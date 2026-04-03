// LFM2 / LFM2.5 hybrid conv+attention model.
//
// Reference: Liquid4All/llama.cpp branch dberrios/updateLlama, src/models/lfm2.cpp

use anyhow::{Context, Result};

use crate::backend::cpu;
use crate::gguf::GgufFile;
use crate::kv_cache::{InferenceState, LayerState};
use crate::model::{BlockType, Model, ModelConfig};
use crate::tensor::DType;

// ── Pre-resolved weight reference ───────────────────────────────────────────

/// Pre-resolved reference to a quantized weight in the mmap.
/// Computed once at load time to avoid HashMap lookups during inference.
#[derive(Debug, Clone)]
struct WeightRef {
    start: usize, // byte offset into mmap
    size: usize,  // byte size
    dtype: DType,
    m: usize, // output dim (rows = shape[1] in GGUF order)
    k: usize, // input dim (elements per row = shape[0] in GGUF order)
}

/// Per-layer weight references for quantized tensors.
#[derive(Debug, Clone)]
struct LayerWeightRefs {
    ffn_gate: WeightRef,
    ffn_up: WeightRef,
    ffn_down: WeightRef,
    // Conv-specific (None for attention layers)
    shortconv_in_proj: Option<WeightRef>,
    shortconv_out_proj: Option<WeightRef>,
    // Attention-specific (None for conv layers)
    attn_q: Option<WeightRef>,
    attn_k: Option<WeightRef>,
    attn_v: Option<WeightRef>,
    attn_output: Option<WeightRef>,
}

// ── LFM2 Model ─────────────────────────────────────────────────────────────

pub struct Lfm2Model {
    gguf: GgufFile,
    config: ModelConfig,
    // Pre-dequantized small F32 weights
    output_norm_weight: Vec<f32>,
    attn_norm_weights: Vec<Vec<f32>>,
    ffn_norm_weights: Vec<Vec<f32>>,
    attn_q_norm_weights: Vec<Option<Vec<f32>>>,
    attn_k_norm_weights: Vec<Option<Vec<f32>>>,
    conv_weights: Vec<Option<Vec<f32>>>,
    // Pre-resolved quantized weight refs
    embd_ref: WeightRef,
    layer_refs: Vec<LayerWeightRefs>,
}

impl Lfm2Model {
    pub fn from_gguf(gguf: GgufFile) -> Result<Self> {
        let prefix = "lfm2";

        let n_layers = gguf
            .get_u32(&format!("{prefix}.block_count"))
            .context("missing lfm2.block_count")? as usize;
        let hidden_size = gguf
            .get_u32(&format!("{prefix}.embedding_length"))
            .context("missing lfm2.embedding_length")? as usize;
        let intermediate_size = gguf
            .get_u32(&format!("{prefix}.feed_forward_length"))
            .context("missing lfm2.feed_forward_length")? as usize;
        let n_heads = gguf
            .get_u32(&format!("{prefix}.attention.head_count"))
            .context("missing lfm2.attention.head_count")? as usize;
        let vocab_size = gguf
            .get_u32(&format!("{prefix}.vocab_size"))
            .context("missing lfm2.vocab_size")? as usize;
        let max_seq_len = gguf
            .get_u32(&format!("{prefix}.context_length"))
            .unwrap_or(128000) as usize;
        let rope_theta = gguf
            .get_f32(&format!("{prefix}.rope.freq_base"))
            .unwrap_or(1_000_000.0);
        let rms_norm_eps = gguf
            .get_f32(&format!("{prefix}.attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-5);
        let conv_kernel_size = gguf
            .get_u32(&format!("{prefix}.shortconv.l_cache"))
            .map(|v| v as usize);

        // Per-layer KV head counts
        let kv_heads_array = gguf
            .get_i32_array(&format!("{prefix}.attention.head_count_kv"))
            .context("missing lfm2.attention.head_count_kv")?;

        // Detect block types from tensor presence
        let mut block_types = Vec::with_capacity(n_layers);
        let mut kv_heads_per_layer = Vec::with_capacity(n_layers);
        for (i, &kv_heads) in kv_heads_array.iter().enumerate().take(n_layers) {
            let is_attn = gguf.tensors.contains_key(&format!("blk.{i}.attn_q.weight"));
            if is_attn {
                block_types.push(BlockType::Attention);
                kv_heads_per_layer.push(kv_heads as usize);
            } else {
                block_types.push(BlockType::GatedConv);
                kv_heads_per_layer.push(0);
            }
        }

        let n_kv_heads = kv_heads_per_layer.iter().copied().max().unwrap_or(0);

        let config = ModelConfig {
            architecture: "lfm2".to_string(),
            n_layers,
            hidden_size,
            intermediate_size,
            n_heads,
            n_kv_heads,
            vocab_size,
            max_seq_len,
            rope_theta,
            rms_norm_eps,
            block_types: block_types.clone(),
            conv_kernel_size,
            kv_heads_per_layer: kv_heads_per_layer.clone(),
        };

        // Pre-extract small F32 weights
        let output_norm_weight = gguf.get_tensor("token_embd_norm.weight")?.to_f32_vec();

        let mut attn_norm_weights = Vec::with_capacity(n_layers);
        let mut ffn_norm_weights = Vec::with_capacity(n_layers);
        let mut attn_q_norm_weights = Vec::with_capacity(n_layers);
        let mut attn_k_norm_weights = Vec::with_capacity(n_layers);
        let mut conv_weights = Vec::with_capacity(n_layers);

        for (i, bt) in block_types.iter().enumerate() {
            attn_norm_weights.push(
                gguf.get_tensor(&format!("blk.{i}.attn_norm.weight"))?
                    .to_f32_vec(),
            );
            ffn_norm_weights.push(
                gguf.get_tensor(&format!("blk.{i}.ffn_norm.weight"))?
                    .to_f32_vec(),
            );

            if *bt == BlockType::Attention {
                attn_q_norm_weights.push(Some(
                    gguf.get_tensor(&format!("blk.{i}.attn_q_norm.weight"))?
                        .to_f32_vec(),
                ));
                attn_k_norm_weights.push(Some(
                    gguf.get_tensor(&format!("blk.{i}.attn_k_norm.weight"))?
                        .to_f32_vec(),
                ));
                conv_weights.push(None);
            } else {
                attn_q_norm_weights.push(None);
                attn_k_norm_weights.push(None);
                conv_weights.push(Some(
                    gguf.get_tensor(&format!("blk.{i}.shortconv.conv.weight"))?
                        .to_f32_vec(),
                ));
            }
        }

        // Pre-resolve quantized weight references
        let embd_ref = Self::resolve_weight(&gguf, "token_embd.weight")?;

        let mut layer_refs = Vec::with_capacity(n_layers);
        for (i, bt) in block_types.iter().enumerate() {
            let ffn_gate = Self::resolve_weight(&gguf, &format!("blk.{i}.ffn_gate.weight"))?;
            let ffn_up = Self::resolve_weight(&gguf, &format!("blk.{i}.ffn_up.weight"))?;
            let ffn_down = Self::resolve_weight(&gguf, &format!("blk.{i}.ffn_down.weight"))?;

            let (shortconv_in_proj, shortconv_out_proj, attn_q, attn_k, attn_v, attn_output) =
                if *bt == BlockType::GatedConv {
                    (
                        Some(Self::resolve_weight(
                            &gguf,
                            &format!("blk.{i}.shortconv.in_proj.weight"),
                        )?),
                        Some(Self::resolve_weight(
                            &gguf,
                            &format!("blk.{i}.shortconv.out_proj.weight"),
                        )?),
                        None,
                        None,
                        None,
                        None,
                    )
                } else {
                    (
                        None,
                        None,
                        Some(Self::resolve_weight(
                            &gguf,
                            &format!("blk.{i}.attn_q.weight"),
                        )?),
                        Some(Self::resolve_weight(
                            &gguf,
                            &format!("blk.{i}.attn_k.weight"),
                        )?),
                        Some(Self::resolve_weight(
                            &gguf,
                            &format!("blk.{i}.attn_v.weight"),
                        )?),
                        Some(Self::resolve_weight(
                            &gguf,
                            &format!("blk.{i}.attn_output.weight"),
                        )?),
                    )
                };

            layer_refs.push(LayerWeightRefs {
                ffn_gate,
                ffn_up,
                ffn_down,
                shortconv_in_proj,
                shortconv_out_proj,
                attn_q,
                attn_k,
                attn_v,
                attn_output,
            });
        }

        Ok(Self {
            gguf,
            config,
            output_norm_weight,
            attn_norm_weights,
            ffn_norm_weights,
            attn_q_norm_weights,
            attn_k_norm_weights,
            conv_weights,
            embd_ref,
            layer_refs,
        })
    }

    /// Resolve a tensor name to a pre-computed byte range in the mmap.
    fn resolve_weight(gguf: &GgufFile, name: &str) -> Result<WeightRef> {
        let info = gguf
            .tensors
            .get(name)
            .with_context(|| format!("tensor not found: {name}"))?;

        // info.offset is already absolute (data_offset + raw_offset from GGUF)
        let start = usize::try_from(info.offset)
            .with_context(|| format!("tensor {name} offset overflow"))?;

        let size = info.size_bytes;
        let dtype = info.dtype;

        // GGUF shape: [inner_dim, outer_dim] → in memory: outer_dim rows of inner_dim elements
        let k = info.shape.first().copied().unwrap_or(1); // inner dim (elements per row)
        let m = if info.shape.len() > 1 {
            info.shape[1]
        } else {
            1
        }; // outer dim (number of rows)

        Ok(WeightRef {
            start,
            size,
            dtype,
            m,
            k,
        })
    }

    /// Get the raw bytes for a pre-resolved weight.
    #[inline]
    fn weight_data(&self, wref: &WeightRef) -> &[u8] {
        &self.gguf.mmap_data()[wref.start..wref.start + wref.size]
    }

    /// GEMV dispatch: y[m] = W[m,k] @ x[k], where W is quantized.
    fn gemv(&self, wref: &WeightRef, x: &[f32], y: &mut [f32]) {
        let data = self.weight_data(wref);
        cpu::gemv_dispatch(wref.dtype, data, x, y, wref.m, wref.k);
    }

    /// Dequantize a single row from a quantized matrix (for embedding lookup).
    fn dequantize_row(&self, wref: &WeightRef, row_idx: usize) -> Vec<f32> {
        let data = self.weight_data(wref);
        let row_bytes = wref.k / wref.dtype.block_size() * wref.dtype.block_bytes();
        let row_start = row_idx * row_bytes;
        let row_data = &data[row_start..row_start + row_bytes];

        let mut out = vec![0.0f32; wref.k];
        match wref.dtype {
            DType::Q6K => crate::quant::dequantize_q6_k_row(row_data, &mut out),
            DType::Q8_0 => crate::quant::dequantize_q8_0_row(row_data, &mut out),
            DType::Q4_0 => crate::quant::dequantize_q4_0_row(row_data, &mut out),
            DType::F32 => {
                let floats: &[f32] = bytemuck::cast_slice(row_data);
                out.copy_from_slice(floats);
            }
            _ => panic!("unsupported embedding dtype: {:?}", wref.dtype),
        }
        out
    }

    /// Process a single conv (recurrent) block.
    fn forward_conv_block(
        &self,
        layer: usize,
        hidden: &[f32],
        state: &mut InferenceState,
    ) -> Vec<f32> {
        let refs = &self.layer_refs[layer];
        let hidden_size = self.config.hidden_size;
        let kernel_size = self.config.conv_kernel_size.unwrap_or(3);
        let d_conv = kernel_size - 1; // number of buffered past states
        let in_proj = refs.shortconv_in_proj.as_ref().unwrap();
        let out_proj = refs.shortconv_out_proj.as_ref().unwrap();
        let conv_weight = self.conv_weights[layer].as_ref().unwrap();

        // in_proj: hidden → 3*hidden
        let proj_size = 3 * hidden_size;
        let mut proj = vec![0.0f32; proj_size];
        self.gemv(in_proj, hidden, &mut proj);

        // Split: b, c, x
        let b = &proj[0..hidden_size];
        let c = &proj[hidden_size..2 * hidden_size];
        let x = &proj[2 * hidden_size..3 * hidden_size];

        // bx = b ⊙ x (element-wise gate before conv)
        let bx: Vec<f32> = b.iter().zip(x.iter()).map(|(bi, xi)| bi * xi).collect();

        // Depthwise conv1d with valid convolution using rolling buffer.
        // Buffer stores d_conv previous bx values (time-major: [d_conv, hidden]).
        // Conv weight layout: [hidden, kernel_size] = weight[ch * kernel_size + k].
        let mut conv_out = vec![0.0f32; hidden_size];
        if let LayerState::Conv { buffer } = &state.layers[layer] {
            for ch in 0..hidden_size {
                let mut sum = 0.0f32;
                for k in 0..d_conv {
                    sum += buffer[k * hidden_size + ch] * conv_weight[ch * kernel_size + k];
                }
                // Current bx is the last element in the kernel window
                sum += bx[ch] * conv_weight[ch * kernel_size + d_conv];
                conv_out[ch] = sum;
            }
        }

        // Update rolling buffer: shift left by one slot, append bx
        if let LayerState::Conv { buffer } = &mut state.layers[layer] {
            if d_conv > 1 {
                buffer.copy_within(hidden_size.., 0);
            }
            let last_slot = (d_conv - 1) * hidden_size;
            buffer[last_slot..last_slot + hidden_size].copy_from_slice(&bx);
        }

        // o = c ⊙ conv_out (second gate)
        let o: Vec<f32> = c
            .iter()
            .zip(conv_out.iter())
            .map(|(ci, co)| ci * co)
            .collect();

        // out_proj: hidden → hidden
        let mut out = vec![0.0f32; hidden_size];
        self.gemv(out_proj, &o, &mut out);

        out
    }

    /// Process a single attention block.
    fn forward_attn_block(
        &self,
        layer: usize,
        hidden: &[f32],
        pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        let refs = &self.layer_refs[layer];
        let cfg = &self.config;
        let head_dim = cfg.hidden_size / cfg.n_heads;
        let n_kv_heads = cfg.kv_heads_per_layer[layer];
        let kv_dim = n_kv_heads * head_dim;

        // Q, K, V projections
        let mut q = vec![0.0f32; cfg.hidden_size];
        let mut k = vec![0.0f32; kv_dim];
        let mut v = vec![0.0f32; kv_dim];
        self.gemv(refs.attn_q.as_ref().unwrap(), hidden, &mut q);
        self.gemv(refs.attn_k.as_ref().unwrap(), hidden, &mut k);
        self.gemv(refs.attn_v.as_ref().unwrap(), hidden, &mut v);

        // Per-head QK norm (RMSnorm each head slice with shared weights)
        let q_norm = self.attn_q_norm_weights[layer].as_ref().unwrap();
        let k_norm = self.attn_k_norm_weights[layer].as_ref().unwrap();
        for h in 0..cfg.n_heads {
            cpu::rmsnorm(
                &mut q[h * head_dim..(h + 1) * head_dim],
                q_norm,
                cfg.rms_norm_eps,
            );
        }
        for h in 0..n_kv_heads {
            cpu::rmsnorm(
                &mut k[h * head_dim..(h + 1) * head_dim],
                k_norm,
                cfg.rms_norm_eps,
            );
        }

        // RoPE
        cpu::rope(
            &mut q,
            &mut k,
            pos,
            cfg.n_heads,
            n_kv_heads,
            head_dim,
            cfg.rope_theta,
        );

        // Append K, V to cache
        state.append_kv(layer, &k, &v);

        // GQA: grouped query attention
        let group_size = cfg.n_heads / n_kv_heads;
        let (k_cache, v_cache) = state.kv_cache(layer);
        // Derive seq_len from actual cache size, not state.seq_len which hasn't
        // been incremented yet for the current token.
        let seq_len = k_cache.len() / kv_dim;

        let mut attn_out = vec![0.0f32; cfg.hidden_size];
        let scale = 1.0 / (head_dim as f32).sqrt();

        for h in 0..cfg.n_heads {
            let kv_h = h / group_size;
            let q_head = &q[h * head_dim..(h + 1) * head_dim];

            // Compute attention scores: Q_h · K_cache^T
            let mut scores = vec![0.0f32; seq_len];
            for t in 0..seq_len {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_head[d] * k_cache[t * kv_dim + kv_h * head_dim + d];
                }
                scores[t] = dot * scale;
            }

            // Softmax
            cpu::softmax_inplace(&mut scores);

            // Weighted sum of V_cache
            for d in 0..head_dim {
                let mut val = 0.0f32;
                for t in 0..seq_len {
                    val += scores[t] * v_cache[t * kv_dim + kv_h * head_dim + d];
                }
                attn_out[h * head_dim + d] = val;
            }
        }

        // Output projection
        let mut out = vec![0.0f32; cfg.hidden_size];
        self.gemv(refs.attn_output.as_ref().unwrap(), &attn_out, &mut out);

        out
    }
}

impl Model for Lfm2Model {
    fn forward(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) -> Vec<f32> {
        assert_eq!(tokens.len(), 1, "LFM2 forward expects single token");
        let token_id = tokens[0] as usize;
        let cfg = &self.config;
        assert!(
            token_id < cfg.vocab_size,
            "token_id {token_id} out of range (vocab_size={})",
            cfg.vocab_size
        );

        // 1. Embedding lookup (no embedding norm — raw into layers)
        let mut hidden = self.dequantize_row(&self.embd_ref, token_id);

        // 2. Per-layer loop
        for i in 0..cfg.n_layers {
            // Pre-norm (operator_norm)
            let prev_cur = hidden.clone();
            cpu::rmsnorm(&mut hidden, &self.attn_norm_weights[i], cfg.rms_norm_eps);

            // Operator: conv or attention
            let block_out = if cfg.block_types[i] == BlockType::GatedConv {
                self.forward_conv_block(i, &hidden, state)
            } else {
                self.forward_attn_block(i, &hidden, pos, state)
            };

            // First residual
            hidden = prev_cur;
            cpu::add_inplace(&mut hidden, &block_out);

            // FFN pre-norm
            let prev_cur = hidden.clone();
            let mut ffn_input = hidden.clone();
            cpu::rmsnorm(&mut ffn_input, &self.ffn_norm_weights[i], cfg.rms_norm_eps);

            // SwiGLU FFN
            let refs = &self.layer_refs[i];
            let mut gate = vec![0.0f32; cfg.intermediate_size];
            let mut up = vec![0.0f32; cfg.intermediate_size];
            self.gemv(&refs.ffn_gate, &ffn_input, &mut gate);
            self.gemv(&refs.ffn_up, &ffn_input, &mut up);
            cpu::silu_inplace(&mut gate);
            cpu::mul_inplace(&mut gate, &up);
            let mut ffn_out = vec![0.0f32; cfg.hidden_size];
            self.gemv(&refs.ffn_down, &gate, &mut ffn_out);

            // Second residual
            hidden = prev_cur;
            cpu::add_inplace(&mut hidden, &ffn_out);
        }

        // 3. Output norm (token_embd_norm is the output norm, not embedding norm)
        cpu::rmsnorm(&mut hidden, &self.output_norm_weight, cfg.rms_norm_eps);

        // 4. Output projection (tied embeddings)
        let mut logits = vec![0.0f32; cfg.vocab_size];
        self.gemv(&self.embd_ref, &hidden, &mut logits);

        // Update sequence length
        state.seq_len += 1;

        logits
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }
}
