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

        // Validate kv_heads_array length matches n_layers
        anyhow::ensure!(
            kv_heads_array.len() >= n_layers,
            "head_count_kv array length ({}) < block_count ({n_layers})",
            kv_heads_array.len()
        );

        // Detect block types from tensor presence
        let mut block_types = Vec::with_capacity(n_layers);
        let mut kv_heads_per_layer = Vec::with_capacity(n_layers);
        for (i, &kv_heads) in kv_heads_array.iter().enumerate().take(n_layers) {
            let is_attn = gguf.tensors.contains_key(&format!("blk.{i}.attn_q.weight"));
            if is_attn {
                let n_kv = kv_heads as usize;
                anyhow::ensure!(
                    n_kv > 0 && n_heads % n_kv == 0,
                    "layer {i}: n_kv_heads ({n_kv}) must be > 0 and divide n_heads ({n_heads})"
                );
                block_types.push(BlockType::Attention);
                kv_heads_per_layer.push(n_kv);
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
        assert!(
            row_idx < wref.m,
            "dequantize_row: row_idx {row_idx} out of range (m={})",
            wref.m
        );
        let data = self.weight_data(wref);
        let row_bytes = wref.k / wref.dtype.block_size() * wref.dtype.block_bytes();
        let row_start = row_idx * row_bytes;
        let row_data = &data[row_start..row_start + row_bytes];

        let mut out = vec![0.0f32; wref.k];
        match wref.dtype {
            DType::Q6K => crate::quant::dequantize_q6_k_row(row_data, &mut out),
            DType::Q8_0 => crate::quant::dequantize_q8_0_row(row_data, &mut out),
            DType::Q4_0 => crate::quant::dequantize_q4_0_row(row_data, &mut out),
            DType::Q4KM => crate::quant::dequantize_q4_k_m_row(row_data, &mut out),
            DType::F32 => {
                let floats: &[f32] = bytemuck::cast_slice(row_data);
                out.copy_from_slice(floats);
            }
            _ => panic!("unsupported embedding dtype: {:?}", wref.dtype),
        }
        out
    }

    /// Process a single conv (recurrent) block using pre-allocated scratch buffers.
    fn forward_conv_block(&self, layer: usize, hidden: &[f32], state: &mut InferenceState) {
        let refs = &self.layer_refs[layer];
        let hidden_size = self.config.hidden_size;
        let kernel_size = self.config.conv_kernel_size.unwrap_or(3);
        let d_conv = kernel_size - 1;
        let in_proj = refs.shortconv_in_proj.as_ref().unwrap();
        let out_proj = refs.shortconv_out_proj.as_ref().unwrap();
        let conv_weight = self.conv_weights[layer].as_ref().unwrap();

        // in_proj: hidden → 3*hidden
        let proj = &mut state.scratch.conv_proj[..3 * hidden_size];
        self.gemv(in_proj, hidden, proj);

        // Split: b, c, x
        let (b, rest) = proj.split_at(hidden_size);
        let (c, x) = rest.split_at(hidden_size);

        // bx = b ⊙ x (element-wise gate before conv)
        let conv_scratch = &mut state.scratch.conv_scratch[..hidden_size];
        for (out, (bi, xi)) in conv_scratch.iter_mut().zip(b.iter().zip(x.iter())) {
            *out = bi * xi;
        }
        // conv_scratch now holds bx

        // Depthwise conv1d with valid convolution using rolling buffer
        let LayerState::Conv { buffer } = &mut state.layers[layer] else {
            panic!("expected Conv state for layer {layer}");
        };

        let out_buf = &mut state.scratch.out[..hidden_size];
        for ch in 0..hidden_size {
            let mut sum = 0.0f32;
            for k in 0..d_conv {
                sum += buffer[k * hidden_size + ch] * conv_weight[ch * kernel_size + k];
            }
            sum += conv_scratch[ch] * conv_weight[ch * kernel_size + d_conv];
            out_buf[ch] = sum;
        }

        // Update rolling buffer: shift left by one slot, append bx
        if d_conv > 0 {
            if d_conv > 1 {
                buffer.copy_within(hidden_size.., 0);
            }
            let last_slot = (d_conv - 1) * hidden_size;
            buffer[last_slot..last_slot + hidden_size].copy_from_slice(conv_scratch);
        }

        // o = c ⊙ conv_out (second gate), reuse conv_scratch
        for (o, (ci, co)) in conv_scratch.iter_mut().zip(c.iter().zip(out_buf.iter())) {
            *o = ci * co;
        }

        // out_proj: hidden → hidden, write result into out_buf
        self.gemv(out_proj, conv_scratch, out_buf);
        // Result is now in state.scratch.out[..hidden_size]
    }

    /// Process a single attention block using pre-allocated scratch buffers.
    fn forward_attn_block(
        &self,
        layer: usize,
        hidden: &[f32],
        pos: usize,
        state: &mut InferenceState,
    ) {
        let refs = &self.layer_refs[layer];
        let cfg = &self.config;
        let head_dim = cfg.hidden_size / cfg.n_heads;
        let n_kv_heads = cfg.kv_heads_per_layer[layer];
        let kv_dim = n_kv_heads * head_dim;

        // Q, K, V projections into scratch buffers
        let q = &mut state.scratch.q[..cfg.hidden_size];
        let k = &mut state.scratch.k[..kv_dim];
        let v = &mut state.scratch.v[..kv_dim];
        self.gemv(refs.attn_q.as_ref().unwrap(), hidden, q);
        self.gemv(refs.attn_k.as_ref().unwrap(), hidden, k);
        self.gemv(refs.attn_v.as_ref().unwrap(), hidden, v);

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
        cpu::rope(q, k, pos, cfg.n_heads, n_kv_heads, head_dim, cfg.rope_theta);

        // Append K, V to cache (need temporaries to satisfy borrow checker)
        let k_to_cache = state.scratch.k[..kv_dim].to_vec();
        let v_to_cache = state.scratch.v[..kv_dim].to_vec();
        state.append_kv(layer, &k_to_cache, &v_to_cache);

        // GQA: grouped query attention
        // Compute attention into a local buffer to avoid borrow conflicts with KV cache.
        let group_size = cfg.n_heads / n_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        {
            // Access layer cache and scratch directly to avoid whole-state borrow
            let (k_cache, v_cache) = match &state.layers[layer] {
                LayerState::Attention {
                    key_cache,
                    value_cache,
                } => (key_cache.as_slice(), value_cache.as_slice()),
                _ => panic!("expected Attention state for layer {layer}"),
            };
            let seq_len = k_cache.len() / kv_dim;
            let attn_out = &mut state.scratch.attn_out[..cfg.hidden_size];
            attn_out.fill(0.0);
            let q = &state.scratch.q[..cfg.hidden_size];

            for h in 0..cfg.n_heads {
                let kv_h = h / group_size;
                let q_head = &q[h * head_dim..(h + 1) * head_dim];

                let mut scores = vec![0.0f32; seq_len];
                for (t, score) in scores.iter_mut().enumerate() {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q_head[d] * k_cache[t * kv_dim + kv_h * head_dim + d];
                    }
                    *score = dot * scale;
                }

                cpu::softmax_inplace(&mut scores);

                for d in 0..head_dim {
                    let mut val = 0.0f32;
                    for (t, &s) in scores.iter().enumerate() {
                        val += s * v_cache[t * kv_dim + kv_h * head_dim + d];
                    }
                    attn_out[h * head_dim + d] = val;
                }
            }
        } // k_cache/v_cache borrows end here

        // Output projection
        let out = &mut state.scratch.out[..cfg.hidden_size];
        self.gemv(
            refs.attn_output.as_ref().unwrap(),
            &state.scratch.attn_out[..cfg.hidden_size],
            out,
        );
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
        // Pre-allocate norm buffers outside the loop to avoid per-layer allocation.
        let hs = cfg.hidden_size;
        let mut normed = vec![0.0f32; hs];
        let mut ffn_input = vec![0.0f32; hs];
        for i in 0..cfg.n_layers {
            // Pre-norm (operator_norm) — copy+norm, keep hidden as residual
            normed.copy_from_slice(&hidden);
            cpu::rmsnorm(&mut normed, &self.attn_norm_weights[i], cfg.rms_norm_eps);

            // Operator: conv or attention (writes result to state.scratch.out)
            if cfg.block_types[i] == BlockType::GatedConv {
                self.forward_conv_block(i, &normed, state);
            } else {
                self.forward_attn_block(i, &normed, pos, state);
            }

            // First residual: hidden += block_out
            cpu::add_inplace(&mut hidden, &state.scratch.out[..hs]);

            // FFN pre-norm
            ffn_input.copy_from_slice(&hidden);
            cpu::rmsnorm(&mut ffn_input, &self.ffn_norm_weights[i], cfg.rms_norm_eps);

            // SwiGLU FFN using scratch buffers
            let refs = &self.layer_refs[i];
            self.gemv(
                &refs.ffn_gate,
                &ffn_input,
                &mut state.scratch.gate[..cfg.intermediate_size],
            );
            self.gemv(
                &refs.ffn_up,
                &ffn_input,
                &mut state.scratch.up[..cfg.intermediate_size],
            );
            cpu::silu_inplace(&mut state.scratch.gate[..cfg.intermediate_size]);
            cpu::mul_inplace(
                &mut state.scratch.gate[..cfg.intermediate_size],
                &state.scratch.up[..cfg.intermediate_size],
            );
            self.gemv(
                &refs.ffn_down,
                &state.scratch.gate[..cfg.intermediate_size],
                &mut state.scratch.out[..cfg.hidden_size],
            );

            // Second residual: hidden += ffn_out
            cpu::add_inplace(&mut hidden, &state.scratch.out[..cfg.hidden_size]);
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
