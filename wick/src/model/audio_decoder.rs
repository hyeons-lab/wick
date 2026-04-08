//! Audio decoder for LFM2.5-Audio: vocoder GGUF loading + weight structures.
//!
//! Three sub-models loaded from the vocoder GGUF:
//! 1. DecoderModel — samples 8 audio codes per frame from LLM embedding
//! 2. DepthformerModel — small 6-layer attention-only transformer (backbone of DecoderModel)
//! 3. Detokenizer — codes → spectrogram → PCM (not loaded here yet — Phase 5)

use anyhow::{Context, Result};

use crate::backend::cpu;
use crate::gguf::GgufFile;
use crate::tensor::DType;

/// GPU-accelerated audio backend. Implementations provide Metal or WGPU
/// dispatch for the depthformer (code sampling) and detokenizer (spectrum).
pub trait AudioGpu {
    /// Sample 8 audio codes from an LLM embedding using the depthformer.
    fn sample_audio_frame(&self, embedding: &[f32], temperature: f32, top_k: usize) -> [i32; 8];

    /// Convert 8 audio codes to spectrum [n_frames × n_fft_bins × 2].
    fn detokenize_to_spectrum(&self, cpu_weights: &DetokenizerWeights, codes: &[i32]) -> Vec<f32>;

    /// Reset depthformer KV caches (called per audio frame).
    fn reset_depthformer(&self);

    /// Reset detokenizer state (conv buffers + KV caches, called per generation).
    fn reset_detokenizer(&self);
}

/// Configuration for the depthformer (small transformer inside the decoder).
#[derive(Debug, Clone)]
pub struct DepthformerConfig {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub n_embd_head: usize,
    pub ffn_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_freq_base: f32,
    pub max_seq_len: usize,
}

/// Configuration for the decoder model (8-codebook sampling).
#[derive(Debug, Clone)]
pub struct DecoderConfig {
    pub n_codebook: usize,
    pub n_vocab: usize,
    pub n_embd: usize, // LLM embedding dimension (2048)
    pub rms_norm_eps: f32,
}

/// A single tensor stored as dequantized f32 for CPU inference.
pub struct F32Weight {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl F32Weight {
    fn from_tensor(gguf: &GgufFile, name: &str) -> Result<Self> {
        let tensor = gguf.get_tensor(name)?;
        let shape = tensor.shape();
        let (rows, cols) = match shape.len() {
            1 => (1, shape[0]),
            2 => (shape[1], shape[0]), // GGUF stores [cols, rows]
            _ => anyhow::bail!("unexpected tensor rank for {name}: {}", shape.len()),
        };
        let data = tensor.to_f32_vec();
        Ok(Self { data, rows, cols })
    }

    /// Matrix-vector multiply: y = self × x. self is [rows, cols], x is [cols].
    pub fn gemv(&self, x: &[f32], y: &mut [f32]) {
        assert_eq!(x.len(), self.cols);
        assert_eq!(y.len(), self.rows);
        for (r, y_r) in y.iter_mut().enumerate() {
            let row = &self.data[r * self.cols..(r + 1) * self.cols];
            *y_r = row.iter().zip(x).map(|(w, x)| w * x).sum();
        }
    }
}

/// A weight tensor stored in its native quantized format (Q4_0).
/// Uses the quantized GEMV path (NEON Q4_0×Q8_0) matching ggml's computation.
pub struct QuantWeight {
    pub data: Vec<u8>,
    pub dtype: DType,
    pub rows: usize,
    pub cols: usize,
}

impl QuantWeight {
    fn from_tensor(gguf: &GgufFile, name: &str) -> Result<Self> {
        let tensor = gguf.get_tensor(name)?;
        let shape = tensor.shape();
        let (rows, cols) = match shape.len() {
            1 => (1, shape[0]),
            2 => (shape[1], shape[0]),
            _ => anyhow::bail!("unexpected tensor rank for {name}: {}", shape.len()),
        };
        let data = tensor.data().to_vec();
        let dtype = tensor.dtype();
        Ok(Self {
            data,
            rows,
            cols,
            dtype,
        })
    }

    /// Matrix-vector multiply using the quantized GEMV path.
    pub fn gemv(&self, x: &[f32], y: &mut [f32]) {
        assert_eq!(x.len(), self.cols);
        assert_eq!(y.len(), self.rows);
        crate::backend::cpu::gemv_dispatch(
            self.dtype, &self.data, x, y, self.rows, self.cols, None,
        );
    }

    /// Matrix-vector multiply for a row subrange [row_start..row_start+n_rows].
    /// Used for per-codebook slicing of depth_linear.
    pub fn gemv_rows(&self, x: &[f32], y: &mut [f32], row_start: usize, n_rows: usize) {
        assert_eq!(x.len(), self.cols);
        assert_eq!(y.len(), n_rows);
        assert!(row_start + n_rows <= self.rows);
        let row_bytes = (self.cols / self.dtype.block_size()) * self.dtype.block_bytes();
        let offset = row_start * row_bytes;
        let slice = &self.data[offset..offset + n_rows * row_bytes];
        crate::backend::cpu::gemv_dispatch(self.dtype, slice, x, y, n_rows, self.cols, None);
    }
}

/// Per-layer weights for one depthformer layer.
/// Uses QuantWeight for GEMV to match ggml's Q4_0 computation path.
pub struct DepthformerLayerWeights {
    pub operator_norm: Vec<f32>,
    pub wqkv: QuantWeight,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub wo: QuantWeight,
    pub ffn_norm: Vec<f32>,
    pub w1: QuantWeight, // gate
    pub w2: QuantWeight, // down
    pub w3: QuantWeight, // up
}

/// Per-codebook embedding layer weights.
pub struct CodebookWeights {
    pub embedding: F32Weight,
    pub norm: Vec<f32>,
    pub to_logits: QuantWeight,
}

/// All weights for the audio decoder (loaded from vocoder GGUF).
pub struct AudioDecoderWeights {
    pub depthformer_config: DepthformerConfig,
    pub decoder_config: DecoderConfig,
    pub depthformer_layers: Vec<DepthformerLayerWeights>,
    pub depth_linear_w: QuantWeight,
    pub depth_linear_b: Vec<f32>,
    pub depth_embeddings: Vec<CodebookWeights>,
    pub audio_embedding: CodebookWeights,
}

impl AudioDecoderWeights {
    /// Load all decoder weights from the vocoder GGUF.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        // Hyperparameters from GGUF metadata.
        let n_layer = gguf
            .get_u32("depthformer_n_layer")
            .context("missing depthformer_n_layer")? as usize;
        let n_embd = gguf
            .get_u32("depthformer_n_embd")
            .context("missing depthformer_n_embd")? as usize;

        // Derive head config from qkv_proj shape.
        // qkv_proj.weight is [n_embd, (n_head + 2*n_head_kv) * head_dim]
        let qkv = F32Weight::from_tensor(gguf, "depthformer.layers.0.operator.qkv_proj.weight")?;
        let q_norm_w =
            gguf.get_tensor("depthformer.layers.0.operator.attention.q_layernorm.weight")?;
        let n_embd_head = q_norm_w.shape()[0]; // head_dim from q_norm shape
        let qkv_out = qkv.rows; // (n_head + 2*n_head_kv) * head_dim
        // qkv_out = n_head*hd + 2*n_kv*hd. With n_kv = n_head/4 (typical):
        // qkv_out = hd*(n_head + n_head/2). Solve for n_head.
        // From decoder.cpp: n_head=32, n_head_kv=8, hd=32.
        let n_head = 32; // hardcoded per decoder.cpp config
        let n_head_kv = 8;
        assert_eq!(
            qkv_out,
            (n_head + 2 * n_head_kv) * n_embd_head,
            "qkv_proj shape mismatch"
        );

        // FFN dim from w1 shape.
        let w1_0 = F32Weight::from_tensor(gguf, "depthformer.layers.0.feed_forward.w1.weight")?;
        let ffn_dim = w1_0.rows;

        let depthformer_config = DepthformerConfig {
            n_layer,
            n_embd,
            n_head,
            n_head_kv,
            n_embd_head,
            ffn_dim,
            rms_norm_eps: 1e-5,
            rope_freq_base: 1_000_000.0,
            max_seq_len: 8,
        };

        // Load depthformer layers.
        let mut depthformer_layers = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let prefix = format!("depthformer.layers.{i}");
            depthformer_layers.push(DepthformerLayerWeights {
                operator_norm: gguf
                    .get_tensor(&format!("{prefix}.operator_norm.weight"))?
                    .to_f32_vec(),
                wqkv: QuantWeight::from_tensor(
                    gguf,
                    &format!("{prefix}.operator.qkv_proj.weight"),
                )?,
                q_norm: gguf
                    .get_tensor(&format!("{prefix}.operator.attention.q_layernorm.weight"))?
                    .to_f32_vec(),
                k_norm: gguf
                    .get_tensor(&format!("{prefix}.operator.attention.k_layernorm.weight"))?
                    .to_f32_vec(),
                wo: QuantWeight::from_tensor(gguf, &format!("{prefix}.operator.out_proj.weight"))?,
                ffn_norm: gguf
                    .get_tensor(&format!("{prefix}.ffn_norm.weight"))?
                    .to_f32_vec(),
                w1: QuantWeight::from_tensor(gguf, &format!("{prefix}.feed_forward.w1.weight"))?,
                w2: QuantWeight::from_tensor(gguf, &format!("{prefix}.feed_forward.w2.weight"))?,
                w3: QuantWeight::from_tensor(gguf, &format!("{prefix}.feed_forward.w3.weight"))?,
            });
        }

        // Decoder model weights.
        let depth_linear_w = QuantWeight::from_tensor(gguf, "depth_linear.weight")?;
        let depth_linear_b = gguf.get_tensor("depth_linear.bias")?.to_f32_vec();

        let n_codebook = 8;
        let mut depth_embeddings = Vec::with_capacity(n_codebook);
        for i in 0..n_codebook {
            let prefix = format!("depth_embeddings.{i}");
            depth_embeddings.push(CodebookWeights {
                embedding: F32Weight::from_tensor(gguf, &format!("{prefix}.embedding.weight"))?,
                norm: gguf
                    .get_tensor(&format!("{prefix}.embedding_norm.weight"))?
                    .to_f32_vec(),
                to_logits: QuantWeight::from_tensor(gguf, &format!("{prefix}.to_logits.weight"))?,
            });
        }

        let audio_embedding = CodebookWeights {
            embedding: F32Weight::from_tensor(gguf, "audio_embedding.embedding.weight")?,
            norm: gguf
                .get_tensor("audio_embedding.embedding_norm.weight")?
                .to_f32_vec(),
            to_logits: QuantWeight::from_tensor(gguf, "audio_embedding.to_logits.weight")?,
        };

        // Derive decoder config from weight shapes.
        let n_embd_llm = depth_linear_w.cols; // LLM embedding dim (2048)
        let n_vocab = depth_embeddings[0].to_logits.rows; // 2049

        let decoder_config = DecoderConfig {
            n_codebook,
            n_vocab,
            n_embd: n_embd_llm,
            rms_norm_eps: 1e-5,
        };

        eprintln!(
            "Audio decoder loaded: depthformer {}L×{}, decoder {}cb×{} vocab, LLM embd={}",
            n_layer, n_embd, n_codebook, n_vocab, n_embd_llm
        );

        Ok(Self {
            depthformer_config,
            decoder_config,
            depthformer_layers,
            depth_linear_w,
            depth_linear_b,
            depth_embeddings,
            audio_embedding,
        })
    }
}

// ── Depthformer runtime ─────────────────────────────────────────────────

/// KV cache for one depthformer layer.
struct LayerKvCache {
    /// [max_seq × n_head_kv × head_dim] for K and V.
    k: Vec<f32>,
    v: Vec<f32>,
}

/// Runtime state for the depthformer (reset per audio frame).
pub struct DepthformerState {
    kv: Vec<LayerKvCache>,
    n_past: usize,
}

impl DepthformerState {
    pub fn new(cfg: &DepthformerConfig) -> Self {
        let cache_size = cfg.max_seq_len * cfg.n_head_kv * cfg.n_embd_head;
        let kv = (0..cfg.n_layer)
            .map(|_| LayerKvCache {
                k: vec![0.0; cache_size],
                v: vec![0.0; cache_size],
            })
            .collect();
        Self { kv, n_past: 0 }
    }

    pub fn reset(&mut self) {
        for layer in &mut self.kv {
            layer.k.fill(0.0);
            layer.v.fill(0.0);
        }
        self.n_past = 0;
    }
}

/// Apply RoPE (interleaved / LLAMA_ROPE_TYPE_NORM) to a single head.
/// Uses iterative theta multiplication to match ggml's `ggml_rope_cache_init`.
fn apply_rope_interleaved(x: &mut [f32], pos: usize, head_dim: usize, freq_base: f32) {
    let theta_scale = freq_base.powf(-2.0 / head_dim as f32);
    let mut theta = pos as f32;
    for i in 0..head_dim / 2 {
        let (sin_t, cos_t) = theta.sin_cos();
        let x0 = x[2 * i];
        let x1 = x[2 * i + 1];
        x[2 * i] = x0 * cos_t - x1 * sin_t;
        x[2 * i + 1] = x0 * sin_t + x1 * cos_t;
        theta *= theta_scale;
    }
}

/// Apply RoPE (NeoX style / LLAMA_ROPE_TYPE_NEOX) to a single head.
/// Uses iterative theta multiplication to match ggml's `ggml_rope_cache_init`.
fn apply_rope_neox(x: &mut [f32], pos: usize, head_dim: usize, freq_base: f32) {
    let theta_scale = freq_base.powf(-2.0 / head_dim as f32);
    let mut theta = pos as f32;
    for i in 0..head_dim / 2 {
        let (sin_t, cos_t) = theta.sin_cos();
        let x0 = x[i];
        let x1 = x[i + head_dim / 2];
        x[i] = x0 * cos_t - x1 * sin_t;
        x[i + head_dim / 2] = x0 * sin_t + x1 * cos_t;
        theta *= theta_scale;
    }
}

/// Run one forward pass through the depthformer (1 token, n_embd input → n_embd output).
pub fn depthformer_forward(
    weights: &AudioDecoderWeights,
    state: &mut DepthformerState,
    input: &[f32],
) -> Vec<f32> {
    let cfg = &weights.depthformer_config;
    let n_embd = cfg.n_embd;
    let n_head = cfg.n_head;
    let n_kv = cfg.n_head_kv;
    let hd = cfg.n_embd_head;
    let pos = state.n_past;
    let kv_dim = n_kv * hd;
    let q_dim = n_head * hd;
    let k_dim = n_kv * hd;
    let qkv_dim = q_dim + k_dim + k_dim; // Q + K + V
    let group_size = n_head / n_kv;
    let scale = 1.0 / (hd as f32).sqrt();

    let mut cur = input.to_vec();
    assert_eq!(cur.len(), n_embd);

    // Pre-allocate buffers outside the layer loop to avoid per-layer allocation.
    let mut residual = vec![0.0f32; n_embd];
    let mut qkv = vec![0.0f32; qkv_dim];
    let mut attn_out = vec![0.0f32; n_head * hd];
    let mut proj = vec![0.0f32; n_embd];
    let mut gate = vec![0.0f32; cfg.ffn_dim];
    let mut up = vec![0.0f32; cfg.ffn_dim];
    let mut scores = vec![0.0f32; cfg.max_seq_len];
    // Q8_0 scratch for pre-quantized GEMV (avoids re-quantizing for each weight matrix).
    let mut q8_scales = Vec::new();
    let mut q8_quants = Vec::new();

    for (il, lw) in weights.depthformer_layers.iter().enumerate() {
        residual.copy_from_slice(&cur);

        // 1. RMSnorm.
        cpu::rmsnorm(&mut cur, &lw.operator_norm, cfg.rms_norm_eps);

        // 2. Fused QKV projection → split.
        // Pre-quantize cur to Q8_0 once, reuse for QKV and later wo.
        #[cfg(target_arch = "aarch64")]
        {
            let nb = n_embd / 32;
            q8_scales.resize(nb, 0.0);
            q8_quants.resize(n_embd, 0);
            unsafe {
                crate::backend::simd::neon::quantize_f32_to_q8_0_neon(
                    &cur,
                    &mut q8_scales,
                    &mut q8_quants,
                );
            }
        }
        qkv.iter_mut().for_each(|v| *v = 0.0);
        crate::backend::cpu::gemv_dispatch(
            lw.wqkv.dtype,
            &lw.wqkv.data,
            &cur,
            &mut qkv,
            lw.wqkv.rows,
            lw.wqkv.cols,
            Some((&mut q8_scales, &mut q8_quants)),
        );

        // Split in-place via index ranges (no copy for V).
        // Q: qkv[0..q_dim], K: qkv[q_dim..q_dim+k_dim], V: qkv[q_dim+k_dim..]

        // 3. Per-head RMSnorm on Q and K (in-place within qkv).
        for h in 0..n_head {
            let s = &mut qkv[h * hd..(h + 1) * hd];
            cpu::rmsnorm(s, &lw.q_norm, cfg.rms_norm_eps);
        }
        for h in 0..n_kv {
            let s = &mut qkv[q_dim + h * hd..q_dim + (h + 1) * hd];
            cpu::rmsnorm(s, &lw.k_norm, cfg.rms_norm_eps);
        }

        // 4. RoPE on Q and K (depthformer uses interleaved/NORM style).
        for h in 0..n_head {
            apply_rope_interleaved(&mut qkv[h * hd..(h + 1) * hd], pos, hd, cfg.rope_freq_base);
        }
        for h in 0..n_kv {
            apply_rope_interleaved(
                &mut qkv[q_dim + h * hd..q_dim + (h + 1) * hd],
                pos,
                hd,
                cfg.rope_freq_base,
            );
        }

        // 5. Write K, V to cache at position n_past.
        let kv = &mut state.kv[il];
        for h in 0..n_kv {
            let cache_off = pos * kv_dim + h * hd;
            kv.k[cache_off..cache_off + hd]
                .copy_from_slice(&qkv[q_dim + h * hd..q_dim + (h + 1) * hd]);
            kv.v[cache_off..cache_off + hd]
                .copy_from_slice(&qkv[q_dim + k_dim + h * hd..q_dim + k_dim + (h + 1) * hd]);
        }

        // 6. Attention: Q×K → softmax → ×V (GQA).
        let seq_len = pos + 1;
        attn_out.iter_mut().for_each(|v| *v = 0.0);

        for h in 0..n_head {
            let kv_h = h / group_size;
            let q_head = &qkv[h * hd..(h + 1) * hd];
            let kv_h_offset = kv_h * hd;

            let sc = &mut scores[..seq_len];
            cpu::attn_scores(q_head, &kv.k, sc, kv_dim, kv_h_offset, hd, scale, seq_len);
            cpu::softmax_inplace(sc);

            let out = &mut attn_out[h * hd..(h + 1) * hd];
            cpu::attn_values(sc, &kv.v, out, kv_dim, kv_h_offset, hd, seq_len);
        }

        // 7. Out projection + residual (in-place).
        proj.iter_mut().for_each(|v| *v = 0.0);
        lw.wo.gemv(&attn_out, &mut proj);
        for (c, (r, p)) in cur.iter_mut().zip(residual.iter().zip(&proj)) {
            *c = r + p;
        }

        // 8. FFN: RMSnorm → SwiGLU(w1, w3) → w2 → residual.
        residual.copy_from_slice(&cur);
        cpu::rmsnorm(&mut cur, &lw.ffn_norm, cfg.rms_norm_eps);

        // Pre-quantize cur for FFN gate + up (same input).
        #[cfg(target_arch = "aarch64")]
        {
            let nb = n_embd / 32;
            q8_scales.resize(nb, 0.0);
            q8_quants.resize(n_embd, 0);
            unsafe {
                crate::backend::simd::neon::quantize_f32_to_q8_0_neon(
                    &cur,
                    &mut q8_scales,
                    &mut q8_quants,
                );
            }
        }
        gate.iter_mut().for_each(|v| *v = 0.0);
        up.iter_mut().for_each(|v| *v = 0.0);
        crate::backend::cpu::gemv_dispatch(
            lw.w1.dtype,
            &lw.w1.data,
            &cur,
            &mut gate,
            lw.w1.rows,
            lw.w1.cols,
            Some((&mut q8_scales, &mut q8_quants)),
        );
        crate::backend::cpu::gemv_dispatch(
            lw.w3.dtype,
            &lw.w3.data,
            &cur,
            &mut up,
            lw.w3.rows,
            lw.w3.cols,
            Some((&mut q8_scales, &mut q8_quants)),
        );
        cpu::silu_mul_inplace(&mut gate, &up);

        proj.iter_mut().for_each(|v| *v = 0.0);
        lw.w2.gemv(&gate, &mut proj);
        for (c, (r, d)) in cur.iter_mut().zip(residual.iter().zip(&proj)) {
            *c = r + d;
        }
    }

    state.n_past += 1;
    cur
}

// ── DecoderModel: 8-codebook audio frame sampling ───────────────────────

/// Sample one audio frame (8 codes) from an LLM embedding.
///
/// Runs 8 sequential passes through the depthformer, one per codebook.
/// Each pass projects the LLM embedding into the depthformer input,
/// optionally adds the previous codebook's token embedding, runs the
/// depthformer, and samples a token from the codebook's logits.
pub fn sample_audio_frame(
    weights: &AudioDecoderWeights,
    state: &mut DepthformerState,
    embedding: &[f32],
    temperature: f32,
    top_k: usize,
) -> [i32; 8] {
    let cfg = &weights.decoder_config;
    let df_cfg = &weights.depthformer_config;
    let mut token = [0i32; 8];
    let mut prev_token: i32 = -1;

    state.reset();

    for j in 0..cfg.n_codebook {
        let cb = &weights.depth_embeddings[j];
        let n_embd_d = cb.embedding.cols; // depthformer input dim per codebook

        // 1. Project LLM embedding → depthformer input for codebook j.
        //    depth_linear.weight is [n_embd_llm, n_codebook * n_embd_d],
        //    slice rows [j*n_embd_d .. (j+1)*n_embd_d].
        let mut depthformer_in = vec![0.0; n_embd_d];
        let row_start = j * n_embd_d;
        weights
            .depth_linear_w
            .gemv_rows(embedding, &mut depthformer_in, row_start, n_embd_d);
        // Add bias.
        for (r, out) in depthformer_in.iter_mut().enumerate() {
            *out += weights.depth_linear_b[row_start + r];
        }

        // 2. Add previous codebook's token embedding (if j > 0).
        if j > 0 && prev_token >= 0 {
            let prev_cb = &weights.depth_embeddings[j - 1];
            let tok = prev_token as usize;
            let emb_row = &prev_cb.embedding.data[tok * n_embd_d..(tok + 1) * n_embd_d];
            for (d, e) in depthformer_in.iter_mut().zip(emb_row) {
                *d += e;
            }
        }

        // 3. Run depthformer.
        let hidden = depthformer_forward(weights, state, &depthformer_in);

        // 4. RMSnorm → to_logits → sample.
        let mut normed = hidden;
        cpu::rmsnorm(&mut normed, &cb.norm, cfg.rms_norm_eps);

        let mut logits = vec![0.0; cfg.n_vocab];
        cb.to_logits.gemv(&normed, &mut logits);

        // Sample (greedy if temperature <= 0, otherwise top-k).
        let sampled = if temperature <= 0.0 {
            crate::sampler::cpu_argmax(&logits) as i32
        } else {
            // Temperature scaling + top-k sampling.
            let inv_temp = 1.0 / temperature;
            for l in &mut logits {
                *l *= inv_temp;
            }
            cpu::softmax_inplace(&mut logits);
            // Simple top-k: find top-k, sample from them.
            let mut indices: Vec<usize> = (0..logits.len()).collect();
            indices.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
            indices.truncate(top_k.min(logits.len()));
            let sum: f32 = indices.iter().map(|&i| logits[i]).sum();
            let mut r = rand::random::<f32>() * sum;
            let mut picked = indices[0];
            for &i in &indices {
                r -= logits[i];
                if r <= 0.0 {
                    picked = i;
                    break;
                }
            }
            picked as i32
        };

        token[j] = sampled;
        prev_token = sampled;
    }

    eprintln!("  codes: {token:?}");
    token
}

/// Convert 8 audio codes back into an embedding for feeding to the LLM.
///
/// Each code indexes into audio_embedding.embedding.weight (with per-codebook
/// offsets), and the embeddings are summed to produce a single vector.
pub fn embed_audio_token(weights: &AudioDecoderWeights, codes: &[i32; 8]) -> Vec<f32> {
    let emb = &weights.audio_embedding.embedding;
    let n_codebook = weights.decoder_config.n_codebook;
    let n_vocab = 2049; // per-codebook vocab size
    let emb_dim = emb.cols;
    let mut result = vec![0.0f32; emb_dim];

    for (j, &code) in codes.iter().enumerate() {
        let offset_idx = j * n_vocab + code as usize;
        let row = &emb.data[offset_idx * emb_dim..(offset_idx + 1) * emb_dim];
        for (r, e) in result.iter_mut().zip(row) {
            *r += e;
        }
    }

    result
}

// ── Detokenizer: codes → PCM ────────────────────────────────────────────

/// Config for the detokenizer's LFM2 backbone.
#[derive(Debug, Clone)]
pub struct DetokenizerConfig {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub n_embd_head: usize,
    pub ffn_dim: usize,
    pub d_conv: usize, // conv kernel - 1
    pub rms_norm_eps: f32,
    pub rope_freq_base: f32,
    pub swa_window_size: usize,
    pub n_codes: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub sample_rate: usize,
    /// Per-layer type: true = conv (recurrent), false = attention.
    pub layer_is_conv: Vec<bool>,
}

/// Weights for one detokenizer LFM2 layer.
pub struct DetokLayerWeights {
    pub operator_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub ffn_w1: F32Weight, // gate
    pub ffn_w2: F32Weight, // down
    pub ffn_w3: F32Weight, // up
    // Conv layers
    pub conv_in_proj: Option<F32Weight>,
    pub conv_out_proj: Option<F32Weight>,
    pub conv_weight: Option<Vec<f32>>, // [kernel_size, n_embd]
    // Attention layers
    pub wq: Option<F32Weight>,
    pub wk: Option<F32Weight>,
    pub wv: Option<F32Weight>,
    pub wo: Option<F32Weight>,
    pub q_norm: Option<Vec<f32>>,
    pub k_norm: Option<Vec<f32>>,
}

/// All detokenizer weights (loaded from vocoder GGUF).
pub struct DetokenizerWeights {
    pub config: DetokenizerConfig,
    pub output_norm: Vec<f32>,
    pub emb_weight: F32Weight, // code embedding [n_codes * n_vocab, emb_dim]
    pub lin_w: F32Weight,      // linear head
    pub lin_b: Vec<f32>,
    pub layers: Vec<DetokLayerWeights>,
}

impl DetokenizerWeights {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        // Layer types (hardcoded from decoder.cpp).
        let layer_is_conv = vec![true, true, false, true, false, true, false, true];
        let n_layer = layer_is_conv.len();

        // Derive config from weight shapes.
        let conv_in = F32Weight::from_tensor(gguf, "lfm.layers.0.conv.in_proj.weight")?;
        let n_embd = conv_in.cols;
        let q_norm_w = gguf.get_tensor("lfm.layers.2.self_attn.q_layernorm.weight")?;
        let n_embd_head = q_norm_w.shape()[0];
        let q_w = F32Weight::from_tensor(gguf, "lfm.layers.2.self_attn.q_proj.weight")?;
        let n_head = q_w.rows / n_embd_head;
        let k_w = F32Weight::from_tensor(gguf, "lfm.layers.2.self_attn.k_proj.weight")?;
        let n_head_kv = k_w.rows / n_embd_head;
        let ffn_w1_0 = F32Weight::from_tensor(gguf, "lfm.layers.0.feed_forward.w1.weight")?;
        let ffn_dim = ffn_w1_0.rows;

        let config = DetokenizerConfig {
            n_layer,
            n_embd,
            n_head,
            n_head_kv,
            n_embd_head,
            ffn_dim,
            d_conv: 2, // kernel_size=3, d_conv=2
            rms_norm_eps: 1e-5,
            rope_freq_base: 1_000_000.0,
            swa_window_size: 30,
            n_codes: 8,
            n_fft: 1280,
            hop_length: 320,
            sample_rate: 24000,
            layer_is_conv,
        };

        let output_norm = gguf.get_tensor("lfm.embedding_norm.weight")?.to_f32_vec();

        let emb_weight = F32Weight::from_tensor(gguf, "emb.emb.weight")?;
        let lin_w = F32Weight::from_tensor(gguf, "lin.weight")?;
        let lin_b = gguf.get_tensor("lin.bias")?.to_f32_vec();

        let mut layers = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let prefix = format!("lfm.layers.{i}");
            let is_conv = config.layer_is_conv[i];

            layers.push(DetokLayerWeights {
                operator_norm: gguf
                    .get_tensor(&format!("{prefix}.operator_norm.weight"))?
                    .to_f32_vec(),
                ffn_norm: gguf
                    .get_tensor(&format!("{prefix}.ffn_norm.weight"))?
                    .to_f32_vec(),
                ffn_w1: F32Weight::from_tensor(gguf, &format!("{prefix}.feed_forward.w1.weight"))?,
                ffn_w2: F32Weight::from_tensor(gguf, &format!("{prefix}.feed_forward.w2.weight"))?,
                ffn_w3: F32Weight::from_tensor(gguf, &format!("{prefix}.feed_forward.w3.weight"))?,
                conv_in_proj: if is_conv {
                    Some(F32Weight::from_tensor(
                        gguf,
                        &format!("{prefix}.conv.in_proj.weight"),
                    )?)
                } else {
                    None
                },
                conv_out_proj: if is_conv {
                    Some(F32Weight::from_tensor(
                        gguf,
                        &format!("{prefix}.conv.out_proj.weight"),
                    )?)
                } else {
                    None
                },
                conv_weight: if is_conv {
                    Some(
                        gguf.get_tensor(&format!("{prefix}.conv.conv.weight"))?
                            .to_f32_vec(),
                    )
                } else {
                    None
                },
                wq: if !is_conv {
                    Some(F32Weight::from_tensor(
                        gguf,
                        &format!("{prefix}.self_attn.q_proj.weight"),
                    )?)
                } else {
                    None
                },
                wk: if !is_conv {
                    Some(F32Weight::from_tensor(
                        gguf,
                        &format!("{prefix}.self_attn.k_proj.weight"),
                    )?)
                } else {
                    None
                },
                wv: if !is_conv {
                    Some(F32Weight::from_tensor(
                        gguf,
                        &format!("{prefix}.self_attn.v_proj.weight"),
                    )?)
                } else {
                    None
                },
                wo: if !is_conv {
                    Some(F32Weight::from_tensor(
                        gguf,
                        &format!("{prefix}.self_attn.out_proj.weight"),
                    )?)
                } else {
                    None
                },
                q_norm: if !is_conv {
                    Some(
                        gguf.get_tensor(&format!("{prefix}.self_attn.q_layernorm.weight"))?
                            .to_f32_vec(),
                    )
                } else {
                    None
                },
                k_norm: if !is_conv {
                    Some(
                        gguf.get_tensor(&format!("{prefix}.self_attn.k_layernorm.weight"))?
                            .to_f32_vec(),
                    )
                } else {
                    None
                },
            });
        }

        eprintln!(
            "Detokenizer loaded: {}L, embd={}, head={}/{}, ffn={}, conv_layers={}",
            n_layer,
            n_embd,
            n_head,
            n_head_kv,
            ffn_dim,
            config.layer_is_conv.iter().filter(|&&c| c).count()
        );

        Ok(Self {
            config,
            output_norm,
            emb_weight,
            lin_w,
            lin_b,
            layers,
        })
    }
}

// ── Detokenizer forward pass ────────────────────────────────────────────

/// Runtime state for the detokenizer's LFM2 backbone.
pub struct DetokenizerState {
    /// Per-layer conv rolling buffer [d_conv, n_embd].
    conv_bufs: Vec<Vec<f32>>,
    /// Per-layer KV cache for attention layers.
    attn_kv: Vec<Option<(Vec<f32>, Vec<f32>)>>,
    n_past: usize,
}

impl DetokenizerState {
    pub fn new(cfg: &DetokenizerConfig) -> Self {
        let mut conv_bufs = Vec::new();
        let mut attn_kv = Vec::new();
        let kv_size = cfg.swa_window_size * cfg.n_head_kv * cfg.n_embd_head;
        for &is_conv in &cfg.layer_is_conv {
            if is_conv {
                conv_bufs.push(vec![0.0; cfg.d_conv * cfg.n_embd]);
                attn_kv.push(None);
            } else {
                conv_bufs.push(vec![]); // placeholder
                attn_kv.push(Some((vec![0.0; kv_size], vec![0.0; kv_size])));
            }
        }
        Self {
            conv_bufs,
            attn_kv,
            n_past: 0,
        }
    }

    pub fn reset(&mut self) {
        for buf in &mut self.conv_bufs {
            buf.fill(0.0);
        }
        for kv in &mut self.attn_kv {
            if let Some((k, v)) = kv {
                k.fill(0.0);
                v.fill(0.0);
            }
        }
        self.n_past = 0;
    }
}

/// Embed 8 audio codes into a single vector for the detokenizer.
/// Looks up each code with per-codebook offset, averages across codebooks.
pub fn detok_embed_codes(weights: &DetokenizerWeights, codes: &[i32]) -> Vec<f32> {
    let n_codes = weights.config.n_codes;
    let emb = &weights.emb_weight;
    let emb_dim = emb.cols;
    let n_vocab_per_cb = emb.rows / n_codes;

    let mut result = vec![0.0f32; emb_dim];
    for (j, &code) in codes.iter().enumerate() {
        let idx = j * n_vocab_per_cb + code as usize;
        let row = &emb.data[idx * emb_dim..(idx + 1) * emb_dim];
        for (r, e) in result.iter_mut().zip(row) {
            *r += e;
        }
    }
    // Mean across codebooks.
    let scale = 1.0 / n_codes as f32;
    for r in &mut result {
        *r *= scale;
    }
    result
}

/// Linear interpolation upsample: 1 token → n_up tokens.
pub fn upsample(input: &[f32], n_embd: usize, n_up: usize) -> Vec<f32> {
    let n_in = input.len() / n_embd;
    let n_out = n_in * n_up;
    let mut output = vec![0.0; n_out * n_embd];
    for i in 0..n_out {
        let src_f = i as f32 / n_up as f32;
        let src_i = src_f as usize;
        let frac = src_f - src_i as f32;
        let i0 = src_i.min(n_in - 1);
        let i1 = (src_i + 1).min(n_in - 1);
        for d in 0..n_embd {
            output[i * n_embd + d] =
                input[i0 * n_embd + d] * (1.0 - frac) + input[i1 * n_embd + d] * frac;
        }
    }
    output
}

/// Process one token through a conv block (gated short conv).
fn detok_conv_block(
    lw: &DetokLayerWeights,
    conv_buf: &mut [f32],
    cur: &[f32],
    n_embd: usize,
    d_conv: usize,
) -> Vec<f32> {
    // in_proj → [b, c, x] chunks.
    let chunk_size = n_embd;
    let mut bcx = vec![0.0; 3 * chunk_size];
    lw.conv_in_proj.as_ref().unwrap().gemv(cur, &mut bcx);

    let b = &bcx[..chunk_size];
    let c = &bcx[chunk_size..2 * chunk_size];
    let x = &bcx[2 * chunk_size..];

    // bx = b * x
    let mut bx: Vec<f32> = b.iter().zip(x).map(|(bi, xi)| bi * xi).collect();

    // conv1d with rolling buffer: concat [conv_buf, bx], convolve, update buf.
    let kernel_size = d_conv + 1;
    let conv_w = lw.conv_weight.as_ref().unwrap();
    let mut conv_out = vec![0.0; chunk_size];
    // The rolling buffer has d_conv previous bx vectors.
    // Kernel applies: sum over k of weight[k] * input[pos - d_conv + k]
    // GGUF stores conv.weight as [kernel_size, n_embd] with dim0 (kernel) fastest.
    // Element (k, ch) is at index ch * kernel_size + k.
    let kernel_size = d_conv + 1;
    for ch in 0..chunk_size {
        let mut sum = 0.0;
        for k in 0..d_conv {
            sum += conv_buf[k * n_embd + ch] * conv_w[ch * kernel_size + k];
        }
        sum += bx[ch] * conv_w[ch * kernel_size + d_conv];
        conv_out[ch] = sum;
    }

    // Update rolling buffer: shift left, append bx.
    if d_conv > 1 {
        let row_size = n_embd;
        for k in 0..d_conv - 1 {
            let src = (k + 1) * row_size;
            let dst = k * row_size;
            conv_buf.copy_within(src..src + row_size, dst);
        }
    }
    conv_buf[(d_conv - 1) * n_embd..d_conv * n_embd].copy_from_slice(&bx);

    // y = c * conv_out
    let y: Vec<f32> = c.iter().zip(&conv_out).map(|(ci, co)| ci * co).collect();

    // out_proj
    let mut out = vec![0.0; n_embd];
    lw.conv_out_proj.as_ref().unwrap().gemv(&y, &mut out);
    out
}

/// Process one token through an attention block.
fn detok_attn_block(
    lw: &DetokLayerWeights,
    kv: &mut (Vec<f32>, Vec<f32>),
    cur: &[f32],
    pos: usize,
    cfg: &DetokenizerConfig,
) -> Vec<f32> {
    let n_embd = cfg.n_embd;
    let n_head = cfg.n_head;
    let n_kv = cfg.n_head_kv;
    let hd = cfg.n_embd_head;

    let mut q = vec![0.0; n_head * hd];
    let mut k = vec![0.0; n_kv * hd];
    let mut v = vec![0.0; n_kv * hd];
    lw.wq.as_ref().unwrap().gemv(cur, &mut q);
    lw.wk.as_ref().unwrap().gemv(cur, &mut k);
    lw.wv.as_ref().unwrap().gemv(cur, &mut v);

    // Per-head norm + RoPE.
    let q_norm = lw.q_norm.as_ref().unwrap();
    let k_norm = lw.k_norm.as_ref().unwrap();
    for h in 0..n_head {
        cpu::rmsnorm(&mut q[h * hd..(h + 1) * hd], q_norm, cfg.rms_norm_eps);
        apply_rope_neox(&mut q[h * hd..(h + 1) * hd], pos, hd, cfg.rope_freq_base);
    }
    for h in 0..n_kv {
        cpu::rmsnorm(&mut k[h * hd..(h + 1) * hd], k_norm, cfg.rms_norm_eps);
        apply_rope_neox(&mut k[h * hd..(h + 1) * hd], pos, hd, cfg.rope_freq_base);
    }

    // Write to KV cache (ring buffer for SWA).
    let (k_cache, v_cache) = kv;
    let kv_stride = n_kv * hd;
    let write_pos = pos % cfg.swa_window_size;
    k_cache[write_pos * kv_stride..(write_pos + 1) * kv_stride].copy_from_slice(&k);
    v_cache[write_pos * kv_stride..(write_pos + 1) * kv_stride].copy_from_slice(&v);

    // Attention with sliding window.
    let kv_start = if pos + 1 > cfg.swa_window_size {
        pos + 1 - cfg.swa_window_size
    } else {
        0
    };
    let kv_len = pos + 1 - kv_start;
    let group_size = n_head / n_kv;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut attn_out = vec![0.0; n_head * hd];

    for h in 0..n_head {
        let kv_h = h / group_size;
        let q_head = &q[h * hd..(h + 1) * hd];
        let mut scores = vec![0.0f32; kv_len];
        for t in 0..kv_len {
            let cache_pos = (kv_start + t) % cfg.swa_window_size;
            let k_off = cache_pos * kv_stride + kv_h * hd;
            let k_t = &k_cache[k_off..k_off + hd];
            scores[t] = q_head.iter().zip(k_t).map(|(a, b)| a * b).sum::<f32>() * scale;
        }
        cpu::softmax_inplace(&mut scores);

        let out = &mut attn_out[h * hd..(h + 1) * hd];
        for t in 0..kv_len {
            let cache_pos = (kv_start + t) % cfg.swa_window_size;
            let v_off = cache_pos * kv_stride + kv_h * hd;
            let v_t = &v_cache[v_off..v_off + hd];
            let s = scores[t];
            for d in 0..hd {
                out[d] += s * v_t[d];
            }
        }
    }

    let mut proj = vec![0.0; n_embd];
    lw.wo.as_ref().unwrap().gemv(&attn_out, &mut proj);
    proj
}

/// Run the detokenizer: codes → spectrogram frames (before ISTFT).
///
/// Returns raw spectrogram data as [n_frames × n_fft_bins × 2] (log_abs, angle).
pub fn detokenize_to_spectrum(
    weights: &DetokenizerWeights,
    detok_weights: &AudioDecoderWeights,
    state: &mut DetokenizerState,
    codes: &[i32],
) -> Vec<f32> {
    let cfg = &weights.config;
    let n_embd = cfg.n_embd;

    // 1. Embed codes → single vector.
    let embedding = detok_embed_codes(weights, codes);

    // 2. Upsample 1 → 6 tokens.
    let tokens = upsample(&embedding, n_embd, 6);
    let n_tokens = tokens.len() / n_embd;

    // 3. LFM2 backbone: layer-by-layer with all n_tokens.
    //    Conv layers: sequential (rolling buffer). Attention: batch (all tokens visible).
    let mut hidden = tokens.clone(); // [n_tokens × n_embd] flat

    for (il, lw) in weights.layers.iter().enumerate() {
        let mut new_hidden = Vec::with_capacity(n_tokens * n_embd);

        if cfg.layer_is_conv[il] {
            for t in 0..n_tokens {
                let mut cur = hidden[t * n_embd..(t + 1) * n_embd].to_vec();
                let residual = cur.clone();
                cpu::rmsnorm(&mut cur, &lw.operator_norm, cfg.rms_norm_eps);
                let out = detok_conv_block(lw, &mut state.conv_bufs[il], &cur, n_embd, cfg.d_conv);
                cur = residual.iter().zip(&out).map(|(r, o)| r + o).collect();
                let ffn_res = cur.clone();
                cpu::rmsnorm(&mut cur, &lw.ffn_norm, cfg.rms_norm_eps);
                let mut gate = vec![0.0; cfg.ffn_dim];
                let mut up = vec![0.0; cfg.ffn_dim];
                lw.ffn_w1.gemv(&cur, &mut gate);
                lw.ffn_w3.gemv(&cur, &mut up);
                cpu::silu_mul_inplace(&mut gate, &up);
                let mut down = vec![0.0; n_embd];
                lw.ffn_w2.gemv(&gate, &mut down);
                cur = ffn_res.iter().zip(&down).map(|(r, d)| r + d).collect();
                new_hidden.extend_from_slice(&cur);
            }
        } else {
            // Attention: write ALL tokens' K/V, then each token attends to ALL.
            let kv = state.attn_kv[il].as_mut().unwrap();
            let n_head = cfg.n_head;
            let n_kv = cfg.n_head_kv;
            let hd = cfg.n_embd_head;
            let group_size = n_head / n_kv;
            let scale = 1.0 / (hd as f32).sqrt();
            let kv_stride = n_kv * hd;
            let mut all_q = vec![0.0f32; n_tokens * n_head * hd];

            for t in 0..n_tokens {
                let pos = state.n_past + t;
                let mut cur = hidden[t * n_embd..(t + 1) * n_embd].to_vec();
                cpu::rmsnorm(&mut cur, &lw.operator_norm, cfg.rms_norm_eps);
                let mut q = vec![0.0; n_head * hd];
                let mut k = vec![0.0; n_kv * hd];
                let mut v = vec![0.0; n_kv * hd];
                lw.wq.as_ref().unwrap().gemv(&cur, &mut q);
                lw.wk.as_ref().unwrap().gemv(&cur, &mut k);
                lw.wv.as_ref().unwrap().gemv(&cur, &mut v);
                for h in 0..n_head {
                    cpu::rmsnorm(
                        &mut q[h * hd..(h + 1) * hd],
                        lw.q_norm.as_ref().unwrap(),
                        cfg.rms_norm_eps,
                    );
                    apply_rope_neox(&mut q[h * hd..(h + 1) * hd], pos, hd, cfg.rope_freq_base);
                }
                for h in 0..n_kv {
                    cpu::rmsnorm(
                        &mut k[h * hd..(h + 1) * hd],
                        lw.k_norm.as_ref().unwrap(),
                        cfg.rms_norm_eps,
                    );
                    apply_rope_neox(&mut k[h * hd..(h + 1) * hd], pos, hd, cfg.rope_freq_base);
                }
                let wp = pos % cfg.swa_window_size;
                kv.0[wp * kv_stride..(wp + 1) * kv_stride].copy_from_slice(&k);
                kv.1[wp * kv_stride..(wp + 1) * kv_stride].copy_from_slice(&v);
                all_q[t * n_head * hd..(t + 1) * n_head * hd].copy_from_slice(&q);
            }

            for t in 0..n_tokens {
                let q = &all_q[t * n_head * hd..(t + 1) * n_head * hd];
                let kv_end = state.n_past + n_tokens;
                let kv_start = kv_end.saturating_sub(cfg.swa_window_size);
                let kv_len = kv_end - kv_start;
                let mut attn_out = vec![0.0; n_head * hd];
                for h in 0..n_head {
                    let kv_h = h / group_size;
                    let qh = &q[h * hd..(h + 1) * hd];
                    let mut scores = vec![0.0f32; kv_len];
                    for tt in 0..kv_len {
                        let cp = (kv_start + tt) % cfg.swa_window_size;
                        let ko = cp * kv_stride + kv_h * hd;
                        scores[tt] = qh
                            .iter()
                            .zip(&kv.0[ko..ko + hd])
                            .map(|(a, b)| a * b)
                            .sum::<f32>()
                            * scale;
                    }
                    cpu::softmax_inplace(&mut scores);
                    let out = &mut attn_out[h * hd..(h + 1) * hd];
                    for tt in 0..kv_len {
                        let cp = (kv_start + tt) % cfg.swa_window_size;
                        let vo = cp * kv_stride + kv_h * hd;
                        let s = scores[tt];
                        for d in 0..hd {
                            out[d] += s * kv.1[vo + d];
                        }
                    }
                }
                let mut proj = vec![0.0; n_embd];
                lw.wo.as_ref().unwrap().gemv(&attn_out, &mut proj);
                let residual = &hidden[t * n_embd..(t + 1) * n_embd];
                let mut cur: Vec<f32> = residual.iter().zip(&proj).map(|(r, p)| r + p).collect();
                let ffn_res = cur.clone();
                cpu::rmsnorm(&mut cur, &lw.ffn_norm, cfg.rms_norm_eps);
                let mut gate = vec![0.0; cfg.ffn_dim];
                let mut up = vec![0.0; cfg.ffn_dim];
                lw.ffn_w1.gemv(&cur, &mut gate);
                lw.ffn_w3.gemv(&cur, &mut up);
                cpu::silu_mul_inplace(&mut gate, &up);
                let mut down = vec![0.0; n_embd];
                lw.ffn_w2.gemv(&gate, &mut down);
                cur = ffn_res.iter().zip(&down).map(|(r, d)| r + d).collect();
                new_hidden.extend_from_slice(&cur);
            }
        }
        hidden = new_hidden;
    }

    state.n_past += n_tokens;

    // Output norm per token.
    let mut outputs = Vec::with_capacity(n_tokens * n_embd);
    for t in 0..n_tokens {
        let mut cur = hidden[t * n_embd..(t + 1) * n_embd].to_vec();
        cpu::rmsnorm(&mut cur, &weights.output_norm, cfg.rms_norm_eps);
        outputs.extend_from_slice(&cur);
    }
    // 4. Linear projection → spectrogram.
    let lin_out_dim = weights.lin_w.rows; // n_fft_bins * 2 = 1282
    let mut spectrum = Vec::with_capacity(n_tokens * lin_out_dim);
    for t in 0..n_tokens {
        let hidden = &outputs[t * n_embd..(t + 1) * n_embd];
        let mut frame = vec![0.0; lin_out_dim];
        weights.lin_w.gemv(hidden, &mut frame);
        for (f, b) in frame.iter_mut().zip(&weights.lin_b) {
            *f += b;
        }
        spectrum.extend_from_slice(&frame);
    }

    spectrum
}

/// Convert spectrogram frames to PCM audio samples via ISTFT.
///
/// Input: [n_frames × n_fft_bins × 2] where each pair is (log_abs, angle).
/// Output: PCM float32 samples at the configured sample rate.
/// Streaming ISTFT matching llama.cpp's `mtmd_audio_streaming_istft`.
///
/// Processes frames one at a time with shift-and-accumulate overlap buffers.
/// Each frame produces exactly `hop_length` output samples.
pub fn istft_to_pcm(spectrum: &[f32], n_fft: usize, hop_length: usize) -> Vec<f32> {
    let n_fft_bins = n_fft / 2 + 1;
    let frame_size = n_fft_bins * 2;
    let n_frames = spectrum.len() / frame_size;
    if n_frames == 0 {
        return vec![];
    }

    // Periodic Hann window: w[n] = 0.5 * (1 - cos(2π*n/N)).
    let hann: Vec<f32> = (0..n_fft)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / n_fft as f32).cos()))
        .collect();

    // Streaming overlap buffers (size = n_fft).
    let mut overlap_buf = vec![0.0f32; n_fft];
    let mut window_sum = vec![0.0f32; n_fft];
    let mut output = Vec::with_capacity(n_frames * hop_length);

    for i in 0..n_frames {
        // Convert (log_abs, angle) → interleaved complex.
        let mut complex = vec![(0.0f32, 0.0f32); n_fft];
        for j in 0..n_fft_bins {
            let log_abs = spectrum[i * frame_size + j];
            let angle = spectrum[i * frame_size + n_fft_bins + j];
            let mag = log_abs.exp();
            complex[j] = (mag * angle.cos(), mag * angle.sin());
        }
        // Mirror negative frequencies (conjugate).
        for j in 1..n_fft_bins - 1 {
            complex[n_fft - j] = (complex[j].0, -complex[j].1);
        }

        // IFFT.
        let time_domain = ifft_frame(&complex, n_fft);

        // Add to overlap buffer with Hann window.
        for j in 0..n_fft {
            overlap_buf[j] += time_domain[j] * hann[j];
            window_sum[j] += hann[j] * hann[j];
        }

        // Extract hop_length samples with normalization.
        for k in 0..hop_length {
            let sample = if window_sum[k] > 1e-8 {
                overlap_buf[k] / window_sum[k]
            } else {
                overlap_buf[k]
            };
            output.push(sample);
        }

        // Shift buffers left by hop_length.
        overlap_buf.copy_within(hop_length..n_fft, 0);
        overlap_buf[n_fft - hop_length..].fill(0.0);
        window_sum.copy_within(hop_length..n_fft, 0);
        window_sum[n_fft - hop_length..].fill(0.0);
    }

    // Strip startup padding: the first (n_fft - hop_length) / 2 samples are
    // overlap-add artifacts. Matches llama.cpp's `padding_to_remove`.
    let padding = (n_fft - hop_length) / 2;
    if output.len() > padding {
        output.drain(..padding);
    }
    output
}

/// Inverse FFT of one frame. Uses rustfft when the `audio` feature is enabled,
/// otherwise falls back to naive O(n²) DFT.
fn ifft_frame(complex: &[(f32, f32)], n_fft: usize) -> Vec<f32> {
    #[cfg(feature = "audio")]
    {
        use rustfft::{FftPlanner, num_complex::Complex32};
        let mut planner = FftPlanner::new();
        let ifft = planner.plan_fft_inverse(n_fft);
        let mut buf: Vec<Complex32> = complex
            .iter()
            .map(|&(re, im)| Complex32::new(re, im))
            .collect();
        ifft.process(&mut buf);
        let inv_n = 1.0 / n_fft as f32;
        buf.iter().map(|c| c.re * inv_n).collect()
    }
    #[cfg(not(feature = "audio"))]
    {
        // Naive O(n²) DFT fallback.
        let inv_n = 1.0 / n_fft as f32;
        (0..n_fft)
            .map(|n| {
                let mut sum = 0.0f32;
                for k in 0..n_fft {
                    let angle = 2.0 * std::f32::consts::PI * k as f32 * n as f32 * inv_n;
                    sum += complex[k].0 * angle.cos() - complex[k].1 * angle.sin();
                }
                sum * inv_n
            })
            .collect()
    }
}
