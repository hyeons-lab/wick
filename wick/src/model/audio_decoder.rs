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

/// Per-layer weights for one depthformer layer.
pub struct DepthformerLayerWeights {
    pub operator_norm: Vec<f32>,
    pub wqkv: F32Weight,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub wo: F32Weight,
    pub ffn_norm: Vec<f32>,
    pub w1: F32Weight, // gate
    pub w2: F32Weight, // down
    pub w3: F32Weight, // up
}

/// Per-codebook embedding layer weights.
pub struct CodebookWeights {
    pub embedding: F32Weight,
    pub norm: Vec<f32>,
    pub to_logits: F32Weight,
}

/// All weights for the audio decoder (loaded from vocoder GGUF).
pub struct AudioDecoderWeights {
    pub depthformer_config: DepthformerConfig,
    pub decoder_config: DecoderConfig,
    pub depthformer_layers: Vec<DepthformerLayerWeights>,
    pub depth_linear_w: F32Weight,
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
                wqkv: F32Weight::from_tensor(gguf, &format!("{prefix}.operator.qkv_proj.weight"))?,
                q_norm: gguf
                    .get_tensor(&format!("{prefix}.operator.attention.q_layernorm.weight"))?
                    .to_f32_vec(),
                k_norm: gguf
                    .get_tensor(&format!("{prefix}.operator.attention.k_layernorm.weight"))?
                    .to_f32_vec(),
                wo: F32Weight::from_tensor(gguf, &format!("{prefix}.operator.out_proj.weight"))?,
                ffn_norm: gguf
                    .get_tensor(&format!("{prefix}.ffn_norm.weight"))?
                    .to_f32_vec(),
                w1: F32Weight::from_tensor(gguf, &format!("{prefix}.feed_forward.w1.weight"))?,
                w2: F32Weight::from_tensor(gguf, &format!("{prefix}.feed_forward.w2.weight"))?,
                w3: F32Weight::from_tensor(gguf, &format!("{prefix}.feed_forward.w3.weight"))?,
            });
        }

        // Decoder model weights.
        let depth_linear_w = F32Weight::from_tensor(gguf, "depth_linear.weight")?;
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
                to_logits: F32Weight::from_tensor(gguf, &format!("{prefix}.to_logits.weight"))?,
            });
        }

        let audio_embedding = CodebookWeights {
            embedding: F32Weight::from_tensor(gguf, "audio_embedding.embedding.weight")?,
            norm: gguf
                .get_tensor("audio_embedding.embedding_norm.weight")?
                .to_f32_vec(),
            to_logits: F32Weight::from_tensor(gguf, "audio_embedding.to_logits.weight")?,
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

/// Apply RoPE (neox style) to a single head's Q or K vector in-place.
fn apply_rope(x: &mut [f32], pos: usize, head_dim: usize, freq_base: f32) {
    let half = head_dim / 2;
    for i in 0..half {
        let freq = 1.0 / freq_base.powf(i as f32 / half as f32);
        let theta = pos as f32 * freq;
        let (sin_t, cos_t) = theta.sin_cos();
        let x0 = x[i];
        let x1 = x[i + half];
        x[i] = x0 * cos_t - x1 * sin_t;
        x[i + half] = x0 * sin_t + x1 * cos_t;
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

    let mut cur = input.to_vec();
    assert_eq!(cur.len(), n_embd);

    for (il, lw) in weights.depthformer_layers.iter().enumerate() {
        let residual = cur.clone();

        // 1. RMSnorm.
        cpu::rmsnorm(&mut cur, &lw.operator_norm, cfg.rms_norm_eps);

        // 2. Fused QKV projection → split.
        let qkv_dim = (n_head + 2 * n_kv) * hd;
        let mut qkv = vec![0.0; qkv_dim];
        lw.wqkv.gemv(&cur, &mut qkv);

        let q_dim = n_head * hd;
        let k_dim = n_kv * hd;
        let mut q = qkv[..q_dim].to_vec();
        let mut k = qkv[q_dim..q_dim + k_dim].to_vec();
        let v = qkv[q_dim + k_dim..].to_vec();

        // 3. Per-head RMSnorm on Q and K.
        for h in 0..n_head {
            let s = &mut q[h * hd..(h + 1) * hd];
            cpu::rmsnorm(s, &lw.q_norm, cfg.rms_norm_eps);
        }
        for h in 0..n_kv {
            let s = &mut k[h * hd..(h + 1) * hd];
            cpu::rmsnorm(s, &lw.k_norm, cfg.rms_norm_eps);
        }

        // 4. RoPE on Q and K.
        for h in 0..n_head {
            apply_rope(&mut q[h * hd..(h + 1) * hd], pos, hd, cfg.rope_freq_base);
        }
        for h in 0..n_kv {
            apply_rope(&mut k[h * hd..(h + 1) * hd], pos, hd, cfg.rope_freq_base);
        }

        // 5. Write K, V to cache at position n_past.
        let kv = &mut state.kv[il];
        for h in 0..n_kv {
            let cache_off = pos * n_kv * hd + h * hd;
            kv.k[cache_off..cache_off + hd].copy_from_slice(&k[h * hd..(h + 1) * hd]);
            kv.v[cache_off..cache_off + hd].copy_from_slice(&v[h * hd..(h + 1) * hd]);
        }

        // 6. Attention: Q×K → softmax → ×V (GQA with group_size = n_head/n_kv).
        let seq_len = pos + 1;
        let group_size = n_head / n_kv;
        let scale = 1.0 / (hd as f32).sqrt();
        let mut attn_out = vec![0.0; n_head * hd];

        for h in 0..n_head {
            let kv_h = h / group_size;
            let q_head = &q[h * hd..(h + 1) * hd];

            // Q×K scores.
            let mut scores = vec![0.0f32; seq_len];
            for t in 0..seq_len {
                let k_off = t * n_kv * hd + kv_h * hd;
                let k_t = &kv.k[k_off..k_off + hd];
                scores[t] = q_head.iter().zip(k_t).map(|(a, b)| a * b).sum::<f32>() * scale;
            }
            cpu::softmax_inplace(&mut scores);

            // Weighted V sum.
            let out = &mut attn_out[h * hd..(h + 1) * hd];
            for t in 0..seq_len {
                let v_off = t * n_kv * hd + kv_h * hd;
                let v_t = &kv.v[v_off..v_off + hd];
                let s = scores[t];
                for d in 0..hd {
                    out[d] += s * v_t[d];
                }
            }
        }

        // 7. Out projection + residual.
        let mut proj = vec![0.0; n_embd];
        lw.wo.gemv(&attn_out, &mut proj);
        cur = residual.iter().zip(&proj).map(|(r, p)| r + p).collect();

        // 8. FFN: RMSnorm → SwiGLU(w1, w3) → w2 → residual.
        let ffn_residual = cur.clone();
        cpu::rmsnorm(&mut cur, &lw.ffn_norm, cfg.rms_norm_eps);

        let mut gate = vec![0.0; cfg.ffn_dim];
        let mut up = vec![0.0; cfg.ffn_dim];
        lw.w1.gemv(&cur, &mut gate);
        lw.w3.gemv(&cur, &mut up);
        cpu::silu_mul_inplace(&mut gate, &up);

        let mut down = vec![0.0; n_embd];
        lw.w2.gemv(&gate, &mut down);
        cur = ffn_residual.iter().zip(&down).map(|(r, d)| r + d).collect();
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
        let w = &weights.depth_linear_w;
        let row_start = j * n_embd_d;
        for (r, out) in depthformer_in.iter_mut().enumerate() {
            let row = &w.data[(row_start + r) * w.cols..(row_start + r + 1) * w.cols];
            *out = row.iter().zip(embedding).map(|(a, b)| a * b).sum::<f32>()
                + weights.depth_linear_b[row_start + r];
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
fn detok_embed_codes(weights: &DetokenizerWeights, codes: &[i32]) -> Vec<f32> {
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
fn upsample(input: &[f32], n_embd: usize, n_up: usize) -> Vec<f32> {
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
    for ch in 0..chunk_size {
        let mut sum = 0.0;
        for k in 0..d_conv {
            sum += conv_buf[k * n_embd + ch] * conv_w[k * n_embd + ch];
        }
        sum += bx[ch] * conv_w[d_conv * n_embd + ch];
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
        apply_rope(&mut q[h * hd..(h + 1) * hd], pos, hd, cfg.rope_freq_base);
    }
    for h in 0..n_kv {
        cpu::rmsnorm(&mut k[h * hd..(h + 1) * hd], k_norm, cfg.rms_norm_eps);
        apply_rope(&mut k[h * hd..(h + 1) * hd], pos, hd, cfg.rope_freq_base);
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

    // 3. LFM2 backbone: process each token sequentially.
    let mut outputs = Vec::with_capacity(n_tokens * n_embd);
    for t in 0..n_tokens {
        let pos = state.n_past + t;
        let mut cur = tokens[t * n_embd..(t + 1) * n_embd].to_vec();

        for (il, lw) in weights.layers.iter().enumerate() {
            let residual = cur.clone();
            cpu::rmsnorm(&mut cur, &lw.operator_norm, cfg.rms_norm_eps);

            let block_out = if cfg.layer_is_conv[il] {
                detok_conv_block(lw, &mut state.conv_bufs[il], &cur, n_embd, cfg.d_conv)
            } else {
                detok_attn_block(lw, state.attn_kv[il].as_mut().unwrap(), &cur, pos, cfg)
            };

            cur = residual
                .iter()
                .zip(&block_out)
                .map(|(r, b)| r + b)
                .collect();

            // FFN.
            let ffn_residual = cur.clone();
            cpu::rmsnorm(&mut cur, &lw.ffn_norm, cfg.rms_norm_eps);
            let mut gate = vec![0.0; cfg.ffn_dim];
            let mut up = vec![0.0; cfg.ffn_dim];
            lw.ffn_w1.gemv(&cur, &mut gate);
            lw.ffn_w3.gemv(&cur, &mut up);
            cpu::silu_mul_inplace(&mut gate, &up);
            let mut down = vec![0.0; n_embd];
            lw.ffn_w2.gemv(&gate, &mut down);
            cur = ffn_residual.iter().zip(&down).map(|(r, d)| r + d).collect();
        }

        // Output norm.
        cpu::rmsnorm(&mut cur, &weights.output_norm, cfg.rms_norm_eps);
        outputs.extend_from_slice(&cur);
    }

    state.n_past += n_tokens;

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
pub fn istft_to_pcm(spectrum: &[f32], n_fft: usize, hop_length: usize) -> Vec<f32> {
    let n_fft_bins = n_fft / 2 + 1;
    let frame_size = n_fft_bins * 2;
    let n_frames = spectrum.len() / frame_size;
    if n_frames == 0 {
        return vec![];
    }

    let output_len = (n_frames - 1) * hop_length;
    let mut output = vec![0.0f32; output_len + n_fft];
    let mut window_sum = vec![0.0f32; output_len + n_fft];

    // Hann window.
    let hann: Vec<f32> = (0..n_fft)
        .map(|i| {
            let t = std::f32::consts::PI * i as f32 / n_fft as f32;
            t.sin().powi(2)
        })
        .collect();

    for i in 0..n_frames {
        // Extract complex spectrum: (log_abs, angle) → complex.
        let mut complex = vec![(0.0f32, 0.0f32); n_fft];
        for j in 0..n_fft_bins {
            let log_abs = spectrum[i * frame_size + j];
            let angle = spectrum[i * frame_size + n_fft_bins + j];
            let mag = log_abs.exp();
            complex[j] = (mag * angle.cos(), mag * angle.sin());
        }
        // Mirror for negative frequencies.
        for j in 1..n_fft_bins - 1 {
            complex[n_fft - j] = (complex[j].0, -complex[j].1);
        }

        // IFFT (simple DFT — n_fft=1280 is small enough).
        let mut time_domain = vec![0.0f32; n_fft];
        let inv_n = 1.0 / n_fft as f32;
        for n in 0..n_fft {
            let mut sum = 0.0f32;
            for k in 0..n_fft {
                let angle = 2.0 * std::f32::consts::PI * k as f32 * n as f32 * inv_n;
                sum += complex[k].0 * angle.cos() - complex[k].1 * angle.sin();
            }
            time_domain[n] = sum * inv_n;
        }

        // Window + overlap-add.
        let offset = i * hop_length;
        for n in 0..n_fft {
            output[offset + n] += time_domain[n] * hann[n];
            window_sum[offset + n] += hann[n] * hann[n];
        }
    }

    // Normalize by window sum.
    for i in 0..output.len() {
        if window_sum[i] > 1e-8 {
            output[i] /= window_sum[i];
        }
    }

    output.truncate(output_len);
    output
}
