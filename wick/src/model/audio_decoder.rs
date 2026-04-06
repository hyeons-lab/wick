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
