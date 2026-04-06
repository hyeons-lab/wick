//! Audio decoder for LFM2.5-Audio: vocoder GGUF loading + weight structures.
//!
//! Three sub-models loaded from the vocoder GGUF:
//! 1. DecoderModel — samples 8 audio codes per frame from LLM embedding
//! 2. DepthformerModel — small 6-layer attention-only transformer (backbone of DecoderModel)
//! 3. Detokenizer — codes → spectrogram → PCM (not loaded here yet — Phase 5)

use anyhow::{Context, Result};

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
