pub mod lfm2;
pub mod llama;

use anyhow::{Result, bail};

use crate::gguf::GgufFile;
use crate::kv_cache::InferenceState;

/// Per-layer block type (for hybrid architectures like LFM2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockType {
    Attention,
    GatedConv,
}

/// Model configuration extracted from GGUF metadata.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub architecture: String,
    pub n_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    /// Per-layer block types. Empty for pure-transformer models.
    pub block_types: Vec<BlockType>,
    /// Convolution kernel size (LFM2-specific).
    pub conv_kernel_size: Option<usize>,
    /// Per-layer KV head counts. Length = n_layers. 0 for conv layers.
    pub kv_heads_per_layer: Vec<usize>,
}

/// Trait for loaded models that can run forward passes.
pub trait Model: Send {
    /// Run a forward pass for a single token and return logits over the vocabulary.
    fn forward(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) -> Vec<f32>;

    /// Batched forward pass for prefill: process all tokens through each layer
    /// using GEMM instead of sequential GEMV. Returns logits for the LAST token only.
    fn forward_prefill(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        // Default: fall back to sequential single-token forward
        let mut logits = Vec::new();
        for (i, &token) in tokens.iter().enumerate() {
            logits = self.forward(&[token], start_pos + i, state);
        }
        logits
    }

    /// Get the model configuration.
    fn config(&self) -> &ModelConfig;
}

/// Load a model from a GGUF file, dispatching on the architecture.
pub fn load_model(gguf: GgufFile) -> Result<Box<dyn Model>> {
    let arch = gguf
        .get_str("general.architecture")
        .unwrap_or("unknown")
        .to_string();
    match arch.as_str() {
        "lfm2" => Ok(Box::new(lfm2::Lfm2Model::from_gguf(gguf)?)),
        other => bail!("unsupported architecture: {other}"),
    }
}
