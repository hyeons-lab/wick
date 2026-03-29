pub mod lfm2;
pub mod llama;

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
}

/// Trait for loaded models that can run forward passes.
pub trait Model: Send {
    /// Run a forward pass and return logits over the vocabulary.
    fn forward(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) -> Vec<f32>;

    /// Get the model configuration.
    fn config(&self) -> &ModelConfig;
}
