pub mod lfm2;
pub mod llama;

#[cfg(feature = "gpu")]
pub mod gpu_lfm2;

#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal_lfm2;

#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal_audio_decoder;

use anyhow::{Result, bail};

use crate::gguf::GgufFile;
use crate::kv_cache::InferenceState;

/// Per-layer block type (for hybrid architectures like LFM2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

    /// Batched forward pass for prefill: process all prompt tokens at once.
    /// Implementations may use GEMM for linear projections. Returns logits for the LAST token only.
    /// Default: falls back to sequential single-token `forward()` calls.
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

    /// Run a forward pass and return the hidden state BEFORE logit projection.
    /// Used by the audio decoder to extract the LLM embedding for audio frame sampling.
    /// Default: panics (must be overridden by backends that support audio).
    fn forward_embedding(
        &self,
        tokens: &[u32],
        _pos: usize,
        _state: &mut InferenceState,
    ) -> Vec<f32> {
        let _ = tokens;
        unimplemented!("forward_embedding not supported by this backend")
    }

    /// Forward pass with a float embedding as input (instead of a token ID).
    /// Used to feed audio codec embeddings back into the LLM after an audio frame.
    /// Default: panics (must be overridden by backends that support audio).
    fn forward_from_embedding(
        &self,
        _embedding: &[f32],
        _pos: usize,
        _state: &mut InferenceState,
    ) -> Vec<f32> {
        unimplemented!("forward_from_embedding not supported by this backend")
    }

    /// Forward pass with embedding input, returning hidden state (not logits).
    /// Used in audio mode: embedding → layers → hidden state → sample audio → embed → loop.
    fn forward_hidden_from_embedding(
        &self,
        _embedding: &[f32],
        _pos: usize,
        _state: &mut InferenceState,
    ) -> Vec<f32> {
        unimplemented!("forward_hidden_from_embedding not supported by this backend")
    }

    /// Greedy (argmax) fast path. Returns just the selected token id,
    /// avoiding a full logits readback when the caller only needs argmax.
    ///
    /// Default impl falls back to `forward()` + CPU argmax. Backends with
    /// a GPU argmax kernel should override to skip the vocab-sized readback.
    fn forward_greedy(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) -> u32 {
        let logits = self.forward(tokens, pos, state);
        crate::sampler::cpu_argmax(&logits)
    }

    /// GPU memory allocated by this model (bytes). 0 for CPU-only backends.
    fn gpu_memory_bytes(&self) -> u64 {
        0
    }

    /// Configure the KV prefix cache. No-op for backends without caching.
    fn configure_cache(&self, _config: crate::kv_cache::KvCacheConfig) {}

    /// Snapshot the current KV and conv state for prefix caching.
    fn snapshot_state(&self) -> crate::kv_cache::StateSnapshot {
        unimplemented!("snapshot_state not supported by this backend")
    }

    /// Restore a previously snapshotted state. Sets internal seq_len.
    fn restore_state(&self, _snapshot: &crate::kv_cache::StateSnapshot) {
        unimplemented!("restore_state not supported by this backend")
    }

    /// Enable TurboQuant KV cache key compression.
    /// Must be called before inference begins. No-op for models that don't support it.
    fn enable_turboquant(&self, _seed: u64) {}

    /// Whether TurboQuant is enabled on this model.
    fn turboquant_enabled(&self) -> bool {
        false
    }
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

/// Load a model with GPU acceleration.
#[cfg(feature = "gpu")]
pub fn load_model_gpu(gguf: GgufFile, context_size: usize) -> Result<Box<dyn Model>> {
    let arch = gguf
        .get_str("general.architecture")
        .unwrap_or("unknown")
        .to_string();
    match arch.as_str() {
        "lfm2" => Ok(Box::new(gpu_lfm2::GpuLfm2Model::from_gguf(
            gguf,
            context_size,
        )?)),
        other => bail!("unsupported architecture for GPU: {other}"),
    }
}

/// Load a model with native Metal acceleration.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn load_model_metal(
    gguf: GgufFile,
    path: &std::path::Path,
    context_size: usize,
) -> Result<Box<dyn Model>> {
    let arch = gguf
        .get_str("general.architecture")
        .unwrap_or("unknown")
        .to_string();
    match arch.as_str() {
        "lfm2" => Ok(Box::new(metal_lfm2::MetalLfm2Model::from_gguf(
            gguf,
            path,
            context_size,
        )?)),
        other => bail!("unsupported architecture for Metal: {other}"),
    }
}
#[allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::manual_saturating_arithmetic,
    unused_variables
)]
pub mod audio_decoder;
