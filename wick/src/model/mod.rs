pub mod lfm2;
pub mod llama;

#[cfg(feature = "gpu")]
pub mod gpu_lfm2;

#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal_lfm2;

#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal_audio_decoder;

use std::sync::atomic::{AtomicBool, Ordering};

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
///
/// `Send + Sync` is required so `std::sync::Arc<dyn Model>` is itself
/// `Send + Sync`, which is the prerequisite for exposing `Session`
/// through UniFFI's foreign-function boundary (the bindgen'd
/// Kotlin/Swift wrappers move the `Arc` between threads and require
/// both bounds).
///
/// **GPU backends are NOT safe across concurrent `Session`s** even
/// though the type bounds permit it: `MetalLfm2Model` / `GpuLfm2Model`
/// keep per-forward scratch buffers + GPU-resident KV caches in their
/// own state. Two threads cloning the same `Arc<dyn Model>` and
/// running `forward()` concurrently would trample those buffers. The
/// plan's multi-session invariant (one GPU model instance per concurrent
/// Session) is enforced at the caller; this bound only covers the
/// *trait-object shape*, not the GPU-state sharing contract. CPU
/// `Lfm2Model` has no such shared state and is safely shareable across
/// concurrent Sessions.
pub trait Model: Send + Sync {
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

    /// Cancelable chunked prefill. Splits `tokens` into `ubatch`-sized slices,
    /// calls [`Self::forward_prefill`] per chunk, and polls `cancel` between
    /// chunks so long prompts can be interrupted without blocking the
    /// caller for the full monolithic duration.
    ///
    /// Returns `(tokens_processed, last_logits)`:
    /// - `tokens_processed <= tokens.len()`; when cancel fires, equals the
    ///   number of tokens that made it into KV before the flag was
    ///   observed (granularity: one ubatch).
    /// - `last_logits` holds the logits from the final processed chunk —
    ///   `Some` whenever any chunk ran. `None` only for the empty-input
    ///   edge case (`tokens.is_empty()`).
    ///
    /// Default impl is correctness-preserving; backend-specific overrides
    /// are free to batch across chunks (none do in v1 — Phase 1.4's
    /// deliberate "probably not in v1" scope). `ubatch == 0` means "no
    /// chunking" (one chunk covering the whole input); this matches the
    /// CLI `--ubatch-size 0` convention for disabling chunking.
    fn forward_prefill_chunked(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut InferenceState,
        ubatch: usize,
        cancel: &AtomicBool,
    ) -> (usize, Option<Vec<f32>>) {
        // `ubatch == 0` → one chunk covering everything (no chunking).
        // Otherwise keep the caller-supplied size.
        let ubatch = if ubatch == 0 {
            tokens.len().max(1)
        } else {
            ubatch
        };
        let mut consumed = 0usize;
        let mut last_logits: Option<Vec<f32>> = None;
        for chunk in tokens.chunks(ubatch) {
            let logits = self.forward_prefill(chunk, start_pos + consumed, state);
            consumed += chunk.len();
            last_logits = Some(logits);
            // Check *after* each chunk so we always make progress on at
            // least one ubatch — avoids the "cancel-before-start leaves
            // the session wedged with no position advance" corner.
            if cancel.load(Ordering::Relaxed) && consumed < tokens.len() {
                break;
            }
        }
        (consumed, last_logits)
    }

    /// Get the model configuration.
    fn config(&self) -> &ModelConfig;

    /// Does this backend support `n_keep` context shift? Static
    /// capability probe — callers MUST check this before invoking
    /// [`Self::shift_kv`].
    ///
    /// The default is `false` so new backends opt in deliberately.
    /// CPU LFM2 overrides to `true`; Metal keeps the default (its
    /// GPU-side K cache needs a shader-based shift that lands as a
    /// follow-up). Non-RoPE architectures also stay `false` — the
    /// shift semantics differ per positional-encoding scheme.
    fn supports_kv_shift(&self) -> bool {
        false
    }

    /// Execute a `n_keep` context shift on this model's state. Drops
    /// attention KV cells `[n_keep .. n_keep + shift)` and re-rotates
    /// remaining K vectors so their RoPE-encoded position matches
    /// their new index. Implemented by overriding; the default is a
    /// no-op, consistent with the default `false` from
    /// [`Self::supports_kv_shift`].
    ///
    /// Callers (today: `Session::append_tokens`) MUST verify
    /// `supports_kv_shift()` is `true` before invoking this. Calling
    /// the default no-op on an overflowed state would leave
    /// `InferenceState` unchanged while the caller proceeds as if a
    /// shift happened — a silent corruption bug.
    fn shift_kv(&self, _state: &mut InferenceState, _n_keep: usize, _shift: usize) {}

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

    /// Whether this model/backend supports TurboQuant KV cache compression.
    /// Used by the CLI to decide whether to request compression or fall back
    /// to f32. TurboQuant is fully driven by `KvCompression` on the
    /// `InferenceState`; models just need to honor the compressed buffers in
    /// their forward pass. Currently only the CPU `Lfm2Model` does.
    fn turboquant_supported(&self) -> bool {
        false
    }
}

/// Load a model from a GGUF file, dispatching on the architecture.
///
/// `context_size` caps the model's `max_seq_len` and determines KV cache
/// pre-allocation in `InferenceState::from_config_with_compression`. Smaller
/// values reduce startup memory; larger values allow longer prompts/decodes.
pub fn load_model(gguf: GgufFile, context_size: usize) -> Result<Box<dyn Model>> {
    let arch = gguf
        .get_str("general.architecture")
        .unwrap_or("unknown")
        .to_string();
    match arch.as_str() {
        "lfm2" => Ok(Box::new(lfm2::Lfm2Model::from_gguf(gguf, context_size)?)),
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
pub mod audio_encoder;
pub mod audio_preprocessor;

// Compile-time proof that `Arc<dyn Model>` is `Send + Sync`. If a new
// backend impl introduces a non-`Sync` field (e.g. a `RefCell` / `Cell`),
// this assertion fires at lib-build time with a clear pointer at the
// invariant, instead of the regression surfacing at a downstream FFI
// crate's build that doesn't have enough context to explain the error.
#[allow(dead_code)]
fn _assert_arc_dyn_model_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<std::sync::Arc<dyn Model>>();
}
