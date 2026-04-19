//! Stateful, multimodal, cancellable inference session.
//!
//! `Session` owns a model's `InferenceState` + `Sampler` and drives
//! prefill/decode through a sink-based streaming API. It replaces the
//! one-shot `engine::generate()` so every downstream consumer — CLI,
//! FFI bindings, browser workers, the AIDL service — shares one core.
//!
//! The API is multimodal from day one even though only text is wired
//! in v1: `append_image` and `append_audio` return
//! `WickError::UnsupportedModality` until the VL / audio loaders land
//! in follow-ups. Callbacks use a `ModalitySink` trait with default-empty
//! methods so text-only consumers override just `on_text_tokens` + `on_done`.

use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::Instant;

use thiserror::Error;

use crate::kv_cache::{InferenceState, KvCompression};
use crate::model::Model;
use crate::sampler::{Sampler, SamplerConfig};
use crate::tokenizer::BpeTokenizer;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Per-session configuration. Set at construction; immutable thereafter.
#[derive(Debug, Clone)]
pub struct SessionConfig {
    /// Cap on total tokens held in KV. `None` → model's default `max_seq_len`.
    pub max_seq_len: Option<u32>,
    /// KV cache compression mode.
    pub kv_compression: KvCompression,
    /// Reserved for Phase 1.5 context shift — tokens pinned at the front on overflow. Ignored in 1.1.
    pub n_keep: u32,
    /// Optional deterministic seed for the sampler.
    pub seed: Option<u64>,
    /// Reserved for Phase 1.4 chunked prefill — ubatch size. Ignored in 1.1 (prefill is monolithic).
    pub ubatch_size: u32,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            max_seq_len: None,
            kv_compression: KvCompression::None,
            n_keep: 0,
            seed: None,
            ubatch_size: 512,
        }
    }
}

/// Per-call generation options.
#[derive(Debug, Clone)]
pub struct GenerateOpts {
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    /// Reserved — repetition penalty not yet supported by the sampler (deferred).
    pub repetition_penalty: f32,
    /// If any of these fires, decode stops with `FinishReason::Stop`.
    pub stop_tokens: Vec<u32>,
    /// Emit `on_text_tokens` at least every N tokens. `0` treats as 1.
    pub flush_every_tokens: u32,
    /// Emit `on_text_tokens` at least every N milliseconds. `0` disables time-based flushing.
    pub flush_every_ms: u32,
}

impl Default for GenerateOpts {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repetition_penalty: 1.0,
            stop_tokens: Vec::new(),
            flush_every_tokens: 16,
            flush_every_ms: 50,
        }
    }
}

/// Summary returned from a completed `generate` call.
#[derive(Debug, Clone)]
pub struct GenerateSummary {
    pub tokens_generated: u32,
    pub prompt_eval_tokens: u32,
    pub prompt_eval_ms: u32,
    pub decode_ms: u32,
    pub finish_reason: FinishReason,
}

/// Why a decode loop ended.
#[derive(Debug, Clone)]
pub enum FinishReason {
    /// Hit `max_tokens`.
    MaxTokens,
    /// Hit an EOS token or an explicit `stop_tokens` entry.
    Stop,
    /// External `cancel()` flipped the atomic.
    Cancelled,
    /// Other error; the outer `Result` is the authoritative channel.
    Error(String),
}

/// Streaming output sink. Default-empty methods let text-only consumers
/// override just `on_text_tokens` + `on_done`; audio callers override
/// `on_audio_frames` as well.
pub trait ModalitySink {
    fn on_text_tokens(&mut self, _tokens: &[u32]) {}
    fn on_audio_frames(&mut self, _pcm: &[f32], _sample_rate: u32) {}
    fn on_done(&mut self, reason: FinishReason);
}

/// Modality support flags for a loaded model.
#[derive(Debug, Clone, Copy, Default)]
pub struct ModalityCapabilities {
    pub text_in: bool,
    pub text_out: bool,
    pub image_in: bool,
    pub audio_in: bool,
    pub audio_out: bool,
}

impl ModalityCapabilities {
    /// Default text-only shape.
    pub fn text_only() -> Self {
        Self {
            text_in: true,
            text_out: true,
            ..Default::default()
        }
    }
}

/// Image input format — scaffolded for the VL path; unused in v1.
#[derive(Debug, Clone, Copy)]
pub enum ImageFormat {
    Png,
    Jpeg,
    RawRgb { width: u32, height: u32 },
}

/// Error type for session operations. `anyhow::Error` upstream consumers
/// can continue to use `?` — we impl `Into<anyhow::Error>` via `thiserror`.
#[derive(Error, Debug)]
pub enum WickError {
    #[error("modality not supported by this model")]
    UnsupportedModality,
    #[error("session is busy with another operation")]
    Busy,
    #[error("cancelled")]
    Cancelled,
    #[error("context window ({max_seq_len}) exceeded by {by} tokens")]
    ContextOverflow { max_seq_len: u32, by: u32 },
    #[error("empty input")]
    EmptyInput,
    #[error("backend: {0}")]
    Backend(String),
    #[error("io: {0}")]
    Io(#[from] io::Error),
}

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

/// Stateful inference session. Borrows the model + tokenizer for its lifetime.
pub struct Session<'a> {
    model: &'a dyn Model,
    tokenizer: &'a BpeTokenizer,
    state: InferenceState,
    sampler: Sampler,
    /// Total tokens currently in KV.
    current_pos: usize,
    /// Mirror of `current_pos` for lock-free external reads via `position()`.
    position_atomic: Arc<AtomicU32>,
    /// External cancel flag. Checked between tokens during decode.
    cancel: Arc<AtomicBool>,
    /// Logits from the last prefill / decode step — seeds the next generate call.
    last_logits: Option<Vec<f32>>,
    /// Copied from config — enforced on `append_tokens`.
    max_seq_len: usize,
    /// Copied from config — retained for introspection.
    _config: SessionConfig,
}

impl<'a> Session<'a> {
    /// Construct a new session backed by an already-loaded model + tokenizer.
    pub fn new(model: &'a dyn Model, tokenizer: &'a BpeTokenizer, config: SessionConfig) -> Self {
        let model_cfg = model.config();
        let max_seq_len = config
            .max_seq_len
            .map(|v| v as usize)
            .unwrap_or(model_cfg.max_seq_len)
            .min(model_cfg.max_seq_len);

        let state = InferenceState::from_config_with_compression(model_cfg, &config.kv_compression);

        let sampler_cfg = SamplerConfig {
            seed: config.seed,
            ..SamplerConfig::default()
        };
        let sampler = Sampler::new(sampler_cfg);

        Self {
            model,
            tokenizer,
            state,
            sampler,
            current_pos: 0,
            position_atomic: Arc::new(AtomicU32::new(0)),
            cancel: Arc::new(AtomicBool::new(false)),
            last_logits: None,
            max_seq_len,
            _config: config,
        }
    }

    /// Capabilities of the loaded model. v1 reports text-only for every
    /// currently-supported model; multimodal reporting lands with the VL
    /// and audio loaders.
    pub fn capabilities(&self) -> ModalityCapabilities {
        ModalityCapabilities::text_only()
    }

    /// Current KV position — tokens live. Atomic; safe from any thread.
    pub fn position(&self) -> u32 {
        self.position_atomic.load(Ordering::Relaxed)
    }

    /// Shared handle to the cancel flag. Clone it into another thread
    /// and call `.store(true, Relaxed)` to interrupt an in-flight generate.
    /// The convenience `cancel()` method does the same for the owning thread.
    pub fn cancel_handle(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancel)
    }

    /// Flip the cancel flag. Safe from any thread.
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    /// Clear KV state and reset position to 0. Does NOT touch the engine-level
    /// disk prefix cache (which lives on `WickEngine`, not `Session`).
    pub fn reset(&mut self) {
        let model_cfg = self.model.config();
        self.state =
            InferenceState::from_config_with_compression(model_cfg, &self._config.kv_compression);
        self.current_pos = 0;
        self.position_atomic.store(0, Ordering::Relaxed);
        self.last_logits = None;
        self.cancel.store(false, Ordering::Relaxed);
    }

    /// Tokenize text and append. Convenience over `append_tokens`.
    pub fn append_text(&mut self, text: &str) -> Result<(), WickError> {
        if text.is_empty() {
            return Err(WickError::EmptyInput);
        }
        let tokens = self.tokenizer.encode(text);
        self.append_tokens(&tokens)
    }

    /// Append raw token IDs, running a prefill pass from the current position
    /// over just the new tail. In v1.1 the prefill is monolithic (no ubatch
    /// chunking yet — that lands in Phase 1.4). Cancel granularity during
    /// append is therefore "whole call"; document that limitation to callers.
    pub fn append_tokens(&mut self, tokens: &[u32]) -> Result<(), WickError> {
        if tokens.is_empty() {
            return Err(WickError::EmptyInput);
        }
        let new_end = self
            .current_pos
            .checked_add(tokens.len())
            .ok_or(WickError::Backend("position overflow".into()))?;
        if new_end > self.max_seq_len {
            return Err(WickError::ContextOverflow {
                max_seq_len: self.max_seq_len as u32,
                by: (new_end - self.max_seq_len) as u32,
            });
        }
        let logits = self
            .model
            .forward_prefill(tokens, self.current_pos, &mut self.state);
        self.current_pos = new_end;
        self.position_atomic
            .store(self.current_pos as u32, Ordering::Relaxed);
        self.last_logits = Some(logits);
        Ok(())
    }

    /// Append an image input. Not supported in v1 for any currently-shipping
    /// model; reserved API shape for the VL loader. Always returns
    /// `UnsupportedModality` until VL support lands.
    pub fn append_image(&mut self, _bytes: &[u8], _format: ImageFormat) -> Result<(), WickError> {
        Err(WickError::UnsupportedModality)
    }

    /// Append an audio input. Reserved for the audio loader wiring; returns
    /// `UnsupportedModality` until that follow-up commit lands.
    pub fn append_audio(&mut self, _pcm: &[f32], _sample_rate: u32) -> Result<(), WickError> {
        Err(WickError::UnsupportedModality)
    }

    /// Run autoregressive decode, emitting token chunks through the sink.
    /// Returns a summary with timing + finish reason. The sink also receives
    /// `on_done(finish_reason)` at the end; callers can treat the `Result`
    /// as authoritative and use `on_done` for UI cleanup.
    pub fn generate<S: ModalitySink + ?Sized>(
        &mut self,
        opts: &GenerateOpts,
        sink: &mut S,
    ) -> Result<GenerateSummary, WickError> {
        // Reset cancel at the start of each generate — stale flips from a
        // prior call shouldn't pre-cancel the next one.
        self.cancel.store(false, Ordering::Relaxed);

        let prompt_eval_tokens = self.current_pos as u32;
        // Synthetic prompt-eval time: prefill already happened in append_*.
        // We don't re-time it here. (Real per-chunk timing arrives with 1.4.)
        let prompt_eval_ms: u32 = 0;

        let mut logits = self.last_logits.take().ok_or(WickError::EmptyInput)?;

        let decode_start = Instant::now();
        let mut finish = FinishReason::MaxTokens;
        let mut generated: u32 = 0;
        let mut pos = self.current_pos;
        // Token batch buffer — flushed to `on_text_tokens` per opts.
        let mut pending: Vec<u32> = Vec::with_capacity(opts.flush_every_tokens.max(1) as usize);
        let mut last_flush = Instant::now();

        let greedy = opts.temperature <= 0.0 || opts.top_k == 1;
        let flush_n = opts.flush_every_tokens.max(1) as usize;
        let flush_ms = opts.flush_every_ms;

        // Seed with the first token sampled from the prefill logits.
        if opts.max_tokens == 0 {
            // Nothing to do — still flush the (empty) sink.
            sink.on_done(FinishReason::MaxTokens);
            let decode_ms = decode_start.elapsed().as_millis() as u32;
            return Ok(GenerateSummary {
                tokens_generated: 0,
                prompt_eval_tokens,
                prompt_eval_ms,
                decode_ms,
                finish_reason: FinishReason::MaxTokens,
            });
        }

        self.sync_sampler_from_opts(opts);
        let mut next_token = self.sampler.sample(&mut logits);

        loop {
            if self.cancel.load(Ordering::Relaxed) {
                finish = FinishReason::Cancelled;
                break;
            }
            if generated >= opts.max_tokens {
                break;
            }
            if pos >= self.max_seq_len {
                break;
            }

            // EOS / stop-token check happens BEFORE emitting the token.
            // Matches llama.cpp/OpenAI semantics: the terminator is not
            // part of the output stream.
            if self.tokenizer.eos_token() == Some(next_token)
                || opts.stop_tokens.contains(&next_token)
            {
                finish = FinishReason::Stop;
                break;
            }

            pending.push(next_token);
            generated += 1;
            pos += 1;

            // Flush the pending batch when N tokens accumulated or
            // flush_every_ms elapsed.
            let should_flush_n = pending.len() >= flush_n;
            let should_flush_t =
                flush_ms > 0 && last_flush.elapsed().as_millis() >= flush_ms as u128;
            if should_flush_n || should_flush_t {
                sink.on_text_tokens(&pending);
                pending.clear();
                last_flush = Instant::now();
            }

            // If we're at the limit, skip the final (wasted) forward pass.
            if generated >= opts.max_tokens || pos >= self.max_seq_len {
                break;
            }

            next_token = if greedy {
                self.model
                    .forward_greedy(&[next_token], pos - 1, &mut self.state)
            } else {
                logits = self.model.forward(&[next_token], pos - 1, &mut self.state);
                self.sampler.sample(&mut logits)
            };
        }

        // Flush any tail tokens.
        if !pending.is_empty() {
            sink.on_text_tokens(&pending);
        }

        self.current_pos = pos;
        self.position_atomic
            .store(self.current_pos as u32, Ordering::Relaxed);

        sink.on_done(finish.clone());

        let decode_ms = decode_start.elapsed().as_millis() as u32;
        Ok(GenerateSummary {
            tokens_generated: generated,
            prompt_eval_tokens,
            prompt_eval_ms,
            decode_ms,
            finish_reason: finish,
        })
    }

    fn sync_sampler_from_opts(&mut self, opts: &GenerateOpts) {
        // `Sampler::new` rebuilds the RNG from the seed; for per-call opts
        // updates within the same session we just replace the config.
        let cfg = SamplerConfig {
            temperature: opts.temperature,
            top_k: opts.top_k as usize,
            top_p: opts.top_p,
            seed: self._config.seed,
        };
        self.sampler.set_config(cfg);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// A trivial sink that records text tokens + done calls in order, for
    /// asserting stream shape without needing a real model.
    #[derive(Default)]
    struct RecordingSink {
        tokens: Vec<u32>,
        done: Option<FinishReason>,
        flushes: u32,
    }

    impl ModalitySink for RecordingSink {
        fn on_text_tokens(&mut self, tokens: &[u32]) {
            self.tokens.extend_from_slice(tokens);
            self.flushes += 1;
        }
        fn on_done(&mut self, reason: FinishReason) {
            self.done = Some(reason);
        }
    }

    #[test]
    fn session_config_default_is_sane() {
        let c = SessionConfig::default();
        assert_eq!(c.n_keep, 0);
        assert_eq!(c.ubatch_size, 512);
        assert!(matches!(c.kv_compression, KvCompression::None));
    }

    #[test]
    fn generate_opts_default_batching() {
        let o = GenerateOpts::default();
        assert_eq!(o.flush_every_tokens, 16);
        assert_eq!(o.flush_every_ms, 50);
    }

    #[test]
    fn capabilities_text_only_shape() {
        let c = ModalityCapabilities::text_only();
        assert!(c.text_in && c.text_out);
        assert!(!c.image_in && !c.audio_in && !c.audio_out);
    }

    #[test]
    fn recording_sink_collects_tokens() {
        let mut s = RecordingSink::default();
        s.on_text_tokens(&[1, 2, 3]);
        s.on_text_tokens(&[4]);
        s.on_done(FinishReason::MaxTokens);
        assert_eq!(s.tokens, vec![1, 2, 3, 4]);
        assert_eq!(s.flushes, 2);
        assert!(matches!(s.done, Some(FinishReason::MaxTokens)));
    }

    // Integration tests that need a real model run under
    // `tests/session_integration.rs` (gated on models being present
    // locally); unit tests here stay dep-free.
}
