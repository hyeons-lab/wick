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

use thiserror::Error;

use crate::time::Instant;

use crate::kv_cache::{InferenceState, KvCompression};
use crate::model::Model;
use crate::model::audio_encoder::AudioEncoderWeights;
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
    /// Reached the session's `max_seq_len`; no room to decode further
    /// without a context shift (landing in Phase 1.5 via `n_keep`).
    ContextFull,
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
    /// Text-in, text-out only. The baseline for LLaMA-family LLMs.
    pub fn text_only() -> Self {
        Self {
            text_in: true,
            text_out: true,
            ..Default::default()
        }
    }

    /// Text + audio bidirectional — LFM2-Audio-class models: PCM audio
    /// in via [`Session::append_audio`], text + audio frames out via
    /// [`ModalitySink`].
    pub fn text_and_audio() -> Self {
        Self {
            text_in: true,
            text_out: true,
            audio_in: true,
            audio_out: true,
            ..Default::default()
        }
    }

    /// Text + image in, text out — VL-class models (LFM2-VL,
    /// LLaVA-family). Image output is not an LFM2-family capability.
    pub fn text_and_image_in() -> Self {
        Self {
            text_in: true,
            text_out: true,
            image_in: true,
            ..Default::default()
        }
    }

    /// Derive capabilities from a manifest's `inference_type`. Unknown
    /// variants fall back to text-only so a bundle we don't understand
    /// at least reports safe minimums.
    pub fn from_inference_type(it: &crate::manifest::InferenceType) -> Self {
        use crate::manifest::InferenceType::*;
        match it {
            LlamaCppTextToText => Self::text_only(),
            LlamaCppImageToText => Self::text_and_image_in(),
            LlamaCppLfm2AudioV1 => Self::text_and_audio(),
            Unknown(_) => Self::text_only(),
        }
    }
}

/// Error type for session operations. Upstream consumers using
/// `anyhow::Error` can continue to use `?` because `thiserror` derives
/// `std::error::Error` for this type, making it compatible with `anyhow`.
#[derive(Error, Debug)]
pub enum WickError {
    #[error("modality not supported by this model")]
    UnsupportedModality,
    #[error("inference_type `{0}` is not supported in this version of wick")]
    UnsupportedInferenceType(String),
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

/// Decide whether `append_tokens` may run a `n_keep` context shift
/// instead of returning `ContextOverflow`. Pure 4-input predicate,
/// extracted from the overflow arm so it can be unit-tested without
/// spinning up a full `Session` (which needs a real `BpeTokenizer`).
///
/// All four must hold:
/// - `supports_kv_shift`: backend opted in via [`Model::supports_kv_shift`]
/// - `n_keep > 0`: user wants to preserve a prefix
/// - `!is_compressed`: TurboQuant caches aren't shiftable yet
/// - `current_pos >= n_keep + shift_needed`: the pinned prefix leaves
///   at least `shift_needed` rotatable cells to drop
pub fn can_shift(
    supports_kv_shift: bool,
    n_keep: usize,
    is_compressed: bool,
    current_pos: usize,
    shift_needed: usize,
) -> bool {
    supports_kv_shift && n_keep > 0 && !is_compressed && current_pos >= n_keep + shift_needed
}

/// One slice of a tokenized chat template, distinguishing text runs
/// from image-marker positions. `Text { start, end }` is a half-open
/// index range into the same `tokens` slice that
/// [`splice_image_markers`] received; `Image` is an in-place marker
/// that the caller should swap for the
/// `<|image_start|>` + image embeddings + `<|image_end|>`
/// envelope at append time.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ChatTemplateSegment {
    Text { start: usize, end: usize },
    Image,
}

/// Walk a tokenized chat-template stream and produce a splice plan:
/// for each contiguous run of non-marker tokens emit a
/// [`ChatTemplateSegment::Text`] range; for each `<image>` token
/// (id == `image_marker_id`) emit a [`ChatTemplateSegment::Image`].
///
/// Empty text runs (two adjacent markers, marker at start/end of
/// stream) are elided so the segment list never carries
/// zero-length text spans — the caller's `append_tokens(&[])`
/// would be a no-op anyway, but keeping the segment list tight
/// makes the unit-test assertions cleaner and the walk loop
/// branch-free on the empty case.
pub(crate) fn splice_image_markers(
    tokens: &[u32],
    image_marker_id: u32,
) -> Vec<ChatTemplateSegment> {
    let mut segments = Vec::new();
    let mut start = 0;
    for (i, &t) in tokens.iter().enumerate() {
        if t == image_marker_id {
            if i > start {
                segments.push(ChatTemplateSegment::Text { start, end: i });
            }
            segments.push(ChatTemplateSegment::Image);
            start = i + 1;
        }
    }
    if start < tokens.len() {
        segments.push(ChatTemplateSegment::Text {
            start,
            end: tokens.len(),
        });
    }
    segments
}

/// Stateful inference session. Owns refcounted handles to the model
/// and tokenizer — no borrow lifetime — so `Session` values can flow
/// across an FFI boundary or be returned from a constructor without
/// tying them to an owning `WickEngine`.
///
/// The `Arc`-based design replaced the earlier `Session<'a> { model:
/// &'a dyn Model, tokenizer: &'a BpeTokenizer }` form because UniFFI
/// and bindgen tools don't marshal Rust lifetimes — the exposed type
/// has to own its dependencies.
pub struct Session {
    model: Arc<dyn Model>,
    tokenizer: Arc<BpeTokenizer>,
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
    /// What this session can accept / emit, derived from the model's
    /// inference_type at construction. Immutable for the session's
    /// lifetime.
    capabilities: ModalityCapabilities,
    /// Retained for `reset()` (rebuild state + sampler) and
    /// `sync_sampler_from_opts` (read back the seed).
    config: SessionConfig,
    /// Audio encoder weights, if attached. None for text-only
    /// sessions; populated via [`Self::attach_audio_encoder`] before
    /// [`Self::append_audio`] is called. Held by `Arc` so the same
    /// encoder can back multiple sessions (one per concurrent
    /// generation) without re-loading the ~hundreds-of-MB weights.
    audio_encoder: Option<Arc<AudioEncoderWeights>>,
    /// Vision encoder weights, if attached. None for non-VL
    /// sessions; populated via [`Self::attach_vision_encoder`]
    /// before [`Self::append_image`] is called. Held by `Arc`
    /// for the same reason as `audio_encoder`.
    vision_encoder: Option<Arc<crate::model::vision_encoder::VisionEncoderWeights>>,
}

impl Session {
    /// Construct a new session backed by an already-loaded model + tokenizer.
    /// Both are taken by `Arc` — in-process callers typically clone from
    /// [`crate::WickEngine`] (see [`crate::WickEngine::new_session`]); FFI
    /// callers wrap owned handles.
    ///
    /// `capabilities` declares what the loaded model accepts / emits.
    /// Direct callers (tests, standalone Model loaders) that don't have
    /// a Manifest handy can pass [`ModalityCapabilities::text_only`].
    pub fn new(
        model: Arc<dyn Model>,
        tokenizer: Arc<BpeTokenizer>,
        capabilities: ModalityCapabilities,
        config: SessionConfig,
    ) -> Self {
        let model_cfg = model.config();
        let max_seq_len = config
            .max_seq_len
            .map(|v| v as usize)
            .unwrap_or(model_cfg.max_seq_len)
            .min(model_cfg.max_seq_len);

        // A `n_keep >= max_seq_len` config can never actually shift —
        // `current_pos` tops out at `max_seq_len` and the shift arm
        // requires `current_pos >= n_keep + shift_needed`. Warn once
        // at construction so users see why their `--n-keep` isn't
        // kicking in instead of discovering it via `ContextOverflow`.
        if config.n_keep > 0 && (config.n_keep as usize) >= max_seq_len {
            tracing::warn!(
                target: "wick::session",
                n_keep = config.n_keep,
                max_seq_len,
                "n_keep >= max_seq_len; context shift will never fire \
                 because there's no room left for shifted cells. Lower \
                 n_keep to enable shifting."
            );
        }
        // Likewise, `n_keep > 0` + TurboQuant is a no-op because the
        // overflow arm gates on `!is_compressed()`. Warn once so the
        // user knows their n_keep value is being silently ignored on
        // overflow.
        if config.n_keep > 0 && !matches!(config.kv_compression, KvCompression::None) {
            tracing::warn!(
                target: "wick::session",
                n_keep = config.n_keep,
                "n_keep configured alongside TurboQuant KV compression; \
                 shift not yet supported for compressed caches, so \
                 overflow will still return ContextOverflow. Disable \
                 compression to enable n_keep."
            );
        }
        // Backend must opt in to shift (CPU LFM2 today; Metal is a
        // follow-up). If the user set `n_keep > 0` on a backend that
        // doesn't implement shift, overflow still returns
        // ContextOverflow — tell them why instead of letting them
        // discover it the hard way.
        if config.n_keep > 0 && !model.supports_kv_shift() {
            tracing::warn!(
                target: "wick::session",
                n_keep = config.n_keep,
                architecture = model_cfg.architecture.as_str(),
                "n_keep configured but this backend doesn't support KV shift; \
                 overflow will still return ContextOverflow. CPU backend \
                 (BackendPreference::Cpu) supports shift today; Metal / GPU \
                 paths land in a follow-up."
            );
        }

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
            capabilities,
            config,
            audio_encoder: None,
            vision_encoder: None,
        }
    }

    /// Attach an audio encoder so [`Self::append_audio`] can encode
    /// PCM samples into LLM-ready embeddings. Callers load the encoder
    /// from the bundle's `multimodal_projector` GGUF via
    /// [`crate::model::audio_encoder::AudioEncoderWeights::from_gguf`].
    ///
    /// Replaces any previously-attached encoder. Preserved across
    /// [`Self::reset`] — the encoder is independent of KV state.
    ///
    /// Does **not** validate that the encoder's `llm_hidden_size`
    /// matches the LLM's `hidden_size`; that check lives on
    /// [`Self::append_audio`] so a stub encoder used in test setup
    /// doesn't have to wire up matching dimensions just to be
    /// attached. Real callers should always pair the encoder with
    /// the LLM it was trained against.
    pub fn attach_audio_encoder(&mut self, encoder: Arc<AudioEncoderWeights>) {
        self.audio_encoder = Some(encoder);
    }

    /// Attach a vision encoder so [`Self::append_image`] can encode
    /// PNG / JPEG bytes into LLM-ready image embeddings. Callers
    /// load the encoder from the bundle's `multimodal_projector`
    /// GGUF via
    /// [`crate::model::vision_encoder::VisionEncoderWeights::from_gguf`].
    /// Mirrors [`Self::attach_audio_encoder`]'s semantics: replaces
    /// any prior attachment, preserved across `reset()`, no
    /// dimension check at attach time (it lives on `append_image`
    /// where we have a real input to size against).
    pub fn attach_vision_encoder(
        &mut self,
        encoder: Arc<crate::model::vision_encoder::VisionEncoderWeights>,
    ) {
        self.vision_encoder = Some(encoder);
    }

    /// What this session accepts as input / emits as output. Derived
    /// from the model's `inference_type` at construction — see
    /// [`ModalityCapabilities::from_inference_type`] for the mapping.
    pub fn capabilities(&self) -> ModalityCapabilities {
        self.capabilities
    }

    /// Borrow the tokenizer the session was constructed with. Useful
    /// for callers (tests, FFI wrappers) that want to encode / decode
    /// without threading the tokenizer through separately.
    pub fn tokenizer(&self) -> &BpeTokenizer {
        self.tokenizer.as_ref()
    }

    /// Borrow the model the session was constructed with. Primarily
    /// for introspection (vocab size, max_seq_len, etc.); hot-path
    /// forward calls still go through `Session`'s own methods.
    pub fn model(&self) -> &dyn Model {
        self.model.as_ref()
    }

    /// Current KV position — tokens live. Atomic; safe from any thread.
    pub fn position(&self) -> u32 {
        self.position_atomic.load(Ordering::Relaxed)
    }

    /// Shared handle to the position counter. Clone into another thread to
    /// watch an in-flight generate's progress without holding `&self` (which
    /// would block on `generate`'s `&mut self` borrow).
    pub fn position_handle(&self) -> Arc<AtomicU32> {
        Arc::clone(&self.position_atomic)
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

    /// Clear KV state and reset position to 0. Rebuilds the sampler from
    /// `SessionConfig::seed` so a seeded session is fully reproducible after
    /// reset. Does NOT touch the engine-level disk prefix cache (which lives
    /// on `WickEngine`, not `Session`).
    pub fn reset(&mut self) {
        let model_cfg = self.model.config();
        self.state =
            InferenceState::from_config_with_compression(model_cfg, &self.config.kv_compression);
        self.current_pos = 0;
        self.position_atomic.store(0, Ordering::Relaxed);
        self.last_logits = None;
        self.cancel.store(false, Ordering::Relaxed);
        // Re-seed the sampler so deterministic runs stay deterministic after reset().
        let sampler_cfg = SamplerConfig {
            seed: self.config.seed,
            ..SamplerConfig::default()
        };
        self.sampler = Sampler::new(sampler_cfg);
    }

    /// Tokenize text and append. Convenience over `append_tokens`.
    pub fn append_text(&mut self, text: &str) -> Result<(), WickError> {
        if text.is_empty() {
            return Err(WickError::EmptyInput);
        }
        let tokens = self.tokenizer.encode(text);
        self.append_tokens(&tokens)
    }

    /// Append PCM audio input to the session's context. Runs the
    /// samples through the attached audio encoder (mel + Conformer +
    /// MLP adapter) and prefills the resulting per-frame hidden
    /// states into KV via [`Self::append_embeddings`].
    ///
    /// `samples` is f32 PCM in `[-1, 1]`, mono. `sample_rate` must
    /// match [`crate::model::audio_encoder::SAMPLE_RATE`] (16 kHz);
    /// resampling is out of scope — caller is expected to resample
    /// externally if their source rate differs.
    ///
    /// Errors are checked in the order listed. The first matching
    /// condition wins — e.g. an empty `samples` buffer paired with
    /// an audio-incapable session returns `UnsupportedModality`,
    /// not `EmptyInput`.
    ///
    /// 1. [`WickError::UnsupportedModality`] when the loaded model
    ///    doesn't support audio input
    ///    ([`Self::capabilities`]`.audio_in == false`).
    /// 2. [`WickError::Backend`] with a "no encoder attached" message
    ///    when the model supports audio but
    ///    [`Self::attach_audio_encoder`] hasn't been called yet.
    /// 3. [`WickError::Backend`] when the attached encoder's
    ///    `llm_hidden_size` doesn't match the LLM's `hidden_size`
    ///    (wrong-bundle encoder).
    /// 4. [`WickError::Backend`] on sample-rate mismatch.
    /// 5. [`WickError::EmptyInput`] when `samples` is empty *or* the
    ///    audio is too short to produce any encoder frames (less than
    ///    one window after center-padded STFT).
    /// 6. [`WickError::ContextOverflow`] / [`WickError::Cancelled`]
    ///    propagated from the underlying [`Self::append_embeddings`]
    ///    call.
    pub fn append_audio(&mut self, samples: &[f32], sample_rate: u32) -> Result<(), WickError> {
        if !self.capabilities.audio_in {
            return Err(WickError::UnsupportedModality);
        }
        let Some(encoder) = self.audio_encoder.as_ref() else {
            return Err(WickError::Backend(
                "Session::append_audio: no audio encoder attached. Call \
                 attach_audio_encoder() with weights loaded via \
                 AudioEncoderWeights::from_gguf(...) on the bundle's \
                 multimodal_projector GGUF before calling append_audio."
                    .to_string(),
            ));
        };
        // Catch encoder/LLM dimension mismatch upfront with a clear
        // message naming both sides. Without this, `append_embeddings`
        // would still reject the resulting buffer downstream — but
        // with a generic shape error that doesn't point at the real
        // cause (encoder loaded from a different bundle than the LLM).
        let llm_hidden = self.model.config().hidden_size;
        let enc_hidden = encoder.config.llm_hidden_size;
        if enc_hidden != llm_hidden {
            return Err(WickError::Backend(format!(
                "Session::append_audio: attached encoder's llm_hidden_size ({enc_hidden}) \
                 does not match the LLM's hidden_size ({llm_hidden}). \
                 The encoder must be the multimodal_projector trained for the \
                 currently-loaded LLM."
            )));
        }
        if sample_rate != crate::model::audio_encoder::SAMPLE_RATE {
            return Err(WickError::Backend(format!(
                "Session::append_audio: sample_rate {} != {} required by \
                 encoder (resampling is out of scope; resample externally \
                 before passing samples in)",
                sample_rate,
                crate::model::audio_encoder::SAMPLE_RATE,
            )));
        }
        if samples.is_empty() {
            return Err(WickError::EmptyInput);
        }
        let (embeddings, n_frames) =
            crate::model::audio_encoder::encode_audio_pcm(samples, encoder.as_ref());
        if n_frames == 0 {
            // Sub-window-length input: log_mel_spectrogram produced
            // zero frames. Caller's audio was too short to encode
            // anything; surface as EmptyInput rather than slicing
            // the empty embedding buffer downstream.
            return Err(WickError::EmptyInput);
        }
        self.append_embeddings(&embeddings, n_frames)
    }

    /// Append raw token IDs, running a prefill pass from the current position
    /// over just the new tail.
    ///
    /// Prefill runs through [`Model::forward_prefill_chunked`] with
    /// `SessionConfig::ubatch_size` so long prompts can be cancelled
    /// mid-flight. Returns `WickError::Cancelled` when cancel fires
    /// before the full slice is consumed.
    ///
    /// On cancellation:
    /// - Tokens already fed through the kernel stay in KV; `position()`
    ///   advances to reflect how many were actually consumed.
    /// - `last_logits` is **cleared** (not set to the partial-prefill
    ///   logits). This forces a subsequent `generate()` to return
    ///   `EmptyInput` rather than silently producing tokens from
    ///   mid-prompt state. The caller's contract is to clear the flag
    ///   via [`Self::clear_cancel`] and resume by appending the
    ///   unconsumed tail before generating. Sketch:
    ///
    ///   ```ignore
    ///   let before = session.position() as usize;
    ///   match session.append_tokens(&tokens) {
    ///       Err(WickError::Cancelled) => {
    ///           let consumed = session.position() as usize - before;
    ///           session.clear_cancel();
    ///           session.append_tokens(&tokens[consumed..])?;
    ///       }
    ///       other => other?,
    ///   }
    ///   ```
    pub fn append_tokens(&mut self, tokens: &[u32]) -> Result<(), WickError> {
        if tokens.is_empty() {
            return Err(WickError::EmptyInput);
        }
        let new_end = self
            .current_pos
            .checked_add(tokens.len())
            .ok_or(WickError::Backend("position overflow".into()))?;
        if new_end > self.max_seq_len {
            // `n_keep` context shift (Phase 1.5): if the backend
            // supports shift, the session was configured with
            // `n_keep > 0`, the state isn't TurboQuant-compressed, and
            // the pinned prefix leaves room to drop — shift to make
            // room. Otherwise fall through to the typed ContextOverflow.
            let n_keep = self.config.n_keep as usize;
            let shift_needed = new_end - self.max_seq_len;
            if !can_shift(
                self.model.supports_kv_shift(),
                n_keep,
                self.state.is_compressed(),
                self.current_pos,
                shift_needed,
            ) {
                return Err(WickError::ContextOverflow {
                    max_seq_len: self.max_seq_len as u32,
                    by: (new_end - self.max_seq_len) as u32,
                });
            }
            self.model.shift_kv(&mut self.state, n_keep, shift_needed);
            let before = self.current_pos;
            self.current_pos -= shift_needed;
            self.position_atomic
                .store(self.current_pos as u32, Ordering::Relaxed);
            // Pre-shift `last_logits` corresponded to position `before - 1`;
            // the positions they encode don't exist anymore. Clear so a
            // subsequent `generate()` that bypasses the upcoming prefill
            // doesn't silently emit from the wrong context.
            self.last_logits = None;
            tracing::info!(
                target: "wick::kv_shift",
                n_keep = n_keep,
                shift = shift_needed,
                seq_len_before = before,
                seq_len_after = self.current_pos,
                "kv context shift"
            );
        }
        // Pass `ubatch_size` straight through — the trait method treats
        // 0 as "no chunking" (single chunk = whole input), matching the
        // CLI `--ubatch-size 0` opt-out.
        let (consumed, logits) = self.model.forward_prefill_chunked(
            tokens,
            self.current_pos,
            &mut self.state,
            self.config.ubatch_size as usize,
            &self.cancel,
        );
        self.current_pos += consumed;
        self.position_atomic
            .store(self.current_pos as u32, Ordering::Relaxed);
        if consumed < tokens.len() {
            // Don't stash the partial-prefill logits. They correspond to
            // the chunk boundary, not the intended end of prompt — letting
            // a subsequent `generate()` read them would silently produce
            // text from mid-prompt state. Force the caller to re-append
            // (or `reset`) before generating.
            self.last_logits = None;
            Err(WickError::Cancelled)
        } else {
            self.last_logits = logits;
            Ok(())
        }
    }

    /// Append a sequence of pre-computed hidden-dim embeddings —
    /// the soft-token analog of [`Self::append_tokens`].
    ///
    /// Each row of `embeddings` is fed straight into the model
    /// at the LLM's input-embedding stage, bypassing the
    /// `embed_tokens` lookup. Used for non-text input modalities
    /// where an external encoder produces hidden states directly
    /// (e.g. the LFM2A audio encoder's per-frame output via
    /// [`crate::model::audio_encoder::encode_audio_pcm`]).
    ///
    /// `embeddings` is a flat row-major buffer of length
    /// `n_tokens * hidden_size`. Position advances by `n_tokens`.
    /// Returns `Err(EmptyInput)` for `n_tokens == 0`,
    /// `Err(Backend(...))` for shape mismatch.
    ///
    /// Mirrors `append_tokens`'s context-shift logic
    /// (`n_keep`-aware) and `last_logits` semantics. Cancellation
    /// is checked between `ubatch_size`-sized chunks (granularity:
    /// one chunk); on cancel, `last_logits` is cleared and
    /// `Err(Cancelled)` is returned with the frames processed so
    /// far still in KV (caller can `clear_cancel` and resume from
    /// `position()` like `append_tokens`).
    ///
    /// Dispatches to [`Model::forward_prefill_from_embeddings`]
    /// in `ubatch`-sized chunks, mirroring
    /// [`Model::forward_prefill_chunked`] for tokens. Backends
    /// with a true batched embedding-prefill path (CPU `Lfm2Model`)
    /// process a whole chunk per call and amortize per-layer GEMM
    /// dispatch across frames; the trait default falls back to a
    /// per-frame `forward_from_embedding` loop, preserving
    /// correctness for backends that haven't overridden.
    pub fn append_embeddings(
        &mut self,
        embeddings: &[f32],
        n_tokens: usize,
    ) -> Result<(), WickError> {
        if n_tokens == 0 {
            return Err(WickError::EmptyInput);
        }
        // Backend capability check: surface a typed error instead
        // of the default `unimplemented!` panic on backends that
        // don't implement `forward_from_embedding`.
        if !self.model.supports_embedding_input() {
            return Err(WickError::UnsupportedModality);
        }
        let hidden_size = self.model.config().hidden_size;
        let expected_len = n_tokens.checked_mul(hidden_size).ok_or_else(|| {
            WickError::Backend("append_embeddings: n_tokens * hidden_size overflow".into())
        })?;
        if embeddings.len() != expected_len {
            return Err(WickError::Backend(format!(
                "append_embeddings: embeddings.len() ({}) != n_tokens ({n_tokens}) * hidden_size ({hidden_size}) = {expected_len}",
                embeddings.len()
            )));
        }

        let new_end = self
            .current_pos
            .checked_add(n_tokens)
            .ok_or(WickError::Backend("position overflow".into()))?;
        if new_end > self.max_seq_len {
            // Same `n_keep` context-shift logic as append_tokens.
            let n_keep = self.config.n_keep as usize;
            let shift_needed = new_end - self.max_seq_len;
            if !can_shift(
                self.model.supports_kv_shift(),
                n_keep,
                self.state.is_compressed(),
                self.current_pos,
                shift_needed,
            ) {
                return Err(WickError::ContextOverflow {
                    max_seq_len: self.max_seq_len as u32,
                    by: (new_end - self.max_seq_len) as u32,
                });
            }
            self.model.shift_kv(&mut self.state, n_keep, shift_needed);
            let before = self.current_pos;
            self.current_pos -= shift_needed;
            self.position_atomic
                .store(self.current_pos as u32, Ordering::Relaxed);
            self.last_logits = None;
            tracing::info!(
                target: "wick::kv_shift",
                n_keep = n_keep,
                shift = shift_needed,
                seq_len_before = before,
                seq_len_after = self.current_pos,
                "kv context shift (append_embeddings)"
            );
        }

        // Chunk over `ubatch_size` frames, calling the batched
        // embedding-prefill once per chunk. Position advances by
        // chunk size after each call. Cancel is checked AFTER each
        // chunk so we always make progress on at least one chunk
        // before observing the flag — mirrors
        // `forward_prefill_chunked`'s "always-one-chunk" guarantee
        // and avoids leaving the session wedged on an entry-time
        // cancel. `ubatch == 0` means "no chunking" (one call
        // covering all frames), matching CLI `--ubatch-size 0`.
        let ubatch = self.config.ubatch_size as usize;
        let chunk_size = if ubatch == 0 { n_tokens } else { ubatch };
        let mut last_logits: Option<Vec<f32>> = None;
        let mut ti = 0usize;
        while ti < n_tokens {
            let end = (ti + chunk_size).min(n_tokens);
            let chunk = &embeddings[ti * hidden_size..end * hidden_size];
            let logits = self.model.forward_prefill_from_embeddings(
                chunk,
                end - ti,
                self.current_pos,
                &mut self.state,
            );
            self.current_pos += end - ti;
            self.position_atomic
                .store(self.current_pos as u32, Ordering::Relaxed);
            last_logits = Some(logits);
            ti = end;
            // Only abort if more frames remain — guarantees ≥ 1
            // chunk of progress before observing the flag.
            if self.cancel.load(Ordering::Relaxed) && ti < n_tokens {
                self.last_logits = None;
                return Err(WickError::Cancelled);
            }
        }

        self.last_logits = last_logits;
        Ok(())
    }

    /// Clear the cancel flag. Call this after handling a
    /// [`WickError::Cancelled`] from `append_tokens` / `generate` when
    /// you want to resume work on the same session (append more tokens,
    /// generate again) without rebuilding it via [`Self::reset`].
    pub fn clear_cancel(&self) {
        self.cancel.store(false, Ordering::Relaxed);
    }

    /// Append an image input. Decodes PNG / JPEG bytes, resizes
    /// to the encoder's native input size, normalises with the
    /// encoder's per-channel mean / std, runs the ViT + projector
    /// forward to produce 64 image tokens × `projection_dim`, and
    /// splices them into the LLM prefill stream at the current
    /// position via [`Self::append_embeddings`].
    ///
    /// **Placement matters.** The model was trained on a specific
    /// surrounding-token envelope (LFM2-VL: `<|image_start|>` /
    /// `<|image_end|>` *inside* the user-turn opening
    /// `<|im_start|>user\n…<|im_end|>` block). Calling
    /// `append_image` at the wrong stream position — before the
    /// `<bos>` token, outside the user turn, or without the
    /// model-specific markers — leaves the LLM unable to
    /// interpret the embeddings as visual content; the visible
    /// failure mode is a generic non-image-conditioned
    /// description (e.g. *"I see a complex and abstract scene
    /// with various shapes…"*).
    ///
    /// **Recommended path:**
    /// [`Self::append_chat_with_images`] handles render +
    /// marker-walk + envelope splice in one call. It's the right
    /// path for any LFM2-VL inference driven by the standard chat
    /// template.
    ///
    /// Use this method directly only when you need the manual
    /// splice — e.g. a non-LFM2-VL model with a different envelope
    /// convention, or custom token routing the helper doesn't
    /// support. Manual recipe:
    ///
    /// ```ignore
    /// let img_start = tokenizer.special_token_id("<|image_start|>")?;
    /// let img_end   = tokenizer.special_token_id("<|image_end|>")?;
    /// session.append_tokens(&prefix_tokens)?;          // BOS + <|im_start|>user\n
    /// session.append_tokens(&[img_start])?;
    /// session.append_image(&jpeg_bytes)?;
    /// session.append_tokens(&[img_end])?;
    /// session.append_tokens(&suffix_tokens)?;          // user text + <|im_end|>\n + asst tag
    /// session.generate(&opts, &mut sink)?;
    /// ```
    ///
    /// `wick/tests/vl_bundle_load.rs::vl_bundle_appends_synthetic_image`
    /// is the reference integration recipe (now via the helper).
    ///
    /// Errors:
    /// 1. [`WickError::EmptyInput`] when `bytes` is empty.
    /// 2. [`WickError::UnsupportedModality`] if the session
    ///    capabilities don't include `image_in` (non-VL bundle).
    /// 3. [`WickError::Backend`] when no vision encoder is
    ///    attached (text-only construction of a VL bundle, or
    ///    test setup that skipped `attach_vision_encoder`).
    /// 4. [`WickError::Backend`] when image decode / resize
    ///    fails (corrupt PNG, unsupported format, etc.).
    /// 5. [`WickError::Backend`] when the encoder's
    ///    `projection_dim` doesn't match the LLM's `hidden_size`
    ///    (mismatched mmproj loaded against a different LLM).
    /// 6. [`WickError::ContextOverflow`] / [`WickError::Cancelled`]
    ///    propagated from [`Self::append_embeddings`].
    #[cfg(feature = "vl-preprocess")]
    pub fn append_image(&mut self, bytes: &[u8]) -> Result<(), WickError> {
        if bytes.is_empty() {
            return Err(WickError::EmptyInput);
        }
        if !self.capabilities.image_in {
            return Err(WickError::UnsupportedModality);
        }
        let Some(encoder) = self.vision_encoder.as_ref() else {
            return Err(WickError::Backend(
                "Session::append_image: no vision encoder attached. \
                 Construct via WickEngine on a VL bundle so the \
                 encoder is auto-attached, or call \
                 attach_vision_encoder(...) in test setup."
                    .into(),
            ));
        };
        let llm_hidden = self.model.config().hidden_size;
        let proj_dim = encoder.config.projection_dim;
        if proj_dim != llm_hidden {
            return Err(WickError::Backend(format!(
                "Session::append_image: vision encoder's projection_dim \
                 ({proj_dim}) does not match LLM hidden_size ({llm_hidden}). \
                 The mmproj must pair with the LLM it was trained against."
            )));
        }
        let pre = crate::model::vision_preprocessor::preprocess_image(bytes, &encoder.config)?;
        let img_tokens = encoder
            .encode_image(&pre.pixels, pre.grid_w, pre.grid_h)
            .map_err(|e| WickError::Backend(format!("encode_image: {e:#}")))?;
        // Sanity-check the encoder output shape before handing off
        // to `append_embeddings`. Integer division below would
        // silently truncate a non-multiple length and surface as a
        // less actionable mismatch error from `append_embeddings`.
        if img_tokens.len() % proj_dim != 0 {
            return Err(WickError::Backend(format!(
                "Session::append_image: encode_image returned {} f32s, \
                 not a multiple of projection_dim ({proj_dim}) — encoder \
                 produced a malformed image-token tensor",
                img_tokens.len(),
            )));
        }
        let n_tokens = img_tokens.len() / proj_dim;
        if n_tokens == 0 {
            return Err(WickError::Backend(
                "Session::append_image: encoder produced zero image tokens \
                 (preprocess + encode succeeded but yielded an empty tensor)"
                    .into(),
            ));
        }
        self.append_embeddings(&img_tokens, n_tokens)
    }

    /// Stub `append_image` for builds without `vl-preprocess`. Same
    /// signature so wasm / embedded code that conditionally calls
    /// it still type-checks; always returns `UnsupportedModality`.
    #[cfg(not(feature = "vl-preprocess"))]
    pub fn append_image(&mut self, _bytes: &[u8]) -> Result<(), WickError> {
        Err(WickError::UnsupportedModality)
    }

    /// Append a multimodal chat conversation in one call: render the
    /// chat template (which emits `<image>` markers for each
    /// [`crate::tokenizer::ContentItem::Image`] in the messages),
    /// then walk the resulting token stream and splice in
    /// `<|image_start|>` + image embeddings + `<|image_end|>` for
    /// each marker. The `images` slice maps positionally onto
    /// `<image>` markers in render order — the order they appear in
    /// the messages' content lists.
    ///
    /// This is the recommended path for VL inference: it replaces
    /// the manual splicing example documented on
    /// [`Self::append_image`]. The manual path stays available for
    /// callers with non-LFM2-VL chat templates or custom token
    /// routing.
    ///
    /// All validation runs *before* any [`Self::append_tokens`] /
    /// [`Self::append_image`] call, so a failed render or a
    /// marker-count mismatch leaves session state untouched.
    ///
    /// Errors:
    /// 1. [`WickError::UnsupportedModality`] when the session
    ///    capabilities don't include `image_in` (non-VL bundle).
    /// 2. [`WickError::Backend`] when the model has no chat
    ///    template, when `<image>` doesn't tokenize to a single
    ///    token (model isn't VL-shaped), when the tokenizer is
    ///    missing the `<|image_start|>` / `<|image_end|>` special
    ///    tokens, or when the marker count doesn't match the
    ///    supplied `images.len()`.
    /// 3. Any error from [`Self::append_tokens`] /
    ///    [`Self::append_image`] propagates once splicing begins.
    #[cfg(feature = "vl-preprocess")]
    pub fn append_chat_with_images(
        &mut self,
        messages: &[crate::tokenizer::ChatMessageMultimodal],
        images: &[&[u8]],
        add_generation_prompt: bool,
    ) -> Result<(), WickError> {
        if !self.capabilities.image_in {
            return Err(WickError::UnsupportedModality);
        }

        // 1. Render template.
        let rendered =
            crate::tokenizer::apply_chat_template(&self.tokenizer, messages, add_generation_prompt)
                .map_err(|e| WickError::Backend(format!("chat template render: {e:#}")))?;

        // 2. Resolve `<image>` token id (probe once, error if not a
        //    single token — that means the vocab doesn't actually
        //    have `<image>` as a merged token, i.e. not a VL model).
        let probed = self.tokenizer.encode("<image>");
        if probed.len() != 1 {
            return Err(WickError::Backend(format!(
                "tokenizer doesn't have `<image>` as a single token \
                 (got {} tokens) — model isn't VL-shaped",
                probed.len(),
            )));
        }
        let image_marker_id = probed[0];

        // 3. Resolve special tokens for the image envelope.
        let img_start = self
            .tokenizer
            .special_token_id("<|image_start|>")
            .ok_or_else(|| {
                WickError::Backend(
                    "tokenizer missing `<|image_start|>` special token — \
                     model isn't VL-shaped"
                        .into(),
                )
            })?;
        let img_end = self
            .tokenizer
            .special_token_id("<|image_end|>")
            .ok_or_else(|| {
                WickError::Backend(
                    "tokenizer missing `<|image_end|>` special token — \
                     model isn't VL-shaped"
                        .into(),
                )
            })?;

        // 4. Tokenize the rendered text + walk for marker positions.
        //    Splice plan is computed eagerly so we can validate the
        //    marker count BEFORE mutating session state.
        let tokens = self.tokenizer.encode(&rendered);
        let segments = splice_image_markers(&tokens, image_marker_id);
        let marker_count = segments
            .iter()
            .filter(|s| matches!(s, ChatTemplateSegment::Image))
            .count();
        if marker_count != images.len() {
            return Err(WickError::Backend(format!(
                "rendered chat template has {marker_count} `<image>` \
                 markers but caller supplied {} images",
                images.len(),
            )));
        }

        // 5. Walk segments and feed the session. Past this point
        //    state is mutated; failures propagate.
        let mut img_idx = 0;
        for seg in &segments {
            match *seg {
                ChatTemplateSegment::Text { start, end } => {
                    self.append_tokens(&tokens[start..end])?;
                }
                ChatTemplateSegment::Image => {
                    self.append_tokens(&[img_start])?;
                    self.append_image(images[img_idx])?;
                    self.append_tokens(&[img_end])?;
                    img_idx += 1;
                }
            }
        }
        Ok(())
    }

    /// Stub for builds without `vl-preprocess`. Same signature so
    /// shared code (wasm / mobile) that conditionally calls into
    /// VL still type-checks; always returns `UnsupportedModality`.
    #[cfg(not(feature = "vl-preprocess"))]
    pub fn append_chat_with_images(
        &mut self,
        _messages: &[crate::tokenizer::ChatMessageMultimodal],
        _images: &[&[u8]],
        _add_generation_prompt: bool,
    ) -> Result<(), WickError> {
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

        let decode_start = Instant::now();
        let mut finish = FinishReason::MaxTokens;
        let mut generated: u32 = 0;
        let mut pos = self.current_pos;
        let mut pending: Vec<u32> = Vec::with_capacity(opts.flush_every_tokens.max(1) as usize);
        let mut last_flush = Instant::now();

        let flush_n = opts.flush_every_tokens.max(1) as usize;
        let flush_ms = opts.flush_every_ms;

        // Early exit before consuming logits or touching the RNG. A no-op
        // `generate()` at full context or with max_tokens=0 has zero
        // side effects — important for stochastic split-generation
        // reproducibility and for callers polling capacity.
        if opts.max_tokens == 0 {
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
        if self.current_pos >= self.max_seq_len {
            sink.on_done(FinishReason::ContextFull);
            let decode_ms = decode_start.elapsed().as_millis() as u32;
            return Ok(GenerateSummary {
                tokens_generated: 0,
                prompt_eval_tokens,
                prompt_eval_ms,
                decode_ms,
                finish_reason: FinishReason::ContextFull,
            });
        }

        // Take the prefill logits. In stochastic mode we keep them pristine
        // across the whole loop and store back at exit for chainable
        // multi-call `generate()`. In greedy mode, logits are only used
        // for the INITIAL argmax; subsequent tokens come from
        // `forward_greedy()` which skips the vocab-sized GPU→CPU readback
        // and returns the argmax token directly — a real win on Metal
        // (~hundreds of μs per token saved at 64 K vocab).
        let mut logits = self.last_logits.take().ok_or(WickError::EmptyInput)?;

        // Greedy mode is decided once per generate call: deterministic
        // argmax + `forward_greedy`. Stochastic mode samples with RNG +
        // `forward` (keeps logits, supports chaining).
        let greedy = opts.temperature <= 0.0 || opts.top_k == 1;

        // Stochastic-only state. Allocating the scratch buffer and syncing
        // the sampler are skipped in greedy mode where neither is touched.
        let mut sample_scratch: Vec<f32> = if greedy {
            Vec::new()
        } else {
            self.sync_sampler_from_opts(opts);
            Vec::with_capacity(logits.len())
        };

        // Greedy-mode token state: the first token comes from `cpu_argmax`
        // on the prefill logits; each subsequent iteration's token comes
        // from the previous `forward_greedy()` return value. No RNG is
        // touched in this path — argmax is deterministic.
        let mut greedy_next: u32 = if greedy {
            crate::sampler::cpu_argmax(&logits)
        } else {
            0
        };

        // Decode loop. One body handles both modes, branching on `greedy`:
        //
        //   Stop checks (no RNG, no forward)
        //   ├─ greedy:     token = greedy_next
        //   └─ stochastic: token = sampler.sample(scratch)  (one RNG advance)
        //   EOS check
        //   Emit + flush
        //   ├─ greedy:     greedy_next = forward_greedy(&[token], pos)  (no vocab readback)
        //   └─ stochastic: logits = forward(&[token], pos)              (keeps logits)
        //   pos += 1, second cancel check
        //
        // Stochastic sampling happens INSIDE the loop body so the RNG
        // advances exactly once per emitted token (plus once on EOS).
        // That preserves seeded-split-generation reproducibility: a
        // single `generate(N)` advances RNG the same number of steps as
        // two `generate(N/2)` calls with the same seed.
        loop {
            if self.cancel.load(Ordering::Relaxed) {
                finish = FinishReason::Cancelled;
                break;
            }
            if generated >= opts.max_tokens {
                break;
            }
            if pos >= self.max_seq_len {
                finish = FinishReason::ContextFull;
                break;
            }

            let token = if greedy {
                greedy_next
            } else {
                sample_scratch.clear();
                sample_scratch.extend_from_slice(&logits);
                self.sampler.sample(&mut sample_scratch)
            };

            if self.tokenizer.eos_token() == Some(token) || opts.stop_tokens.contains(&token) {
                finish = FinishReason::Stop;
                break;
            }

            pending.push(token);
            generated += 1;

            let should_flush_n = pending.len() >= flush_n;
            let should_flush_t =
                flush_ms > 0 && last_flush.elapsed().as_millis() >= flush_ms as u128;
            if should_flush_n || should_flush_t {
                sink.on_text_tokens(&pending);
                pending.clear();
                last_flush = Instant::now();
            }

            if greedy {
                // Fast path: argmax on GPU, returns the 4-byte next token.
                // Skips the vocab-sized logits readback. `logits` goes stale
                // — that's fine, greedy mode never reads it after the
                // initial argmax.
                greedy_next = self.model.forward_greedy(&[token], pos, &mut self.state);
            } else {
                // Stochastic: full forward. Logits stay pristine for the
                // next iteration's sample and for chaining across calls.
                logits = self.model.forward(&[token], pos, &mut self.state);
            }
            pos += 1;
            self.position_atomic.store(pos as u32, Ordering::Relaxed);

            if self.cancel.load(Ordering::Relaxed) {
                finish = FinishReason::Cancelled;
                break;
            }
        }

        if !pending.is_empty() {
            sink.on_text_tokens(&pending);
        }

        self.current_pos = pos;
        self.position_atomic
            .store(self.current_pos as u32, Ordering::Relaxed);

        // Chain support:
        //
        // - Stochastic: `logits` holds the most recent `forward()` output
        //   (pristine). Save so the next `generate()` can continue
        //   without an intervening `append_tokens`.
        //
        // - Greedy, `generated > 0`: at least one `forward_greedy()` ran,
        //   advancing state and leaving `logits` stale (never read after
        //   the initial argmax). Clear — honest "nothing to save."
        //   Subsequent greedy `generate()` needs an `append_tokens` first,
        //   matching the standard chat loop (append user → generate →
        //   append next user → generate).
        //
        // - Greedy, `generated == 0`: we broke before any `forward_greedy`
        //   (e.g., cancel-before-first-iter, or the first predicted token
        //   was already EOS). `logits` still holds the untouched prefill
        //   distribution; restoring it keeps the session usable for a
        //   retry without the caller having to re-prefill.
        if greedy && generated > 0 {
            self.last_logits = None;
        } else {
            self.last_logits = Some(logits);
        }

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
            seed: self.config.seed,
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
    use crate::manifest::InferenceType;

    #[test]
    fn capabilities_from_inference_type_covers_every_variant() {
        let text = ModalityCapabilities::from_inference_type(&InferenceType::LlamaCppTextToText);
        assert!(text.text_in && text.text_out);
        assert!(!text.audio_in && !text.audio_out && !text.image_in);

        let audio = ModalityCapabilities::from_inference_type(&InferenceType::LlamaCppLfm2AudioV1);
        assert!(audio.text_in && audio.text_out);
        assert!(audio.audio_in && audio.audio_out);
        assert!(!audio.image_in);

        // VL bundles report `text_and_image_in` now that
        // `Session::append_image` is fully wired (Phase 3 slice 1).
        let vl = ModalityCapabilities::from_inference_type(&InferenceType::LlamaCppImageToText);
        assert!(vl.text_in && vl.text_out && vl.image_in);
        assert!(!vl.audio_in && !vl.audio_out);

        // Unknown variants fall back to text-only so an unfamiliar bundle
        // at least reports safe minimums rather than crashing.
        let unknown = ModalityCapabilities::from_inference_type(&InferenceType::Unknown(
            "llama.cpp/x".into(),
        ));
        assert!(unknown.text_in && unknown.text_out);
        assert!(!unknown.audio_in && !unknown.audio_out && !unknown.image_in);
    }

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

    // Integration tests that need a real model live under
    // `wick/tests/session_chain.rs` (gated behind `#[ignore]` and a
    // `find_model()` helper so they skip silently when no GGUF is
    // available locally). Unit tests here stay dep-free.

    /// Token stream with no `<image>` markers collapses to a single
    /// `Text` segment covering the whole range.
    #[test]
    fn splice_image_markers_no_markers_one_text_run() {
        let segs = splice_image_markers(&[1, 2, 3, 4], 99);
        assert_eq!(segs, vec![ChatTemplateSegment::Text { start: 0, end: 4 }]);
    }

    /// Single mid-stream marker splits into Text - Image - Text.
    #[test]
    fn splice_image_markers_mid_stream() {
        let segs = splice_image_markers(&[1, 2, 99, 3, 4], 99);
        assert_eq!(
            segs,
            vec![
                ChatTemplateSegment::Text { start: 0, end: 2 },
                ChatTemplateSegment::Image,
                ChatTemplateSegment::Text { start: 3, end: 5 },
            ]
        );
    }

    /// Marker at index 0: leading text run is elided so the segment
    /// list stays tight (no zero-length spans).
    #[test]
    fn splice_image_markers_at_start() {
        let segs = splice_image_markers(&[99, 1, 2], 99);
        assert_eq!(
            segs,
            vec![
                ChatTemplateSegment::Image,
                ChatTemplateSegment::Text { start: 1, end: 3 },
            ]
        );
    }

    /// Marker at the final index: trailing text run is elided.
    #[test]
    fn splice_image_markers_at_end() {
        let segs = splice_image_markers(&[1, 2, 99], 99);
        assert_eq!(
            segs,
            vec![
                ChatTemplateSegment::Text { start: 0, end: 2 },
                ChatTemplateSegment::Image,
            ]
        );
    }

    /// Two adjacent markers: empty-text-run between them is elided.
    #[test]
    fn splice_image_markers_adjacent_markers() {
        let segs = splice_image_markers(&[1, 99, 99, 2], 99);
        assert_eq!(
            segs,
            vec![
                ChatTemplateSegment::Text { start: 0, end: 1 },
                ChatTemplateSegment::Image,
                ChatTemplateSegment::Image,
                ChatTemplateSegment::Text { start: 3, end: 4 },
            ]
        );
    }

    /// Two well-separated markers — the canonical multi-image case.
    /// Verifies image count round-trips for caller validation.
    #[test]
    fn splice_image_markers_two_separated() {
        let segs = splice_image_markers(&[1, 99, 2, 99, 3], 99);
        let images = segs
            .iter()
            .filter(|s| matches!(s, ChatTemplateSegment::Image))
            .count();
        assert_eq!(images, 2);
        assert_eq!(
            segs,
            vec![
                ChatTemplateSegment::Text { start: 0, end: 1 },
                ChatTemplateSegment::Image,
                ChatTemplateSegment::Text { start: 2, end: 3 },
                ChatTemplateSegment::Image,
                ChatTemplateSegment::Text { start: 4, end: 5 },
            ]
        );
    }

    /// All-marker stream — no text runs at all.
    #[test]
    fn splice_image_markers_all_markers() {
        let segs = splice_image_markers(&[99, 99, 99], 99);
        assert_eq!(
            segs,
            vec![
                ChatTemplateSegment::Image,
                ChatTemplateSegment::Image,
                ChatTemplateSegment::Image,
            ]
        );
    }

    /// Empty stream — empty segment list.
    #[test]
    fn splice_image_markers_empty() {
        let segs = splice_image_markers(&[], 99);
        assert!(segs.is_empty());
    }
}
