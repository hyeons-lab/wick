//! `wick-ffi` — foreign-language bindings to [`wick`] via UniFFI.
//!
//! This crate exposes a subset of the `wick` inference engine to
//! Kotlin, Swift, Python, and any other language UniFFI supports. It
//! is structured around the **proc-macro** path (rather than a UDL
//! file) so the Rust types we expose are the source of truth and
//! annotations stay colocated with the code they describe.
//!
//! ## Current surface
//!
//! Engine-level:
//! - [`WickEngine`] — load a model; open sessions; introspect metadata
//!   + capabilities.
//! - [`EngineConfig`] + [`BackendPreference`] — load-time config.
//! - [`ModelMetadata`] + [`ModalityCapabilities`] — model-level info.
//!
//! Session-level:
//! - [`Session`] — stateful inference handle (one per conversation).
//!   `append_text` / `append_tokens` for input, synchronous
//!   [`Session::generate`] returning [`GenerateOutput`] (tokens +
//!   [`GenerateSummary`]), or [`Session::generate_streaming`] that
//!   delivers tokens + audio frames through a foreign [`ModalitySink`]
//!   as they're produced.
//! - [`SessionConfig`] + [`KvCompression`] — per-session knobs.
//! - [`GenerateOpts`] + [`FinishReason`] — per-call decode config + exit reason.
//! - [`Session::cancel`] / [`Session::position`] for cooperative
//!   interrupt + progress monitoring across threads.
//! - [`ModalitySink`] — UniFFI foreign-trait callback for streaming
//!   decode output to Kotlin / Swift / Python implementations.
//!
//! Error:
//! - [`FfiError`] — minimal error shell (string-based). Typed variants
//!   (`ContextOverflow { max_seq_len, by }` etc.) split into separate
//!   enum cases in a later PR.
//!
//! ## Not exposed yet
//!
//! Future PRs grow the surface per the roadmap in
//! `wick-ffi/README.md`. Highlights: `async` `generate` via
//! `#[uniffi::export(async_runtime = "tokio")]` (PR 5), remote URL
//! loading through `BundleRepo` (gated on the `remote` feature),
//! binding generation for Kotlin / Swift / Python, and typed error
//! variants.
//!
//! ## Design notes
//!
//! - **Wrapper types, not annotations on `wick` core.** Every
//!   UniFFI-exposed type is a wrapper defined in this crate with
//!   `From` conversions to/from the `wick` equivalent. The core crate
//!   stays UniFFI-agnostic.
//! - **`u64` on the wire, `usize` internally.** UniFFI records can't
//!   marshal `usize` (pointer-sized). Convert at the boundary.

use std::sync::Arc;

uniffi::setup_scaffolding!();

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Minimal error type for v1 of the FFI surface. Wraps a human-readable
/// message; the typed variants (`ContextOverflow`, `UnsupportedModality`,
/// etc.) will land as additional variants on this enum in a later PR
/// so callers can pattern-match on error class rather than string-sniff.
#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum FfiError {
    #[error("{message}")]
    Backend { message: String },
}

impl From<wick::WickError> for FfiError {
    fn from(e: wick::WickError) -> Self {
        FfiError::Backend {
            message: e.to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Config + enums
// ---------------------------------------------------------------------------

/// Compute-backend selector. Mirrors [`wick::BackendPreference`];
/// kept as a separate type so the `wick` crate doesn't carry UniFFI
/// annotations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum BackendPreference {
    /// Probe Metal → GPU → CPU at load time.
    Auto,
    Cpu,
    /// `wgpu` (Vulkan / Metal / DX12). Requires the `gpu` feature.
    Gpu,
    /// Native Metal. Requires the `metal` feature + macOS.
    Metal,
}

impl From<BackendPreference> for wick::BackendPreference {
    fn from(b: BackendPreference) -> Self {
        match b {
            BackendPreference::Auto => wick::BackendPreference::Auto,
            BackendPreference::Cpu => wick::BackendPreference::Cpu,
            BackendPreference::Gpu => wick::BackendPreference::Gpu,
            BackendPreference::Metal => wick::BackendPreference::Metal,
        }
    }
}

impl From<wick::BackendPreference> for BackendPreference {
    fn from(b: wick::BackendPreference) -> Self {
        match b {
            wick::BackendPreference::Auto => BackendPreference::Auto,
            wick::BackendPreference::Cpu => BackendPreference::Cpu,
            wick::BackendPreference::Gpu => BackendPreference::Gpu,
            wick::BackendPreference::Metal => BackendPreference::Metal,
        }
    }
}

/// Per-engine configuration at load time. Mirrors [`wick::EngineConfig`]
/// with `u64` fields (UniFFI doesn't marshal `usize`).
#[derive(Debug, Clone, uniffi::Record)]
pub struct EngineConfig {
    /// KV-cache capacity in tokens. Capped by the model's own
    /// `max_seq_len`. Pass `0` to use the model's full declared
    /// `max_seq_len` (translated to `usize::MAX` internally, then
    /// capped by the loader).
    pub context_size: u64,
    pub backend: BackendPreference,
}

impl Default for EngineConfig {
    fn default() -> Self {
        // Delegate to `wick::EngineConfig::default()` so the
        // defaults stay in one place. `usize → u64` is infallible on
        // every platform wick targets (`usize` is 32 or 64 bit; both
        // fit in u64).
        let core = wick::EngineConfig::default();
        Self {
            context_size: core.context_size as u64,
            backend: core.backend.into(),
        }
    }
}

impl TryFrom<EngineConfig> for wick::EngineConfig {
    type Error = FfiError;

    fn try_from(c: EngineConfig) -> Result<Self, FfiError> {
        // Checked `u64 → usize` conversion. On 32-bit targets (Android
        // armv7 is still a supported ABI) `u64` can exceed `usize::MAX`
        // and a bare `as usize` would silently truncate — producing a
        // much smaller KV cache than the caller intended. Surface the
        // overflow as a typed error instead.
        let context_size = if c.context_size == 0 {
            // Sentinel for "use model default" — wick caps at model.max_seq_len.
            usize::MAX
        } else {
            usize::try_from(c.context_size).map_err(|_| FfiError::Backend {
                message: format!(
                    "context_size {} exceeds usize::MAX on this target",
                    c.context_size
                ),
            })?
        };
        // `..Default::default()` is intentional forward-compat — under
        // the `remote` feature `wick::EngineConfig` gains a
        // `bundle_repo: Option<BundleRepo>` field, and the spread
        // supplies `None` without this file needing to know. Without
        // the feature clippy sees two specified fields + a redundant
        // update; the #[allow] keeps the spread in place so a future
        // remote-feature activation doesn't require a diff here.
        #[allow(clippy::needless_update)]
        Ok(wick::EngineConfig {
            context_size,
            backend: c.backend.into(),
            ..Default::default()
        })
    }
}

// ---------------------------------------------------------------------------
// Metadata + capabilities
// ---------------------------------------------------------------------------

/// Short summary of a loaded model. Mirrors [`wick::ModelMetadata`].
#[derive(Debug, Clone, uniffi::Record)]
pub struct ModelMetadata {
    pub architecture: String,
    pub max_seq_len: u32,
    pub vocab_size: u32,
    pub has_chat_template: bool,
    pub quantization: String,
    /// Mirror of GGUF `tokenizer.ggml.add_bos_token`. Consumers that
    /// want to insert a BOS at the head of a raw prompt should honor it.
    pub add_bos_token: bool,
}

impl From<&wick::ModelMetadata> for ModelMetadata {
    fn from(m: &wick::ModelMetadata) -> Self {
        ModelMetadata {
            architecture: m.architecture.clone(),
            max_seq_len: m.max_seq_len,
            vocab_size: m.vocab_size,
            has_chat_template: m.has_chat_template,
            quantization: m.quantization.clone(),
            add_bos_token: m.add_bos_token,
        }
    }
}

/// Modality support flags for a loaded model. Mirrors
/// [`wick::ModalityCapabilities`].
#[derive(Debug, Clone, Copy, uniffi::Record)]
pub struct ModalityCapabilities {
    pub text_in: bool,
    pub text_out: bool,
    pub image_in: bool,
    pub audio_in: bool,
    pub audio_out: bool,
}

impl From<wick::ModalityCapabilities> for ModalityCapabilities {
    fn from(c: wick::ModalityCapabilities) -> Self {
        ModalityCapabilities {
            text_in: c.text_in,
            text_out: c.text_out,
            image_in: c.image_in,
            audio_in: c.audio_in,
            audio_out: c.audio_out,
        }
    }
}

// ---------------------------------------------------------------------------
// WickEngine
// ---------------------------------------------------------------------------

/// Owning handle to a loaded model. Mirrors [`wick::WickEngine`];
/// `#[uniffi::Object]` requires `Arc<Self>` wrapping which matches how
/// the underlying engine is already used internally.
#[derive(uniffi::Object)]
pub struct WickEngine {
    inner: wick::WickEngine,
}

#[uniffi::export]
impl WickEngine {
    /// Load a model from a local filesystem path. Accepts the same
    /// inputs as the native [`wick::WickEngine::from_path`]:
    /// a bare `.gguf`, a LeapBundles `.json` manifest, or a directory
    /// containing exactly one `.json` manifest.
    ///
    /// Remote URLs in manifests are rejected in this PR — `BundleRepo`
    /// wiring lands when the `remote` feature activates here.
    #[uniffi::constructor]
    pub fn from_path(path: String, config: EngineConfig) -> Result<Arc<Self>, FfiError> {
        let inner = wick::WickEngine::from_path(&path, config.try_into()?)?;
        Ok(Arc::new(Self { inner }))
    }

    /// Short summary of the loaded model (architecture, vocab size,
    /// max context, etc.). Returns a `Clone` of the stored metadata.
    pub fn metadata(&self) -> ModelMetadata {
        ModelMetadata::from(self.inner.metadata())
    }

    /// What this model accepts as input / emits as output. Derived at
    /// load time from the manifest's `inference_type`.
    pub fn capabilities(&self) -> ModalityCapabilities {
        self.inner.capabilities().into()
    }
}

// ---------------------------------------------------------------------------
// Session types (PR 3)
// ---------------------------------------------------------------------------

/// KV-cache compression mode. Mirrors [`wick::kv_cache::KvCompression`].
/// `TurboQuant` is honored by the CPU backend only; Metal / GPU ignore
/// the setting and use the f32 path.
#[derive(Debug, Clone, Default, uniffi::Enum)]
pub enum KvCompression {
    /// No compression — f32 keys and values (default).
    #[default]
    None,
    /// TurboQuant compression. Both `keys` + `values` true is the
    /// production configuration; toggling them individually is
    /// primarily for debugging the drift contribution of each side.
    /// `seed` drives the per-layer randomized Hadamard rotations.
    TurboQuant { seed: u64, keys: bool, values: bool },
}

impl From<KvCompression> for wick::kv_cache::KvCompression {
    fn from(c: KvCompression) -> Self {
        match c {
            KvCompression::None => wick::kv_cache::KvCompression::None,
            KvCompression::TurboQuant { seed, keys, values } => {
                wick::kv_cache::KvCompression::TurboQuant { seed, keys, values }
            }
        }
    }
}

impl From<wick::kv_cache::KvCompression> for KvCompression {
    fn from(c: wick::kv_cache::KvCompression) -> Self {
        match c {
            wick::kv_cache::KvCompression::None => KvCompression::None,
            wick::kv_cache::KvCompression::TurboQuant { seed, keys, values } => {
                KvCompression::TurboQuant { seed, keys, values }
            }
        }
    }
}

/// Per-session configuration. Mirrors [`wick::SessionConfig`].
#[derive(Debug, Clone, uniffi::Record)]
pub struct SessionConfig {
    /// Cap on total tokens held in KV. `None` → model's default
    /// `max_seq_len`.
    pub max_seq_len: Option<u32>,
    /// KV cache compression mode.
    pub kv_compression: KvCompression,
    /// Pinned-prefix length for Phase-1.5 context shift on overflow.
    /// `0` disables shift; overflow returns `ContextOverflow` error.
    pub n_keep: u32,
    /// Deterministic sampling seed. `None` = fresh entropy per call.
    pub seed: Option<u64>,
    /// Chunked-prefill ubatch size. `0` = monolithic prefill.
    pub ubatch_size: u32,
}

impl Default for SessionConfig {
    fn default() -> Self {
        // Delegate to `wick::SessionConfig::default()` so the defaults
        // stay in one place; `kv_compression` flows through the
        // wrapper's `From` impl (both directions live above).
        let core = wick::SessionConfig::default();
        Self {
            max_seq_len: core.max_seq_len,
            kv_compression: core.kv_compression.into(),
            n_keep: core.n_keep,
            seed: core.seed,
            ubatch_size: core.ubatch_size,
        }
    }
}

impl From<SessionConfig> for wick::SessionConfig {
    fn from(c: SessionConfig) -> Self {
        wick::SessionConfig {
            max_seq_len: c.max_seq_len,
            kv_compression: c.kv_compression.into(),
            n_keep: c.n_keep,
            seed: c.seed,
            ubatch_size: c.ubatch_size,
        }
    }
}

/// Per-call decode options. Mirrors [`wick::GenerateOpts`].
///
/// `flush_every_tokens` / `flush_every_ms` are accepted but have no
/// effect under the synchronous [`Session::generate`] — they're
/// meaningful once streaming (foreign-trait `ModalitySink`) lands
/// in a follow-up PR. Including them in the record now keeps the FFI
/// surface stable across that transition.
#[derive(Debug, Clone, uniffi::Record)]
pub struct GenerateOpts {
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    /// Reserved — the sampler doesn't implement rep-penalty yet.
    pub repetition_penalty: f32,
    /// Early-stop IDs (EOS / instruction markers / end-of-turn).
    pub stop_tokens: Vec<u32>,
    /// Ignored under synchronous generate; reserved for streaming.
    pub flush_every_tokens: u32,
    /// Ignored under synchronous generate; reserved for streaming.
    pub flush_every_ms: u32,
}

impl Default for GenerateOpts {
    fn default() -> Self {
        let core = wick::GenerateOpts::default();
        Self {
            max_tokens: core.max_tokens,
            temperature: core.temperature,
            top_p: core.top_p,
            top_k: core.top_k,
            repetition_penalty: core.repetition_penalty,
            stop_tokens: core.stop_tokens,
            flush_every_tokens: core.flush_every_tokens,
            flush_every_ms: core.flush_every_ms,
        }
    }
}

impl From<GenerateOpts> for wick::GenerateOpts {
    fn from(o: GenerateOpts) -> Self {
        wick::GenerateOpts {
            max_tokens: o.max_tokens,
            temperature: o.temperature,
            top_p: o.top_p,
            top_k: o.top_k,
            repetition_penalty: o.repetition_penalty,
            stop_tokens: o.stop_tokens,
            flush_every_tokens: o.flush_every_tokens,
            flush_every_ms: o.flush_every_ms,
        }
    }
}

/// Why a decode loop exited. Mirrors [`wick::FinishReason`].
#[derive(Debug, Clone, uniffi::Enum)]
pub enum FinishReason {
    MaxTokens,
    Stop,
    Cancelled,
    ContextFull,
    Error { message: String },
}

impl From<wick::FinishReason> for FinishReason {
    fn from(r: wick::FinishReason) -> Self {
        match r {
            wick::FinishReason::MaxTokens => FinishReason::MaxTokens,
            wick::FinishReason::Stop => FinishReason::Stop,
            wick::FinishReason::Cancelled => FinishReason::Cancelled,
            wick::FinishReason::ContextFull => FinishReason::ContextFull,
            wick::FinishReason::Error(msg) => FinishReason::Error { message: msg },
        }
    }
}

/// Decode-run metadata. Mirrors [`wick::GenerateSummary`].
#[derive(Debug, Clone, uniffi::Record)]
pub struct GenerateSummary {
    pub tokens_generated: u32,
    pub prompt_eval_tokens: u32,
    pub prompt_eval_ms: u32,
    pub decode_ms: u32,
    pub finish_reason: FinishReason,
}

impl From<wick::GenerateSummary> for GenerateSummary {
    fn from(s: wick::GenerateSummary) -> Self {
        Self {
            tokens_generated: s.tokens_generated,
            prompt_eval_tokens: s.prompt_eval_tokens,
            prompt_eval_ms: s.prompt_eval_ms,
            decode_ms: s.decode_ms,
            finish_reason: s.finish_reason.into(),
        }
    }
}

/// Bundle of everything a synchronous `generate` call produces:
/// the generated token IDs plus the decode summary. The two are
/// returned together so callers don't have to manage a separate
/// callback channel; streaming (per-chunk delivery) lands in PR 4.
#[derive(Debug, Clone, uniffi::Record)]
pub struct GenerateOutput {
    /// Generated token IDs, in order, not including any prompt
    /// tokens. Decode with [`wick::tokenizer::BpeTokenizer`] on the
    /// Rust side or (once exposed) through a tokenizer handle on the
    /// FFI side.
    pub tokens: Vec<u32>,
    pub summary: GenerateSummary,
}

// ---------------------------------------------------------------------------
// ModalitySink (foreign trait — PR 4)
// ---------------------------------------------------------------------------

/// Streaming sink for decode output. Foreign callers implement this
/// trait (Kotlin class, Swift class, Python subclass) and pass an
/// `Arc<dyn ModalitySink>` to [`Session::generate_streaming`] to
/// receive tokens + audio frames + the finish reason as they happen.
///
/// All methods are required from foreign implementations (UniFFI 0.28
/// foreign traits don't carry Rust's default-impl fallbacks). Callers
/// that don't care about a modality can provide an empty body.
///
/// Threading: every method is invoked on the same Rust thread running
/// `generate` — the decode thread. If the foreign runtime requires
/// marshalling onto a different thread (e.g. Swift's `@MainActor`) it
/// is the implementer's responsibility to dispatch the call there.
#[uniffi::export(with_foreign)]
pub trait ModalitySink: Send + Sync {
    /// Called with each chunk of generated token IDs. Ownership of the
    /// `Vec<u32>` is transferred to the callback, so implementations
    /// may retain or store it directly if needed — no clone required.
    fn on_text_tokens(&self, tokens: Vec<u32>);

    /// Called with each chunk of generated PCM audio samples. Not
    /// called for text-only models; LFM2-Audio-class models emit here.
    /// The `sample_rate` is the model's native output rate (typically
    /// 24000 for LFM2-Audio) and is stable across the whole generate
    /// call.
    fn on_audio_frames(&self, pcm: Vec<f32>, sample_rate: u32);

    /// Called exactly once per [`Session::generate_streaming`] call,
    /// as the last thing before the wrapper returns. Fires for both
    /// success (`MaxTokens`, `Stop`, `Cancelled`, `ContextFull`) and
    /// failure paths: on error the wrapper synthesizes a
    /// [`FinishReason::Error`] so foreign consumers have a reliable
    /// end-of-stream signal regardless of how the call exits.
    fn on_done(&self, reason: FinishReason);
}

/// Adapter from the UniFFI foreign trait to the internal
/// [`wick::ModalitySink`]. Forwards every call; unavoidable `Vec`
/// copy per chunk because UniFFI can't marshal a borrowed `&[u32]`
/// or `&[f32]` across the ABI boundary. Impact is bounded: the
/// decode loop emits chunks of at most a few tokens at a time, so
/// the allocation volume is orders of magnitude lower than the decode
/// itself. For audio the copy is larger but a single frame per decode
/// step is still small (a few hundred f32s).
///
/// `done_called` tracks whether the underlying `wick::Session::generate`
/// fired `on_done`. The FFI wrapper uses this to synthesize a
/// terminal `on_done(Error)` if core returns an error before getting
/// to its own `on_done` call (currently only possible on
/// `WickError::EmptyInput`, but robust against future error paths).
/// Guards against double-firing if the core ever starts calling
/// `on_done` internally on error paths.
struct ForeignSinkAdapter {
    inner: Arc<dyn ModalitySink>,
    done_called: bool,
}

impl wick::ModalitySink for ForeignSinkAdapter {
    fn on_text_tokens(&mut self, tokens: &[u32]) {
        self.inner.on_text_tokens(tokens.to_vec());
    }
    fn on_audio_frames(&mut self, pcm: &[f32], sample_rate: u32) {
        self.inner.on_audio_frames(pcm.to_vec(), sample_rate);
    }
    fn on_done(&mut self, reason: wick::FinishReason) {
        self.done_called = true;
        self.inner.on_done(reason.into());
    }
}

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

/// Stateful inference handle. Wraps [`wick::Session`] behind a
/// `Mutex` so UniFFI's `Arc<Session>` shape works with methods that
/// need `&mut self` on the inner session (prefill, generate, reset).
///
/// Call [`WickEngine::new_session`] to open a session; the engine's
/// `Arc<Model>` and `Arc<BpeTokenizer>` are cloned into the new
/// session so it outlives the engine handle across FFI calls.
#[derive(uniffi::Object)]
pub struct Session {
    inner: std::sync::Mutex<wick::Session>,
    /// Cloned from the inner session at construction time. Shared
    /// atomic — `position()` / `cancel()` don't need to acquire the
    /// mutex, so they're safe to call from a different thread while
    /// `generate()` is running.
    position: Arc<std::sync::atomic::AtomicU32>,
    cancel: Arc<std::sync::atomic::AtomicBool>,
    /// Stored at construction so `capabilities()` doesn't need a lock.
    capabilities: ModalityCapabilities,
}

impl Session {
    /// Lock the inner session, converting `PoisonError` into
    /// `FfiError::Backend` instead of panicking. `expect` on a
    /// poisoned mutex would propagate as a panic across the FFI
    /// boundary — Kotlin / Swift / Python callers see that as an
    /// uncatchable abort of the host process, which is unusable in
    /// production. Returning an error lets callers decide whether to
    /// retry, reset, or surface the failure.
    ///
    /// A poisoned mutex here means a prior session method panicked
    /// while holding the lock — the session's internal state (KV
    /// cache, sampler, position counters) is therefore in an unknown
    /// state. The error message gives the caller enough context to
    /// decide whether to reset or drop the session entirely.
    fn lock_inner(&self) -> Result<std::sync::MutexGuard<'_, wick::Session>, FfiError> {
        self.inner.lock().map_err(|e| FfiError::Backend {
            message: format!(
                "session mutex poisoned (a prior call panicked mid-lock; session state is \
                 inconsistent): {e}"
            ),
        })
    }
}

#[uniffi::export]
impl Session {
    /// Append raw text to the context, running a prefill over just
    /// the new tokens. `EmptyInput` error if `text` is empty.
    pub fn append_text(&self, text: String) -> Result<(), FfiError> {
        self.lock_inner()?.append_text(&text)?;
        Ok(())
    }

    /// Append pre-tokenized IDs. Useful when the caller has its own
    /// tokenizer + chat-template pipeline.
    pub fn append_tokens(&self, tokens: Vec<u32>) -> Result<(), FfiError> {
        self.lock_inner()?.append_tokens(&tokens)?;
        Ok(())
    }

    /// Run autoregressive decode and return all emitted tokens +
    /// a summary. Synchronous — the call blocks until the decode
    /// loop exits (`max_tokens`, EOS, `cancel()`, or error).
    ///
    /// For streaming (per-chunk delivery) and async, see the PR 4 /
    /// PR 5 follow-ups in `wick-ffi/README.md`.
    pub fn generate(&self, opts: GenerateOpts) -> Result<GenerateOutput, FfiError> {
        // Collector sink: captures every token the decode loop emits.
        // `on_done` is invoked once at the end regardless of exit
        // reason; we read the Result from `session.generate` to see
        // whether the run succeeded.
        struct CollectSink(Vec<u32>);
        impl wick::ModalitySink for CollectSink {
            fn on_text_tokens(&mut self, tokens: &[u32]) {
                self.0.extend_from_slice(tokens);
            }
            fn on_done(&mut self, _reason: wick::FinishReason) {}
        }
        let mut sink = CollectSink(Vec::new());
        let summary = self.lock_inner()?.generate(&opts.into(), &mut sink)?;
        Ok(GenerateOutput {
            tokens: sink.0,
            summary: summary.into(),
        })
    }

    /// Run autoregressive decode, streaming every token (and audio
    /// frame, for audio-capable models) to a foreign [`ModalitySink`]
    /// as soon as it's produced. Returns only a [`GenerateSummary`] —
    /// token IDs are delivered through `sink.on_text_tokens`, not a
    /// return value.
    ///
    /// Synchronous: the call blocks on the decode thread and each
    /// `sink` method runs on that same thread before decoding
    /// continues. For async, see PR 5 in `wick-ffi/README.md`.
    ///
    /// **Callback reentrancy — deadlock hazard.** The session mutex is
    /// held for the entire call, and sink callbacks run while that
    /// lock is held. Calling back into methods that also take the
    /// mutex ([`Session::append_text`], [`Session::append_tokens`],
    /// [`Session::generate`], [`Session::generate_streaming`],
    /// [`Session::reset`]) from inside a sink method will deadlock.
    /// [`Session::cancel`] and [`Session::position`] are atomic-backed
    /// and safe to call from the sink or from any other thread.
    ///
    /// Cancellation: call [`Session::cancel`] from any thread (or from
    /// inside a sink callback on this thread) to terminate the loop at
    /// the next between-token check; `sink.on_done` fires with
    /// [`FinishReason::Cancelled`].
    ///
    /// End-of-stream guarantee: `sink.on_done` fires exactly once per
    /// call, even on error paths. If the underlying decode returns an
    /// error before reaching its own `on_done` call (e.g.,
    /// `EmptyInput` with no prefill logits), the wrapper synthesizes
    /// a terminal `on_done(FinishReason::Error { message })` so
    /// foreign consumers have a reliable end-of-stream signal
    /// regardless of how the call exits.
    pub fn generate_streaming(
        &self,
        opts: GenerateOpts,
        sink: Arc<dyn ModalitySink>,
    ) -> Result<GenerateSummary, FfiError> {
        let mut adapter = ForeignSinkAdapter {
            inner: sink,
            done_called: false,
        };
        // Scope the lock so the synthesized on_done on the error path
        // doesn't run with the session mutex held — foreign sinks
        // already have to avoid session-reentrancy during success
        // callbacks; reusing that contract on the error path keeps
        // the hazard set minimal.
        let outcome = match self.lock_inner() {
            Ok(mut guard) => guard
                .generate(&opts.into(), &mut adapter)
                .map_err(FfiError::from),
            Err(e) => Err(e),
        };
        match outcome {
            Ok(summary) => Ok(summary.into()),
            Err(err) => {
                if !adapter.done_called {
                    let message = match &err {
                        FfiError::Backend { message } => message.clone(),
                    };
                    adapter.inner.on_done(FinishReason::Error { message });
                }
                Err(err)
            }
        }
    }

    /// Current KV position — how many tokens live in the cache.
    /// Atomic-backed; safe to call from a different thread while
    /// `generate()` is in flight.
    pub fn position(&self) -> u32 {
        self.position.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Signal in-flight `generate()` to exit with
    /// `FinishReason::Cancelled` at the next between-token check.
    /// Safe from any thread. No-op if no `generate()` is running.
    pub fn cancel(&self) {
        self.cancel
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Drop cached state + resample the seed. After `reset()` the
    /// session behaves like a freshly-opened one (same
    /// model/tokenizer/config, no accumulated context).
    ///
    /// Returns `Result` so a poisoned-mutex case surfaces as an error
    /// instead of panicking across the FFI boundary.
    pub fn reset(&self) -> Result<(), FfiError> {
        self.lock_inner()?.reset();
        Ok(())
    }

    /// Capabilities reported by the loaded model. Cheap — reads a
    /// cached copy, no lock.
    pub fn capabilities(&self) -> ModalityCapabilities {
        self.capabilities
    }
}

// Session-level method on WickEngine.
#[uniffi::export]
impl WickEngine {
    /// Open a new [`Session`] sharing this engine's model + tokenizer
    /// by `Arc` clone. The returned session outlives `&self`; the
    /// engine keeps the shared state live for every session it hands
    /// out. Cheap — no model load, just config + state allocation.
    pub fn new_session(&self, config: SessionConfig) -> Arc<Session> {
        let session = self.inner.new_session(config.into());
        let position = session.position_handle();
        let cancel = session.cancel_handle();
        let capabilities = session.capabilities().into();
        Arc::new(Session {
            inner: std::sync::Mutex::new(session),
            position,
            cancel,
            capabilities,
        })
    }
}

// ---------------------------------------------------------------------------
// Smoke test (from PR 1)
// ---------------------------------------------------------------------------

/// Version string of the `wick-ffi` crate. Useful as a smoke test
/// from the foreign-language side — if this is callable, the binding
/// pipeline works end-to-end.
#[uniffi::export]
pub fn wick_ffi_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_non_empty() {
        // Smoke test: proves the proc-macro expanded and the export is
        // callable. No shape check on the string — SemVer allows
        // pre-release + build-metadata suffixes (`0.1.0-alpha.1+deadbe`)
        // that a strict `x.y.z` split would reject.
        let v = wick_ffi_version();
        assert!(!v.is_empty(), "version string must not be empty");
    }

    #[test]
    fn engine_config_default_roundtrips_to_wick() {
        let ffi = EngineConfig::default();
        let core: wick::EngineConfig = ffi.try_into().unwrap();
        assert_eq!(core.context_size, 4096);
        assert_eq!(core.backend, wick::BackendPreference::Auto);
    }

    #[test]
    fn engine_config_zero_context_size_means_max() {
        // `0` on the wire is the FFI's "use model default" signal;
        // translate to `usize::MAX` so wick caps at model.max_seq_len.
        let ffi = EngineConfig {
            context_size: 0,
            backend: BackendPreference::Cpu,
        };
        let core: wick::EngineConfig = ffi.try_into().unwrap();
        assert_eq!(core.context_size, usize::MAX);
    }

    #[test]
    fn engine_config_oversize_context_errors_on_32bit_targets() {
        // On 32-bit targets, `u64::MAX` exceeds `usize::MAX` and the
        // checked conversion must surface an error rather than
        // silently truncating. On 64-bit targets `usize::MAX ==
        // u64::MAX`, so the conversion succeeds — skip the assert
        // there. This test proves the error path compiles + is
        // reachable under the narrow condition where it matters.
        let ffi = EngineConfig {
            context_size: u64::MAX,
            backend: BackendPreference::Cpu,
        };
        let result: Result<wick::EngineConfig, FfiError> = ffi.try_into();
        #[cfg(target_pointer_width = "32")]
        {
            let err = result.expect_err("u64::MAX must fail on 32-bit");
            match err {
                FfiError::Backend { message } => {
                    assert!(
                        message.contains("exceeds usize::MAX"),
                        "unexpected: {message}"
                    );
                }
            }
        }
        #[cfg(target_pointer_width = "64")]
        {
            // On 64-bit `u64::MAX == usize::MAX`; the sentinel check
            // has already rejected `0`, so `u64::MAX` converts cleanly.
            let core = result.expect("u64::MAX fits usize::MAX on 64-bit");
            assert_eq!(core.context_size, usize::MAX);
        }
    }

    #[test]
    fn backend_preference_roundtrips() {
        for ffi in [
            BackendPreference::Auto,
            BackendPreference::Cpu,
            BackendPreference::Gpu,
            BackendPreference::Metal,
        ] {
            let core: wick::BackendPreference = ffi.into();
            let back: BackendPreference = core.into();
            assert_eq!(ffi, back, "{ffi:?} didn't round-trip");
        }
    }

    #[test]
    fn ffi_error_wraps_wick_error_message() {
        let we = wick::WickError::EmptyInput;
        let expected = we.to_string();
        let fe: FfiError = we.into();
        match fe {
            FfiError::Backend { message } => assert_eq!(message, expected),
        }
    }

    #[test]
    fn session_config_default_roundtrips_to_wick() {
        let ffi = SessionConfig::default();
        let core: wick::SessionConfig = ffi.into();
        let default_core = wick::SessionConfig::default();
        assert_eq!(core.max_seq_len, default_core.max_seq_len);
        assert_eq!(core.n_keep, default_core.n_keep);
        assert_eq!(core.seed, default_core.seed);
        assert_eq!(core.ubatch_size, default_core.ubatch_size);
    }

    #[test]
    fn kv_compression_none_roundtrips() {
        let ffi = KvCompression::None;
        let core: wick::kv_cache::KvCompression = ffi.into();
        assert!(matches!(core, wick::kv_cache::KvCompression::None));
    }

    #[test]
    fn kv_compression_turboquant_roundtrips() {
        let ffi = KvCompression::TurboQuant {
            seed: 42,
            keys: true,
            values: false,
        };
        let core: wick::kv_cache::KvCompression = ffi.into();
        match core {
            wick::kv_cache::KvCompression::TurboQuant { seed, keys, values } => {
                assert_eq!(seed, 42);
                assert!(keys);
                assert!(!values);
            }
            _ => panic!("expected TurboQuant variant"),
        }
    }

    #[test]
    fn generate_opts_default_roundtrips_to_wick() {
        let ffi = GenerateOpts::default();
        let core: wick::GenerateOpts = ffi.into();
        let default_core = wick::GenerateOpts::default();
        // Field-by-field so a future wick field-add breaks here loudly.
        assert_eq!(core.max_tokens, default_core.max_tokens);
        assert_eq!(core.temperature, default_core.temperature);
        assert_eq!(core.top_p, default_core.top_p);
        assert_eq!(core.top_k, default_core.top_k);
        assert_eq!(core.repetition_penalty, default_core.repetition_penalty);
        assert_eq!(core.stop_tokens, default_core.stop_tokens);
        assert_eq!(core.flush_every_tokens, default_core.flush_every_tokens);
        assert_eq!(core.flush_every_ms, default_core.flush_every_ms);
    }

    #[test]
    fn finish_reason_covers_every_variant() {
        use wick::FinishReason as Core;
        let cases = [
            (Core::MaxTokens, "MaxTokens"),
            (Core::Stop, "Stop"),
            (Core::Cancelled, "Cancelled"),
            (Core::ContextFull, "ContextFull"),
            (Core::Error("boom".into()), "Error"),
        ];
        for (core, tag) in cases {
            let ffi: FinishReason = core.into();
            match (&ffi, tag) {
                (FinishReason::MaxTokens, "MaxTokens") => {}
                (FinishReason::Stop, "Stop") => {}
                (FinishReason::Cancelled, "Cancelled") => {}
                (FinishReason::ContextFull, "ContextFull") => {}
                (FinishReason::Error { message }, "Error") => assert_eq!(message, "boom"),
                _ => panic!("variant mismatch: {ffi:?} tagged {tag}"),
            }
        }
    }

    /// Exercises the ForeignSinkAdapter by implementing the FFI
    /// `ModalitySink` trait from Rust (what UniFFI codegens the foreign
    /// binding to look like on the Rust side) and driving it through
    /// the internal `wick::ModalitySink` impl. Confirms:
    /// - `on_text_tokens` forwards with the exact bytes.
    /// - `on_audio_frames` forwards with the exact bytes + rate.
    /// - `on_done` forwards and maps the FinishReason through `.into()`.
    /// - All three run without the foreign side needing `&mut self` —
    ///   interior mutability (Mutex/atomic) is the caller's burden,
    ///   mirroring what Kotlin/Swift will see.
    #[test]
    fn foreign_sink_adapter_forwards_every_method() {
        use std::sync::Mutex;
        use wick::ModalitySink as CoreSink;

        #[derive(Default)]
        struct Recorder {
            text: Mutex<Vec<u32>>,
            audio: Mutex<Vec<(Vec<f32>, u32)>>,
            done: Mutex<Option<FinishReason>>,
        }

        impl ModalitySink for Recorder {
            fn on_text_tokens(&self, tokens: Vec<u32>) {
                self.text.lock().unwrap().extend(tokens);
            }
            fn on_audio_frames(&self, pcm: Vec<f32>, sample_rate: u32) {
                self.audio.lock().unwrap().push((pcm, sample_rate));
            }
            fn on_done(&self, reason: FinishReason) {
                *self.done.lock().unwrap() = Some(reason);
            }
        }

        let recorder: Arc<Recorder> = Arc::new(Recorder::default());
        let mut adapter = ForeignSinkAdapter {
            inner: recorder.clone() as Arc<dyn ModalitySink>,
            done_called: false,
        };

        // Drive the adapter as wick's decode loop would.
        adapter.on_text_tokens(&[1, 2, 3]);
        adapter.on_text_tokens(&[4, 5]);
        adapter.on_audio_frames(&[0.1, 0.2, 0.3], 24_000);
        adapter.on_done(wick::FinishReason::MaxTokens);

        assert_eq!(&*recorder.text.lock().unwrap(), &[1, 2, 3, 4, 5]);
        let audio = recorder.audio.lock().unwrap();
        assert_eq!(audio.len(), 1);
        assert_eq!(audio[0].0, vec![0.1, 0.2, 0.3]);
        assert_eq!(audio[0].1, 24_000);
        assert!(matches!(
            &*recorder.done.lock().unwrap(),
            Some(FinishReason::MaxTokens)
        ));
        assert!(
            adapter.done_called,
            "adapter.done_called must flip after on_done"
        );
    }

    /// Before `adapter.on_done` has been forwarded, `done_called`
    /// stays `false`. Protects the error-synthesis branch in
    /// `Session::generate_streaming` — we can only safely synthesize
    /// a terminal `on_done(Error)` on failure when the inner
    /// `wick::Session::generate` hasn't already fired its own `on_done`.
    #[test]
    fn adapter_done_called_starts_false_and_guards_error_synthesis() {
        use std::sync::Mutex;

        #[derive(Default)]
        struct Recorder {
            calls: Mutex<usize>,
        }
        impl ModalitySink for Recorder {
            fn on_text_tokens(&self, _: Vec<u32>) {}
            fn on_audio_frames(&self, _: Vec<f32>, _: u32) {}
            fn on_done(&self, _: FinishReason) {
                *self.calls.lock().unwrap() += 1;
            }
        }

        let recorder: Arc<Recorder> = Arc::new(Recorder::default());
        let adapter = ForeignSinkAdapter {
            inner: recorder.clone() as Arc<dyn ModalitySink>,
            done_called: false,
        };

        // Simulate the error-branch logic: never forwarded on_done,
        // so the wrapper should synthesize one.
        assert!(!adapter.done_called);
        if !adapter.done_called {
            adapter.inner.on_done(FinishReason::Error {
                message: "simulated pre-decode error".into(),
            });
        }
        assert_eq!(*recorder.calls.lock().unwrap(), 1, "synthesized once");

        // And the double-fire guard: if done_called were already true,
        // the wrapper must skip synthesis.
        let adapter_already_done = ForeignSinkAdapter {
            inner: recorder.clone() as Arc<dyn ModalitySink>,
            done_called: true,
        };
        if !adapter_already_done.done_called {
            adapter_already_done
                .inner
                .on_done(FinishReason::Error { message: "".into() });
        }
        assert_eq!(
            *recorder.calls.lock().unwrap(),
            1,
            "still one — no double-fire"
        );
    }
}
