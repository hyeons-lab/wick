//! `cera-ffi` — foreign-language bindings to [`cera`] via UniFFI.
//!
//! This crate exposes a subset of the `cera` inference engine to
//! Kotlin, Swift, Python, and any other language UniFFI supports. It
//! is structured around the **proc-macro** path (rather than a UDL
//! file) so the Rust types we expose are the source of truth and
//! annotations stay colocated with the code they describe.
//!
//! ## Current surface
//!
//! Engine-level:
//! - [`CeraEngine`] — model loader, session factory, and tokenizer
//!   accessor. Constructors: [`CeraEngine::from_path`] for a local
//!   GGUF, manifest, or directory; [`CeraEngine::from_bundle_id`]
//!   for LeapBundles-style remote loading;
//!   [`CeraEngine::from_path_async`], [`CeraEngine::from_bytes_async`]
//!   and [`CeraEngine::from_bundle_id_async`] for the tokio-async
//!   variants, which move the load off the calling thread.
//!   Tokenizer methods ([`CeraEngine::encode_text`],
//!   [`CeraEngine::decode_tokens`],
//!   [`CeraEngine::apply_chat_template`]) let foreign callers
//!   tokenize / detokenize / format messages without `Session`.
//! - [`ChatMessage`] — input record for `apply_chat_template`.
//! - [`BundleRepo`] — HTTP model cache; construct once per app and
//!   attach via [`EngineConfig::bundle_repo`] for remote loading.
//!   [`BundleRepo::with_progress`] takes a [`DownloadProgressSink`]
//!   foreign-trait callback for download progress UI.
//!   [`BundleRepo::cache_size`] / [`BundleRepo::clear_cache`] for
//!   on-disk usage queries + cleanup.
//! - [`EngineConfig`] + [`BackendPreference`] — load-time config.
//! - [`ModelMetadata`] + [`ModalityCapabilities`] — model-level info.
//!
//! Session-level:
//! - [`Session`] — stateful inference handle (one per conversation).
//!   `append_text` / `append_tokens` for input, synchronous
//!   [`Session::generate`] returning [`GenerateOutput`] (tokens +
//!   [`GenerateSummary`]), or [`Session::generate_streaming`] that
//!   delivers tokens + audio frames through a foreign [`ModalitySink`]
//!   as they're produced. Async twins
//!   [`Session::generate_async`] + [`Session::generate_streaming_async`]
//!   let foreign async runtimes — Kotlin coroutines, Swift `async`,
//!   Python `asyncio` — `.await` decode without stalling the caller.
//! - [`SessionConfig`] + [`KvCompression`] — per-session knobs.
//! - [`GenerateOpts`] + [`FinishReason`] — per-call decode config + exit reason.
//! - [`Session::cancel`] / [`Session::position`] for cooperative
//!   interrupt + progress monitoring across threads.
//! - [`ModalitySink`] — UniFFI foreign-trait callback for streaming
//!   decode output to Kotlin / Swift / Python implementations.
//!
//! Voice Activity Detection (VAD):
//! - [`FfiSileroVad`] — native Silero VAD v5 speech detector for 16 kHz and 8 kHz audio.
//! - [`FfiVadIterator`] — stateful speech boundary detector emitting start/end events for live audio streams.
//! - [`FfiVadConfig`], [`FfiVadSampleRate`], [`FfiSpeechTimestamp`], [`FfiVadEvent`].
//!
//! Keyword Spotting (KWS):
//! - [`FfiHotwordDetector`]: native keyword spotting detector evaluating acoustic models from GGUF.
//! - [`FfiHotwordIterator`]: streaming manager with integrated VAD gating, ring buffering, and debounce.
//! - [`FfiHotwordConfig`], [`FfiHotwordScore`], [`FfiHotwordEvent`].
//!
//! Speech Recognition (ASR / Whisper):
//! - [`FfiWhisperModel`]: standalone pure-Rust OpenAI Whisper speech recognition engine.
//! - [`FfiWhisperTranscribeOpts`]: options for language, translation, timestamps, max tokens, and temperature.
//! - [`whisper_default_transcribe_opts`]: factory for default transcription options.
//!
//! Error:
//! - [`FfiError`] — typed error surface mirroring [`cera::CeraError`]
//!   one-to-one (`ContextOverflow { max_seq_len, by }`,
//!   `UnsupportedModality`, `UnsupportedInferenceType`, `Busy`,
//!   `Cancelled`, `EmptyInput`, `Io`), plus `Backend` for FFI-internal
//!   errors that have no cera analog (poisoned mutex, `JoinError`).
//!   The `From<CeraError>` conversion is exhaustive — new cera
//!   variants break compilation, never silently fall through to
//!   `Backend`.
//!
//! ## Not exposed yet
//!
//! Future PRs grow the surface per the roadmap in
//! `cera-ffi/README.md`. Highlights: remote URL loading through
//! `BundleRepo` (gated on the `remote` feature) and a parity harness
//! crate that cross-checks `cera-ffi` output against a reference
//! implementation.
//!
//! ## Design notes
//!
//! - **Wrapper types, not annotations on `cera` core.** Every
//!   UniFFI-exposed type is a wrapper defined in this crate with
//!   `From` conversions to/from the `cera` equivalent. The core crate
//!   stays UniFFI-agnostic.
//! - **`u64` on the wire, `usize` internally.** UniFFI records can't
//!   marshal `usize` (pointer-sized). Convert at the boundary.

use std::sync::Arc;

uniffi::setup_scaffolding!();

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Typed error surface for `cera-ffi`. Mirrors [`cera::CeraError`] one-
/// to-one so foreign callers can pattern-match on error class (Kotlin
/// `when`, Swift `switch`, Python `match`) instead of string-sniffing
/// a generic message.
///
/// `Backend` is **not** a silent fallback for unmapped `cera::CeraError`
/// variants — the `From<CeraError>` impl is exhaustive, so adding a
/// new cera variant breaks compilation here. `Backend` exists solely
/// for FFI-internal errors that have no cera analog: `JoinError` from
/// a panicking `spawn_blocking` task, a poisoned `Session::inner`
/// mutex, 32-bit `u64 → usize` overflow in `EngineConfig::try_from`.
///
/// Every variant carries the data needed to act on it:
/// `ContextOverflow` exposes `max_seq_len` and `by` so callers can
/// reset or truncate rather than re-reading the message;
/// `UnsupportedInferenceType` exposes the offending value;
/// `Io` preserves the underlying OS error message as a string since
/// `io::Error` isn't UniFFI-marshallable.
///
/// `#[error(...)]` format strings match `cera::CeraError` exactly for
/// every shared variant, so `Display` output is identical whether the
/// error originates from cera directly or routes through the FFI
/// wrapper. Pinned by `ffi_error_display_matches_cera_error_for_every_shared_variant`.
#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum FfiError {
    /// The loaded model doesn't support the modality the caller
    /// requested (e.g. `append_audio` on a text-only LLM).
    #[error("modality not supported by this model")]
    UnsupportedModality,

    /// The manifest's `inference_type` is one cera doesn't recognize
    /// at this version. Field carries the offending string.
    #[error("inference_type `{inference_type}` is not supported in this version of cera")]
    UnsupportedInferenceType { inference_type: String },

    /// A concurrent `generate*` call is already in flight on this
    /// session. Rust side guards with a mutex; this surfaces when the
    /// FFI detects contention.
    #[error("session is busy with another operation")]
    Busy,

    /// The caller (or the cancel-on-drop guard) flipped the cancel
    /// atomic mid-call. Surfaces from `append_text`, `append_tokens`,
    /// and `append_audio` when chunked prefill detects the cancel
    /// flag between micro-batches and aborts (see
    /// [`cera::Session::append_tokens`] for the chunked-prefill
    /// mechanism). Call [`Session::clear_cancel`] to reset the flag
    /// so the next call can proceed.
    ///
    /// `generate` reports cancellation via a different path: the
    /// call still returns `Ok` with a [`GenerateOutput`] whose
    /// `finish_reason` is set to `Cancelled`. Two paths because
    /// chunked prefill has nothing useful to return on cancel (no
    /// decoded tokens) while decode has accumulated tokens worth
    /// preserving.
    #[error("cancelled")]
    Cancelled,

    /// The context window is full and the session can't shift to make
    /// room (e.g. `n_keep == 0`, TurboQuant caches, or the active
    /// model doesn't support rope-shift). `max_seq_len` is the cap
    /// that was hit; `by` is the overshoot in tokens.
    #[error("context window ({max_seq_len}) exceeded by {by} tokens")]
    ContextOverflow { max_seq_len: u32, by: u32 },

    /// Input buffer was empty (e.g. `append_text("")`, or decode with
    /// no prefill state).
    #[error("empty input")]
    EmptyInput,

    /// Filesystem / mmap / network error surfaced from cera. The
    /// underlying `io::Error` isn't marshallable, so the message is
    /// flattened to a string. Callers that need the raw kind should
    /// parse the `detail` field or open an issue to request a typed
    /// field.
    ///
    /// Field is named `detail` rather than `message` because UniFFI's
    /// 0.31 Kotlin generator emits `class Io(val `message`) : FfiException()`
    /// AND `override val message` in the body when the field is literally
    /// named `message`, producing a "conflicting declarations" error
    /// (the constructor param collides with the inherited
    /// `Throwable.message` override). Renaming to `detail` sidesteps
    /// the collision.
    ///
    /// Format string matches `cera::CeraError::Io`'s `"io: {0}"` so
    /// foreign `.toString()` / `String(describing:)` gives the same
    /// output Rust consumers see.
    #[error("io: {detail}")]
    Io { detail: String },

    /// FFI-internal error with no cera analog: `JoinError` from a
    /// panicking `spawn_blocking` task, poisoned `Session::inner`
    /// mutex, 32-bit `u64 → usize` overflow in `EngineConfig::try_from`,
    /// or `cera::CeraError::Backend` routed through the `From` impl.
    /// Format string matches `cera::CeraError::Backend`'s
    /// `"backend: {0}"` — FFI-internal constructors that have already
    /// formatted a descriptive message (e.g. "generate_async join
    /// error: ...") still read cleanly with the `backend:` label.
    ///
    /// Field is named `detail` rather than `message` for the same
    /// `Throwable.message` collision reason as [`FfiError::Io`].
    #[error("backend: {detail}")]
    Backend { detail: String },

    /// The GBNF grammar string passed in `GenerateOpts.grammar` failed to
    /// compile. Grammar compilation happens in the FFI wrapper (the compiled
    /// grammar object can't cross the boundary, so callers pass the source text
    /// and it's parsed here). `detail` carries the parser's diagnostic.
    #[error("grammar: {detail}")]
    GrammarParse { detail: String },

    /// A token id passed to `hidden_states_for_tokens` (or another
    /// token-taking method) was `>= vocab_size`. Returned as a typed error
    /// rather than tripping the model-layer `assert!` (whose panic would
    /// unwind through the held session lock and poison it). Mirrors
    /// `cera::CeraError::InvalidToken`.
    #[error("token id {id} out of range (vocab_size {vocab_size})")]
    InvalidToken { id: u32, vocab_size: u32 },

    /// A LoRA adapter failed to load ([`LoraAdapters::from_gguf`] /
    /// [`LoraAdapters::from_safetensors`]) or was incompatible with the model at
    /// attach time (wrong dimensions). `detail` carries the diagnostic.
    #[error("lora: {detail}")]
    LoraParse { detail: String },

    /// A large model/KV allocation could not be satisfied — the device is out of
    /// memory for this model at this context size. Returned instead of aborting
    /// the process, so a caller can fall back (smaller model or context) or
    /// surface a clean error. Mirrors `cera::CeraError::OutOfMemory`.
    #[error("out of memory: could not allocate {requested_bytes} bytes")]
    OutOfMemory { requested_bytes: u64 },

    /// A backend's KV-cache compression mode is fixed by the first session that
    /// configures it — the compressed and uncompressed caches have different
    /// buffer layouts (and the uncompressed one is f32 on CPU/wgpu but f16 on
    /// Metal), so only the configured one is ever allocated. Two sessions wanting
    /// different modes need two `CeraModel` instances. Mirrors
    /// `cera::CeraError::KvCompressionConflict`.
    #[error(
        "model already configured for KV compression mode `{configured}`; \
         cannot reconfigure to `{requested}` — create a separate model instance"
    )]
    KvCompressionConflict {
        configured: String,
        requested: String,
    },

    /// The adapter fits the model, but the active backend has no hook for
    /// something it adapts. Mirrors [`cera::CeraError::LoraUnsupportedByBackend`].
    ///
    /// Separate from [`FfiError::LoraParse`] because the two need different
    /// handling on the foreign side: `LoraParse` means the adapter or the model
    /// pairing is wrong, while this one means only the backend is, so a caller
    /// can retry on CPU instead of surfacing "bad adapter" to a user. Today the
    /// case is a routed feed-forward (mixture-of-experts) delta on a GPU
    /// backend.
    ///
    /// **Appended, not grouped next to `LoraParse`.** UniFFI serializes this
    /// enum by ordinal, and the committed Kotlin/Swift/Dart bindings decode it
    /// the same way, so inserting mid-enum renumbers every later variant and a
    /// prebuilt consumer would decode this one as whatever now holds its old
    /// ordinal. New variants go at the end.
    #[error("LoRA adapter not supported by this backend: {detail}")]
    LoraUnsupportedByBackend { detail: String },
}

impl From<cera::CeraError> for FfiError {
    fn from(e: cera::CeraError) -> Self {
        // Match exhaustively on the upstream enum so a future cera
        // variant-add breaks compilation here loudly rather than
        // silently routing through the `Backend` catch-all.
        match e {
            cera::CeraError::UnsupportedModality => FfiError::UnsupportedModality,
            cera::CeraError::UnsupportedInferenceType(s) => {
                FfiError::UnsupportedInferenceType { inference_type: s }
            }
            cera::CeraError::Busy => FfiError::Busy,
            cera::CeraError::Cancelled => FfiError::Cancelled,
            cera::CeraError::ContextOverflow { max_seq_len, by } => {
                FfiError::ContextOverflow { max_seq_len, by }
            }
            cera::CeraError::EmptyInput => FfiError::EmptyInput,
            cera::CeraError::InvalidToken { id, vocab_size } => {
                FfiError::InvalidToken { id, vocab_size }
            }
            cera::CeraError::Backend(s) => FfiError::Backend { detail: s },
            cera::CeraError::OutOfMemory { requested_bytes } => {
                FfiError::OutOfMemory { requested_bytes }
            }
            cera::CeraError::KvCompressionConflict {
                configured,
                requested,
            } => FfiError::KvCompressionConflict {
                configured,
                requested,
            },
            cera::CeraError::LoraDimMismatch(s) => FfiError::LoraParse { detail: s },
            cera::CeraError::LoraUnsupportedByBackend(s) => {
                FfiError::LoraUnsupportedByBackend { detail: s }
            }
            cera::CeraError::Io(io_err) => FfiError::Io {
                detail: io_err.to_string(),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Config + enums
// ---------------------------------------------------------------------------

/// Compute-backend selector. Mirrors [`cera::BackendPreference`];
/// kept as a separate type so the `cera` crate doesn't carry UniFFI
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

impl From<BackendPreference> for cera::BackendPreference {
    fn from(b: BackendPreference) -> Self {
        match b {
            BackendPreference::Auto => cera::BackendPreference::Auto,
            BackendPreference::Cpu => cera::BackendPreference::Cpu,
            BackendPreference::Gpu => cera::BackendPreference::Gpu,
            BackendPreference::Metal => cera::BackendPreference::Metal,
        }
    }
}

impl From<cera::BackendPreference> for BackendPreference {
    fn from(b: cera::BackendPreference) -> Self {
        match b {
            cera::BackendPreference::Auto => BackendPreference::Auto,
            cera::BackendPreference::Cpu => BackendPreference::Cpu,
            cera::BackendPreference::Gpu => BackendPreference::Gpu,
            cera::BackendPreference::Metal => BackendPreference::Metal,
        }
    }
}

/// Per-engine configuration at load time. Mirrors [`cera::EngineConfig`]
/// with `u64` fields (UniFFI doesn't marshal `usize`).
#[derive(Debug, Clone, uniffi::Record)]
pub struct EngineConfig {
    /// KV-cache capacity in tokens. Capped by the model's own
    /// `max_seq_len`. Pass `0` to use the model's full declared
    /// `max_seq_len` (translated to `usize::MAX` internally, then
    /// capped by the loader).
    #[uniffi(default = 4096)]
    pub context_size: u64,
    pub backend: BackendPreference,
    /// Bundle repository for resolving `http(s)://` URLs in manifests
    /// (or for [`CeraEngine::from_bundle_id`]). `None` means "remote
    /// URLs will fail with an error"; set this to a [`BundleRepo`]
    /// rooted at a persistent cache directory to enable remote
    /// downloads. Construct the repo once + reuse it across engine
    /// loads so its HTTP client pool + on-disk cache are shared.
    #[uniffi(default = None)]
    pub bundle_repo: Option<Arc<BundleRepo>>,
    /// Optional path to a DSpark speculative draft model GGUF file.
    #[uniffi(default = None)]
    pub draft_model: Option<String>,
    /// Whether to prefer GPU depthformer for audio decoder generation.
    #[uniffi(default = false)]
    pub gpu_depthformer: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        // Delegate to `cera::EngineConfig::default()` so the
        // defaults stay in one place. `usize → u64` is infallible on
        // every platform cera targets (`usize` is 32 or 64 bit; both
        // fit in u64).
        let core = cera::EngineConfig::default();
        Self {
            context_size: core.context_size as u64,
            backend: core.backend.into(),
            // `bundle_repo` defaults to None; foreign callers who want
            // remote-URL loading set it explicitly before passing the
            // config to `CeraEngine::from_path` / `from_bundle_id`.
            bundle_repo: None,
            draft_model: core.draft_model.map(|p| p.to_string_lossy().to_string()),
            gpu_depthformer: core.gpu_depthformer,
        }
    }
}

impl TryFrom<EngineConfig> for cera::EngineConfig {
    type Error = FfiError;

    fn try_from(c: EngineConfig) -> Result<Self, FfiError> {
        // Checked `u64 → usize` conversion. On 32-bit targets (Android
        // armv7 is still a supported ABI) `u64` can exceed `usize::MAX`
        // and a bare `as usize` would silently truncate — producing a
        // much smaller KV cache than the caller intended. Surface the
        // overflow as a typed error instead.
        let context_size = if c.context_size == 0 {
            // Sentinel for "use model default" — cera caps at model.max_seq_len.
            usize::MAX
        } else {
            usize::try_from(c.context_size).map_err(|_| FfiError::Backend {
                detail: format!(
                    "context_size {} exceeds usize::MAX on this target",
                    c.context_size
                ),
            })?
        };
        // Under the `remote` feature `cera::EngineConfig` carries a
        // `bundle_repo: Option<cera::bundle::BundleRepo>` field. Pull
        // the inner from our FFI `Arc<BundleRepo>` wrapper (cheap —
        // `cera::bundle::BundleRepo` is `Clone` and the two reqwest
        // clients inside share their connection pool via Arc-backed
        // refcounts).
        Ok(cera::EngineConfig {
            context_size,
            backend: c.backend.into(),
            draft_model: c.draft_model.map(std::path::PathBuf::from),
            gpu_depthformer: c.gpu_depthformer,
            bundle_repo: c.bundle_repo.map(|r| r.inner.clone()),
        })
    }
}

// ---------------------------------------------------------------------------
// Metadata + capabilities
// ---------------------------------------------------------------------------

/// Short summary of a loaded model. Mirrors [`cera::ModelMetadata`].
#[derive(Debug, Clone, uniffi::Record)]
pub struct ModelMetadata {
    pub architecture: String,
    pub max_seq_len: u32,
    pub vocab_size: u32,
    pub has_chat_template: bool,
    pub quantization: String,
    /// Mirror of GGUF `tokenizer.ggml.add_bos_token`. Consumers that
    /// want to insert a BOS at the head of a raw prompt should honor it —
    /// or, better, tokenize via `encode_text_special`, which applies both
    /// this and `add_eos_token`.
    pub add_bos_token: bool,
    /// Mirror of GGUF `tokenizer.ggml.add_eos_token`. See `add_bos_token`.
    pub add_eos_token: bool,
    /// SIMD backend tier the runtime resolved for this host (e.g.
    /// `"neon+dotprod"`, `"avx2"`, `"scalar"`). A host property, not
    /// model-specific — surfaced here so consumers fetching metadata also
    /// get backend diagnostics for telemetry / bug reports. For the full
    /// feature list, see [`cpu_backend_report`].
    pub cpu_backend: String,
}

impl From<&cera::ModelMetadata> for ModelMetadata {
    fn from(m: &cera::ModelMetadata) -> Self {
        ModelMetadata {
            architecture: m.architecture.clone(),
            max_seq_len: m.max_seq_len,
            vocab_size: m.vocab_size,
            has_chat_template: m.has_chat_template,
            quantization: m.quantization.clone(),
            add_bos_token: m.add_bos_token,
            add_eos_token: m.add_eos_token,
            cpu_backend: cera::cpu_tier().label().to_string(),
        }
    }
}

/// Modality support flags for a loaded model. Mirrors
/// [`cera::ModalityCapabilities`].
#[derive(Debug, Clone, Copy, uniffi::Record)]
pub struct ModalityCapabilities {
    pub text_in: bool,
    pub text_out: bool,
    pub image_in: bool,
    pub audio_in: bool,
    pub audio_out: bool,
}

impl From<cera::ModalityCapabilities> for ModalityCapabilities {
    fn from(c: cera::ModalityCapabilities) -> Self {
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
// ChatMessage (PR 13 — chat template input)
// ---------------------------------------------------------------------------

/// One message in a chat-template conversation. Mirrors
/// [`cera::tokenizer::ChatMessage`]. Pass a `Vec<ChatMessage>` to
/// [`CeraEngine::apply_chat_template`] to render the model's
/// chat-template (Jinja2 from GGUF metadata) into a prompt string
/// ready to feed into [`Session::append_text`].
///
/// `role` follows the OpenAI / chat-template convention — typically
/// one of `"system"`, `"user"`, `"assistant"`, occasionally
/// `"tool"`. cera-ffi doesn't validate the role string; whatever is
/// passed flows directly into the Jinja template. Whether an
/// unknown role errors or silently no-ops depends on the template's
/// own logic — many templates have an explicit error path for
/// unrecognized roles, but it's template-dependent rather than
/// enforced by [`CeraEngine::apply_chat_template`].
#[derive(Debug, Clone, uniffi::Record)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl From<ChatMessage> for cera::tokenizer::ChatMessage {
    fn from(m: ChatMessage) -> Self {
        cera::tokenizer::ChatMessage {
            role: m.role,
            content: m.content,
        }
    }
}

// ---------------------------------------------------------------------------
// Tool calling
// ---------------------------------------------------------------------------

/// The tool-call wire format a model family uses. Mirrors
/// [`cera::tools::ToolFormat`]. Get one from
/// [`CeraEngine::tool_format`] (auto-detected from the model) or set it
/// explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum ToolFormat {
    /// LFM2 / LFM2.5: Pythonic `[get_weather(city="Paris")]` in
    /// `<|tool_call_start|>…<|tool_call_end|>`.
    Lfm2Pythonic,
    /// Hermes / Qwen: JSON `{"name":…,"arguments":{…}}` in
    /// `<tool_call>…</tool_call>`.
    Hermes,
}

impl From<ToolFormat> for cera::tools::ToolFormat {
    fn from(f: ToolFormat) -> Self {
        match f {
            ToolFormat::Lfm2Pythonic => cera::tools::ToolFormat::Lfm2Pythonic,
            ToolFormat::Hermes => cera::tools::ToolFormat::Hermes,
        }
    }
}

impl From<cera::tools::ToolFormat> for ToolFormat {
    fn from(f: cera::tools::ToolFormat) -> Self {
        match f {
            cera::tools::ToolFormat::Lfm2Pythonic => ToolFormat::Lfm2Pythonic,
            cera::tools::ToolFormat::Hermes => ToolFormat::Hermes,
        }
    }
}

/// A tool the model may call. Mirrors [`cera::tools::ToolDef`], but the
/// JSON Schema for the arguments crosses the boundary as a JSON **string**
/// (`parameters_json`) since UniFFI has no arbitrary-JSON type. An empty
/// `parameters_json` means "no parameters".
#[derive(Debug, Clone, uniffi::Record)]
pub struct ToolDef {
    pub name: String,
    pub description: Option<String>,
    /// JSON Schema object for the arguments, as a JSON string (e.g.
    /// `{"type":"object","properties":{…},"required":[…]}`). Empty → none.
    pub parameters_json: String,
}

impl TryFrom<ToolDef> for cera::tools::ToolDef {
    type Error = FfiError;
    fn try_from(t: ToolDef) -> Result<Self, FfiError> {
        let parameters = if t.parameters_json.trim().is_empty() {
            serde_json::json!({ "type": "object", "properties": {} })
        } else {
            let v: serde_json::Value =
                serde_json::from_str(&t.parameters_json).map_err(|e| FfiError::Backend {
                    detail: format!("tool `{}` parameters_json is not valid JSON: {e}", t.name),
                })?;
            // A JSON Schema for arguments must be an object; a scalar/array
            // would silently yield zero constraints and can break the chat
            // template's `tool.parameters.properties` access.
            if !v.is_object() {
                return Err(FfiError::Backend {
                    detail: format!(
                        "tool `{}` parameters_json must be a JSON object schema, got {}",
                        t.name,
                        match v {
                            serde_json::Value::Array(_) => "an array",
                            serde_json::Value::String(_) => "a string",
                            serde_json::Value::Number(_) => "a number",
                            serde_json::Value::Bool(_) => "a boolean",
                            serde_json::Value::Null => "null",
                            serde_json::Value::Object(_) => "an object",
                        }
                    ),
                });
            }
            v
        };
        Ok(cera::tools::ToolDef {
            name: t.name,
            description: t.description,
            parameters,
        })
    }
}

/// A tool call parsed from model output. Mirrors [`cera::tools::ToolCall`];
/// `arguments_json` is the call's arguments encoded as a JSON string.
#[derive(Debug, Clone, uniffi::Record)]
pub struct ToolCall {
    pub name: String,
    /// The call's arguments as a JSON string — normally an object
    /// (e.g. `{"city":"Paris"}`), but a malformed Hermes/Qwen reply may pass
    /// through a non-object value, so decode defensively.
    pub arguments_json: String,
}

impl From<cera::tools::ToolCall> for ToolCall {
    fn from(c: cera::tools::ToolCall) -> Self {
        ToolCall {
            name: c.name,
            arguments_json: serde_json::to_string(&c.arguments)
                .unwrap_or_else(|_| "{}".to_string()),
        }
    }
}

fn to_core_tools(tools: Vec<ToolDef>) -> Result<Vec<cera::tools::ToolDef>, FfiError> {
    tools.into_iter().map(TryInto::try_into).collect()
}

/// Detect the tool-call format for a model architecture string (e.g.
/// `"lfm2"`, `"qwen3"`). Returns `None` for architectures with no known
/// convention — the caller may still choose a format explicitly.
#[uniffi::export]
pub fn detect_tool_format(architecture: String) -> Option<ToolFormat> {
    cera::tools::ToolFormat::detect(&architecture).map(Into::into)
}

/// Parse tool calls out of generated model text for the given `format`.
/// Returns an empty list when the reply contains no tool call (the model
/// answered in prose). Errors only when a call section is present but
/// unrecoverably malformed.
#[uniffi::export]
pub fn parse_tool_calls(text: String, format: ToolFormat) -> Result<Vec<ToolCall>, FfiError> {
    cera::tools::parse_tool_calls(&text, format.into())
        .map(|calls| calls.into_iter().map(Into::into).collect())
        .map_err(|e| FfiError::Backend {
            detail: format!("parse_tool_calls: {e}"),
        })
}

/// Build a GBNF grammar string constraining output to a valid call for one
/// of `tools`, in `format`. Put the result in `GenerateOpts.grammar` and set
/// `GenerateOpts.grammar_trigger_tokens` (see
/// [`CeraEngine::tool_call_start_token`]) for a lazy tool-call trigger.
#[uniffi::export]
pub fn tool_grammar(tools: Vec<ToolDef>, format: ToolFormat) -> Result<String, FfiError> {
    let core = to_core_tools(tools)?;
    cera::tools::tool_grammar(&core, format.into()).map_err(|e| FfiError::Backend {
        detail: format!("tool_grammar: {e}"),
    })
}

// ---------------------------------------------------------------------------
// LeapBundles catalog
// ---------------------------------------------------------------------------

/// One bundle published on `huggingface.co/LiquidAI/LeapBundles`: the
/// model directory plus every per-quant manifest inside it. Feed
/// `name` and one element of `quants` straight to
/// [`CeraEngine::from_bundle_id`].
///
/// Both fields are sorted ascending, so a menu built from this list is
/// stable across runs even if the upstream API reorders its response.
#[derive(Debug, Clone, uniffi::Record)]
pub struct LeapBundleEntry {
    pub name: String,
    pub quants: Vec<String>,
}

/// List every bundle published on `LiquidAI/LeapBundles`, so a picker
/// can offer `<name>, <quant>` pairs instead of making the user type a
/// bundle id. Pair with [`CeraEngine::from_bundle_id`], which takes
/// exactly these two strings.
///
/// One blocking HTTP GET with a 30 s timeout and no retry. Prefer
/// [`list_leap_bundles_async`] anywhere a UI thread is involved: this
/// twin stalls the calling thread for the whole round-trip.
///
/// Needs no [`BundleRepo`]: the catalog is a single small JSON
/// response and is deliberately not cached, so a picker opened twice
/// in one session reflects newly published bundles.
#[uniffi::export]
pub fn list_leap_bundles() -> Result<Vec<LeapBundleEntry>, FfiError> {
    cera::bundle::list_leap_bundles()
        .map(|entries| {
            entries
                .into_iter()
                .map(|e| LeapBundleEntry {
                    name: e.name,
                    quants: e.quants,
                })
                .collect()
        })
        .map_err(|e| FfiError::Backend {
            detail: format!("list_leap_bundles: {e}"),
        })
}

/// Async variant of [`list_leap_bundles`]: moves the blocking HTTP
/// round-trip onto a tokio blocking worker so a coroutine, a Swift
/// `async` context or a Dart `Future` can await the catalog without
/// stalling the thread that asked for it.
///
/// `async_runtime = "tokio"` is load-bearing, not decoration: it is
/// what makes uniffi poll this future inside a tokio context. Without
/// it the foreign executor drives the future with no runtime
/// installed and the `spawn_blocking` below panics with "must be
/// called from the context of a Tokio 1.x runtime" on the very first
/// call.
///
/// Cancellation: dropping the returned future aborts the task if it
/// has not started, so a dismissed picker does not leave a 30 s
/// blocking GET queued on the pool. A request already in flight runs
/// to completion; `reqwest::blocking` offers nothing to interrupt, and
/// the response is small.
#[uniffi::export(async_runtime = "tokio")]
pub async fn list_leap_bundles_async() -> Result<Vec<LeapBundleEntry>, FfiError> {
    spawn_blocking_guarded("list_leap_bundles_async", list_leap_bundles).await
}

// ---------------------------------------------------------------------------
// BundleRepo
// ---------------------------------------------------------------------------

/// Remote model-bundle downloader + on-disk cache. Wraps
/// [`cera::bundle::BundleRepo`]; construct once per application with
/// a persistent `store_dir` and reuse across engine loads so the
/// HTTP client pool + downloaded-file cache are shared.
///
/// On Android the `store_dir` should typically be
/// `Context.getFilesDir()` (persistent), not `getCacheDir()` (OS-
/// purgeable under storage pressure). On iOS / macOS, the app's
/// Application Support or a dedicated subdirectory under Documents
/// is a reasonable baseline.
///
/// Cache layout mirrors the remote URL structure under
/// `<store_dir>/huggingface.co/<full path>`, so inspecting the
/// on-disk state with a file browser is straightforward and multiple
/// cera-powered apps on the same device can share the same cache
/// directory without conflicting.
#[derive(Debug, uniffi::Object)]
pub struct BundleRepo {
    inner: cera::bundle::BundleRepo,
}

#[uniffi::export]
impl BundleRepo {
    /// Create a new repo rooted at `store_dir`. The directory doesn't
    /// need to exist yet — it's created on the first download. Pass
    /// the same path to subsequent runs to reuse the cached bundles.
    #[uniffi::constructor]
    pub fn new(store_dir: String) -> Arc<Self> {
        Arc::new(Self {
            inner: cera::bundle::BundleRepo::new(store_dir),
        })
    }

    /// Create a new repo rooted at `store_dir` with a foreign
    /// [`DownloadProgressSink`] attached. The sink fires periodically
    /// during cache-miss downloads (every ~256 KB written + once at
    /// end-of-stream). Cache-hit resolves don't fire any callbacks.
    /// The same sink receives events for every file the repo
    /// downloads — distinguish per-file progress by the `url`
    /// argument on each callback.
    ///
    /// Construction-time attachment (rather than per-call) matches
    /// how mobile apps drive a single download-progress UI across
    /// multiple files in one logical bundle (manifest + GGUF + …):
    /// one repo, one sink, one progress bar. If you need to tear
    /// down the sink mid-app-lifecycle, drop the repo + construct a
    /// new one — Arc-based, so all in-flight calls finish on the
    /// old sink and new calls go to the new one.
    #[uniffi::constructor]
    pub fn with_progress(store_dir: String, progress: Arc<dyn DownloadProgressSink>) -> Arc<Self> {
        let adapter: Arc<dyn cera::bundle::DownloadProgress> =
            Arc::new(DownloadProgressAdapter { inner: progress });
        Arc::new(Self {
            inner: cera::bundle::BundleRepo::with_progress(store_dir, adapter),
        })
    }

    /// The directory this repo caches bundles under. Matches what was
    /// passed to [`BundleRepo::new`] / [`BundleRepo::with_progress`],
    /// useful for log / telemetry.
    pub fn store_dir(&self) -> String {
        self.inner.store_dir().to_string_lossy().into_owned()
    }

    /// Total bytes currently held in the cache. Returns `0` if the
    /// `store_dir` doesn't exist yet (no downloads have run).
    /// O(n) over the cache contents; for a multi-GB cache it's a
    /// real walk, not a constant-time query — UIs surfacing the
    /// value should run it off the main thread (e.g. via
    /// `withContext(Dispatchers.IO)` on Kotlin or
    /// `Task.detached` on Swift).
    ///
    /// Mobile apps use this to drive a "Storage: X MB used" line in
    /// settings or to gate a "Clear cache" button on actual
    /// non-zero usage.
    pub fn cache_size(&self) -> Result<u64, FfiError> {
        Ok(self.inner.cache_size()?)
    }

    /// Wipe every file the repo has cached, leaving `store_dir`
    /// itself in place so subsequent downloads land in the same
    /// path. Idempotent — calling on an empty repo or non-existent
    /// `store_dir` is a no-op success.
    ///
    /// Mobile apps trigger this from a "Clear downloaded models"
    /// settings action. Caller is responsible for serializing
    /// against in-flight downloads — typically trivial since the
    /// action is user-driven.
    pub fn clear_cache(&self) -> Result<(), FfiError> {
        Ok(self.inner.clear_cache()?)
    }

    /// Download all assets for a bundle ID and quantization to the local cache
    /// without loading model weights into memory or creating an engine.
    pub fn download_bundle(&self, bundle_id: String, quant: String) -> Result<(), FfiError> {
        Ok(self.inner.download_bundle(&bundle_id, &quant)?)
    }
}

// ---------------------------------------------------------------------------
// DownloadProgressSink (foreign trait — PR 12)
// ---------------------------------------------------------------------------

/// Foreign-trait callback for download progress events from
/// [`BundleRepo::with_progress`]. Implementers (Kotlin class, Swift
/// class, Python subclass) drive a progress UI from these events.
///
/// All methods are required from foreign implementations (UniFFI
/// 0.31 foreign traits don't carry Rust default-impl fallbacks).
///
/// Threading: `on_progress` is invoked from the thread driving the
/// download. For sync `from_bundle_id` that's the caller's thread;
/// for `from_bundle_id_async` it's a tokio blocking worker. If your
/// progress UI requires marshalling onto a UI thread (`@MainActor`,
/// `runOnUiThread`, etc.), the implementer is responsible for the
/// dispatch.
#[uniffi::export(with_foreign)]
pub trait DownloadProgressSink: Send + Sync {
    /// Called periodically during a download. `bytes_downloaded` is
    /// monotonic across the same call's stream; `total_bytes` is the
    /// `Content-Length` reported by the server (may be `None` for
    /// chunked-transfer responses or when HEAD didn't surface a
    /// length). Same `url` value across all calls for one download
    /// — pattern-match on it to drive a per-file UI within a
    /// multi-file bundle download.
    ///
    /// Throttled by `cera-core` to ~256 KB granularity + one final
    /// callback at end-of-stream so the consumer always sees the
    /// final byte count.
    fn on_progress(&self, url: String, bytes_downloaded: u64, total_bytes: Option<u64>);
}

/// Adapter from the UniFFI foreign trait to cera-core's
/// [`cera::bundle::DownloadProgress`]. Same shape as
/// [`ForeignSinkAdapter`] for `ModalitySink` (PR 4): the foreign
/// arg's `&str` becomes an owned `String` because UniFFI can't
/// marshal a borrowed slice across the boundary.
struct DownloadProgressAdapter {
    inner: Arc<dyn DownloadProgressSink>,
}

// Manual Debug impl because `dyn DownloadProgressSink` is a UniFFI
// foreign-trait object — the foreign side (Kotlin / Swift / Python
// implementations) has no Rust Debug, so we can't blanket-derive.
// `cera::bundle::DownloadProgress` requires Debug for `BundleRepo`'s
// own derived Debug to work; printing the adapter as a typed handle
// is sufficient for any Rust-side log line that touches it.
impl std::fmt::Debug for DownloadProgressAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DownloadProgressAdapter")
            .field("inner", &"<foreign DownloadProgressSink>")
            .finish()
    }
}

impl cera::bundle::DownloadProgress for DownloadProgressAdapter {
    fn on_progress(&self, url: &str, bytes_downloaded: u64, total_bytes: Option<u64>) {
        self.inner
            .on_progress(url.to_string(), bytes_downloaded, total_bytes);
    }
}

// ---------------------------------------------------------------------------
// CeraEngine
// ---------------------------------------------------------------------------

/// Owning handle to a loaded model. Mirrors [`cera::CeraEngine`];
/// `#[uniffi::Object]` requires `Arc<Self>` wrapping which matches how
/// the underlying engine is already used internally.
#[derive(uniffi::Object)]
pub struct CeraEngine {
    inner: cera::CeraEngine,
}

#[uniffi::export]
impl CeraEngine {
    /// Load a model from a local filesystem path. Accepts the same
    /// inputs as the native [`cera::CeraEngine::from_path`]: a bare
    /// `.gguf`, a LeapBundles `.json` manifest, or a directory
    /// containing exactly one `.json` manifest.
    ///
    /// If the manifest carries `http(s)://` URLs for its files,
    /// `config.bundle_repo` must be set — otherwise those URLs fail
    /// to resolve. For a pure-local workflow (bundle already on
    /// disk) leave `bundle_repo = None`.
    #[uniffi::constructor]
    pub fn from_path(path: String, config: EngineConfig) -> Result<Arc<Self>, FfiError> {
        let inner = cera::CeraEngine::from_path(&path, config.try_into()?)?;
        Ok(Arc::new(Self { inner }))
    }

    /// Load a model from GGUF bytes already in memory.
    ///
    /// For callers with no filesystem to point [`CeraEngine::from_path`]
    /// at: a browser, an encrypted blob decrypted in memory, an asset
    /// read out of an archive. It is the one constructor a WebAssembly
    /// build can also offer, so code written against it ports across.
    ///
    /// **Not a streaming API.** GGUF is random-access: tensor data is
    /// addressed by offset and read throughout inference, so the whole
    /// file has to be resident before the first token. You can download
    /// over a stream, but you must accumulate it all before calling
    /// this. There is no partial-model inference.
    ///
    /// **Prefer [`CeraEngine::from_path`] whenever a path exists.** That
    /// route memory-maps the file, so tensor pages stay owned by the
    /// kernel: shared between processes and evictable under pressure.
    /// These bytes are committed resident memory for as long as the
    /// engine lives, which on a phone is the difference between a model
    /// the OS can page out and one that counts against your footprint.
    /// To load from the network on a platform that has a filesystem,
    /// stream to disk and use `from_path` (which is what [`BundleRepo`]
    /// does), rather than buffering the model here.
    ///
    /// Text-only: the bytes are a bare GGUF with no accompanying
    /// manifest, so there is nothing to point at a vision encoder or an
    /// audio decoder. Multimodal models need
    /// [`CeraEngine::from_parts`], `from_path`, or
    /// [`CeraEngine::from_bundle_id`]. `config.bundle_repo` is ignored.
    #[uniffi::constructor]
    pub fn from_bytes(bytes: Vec<u8>, config: EngineConfig) -> Result<Arc<Self>, FfiError> {
        let inner = cera::CeraEngine::from_bytes(bytes, config.try_into()?)?;
        Ok(Arc::new(Self { inner }))
    }

    /// Load a multi-file bundle from memory: the model GGUF plus its
    /// multimodal projector ("mmproj").
    ///
    /// This is the constructor a VL or audio model needs when there is
    /// no filesystem, and [`CeraEngine::from_bytes`] structurally cannot
    /// be: the vision tower and the audio encoder live in a *second*
    /// GGUF, and that one takes a single buffer. Same inputs and same
    /// rules as the wasm build's `fromGgufParts`, so a portable layer
    /// over both has one shape to target.
    ///
    /// `multimodal_projector` may be `None`, which makes this exactly
    /// `from_bytes` with an explicit config.
    ///
    /// **Modality is inferred from the arguments, not just the header.**
    /// Every published LFM2-VL model reports `architecture = "lfm2"`,
    /// the same string a text model reports, because the vision half is
    /// entirely in the mmproj. So supplying one alongside a text-arch
    /// model is taken as the statement of intent it is and loads as
    /// image-to-text; audio models already identify themselves and are
    /// unaffected. Pass `inference_type` explicitly to override
    /// (`"llama.cpp/text-to-text"`, `"llama.cpp/image-to-text"`,
    /// `"llama.cpp/lfm2-audio-v1"`).
    ///
    /// A malformed or mismatched mmproj is **not** fatal: it warns, and
    /// the bundle still serves text with `capabilities().image_in`
    /// staying false. That mirrors the path-based loaders rather than
    /// failing a whole load over a sidecar.
    ///
    /// **Prefer `from_path` whenever a path exists**, for the same
    /// memory reason as [`CeraEngine::from_bytes`]: these buffers are
    /// committed resident memory for the engine's lifetime, and a VL
    /// bundle is the model *plus* the tower. `config.bundle_repo` is
    /// ignored.
    #[uniffi::constructor]
    pub fn from_parts(
        bytes: Vec<u8>,
        multimodal_projector: Option<Vec<u8>>,
        inference_type: Option<String>,
        config: EngineConfig,
    ) -> Result<Arc<Self>, FfiError> {
        let parts = cera::ModelBytes {
            model: bytes.into(),
            multimodal_projector: multimodal_projector.map(Into::into),
            audio_decoder: None,
            audio_tokenizer: None,
            draft_model: None,
            // `parse_str` maps anything unrecognized to `Unknown(s)`,
            // which `from_parts` rejects by name, better than silently
            // falling back to text when a caller fat-fingers the string.
            inference_type: inference_type
                .as_deref()
                .map(cera::manifest::InferenceType::parse_str),
            chat_template: None,
            generation_defaults: None,
        };
        let inner = cera::CeraEngine::from_parts(parts, config.try_into()?)?;
        Ok(Arc::new(Self { inner }))
    }

    /// Load a model by LeapBundles ID + quantization selector, e.g.
    /// `from_bundle_id("LFM2-1.2B-GGUF", "Q4_0", config)`. Resolves
    /// to the matching `<bundle_id>/<quant>.json` manifest under
    /// `huggingface.co/LiquidAI/LeapBundles` and downloads whatever
    /// isn't already in `config.bundle_repo`'s on-disk cache.
    ///
    /// `config.bundle_repo` must be set; otherwise this returns an
    /// [`FfiError::Backend`] telling the caller to construct a
    /// [`BundleRepo`] and attach it. Idempotent across calls — the
    /// repo's cache deduplicates subsequent downloads.
    ///
    /// Blocking: this call fetches over the network on first run +
    /// opens / parses the GGUF. Foreign async runtimes should wrap
    /// the call in `spawn_blocking` / its equivalent. (An async
    /// counterpart matching `generate_async` could be added later;
    /// not in this PR.)
    #[uniffi::constructor]
    pub fn from_bundle_id(
        bundle_id: String,
        quant: String,
        config: EngineConfig,
    ) -> Result<Arc<Self>, FfiError> {
        let inner = cera::CeraEngine::from_bundle_id(&bundle_id, &quant, config.try_into()?)?;
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

    /// Transcribe mono `f32` PCM audio (normalized to roughly `[-1.0, 1.0]`) to text using the
    /// model's trained `"Perform ASR."` chat mode. `sample_rate` must match the audio encoder's
    /// expected rate (resample beforehand if needed). Requires an audio-capable bundle; a text-only
    /// model returns an [`FfiError`] for unsupported modality.
    ///
    /// Blocking: runs a full prefill + greedy decode. Foreign async runtimes should wrap the call in
    /// `spawn_blocking` / its equivalent.
    pub fn transcribe(&self, pcm: Vec<f32>, sample_rate: u32) -> Result<String, FfiError> {
        Ok(self.inner.transcribe(&pcm, sample_rate)?)
    }

    /// Resolved context-window size (KV cache cap) the engine was
    /// configured with. Mirrors the `context_size` field of the
    /// [`EngineConfig`] passed to `from_path` / `from_bundle_id`,
    /// with the `0` → `model.max_seq_len` defaulting already
    /// applied so callers always see a meaningful number rather
    /// than the internal `usize::MAX` sentinel.
    ///
    /// Note this is the **engine-level** requested cap, not a
    /// per-session ceiling. cera core clamps the model's
    /// `max_seq_len` at load time to `min(requested_context,
    /// gguf_max_seq_len)` (see `cera/src/model/lfm2.rs`), so
    /// [`Self::metadata`]`.max_seq_len` is already the effective
    /// ceiling for any session built from this engine — `context_size`
    /// is informational ("what cap did this engine load with?")
    /// rather than a value callers should `min(...)` against.
    pub fn context_size(&self) -> u64 {
        let cs = self.inner.config().context_size;
        // EngineConfig::try_from maps a `0` request to `usize::MAX`
        // as a "use the model's own max" sentinel; resolve it back
        // to a real number for FFI consumers so they don't see a
        // 18-quintillion-token cap.
        if cs == usize::MAX {
            self.inner.metadata().max_seq_len as u64
        } else {
            cs as u64
        }
    }

    /// Returns default `GenerateOpts` for this engine, pre-populated with
    /// advisory sampling defaults from the bundle manifest (if any) or standard defaults.
    pub fn default_generate_opts(&self) -> GenerateOpts {
        GenerateOpts::from(&self.inner.default_generate_opts())
    }

    // ----- Tokenizer surface (PR 13) ---------------------------------
    //
    // Wraps `cera::tokenizer::BpeTokenizer` so foreign callers can
    // tokenize / detokenize / introspect the model's vocab without
    // going through `Session::append_text`. Useful for: pre-counting
    // prompt tokens before deciding to start a session, manual prompt
    // construction with explicit special tokens, decoding token IDs
    // returned from `generate` for incremental UI display.
    //
    // Tokenizer is shared across all sessions opened from this engine
    // (via Arc internally), so calling these methods concurrently with
    // a `generate*` is safe — they only read.

    /// Encode `text` into token IDs using the model's BPE tokenizer.
    /// Empty input returns an empty vec.
    pub fn encode_text(&self, text: String) -> Vec<u32> {
        self.inner.tokenizer().encode(&text)
    }

    /// Encode `text` with optional special markers — the analog of llama.cpp's
    /// `llama_tokenize(..., add_special)`. When `add_special` is true, BOS is
    /// prepended iff the GGUF declares `tokenizer.ggml.add_bos_token` and EOS
    /// appended iff it declares `tokenizer.ggml.add_eos_token`, so token counts
    /// match llama.cpp for the same text (benchmark parity). With
    /// `add_special = false` this is exactly [`Self::encode_text`]. Prefer this
    /// over hand-prepending BOS via [`ModelMetadata::add_bos_token`].
    pub fn encode_text_special(&self, text: String, add_special: bool) -> Vec<u32> {
        self.inner.tokenizer().encode_special(&text, add_special)
    }

    /// Decode token IDs back to text. Out-of-vocab IDs are silently
    /// skipped (omitted from the decoded output) — `BpeTokenizer::decode`
    /// only appends bytes for IDs it has in `vocab.get(id)`. No
    /// substitution glyph, no error. Callers that want to detect
    /// invalid IDs should validate against `vocab_size()` first.
    pub fn decode_tokens(&self, tokens: Vec<u32>) -> String {
        self.inner.tokenizer().decode(&tokens)
    }

    /// Total vocabulary size — the number of distinct token IDs the
    /// model can emit. Sourced from the model's config (matches
    /// [`ModelMetadata::vocab_size`]) rather than the tokenizer's
    /// own count: in healthy models they match, but the model's
    /// config is the authoritative range for valid logit indices.
    pub fn vocab_size(&self) -> u32 {
        self.inner.metadata().vocab_size
    }

    /// Beginning-of-sequence token ID, if the model has one.
    /// LLaMA-family models typically do; some don't. Honor
    /// [`ModelMetadata::add_bos_token`] when deciding whether to
    /// prepend it manually to a prompt.
    pub fn bos_token(&self) -> Option<u32> {
        self.inner.tokenizer().bos_token()
    }

    /// End-of-sequence / end-of-text token ID, if the model has one.
    /// Used as a default stop-token by the sampler; callers can also
    /// pass it explicitly in [`GenerateOpts::stop_tokens`].
    pub fn eos_token(&self) -> Option<u32> {
        self.inner.tokenizer().eos_token()
    }

    /// Look up a special token by name (e.g. `<|im_start|>`,
    /// `<|im_end|>`, `<|tool_call|>`). Returns `None` if the token
    /// isn't defined in the tokenizer's vocab.
    pub fn special_token_id(&self, name: String) -> Option<u32> {
        self.inner.tokenizer().special_token_id(&name)
    }

    /// `true` when `id` is registered as a control or user-defined
    /// special token in the model's GGUF metadata
    /// (`tokenizer.ggml.token_type` types `3` / `4`). Useful for
    /// output filtering — e.g. dropping `<|im_end|>` from streamed
    /// tokens before rendering them to a UI — and for token-class
    /// classification in analysis tools.
    ///
    /// Out-of-range IDs (>= vocab size) and regular vocab tokens
    /// both return `false`. Companion to [`Self::special_token_id`]
    /// which goes the other direction (name → ID).
    pub fn is_special_token(&self, id: u32) -> bool {
        self.inner.tokenizer().is_special_token(id)
    }

    /// `true` if the model's tokenizer carries a chat template (a
    /// minijinja string from GGUF metadata). Foreign callers should
    /// check this before calling [`CeraEngine::apply_chat_template`].
    pub fn has_chat_template(&self) -> bool {
        self.inner.tokenizer().chat_template().is_some()
    }

    /// Render the model's chat template against a sequence of
    /// `ChatMessage`s. `add_generation_prompt = true` appends the
    /// model's "now it's the assistant's turn" suffix (typical when
    /// driving an interactive chat); `false` produces a transcript
    /// the model can keep continuing.
    ///
    /// Returns [`FfiError::Backend`] if the model has no chat
    /// template (check [`CeraEngine::has_chat_template`] first) or
    /// if the template fails to render against the supplied messages.
    pub fn apply_chat_template(
        &self,
        messages: Vec<ChatMessage>,
        add_generation_prompt: bool,
    ) -> Result<String, FfiError> {
        let core_messages: Vec<cera::tokenizer::ChatMessage> =
            messages.into_iter().map(Into::into).collect();
        cera::tokenizer::apply_chat_template(
            self.inner.tokenizer(),
            &core_messages,
            add_generation_prompt,
        )
        .map_err(|e| FfiError::Backend {
            detail: format!("apply_chat_template: {e}"),
        })
    }

    /// Like [`CeraEngine::apply_chat_template`], but also passes a `tools`
    /// array so a tool-trained model renders its tool-definition block. Pass an
    /// empty `tools` for identical behavior to the plain call.
    pub fn apply_chat_template_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Vec<ToolDef>,
        add_generation_prompt: bool,
    ) -> Result<String, FfiError> {
        let core_messages: Vec<cera::tokenizer::ChatMessage> =
            messages.into_iter().map(Into::into).collect();
        let core_tools = to_core_tools(tools)?;
        cera::tokenizer::apply_chat_template_with_tools(
            self.inner.tokenizer(),
            &core_messages,
            &core_tools,
            add_generation_prompt,
        )
        .map_err(|e| FfiError::Backend {
            detail: format!("apply_chat_template_with_tools: {e}"),
        })
    }

    /// The tool-call format auto-detected from this model's architecture, or
    /// `None` if the architecture has no known tool convention.
    pub fn tool_format(&self) -> Option<ToolFormat> {
        cera::tools::ToolFormat::detect(&self.inner.model().config().architecture).map(Into::into)
    }

    /// The token id of `format`'s tool-call start marker (e.g.
    /// `<|tool_call_start|>`) in this model's vocab, for use as a lazy grammar
    /// trigger in `GenerateOpts.grammar_trigger_tokens`. `None` if the model's
    /// tokenizer lacks that special token.
    pub fn tool_call_start_token(&self, format: ToolFormat) -> Option<u32> {
        let fmt: cera::tools::ToolFormat = format.into();
        self.inner
            .tokenizer()
            .special_token_id(fmt.call_start_marker())
    }

    /// Clear the engine's in-memory warm KV prefix cache, preserving on-disk cold cache.
    /// Call this from host OS memory pressure warnings (e.g. iOS `applicationDidReceiveMemoryWarning`
    /// or Android `onTrimMemory`) to immediately free RAM without losing persistent cached prefixes.
    pub fn clear_prefix_cache(&self) {
        self.inner.clear_warm_cache();
    }

    /// Wipe all KV prefix caches (both in-memory RAM tier and on-disk files).
    pub fn wipe_all_prefix_caches(&self) {
        self.inner.clear_cache();
    }

    /// Detect PII entity spans in text using the loaded token classification model.
    pub fn detect_pii(&self, text: String) -> Result<Vec<FfiEntitySpan>, FfiError> {
        let spans = self.inner.detect_pii(&text)?;
        Ok(spans.into_iter().map(Into::into).collect())
    }
}

// ---------------------------------------------------------------------------
// Session types (PR 3)
// ---------------------------------------------------------------------------

/// KV-cache compression mode. Mirrors [`cera::kv_cache::KvCompression`].
/// `TurboQuant` is honored by the CPU backend and by both GPU backends (wgpu
/// and native Metal). The GPU paths implement the both-sides mode only: a
/// single-sided (debug) request, or a `head_dim` their kernels can't handle,
/// warns and falls back to that backend's uncompressed KV (f32 on wgpu, f16 on
/// Metal). `F16` is honored by the CPU backend only.
#[derive(Debug, Clone, Default, uniffi::Enum)]
pub enum KvCompression {
    /// No compression — the backend's uncompressed KV: f32 on CPU and wgpu,
    /// f16 on native Metal, whose cache has always been half precision.
    #[default]
    None,
    /// f16 KV cache — half-precision keys + values (2 bytes/elem), ~2× less KV
    /// bandwidth at decode-at-depth. Near-lossless. CPU LFM2 and
    /// dense-transformer paths.
    F16,
    /// TurboQuant compression. Both `keys` + `values` true is the
    /// production configuration; toggling them individually is
    /// primarily for debugging the drift contribution of each side.
    /// `seed` drives the per-layer randomized Hadamard rotations.
    TurboQuant { seed: u64, keys: bool, values: bool },
}

impl From<KvCompression> for cera::kv_cache::KvCompression {
    fn from(c: KvCompression) -> Self {
        match c {
            KvCompression::None => cera::kv_cache::KvCompression::None,
            KvCompression::F16 => cera::kv_cache::KvCompression::F16,
            KvCompression::TurboQuant { seed, keys, values } => {
                cera::kv_cache::KvCompression::TurboQuant { seed, keys, values }
            }
        }
    }
}

impl From<cera::kv_cache::KvCompression> for KvCompression {
    fn from(c: cera::kv_cache::KvCompression) -> Self {
        match c {
            cera::kv_cache::KvCompression::None => KvCompression::None,
            cera::kv_cache::KvCompression::F16 => KvCompression::F16,
            cera::kv_cache::KvCompression::TurboQuant { seed, keys, values } => {
                KvCompression::TurboQuant { seed, keys, values }
            }
        }
    }
}

/// Per-session configuration. Mirrors [`cera::SessionConfig`].
#[derive(Debug, Clone, uniffi::Record)]
pub struct SessionConfig {
    /// Cap on total tokens held in KV. `None` → model's default
    /// `max_seq_len`.
    #[uniffi(default = None)]
    pub max_seq_len: Option<u32>,
    /// KV cache compression mode. `None` → no compression (the default).
    #[uniffi(default = None)]
    pub kv_compression: Option<KvCompression>,
    /// Pinned-prefix length for Phase-1.5 context shift on overflow.
    /// `0` disables shift; overflow returns `ContextOverflow` error.
    #[uniffi(default = 0)]
    pub n_keep: u32,
    /// Deterministic sampling seed. `None` = fresh entropy per call.
    #[uniffi(default = None)]
    pub seed: Option<u64>,
    /// Chunked-prefill ubatch size. `0` = monolithic prefill.
    #[uniffi(default = 512)]
    pub ubatch_size: u32,
    /// Whether to prefer GPU depthformer for audio decoder generation.
    #[uniffi(default = false)]
    pub gpu_depthformer: bool,
}

impl Default for SessionConfig {
    fn default() -> Self {
        // Delegate to `cera::SessionConfig::default()` so the defaults
        // stay in one place; `kv_compression` flows through the
        // wrapper's `From` impl (both directions live above).
        let core = cera::SessionConfig::default();
        Self {
            max_seq_len: core.max_seq_len,
            // `None` == "use the default (no) compression" — matches the
            // foreign-binding `#[uniffi(default = None)]` and the `From` mapping,
            // so the Rust `Default` and the no-arg `SessionConfig()` agree.
            kv_compression: None,
            n_keep: core.n_keep,
            seed: core.seed,
            ubatch_size: core.ubatch_size,
            gpu_depthformer: core.gpu_depthformer,
        }
    }
}

impl From<SessionConfig> for cera::SessionConfig {
    fn from(c: SessionConfig) -> Self {
        cera::SessionConfig {
            max_seq_len: c.max_seq_len,
            // `None` → no compression (the core default), so an omitted
            // `kv_compression` behaves like `KvCompression::None`.
            kv_compression: c.kv_compression.map(Into::into).unwrap_or_default(),
            n_keep: c.n_keep,
            seed: c.seed,
            ubatch_size: c.ubatch_size,
            gpu_depthformer: c.gpu_depthformer,
        }
    }
}

/// Speculative decoding configuration for prompt-lookup drafting. Mirrors [`cera::SpecDecode`].
///
/// Prompt-lookup drafting matches trailing n-grams in history to propose draft candidates
/// and verifies them in a single batched prefill step. This is optimal for memory-bandwidth-bound
/// CPU inference; on high-throughput GPU backends, verification overhead may reduce net speedup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Record)]
pub struct SpecDecodeConfig {
    /// Length of the trailing n-gram matched to locate a draft. Defaults to 2.
    #[uniffi(default = 2)]
    pub ngram: u32,
    /// Maximum draft length verified per round (speculation depth). Defaults to 6.
    #[uniffi(default = 6)]
    pub k: u32,
}

impl Default for SpecDecodeConfig {
    fn default() -> Self {
        Self { ngram: 2, k: 6 }
    }
}

impl From<cera::SpecDecode> for SpecDecodeConfig {
    fn from(sd: cera::SpecDecode) -> Self {
        Self {
            ngram: sd.ngram as u32,
            k: sd.k as u32,
        }
    }
}

impl From<SpecDecodeConfig> for cera::SpecDecode {
    fn from(c: SpecDecodeConfig) -> Self {
        Self {
            ngram: (c.ngram as usize).clamp(1, 32),
            k: (c.k as usize).clamp(1, 64),
        }
    }
}

/// Per-call decode options. Mirrors [`cera::GenerateOpts`].
///
/// `flush_every_tokens` / `flush_every_ms` are accepted but have no
/// effect under the synchronous [`Session::generate`]; they are
/// meaningful once streaming (foreign-trait `ModalitySink`) lands
/// in a follow-up PR. Including them in the record now keeps the FFI
/// surface stable across that transition.
#[derive(Debug, Clone, uniffi::Record)]
pub struct GenerateOpts {
    #[uniffi(default = 256)]
    pub max_tokens: u32,
    #[uniffi(default = 0.7)]
    pub temperature: f32,
    #[uniffi(default = 0.9)]
    pub top_p: f32,
    #[uniffi(default = 40)]
    pub top_k: u32,
    /// Min-p (relative) nucleus cutoff: drop tokens below `min_p * p_max`. `0.0`
    /// disables it. Honored in the stochastic path.
    #[uniffi(default = 0.05)]
    pub min_p: f32,
    /// Repetition penalty over tokens generated this call. `1.0` disables it.
    /// Honored in the stochastic path (greedy/argmax decoding is unaffected).
    #[uniffi(default = 1.1)]
    pub repetition_penalty: f32,
    /// Early-stop IDs (EOS / instruction markers / end-of-turn).
    #[uniffi(default = [])]
    pub stop_tokens: Vec<u32>,
    /// Ignore end-of-generation: EOS and `stop_tokens` are not honored, so
    /// decode always runs to `max_tokens`. For benchmark loops that must
    /// cover an exact token count.
    #[uniffi(default = false)]
    pub ignore_eos: bool,
    /// Optional GBNF grammar **source text** constraining the output (e.g. a
    /// JSON grammar). When absent (the default), decoding is unconstrained. The
    /// grammar is compiled on the Rust side when generation starts; a malformed
    /// grammar is reported as a `GrammarParse` error.
    #[uniffi(default = None)]
    pub grammar: Option<String>,
    /// Lazy-grammar trigger token ids (tool calling). When non-empty and
    /// `grammar` is set, the grammar stays inactive until the model emits one
    /// of these tokens (e.g. the tool-call start marker from
    /// [`CeraEngine::tool_call_start_token`]), then constrains the call and
    /// deactivates on completion. Empty -> `grammar` is active from the start.
    #[uniffi(default = [])]
    pub grammar_trigger_tokens: Vec<u32>,
    /// Ignored under synchronous generate; reserved for streaming.
    #[uniffi(default = 16)]
    pub flush_every_tokens: u32,
    /// Ignored under synchronous generate; reserved for streaming.
    #[uniffi(default = 50)]
    pub flush_every_ms: u32,
    /// Optional speculative decoding configuration (prompt-lookup drafting).
    /// When set, runs prompt-lookup speculative drafting to accelerate greedy decoding.
    #[uniffi(default = None)]
    pub spec: Option<SpecDecodeConfig>,
}

impl From<&cera::GenerateOpts> for GenerateOpts {
    fn from(core: &cera::GenerateOpts) -> Self {
        Self {
            max_tokens: core.max_tokens,
            temperature: core.temperature,
            top_p: core.top_p,
            top_k: core.top_k,
            min_p: core.min_p,
            repetition_penalty: core.repetition_penalty,
            stop_tokens: core.stop_tokens.clone(),
            ignore_eos: core.ignore_eos,
            // Core default is no grammar; the compiled `Arc` has no FFI form, so
            // the mirrored field is the (absent) source string.
            grammar: None,
            grammar_trigger_tokens: core.grammar_trigger_tokens.clone(),
            flush_every_tokens: core.flush_every_tokens,
            flush_every_ms: core.flush_every_ms,
            spec: core.spec.map(SpecDecodeConfig::from),
        }
    }
}

impl From<cera::GenerateOpts> for GenerateOpts {
    fn from(core: cera::GenerateOpts) -> Self {
        GenerateOpts::from(&core)
    }
}

impl Default for GenerateOpts {
    fn default() -> Self {
        GenerateOpts::from(&cera::GenerateOpts::default())
    }
}

impl TryFrom<GenerateOpts> for cera::GenerateOpts {
    type Error = FfiError;

    /// Fallible because the GBNF `grammar` source is compiled here: a malformed
    /// grammar becomes [`FfiError::GrammarParse`] rather than silently decoding
    /// unconstrained.
    fn try_from(o: GenerateOpts) -> Result<Self, FfiError> {
        let grammar = match o.grammar {
            Some(src) => Some(Arc::new(cera::grammar::Grammar::parse(&src).map_err(
                |e| FfiError::GrammarParse {
                    detail: format!("{e:#}"),
                },
            )?)),
            None => None,
        };
        Ok(cera::GenerateOpts {
            max_tokens: o.max_tokens,
            temperature: o.temperature,
            top_p: o.top_p,
            top_k: o.top_k,
            min_p: o.min_p,
            repetition_penalty: o.repetition_penalty,
            stop_tokens: o.stop_tokens,
            ignore_eos: o.ignore_eos,
            grammar,
            grammar_trigger_tokens: o.grammar_trigger_tokens,
            flush_every_tokens: o.flush_every_tokens,
            flush_every_ms: o.flush_every_ms,
            spec: o.spec.map(cera::SpecDecode::from),
        })
    }
}

/// Why a decode loop exited. Mirrors [`cera::FinishReason`].
#[derive(Debug, Clone, uniffi::Enum)]
pub enum FinishReason {
    MaxTokens,
    Stop,
    Cancelled,
    ContextFull,
    /// A grammar constraint left no token allowed at this step — decoding
    /// stopped because the grammar dead-ended. Only reachable when
    /// `GenerateOpts.grammar` is set.
    GrammarDeadEnd,
    Error {
        message: String,
    },
}

impl From<cera::FinishReason> for FinishReason {
    fn from(r: cera::FinishReason) -> Self {
        match r {
            cera::FinishReason::MaxTokens => FinishReason::MaxTokens,
            cera::FinishReason::Stop => FinishReason::Stop,
            cera::FinishReason::Cancelled => FinishReason::Cancelled,
            cera::FinishReason::ContextFull => FinishReason::ContextFull,
            cera::FinishReason::GrammarDeadEnd => FinishReason::GrammarDeadEnd,
            cera::FinishReason::Error(msg) => FinishReason::Error { message: msg },
        }
    }
}

/// Decode-run metadata. Mirrors [`cera::GenerateSummary`].
#[derive(Debug, Clone, uniffi::Record)]
pub struct GenerateSummary {
    pub tokens_generated: u32,
    pub prompt_eval_tokens: u32,
    pub prompt_eval_ms: u32,
    pub decode_ms: u32,
    pub total_duration_ms: u32,
    pub decode_tok_per_sec: f64,
    pub prompt_eval_tok_per_sec: f64,
    pub finish_reason: FinishReason,
}

impl From<cera::GenerateSummary> for GenerateSummary {
    fn from(s: cera::GenerateSummary) -> Self {
        let decode_tok_per_sec = s.decode_tok_per_sec();
        let prompt_eval_tok_per_sec = s.prompt_eval_tok_per_sec();
        let total_duration_ms = s.total_duration_ms();
        Self {
            tokens_generated: s.tokens_generated,
            prompt_eval_tokens: s.prompt_eval_tokens,
            prompt_eval_ms: s.prompt_eval_ms,
            decode_ms: s.decode_ms,
            total_duration_ms,
            decode_tok_per_sec,
            prompt_eval_tok_per_sec,
            finish_reason: s.finish_reason.into(),
        }
    }
}

/// Bundle of everything a synchronous `generate` call produces:
/// the generated text string, token IDs, and decode summary.
#[derive(Debug, Clone, uniffi::Record)]
pub struct GenerateOutput {
    /// Generated text (UTF-8 decoded).
    pub text: String,
    /// Generated token IDs, in order, not including any prompt tokens.
    pub tokens: Vec<u32>,
    pub summary: GenerateSummary,
}

/// PCM audio input buffer.
#[derive(Debug, Clone, uniffi::Record)]
pub struct AudioInput {
    pub pcm: Vec<f32>,
    #[uniffi(default = 16000)]
    pub sample_rate: u32,
}

impl From<AudioInput> for cera::tokenizer::AudioInput {
    fn from(a: AudioInput) -> Self {
        Self {
            pcm: a.pcm,
            sample_rate: a.sample_rate,
        }
    }
}

/// User-facing multimodal input envelope.
#[derive(Debug, Clone, Default, uniffi::Record)]
pub struct UserMessage {
    #[uniffi(default = None)]
    pub text: Option<String>,
    #[uniffi(default = [])]
    pub images: Vec<Vec<u8>>,
    #[uniffi(default = None)]
    pub audio: Option<AudioInput>,
}

impl From<UserMessage> for cera::tokenizer::UserMessage {
    fn from(m: UserMessage) -> Self {
        Self {
            text: m.text,
            images: m.images,
            audio: m.audio.map(Into::into),
        }
    }
}

// ---------------------------------------------------------------------------
// ModalitySink (foreign trait)
// ---------------------------------------------------------------------------

/// Streaming sink for decode output. Foreign callers implement this
/// trait (Kotlin class, Swift class, Python subclass) and pass an
/// `Arc<dyn ModalitySink>` to [`Session::generate_streaming`] to
/// receive text chunks + thought chunks + audio frames + the finish reason
/// as they happen.
///
/// All methods are required from foreign implementations (UniFFI 0.31
/// foreign traits don't carry Rust's default-impl fallbacks). Callers
/// that don't care about a modality can provide an empty body.
///
/// Threading: every method is invoked on the same Rust thread running
/// `generate` (the decode thread). If the foreign runtime requires
/// marshalling onto a different thread (e.g. Swift's `@MainActor`) it
/// is the implementer's responsibility to dispatch the call there.
#[uniffi::export(with_foreign)]
pub trait ModalitySink: Send + Sync {
    /// Called with each chunk of reasoning or chain-of-thought text
    /// extracted from thinking delimiters (<think>...</think>).
    fn on_thought_chunk(&self, text: String);

    /// Called with each chunk of generated user-facing text
    /// as soon as valid characters are produced.
    fn on_text_chunk(&self, text: String);

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

const OPEN_THOUGHT_TAGS: &[&str] = &["<think>", "<thought>", "<|thought_start|>"];
const CLOSE_THOUGHT_TAGS: &[&str] = &["</think>", "</thought>", "<|thought_end|>"];

fn find_thought_tag<'a>(text: &str, tags: &[&'a str]) -> Option<(usize, &'a str)> {
    tags.iter()
        .filter_map(|&tag| text.find(tag).map(|pos| (pos, tag)))
        .min_by_key(|&(pos, _)| pos)
}

fn thought_partial_suffix_len(text: &str, tags: &[&str]) -> usize {
    let max_tag_len = tags.iter().map(|t| t.len()).max().unwrap_or(0);
    let max_check = (max_tag_len.saturating_sub(1)).min(text.len());
    for len in (1..=max_check).rev() {
        let start = text.len() - len;
        if text.is_char_boundary(start) {
            let suffix = &text[start..];
            if tags.iter().any(|t| t.starts_with(suffix)) {
                return len;
            }
        }
    }
    0
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ThinkingState {
    Content,
    Thinking,
}

struct StreamingThinkingParser {
    state: ThinkingState,
    buffer: String,
}

impl StreamingThinkingParser {
    fn new() -> Self {
        Self {
            state: ThinkingState::Content,
            buffer: String::new(),
        }
    }

    fn feed(&mut self, chunk: &str) -> Vec<(bool, String)> {
        self.buffer.push_str(chunk);
        let mut emissions = Vec::new();

        loop {
            let (tags, is_thought, next_state) = match self.state {
                ThinkingState::Content => (OPEN_THOUGHT_TAGS, false, ThinkingState::Thinking),
                ThinkingState::Thinking => (CLOSE_THOUGHT_TAGS, true, ThinkingState::Content),
            };

            if let Some((pos, tag)) = find_thought_tag(&self.buffer, tags) {
                if pos > 0 {
                    let text = self.buffer[..pos].to_string();
                    emissions.push((is_thought, text));
                }
                let tag_len = tag.len();
                self.buffer.drain(..pos + tag_len);
                self.state = next_state;
                continue;
            }

            let hold = thought_partial_suffix_len(&self.buffer, tags);
            let safe_len = self.buffer.len() - hold;
            if safe_len > 0 {
                let text = self.buffer[..safe_len].to_string();
                self.buffer.drain(..safe_len);
                emissions.push((is_thought, text));
            }
            break;
        }

        emissions
    }

    fn flush(&mut self) -> Option<(bool, String)> {
        if self.buffer.is_empty() {
            None
        } else {
            let is_thought = self.state == ThinkingState::Thinking;
            let text = std::mem::take(&mut self.buffer);
            Some((is_thought, text))
        }
    }
}

/// Adapter from the UniFFI foreign trait to the internal
/// [`cera::ModalitySink`]. Decodes tokens to valid UTF-8 text chunks
/// incrementally using the session's tokenizer, and forwards audio frames
/// and terminal completion events.
struct ForeignSinkAdapter {
    inner: Arc<dyn ModalitySink>,
    tokenizer: Arc<cera::tokenizer::BpeTokenizer>,
    pending_bytes: Vec<u8>,
    parser: StreamingThinkingParser,
    done_called: bool,
    done_reason: Option<FinishReason>,
}

impl ForeignSinkAdapter {
    fn new(inner: Arc<dyn ModalitySink>, tokenizer: Arc<cera::tokenizer::BpeTokenizer>) -> Self {
        Self {
            inner,
            tokenizer,
            pending_bytes: Vec::new(),
            parser: StreamingThinkingParser::new(),
            done_called: false,
            done_reason: None,
        }
    }

    fn emit_chunk(&self, is_thought: bool, text: String) {
        if !text.is_empty() {
            if is_thought {
                self.inner.on_thought_chunk(text);
            } else {
                self.inner.on_text_chunk(text);
            }
        }
    }

    fn flush_pending(&mut self) {
        if !self.pending_bytes.is_empty() {
            let piece = String::from_utf8_lossy(&self.pending_bytes);
            if !piece.is_empty() {
                for (is_thought, text) in self.parser.feed(&piece) {
                    self.emit_chunk(is_thought, text);
                }
            }
            self.pending_bytes.clear();
        }
        if let Some((is_thought, text)) = self.parser.flush() {
            self.emit_chunk(is_thought, text);
        }
    }

    fn notify_done(&mut self, fallback: Option<FinishReason>) {
        if let Some(reason) = self.done_reason.take() {
            self.inner.on_done(reason);
        } else if let Some(fallback_reason) = fallback {
            self.inner.on_done(fallback_reason);
        }
    }
}

impl cera::ModalitySink for ForeignSinkAdapter {
    fn on_text_tokens(&mut self, tokens: &[u32]) {
        self.pending_bytes
            .extend_from_slice(&self.tokenizer.decode_bytes(tokens));
        loop {
            if self.pending_bytes.is_empty() {
                break;
            }
            match std::str::from_utf8(&self.pending_bytes) {
                Ok(s) => {
                    for (is_thought, text) in self.parser.feed(s) {
                        self.emit_chunk(is_thought, text);
                    }
                    self.pending_bytes.clear();
                    break;
                }
                Err(e) => {
                    let valid_len = e.valid_up_to();
                    if valid_len > 0 {
                        if let Ok(piece) = std::str::from_utf8(&self.pending_bytes[..valid_len]) {
                            for (is_thought, text) in self.parser.feed(piece) {
                                self.emit_chunk(is_thought, text);
                            }
                        }
                        self.pending_bytes.drain(..valid_len);
                    }
                    if let Some(err_len) = e.error_len() {
                        for (is_thought, text) in self.parser.feed("\u{FFFD}") {
                            self.emit_chunk(is_thought, text);
                        }
                        self.pending_bytes.drain(..err_len);
                    } else {
                        // Incomplete multi-byte sequence at end of buffer, wait for more tokens.
                        break;
                    }
                }
            }
        }
    }
    fn on_audio_frames(&mut self, pcm: &[f32], sample_rate: u32) {
        self.inner.on_audio_frames(pcm.to_vec(), sample_rate);
    }
    fn on_done(&mut self, reason: cera::FinishReason) {
        self.flush_pending();
        self.done_called = true;
        self.done_reason = Some(reason.into());
    }
}

// ---------------------------------------------------------------------------
// LoRA adapters
// ---------------------------------------------------------------------------

/// A loaded LoRA adapter, ready to attach to a [`Session`] via
/// [`Session::attach_lora`]. Load it once and share the handle across sessions —
/// it's reference-counted internally, so attaching to multiple sessions doesn't
/// re-parse or re-allocate the factors.
#[derive(uniffi::Object)]
pub struct LoraAdapters {
    inner: Arc<cera::lora::LoraAdapterWeights>,
}

#[uniffi::export]
impl LoraAdapters {
    /// Load a llama.cpp-format GGUF adapter (`convert_lora_to_gguf` output) from
    /// a local path. `alpha` is read from the adapter's `adapter.lora.alpha`
    /// metadata (missing ⇒ scale = 1).
    #[uniffi::constructor]
    pub fn from_gguf(path: String) -> Result<Arc<Self>, FfiError> {
        let inner = cera::lora::LoraAdapterWeights::from_gguf(std::path::Path::new(&path))
            .map_err(|e| FfiError::LoraParse {
                detail: e.to_string(),
            })?;
        Ok(Arc::new(Self { inner }))
    }

    /// Load a PEFT `.safetensors` adapter from a local path. PEFT stores `alpha`
    /// in a sibling `adapter_config.json`, so pass it explicitly here (`None` ⇒
    /// scale = 1, i.e. `alpha == rank`).
    #[uniffi::constructor]
    pub fn from_safetensors(path: String, alpha: Option<f32>) -> Result<Arc<Self>, FfiError> {
        let inner =
            cera::lora::LoraAdapterWeights::from_safetensors(std::path::Path::new(&path), alpha)
                .map_err(|e| FfiError::LoraParse {
                    detail: e.to_string(),
                })?;
        Ok(Arc::new(Self { inner }))
    }

    /// Number of `(layer, target)` low-rank deltas the adapter carries — for
    /// diagnostics / logging.
    pub fn target_count(&self) -> u32 {
        self.inner.target_count() as u32
    }
}

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

/// Stateful inference handle. Wraps [`cera::Session`] behind a
/// `Mutex` so UniFFI's `Arc<Session>` shape works with methods that
/// need `&mut self` on the inner session (prefill, generate, reset).
///
/// Call [`CeraEngine::new_session`] to open a session; the engine's
/// `Arc<Model>` and `Arc<BpeTokenizer>` are cloned into the new
/// session so it outlives the engine handle across FFI calls.
#[derive(uniffi::Object)]
pub struct Session {
    inner: std::sync::Mutex<cera::Session>,
    /// Cloned from the inner session at construction time. Shared
    /// atomic — `position()` / `cancel()` don't need to acquire the
    /// mutex, so they're safe to call from a different thread while
    /// `generate()` is running.
    position: Arc<std::sync::atomic::AtomicU32>,
    cancel: Arc<std::sync::atomic::AtomicBool>,
    /// Stored at construction so `capabilities()` doesn't need a lock.
    capabilities: ModalityCapabilities,
    /// Model hidden dimension, cached at construction so `hidden_size()` is a
    /// lock-free read — safe to call from a `generate_streaming` sink callback
    /// (which runs while `generate` holds the mutex), same as `position()`.
    hidden_size: u32,
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
    fn lock_inner(&self) -> Result<std::sync::MutexGuard<'_, cera::Session>, FfiError> {
        self.inner.lock().map_err(|e| FfiError::Backend {
            detail: format!(
                "session mutex poisoned (a prior call panicked mid-lock; session state is \
                 inconsistent): {e}"
            ),
        })
    }
}

/// Flatten `f32`s into a little-endian byte buffer (`4 * len` bytes) so the wire
/// format is stable across host architectures; callers reinterpret 4-byte groups
/// as `f32`. On little-endian hosts (every UniFFI target in practice) the
/// in-memory `f32` bytes already ARE the LE wire format, so a single bulk copy
/// beats the per-element `to_le_bytes()` loop on the large `[T*D]` payloads.
fn f32_vec_to_le_bytes(v: &[f32]) -> Vec<u8> {
    #[cfg(target_endian = "little")]
    {
        bytemuck::cast_slice::<f32, u8>(v).to_vec()
    }
    #[cfg(target_endian = "big")]
    {
        let mut bytes = Vec::with_capacity(v.len() * 4);
        for &x in v {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        bytes
    }
}

/// Collector sink: captures every token the decode loop emits.
/// Used by synchronous `generate` and `send_message_and_generate`.
struct TokenCollectSink(Vec<u32>);

impl cera::ModalitySink for TokenCollectSink {
    fn on_text_tokens(&mut self, tokens: &[u32]) {
        self.0.extend_from_slice(tokens);
    }
    fn on_done(&mut self, _reason: cera::FinishReason) {}
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

    /// Append PCM audio samples (mono `f32`, normalized to roughly
    /// `[-1.0, 1.0]`) at `sample_rate` Hz. The audio is encoded via
    /// the bundle's mmproj (`AudioEncoderWeights`) and prefilled
    /// into the LLM as soft tokens — see
    /// [`cera::Session::append_audio`] for the underlying flow.
    ///
    /// `CeraEngine::new_session` auto-attaches the encoder when the
    /// loaded bundle's `inference_type == LlamaCppLfm2AudioV1` and
    /// has `multimodal_projector` set in the manifest, so FFI
    /// consumers don't need any separate "load encoder" call.
    /// Bundles where the manifest omits `multimodal_projector`
    /// silently end up with no encoder attached (no log). Bundles
    /// where the file is named but fails to open or parse log a
    /// `tracing::warn!` at `CeraEngine` construction. Both cases
    /// surface here as a "no audio encoder attached" `Backend`
    /// error.
    ///
    /// `sample_rate` must be 16000 — resampling is out of scope.
    /// Callers should resample externally before passing samples in.
    ///
    /// **Marshaling cost**: UniFFI maps `Vec<f32>` to `List<Float>`
    /// in Kotlin and `[Float]` in Swift. The Kotlin side boxes each
    /// `Float` to `java.lang.Float`, a ~4× memory overhead vs the
    /// underlying `f32` wire bytes — negligible for O(seconds ×
    /// sample-rate) chunks but worth knowing if you're streaming
    /// continuous audio in tight loops.
    ///
    /// Errors:
    /// - `EmptyInput` either when `samples` is empty (fast-fail,
    ///   enforced here for parity with `append_text` /
    ///   `append_tokens`) **or** when the audio is too short to
    ///   produce any encoder frames (e.g. shorter than one
    ///   center-padded STFT window).
    /// - `UnsupportedModality` if the loaded model's
    ///   [`ModalityCapabilities::audio_in`] is `false`.
    /// - `Backend(...)` for sample-rate mismatch, encoder/LLM
    ///   `hidden_size` mismatch, or missing encoder. The latter
    ///   includes both "manifest didn't list a mmproj" (no warn
    ///   logged) and "mmproj listed but failed to open/parse"
    ///   (warn logged at `CeraEngine::from_path`).
    /// - `ContextOverflow` / `Cancelled` propagate from the
    ///   underlying prefill.
    pub fn append_audio(&self, samples: Vec<f32>, sample_rate: u32) -> Result<(), FfiError> {
        if samples.is_empty() {
            return Err(FfiError::EmptyInput);
        }
        self.lock_inner()?.append_audio(&samples, sample_rate)?;
        Ok(())
    }

    /// Model hidden dimension `D`. Reshape a raw `[T*D]` byte buffer from
    /// [`Self::hidden_states_for_tokens`] into `[T][D]` with this. Lock-free
    /// (cached at construction), so — like `position()` — it's safe to call from
    /// a `generate_streaming` sink callback.
    pub fn hidden_size(&self) -> u32 {
        self.hidden_size
    }

    /// Per-token last-layer hidden states (post-final-RMSNorm — the llama.cpp
    /// `--pooling none` / `llama_get_embeddings_ith` vector) for `tokens`,
    /// returned as **little-endian f32 bytes**: `n_tokens * hidden_size * 4`
    /// bytes, row-major, token `t` channel `c` at `(t*D + c) * 4`.
    ///
    /// Bytes (UniFFI `Data` in Swift, `ByteArray` in Kotlin) rather than
    /// `List<Float>` to avoid Kotlin's per-element boxing on the potentially
    /// large `[T*D]` payload. Swift decodes via `Data.withUnsafeBytes`; reflects
    /// the active LoRA once that lands. Side-effect-free: does not disturb the
    /// session's generation KV.
    ///
    /// Like `append_*` / `generate`, this holds the session mutex for the
    /// duration of the compute, so it must NOT be called re-entrantly from
    /// within a `generate_streaming` sink callback (would self-deadlock).
    ///
    /// Errors: `EmptyInput` on empty input; `UnsupportedModality` if the backend
    /// doesn't implement hidden-state extraction; `InvalidToken` if any id is
    /// `>= vocab_size`.
    pub fn hidden_states_for_tokens(&self, tokens: Vec<u32>) -> Result<Vec<u8>, FfiError> {
        let hs = self.lock_inner()?.hidden_states_for_tokens(&tokens)?;
        Ok(f32_vec_to_le_bytes(&hs))
    }

    /// Like [`Self::hidden_states_for_tokens`] but tokenizes `text` first
    /// (Swift `hiddenStatesForText(text:)`). Returns the same LE-f32 byte layout.
    pub fn hidden_states_for_text(&self, text: String) -> Result<Vec<u8>, FfiError> {
        let hs = self.lock_inner()?.hidden_states_for_text(&text)?;
        Ok(f32_vec_to_le_bytes(&hs))
    }

    /// Mean-pooled hidden state — a single `[hidden_size]` vector (the common
    /// classifier path: pool in Rust, ship `D` floats not `T*D`). Returned as
    /// `[Float]` / `List<Float>`; only `D` elements, so boxing is negligible.
    pub fn hidden_states_mean_pooled(&self, tokens: Vec<u32>) -> Result<Vec<f32>, FfiError> {
        Ok(self.lock_inner()?.hidden_states_mean_pooled(&tokens)?)
    }

    /// Attach a [`LoraAdapters`] to this session (generated as `attachLora` in
    /// Swift/Kotlin — this is the engine's equivalent of a `setLoraAdapters`
    /// call). It's applied to every subsequent forward pass — generation **and**
    /// hidden-states extraction — until removed or replaced (hot-swap), and is
    /// preserved across [`Self::reset`]. Only affects tokens processed after the
    /// call (doesn't retroactively re-adapt cached KV).
    ///
    /// Two distinct failures, worth catching separately: [`FfiError::LoraParse`]
    /// means the adapter's dimensions don't match the loaded model, so the
    /// adapter or the pairing is wrong; [`FfiError::LoraUnsupportedByBackend`]
    /// means it fits but this backend has no hook for something it adapts, so
    /// the same adapter works on another backend (today: a mixture-of-experts
    /// adapter needs the CPU backend).
    pub fn attach_lora(&self, adapters: Arc<LoraAdapters>) -> Result<(), FfiError> {
        self.lock_inner()?
            .attach_lora_adapters(adapters.inner.clone())?;
        Ok(())
    }

    /// Remove any attached LoRA adapter, returning to base-model inference.
    pub fn remove_lora(&self) -> Result<(), FfiError> {
        self.lock_inner()?.remove_lora_adapters();
        Ok(())
    }

    /// Whether a LoRA adapter is currently attached to this session.
    pub fn has_lora(&self) -> Result<bool, FfiError> {
        Ok(self.lock_inner()?.has_lora_adapters())
    }

    /// Append an encoded image (PNG / JPEG bytes, auto-detected) to the
    /// context. The image is decoded, resized, normalized, and run
    /// through the bundle's vision mmproj (`VisionEncoderWeights`), then
    /// prefilled into the LLM as soft tokens — see
    /// [`cera::Session::append_image`] for the underlying flow.
    ///
    /// `CeraEngine::new_session` auto-attaches the vision encoder when
    /// the loaded bundle's `inference_type` is a VL type with
    /// `multimodal_projector` set in the manifest, so FFI consumers
    /// don't need a separate "load encoder" call. Bundles whose
    /// manifest omits the mmproj end up with no encoder attached (no
    /// log); bundles where it's named but fails to open/parse log a
    /// `tracing::warn!` at `CeraEngine` construction. Both surface here
    /// as a "no vision encoder attached" `Backend` error.
    ///
    /// `max_long_size` controls the per-call cap on the longest side of
    /// the *encoded* image, with three cases distinguished so the
    /// session default stays reachable through FFI:
    /// - `None` — defer to the session default set via
    ///   [`Self::set_image_max_long_size`] (no cap if none was set).
    /// - `Some(0)` — explicitly force *no cap* for this call, ignoring
    ///   the session default.
    /// - `Some(n)` (`n > 0`) — cap this call at `n`, overriding the
    ///   session default.
    ///
    /// When a cap applies, the resize target is shrunk
    /// (aspect-preserving) so its longer side is at most `n` pixels,
    /// floored at one aligned patch block (so a very small `n` can still
    /// round up to that minimum) — a quality/cost knob (smaller = fewer
    /// image tokens, faster, less detail). It only shrinks (never
    /// upscales) and takes precedence over the model's
    /// minimum-resolution floor. The cap bounds the *encode*, not the
    /// *decode* (a huge source image is still decoded, bounded by
    /// internal limits).
    ///
    /// **Placement matters.** Prefer driving multimodal turns through
    /// the chat template; calling this at the wrong stream position
    /// (outside the model's image-marker envelope) leaves the LLM
    /// unable to interpret the embeddings as visual content. See
    /// [`cera::Session::append_image`] for the marker recipe.
    ///
    /// Errors (capability is checked before emptiness, matching core):
    /// - `UnsupportedModality` if the loaded model's
    ///   [`ModalityCapabilities::image_in`] is `false`.
    /// - `EmptyInput` when `bytes` is empty (on a VL session).
    /// - `Backend(...)` for image decode failure, missing vision
    ///   encoder, or encoder/LLM `projection_dim` ≠ `hidden_size`
    ///   mismatch.
    /// - `ContextOverflow` / `Cancelled` propagate from the
    ///   underlying prefill.
    pub fn append_image(&self, bytes: Vec<u8>, max_long_size: Option<u32>) -> Result<(), FfiError> {
        // Delegate to the core methods (rather than always calling
        // `append_image_with_opts`) so the session default stays
        // reachable through FFI and core's capability-before-empty error
        // precedence is preserved: `None` -> session default, `Some(0)`
        // -> force no cap, `Some(n)` -> cap at `n`. The empty-bytes guard
        // lives in core (`preprocess_image_with_opts`), which runs after
        // the capability check, so a non-VL session still reports
        // `UnsupportedModality` rather than `EmptyInput` for empty input.
        let mut inner = self.lock_inner()?;
        match max_long_size {
            None => inner.append_image(&bytes),
            Some(0) => inner.append_image_with_opts(&bytes, None),
            Some(n) => inner.append_image_with_opts(&bytes, Some(n)),
        }?;
        Ok(())
    }

    /// Set a session-default cap on the longest side of an appended
    /// image, in pixels (`None` = no cap). Unlike the per-call
    /// `max_long_size` argument to [`Self::append_image`], this default
    /// is honored by every image-append path the session drives —
    /// including chat-template flows — so a host can configure the
    /// image-encode budget once. See [`Self::append_image`] for the cap
    /// semantics (shrinks the encoded target, never upscales, takes
    /// precedence over the model's minimum-resolution floor).
    pub fn set_image_max_long_size(&self, max_long_size: Option<u32>) -> Result<(), FfiError> {
        self.lock_inner()?.set_image_max_long_size(max_long_size);
        Ok(())
    }

    /// Returns default `GenerateOpts` for this session, pre-populated with
    /// advisory sampling defaults from the bundle manifest (if any) or standard defaults.
    pub fn default_generate_opts(&self) -> Result<GenerateOpts, FfiError> {
        let guard = self.lock_inner()?;
        Ok(GenerateOpts::from(guard.default_generate_opts()))
    }

    /// Run autoregressive decode and return all emitted text, tokens, and
    /// summary. Synchronous: the call blocks until the decode loop exits
    /// (`max_tokens`, EOS, `cancel()`, or error).
    pub fn generate(&self, opts: GenerateOpts) -> Result<GenerateOutput, FfiError> {
        let mut sink = TokenCollectSink(Vec::new());
        // Compile the grammar (if any) before taking the session lock so a
        // malformed GBNF fails fast with `FfiError::GrammarParse`.
        let core: cera::GenerateOpts = opts.try_into()?;
        let mut guard = self.lock_inner()?;
        let summary = guard.generate(&core, &mut sink)?;
        let text = guard.tokenizer().decode(&sink.0);
        Ok(GenerateOutput {
            text,
            tokens: sink.0,
            summary: summary.into(),
        })
    }

    /// Run autoregressive decode, streaming every text chunk (and audio
    /// frame, for audio-capable models) to a foreign [`ModalitySink`]
    /// as soon as it is produced. Returns only a [`GenerateSummary`]:
    /// text chunks are delivered through `sink.on_text_chunk`, not a
    /// return value.
    ///
    /// Synchronous: the call blocks on the decode thread and each
    /// `sink` method runs on that same thread before decoding
    /// continues.
    ///
    /// **Callback reentrancy: deadlock hazard.** The session mutex is
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
        let outcome = match cera::GenerateOpts::try_from(opts) {
            Ok(core) => match self.lock_inner() {
                Ok(mut guard) => {
                    let mut adapter = ForeignSinkAdapter::new(sink, guard.tokenizer_arc());
                    let result = guard.generate(&core, &mut adapter).map_err(FfiError::from);
                    drop(guard);
                    (result, Some(adapter), None)
                }
                Err(e) => (Err(e), None, Some(sink)),
            },
            Err(e) => (Err(e), None, Some(sink)),
        };
        match outcome {
            (Ok(summary), Some(mut adapter), _) => {
                adapter.notify_done(None);
                Ok(summary.into())
            }
            (Err(err), Some(mut adapter), _) => {
                if !adapter.done_called {
                    adapter.flush_pending();
                    let finish_reason = match &err {
                        FfiError::Cancelled => FinishReason::Cancelled,
                        _ => FinishReason::Error {
                            message: err.to_string(),
                        },
                    };
                    adapter.notify_done(Some(finish_reason));
                } else {
                    adapter.notify_done(None);
                }
                Err(err)
            }
            (Err(err), None, Some(inner)) => {
                let finish_reason = match &err {
                    FfiError::Cancelled => FinishReason::Cancelled,
                    _ => FinishReason::Error {
                        message: err.to_string(),
                    },
                };
                inner.on_done(finish_reason);
                Err(err)
            }
            (Err(err), None, None) => Err(err),
            (Ok(_), None, _) => unreachable!(),
        }
    }

    /// Append a multimodal message, automatically enforcing model-canonical
    /// media ordering, boundary token envelopes, and sample rate normalization.
    pub fn send_message(&self, message: UserMessage) -> Result<(), FfiError> {
        let mut guard = self.lock_inner()?;
        let core_msg: cera::tokenizer::UserMessage = message.into();
        guard.append_user_message(&core_msg).map_err(FfiError::from)
    }

    /// Append a multimodal message and run generation synchronously while holding
    /// the session lock continuously across prefill and decode.
    pub fn send_message_and_generate(
        &self,
        message: UserMessage,
        opts: GenerateOpts,
    ) -> Result<GenerateOutput, FfiError> {
        let core_opts: cera::GenerateOpts = opts.try_into()?;
        let core_msg: cera::tokenizer::UserMessage = message.into();
        let mut guard = self.lock_inner()?;
        guard.append_user_message(&core_msg)?;

        let mut sink = TokenCollectSink(Vec::new());
        let summary = guard.generate(&core_opts, &mut sink)?;
        let text = guard.tokenizer().decode(&sink.0);
        Ok(GenerateOutput {
            text,
            tokens: sink.0,
            summary: summary.into(),
        })
    }

    /// Append a multimodal message and run streaming generation while holding
    /// the session lock continuously across prefill and decode.
    pub fn send_message_streaming(
        &self,
        message: UserMessage,
        opts: GenerateOpts,
        sink: Arc<dyn ModalitySink>,
    ) -> Result<GenerateSummary, FfiError> {
        let core_opts: cera::GenerateOpts = match opts.try_into() {
            Ok(o) => o,
            Err(err) => {
                sink.on_done(FinishReason::Error {
                    message: err.to_string(),
                });
                return Err(err);
            }
        };
        let core_msg: cera::tokenizer::UserMessage = message.into();
        let mut guard = match self.lock_inner() {
            Ok(g) => g,
            Err(err) => {
                sink.on_done(FinishReason::Error {
                    message: err.to_string(),
                });
                return Err(err);
            }
        };
        if let Err(err) = guard.append_user_message(&core_msg) {
            drop(guard);
            let ffi_err = FfiError::from(err);
            let finish_reason = match &ffi_err {
                FfiError::Cancelled => FinishReason::Cancelled,
                _ => FinishReason::Error {
                    message: ffi_err.to_string(),
                },
            };
            sink.on_done(finish_reason);
            return Err(ffi_err);
        }

        let mut adapter = ForeignSinkAdapter::new(sink, guard.tokenizer_arc());
        let result = guard
            .generate(&core_opts, &mut adapter)
            .map_err(FfiError::from);
        drop(guard);
        match result {
            Ok(summary) => {
                adapter.notify_done(None);
                Ok(summary.into())
            }
            Err(err) => {
                if !adapter.done_called {
                    adapter.flush_pending();
                    let finish_reason = match &err {
                        FfiError::Cancelled => FinishReason::Cancelled,
                        _ => FinishReason::Error {
                            message: err.to_string(),
                        },
                    };
                    adapter.notify_done(Some(finish_reason));
                } else {
                    adapter.notify_done(None);
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

    /// Clear the cancel flag without dropping any session state.
    /// Use this after observing a cancellation signal — either
    /// [`FfiError::Cancelled`] from `append_text` / `append_tokens`
    /// / `append_audio` (mid-prefill cancellation surfaces
    /// typed), or `finish_reason = "Cancelled"` on the
    /// [`GenerateOutput`] returned from `generate` (cancellation
    /// during decode is reported as an `Ok` with that finish
    /// reason rather than an `Err`) — when you want to resume
    /// work on the same session without losing the accumulated
    /// KV cache.
    ///
    /// Compared to [`Self::reset`]:
    /// - `clear_cancel`: keeps KV state + position + sampler
    ///   intact; only flips the cancel atomic back to `false`.
    ///   Use for "interrupted but continuing" flows.
    /// - `reset`: drops KV cache + position + last logits +
    ///   re-seeds sampler. Use for "clear conversation" flows.
    ///
    /// Atomic-backed; no mutex acquire, infallible, safe from
    /// any thread (mirrors the shape of [`Self::cancel`]).
    pub fn clear_cancel(&self) {
        self.cancel
            .store(false, std::sync::atomic::Ordering::Relaxed);
    }

    /// Drop cached state + resample the seed. After `reset()` the
    /// session behaves like a freshly-opened one (same
    /// model/tokenizer/config, no accumulated context).
    ///
    /// Returns `Result` so a poisoned-mutex case surfaces as an error
    /// instead of panicking across the FFI boundary.
    pub fn reset(&self) -> Result<(), FfiError> {
        self.lock_inner()?.reset()?;
        Ok(())
    }

    /// Capabilities reported by the loaded model. Cheap — reads a
    /// cached copy, no lock.
    pub fn capabilities(&self) -> ModalityCapabilities {
        self.capabilities
    }
}

// ---------------------------------------------------------------------------
// Async Session methods (PR 5)
// ---------------------------------------------------------------------------
//
// Foreign callers driving an async runtime (Kotlin coroutines, Swift
// `async`, Python `asyncio`) can `.await` these without bouncing into
// a sync context. Every method defers to its synchronous twin inside
// `tokio::task::spawn_blocking`, which moves the actual decode work
// onto a blocking worker thread — so the tokio async worker pool
// stays free to poll other futures while decoding is in flight.
//
// `self: Arc<Self>` rather than `&self` so the session handle can
// cross the `spawn_blocking` boundary (requires `'static`). UniFFI
// wraps `#[uniffi::Object]` types in `Arc` on the foreign side anyway,
// so this doesn't change the foreign API shape — it's still
// `session.generateAsync(opts)` on Kotlin / `session.generateAsync(opts)`
// on Swift / `session.generate_async(opts)` on Python.
//
// UniFFI's `tokio` feature starts an internal multi-thread tokio
// runtime the first time a `#[uniffi::export(async_runtime = "tokio")]`
// method is invoked. Spawned blocking tasks inherit that runtime's
// blocking worker pool (`tokio::runtime::Builder::new_multi_thread`
// default). We don't need to create or enter a runtime ourselves.

/// RAII guard that cancels the in-flight `spawn_blocking` decode on
/// future-drop. Addresses a subtle hazard of wrapping sync decode in
/// `tokio::task::spawn_blocking`: dropping the outer future drops the
/// `JoinHandle`, but tokio does **not** abort a `spawn_blocking` task
/// on handle-drop — the blocking worker keeps decoding, keeps holding
/// the session mutex, keeps mutating `Session::state`.
///
/// Without this guard, a foreign-side cancellation (Kotlin coroutine
/// scope exit, Swift `Task.cancel`, Python `asyncio.Task.cancel`) would
/// silently leak decode work into the background. The caller's next
/// `generate*` call would block on the still-held mutex or observe
/// state advanced by the "cancelled" call.
///
/// Two code paths, two mitigations — both fire together because the
/// guard can't know which path applies:
///
/// 1. **Running decode.** The task is executing `cera::Session::generate`
///    on a blocking worker. `session.cancel()` flips the cancel atomic;
///    the decode loop polls it between tokens and exits with
///    `FinishReason::Cancelled`. `JoinHandle::abort` has no effect
///    here — `spawn_blocking` tasks are opaque synchronous code with
///    no await points to interrupt.
///
/// 2. **Queued decode.** The task is in the blocking pool's queue
///    waiting for a worker (pool saturated, or just hasn't been
///    scheduled). `JoinHandle::abort` cancels queued-but-not-started
///    blocking tasks before their closure runs — the closure never
///    executes, so `cera::Session::generate` never starts, so the
///    session's cancel flag is never reset. Without this, the race is:
///    guard sets cancel → task eventually dequeues → decode's first
///    line clears cancel back to `false` (`cera/src/session.rs:603-605`)
///    → decode runs to completion despite the caller having dropped
///    the future.
///
/// Both operations are idempotent / harmless on the irrelevant path:
/// `session.cancel()` on a queued task is overridden by `abort`; an
/// already-completed task ignores both. `cera::Session::generate`
/// resets the cancel atomic on entry, so a spurious late-arriving
/// cancel from a guard that dropped just after the await resolved
/// (not reachable in practice — futures aren't preemptively dropped
/// between synchronous statements) wouldn't affect the next call.
struct AsyncCancelGuard {
    session: Arc<Session>,
    /// Abort handle for the `spawn_blocking` task. Calling `abort()`
    /// on a queued task removes it from the pool's queue; on a running
    /// task it's a no-op (no await point to unwind through). Kept as
    /// an `AbortHandle` rather than a `JoinHandle` so the guard can
    /// coexist with the outer `.await` on the same handle (we take
    /// `abort_handle()` before awaiting).
    abort: tokio::task::AbortHandle,
    /// `true` until the await successfully resolves. Dropping with
    /// `armed = true` means we're being dropped mid-await: fire both
    /// abort (for queued-but-not-started) and cancel (for in-flight).
    armed: bool,
}

impl Drop for AsyncCancelGuard {
    fn drop(&mut self) {
        if self.armed {
            self.abort.abort();
            self.session.cancel();
        }
    }
}

/// Lighter-weight sibling of [`AsyncCancelGuard`] for `spawn_blocking`
/// tasks that don't share mutable state with anything the caller can
/// signal. Reached through `spawn_blocking_guarded`, so it covers the
/// three async [`CeraEngine`] constructors (via `spawn_engine_build`)
/// and `list_leap_bundles_async`: engine construction holds no cross-thread
/// cancel flag, and neither the tokenizer build nor (for
/// `from_bundle_id`) the `reqwest::blocking` download can be
/// cooperatively cancelled, so there's nothing like `Session::cancel`
/// to call on drop. All we can do is abort the queued task before its closure
/// runs — `AbortHandle::abort` (taken from the task's `JoinHandle`
/// via `JoinHandle::abort_handle()` so the guard doesn't fight the
/// outer `.await` for ownership of the handle) on a queued
/// `spawn_blocking` task is effective; on a running one it's a no-op
/// and the download finishes to cache (which is arguably a feature:
/// a dropped future's bandwidth isn't wasted, the next call finds
/// the bundle cached and returns instantly).
///
/// Drop logic is one conditional (`if armed { abort.abort() }`) —
/// structurally identical to [`AsyncCancelGuard`]'s. The
/// `async_cancel_guard_drop_fires_when_armed_only` ProbeGuard test in
/// the test module already exercises that exact branch shape; a
/// duplicate test for AbortOnDrop would add no coverage. End-to-end
/// "abort actually cancels a queued tokio task" is upstream tokio's
/// behavior to test, not ours.
struct AbortOnDrop {
    abort: tokio::task::AbortHandle,
    /// Set to `false` once the outer `.await` resolves; prevents
    /// `abort()` from running on a task that already completed.
    armed: bool,
}

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        if self.armed {
            self.abort.abort();
        }
    }
}

#[uniffi::export(async_runtime = "tokio")]
impl Session {
    /// Async variant of [`Session::generate`] — runs buffered decode
    /// (returning every emitted token + a summary) on a tokio blocking
    /// worker so the caller's async context isn't stalled by the
    /// synchronous decode loop.
    ///
    /// Cancellation: dropping the returned future (Kotlin coroutine
    /// scope exit, Swift `Task.cancel`, Python `asyncio.Task.cancel`)
    /// triggers both an abort of the queued `spawn_blocking` task (so
    /// a not-yet-started decode never runs) and a
    /// [`Session::cancel`] call (so an in-flight decode exits at its
    /// next between-token check with [`FinishReason::Cancelled`]).
    /// Either path releases the session mutex; subsequent calls see
    /// a clean session. You can also call [`Session::cancel`]
    /// directly from any thread to trigger the same in-flight exit
    /// without dropping the future. See `AsyncCancelGuard` for the
    /// full rationale.
    ///
    /// On error the wrapper performs the same poisoned-mutex handling
    /// as sync [`Session::generate`]. `JoinError` from a panic in the
    /// blocking closure surfaces as [`FfiError::Backend`] with a
    /// diagnostic prefix.
    pub async fn generate_async(
        self: Arc<Self>,
        opts: GenerateOpts,
    ) -> Result<GenerateOutput, FfiError> {
        let session_for_guard = Arc::clone(&self);
        let handle = tokio::task::spawn_blocking(move || self.generate(opts));
        let mut guard = AsyncCancelGuard {
            session: session_for_guard,
            abort: handle.abort_handle(),
            armed: true,
        };
        let join_result = handle.await;
        guard.armed = false;
        join_result.map_err(|e| FfiError::Backend {
            detail: format!("generate_async join error: {e}"),
        })?
    }

    /// Async variant of [`Session::generate_streaming`] — delivers
    /// tokens and audio frames to the foreign [`ModalitySink`] as the
    /// decode loop produces them, from within a blocking worker so
    /// the caller's async runtime stays responsive.
    ///
    /// Sink callbacks run on the blocking worker thread that's
    /// executing the decode — **not** on the caller's async thread.
    /// The reentrancy hazard documented on
    /// [`Session::generate_streaming`] still applies: sink callbacks
    /// that call back into `append_text` / `generate*` / `reset` from
    /// inside the session will deadlock on the session mutex.
    /// [`Session::cancel`] and [`Session::position`] remain atomic-
    /// backed and safe to invoke from any thread (including from
    /// inside a callback).
    ///
    /// Cancellation: dropping the returned future fires the same
    /// abort + [`Session::cancel`] pair as [`Session::generate_async`]
    /// (see `AsyncCancelGuard`). For an in-flight decode, the loop
    /// exits with [`FinishReason::Cancelled`] and the sink's `on_done`
    /// fires on the blocking worker before the task completes —
    /// foreign consumers get the terminal signal even though they've
    /// already stopped awaiting. For a queued-but-not-started decode,
    /// abort cancels the task without ever running the closure; no
    /// sink callbacks fire for that case (the decode never began).
    pub async fn generate_streaming_async(
        self: Arc<Self>,
        opts: GenerateOpts,
        sink: Arc<dyn ModalitySink>,
    ) -> Result<GenerateSummary, FfiError> {
        let session_for_guard = Arc::clone(&self);
        let handle = tokio::task::spawn_blocking(move || self.generate_streaming(opts, sink));
        let mut guard = AsyncCancelGuard {
            session: session_for_guard,
            abort: handle.abort_handle(),
            armed: true,
        };
        let join_result = handle.await;
        guard.armed = false;
        join_result.map_err(|e| FfiError::Backend {
            detail: format!("generate_streaming_async join error: {e}"),
        })?
    }
}

/// Runs `work` on a tokio blocking worker under an [`AbortOnDrop`] guard.
///
/// The one place the `spawn_blocking` + guard + join-error-mapping sequence
/// lives, for the exports whose blocking work has no cooperative cancel point:
/// the async engine constructors (via [`spawn_engine_build`]) and
/// [`list_leap_bundles_async`]. They differ only in `what`, the name in the
/// join-error message, and in what they do with the value.
///
/// Not for [`Session::generate_async`] or `generate_streaming_async`. Those
/// hold a cancel flag the caller can signal, so they use [`AsyncCancelGuard`]
/// to call `Session::cancel` on drop; routing them through here would silently
/// drop that.
///
/// Cancellation is therefore the weak form those exports document: dropping the
/// returned future aborts the task only while it is still queued, since neither
/// a blocking HTTP GET nor an engine build can be interrupted once running.
async fn spawn_blocking_guarded<T, F>(what: &str, work: F) -> Result<T, FfiError>
where
    F: FnOnce() -> Result<T, FfiError> + Send + 'static,
    T: Send + 'static,
{
    let handle = tokio::task::spawn_blocking(work);
    let mut guard = AbortOnDrop {
        abort: handle.abort_handle(),
        armed: true,
    };
    let join_result = handle.await;
    guard.armed = false;
    join_result.map_err(|e| FfiError::Backend {
        detail: format!("{what} join error: {e}"),
    })?
}

/// Runs a blocking engine construction on tokio and wraps the result.
///
/// The three async constructors differ only in the call they make and the name
/// in their join-error message, so this adds the one thing they share on top of
/// [`spawn_blocking_guarded`]: the `Arc` wrap. Three copies of it is how
/// families of near-identical methods start to diverge, so there is one.
///
/// Cancellation is the weak form the constructors document: dropping the
/// returned future aborts the task only while it is still queued, because
/// engine construction has no cooperative cancel point.
async fn spawn_engine_build<F>(what: &str, build: F) -> Result<Arc<CeraEngine>, FfiError>
where
    F: FnOnce() -> Result<cera::CeraEngine, cera::CeraError> + Send + 'static,
{
    spawn_blocking_guarded(what, move || build().map_err(FfiError::from))
        .await
        .map(|inner| Arc::new(CeraEngine { inner }))
}

// Async CeraEngine constructors (PR 11).
#[uniffi::export(async_runtime = "tokio")]
impl CeraEngine {
    /// Async variant of [`CeraEngine::from_bundle_id`] — offloads the
    /// manifest + GGUF download and the engine construction onto a
    /// tokio blocking worker so the caller's async context isn't
    /// stalled. Foreign async runtimes (Kotlin coroutines, Swift
    /// `async`, Python `asyncio`) `.await` it directly.
    ///
    /// `config.bundle_repo` must be set (same constraint as the sync
    /// twin); construct a [`BundleRepo`] rooted at a persistent cache
    /// directory and attach it to the config before calling.
    ///
    /// Cancellation semantics (weaker than [`Session::generate_async`]):
    /// dropping the returned future drops the `AbortOnDrop` guard,
    /// which calls `AbortHandle::abort` on the spawned task. That
    /// cancels the task if it's still queued on tokio's blocking
    /// pool, so a not-yet-started download never runs. But if the
    /// task has started, abort is a no-op — the download is a
    /// `reqwest::blocking` call with no cooperative cancel point,
    /// and cera's engine-construction code (tokenizer build, model
    /// load, KV alloc) also isn't interruptible. In that case the
    /// task runs to completion and the engine is constructed then
    /// dropped; the downloaded bundle stays cached, so the caller's
    /// next attempt starts from that cache hit. Bandwidth isn't
    /// wasted, it's just shifted.
    ///
    /// `JoinError` from a panicking blocking closure surfaces as
    /// [`FfiError::Backend`] with a diagnostic prefix, same as
    /// [`Session::generate_async`].
    #[uniffi::constructor]
    pub async fn from_bundle_id_async(
        bundle_id: String,
        quant: String,
        config: EngineConfig,
    ) -> Result<Arc<Self>, FfiError> {
        // Convert the config synchronously. Any `TryFrom` error
        // (e.g. 32-bit `u64 → usize` overflow on the context size)
        // fails fast without spawning a blocking task.
        let cera_config: cera::EngineConfig = config.try_into()?;
        spawn_engine_build("from_bundle_id_async", move || {
            cera::CeraEngine::from_bundle_id(&bundle_id, &quant, cera_config)
        })
        .await
    }

    /// Async variant of [`CeraEngine::from_path`]: moves the GGUF open,
    /// tokenizer build, and KV allocation onto a tokio blocking worker.
    ///
    /// The sync twin is not cheap enough to call from a UI thread. GGUF
    /// tensor data is memory-mapped rather than read, so the cost is not
    /// proportional to file size, but the tokenizer is built eagerly and
    /// a large vocabulary's merge table is real work: enough to drop
    /// frames, and on a cold page cache the metadata reads are disk-bound
    /// on top. Foreign UI code should prefer this everywhere.
    ///
    /// Cancellation is the weak form documented on
    /// [`CeraEngine::from_bundle_id_async`]: dropping the future aborts
    /// the task only while it is still queued. Engine construction has no
    /// cooperative cancel point, so once started it runs to completion and
    /// the result is dropped.
    #[uniffi::constructor]
    pub async fn from_path_async(
        path: String,
        config: EngineConfig,
    ) -> Result<Arc<Self>, FfiError> {
        // Convert the config synchronously so a bad one fails fast without
        // spawning, exactly as `from_bundle_id_async` does.
        let cera_config: cera::EngineConfig = config.try_into()?;
        spawn_engine_build("from_path_async", move || {
            cera::CeraEngine::from_path(&path, cera_config)
        })
        .await
    }

    /// Async variant of [`CeraEngine::from_bytes`]: the in-memory twin of
    /// [`CeraEngine::from_path_async`], for callers with no filesystem.
    ///
    /// This one benefits more than the path variant: `from_bytes` has no
    /// mmap to lean on, so every tensor is already resident and the whole
    /// parse plus tokenizer build happens inline. Same weak cancellation.
    ///
    /// The `bytes` are moved into the blocking task, so a dropped future
    /// releases them when the task finishes rather than when it is
    /// dropped.
    #[uniffi::constructor]
    pub async fn from_bytes_async(
        bytes: Vec<u8>,
        config: EngineConfig,
    ) -> Result<Arc<Self>, FfiError> {
        let cera_config: cera::EngineConfig = config.try_into()?;
        spawn_engine_build("from_bytes_async", move || {
            cera::CeraEngine::from_bytes(bytes, cera_config)
        })
        .await
    }

    /// Async variant of [`CeraEngine::from_parts`].
    ///
    /// Wanted more than the text-only twin, not less: a VL bundle is the
    /// model *plus* its tower, so there is strictly more parsing to keep
    /// off the caller's thread, and the vision encoder's weights are
    /// built during the load. Same weak cancellation, and both buffers
    /// are moved into the blocking task.
    #[uniffi::constructor]
    pub async fn from_parts_async(
        bytes: Vec<u8>,
        multimodal_projector: Option<Vec<u8>>,
        inference_type: Option<String>,
        config: EngineConfig,
    ) -> Result<Arc<Self>, FfiError> {
        let cera_config: cera::EngineConfig = config.try_into()?;
        spawn_engine_build("from_parts_async", move || {
            let parts = cera::ModelBytes {
                model: bytes.into(),
                multimodal_projector: multimodal_projector.map(Into::into),
                audio_decoder: None,
                audio_tokenizer: None,
                draft_model: None,
                inference_type: inference_type
                    .as_deref()
                    .map(cera::manifest::InferenceType::parse_str),
                chat_template: None,
                generation_defaults: None,
            };
            cera::CeraEngine::from_parts(parts, cera_config)
        })
        .await
    }
}

// Session-level method on CeraEngine.
#[uniffi::export]
impl CeraEngine {
    /// Open a new [`Session`] sharing this engine's model + tokenizer
    /// by `Arc` clone. The returned session outlives `&self`; the
    /// engine keeps the shared state live for every session it hands
    /// out. Cheap — no model load, just config + state allocation.
    pub fn new_session(&self, config: SessionConfig) -> Result<Arc<Session>, FfiError> {
        let session = self.inner.new_session(config.into())?;
        let position = session.position_handle();
        let cancel = session.cancel_handle();
        let capabilities = session.capabilities().into();
        let hidden_size = u32::try_from(session.hidden_size()).unwrap_or(u32::MAX);
        Ok(Arc::new(Session {
            inner: std::sync::Mutex::new(session),
            position,
            cancel,
            capabilities,
            hidden_size,
        }))
    }
}

// ---------------------------------------------------------------------------
// Smoke test (from PR 1)
// ---------------------------------------------------------------------------

/// Version string of the `cera-ffi` crate. Useful as a smoke test
/// from the foreign-language side — if this is callable, the binding
/// pipeline works end-to-end.
#[uniffi::export]
pub fn cera_ffi_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// One-line CPU backend report for this host — the resolved SIMD tier plus the
/// detected feature flags, e.g. `cpu: tier=neon+dotprod [neon dotprod]`. A host
/// property independent of any loaded model; callable without an engine. Handy
/// for telemetry and bug reports (tells you which kernel path actually ran).
#[uniffi::export]
pub fn cpu_backend_report() -> String {
    cera::cpu_features().report()
}

// ---------------------------------------------------------------------------
// Voice Activity Detection (VAD)
// ---------------------------------------------------------------------------

/// Audio sample rate supported by Silero VAD.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, uniffi::Enum)]
pub enum FfiVadSampleRate {
    Rate16kHz,
    Rate8kHz,
}

impl From<FfiVadSampleRate> for cera::vad::VadSampleRate {
    fn from(r: FfiVadSampleRate) -> Self {
        match r {
            FfiVadSampleRate::Rate16kHz => cera::vad::VadSampleRate::Rate16kHz,
            FfiVadSampleRate::Rate8kHz => cera::vad::VadSampleRate::Rate8kHz,
        }
    }
}

impl From<cera::vad::VadSampleRate> for FfiVadSampleRate {
    fn from(r: cera::vad::VadSampleRate) -> Self {
        match r {
            cera::vad::VadSampleRate::Rate16kHz => FfiVadSampleRate::Rate16kHz,
            cera::vad::VadSampleRate::Rate8kHz => FfiVadSampleRate::Rate8kHz,
        }
    }
}

/// A detected speech segment with sample and millisecond boundaries.
#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct FfiSpeechTimestamp {
    pub start_sample: u64,
    pub end_sample: u64,
    pub start_ms: f32,
    pub end_ms: f32,
}

impl From<cera::vad::SpeechTimestamp> for FfiSpeechTimestamp {
    fn from(ts: cera::vad::SpeechTimestamp) -> Self {
        Self {
            start_sample: ts.start_sample,
            end_sample: ts.end_sample,
            start_ms: ts.start_ms,
            end_ms: ts.end_ms,
        }
    }
}

/// Configuration options for batch speech detection and segmentation.
#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct FfiVadConfig {
    #[uniffi(default = 0.5)]
    pub threshold: f32,
    #[uniffi(default = 0.35)]
    pub neg_threshold: f32,
    #[uniffi(default = 64)]
    pub min_speech_duration_ms: u32,
    #[uniffi(default = 100)]
    pub min_silence_duration_ms: u32,
    #[uniffi(default = 30)]
    pub speech_pad_ms: u32,
}

impl From<cera::vad::VadConfig> for FfiVadConfig {
    fn from(cfg: cera::vad::VadConfig) -> Self {
        Self {
            threshold: cfg.threshold,
            neg_threshold: cfg.neg_threshold,
            min_speech_duration_ms: cfg.min_speech_duration_ms as u32,
            min_silence_duration_ms: cfg.min_silence_duration_ms as u32,
            speech_pad_ms: cfg.speech_pad_ms as u32,
        }
    }
}

impl Default for FfiVadConfig {
    fn default() -> Self {
        cera::vad::VadConfig::default().into()
    }
}

impl From<FfiVadConfig> for cera::vad::VadConfig {
    fn from(cfg: FfiVadConfig) -> Self {
        Self {
            threshold: cfg.threshold,
            neg_threshold: cfg.neg_threshold,
            min_speech_duration_ms: cfg.min_speech_duration_ms as usize,
            min_silence_duration_ms: cfg.min_silence_duration_ms as usize,
            speech_pad_ms: cfg.speech_pad_ms as usize,
        }
    }
}

/// Default VAD configuration parameters.
#[uniffi::export]
pub fn silero_vad_default_config() -> FfiVadConfig {
    FfiVadConfig::default()
}

/// Stateful Silero Voice Activity Detector (VAD) session.
#[derive(uniffi::Object)]
pub struct FfiSileroVad {
    inner: std::sync::Mutex<cera::vad::SileroVad>,
}

impl FfiSileroVad {
    fn lock_inner(&self) -> Result<std::sync::MutexGuard<'_, cera::vad::SileroVad>, FfiError> {
        self.inner.lock().map_err(|e| FfiError::Backend {
            detail: format!("VAD mutex poisoned: {e}"),
        })
    }
}

#[uniffi::export]
impl FfiSileroVad {
    /// Load a Silero VAD model from a `.gguf` file path.
    #[uniffi::constructor]
    pub fn from_file(path: String) -> Result<Arc<Self>, FfiError> {
        let vad = cera::vad::SileroVad::from_file(&path).map_err(|e| FfiError::Backend {
            detail: e.to_string(),
        })?;
        Ok(Arc::new(Self {
            inner: std::sync::Mutex::new(vad),
        }))
    }

    /// Load a Silero VAD model from in-memory GGUF bytes.
    #[uniffi::constructor]
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Arc<Self>, FfiError> {
        let vad = cera::vad::SileroVad::from_bytes(bytes).map_err(|e| FfiError::Backend {
            detail: e.to_string(),
        })?;
        Ok(Arc::new(Self {
            inner: std::sync::Mutex::new(vad),
        }))
    }

    /// Reset recurrent state tensors and streaming context to zeros.
    pub fn reset(&self) -> Result<(), FfiError> {
        let mut vad = self.lock_inner()?;
        vad.reset();
        Ok(())
    }

    /// Process a single chunk of audio and return the speech probability in `[0.0, 1.0]`.
    ///
    /// - 16 kHz: chunk must have exactly 512 samples.
    /// - 8 kHz: chunk must have exactly 256 samples.
    pub fn process_chunk(&self, chunk: Vec<f32>, rate: FfiVadSampleRate) -> Result<f32, FfiError> {
        let mut vad = self.lock_inner()?;
        vad.process_chunk(&chunk, rate.into())
            .map_err(|e| FfiError::Backend {
                detail: e.to_string(),
            })
    }

    /// Process an entire audio buffer and return speech timestamps.
    pub fn get_speech_timestamps(
        &self,
        audio: Vec<f32>,
        rate: FfiVadSampleRate,
        config: Option<FfiVadConfig>,
    ) -> Result<Vec<FfiSpeechTimestamp>, FfiError> {
        let mut vad = self.lock_inner()?;
        let cfg = config.unwrap_or_default().into();
        let timestamps = vad
            .get_speech_timestamps(&audio, rate.into(), &cfg)
            .map_err(|e| FfiError::Backend {
                detail: e.to_string(),
            })?;
        Ok(timestamps.into_iter().map(Into::into).collect())
    }
}

/// A speech boundary event emitted during streaming audio processing.
#[derive(Debug, Clone, PartialEq, uniffi::Enum)]
pub enum FfiVadEvent {
    SpeechStart {
        sample: u64,
        ms: f32,
    },
    SpeechEnd {
        start_sample: u64,
        end_sample: u64,
        start_ms: f32,
        end_ms: f32,
    },
}

impl From<cera::vad::VadEvent> for FfiVadEvent {
    fn from(ev: cera::vad::VadEvent) -> Self {
        match ev {
            cera::vad::VadEvent::SpeechStart { sample, ms } => Self::SpeechStart { sample, ms },
            cera::vad::VadEvent::SpeechEnd {
                start_sample,
                end_sample,
                start_ms,
                end_ms,
            } => Self::SpeechEnd {
                start_sample,
                end_sample,
                start_ms,
                end_ms,
            },
        }
    }
}

/// Stateful speech boundary detector for live audio streams.
#[derive(uniffi::Object)]
pub struct FfiVadIterator {
    inner: std::sync::Mutex<cera::vad::VadIterator>,
}

impl FfiVadIterator {
    fn lock_inner(&self) -> Result<std::sync::MutexGuard<'_, cera::vad::VadIterator>, FfiError> {
        self.inner.lock().map_err(|e| FfiError::Backend {
            detail: format!("VAD iterator mutex poisoned: {e}"),
        })
    }
}

#[uniffi::export]
impl FfiVadIterator {
    /// Create a new streaming speech boundary iterator.
    #[uniffi::constructor]
    pub fn new(rate: FfiVadSampleRate, config: Option<FfiVadConfig>) -> Arc<Self> {
        let cfg = config.unwrap_or_default().into();
        Arc::new(Self {
            inner: std::sync::Mutex::new(cera::vad::VadIterator::new(rate.into(), cfg)),
        })
    }

    /// Reset iterator state.
    pub fn reset(&self) -> Result<(), FfiError> {
        let mut it = self.lock_inner()?;
        it.reset();
        Ok(())
    }

    /// Flush any pending in-flight speech segment at the end of an audio stream.
    pub fn flush(&self) -> Result<Option<FfiVadEvent>, FfiError> {
        let mut it = self.lock_inner()?;
        Ok(it.flush().map(Into::into))
    }

    /// Process a single chunk of audio and return any speech start or end event.
    pub fn process_chunk(
        &self,
        vad: &FfiSileroVad,
        chunk: Vec<f32>,
    ) -> Result<Option<FfiVadEvent>, FfiError> {
        let mut it = self.lock_inner()?;
        let mut v = vad.lock_inner()?;
        let ev = it
            .process_chunk(&mut v, &chunk)
            .map_err(|e| FfiError::Backend {
                detail: e.to_string(),
            })?;
        Ok(ev.map(Into::into))
    }
}

// ---------------------------------------------------------------------------
// Keyword Spotting (KWS) types
// ---------------------------------------------------------------------------

/// Configuration options for keyword spotting.
#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct FfiHotwordConfig {
    /// Activation probability threshold (default: 0.75).
    pub threshold: f32,
    /// Post-detection debounce cooldown in milliseconds (default: 2000 ms).
    pub cooldown_ms: u32,
    /// Evaluation step interval in milliseconds (default: 80 ms).
    pub step_ms: u32,
    /// Sliding window length in milliseconds (default: 1200 ms).
    pub window_ms: u32,
    /// Audio pre-roll margin in milliseconds preserved before command (default: 150 ms).
    pub pre_roll_ms: u32,
    /// VAD speech probability threshold for gating KWS (default: 0.5).
    pub vad_threshold: f32,
}

impl From<cera::hotword::HotwordConfig> for FfiHotwordConfig {
    fn from(cfg: cera::hotword::HotwordConfig) -> Self {
        Self {
            threshold: cfg.threshold,
            cooldown_ms: cfg.cooldown_ms as u32,
            step_ms: cfg.step_ms as u32,
            window_ms: cfg.window_ms as u32,
            pre_roll_ms: cfg.pre_roll_ms as u32,
            vad_threshold: cfg.vad_threshold,
        }
    }
}

impl From<FfiHotwordConfig> for cera::hotword::HotwordConfig {
    fn from(cfg: FfiHotwordConfig) -> Self {
        Self {
            threshold: cfg.threshold,
            cooldown_ms: cfg.cooldown_ms as usize,
            step_ms: cfg.step_ms as usize,
            window_ms: cfg.window_ms as usize,
            pre_roll_ms: cfg.pre_roll_ms as usize,
            vad_threshold: cfg.vad_threshold,
        }
    }
}

impl Default for FfiHotwordConfig {
    fn default() -> Self {
        cera::hotword::HotwordConfig::default().into()
    }
}

/// Default KWS configuration parameters.
#[uniffi::export]
pub fn hotword_default_config() -> FfiHotwordConfig {
    FfiHotwordConfig::default()
}

/// Confidence score for a specific keyword candidate.
#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct FfiHotwordScore {
    /// Target keyword string.
    pub keyword: String,
    /// Model activation probability between 0.0 and 1.0.
    pub score: f32,
}

impl From<cera::hotword::HotwordScore> for FfiHotwordScore {
    fn from(s: cera::hotword::HotwordScore) -> Self {
        Self {
            keyword: s.keyword,
            score: s.score,
        }
    }
}

/// Event emitted when a keyword spotting threshold is crossed.
#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct FfiHotwordEvent {
    /// The matched keyword string.
    pub keyword: String,
    /// Exact audio stream sample index where the keyword completed.
    pub sample_offset: u64,
    /// Audio stream sample index including pre-roll safety margin for downstream ASR.
    pub command_start_sample: u64,
    /// Timestamp in milliseconds from stream origin where keyword completed.
    pub timestamp_ms: f32,
    /// Model confidence probability (0.0 to 1.0).
    pub confidence: f32,
}

impl From<cera::hotword::HotwordEvent> for FfiHotwordEvent {
    fn from(ev: cera::hotword::HotwordEvent) -> Self {
        Self {
            keyword: ev.keyword,
            sample_offset: ev.sample_offset,
            command_start_sample: ev.command_start_sample,
            timestamp_ms: ev.timestamp_ms,
            confidence: ev.confidence,
        }
    }
}

/// Stateful Keyword Spotting detector executing pure-Rust forward inference.
#[derive(uniffi::Object)]
pub struct FfiHotwordDetector {
    inner: std::sync::Mutex<cera::hotword::HotwordDetector>,
}

impl FfiHotwordDetector {
    fn lock_inner(&self) -> std::sync::MutexGuard<'_, cera::hotword::HotwordDetector> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }
}

#[uniffi::export]
impl FfiHotwordDetector {
    /// Load a KWS model from a `.gguf` file path.
    #[uniffi::constructor]
    pub fn from_file(path: String) -> Result<Arc<Self>, FfiError> {
        let detector =
            cera::hotword::HotwordDetector::from_file(&path).map_err(|e| FfiError::Backend {
                detail: e.to_string(),
            })?;
        Ok(Arc::new(Self {
            inner: std::sync::Mutex::new(detector),
        }))
    }

    /// Load a KWS model from in-memory GGUF bytes.
    #[uniffi::constructor]
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Arc<Self>, FfiError> {
        let detector =
            cera::hotword::HotwordDetector::from_bytes(bytes).map_err(|e| FfiError::Backend {
                detail: e.to_string(),
            })?;
        Ok(Arc::new(Self {
            inner: std::sync::Mutex::new(detector),
        }))
    }

    /// List of target keywords supported by this model.
    pub fn keywords(&self) -> Result<Vec<String>, FfiError> {
        let det = self.lock_inner();
        Ok(det.keywords().to_vec())
    }

    /// Get default configuration suggested by model metadata.
    pub fn default_config(&self) -> Result<FfiHotwordConfig, FfiError> {
        let det = self.lock_inner();
        Ok(det.default_config().into())
    }

    /// Process a full audio window and return probability scores for each keyword.
    pub fn process_window(&self, window: Vec<f32>) -> Result<Vec<f32>, FfiError> {
        let mut det = self.lock_inner();
        let scores = det.process_window(&window).map_err(|e| FfiError::Backend {
            detail: e.to_string(),
        })?;
        Ok(scores.to_vec())
    }
}

/// Streaming Keyword Spotting manager with VAD gating and debounce state.
#[derive(uniffi::Object)]
pub struct FfiHotwordIterator {
    inner: std::sync::Mutex<cera::hotword::HotwordIterator>,
}

impl FfiHotwordIterator {
    fn lock_inner(&self) -> std::sync::MutexGuard<'_, cera::hotword::HotwordIterator> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }
}

#[uniffi::export]
impl FfiHotwordIterator {
    /// Load and construct a streaming hotword iterator from file paths.
    #[uniffi::constructor]
    pub fn from_files(
        model_path: String,
        vad_model_path: Option<String>,
        config: Option<FfiHotwordConfig>,
    ) -> Result<Arc<Self>, FfiError> {
        let detector = cera::hotword::HotwordDetector::from_file(&model_path).map_err(|e| {
            FfiError::Backend {
                detail: e.to_string(),
            }
        })?;
        let vad = if let Some(vp) = vad_model_path {
            let v = cera::vad::SileroVad::from_file(&vp).map_err(|e| FfiError::Backend {
                detail: e.to_string(),
            })?;
            Some(v)
        } else {
            None
        };
        let cfg = config.unwrap_or_default().into();
        Ok(Arc::new(Self {
            inner: std::sync::Mutex::new(cera::hotword::HotwordIterator::new(detector, vad, cfg)),
        }))
    }

    /// Reset iterator state, ring buffer, and debounce timers.
    pub fn reset(&self) -> Result<(), FfiError> {
        let mut it = self.lock_inner();
        it.reset();
        Ok(())
    }

    /// Process a streaming audio chunk and return a detection event if triggered.
    ///
    /// For chunks containing multiple hops, returns the first detected event encountered
    /// during the chunk evaluation steps (or `None` if silence or cooldown persists).
    pub fn process_chunk(&self, chunk: Vec<f32>) -> Result<Option<FfiHotwordEvent>, FfiError> {
        let mut it = self.lock_inner();
        let ev = it.process_chunk(&chunk).map_err(|e| FfiError::Backend {
            detail: e.to_string(),
        })?;
        Ok(ev.map(Into::into))
    }
}

// ---------------------------------------------------------------------------
// Speech Recognition / Whisper ASR types
// ---------------------------------------------------------------------------

/// Options for Whisper speech transcription.
#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct FfiWhisperTranscribeOpts {
    /// Language code (e.g. "en", "es", "fr").
    /// If None or Some("auto"), dynamic language auto-detection is performed.
    pub language: Option<String>,
    /// Whether to translate speech into English instead of transcribing in source language.
    pub translate: bool,
    /// Whether to output segment timestamps (<|0.00|> to <|30.00|>).
    pub timestamps: bool,
    /// Maximum new tokens to decode (defaults to 448).
    pub max_tokens: Option<u32>,
    /// Temperature for sampling (0.0 = greedy).
    pub temperature: Option<f32>,
}

impl Default for FfiWhisperTranscribeOpts {
    fn default() -> Self {
        Self {
            language: None,
            translate: false,
            timestamps: false,
            max_tokens: Some(448),
            temperature: Some(0.0),
        }
    }
}

impl From<FfiWhisperTranscribeOpts> for cera::WhisperTranscribeOpts {
    fn from(opts: FfiWhisperTranscribeOpts) -> Self {
        Self {
            language: opts.language,
            translate: opts.translate,
            timestamps: opts.timestamps,
            max_tokens: opts.max_tokens.map(|v| v as usize).unwrap_or(448),
            temperature: opts.temperature.unwrap_or(0.0),
            cancel: None,
        }
    }
}

impl From<cera::WhisperTranscribeOpts> for FfiWhisperTranscribeOpts {
    fn from(opts: cera::WhisperTranscribeOpts) -> Self {
        Self {
            language: opts.language,
            translate: opts.translate,
            timestamps: opts.timestamps,
            max_tokens: Some(opts.max_tokens as u32),
            temperature: Some(opts.temperature),
        }
    }
}

/// Default transcription options for Whisper ASR.
#[uniffi::export]
pub fn whisper_default_transcribe_opts() -> FfiWhisperTranscribeOpts {
    FfiWhisperTranscribeOpts::default()
}

/// Standalone pure-Rust OpenAI Whisper speech recognition engine.
#[derive(uniffi::Object)]
pub struct FfiWhisperModel {
    model: cera::WhisperModel,
    tokenizer: cera::tokenizer::BpeTokenizer,
}

#[uniffi::export]
impl FfiWhisperModel {
    /// Load a Whisper ASR model from a local `.gguf` file path.
    #[uniffi::constructor]
    pub fn from_file(path: String) -> Result<Arc<Self>, FfiError> {
        let (model, tokenizer) =
            cera::WhisperModel::from_file(&path).map_err(|e| FfiError::Backend {
                detail: format!("failed to load Whisper model from {path}: {e}"),
            })?;
        Ok(Arc::new(Self { model, tokenizer }))
    }

    /// Load a Whisper ASR model from an in-memory GGUF byte buffer.
    #[uniffi::constructor]
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Arc<Self>, FfiError> {
        let (model, tokenizer) =
            cera::WhisperModel::from_bytes(bytes).map_err(|e| FfiError::Backend {
                detail: format!("failed to load Whisper model from bytes: {e}"),
            })?;
        Ok(Arc::new(Self { model, tokenizer }))
    }

    /// Transcribe 16 kHz mono PCM audio samples synchronously.
    pub fn transcribe(
        &self,
        pcm: Vec<f32>,
        opts: Option<FfiWhisperTranscribeOpts>,
    ) -> Result<String, FfiError> {
        let cera_opts: cera::WhisperTranscribeOpts = opts.unwrap_or_default().into();
        self.model
            .transcribe(&self.tokenizer, &pcm, &cera_opts)
            .map_err(|e| FfiError::Backend {
                detail: e.to_string(),
            })
    }

    /// List standard 100 language codes supported by OpenAI Whisper in sequential token order.
    pub fn languages(&self) -> Vec<String> {
        cera::WHISPER_LANGUAGES
            .iter()
            .map(|&s| s.to_string())
            .collect()
    }

    /// Whether this Whisper model is multilingual (contains `<|transcribe|>` task token).
    pub fn is_multilingual(&self) -> bool {
        self.tokenizer.token_to_id("<|transcribe|>").is_some()
    }
}

#[uniffi::export(async_runtime = "tokio")]
impl FfiWhisperModel {
    /// Transcribe 16 kHz mono PCM audio samples asynchronously on a background blocking worker.
    pub async fn transcribe_async(
        self: Arc<Self>,
        pcm: Vec<f32>,
        opts: Option<FfiWhisperTranscribeOpts>,
    ) -> Result<String, FfiError> {
        let handle = tokio::task::spawn_blocking(move || self.transcribe(pcm, opts));
        handle.await.map_err(|e| FfiError::Backend {
            detail: format!("transcribe_async worker task failed: {e}"),
        })?
    }
}

// ---------------------------------------------------------------------------
// PII Classification types
// ---------------------------------------------------------------------------

/// An identified PII entity span in source text.
#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct FfiEntitySpan {
    /// Entity label type (e.g. "NAME", "EMAIL", "PHONE_NUMBER", "STREET_ADDRESS").
    pub entity_type: String,
    /// UTF-8 character start index in source text (inclusive).
    pub start_char: u64,
    /// UTF-8 character end index in source text (exclusive).
    pub end_char: u64,
    /// Token start index in sequence (inclusive).
    pub start_token: u64,
    /// Token end index in sequence (exclusive).
    pub end_token: u64,
    /// Extracted text slice.
    pub text: String,
    /// Mean classification confidence score across the span tokens [0.0..1.0].
    pub score: f32,
}

impl From<cera::EntitySpan> for FfiEntitySpan {
    fn from(s: cera::EntitySpan) -> Self {
        Self {
            entity_type: s.entity_type,
            start_char: s.start_char as u64,
            end_char: s.end_char as u64,
            start_token: s.start_token as u64,
            end_token: s.end_token as u64,
            text: s.text,
            score: s.score,
        }
    }
}

/// Zero-dependency PII Classifier for named entity recognition.
#[derive(uniffi::Object)]
pub struct PiiClassifier {
    engine: Arc<CeraEngine>,
    adapter: Option<Arc<cera::lora::LoraAdapterWeights>>,
}

#[uniffi::export]
impl PiiClassifier {
    /// Load a PII classification model from a local GGUF path.
    #[uniffi::constructor]
    pub fn from_path(path: String) -> Result<Arc<Self>, FfiError> {
        let engine = CeraEngine::from_path(path, EngineConfig::default())?;
        Ok(Arc::new(Self {
            engine,
            adapter: None,
        }))
    }

    /// Load a base model with a separate LoRA classifier adapter.
    #[uniffi::constructor]
    pub fn from_base_and_adapter(
        base_path: String,
        adapter_path: String,
    ) -> Result<Arc<Self>, FfiError> {
        let engine = CeraEngine::from_path(base_path, EngineConfig::default())?;
        let adapter =
            cera::lora::LoraAdapterWeights::load_from_path(std::path::Path::new(&adapter_path))
                .map_err(|e| FfiError::Backend {
                    detail: format!("load adapter {adapter_path}: {e}"),
                })?;
        Ok(Arc::new(Self {
            engine,
            adapter: Some(adapter),
        }))
    }

    /// Detect PII entities in the input text.
    pub fn detect(&self, text: String) -> Result<Vec<FfiEntitySpan>, FfiError> {
        let spans = self
            .engine
            .inner
            .detect_pii_with_lora(&text, self.adapter.clone())
            .map_err(FfiError::from)?;
        Ok(spans.into_iter().map(Into::into).collect())
    }
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
        let v = cera_ffi_version();
        assert!(!v.is_empty(), "version string must not be empty");
    }

    #[test]
    fn engine_config_default_roundtrips_to_cera() {
        let ffi = EngineConfig::default();
        let core: cera::EngineConfig = ffi.try_into().unwrap();
        assert_eq!(core.context_size, 4096);
        assert_eq!(core.backend, cera::BackendPreference::Auto);
        assert!(!core.gpu_depthformer);
        // bundle_repo defaults to None: foreign callers opt in by
        // attaching a BundleRepo before the try_into.
        assert!(core.bundle_repo.is_none());
    }

    /// `ChatMessage` round-trips its `role` + `content` fields
    /// through the cera-core conversion. `From<ChatMessage> for
    /// cera::tokenizer::ChatMessage` is a trivial field-copy; this
    /// test pins the field shape so a future cera-core rename
    /// breaks compilation here loudly instead of silently dropping
    /// data on the FFI boundary.
    #[test]
    fn chat_message_converts_to_cera_core() {
        let m = ChatMessage {
            role: "user".to_string(),
            content: "hello".to_string(),
        };
        let core: cera::tokenizer::ChatMessage = m.clone().into();
        assert_eq!(core.role, "user");
        assert_eq!(core.content, "hello");
        // Original FFI value is unchanged (we cloned for the
        // conversion); proves the From doesn't mutate by ref.
        assert_eq!(m.role, "user");
    }

    /// Pick a temp path scoped to this process + test name so parallel
    /// test binaries and prior runs don't collide. `std::env::temp_dir()`
    /// honors `TMPDIR` on macOS and `/tmp` elsewhere; the process-id
    /// suffix is stable across the test's lifetime but unique per run.
    /// `remove_dir_all` on entry makes the existence assertion below
    /// deterministic even if a previous run's panic left the dir behind.
    fn unique_test_bundle_dir(test_name: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "cera-ffi-test-{}-{}",
            test_name,
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        path
    }

    /// `cache_size` round-trips through the FFI wrapper. Builds a
    /// small synthetic cache, verifies the FFI method returns the
    /// same byte count as the cera-core method (which has its own
    /// unit-test coverage in `cera/src/bundle/mod.rs`).
    #[test]
    fn bundle_repo_cache_size_forwards_to_cera_core() {
        use std::fs;
        let dir = unique_test_bundle_dir("size");
        fs::create_dir_all(dir.join("huggingface.co/test")).unwrap();
        fs::write(dir.join("huggingface.co/test/file"), vec![0u8; 2048]).unwrap();
        let repo = BundleRepo::new(dir.to_string_lossy().into_owned());
        assert_eq!(repo.cache_size().unwrap(), 2048);
        let _ = fs::remove_dir_all(&dir);
    }

    /// `clear_cache` round-trips through the FFI wrapper. After the
    /// clear, `cache_size` reports 0 and `store_dir` still exists
    /// (subsequent downloads can land there).
    #[test]
    fn bundle_repo_clear_cache_wipes_files_via_ffi() {
        use std::fs;
        let dir = unique_test_bundle_dir("clear");
        fs::create_dir_all(dir.join("huggingface.co/test")).unwrap();
        fs::write(dir.join("huggingface.co/test/file"), vec![0u8; 512]).unwrap();
        let repo = BundleRepo::new(dir.to_string_lossy().into_owned());
        assert_eq!(repo.cache_size().unwrap(), 512);
        repo.clear_cache().unwrap();
        assert!(dir.exists(), "store_dir must survive clear_cache");
        assert_eq!(repo.cache_size().unwrap(), 0);
        let _ = fs::remove_dir_all(&dir);
    }

    /// `BundleRepo::new` wraps a `cera::bundle::BundleRepo` without
    /// creating the store_dir on disk (that happens lazily on first
    /// download). Store_dir round-trips.
    #[test]
    fn bundle_repo_constructs_and_store_dir_roundtrips() {
        let dir = unique_test_bundle_dir("construct");
        let dir_str = dir.to_string_lossy().into_owned();
        let repo = BundleRepo::new(dir_str.clone());
        assert_eq!(repo.store_dir(), dir_str);
        // Directory creation is lazy; the path must not exist yet.
        assert!(
            !dir.exists(),
            "BundleRepo::new eagerly created {}",
            dir.display()
        );
    }

    /// Attaching a `BundleRepo` to `EngineConfig` plumbs through to
    /// `cera::EngineConfig::bundle_repo` — proves the From/TryFrom
    /// conversion handles the Arc<BundleRepo> → Option<cera::BundleRepo>
    /// path correctly.
    #[test]
    fn engine_config_carries_bundle_repo_through_try_from() {
        let dir = unique_test_bundle_dir("carry");
        let repo = BundleRepo::new(dir.to_string_lossy().into_owned());
        let ffi = EngineConfig {
            context_size: 0,
            backend: BackendPreference::Cpu,
            bundle_repo: Some(repo.clone()),
            draft_model: None,
            gpu_depthformer: false,
        };
        let core: cera::EngineConfig = ffi.try_into().unwrap();
        let core_repo = core.bundle_repo.expect("bundle_repo must be Some");
        assert_eq!(core_repo.store_dir(), repo.inner.store_dir());
    }

    /// `DownloadProgressAdapter` forwards `on_progress` calls from
    /// cera-core to a foreign-trait implementation (impl'd here as a
    /// recording Rust struct that mirrors how UniFFI codegens the
    /// foreign side). Verifies the URL / bytes / total round-trip
    /// without dropping data.
    #[test]
    fn download_progress_adapter_forwards() {
        use cera::bundle::DownloadProgress as _;
        use std::sync::Mutex;

        #[derive(Debug, Default)]
        struct Recorder {
            calls: Mutex<Vec<(String, u64, Option<u64>)>>,
        }
        impl DownloadProgressSink for Recorder {
            fn on_progress(&self, url: String, bytes: u64, total: Option<u64>) {
                self.calls.lock().unwrap().push((url, bytes, total));
            }
        }

        let recorder: Arc<Recorder> = Arc::new(Recorder::default());
        let adapter = DownloadProgressAdapter {
            inner: recorder.clone() as Arc<dyn DownloadProgressSink>,
        };

        // Drive the adapter as cera-core's download_to would.
        adapter.on_progress("https://example.com/a.gguf", 1024, Some(2048));
        adapter.on_progress("https://example.com/a.gguf", 2048, Some(2048));
        adapter.on_progress("https://example.com/no-length", 512, None);

        let calls = recorder.calls.lock().unwrap();
        assert_eq!(calls.len(), 3);
        assert_eq!(calls[0].0, "https://example.com/a.gguf");
        assert_eq!(calls[0].1, 1024);
        assert_eq!(calls[0].2, Some(2048));
        assert_eq!(calls[2].0, "https://example.com/no-length");
        assert_eq!(calls[2].1, 512);
        assert_eq!(calls[2].2, None);
    }

    #[test]
    fn engine_config_zero_context_size_means_max() {
        // `0` on the wire is the FFI's "use model default" signal;
        // translate to `usize::MAX` so cera caps at model.max_seq_len.
        let ffi = EngineConfig {
            context_size: 0,
            backend: BackendPreference::Cpu,
            bundle_repo: None,
            draft_model: None,
            gpu_depthformer: false,
        };
        let core: cera::EngineConfig = ffi.try_into().unwrap();
        assert_eq!(core.context_size, usize::MAX);
    }

    #[test]
    fn engine_config_oversize_context_errors_on_32bit_targets() {
        // On 32-bit targets, `u64::MAX` exceeds `usize::MAX` and the
        // checked conversion must surface an error rather than
        // silently truncating. On 64-bit targets `usize::MAX ==
        // u64::MAX`, so the conversion succeeds: skip the assert
        // there. This test proves the error path compiles + is
        // reachable under the narrow condition where it matters.
        let ffi = EngineConfig {
            context_size: u64::MAX,
            backend: BackendPreference::Cpu,
            bundle_repo: None,
            draft_model: None,
            gpu_depthformer: false,
        };
        let result: Result<cera::EngineConfig, FfiError> = ffi.try_into();
        #[cfg(target_pointer_width = "32")]
        {
            let err = result.expect_err("u64::MAX must fail on 32-bit");
            match err {
                FfiError::Backend { detail } => {
                    assert!(
                        detail.contains("exceeds usize::MAX"),
                        "unexpected: {detail}"
                    );
                }
                other => panic!("expected Backend, got: {other:?}"),
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
            let core: cera::BackendPreference = ffi.into();
            let back: BackendPreference = core.into();
            assert_eq!(ffi, back, "{ffi:?} didn't round-trip");
        }
    }

    /// Every `cera::CeraError` variant maps to a specific `FfiError`
    /// variant (not the generic `Backend` catch-all) so foreign
    /// callers can pattern-match on class. If cera adds a new
    /// `CeraError` variant and forgets to update `From<CeraError>`,
    /// the exhaustive match in that impl breaks compilation loudly —
    /// this test just asserts the existing mapping is correct.
    #[test]
    fn cera_error_variants_map_to_typed_ffi_error_variants() {
        // Payload-free variants.
        assert!(matches!(
            FfiError::from(cera::CeraError::UnsupportedModality),
            FfiError::UnsupportedModality
        ));
        assert!(matches!(
            FfiError::from(cera::CeraError::Busy),
            FfiError::Busy
        ));
        assert!(matches!(
            FfiError::from(cera::CeraError::Cancelled),
            FfiError::Cancelled
        ));
        assert!(matches!(
            FfiError::from(cera::CeraError::EmptyInput),
            FfiError::EmptyInput
        ));

        // Payload-carrying variants preserve their fields.
        match FfiError::from(cera::CeraError::UnsupportedInferenceType(
            "audio-magic".into(),
        )) {
            FfiError::UnsupportedInferenceType { inference_type } => {
                assert_eq!(inference_type, "audio-magic");
            }
            other => panic!("expected UnsupportedInferenceType, got: {other:?}"),
        }

        match FfiError::from(cera::CeraError::ContextOverflow {
            max_seq_len: 4096,
            by: 17,
        }) {
            FfiError::ContextOverflow { max_seq_len, by } => {
                assert_eq!(max_seq_len, 4096);
                assert_eq!(by, 17);
            }
            other => panic!("expected ContextOverflow, got: {other:?}"),
        }

        match FfiError::from(cera::CeraError::Backend("metal driver crashed".into())) {
            FfiError::Backend { detail } => {
                assert_eq!(detail, "metal driver crashed");
            }
            other => panic!("expected Backend, got: {other:?}"),
        }

        // Io flattens the OS error to a string.
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "no such file");
        let io_str = io_err.to_string();
        match FfiError::from(cera::CeraError::Io(io_err)) {
            FfiError::Io { detail } => {
                assert_eq!(detail, io_str);
            }
            other => panic!("expected Io, got: {other:?}"),
        }
    }

    /// Display (`thiserror`-derived) produces byte-identical message
    /// text to the equivalent `cera::CeraError` for every variant
    /// that has a cera analog. Foreign callers logging the error via
    /// `.toString()` / `String(describing:)` / `str()` see the same
    /// output whether the error originates from cera directly or
    /// routes through the FFI wrapper.
    ///
    /// Tests every shared variant including `Backend`, `Io`, and
    /// `UnsupportedInferenceType` — an earlier iteration of this test
    /// quietly excluded them, which masked a real drift where the FFI
    /// side had dropped the `"backend: "` and `"io: "` label prefixes.
    #[test]
    fn ffi_error_display_matches_cera_error_for_every_shared_variant() {
        // One deliberate exclusion: `CeraError::LoraDimMismatch` maps to
        // `FfiError::LoraParse`, whose Display is the wider "lora: {detail}"
        // because it also covers adapter *load* failures that have no
        // `CeraError` counterpart. Every other shared variant is paired below,
        // and a new one belongs here rather than being quietly left out.

        // Prep the Io pair outside the vec since io::Error isn't
        // `Clone`: we need to consume one into `CeraError::Io` and
        // stash its pre-wrap display string for the FFI side.
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "no such file");
        let io_msg = io_err.to_string();

        let pairs: Vec<(FfiError, cera::CeraError)> = vec![
            (
                FfiError::UnsupportedModality,
                cera::CeraError::UnsupportedModality,
            ),
            (
                FfiError::UnsupportedInferenceType {
                    inference_type: "audio-magic".into(),
                },
                cera::CeraError::UnsupportedInferenceType("audio-magic".into()),
            ),
            (FfiError::Busy, cera::CeraError::Busy),
            (FfiError::Cancelled, cera::CeraError::Cancelled),
            (
                FfiError::ContextOverflow {
                    max_seq_len: 2048,
                    by: 5,
                },
                cera::CeraError::ContextOverflow {
                    max_seq_len: 2048,
                    by: 5,
                },
            ),
            (FfiError::EmptyInput, cera::CeraError::EmptyInput),
            (
                FfiError::Backend {
                    detail: "metal driver crashed".into(),
                },
                cera::CeraError::Backend("metal driver crashed".into()),
            ),
            (
                FfiError::InvalidToken {
                    id: 99,
                    vocab_size: 32,
                },
                cera::CeraError::InvalidToken {
                    id: 99,
                    vocab_size: 32,
                },
            ),
            (
                FfiError::LoraUnsupportedByBackend {
                    detail: "no routed-FFN hooks".into(),
                },
                cera::CeraError::LoraUnsupportedByBackend("no routed-FFN hooks".into()),
            ),
            (
                FfiError::OutOfMemory {
                    requested_bytes: 1 << 40,
                },
                cera::CeraError::OutOfMemory {
                    requested_bytes: 1 << 40,
                },
            ),
            (
                FfiError::KvCompressionConflict {
                    configured: "tq3".into(),
                    requested: "none".into(),
                },
                cera::CeraError::KvCompressionConflict {
                    configured: "tq3".into(),
                    requested: "none".into(),
                },
            ),
            (FfiError::Io { detail: io_msg }, cera::CeraError::Io(io_err)),
        ];
        for (ffi, core) in pairs {
            assert_eq!(
                ffi.to_string(),
                core.to_string(),
                "display mismatch for {ffi:?} vs {core:?}"
            );
        }
    }

    #[test]
    fn session_config_default_roundtrips_to_cera() {
        let ffi = SessionConfig::default();
        let core: cera::SessionConfig = ffi.into();
        let default_core = cera::SessionConfig::default();
        assert_eq!(core.max_seq_len, default_core.max_seq_len);
        assert_eq!(core.n_keep, default_core.n_keep);
        assert_eq!(core.seed, default_core.seed);
        assert_eq!(core.ubatch_size, default_core.ubatch_size);
        assert_eq!(core.gpu_depthformer, default_core.gpu_depthformer);
    }

    #[test]
    fn kv_compression_none_roundtrips() {
        let ffi = KvCompression::None;
        let core: cera::kv_cache::KvCompression = ffi.into();
        assert!(matches!(core, cera::kv_cache::KvCompression::None));
    }

    #[test]
    fn kv_compression_turboquant_roundtrips() {
        let ffi = KvCompression::TurboQuant {
            seed: 42,
            keys: true,
            values: false,
        };
        let core: cera::kv_cache::KvCompression = ffi.into();
        match core {
            cera::kv_cache::KvCompression::TurboQuant { seed, keys, values } => {
                assert_eq!(seed, 42);
                assert!(keys);
                assert!(!values);
            }
            _ => panic!("expected TurboQuant variant"),
        }
    }

    #[test]
    fn generate_opts_default_roundtrips_to_cera() {
        let ffi = GenerateOpts::default();
        let core: cera::GenerateOpts = ffi.try_into().expect("default opts have no grammar");
        let default_core = cera::GenerateOpts::default();
        // Field-by-field so a future cera field-add breaks here loudly.
        assert_eq!(core.max_tokens, default_core.max_tokens);
        assert_eq!(core.temperature, default_core.temperature);
        assert_eq!(core.top_p, default_core.top_p);
        assert_eq!(core.top_k, default_core.top_k);
        assert_eq!(core.min_p, default_core.min_p);
        assert_eq!(core.repetition_penalty, default_core.repetition_penalty);
        assert_eq!(core.stop_tokens, default_core.stop_tokens);
        assert_eq!(core.ignore_eos, default_core.ignore_eos);
        assert!(core.grammar.is_none());
        assert_eq!(core.flush_every_tokens, default_core.flush_every_tokens);
        assert_eq!(core.flush_every_ms, default_core.flush_every_ms);
        assert_eq!(
            core.spec.map(|s| (s.ngram, s.k)),
            default_core.spec.map(|s| (s.ngram, s.k))
        );
    }

    #[test]
    fn generate_opts_spec_decode_roundtrip() {
        let ffi = GenerateOpts {
            spec: Some(SpecDecodeConfig { ngram: 3, k: 8 }),
            ..GenerateOpts::default()
        };
        let core: cera::GenerateOpts = ffi.clone().try_into().expect("valid spec decode config");
        assert_eq!(core.spec.as_ref().map(|s| (s.ngram, s.k)), Some((3, 8)));
        let back = GenerateOpts::from(&core);
        assert_eq!(back.spec, Some(SpecDecodeConfig { ngram: 3, k: 8 }));
    }

    #[test]
    fn generate_opts_spec_decode_boundary_clamping() {
        let ffi_zero = GenerateOpts {
            spec: Some(SpecDecodeConfig { ngram: 0, k: 0 }),
            ..GenerateOpts::default()
        };
        let core_zero: cera::GenerateOpts = ffi_zero.try_into().expect("clamps zero");
        assert_eq!(
            core_zero.spec.as_ref().map(|s| (s.ngram, s.k)),
            Some((1, 1))
        );

        let ffi_huge = GenerateOpts {
            spec: Some(SpecDecodeConfig {
                ngram: 10_000,
                k: 50_000,
            }),
            ..GenerateOpts::default()
        };
        let core_huge: cera::GenerateOpts = ffi_huge.try_into().expect("clamps excessive values");
        assert_eq!(
            core_huge.spec.as_ref().map(|s| (s.ngram, s.k)),
            Some((32, 64))
        );
    }

    #[test]
    fn generate_opts_compiles_valid_grammar() {
        let ffi = GenerateOpts {
            grammar: Some(r#"root ::= "yes" | "no""#.to_string()),
            ..GenerateOpts::default()
        };
        let core: cera::GenerateOpts = ffi.try_into().expect("valid GBNF compiles");
        assert!(core.grammar.is_some());
    }

    #[test]
    fn generate_opts_rejects_malformed_grammar() {
        let ffi = GenerateOpts {
            // Unterminated string literal — the GBNF parser rejects it.
            grammar: Some(r#"root ::= "oops"#.to_string()),
            ..GenerateOpts::default()
        };
        let err = cera::GenerateOpts::try_from(ffi).unwrap_err();
        assert!(matches!(err, FfiError::GrammarParse { .. }), "got: {err:?}");
    }

    #[test]
    fn finish_reason_covers_every_variant() {
        use cera::FinishReason as Core;
        let cases = [
            (Core::MaxTokens, "MaxTokens"),
            (Core::Stop, "Stop"),
            (Core::Cancelled, "Cancelled"),
            (Core::ContextFull, "ContextFull"),
            (Core::GrammarDeadEnd, "GrammarDeadEnd"),
            (Core::Error("boom".into()), "Error"),
        ];
        for (core, tag) in cases {
            let ffi: FinishReason = core.into();
            match (&ffi, tag) {
                (FinishReason::MaxTokens, "MaxTokens") => {}
                (FinishReason::Stop, "Stop") => {}
                (FinishReason::Cancelled, "Cancelled") => {}
                (FinishReason::ContextFull, "ContextFull") => {}
                (FinishReason::GrammarDeadEnd, "GrammarDeadEnd") => {}
                (FinishReason::Error { message }, "Error") => assert_eq!(message, "boom"),
                _ => panic!("variant mismatch: {ffi:?} tagged {tag}"),
            }
        }
    }

    /// Exercises the ForeignSinkAdapter by implementing the FFI
    /// `ModalitySink` trait from Rust (what UniFFI codegens the foreign
    /// binding to look like on the Rust side) and driving it through
    /// the internal `cera::ModalitySink` impl. Confirms:
    /// - `on_text_tokens` decodes to text and forwards as `on_text_chunk`.
    /// - `on_audio_frames` forwards with the exact bytes + rate.
    /// - `on_done` forwards and maps the FinishReason through `.into()`.
    /// - Multi-byte UTF-8 character sequences split across token boundaries are
    ///   buffered and emitted cleanly once completed.
    #[test]
    fn foreign_sink_adapter_forwards_every_method() {
        use cera::ModalitySink as CoreSink;
        use std::sync::Mutex;

        #[derive(Default)]
        struct Recorder {
            thought: Mutex<Vec<String>>,
            text: Mutex<Vec<String>>,
            audio: Mutex<Vec<(Vec<f32>, u32)>>,
            done: Mutex<Option<FinishReason>>,
        }

        impl ModalitySink for Recorder {
            fn on_thought_chunk(&self, text: String) {
                self.thought.lock().unwrap().push(text);
            }
            fn on_text_chunk(&self, text: String) {
                self.text.lock().unwrap().push(text);
            }
            fn on_audio_frames(&self, pcm: Vec<f32>, sample_rate: u32) {
                self.audio.lock().unwrap().push((pcm, sample_rate));
            }
            fn on_done(&self, reason: FinishReason) {
                *self.done.lock().unwrap() = Some(reason);
            }
        }

        let vocab = vec![
            b"Hello".to_vec(), // 0
            b" ".to_vec(),     // 1
            b"world".to_vec(), // 2
            vec![0xF0, 0x9F],  // 3 (first half of wave emoji: F0 9F 91 8B)
            vec![0x91, 0x8B],  // 4 (second half of wave emoji)
        ];
        let tokenizer = Arc::new(cera::tokenizer::BpeTokenizer::from_vocab(vocab));

        let recorder: Arc<Recorder> = Arc::new(Recorder::default());
        let mut adapter =
            ForeignSinkAdapter::new(recorder.clone() as Arc<dyn ModalitySink>, tokenizer);

        // Drive the adapter as cera's decode loop would.
        adapter.on_text_tokens(&[0, 1]);
        adapter.on_text_tokens(&[2]);
        // Split UTF-8 sequence across chunks:
        adapter.on_text_tokens(&[3]); // incomplete UTF-8: should not emit yet
        assert_eq!(&*recorder.text.lock().unwrap(), &["Hello ", "world"]);
        adapter.on_text_tokens(&[4]); // completes wave emoji: should emit
        assert_eq!(&*recorder.text.lock().unwrap(), &["Hello ", "world", "👋"]);

        adapter.on_audio_frames(&[0.1, 0.2, 0.3], 24_000);
        adapter.on_done(cera::FinishReason::MaxTokens);
        adapter.notify_done(None);

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

    #[test]
    fn streaming_thinking_parser_splits_delimiters_across_chunks() {
        let mut parser = StreamingThinkingParser::new();

        // Chunk 1: regular text starting with partial open tag "<th"
        let out1 = parser.feed("Intro text. <th");
        assert_eq!(out1, vec![(false, "Intro text. ".to_string())]);

        // Chunk 2: completes "<think>" and begins thought
        let out2 = parser.feed("ink>Reasoning step 1. ");
        assert_eq!(out2, vec![(true, "Reasoning step 1. ".to_string())]);

        // Chunk 3: thought text with partial close tag "</th"
        let out3 = parser.feed("Reasoning step 2. </th");
        assert_eq!(out3, vec![(true, "Reasoning step 2. ".to_string())]);

        // Chunk 4: completes "</think>" and gives final answer
        let out4 = parser.feed("ink>Final answer.");
        assert_eq!(out4, vec![(false, "Final answer.".to_string())]);

        let flushed = parser.flush();
        assert!(flushed.is_none());
    }

    #[test]
    fn generate_summary_computes_throughput_stats() {
        let core = cera::GenerateSummary {
            tokens_generated: 100,
            prompt_eval_tokens: 50,
            prompt_eval_ms: 250,
            decode_ms: 1000,
            finish_reason: cera::FinishReason::Stop,
        };
        let ffi: GenerateSummary = core.into();
        assert_eq!(ffi.tokens_generated, 100);
        assert_eq!(ffi.prompt_eval_tokens, 50);
        assert_eq!(ffi.prompt_eval_ms, 250);
        assert_eq!(ffi.decode_ms, 1000);
        assert_eq!(ffi.total_duration_ms, 1250);
        assert!((ffi.decode_tok_per_sec - 100.0).abs() < 1e-4);
        assert!((ffi.prompt_eval_tok_per_sec - 200.0).abs() < 1e-4);
        assert!(matches!(ffi.finish_reason, FinishReason::Stop));
    }

    #[test]
    fn user_message_and_audio_input_defaults() {
        let msg = UserMessage {
            text: Some("Describe this part".to_string()),
            images: vec![vec![1, 2, 3]],
            audio: Some(AudioInput {
                pcm: vec![0.1, 0.2],
                sample_rate: 16000,
            }),
        };
        assert_eq!(msg.text.as_deref(), Some("Describe this part"));
        assert_eq!(msg.images.len(), 1);
        let core: cera::tokenizer::UserMessage = msg.clone().into();
        assert_eq!(core.text.as_deref(), Some("Describe this part"));
        assert_eq!(core.images.len(), 1);
        let audio = msg.audio.unwrap();
        assert_eq!(audio.sample_rate, 16000);
        assert_eq!(audio.pcm.len(), 2);
        let core_audio: cera::tokenizer::AudioInput = audio.into();
        assert_eq!(core_audio.sample_rate, 16000);
        assert_eq!(core_audio.pcm.len(), 2);
    }

    /// Before `adapter.on_done` has been forwarded, `done_called`
    /// stays `false`. Protects the error-synthesis branch in
    /// `Session::generate_streaming`: we can only safely synthesize
    /// a terminal `on_done(Error)` on failure when the inner
    /// `cera::Session::generate` hasn't already fired its own `on_done`.
    #[test]
    fn adapter_done_called_starts_false_and_guards_error_synthesis() {
        use std::sync::Mutex;

        #[derive(Default)]
        struct Recorder {
            calls: Mutex<usize>,
        }
        impl ModalitySink for Recorder {
            fn on_thought_chunk(&self, _: String) {}
            fn on_text_chunk(&self, _: String) {}
            fn on_audio_frames(&self, _: Vec<f32>, _: u32) {}
            fn on_done(&self, _: FinishReason) {
                *self.calls.lock().unwrap() += 1;
            }
        }

        let tokenizer = Arc::new(cera::tokenizer::BpeTokenizer::from_vocab(vec![]));
        let recorder: Arc<Recorder> = Arc::new(Recorder::default());
        let adapter = ForeignSinkAdapter::new(recorder.clone() as Arc<dyn ModalitySink>, tokenizer);

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
        let tokenizer2 = Arc::new(cera::tokenizer::BpeTokenizer::from_vocab(vec![]));
        let mut adapter_already_done =
            ForeignSinkAdapter::new(recorder.clone() as Arc<dyn ModalitySink>, tokenizer2);
        adapter_already_done.done_called = true;
        if !adapter_already_done.done_called {
            adapter_already_done
                .inner
                .on_done(FinishReason::Error { message: "".into() });
        }
        assert_eq!(
            *recorder.calls.lock().unwrap(),
            1,
            "still one: no double-fire"
        );
    }

    /// Mirrors the exact `if armed { ... }` branch in
    /// `AsyncCancelGuard::drop`. Can't build a real `cera::Session`
    /// in a unit test (no model to load), but the guard's logic is
    /// one conditional — a structurally identical probe guard gives
    /// the same coverage. End-to-end verification of
    /// "drop-future-cancels-decode" requires a real model and lands
    /// with PR 6's binding smoke tests / PR 7+'s parity harness.
    #[test]
    fn async_cancel_guard_drop_fires_when_armed_only() {
        use std::sync::atomic::{AtomicBool, Ordering};

        struct ProbeGuard {
            fired: Arc<AtomicBool>,
            armed: bool,
        }
        impl Drop for ProbeGuard {
            fn drop(&mut self) {
                if self.armed {
                    self.fired.store(true, Ordering::Relaxed);
                }
            }
        }

        // Armed drop → fires.
        let armed_fired = Arc::new(AtomicBool::new(false));
        drop(ProbeGuard {
            fired: armed_fired.clone(),
            armed: true,
        });
        assert!(armed_fired.load(Ordering::Relaxed), "armed drop must fire");

        // Disarmed drop → does not fire (the await-resolved path).
        let disarmed_fired = Arc::new(AtomicBool::new(false));
        let mut g = ProbeGuard {
            fired: disarmed_fired.clone(),
            armed: true,
        };
        g.armed = false;
        drop(g);
        assert!(
            !disarmed_fired.load(Ordering::Relaxed),
            "disarmed drop must not fire"
        );
    }

    /// Replicates the `spawn_blocking(..).await.map_err(..)?` pattern
    /// used by `generate_async` / `generate_streaming_async` — the
    /// wrapper's entire logic. Proves:
    ///
    /// - Successful blocking-closure results propagate through the
    ///   await + ?-sugar unchanged.
    /// - A panic in the blocking closure surfaces as
    ///   `tokio::task::JoinError`, which our `map_err` folds into
    ///   `FfiError::Backend` with the documented prefix.
    ///
    /// Can't construct a real `Session` in a unit test (no model to
    /// load), so the actual async wrappers are exercised only via
    /// this shape-equivalent stand-in. The binding generation step
    /// (PR 6) and the parity harness (PR 7+) will end-to-end exercise
    /// the real methods.
    #[tokio::test]
    async fn spawn_blocking_pattern_propagates_ok_and_maps_join_error() {
        // Ok path: same shape as `generate_async`'s body — the sync
        // closure returns the final value, spawn_blocking + await +
        // map_err hands it back via `?`.
        let map_join = |e: tokio::task::JoinError| FfiError::Backend {
            detail: format!("test join error: {e}"),
        };

        let ok: u32 = tokio::task::spawn_blocking(|| 42u32)
            .await
            .map_err(map_join)
            .expect("tokio should not drop the blocking task");
        assert_eq!(ok, 42);

        // Panic path: the blocking closure panics, JoinError bubbles
        // out, map_err converts it. No `?` here so we can inspect the
        // error variant.
        let panicked = tokio::task::spawn_blocking(|| -> u32 {
            panic!("simulated decode panic");
        })
        .await
        .map_err(map_join);
        match panicked {
            Err(FfiError::Backend { detail }) => {
                assert!(
                    detail.contains("test join error"),
                    "expected prefix, got: {detail}"
                );
            }
            other => panic!("expected Err(Backend), got: {other:?}"),
        }
    }

    #[test]
    fn tool_def_json_marshals_to_core() {
        let ffi = ToolDef {
            name: "get_weather".into(),
            description: Some("weather".into()),
            parameters_json: r#"{"type":"object","properties":{"city":{"type":"string"}}}"#.into(),
        };
        let core: cera::tools::ToolDef = ffi.try_into().expect("valid schema");
        assert_eq!(core.name, "get_weather");
        assert_eq!(core.parameters["properties"]["city"]["type"], "string");

        // Empty parameters_json → default empty object schema.
        let bare = ToolDef {
            name: "ping".into(),
            description: None,
            parameters_json: String::new(),
        };
        let core: cera::tools::ToolDef = bare.try_into().unwrap();
        assert_eq!(core.parameters["type"], "object");

        // Malformed schema → error, not panic.
        let bad = ToolDef {
            name: "x".into(),
            description: None,
            parameters_json: "{not json".into(),
        };
        let core: Result<cera::tools::ToolDef, _> = bad.try_into();
        assert!(core.is_err());
    }

    #[test]
    fn tool_def_rejects_non_object_parameters() {
        // A scalar/array schema must error rather than silently yield zero
        // constraints (and risk breaking the chat template).
        for bad in ["[]", "42", "\"x\"", "true", "null"] {
            let d = ToolDef {
                name: "f".into(),
                description: None,
                parameters_json: bad.into(),
            };
            let r: Result<cera::tools::ToolDef, _> = d.try_into();
            assert!(r.is_err(), "parameters_json={bad:?} should be rejected");
        }
    }

    #[test]
    fn parse_tool_calls_ffi_returns_json_args() {
        let calls = parse_tool_calls(
            "<|tool_call_start|>[get_weather(city=\"Paris\")]<|tool_call_end|>".into(),
            ToolFormat::Lfm2Pythonic,
        )
        .expect("parse");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        let args: serde_json::Value = serde_json::from_str(&calls[0].arguments_json).unwrap();
        assert_eq!(args["city"], "Paris");
    }

    #[test]
    fn tool_grammar_ffi_compiles() {
        let tools = vec![ToolDef {
            name: "get_weather".into(),
            description: None,
            parameters_json: r#"{"type":"object","properties":{"city":{"type":"string"}}}"#.into(),
        }];
        let gbnf = tool_grammar(tools, ToolFormat::Lfm2Pythonic).expect("grammar");
        assert!(cera::grammar::Grammar::parse(&gbnf).is_ok());
    }

    #[test]
    fn detect_tool_format_ffi() {
        assert_eq!(
            detect_tool_format("lfm2".into()),
            Some(ToolFormat::Lfm2Pythonic)
        );
        assert_eq!(detect_tool_format("qwen3".into()), Some(ToolFormat::Hermes));
        assert_eq!(detect_tool_format("gpt2".into()), None);
    }

    #[test]
    fn vad_config_and_sample_rate_roundtrip() {
        let def = silero_vad_default_config();
        assert_eq!(def.threshold, 0.5);
        assert_eq!(def.neg_threshold, 0.35);
        assert_eq!(def.min_speech_duration_ms, 64);
        assert_eq!(def.min_silence_duration_ms, 100);
        assert_eq!(def.speech_pad_ms, 30);

        let core_cfg: cera::vad::VadConfig = def.clone().into();
        assert_eq!(core_cfg.threshold, def.threshold);

        let r16 = FfiVadSampleRate::Rate16kHz;
        let core_r16: cera::vad::VadSampleRate = r16.into();
        assert_eq!(core_r16, cera::vad::VadSampleRate::Rate16kHz);
        let back_r16: FfiVadSampleRate = core_r16.into();
        assert_eq!(back_r16, r16);

        let r8 = FfiVadSampleRate::Rate8kHz;
        let core_r8: cera::vad::VadSampleRate = r8.into();
        assert_eq!(core_r8, cera::vad::VadSampleRate::Rate8kHz);
        let back_r8: FfiVadSampleRate = core_r8.into();
        assert_eq!(back_r8, r8);
    }

    #[test]
    fn generate_opts_defaults_match_core() {
        let opts = GenerateOpts::default();
        let core = cera::GenerateOpts::default();
        assert_eq!(opts.temperature, core.temperature);
        assert_eq!(opts.top_p, core.top_p);
        assert_eq!(opts.top_k, core.top_k);
        assert_eq!(opts.min_p, core.min_p);
        assert_eq!(opts.repetition_penalty, core.repetition_penalty);
    }

    #[test]
    fn entity_span_ffi_conversion() {
        let span = cera::EntitySpan {
            entity_type: "NAME".into(),
            start_char: 5,
            end_char: 15,
            start_token: 2,
            end_token: 4,
            text: "Alice Smith".into(),
            score: 0.98,
        };
        let ffi_span: FfiEntitySpan = span.clone().into();
        assert_eq!(ffi_span.entity_type, span.entity_type);
        assert_eq!(ffi_span.start_char, 5);
        assert_eq!(ffi_span.end_char, 15);
        assert_eq!(ffi_span.start_token, 2);
        assert_eq!(ffi_span.end_token, 4);
        assert_eq!(ffi_span.text, "Alice Smith");
        assert_eq!(ffi_span.score, 0.98);
    }

    #[test]
    fn hotword_ffi_conversions() {
        let def = hotword_default_config();
        assert_eq!(def.threshold, 0.75);
        assert_eq!(def.cooldown_ms, 2000);
        assert_eq!(def.step_ms, 80);
        assert_eq!(def.window_ms, 1200);
        assert_eq!(def.pre_roll_ms, 150);
        assert_eq!(def.vad_threshold, 0.5);

        let core_cfg: cera::HotwordConfig = def.clone().into();
        assert_eq!(core_cfg.threshold, def.threshold);
        assert_eq!(core_cfg.cooldown_ms, def.cooldown_ms as usize);
        assert_eq!(core_cfg.step_ms, def.step_ms as usize);
        assert_eq!(core_cfg.window_ms, def.window_ms as usize);
        assert_eq!(core_cfg.pre_roll_ms, def.pre_roll_ms as usize);

        let event = cera::HotwordEvent {
            keyword: "Hey Liquid".into(),
            sample_offset: 19200,
            command_start_sample: 16800,
            timestamp_ms: 1200.0,
            confidence: 0.95,
        };
        let ffi_event: FfiHotwordEvent = event.clone().into();
        assert_eq!(ffi_event.keyword, "Hey Liquid");
        assert_eq!(ffi_event.sample_offset, 19200);
        assert_eq!(ffi_event.command_start_sample, 16800);
        assert_eq!(ffi_event.timestamp_ms, 1200.0);
        assert_eq!(ffi_event.confidence, 0.95);
    }

    #[test]
    fn whisper_ffi_conversions() {
        let def = whisper_default_transcribe_opts();
        assert_eq!(def.language, None);
        assert!(!def.translate);
        assert!(!def.timestamps);
        assert_eq!(def.max_tokens, Some(448));
        assert_eq!(def.temperature, Some(0.0));

        let custom = FfiWhisperTranscribeOpts {
            language: Some("es".to_string()),
            translate: true,
            timestamps: true,
            max_tokens: Some(256),
            temperature: Some(0.2),
        };
        let core_opts: cera::WhisperTranscribeOpts = custom.clone().into();
        assert_eq!(core_opts.language.as_deref(), Some("es"));
        assert!(core_opts.translate);
        assert!(core_opts.timestamps);
        assert_eq!(core_opts.max_tokens, 256);
        assert_eq!(core_opts.temperature, 0.2);

        let roundtrip: FfiWhisperTranscribeOpts = core_opts.into();
        assert_eq!(roundtrip, custom);
    }
}
