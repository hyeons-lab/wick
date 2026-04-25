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
//! - [`WickEngine`] — model loader, session factory, and tokenizer
//!   accessor. Constructors: [`WickEngine::from_path`] for a local
//!   GGUF, manifest, or directory; [`WickEngine::from_bundle_id`]
//!   for LeapBundles-style remote loading;
//!   [`WickEngine::from_bundle_id_async`] for the tokio-async variant.
//!   Tokenizer methods ([`WickEngine::encode_text`],
//!   [`WickEngine::decode_tokens`],
//!   [`WickEngine::apply_chat_template`]) let foreign callers
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
//! Error:
//! - [`FfiError`] — typed error surface mirroring [`wick::WickError`]
//!   one-to-one (`ContextOverflow { max_seq_len, by }`,
//!   `UnsupportedModality`, `UnsupportedInferenceType`, `Busy`,
//!   `Cancelled`, `EmptyInput`, `Io`), plus `Backend` for FFI-internal
//!   errors that have no wick analog (poisoned mutex, `JoinError`).
//!   The `From<WickError>` conversion is exhaustive — new wick
//!   variants break compilation, never silently fall through to
//!   `Backend`.
//!
//! ## Not exposed yet
//!
//! Future PRs grow the surface per the roadmap in
//! `wick-ffi/README.md`. Highlights: remote URL loading through
//! `BundleRepo` (gated on the `remote` feature) and a parity harness
//! crate that cross-checks `wick-ffi` output against a reference
//! implementation.
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

/// Typed error surface for `wick-ffi`. Mirrors [`wick::WickError`] one-
/// to-one so foreign callers can pattern-match on error class (Kotlin
/// `when`, Swift `switch`, Python `match`) instead of string-sniffing
/// a generic message.
///
/// `Backend` is **not** a silent fallback for unmapped `wick::WickError`
/// variants — the `From<WickError>` impl is exhaustive, so adding a
/// new wick variant breaks compilation here. `Backend` exists solely
/// for FFI-internal errors that have no wick analog: `JoinError` from
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
/// `#[error(...)]` format strings match `wick::WickError` exactly for
/// every shared variant, so `Display` output is identical whether the
/// error originates from wick directly or routes through the FFI
/// wrapper. Pinned by `ffi_error_display_matches_wick_error_for_every_shared_variant`.
#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum FfiError {
    /// The loaded model doesn't support the modality the caller
    /// requested (e.g. `append_audio` on a text-only LLM).
    #[error("modality not supported by this model")]
    UnsupportedModality,

    /// The manifest's `inference_type` is one wick doesn't recognize
    /// at this version. Field carries the offending string.
    #[error("inference_type `{inference_type}` is not supported in this version of wick")]
    UnsupportedInferenceType { inference_type: String },

    /// A concurrent `generate*` call is already in flight on this
    /// session. Rust side guards with a mutex; this surfaces when the
    /// FFI detects contention.
    #[error("session is busy with another operation")]
    Busy,

    /// The caller (or the cancel-on-drop guard) flipped the cancel
    /// atomic. Currently wick's `generate` returns this as a
    /// `FinishReason::Cancelled` success rather than an `Err`, but
    /// the variant exists so a future `append_tokens` cancel (or
    /// similar) can surface typed.
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

    /// Filesystem / mmap / network error surfaced from wick. The
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
    /// Format string matches `wick::WickError::Io`'s `"io: {0}"` so
    /// foreign `.toString()` / `String(describing:)` gives the same
    /// output Rust consumers see.
    #[error("io: {detail}")]
    Io { detail: String },

    /// FFI-internal error with no wick analog: `JoinError` from a
    /// panicking `spawn_blocking` task, poisoned `Session::inner`
    /// mutex, 32-bit `u64 → usize` overflow in `EngineConfig::try_from`,
    /// or `wick::WickError::Backend` routed through the `From` impl.
    /// Format string matches `wick::WickError::Backend`'s
    /// `"backend: {0}"` — FFI-internal constructors that have already
    /// formatted a descriptive message (e.g. "generate_async join
    /// error: ...") still read cleanly with the `backend:` label.
    ///
    /// Field is named `detail` rather than `message` for the same
    /// `Throwable.message` collision reason as [`FfiError::Io`].
    #[error("backend: {detail}")]
    Backend { detail: String },
}

impl From<wick::WickError> for FfiError {
    fn from(e: wick::WickError) -> Self {
        // Match exhaustively on the upstream enum so a future wick
        // variant-add breaks compilation here loudly rather than
        // silently routing through the `Backend` catch-all.
        match e {
            wick::WickError::UnsupportedModality => FfiError::UnsupportedModality,
            wick::WickError::UnsupportedInferenceType(s) => {
                FfiError::UnsupportedInferenceType { inference_type: s }
            }
            wick::WickError::Busy => FfiError::Busy,
            wick::WickError::Cancelled => FfiError::Cancelled,
            wick::WickError::ContextOverflow { max_seq_len, by } => {
                FfiError::ContextOverflow { max_seq_len, by }
            }
            wick::WickError::EmptyInput => FfiError::EmptyInput,
            wick::WickError::Backend(s) => FfiError::Backend { detail: s },
            wick::WickError::Io(io_err) => FfiError::Io {
                detail: io_err.to_string(),
            },
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
    /// Bundle repository for resolving `http(s)://` URLs in manifests
    /// (or for [`WickEngine::from_bundle_id`]). `None` means "remote
    /// URLs will fail with an error"; set this to a [`BundleRepo`]
    /// rooted at a persistent cache directory to enable remote
    /// downloads. Construct the repo once + reuse it across engine
    /// loads so its HTTP client pool + on-disk cache are shared.
    pub bundle_repo: Option<Arc<BundleRepo>>,
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
            // `bundle_repo` defaults to None; foreign callers who want
            // remote-URL loading set it explicitly before passing the
            // config to `WickEngine::from_path` / `from_bundle_id`.
            bundle_repo: None,
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
                detail: format!(
                    "context_size {} exceeds usize::MAX on this target",
                    c.context_size
                ),
            })?
        };
        // Under the `remote` feature `wick::EngineConfig` carries a
        // `bundle_repo: Option<wick::bundle::BundleRepo>` field. Pull
        // the inner from our FFI `Arc<BundleRepo>` wrapper (cheap —
        // `wick::bundle::BundleRepo` is `Clone` and the two reqwest
        // clients inside share their connection pool via Arc-backed
        // refcounts).
        Ok(wick::EngineConfig {
            context_size,
            backend: c.backend.into(),
            bundle_repo: c.bundle_repo.map(|r| r.inner.clone()),
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
// ChatMessage (PR 13 — chat template input)
// ---------------------------------------------------------------------------

/// One message in a chat-template conversation. Mirrors
/// [`wick::tokenizer::ChatMessage`]. Pass a `Vec<ChatMessage>` to
/// [`WickEngine::apply_chat_template`] to render the model's
/// chat-template (Jinja2 from GGUF metadata) into a prompt string
/// ready to feed into [`Session::append_text`].
///
/// `role` follows the OpenAI / chat-template convention — typically
/// one of `"system"`, `"user"`, `"assistant"`, occasionally
/// `"tool"`. wick-ffi doesn't validate the role string; whatever is
/// passed flows directly into the Jinja template. Whether an
/// unknown role errors or silently no-ops depends on the template's
/// own logic — many templates have an explicit error path for
/// unrecognized roles, but it's template-dependent rather than
/// enforced by [`WickEngine::apply_chat_template`].
#[derive(Debug, Clone, uniffi::Record)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl From<ChatMessage> for wick::tokenizer::ChatMessage {
    fn from(m: ChatMessage) -> Self {
        wick::tokenizer::ChatMessage {
            role: m.role,
            content: m.content,
        }
    }
}

// ---------------------------------------------------------------------------
// BundleRepo
// ---------------------------------------------------------------------------

/// Remote model-bundle downloader + on-disk cache. Wraps
/// [`wick::bundle::BundleRepo`]; construct once per application with
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
/// wick-powered apps on the same device can share the same cache
/// directory without conflicting.
#[derive(Debug, uniffi::Object)]
pub struct BundleRepo {
    inner: wick::bundle::BundleRepo,
}

#[uniffi::export]
impl BundleRepo {
    /// Create a new repo rooted at `store_dir`. The directory doesn't
    /// need to exist yet — it's created on the first download. Pass
    /// the same path to subsequent runs to reuse the cached bundles.
    #[uniffi::constructor]
    pub fn new(store_dir: String) -> Arc<Self> {
        Arc::new(Self {
            inner: wick::bundle::BundleRepo::new(store_dir),
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
        let adapter: Arc<dyn wick::bundle::DownloadProgress> =
            Arc::new(DownloadProgressAdapter { inner: progress });
        Arc::new(Self {
            inner: wick::bundle::BundleRepo::with_progress(store_dir, adapter),
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
    /// Throttled by `wick-core` to ~256 KB granularity + one final
    /// callback at end-of-stream so the consumer always sees the
    /// final byte count.
    fn on_progress(&self, url: String, bytes_downloaded: u64, total_bytes: Option<u64>);
}

/// Adapter from the UniFFI foreign trait to wick-core's
/// [`wick::bundle::DownloadProgress`]. Same shape as
/// [`ForeignSinkAdapter`] for `ModalitySink` (PR 4): the foreign
/// arg's `&str` becomes an owned `String` because UniFFI can't
/// marshal a borrowed slice across the boundary.
struct DownloadProgressAdapter {
    inner: Arc<dyn DownloadProgressSink>,
}

// Manual Debug impl because `dyn DownloadProgressSink` is a UniFFI
// foreign-trait object — the foreign side (Kotlin / Swift / Python
// implementations) has no Rust Debug, so we can't blanket-derive.
// `wick::bundle::DownloadProgress` requires Debug for `BundleRepo`'s
// own derived Debug to work; printing the adapter as a typed handle
// is sufficient for any Rust-side log line that touches it.
impl std::fmt::Debug for DownloadProgressAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DownloadProgressAdapter")
            .field("inner", &"<foreign DownloadProgressSink>")
            .finish()
    }
}

impl wick::bundle::DownloadProgress for DownloadProgressAdapter {
    fn on_progress(&self, url: &str, bytes_downloaded: u64, total_bytes: Option<u64>) {
        self.inner
            .on_progress(url.to_string(), bytes_downloaded, total_bytes);
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
    /// inputs as the native [`wick::WickEngine::from_path`]: a bare
    /// `.gguf`, a LeapBundles `.json` manifest, or a directory
    /// containing exactly one `.json` manifest.
    ///
    /// If the manifest carries `http(s)://` URLs for its files,
    /// `config.bundle_repo` must be set — otherwise those URLs fail
    /// to resolve. For a pure-local workflow (bundle already on
    /// disk) leave `bundle_repo = None`.
    #[uniffi::constructor]
    pub fn from_path(path: String, config: EngineConfig) -> Result<Arc<Self>, FfiError> {
        let inner = wick::WickEngine::from_path(&path, config.try_into()?)?;
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
        let inner = wick::WickEngine::from_bundle_id(&bundle_id, &quant, config.try_into()?)?;
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

    // ----- Tokenizer surface (PR 13) ---------------------------------
    //
    // Wraps `wick::tokenizer::BpeTokenizer` so foreign callers can
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

    /// `true` if the model's tokenizer carries a chat template (a
    /// minijinja string from GGUF metadata). Foreign callers should
    /// check this before calling [`WickEngine::apply_chat_template`].
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
    /// template (check [`WickEngine::has_chat_template`] first) or
    /// if the template fails to render against the supplied messages.
    pub fn apply_chat_template(
        &self,
        messages: Vec<ChatMessage>,
        add_generation_prompt: bool,
    ) -> Result<String, FfiError> {
        let core_messages: Vec<wick::tokenizer::ChatMessage> =
            messages.into_iter().map(Into::into).collect();
        wick::tokenizer::apply_chat_template(
            self.inner.tokenizer(),
            &core_messages,
            add_generation_prompt,
        )
        .map_err(|e| FfiError::Backend {
            detail: format!("apply_chat_template: {e}"),
        })
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
            detail: format!(
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
                    // `FinishReason::Error` only carries a message string,
                    // so flatten whichever typed FfiError variant via
                    // Display. Foreign callers still receive the full
                    // typed error from the return value; the sink's
                    // on_done(Error) is a best-effort terminal signal.
                    adapter.inner.on_done(FinishReason::Error {
                        message: err.to_string(),
                    });
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
/// 1. **Running decode.** The task is executing `wick::Session::generate`
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
///    executes, so `wick::Session::generate` never starts, so the
///    session's cancel flag is never reset. Without this, the race is:
///    guard sets cancel → task eventually dequeues → decode's first
///    line clears cancel back to `false` (`wick/src/session.rs:603-605`)
///    → decode runs to completion despite the caller having dropped
///    the future.
///
/// Both operations are idempotent / harmless on the irrelevant path:
/// `session.cancel()` on a queued task is overridden by `abort`; an
/// already-completed task ignores both. `wick::Session::generate`
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
/// signal. Used by [`WickEngine::from_bundle_id_async`]: the underlying
/// `wick::WickEngine::from_bundle_id` holds no cross-thread cancel flag
/// and the `reqwest::blocking` download can't be cooperatively
/// cancelled, so there's nothing like `Session::cancel` to call on
/// drop. All we can do is abort the queued task before its closure
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
    /// without dropping the future. See [`AsyncCancelGuard`] for the
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
    /// (see [`AsyncCancelGuard`]). For an in-flight decode, the loop
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

// Async WickEngine constructors (PR 11).
#[uniffi::export(async_runtime = "tokio")]
impl WickEngine {
    /// Async variant of [`WickEngine::from_bundle_id`] — offloads the
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
    /// dropping the returned future drops the [`AbortOnDrop`] guard,
    /// which calls `AbortHandle::abort` on the spawned task. That
    /// cancels the task if it's still queued on tokio's blocking
    /// pool, so a not-yet-started download never runs. But if the
    /// task has started, abort is a no-op — the download is a
    /// `reqwest::blocking` call with no cooperative cancel point,
    /// and wick's engine-construction code (tokenizer build, model
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
        let wick_config: wick::EngineConfig = config.try_into()?;
        let handle = tokio::task::spawn_blocking(move || {
            wick::WickEngine::from_bundle_id(&bundle_id, &quant, wick_config)
                .map_err(FfiError::from)
        });
        let mut guard = AbortOnDrop {
            abort: handle.abort_handle(),
            armed: true,
        };
        let join_result = handle.await;
        guard.armed = false;
        join_result
            .map_err(|e| FfiError::Backend {
                detail: format!("from_bundle_id_async join error: {e}"),
            })?
            .map(|inner| Arc::new(Self { inner }))
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
        // bundle_repo defaults to None — foreign callers opt in by
        // attaching a BundleRepo before the try_into.
        assert!(core.bundle_repo.is_none());
    }

    /// `ChatMessage` round-trips its `role` + `content` fields
    /// through the wick-core conversion. `From<ChatMessage> for
    /// wick::tokenizer::ChatMessage` is a trivial field-copy; this
    /// test pins the field shape so a future wick-core rename
    /// breaks compilation here loudly instead of silently dropping
    /// data on the FFI boundary.
    #[test]
    fn chat_message_converts_to_wick_core() {
        let m = ChatMessage {
            role: "user".to_string(),
            content: "hello".to_string(),
        };
        let core: wick::tokenizer::ChatMessage = m.clone().into();
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
            "wick-ffi-test-{}-{}",
            test_name,
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        path
    }

    /// `cache_size` round-trips through the FFI wrapper. Builds a
    /// small synthetic cache, verifies the FFI method returns the
    /// same byte count as the wick-core method (which has its own
    /// unit-test coverage in `wick/src/bundle/mod.rs`).
    #[test]
    fn bundle_repo_cache_size_forwards_to_wick_core() {
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

    /// `BundleRepo::new` wraps a `wick::bundle::BundleRepo` without
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
    /// `wick::EngineConfig::bundle_repo` — proves the From/TryFrom
    /// conversion handles the Arc<BundleRepo> → Option<wick::BundleRepo>
    /// path correctly.
    #[test]
    fn engine_config_carries_bundle_repo_through_try_from() {
        let dir = unique_test_bundle_dir("carry");
        let repo = BundleRepo::new(dir.to_string_lossy().into_owned());
        let ffi = EngineConfig {
            context_size: 0,
            backend: BackendPreference::Cpu,
            bundle_repo: Some(repo.clone()),
        };
        let core: wick::EngineConfig = ffi.try_into().unwrap();
        let core_repo = core.bundle_repo.expect("bundle_repo must be Some");
        assert_eq!(core_repo.store_dir(), repo.inner.store_dir());
    }

    /// `DownloadProgressAdapter` forwards `on_progress` calls from
    /// wick-core to a foreign-trait implementation (impl'd here as a
    /// recording Rust struct that mirrors how UniFFI codegens the
    /// foreign side). Verifies the URL / bytes / total round-trip
    /// without dropping data.
    #[test]
    fn download_progress_adapter_forwards() {
        use std::sync::Mutex;
        use wick::bundle::DownloadProgress as _;

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

        // Drive the adapter as wick-core's download_to would.
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
        // translate to `usize::MAX` so wick caps at model.max_seq_len.
        let ffi = EngineConfig {
            context_size: 0,
            backend: BackendPreference::Cpu,
            bundle_repo: None,
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
            bundle_repo: None,
        };
        let result: Result<wick::EngineConfig, FfiError> = ffi.try_into();
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
            let core: wick::BackendPreference = ffi.into();
            let back: BackendPreference = core.into();
            assert_eq!(ffi, back, "{ffi:?} didn't round-trip");
        }
    }

    /// Every `wick::WickError` variant maps to a specific `FfiError`
    /// variant (not the generic `Backend` catch-all) so foreign
    /// callers can pattern-match on class. If wick adds a new
    /// `WickError` variant and forgets to update `From<WickError>`,
    /// the exhaustive match in that impl breaks compilation loudly —
    /// this test just asserts the existing mapping is correct.
    #[test]
    fn wick_error_variants_map_to_typed_ffi_error_variants() {
        // Payload-free variants.
        assert!(matches!(
            FfiError::from(wick::WickError::UnsupportedModality),
            FfiError::UnsupportedModality
        ));
        assert!(matches!(
            FfiError::from(wick::WickError::Busy),
            FfiError::Busy
        ));
        assert!(matches!(
            FfiError::from(wick::WickError::Cancelled),
            FfiError::Cancelled
        ));
        assert!(matches!(
            FfiError::from(wick::WickError::EmptyInput),
            FfiError::EmptyInput
        ));

        // Payload-carrying variants preserve their fields.
        match FfiError::from(wick::WickError::UnsupportedInferenceType(
            "audio-magic".into(),
        )) {
            FfiError::UnsupportedInferenceType { inference_type } => {
                assert_eq!(inference_type, "audio-magic");
            }
            other => panic!("expected UnsupportedInferenceType, got: {other:?}"),
        }

        match FfiError::from(wick::WickError::ContextOverflow {
            max_seq_len: 4096,
            by: 17,
        }) {
            FfiError::ContextOverflow { max_seq_len, by } => {
                assert_eq!(max_seq_len, 4096);
                assert_eq!(by, 17);
            }
            other => panic!("expected ContextOverflow, got: {other:?}"),
        }

        match FfiError::from(wick::WickError::Backend("metal driver crashed".into())) {
            FfiError::Backend { detail } => {
                assert_eq!(detail, "metal driver crashed");
            }
            other => panic!("expected Backend, got: {other:?}"),
        }

        // Io flattens the OS error to a string.
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "no such file");
        let io_str = io_err.to_string();
        match FfiError::from(wick::WickError::Io(io_err)) {
            FfiError::Io { detail } => {
                assert_eq!(detail, io_str);
            }
            other => panic!("expected Io, got: {other:?}"),
        }
    }

    /// Display (`thiserror`-derived) produces byte-identical message
    /// text to the equivalent `wick::WickError` for every variant
    /// that has a wick analog. Foreign callers logging the error via
    /// `.toString()` / `String(describing:)` / `str()` see the same
    /// output whether the error originates from wick directly or
    /// routes through the FFI wrapper.
    ///
    /// Tests every shared variant including `Backend`, `Io`, and
    /// `UnsupportedInferenceType` — an earlier iteration of this test
    /// quietly excluded them, which masked a real drift where the FFI
    /// side had dropped the `"backend: "` and `"io: "` label prefixes.
    #[test]
    fn ffi_error_display_matches_wick_error_for_every_shared_variant() {
        // Prep the Io pair outside the vec since io::Error isn't
        // `Clone`: we need to consume one into `WickError::Io` and
        // stash its pre-wrap display string for the FFI side.
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "no such file");
        let io_msg = io_err.to_string();

        let pairs: Vec<(FfiError, wick::WickError)> = vec![
            (
                FfiError::UnsupportedModality,
                wick::WickError::UnsupportedModality,
            ),
            (
                FfiError::UnsupportedInferenceType {
                    inference_type: "audio-magic".into(),
                },
                wick::WickError::UnsupportedInferenceType("audio-magic".into()),
            ),
            (FfiError::Busy, wick::WickError::Busy),
            (FfiError::Cancelled, wick::WickError::Cancelled),
            (
                FfiError::ContextOverflow {
                    max_seq_len: 2048,
                    by: 5,
                },
                wick::WickError::ContextOverflow {
                    max_seq_len: 2048,
                    by: 5,
                },
            ),
            (FfiError::EmptyInput, wick::WickError::EmptyInput),
            (
                FfiError::Backend {
                    detail: "metal driver crashed".into(),
                },
                wick::WickError::Backend("metal driver crashed".into()),
            ),
            (FfiError::Io { detail: io_msg }, wick::WickError::Io(io_err)),
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

    /// Mirrors the exact `if armed { ... }` branch in
    /// `AsyncCancelGuard::drop`. Can't build a real `wick::Session`
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
}
