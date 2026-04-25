//! `WickEngine` — the owning, loader-aware front door for the core crate.
//!
//! Previous versions exposed `wick::engine::generate()`, a one-shot helper
//! that owned model + tokenizer only for the duration of a single call.
//! That was retired in PR #27 (Phase 1.1) when `Session` became the
//! canonical stateful API. This module reclaims the `engine` name for
//! what the FFI / CLI / web demos all actually need: a handle that owns
//! the loaded model + tokenizer + manifest for a process's lifetime and
//! hands out cheap `Session<'_>` instances.
//!
//! `from_path` accepts three shapes — a bare `.gguf` (synthesized text
//! manifest), a `.json` LeapBundles manifest, or a directory containing
//! exactly one `.json` manifest. All three converge on an internal
//! `from_manifest` routine that dispatches on `InferenceType`. For
//! callers who have explicit file paths and don't want to fabricate a
//! manifest, `from_files(ModelFiles, cfg)` is the overload.
//!
//! Scope notes (Phase 1.2):
//! - Text models: loaded via the existing CPU / wgpu / Metal paths,
//!   selected by [`BackendPreference`] on [`EngineConfig`].
//! - Audio models (`llama.cpp/lfm2-audio-v1`): the primary text LLM is
//!   loaded the same way as text; the audio decoder + detokenizer +
//!   safetensors tokenizer are not consumed by the engine itself —
//!   they're surfaced via [`WickEngine::manifest`] for callers (the
//!   CLI today) that drive `wick::audio_engine::generate_audio`
//!   directly. Unified `Session::append_audio` wiring lands in a
//!   follow-up.
//! - VL models (`llama.cpp/image-to-text`): parsed but rejected at load
//!   with [`WickError::UnsupportedInferenceType`]. The VL loader lands
//!   in a future phase.
//! - Remote manifests: `from_path` only resolves local paths in v1. A
//!   manifest whose `load_time_parameters.model` looks like an HTTP(S)
//!   URL is rejected with a typed error pointing at Phase 1.6's
//!   [`BundleRepo`] as the follow-up. Callers who already have the
//!   bundle on disk should point the manifest at the on-disk file.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::gguf::GgufFile;
use crate::kv_cache::KvCacheConfig;
#[cfg(feature = "mmap")]
use crate::manifest::ManifestFiles;
use crate::manifest::{InferenceType, Manifest};
use crate::model::{self, Model};
use crate::session::{ModalityCapabilities, Session, SessionConfig, WickError};
use crate::tokenizer::BpeTokenizer;

// ---------------------------------------------------------------------------
// Public configuration + metadata types
// ---------------------------------------------------------------------------

/// Which compute backend to use when loading a model.
///
/// `Auto` probes `metal → gpu → cpu` at load time with runtime fallback,
/// matching the existing CLI `--device auto` behavior. Explicit variants
/// error if their feature isn't compiled in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackendPreference {
    #[default]
    Auto,
    Cpu,
    /// `wgpu` (Vulkan / Metal / DX12). Requires the `gpu` feature.
    Gpu,
    /// Native Metal. Requires the `metal` feature + macOS.
    Metal,
}

impl BackendPreference {
    /// Parse a case-insensitive string (`"auto"`, `"cpu"`, `"gpu"`, `"wgpu"`, `"metal"`).
    /// Returns `Err` on an unknown label.
    pub fn parse_str(s: &str) -> Result<Self, WickError> {
        match s.to_ascii_lowercase().as_str() {
            "auto" | "" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "gpu" | "wgpu" => Ok(Self::Gpu),
            "metal" => Ok(Self::Metal),
            other => Err(WickError::Backend(format!(
                "unknown backend preference `{other}` (use auto, cpu, gpu, or metal)"
            ))),
        }
    }
}

/// Per-engine configuration. Set at `from_path` / `from_files` time;
/// immutable for the engine's lifetime.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// KV cache capacity in tokens. Capped by the model's own `max_seq_len`.
    pub context_size: usize,
    /// Which compute backend to prefer.
    pub backend: BackendPreference,
    /// Optional repository used to resolve `http(s)://` URLs found in a
    /// manifest's `files` entries. When `None`, remote URLs fail with a
    /// clear error asking the caller to either set this field or
    /// pre-download the bundle. Requires the `remote` feature.
    #[cfg(feature = "remote")]
    pub bundle_repo: Option<crate::bundle::BundleRepo>,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            context_size: 4096,
            backend: BackendPreference::Auto,
            #[cfg(feature = "remote")]
            bundle_repo: None,
        }
    }
}

/// Explicit file paths + metadata for `WickEngine::from_files`. Mirrors
/// [`ManifestFiles`] but with local paths instead of URLs and an optional
/// `inference_type` override (auto-detected from the GGUF header when
/// absent).
#[derive(Debug, Clone)]
pub struct ModelFiles {
    /// Required: primary GGUF path.
    pub model: PathBuf,
    /// Optional: multimodal projector GGUF (VL + audio models).
    pub multimodal_projector: Option<PathBuf>,
    /// Optional: audio-decoder GGUF (audio-out models).
    pub audio_decoder: Option<PathBuf>,
    /// Optional: audio tokenizer (usually a `.safetensors` checkpoint).
    pub audio_tokenizer: Option<PathBuf>,
    /// Forward-compat: any additional named aux file.
    pub extras: std::collections::HashMap<String, PathBuf>,
    /// Explicit inference type. `None` → auto-detect from GGUF
    /// `general.architecture` metadata + aux-file heuristic.
    pub inference_type: Option<InferenceType>,
    /// Optional chat-template override. If set, replaces any template
    /// embedded in the GGUF.
    pub chat_template: Option<String>,
}

impl ModelFiles {
    /// Convenience: construct a text-only `ModelFiles` from a single path.
    pub fn text(path: impl Into<PathBuf>) -> Self {
        Self {
            model: path.into(),
            multimodal_projector: None,
            audio_decoder: None,
            audio_tokenizer: None,
            extras: std::collections::HashMap::new(),
            inference_type: Some(InferenceType::LlamaCppTextToText),
            chat_template: None,
        }
    }
}

/// Short summary of the loaded model. Matches the shape planned for the
/// UniFFI `ModelMetadata` record so FFI bindings can surface it without
/// re-deriving.
#[derive(Debug, Clone)]
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

// ---------------------------------------------------------------------------
// WickEngine
// ---------------------------------------------------------------------------

/// Owning handle to a loaded model + tokenizer + manifest.
///
/// `model` and `tokenizer` are stored as `Arc` rather than `Box`/owned
/// so [`new_session`](Self::new_session) can hand out cheap
/// lifetime-free [`Session`] handles (see [`Session`]'s doc comment for
/// why the FFI story requires this).
pub struct WickEngine {
    manifest: Manifest,
    model: Arc<dyn Model>,
    tokenizer: Arc<BpeTokenizer>,
    metadata: ModelMetadata,
    config: EngineConfig,
}

impl WickEngine {
    /// Load from a path that may be:
    /// - a bare `.gguf` file → internally synthesizes a text manifest,
    /// - a `.json` LeapBundles manifest → parsed + dispatched on `inference_type`,
    /// - a directory → scanned for exactly one `.json` manifest.
    ///
    /// Requires both `std-fs` (for directory + manifest I/O) and `mmap`
    /// (to mmap-open the GGUF). Both are default-on. Builds without
    /// them (e.g. wasm32) should use [`Self::from_reader`] or
    /// [`Self::from_bytes`] with externally-sourced bytes.
    #[cfg(feature = "mmap")]
    pub fn from_path<P: AsRef<Path>>(path: P, cfg: EngineConfig) -> Result<Self, WickError> {
        let path = path.as_ref();
        if path.is_dir() {
            let manifest_path = find_single_manifest(path)?;
            Self::from_manifest_file(&manifest_path, cfg)
        } else if has_extension(path, "json") {
            Self::from_manifest_file(path, cfg)
        } else if has_extension(path, "gguf") {
            // Bare `.gguf` → peek at `general.architecture` and fail
            // early for known non-text arches (VL, audio) so the caller
            // gets the `UnsupportedInferenceType` error they'd see with
            // a manifest, not a mid-load "unsupported architecture"
            // failure from the text loader. Unknown arches fall back
            // to text (auto-detect's existing policy).
            let detected = auto_detect_inference_type(path)?;
            match detected {
                InferenceType::LlamaCppTextToText | InferenceType::LlamaCppLfm2AudioV1 => {
                    // Audio loads the same text LLM under the hood today
                    // (aux files are manifest-driven). Proceed with the
                    // synthetic text manifest; audio consumers who need
                    // the aux files must load via manifest or `from_files`.
                    let manifest = Manifest::synthetic_text(path);
                    Self::from_manifest(manifest, cfg)
                }
                InferenceType::LlamaCppImageToText => Err(WickError::UnsupportedInferenceType(
                    detected.as_str().to_string(),
                )),
                // `auto_detect_inference_type` defaults unknown arches to
                // Text, so this arm is unreachable today — matched for
                // exhaustiveness if the policy changes.
                InferenceType::Unknown(s) => Err(WickError::UnsupportedInferenceType(s)),
            }
        } else {
            Err(WickError::Backend(format!(
                "don't know how to load `{}` — expected a .gguf file, a .json manifest, or a directory containing one",
                path.display()
            )))
        }
    }

    /// Load from an in-memory byte buffer. Text-only; for multi-file loads
    /// (VL / audio) use [`Self::from_path`] with a manifest or
    /// [`Self::from_files`]. Documented as `<50 MB or testing only` —
    /// production paths should stream from disk.
    ///
    /// Unconditional — works in every feature configuration, including
    /// `--no-default-features`. Phase 3's `wick-wasm` crate uses this
    /// (plus [`Self::from_reader`]) to back its OPFS-loaded paths.
    pub fn from_bytes(bytes: impl Into<Arc<[u8]>>, cfg: EngineConfig) -> Result<Self, WickError> {
        let arc_bytes: Arc<[u8]> = bytes.into();
        let gguf = GgufFile::from_bytes(arc_bytes)
            .map_err(|e| WickError::Backend(format!("parsing GGUF bytes: {e}")))?;
        let manifest = Manifest::synthetic_text(Path::new("<bytes>"));
        Self::from_gguf(gguf, manifest, cfg, None)
    }

    /// Load from any `std::io::Read`. Streams the full GGUF into an
    /// owned buffer before parsing. Unconditional — works in every
    /// feature configuration.
    ///
    /// Intended backend for Phase 3's `wick-wasm` (an OPFS-backed
    /// `Read + Seek` shim) and for any consumer that has the bytes
    /// coming from a source other than a filesystem path (decrypted
    /// blob, network stream, archive entry).
    pub fn from_reader<R: Read>(reader: R, cfg: EngineConfig) -> Result<Self, WickError> {
        let gguf = GgufFile::from_reader(reader)
            .map_err(|e| WickError::Backend(format!("reading GGUF stream: {e}")))?;
        let manifest = Manifest::synthetic_text(Path::new("<reader>"));
        Self::from_gguf(gguf, manifest, cfg, None)
    }

    /// Load from explicit file paths — skips manifest JSON parsing.
    /// `files.inference_type` decides the loader; `None` auto-detects
    /// from the GGUF header.
    ///
    /// Requires the `mmap` feature (default-on) because it mmap-opens
    /// the primary file. Callers without `mmap` should read the file
    /// manually and use [`Self::from_reader`].
    #[cfg(feature = "mmap")]
    pub fn from_files(files: ModelFiles, cfg: EngineConfig) -> Result<Self, WickError> {
        let manifest = synthesize_manifest_from_files(&files)?;
        // If the caller overrode the chat template, apply it by threading
        // it through the manifest; the text loader doesn't need to know.
        // (The tokenizer will still be built from the GGUF; template
        // precedence lives on the manifest for downstream consumers.)
        Self::from_manifest_with_primary(manifest, files.model.as_path(), cfg)
    }

    /// Load from a LeapBundles ID + quantization selector, e.g.
    /// `from_bundle_id("LFM2-1.2B-GGUF", "Q4_0", cfg)`.
    ///
    /// Resolves to
    /// `https://huggingface.co/LiquidAI/LeapBundles/resolve/main/{bundle_id}/{quant}.json`,
    /// downloads + caches it via `cfg.bundle_repo`, then loads the
    /// engine through the normal manifest path — which in turn fetches
    /// the GGUF (also via `bundle_repo`) since the manifest's model URL
    /// is an `http(s)://` reference to the model's own HF repo.
    ///
    /// `cfg.bundle_repo` **must** be set; otherwise this returns an
    /// error telling the caller to set it. Requires both `remote` and
    /// `mmap` features.
    #[cfg(all(feature = "remote", feature = "mmap"))]
    pub fn from_bundle_id(
        bundle_id: &str,
        quant: &str,
        cfg: EngineConfig,
    ) -> Result<Self, WickError> {
        let repo = cfg.bundle_repo.as_ref().ok_or_else(|| {
            WickError::Backend(
                "`WickEngine::from_bundle_id` requires `EngineConfig::bundle_repo` to be set — \
                 construct a `BundleRepo` rooted at your desired store directory and assign it \
                 before calling this constructor."
                    .to_string(),
            )
        })?;
        let manifest_url = crate::bundle::leap_bundles_manifest_url(bundle_id, quant)?;
        // No caller-supplied hash for manifest JSONs (LeapBundles schema
        // doesn't carry one, and the file is tiny — etag fallback is
        // sufficient). Manifest-level per-file hashes, when they land,
        // would be threaded through from inside `from_manifest_file`.
        let manifest_path = repo.resolve_url(&manifest_url, None)?;
        Self::from_manifest_file(&manifest_path, cfg)
    }

    // --- internal constructors ---

    /// Core assembly: take a pre-constructed `GgufFile` + parsed
    /// manifest, build the tokenizer, load the model, wrap in
    /// `WickEngine`. All three public constructors funnel through here:
    ///
    /// - `from_bytes` / `from_reader` pass `path = None` — no on-disk
    ///   file to hand to backends.
    /// - `from_manifest_with_primary` (via `from_path` / `from_files`)
    ///   passes `Some(primary)` — Metal and wgpu's auto-dispatch may
    ///   reopen the file by path for their own mmap, so they need the
    ///   original filesystem path even though we also hand them the
    ///   already-parsed `GgufFile`.
    fn from_gguf(
        gguf: GgufFile,
        manifest: Manifest,
        cfg: EngineConfig,
        path: Option<&Path>,
    ) -> Result<Self, WickError> {
        // Covers `from_bytes` / `from_reader`, which skip the pre-filter
        // in `from_manifest_with_primary`. Text LLMs AND LFM2-audio
        // models both load the primary GGUF through the same path;
        // audio aux files (decoder, mmproj, safetensors tokenizer) stay
        // on the manifest for the audio pipeline to pick up separately.
        check_inference_type_supported(&manifest.inference_type)?;

        let tokenizer = BpeTokenizer::from_gguf(&gguf)
            .map_err(|e| WickError::Backend(format!("loading tokenizer: {e}")))?;
        let add_bos_token = gguf
            .get_bool("tokenizer.ggml.add_bos_token")
            .unwrap_or(false);
        // Extract `general.file_type` BEFORE `load_text_model` consumes
        // the gguf — that's the only place this metadata exists, and
        // we need it for the metadata's quantization label.
        let quantization = gguf
            .get_u32("general.file_type")
            .map(ftype_label)
            .unwrap_or_else(|| "unknown".to_string());
        // `load_text_model` returns `Box<dyn Model>`; convert to `Arc`
        // at the engine boundary. `Arc::from(Box<T>)` is documented on
        // `Arc` for exactly this sizing dance (including `T: ?Sized`).
        let model: Arc<dyn Model> = Arc::from(load_text_model(gguf, path, &cfg)?);
        let metadata = build_metadata(
            model.as_ref(),
            &tokenizer,
            &manifest,
            add_bos_token,
            quantization,
        );
        Ok(Self {
            manifest,
            model,
            tokenizer: Arc::new(tokenizer),
            metadata,
            config: cfg,
        })
    }

    #[cfg(feature = "mmap")]
    fn from_manifest_file(path: &Path, cfg: EngineConfig) -> Result<Self, WickError> {
        let mut manifest = Manifest::from_file(path).map_err(|e| {
            WickError::Backend(format!("parsing manifest `{}`: {e}", path.display()))
        })?;
        resolve_all_manifest_files(&mut manifest, path.parent(), &cfg)?;
        let primary = PathBuf::from(&manifest.files.model);
        Self::from_manifest_with_primary(manifest, &primary, cfg)
    }

    /// Opens the primary GGUF at `primary` and delegates assembly to
    /// [`Self::from_gguf`] with `Some(primary)` so Metal/GPU backends
    /// can reach the on-disk file.
    ///
    /// Requires `mmap` because it opens the primary via `GgufFile::open`.
    #[cfg(feature = "mmap")]
    fn from_manifest_with_primary(
        manifest: Manifest,
        primary: &Path,
        cfg: EngineConfig,
    ) -> Result<Self, WickError> {
        // Pre-filter on inference_type so VL / Unknown manifests fail
        // fast without paying for the GGUF mmap + header parse. `from_gguf`
        // checks again for the in-memory constructors that skip this path.
        check_inference_type_supported(&manifest.inference_type)?;
        let gguf = GgufFile::open(primary)
            .map_err(|e| WickError::Backend(format!("opening `{}`: {e}", primary.display())))?;
        Self::from_gguf(gguf, manifest, cfg, Some(primary))
    }

    /// Convergence point for `from_path(.gguf)`. Re-resolves the primary
    /// from the synthetic manifest and dispatches through
    /// [`Self::from_manifest_with_primary`].
    #[cfg(feature = "mmap")]
    fn from_manifest(mut manifest: Manifest, cfg: EngineConfig) -> Result<Self, WickError> {
        resolve_all_manifest_files(&mut manifest, None, &cfg)?;
        let primary = PathBuf::from(&manifest.files.model);
        Self::from_manifest_with_primary(manifest, &primary, cfg)
    }

    // --- accessors ---

    /// Create a new [`Session`] sharing ownership of the engine's model
    /// and tokenizer via `Arc` clones. The returned session outlives
    /// `&self`; the engine keeps the originals live for every session
    /// it handed out. The session's [`ModalityCapabilities`] is derived
    /// from the manifest's `inference_type`.
    pub fn new_session(&self, cfg: SessionConfig) -> Session {
        Session::new(
            Arc::clone(&self.model),
            Arc::clone(&self.tokenizer),
            self.capabilities(),
            cfg,
        )
    }

    /// Modality capabilities reported by the loaded model, derived from
    /// the manifest's `inference_type`. Useful for FFI consumers that
    /// want to gate UI / API surfaces on what the model supports
    /// without constructing a [`Session`].
    pub fn capabilities(&self) -> ModalityCapabilities {
        ModalityCapabilities::from_inference_type(&self.manifest.inference_type)
    }

    /// Borrow the loaded model. Used by the audio pipeline today;
    /// unified `Session::append_audio` will subsume this in a follow-up.
    pub fn model(&self) -> &dyn Model {
        self.model.as_ref()
    }

    /// Shared refcounted handle to the loaded model. Used by callers
    /// (FFI wrappers, the audio pipeline, future trait impls) that
    /// need to keep the model alive independently of the engine.
    pub fn model_arc(&self) -> Arc<dyn Model> {
        Arc::clone(&self.model)
    }

    /// Borrow the tokenizer.
    pub fn tokenizer(&self) -> &BpeTokenizer {
        self.tokenizer.as_ref()
    }

    /// Shared refcounted handle to the tokenizer.
    pub fn tokenizer_arc(&self) -> Arc<BpeTokenizer> {
        Arc::clone(&self.tokenizer)
    }

    /// Borrow the parsed manifest.
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// Borrow the metadata summary.
    pub fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    /// Borrow the engine config.
    pub fn config(&self) -> &EngineConfig {
        &self.config
    }

    /// Configure the model's KV prefix cache. Passthrough to
    /// `Model::configure_cache`; exposed here so callers that only hold
    /// a `WickEngine` don't need to reach into `engine.model()`.
    pub fn configure_cache(&self, cfg: KvCacheConfig) {
        self.model.configure_cache(cfg);
    }
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

// Only used on `mmap` builds (by `from_path` + `find_single_manifest`).
#[cfg(feature = "mmap")]
fn has_extension(p: &Path, ext: &str) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case(ext))
}

#[cfg(feature = "mmap")]
fn find_single_manifest(dir: &Path) -> Result<PathBuf, WickError> {
    let entries = std::fs::read_dir(dir)
        .map_err(|e| WickError::Backend(format!("reading directory `{}`: {e}", dir.display())))?;
    let mut jsons: Vec<PathBuf> = Vec::new();
    for entry in entries {
        let entry =
            entry.map_err(|e| WickError::Backend(format!("reading directory entry: {e}")))?;
        let path = entry.path();
        if path.is_file() && has_extension(&path, "json") {
            jsons.push(path);
        }
    }
    match jsons.len() {
        0 => Err(WickError::Backend(format!(
            "no .json manifest in directory `{}`",
            dir.display()
        ))),
        1 => Ok(jsons.into_iter().next().unwrap()),
        n => {
            jsons.sort();
            let names: Vec<String> = jsons
                .iter()
                .filter_map(|p| p.file_name().map(|f| f.to_string_lossy().into_owned()))
                .collect();
            Err(WickError::Backend(format!(
                "{n} .json manifests in directory `{}` (expected exactly one): {}",
                dir.display(),
                names.join(", ")
            )))
        }
    }
}

/// Resolve every file reference in `manifest.files` to a local path,
/// rewriting the manifest's URL/path strings in place. A remote
/// `http(s)://` URL is downloaded + cached via `EngineConfig::bundle_repo`
/// (with the `remote` feature); a relative path is joined against
/// `manifest_dir` when provided; an absolute path is kept as-is.
///
/// Fields walked (in declaration order):
/// - `files.model` (required)
/// - `files.multimodal_projector` (optional — VL + audio bundles)
/// - `files.audio_decoder` (optional — audio-out bundles)
/// - `files.audio_tokenizer` (optional — audio-in bundles)
/// - `files.extras` (every entry — forward-compat aux roles)
///
/// Consumers downstream of the loader (audio pipeline, VL loader, etc.)
/// read back from `engine.manifest().files.*` and expect local paths,
/// so every URL must be rewritten before we hand the manifest on.
///
/// Gated on `mmap` — the callers (`from_manifest_file`, `from_manifest`)
/// are both `mmap`-only. `from_bytes` / `from_reader` skip path
/// resolution entirely since they receive bytes.
#[cfg(feature = "mmap")]
fn resolve_all_manifest_files(
    manifest: &mut Manifest,
    manifest_dir: Option<&Path>,
    cfg: &EngineConfig,
) -> Result<(), WickError> {
    manifest.files.model = resolve_url_or_path(&manifest.files.model, manifest_dir, cfg)?
        .to_string_lossy()
        .into_owned();

    for slot in [
        &mut manifest.files.multimodal_projector,
        &mut manifest.files.audio_decoder,
        &mut manifest.files.audio_tokenizer,
    ] {
        if let Some(s) = slot.as_ref() {
            let resolved = resolve_url_or_path(s, manifest_dir, cfg)?;
            *slot = Some(resolved.to_string_lossy().into_owned());
        }
    }

    for value in manifest.files.extras.values_mut() {
        *value = resolve_url_or_path(value, manifest_dir, cfg)?
            .to_string_lossy()
            .into_owned();
    }

    Ok(())
}

#[cfg(feature = "mmap")]
fn resolve_url_or_path(
    value: &str,
    base_dir: Option<&Path>,
    cfg: &EngineConfig,
) -> Result<PathBuf, WickError> {
    if is_remote_url(value) {
        #[cfg(feature = "remote")]
        {
            if let Some(repo) = cfg.bundle_repo.as_ref() {
                return repo.resolve_url(value, None);
            }
            return Err(WickError::Backend(format!(
                "manifest references remote URL `{value}` — set `EngineConfig::bundle_repo` \
                 to a `BundleRepo` rooted at your desired store directory, or pre-download \
                 the bundle and pass a local file path."
            )));
        }
        #[cfg(not(feature = "remote"))]
        {
            let _ = cfg;
            return Err(WickError::Backend(format!(
                "manifest references remote URL `{value}` — rebuild wick with the `remote` \
                 feature + set `EngineConfig::bundle_repo`, or pre-download the bundle \
                 and pass a local file path."
            )));
        }
    }
    if let Some(rest) = strip_file_scheme(value) {
        // `file://…` isn't portable via `Path::new` (Windows especially),
        // so reject until we take a real URI dependency. Users with a
        // `file://` URI in hand can drop the scheme before calling.
        return Err(WickError::Backend(format!(
            "manifest references `file://` URI `{value}` — wick doesn't parse file URIs yet; \
             pass the local path directly (e.g. `{rest}`)."
        )));
    }
    let p = Path::new(value);
    if p.is_absolute() {
        Ok(p.to_path_buf())
    } else if let Some(base) = base_dir {
        Ok(base.join(p))
    } else {
        Ok(p.to_path_buf())
    }
}

#[cfg(feature = "mmap")]
fn is_remote_url(s: &str) -> bool {
    let lower = s.to_ascii_lowercase();
    lower.starts_with("http://") || lower.starts_with("https://")
}

/// Return the path-like tail of a `file://` URI, or `None` if the input
/// isn't a file URI. Case-insensitive on the scheme. Does NOT decode
/// percent-encoding or handle Windows drive letters; the caller errors
/// out rather than trying to interpret it.
#[cfg(feature = "mmap")]
fn strip_file_scheme(s: &str) -> Option<&str> {
    let lower = s.to_ascii_lowercase();
    if let Some(rest) = lower.strip_prefix("file://") {
        // Slice from the same byte offset in the original string so we
        // preserve case.
        let offset = s.len() - rest.len();
        Some(&s[offset..])
    } else {
        None
    }
}

/// Build a minimal `Manifest` from an explicit `ModelFiles`.
#[cfg(feature = "mmap")]
fn synthesize_manifest_from_files(files: &ModelFiles) -> Result<Manifest, WickError> {
    let inference_type = match files.inference_type.clone() {
        Some(it) => it,
        None => auto_detect_inference_type(&files.model)?,
    };

    let model_str = files.model.to_string_lossy().into_owned();
    let mmproj = files
        .multimodal_projector
        .as_ref()
        .map(|p| p.to_string_lossy().into_owned());
    let audio_decoder = files
        .audio_decoder
        .as_ref()
        .map(|p| p.to_string_lossy().into_owned());
    let audio_tokenizer = files
        .audio_tokenizer
        .as_ref()
        .map(|p| p.to_string_lossy().into_owned());
    let mut extras_str = std::collections::HashMap::with_capacity(files.extras.len());
    for (k, v) in &files.extras {
        extras_str.insert(k.clone(), v.to_string_lossy().into_owned());
    }

    // Build a serde_json::Value that mirrors the typed shape so
    // `Manifest::raw` stays useful for consumers that inspect it.
    let mut load_params = serde_json::Map::new();
    load_params.insert("model".into(), serde_json::Value::String(model_str.clone()));
    if let Some(v) = &mmproj {
        load_params.insert(
            "multimodal_projector".into(),
            serde_json::Value::String(v.clone()),
        );
    }
    if let Some(v) = &audio_decoder {
        load_params.insert("audio_decoder".into(), serde_json::Value::String(v.clone()));
    }
    if let Some(v) = &audio_tokenizer {
        load_params.insert(
            "audio_tokenizer".into(),
            serde_json::Value::String(v.clone()),
        );
    }
    for (k, v) in &extras_str {
        load_params.insert(k.clone(), serde_json::Value::String(v.clone()));
    }
    if let Some(t) = &files.chat_template {
        load_params.insert("chat_template".into(), serde_json::Value::String(t.clone()));
    }

    let mut raw_map = serde_json::Map::new();
    raw_map.insert(
        "inference_type".into(),
        serde_json::Value::String(inference_type.as_str().to_string()),
    );
    raw_map.insert(
        "schema_version".into(),
        serde_json::Value::String("1.0.0".into()),
    );
    raw_map.insert(
        "load_time_parameters".into(),
        serde_json::Value::Object(load_params),
    );

    let defaults_shape = inference_type_defaults_shape(&inference_type);
    Ok(Manifest {
        inference_type,
        schema_version: "1.0.0".into(),
        files: ManifestFiles {
            model: model_str,
            multimodal_projector: mmproj,
            audio_decoder,
            audio_tokenizer,
            extras: extras_str,
        },
        chat_template: files.chat_template.clone(),
        // For `from_files` the caller hasn't provided sampling defaults;
        // surface a zero-info `Text` variant for text/VL models and an
        // empty `Audio` variant for audio models. Consumers who need
        // defaults should go through a real manifest.
        //
        // Key on the *resolved* `inference_type` — using `files.inference_type`
        // (pre-resolution) would hand the `Text` defaults shape to an
        // auto-detected audio model.
        generation_defaults: match defaults_shape {
            DefaultsShape::Text => crate::manifest::GenerationDefaults::Text {
                temperature: None,
                min_p: None,
                top_p: None,
                top_k: None,
                repetition_penalty: None,
            },
            DefaultsShape::Audio => crate::manifest::GenerationDefaults::Audio {
                number_of_decoding_threads: None,
            },
            DefaultsShape::Other => crate::manifest::GenerationDefaults::Other {
                raw: serde_json::Value::Null,
            },
        },
        raw: serde_json::Value::Object(raw_map),
    })
}

// Only used by `synthesize_manifest_from_files` (mmap-gated).
#[cfg(feature = "mmap")]
enum DefaultsShape {
    Text,
    Audio,
    Other,
}

#[cfg(feature = "mmap")]
fn inference_type_defaults_shape(it: &InferenceType) -> DefaultsShape {
    match it {
        InferenceType::LlamaCppLfm2AudioV1 => DefaultsShape::Audio,
        InferenceType::LlamaCppTextToText | InferenceType::LlamaCppImageToText => {
            DefaultsShape::Text
        }
        InferenceType::Unknown(_) => DefaultsShape::Other,
    }
}

/// Peek at the GGUF header and guess an inference type. Minimal mapping
/// for v1 — only `lfm2` is actually loadable today; the other arches
/// are listed so auto-detect doesn't silently confuse a future non-text
/// model for text.
/// Shared gate for the set of `InferenceType`s the engine can actually
/// load today. Returns `Ok(())` for text + LFM2-audio; returns
/// `WickError::UnsupportedInferenceType` for VL and anything unknown.
/// Unconditional so both the mmap-backed path (pre-file-open) and the
/// in-memory paths (`from_bytes` / `from_reader`) use the same rule.
fn check_inference_type_supported(it: &InferenceType) -> Result<(), WickError> {
    match it {
        InferenceType::LlamaCppTextToText | InferenceType::LlamaCppLfm2AudioV1 => Ok(()),
        InferenceType::LlamaCppImageToText => {
            Err(WickError::UnsupportedInferenceType(it.as_str().to_string()))
        }
        InferenceType::Unknown(s) => Err(WickError::UnsupportedInferenceType(s.clone())),
    }
}

#[cfg(feature = "mmap")]
fn auto_detect_inference_type(model_path: &Path) -> Result<InferenceType, WickError> {
    let gguf = GgufFile::open(model_path).map_err(|e| {
        WickError::Backend(format!(
            "opening `{}` for inference-type auto-detect: {e}",
            model_path.display()
        ))
    })?;
    let arch = gguf.get_str("general.architecture").unwrap_or("");
    Ok(match arch {
        "lfm2" | "llama" | "qwen2" | "qwen3" => InferenceType::LlamaCppTextToText,
        "lfm2vl" => InferenceType::LlamaCppImageToText,
        "lfm2-audio" => InferenceType::LlamaCppLfm2AudioV1,
        // Unknown arch → assume text. Callers who need a different
        // mapping can set `ModelFiles::inference_type` explicitly.
        _ => InferenceType::LlamaCppTextToText,
    })
}

/// Dispatch the text-model loader on [`BackendPreference`]. Single source
/// of truth for "how to turn a `GgufFile` + a preference into a
/// `Box<dyn Model>`" — the CLI used to carry this logic.
fn load_text_model(
    gguf: GgufFile,
    path: Option<&Path>,
    cfg: &EngineConfig,
) -> Result<Box<dyn Model>, WickError> {
    match cfg.backend {
        BackendPreference::Auto => load_text_model_auto(gguf, path, cfg.context_size),
        BackendPreference::Cpu => model::load_model(gguf, cfg.context_size)
            .map_err(|e| WickError::Backend(format!("CPU model load failed: {e}"))),
        #[cfg(feature = "gpu")]
        BackendPreference::Gpu => model::load_model_gpu(gguf, cfg.context_size)
            .map_err(|e| WickError::Backend(format!("GPU model load failed: {e}"))),
        #[cfg(not(feature = "gpu"))]
        BackendPreference::Gpu => Err(WickError::Backend(
            "GPU backend not available (compile with --features gpu)".into(),
        )),
        #[cfg(all(feature = "metal", target_os = "macos"))]
        BackendPreference::Metal => {
            let p = path.ok_or_else(|| {
                WickError::Backend("Metal backend requires a file path (not from_bytes)".into())
            })?;
            model::load_model_metal(gguf, p, cfg.context_size)
                .map_err(|e| WickError::Backend(format!("Metal model load failed: {e}")))
        }
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        BackendPreference::Metal => Err(WickError::Backend(
            "Metal backend not available (compile with --features metal on macOS)".into(),
        )),
    }
}

fn load_text_model_auto(
    gguf: GgufFile,
    path: Option<&Path>,
    context_size: usize,
) -> Result<Box<dyn Model>, WickError> {
    // Without a path (today: `from_bytes` only), Metal is unreachable
    // (it requires a file) and wgpu-then-CPU fallback can't re-open the
    // source. Short-circuit to CPU so `from_bytes` stays robust — this
    // matches the documented "testing / <50 MB" intent of that
    // constructor. Callers who want GPU with in-memory bytes must
    // opt in explicitly via `BackendPreference::Gpu`.
    if path.is_none() {
        tracing::debug!("wick::engine: no path available (from_bytes); using CPU backend (auto)");
        return model::load_model(gguf, context_size)
            .map_err(|e| WickError::Backend(format!("CPU model load failed: {e}")));
    }

    // Metal → wgpu → CPU. Mirrors the CLI's previous `load_model_auto`.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    if let Some(p) = path {
        match model::load_model_metal(clone_gguf_like(&gguf, p)?, p, context_size) {
            Ok(m) => {
                tracing::debug!("wick::engine: using native Metal backend (auto)");
                return Ok(m);
            }
            Err(e) => {
                tracing::debug!("wick::engine: Metal unavailable ({e}); trying next backend");
            }
        }
    }

    #[cfg(feature = "gpu")]
    {
        // Path guaranteed present (short-circuited above otherwise).
        let p = path.expect("path guaranteed by early return");
        let gguf_for_gpu = clone_gguf_like(&gguf, p)?;
        match model::load_model_gpu(gguf_for_gpu, context_size) {
            Ok(m) => {
                tracing::debug!("wick::engine: using wgpu GPU backend (auto)");
                return Ok(m);
            }
            Err(e) => {
                tracing::debug!("wick::engine: wgpu unavailable ({e}); falling back to CPU");
            }
        }
        // Re-open the file for CPU — original `gguf` may have been
        // consumed by the Metal attempt above.
        let gguf_for_cpu = GgufFile::open(p).map_err(|e| {
            WickError::Backend(format!("reopening `{}` for CPU fallback: {e}", p.display()))
        })?;
        return model::load_model(gguf_for_cpu, context_size)
            .map_err(|e| WickError::Backend(format!("CPU model load failed: {e}")));
    }

    #[cfg(not(feature = "gpu"))]
    {
        tracing::debug!("wick::engine: using CPU backend (auto)");
        model::load_model(gguf, context_size)
            .map_err(|e| WickError::Backend(format!("CPU model load failed: {e}")))
    }
}

/// Re-open a GGUF from its path. The Metal and wgpu loaders consume
/// `GgufFile` by value, so the auto-dispatch path has to freshly map
/// the file for each backend it tries.
#[cfg(any(all(feature = "metal", target_os = "macos"), feature = "gpu"))]
fn clone_gguf_like(_: &GgufFile, path: &Path) -> Result<GgufFile, WickError> {
    GgufFile::open(path)
        .map_err(|e| WickError::Backend(format!("reopening `{}`: {e}", path.display())))
}

fn build_metadata(
    model: &dyn Model,
    tokenizer: &BpeTokenizer,
    manifest: &Manifest,
    add_bos_token: bool,
    quantization: String,
) -> ModelMetadata {
    let cfg = model.config();
    // Reflect the effective template availability: a manifest override
    // OR a GGUF-embedded template (the common case for bare `.gguf`
    // loads). Consumers asking `metadata().has_chat_template` expect a
    // truthful answer, not just "does the manifest have one".
    let has_chat_template = manifest.chat_template.is_some() || tokenizer.chat_template().is_some();
    ModelMetadata {
        architecture: cfg.architecture.clone(),
        max_seq_len: cfg.max_seq_len as u32,
        vocab_size: cfg.vocab_size as u32,
        has_chat_template,
        quantization,
        add_bos_token,
    }
}

/// Map a GGUF `general.file_type` value (the llama.cpp `LLAMA_FTYPE_*`
/// enum) to the canonical short label used in filenames and tooling
/// (`Q4_0`, `Q4_K_M`, `BF16`, etc.). Falls back to `ftype:N` for
/// unrecognized values rather than dropping information — when a new
/// quantization scheme appears, the number itself is enough for a
/// human to look up. Returns `"unknown"` when the GGUF doesn't carry
/// the field at all.
///
/// List mirrors llama.cpp's enum as of early 2026; extend as new
/// quants ship upstream.
fn ftype_label(ftype: u32) -> String {
    match ftype {
        0 => "F32".into(),
        1 => "F16".into(),
        2 => "Q4_0".into(),
        3 => "Q4_1".into(),
        7 => "Q8_0".into(),
        8 => "Q5_0".into(),
        9 => "Q5_1".into(),
        10 => "Q2_K".into(),
        11 => "Q3_K_S".into(),
        12 => "Q3_K_M".into(),
        13 => "Q3_K_L".into(),
        14 => "Q4_K_S".into(),
        15 => "Q4_K_M".into(),
        16 => "Q5_K_S".into(),
        17 => "Q5_K_M".into(),
        18 => "Q6_K".into(),
        19 => "IQ2_XXS".into(),
        20 => "IQ2_XS".into(),
        21 => "Q2_K_S".into(),
        22 => "IQ3_XS".into(),
        23 => "IQ3_XXS".into(),
        24 => "IQ1_S".into(),
        25 => "IQ4_NL".into(),
        26 => "IQ3_S".into(),
        27 => "IQ3_M".into(),
        28 => "IQ2_S".into(),
        29 => "IQ2_M".into(),
        30 => "IQ4_XS".into(),
        31 => "IQ1_M".into(),
        32 => "BF16".into(),
        36 => "TQ1_0".into(),
        37 => "TQ2_0".into(),
        other => format!("ftype:{other}"),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_preference_default_is_auto() {
        assert_eq!(BackendPreference::default(), BackendPreference::Auto);
    }

    #[test]
    fn backend_preference_parse_str_known_labels() {
        assert_eq!(
            BackendPreference::parse_str("auto").unwrap(),
            BackendPreference::Auto
        );
        assert_eq!(
            BackendPreference::parse_str("").unwrap(),
            BackendPreference::Auto
        );
        assert_eq!(
            BackendPreference::parse_str("CPU").unwrap(),
            BackendPreference::Cpu
        );
        assert_eq!(
            BackendPreference::parse_str("gpu").unwrap(),
            BackendPreference::Gpu
        );
        assert_eq!(
            BackendPreference::parse_str("wgpu").unwrap(),
            BackendPreference::Gpu
        );
        assert_eq!(
            BackendPreference::parse_str("Metal").unwrap(),
            BackendPreference::Metal
        );
        assert!(BackendPreference::parse_str("nvidia").is_err());
    }

    #[test]
    fn engine_config_default_is_4k_auto() {
        let c = EngineConfig::default();
        assert_eq!(c.context_size, 4096);
        assert_eq!(c.backend, BackendPreference::Auto);
    }

    #[test]
    fn is_remote_url_covers_http_https() {
        assert!(is_remote_url("http://example.com/x.gguf"));
        assert!(is_remote_url("HTTPS://example.com/x.gguf"));
        assert!(!is_remote_url("/local/path.gguf"));
        assert!(!is_remote_url("./rel/path.gguf"));
        assert!(!is_remote_url("file:///local/path.gguf"));
    }

    #[test]
    fn has_extension_case_insensitive() {
        assert!(has_extension(Path::new("foo.gguf"), "gguf"));
        assert!(has_extension(Path::new("foo.GGUF"), "gguf"));
        assert!(has_extension(Path::new("foo.json"), "json"));
        assert!(!has_extension(Path::new("foo.txt"), "gguf"));
        assert!(!has_extension(Path::new("foo"), "gguf"));
    }

    #[test]
    fn resolve_url_or_path_rejects_remote_without_repo() {
        let cfg = EngineConfig::default();
        let e = resolve_url_or_path("https://hf.co/x.gguf", None, &cfg)
            .expect_err("remote URL must error without a BundleRepo");
        let msg = format!("{e}");
        // Without the `remote` feature or with `bundle_repo = None`, the
        // error should steer the user toward the fix.
        assert!(
            msg.contains("remote URL"),
            "error should mention remote URL; got `{msg}`"
        );
        #[cfg(feature = "remote")]
        assert!(
            msg.contains("bundle_repo"),
            "error under `remote` feature should point at the config field; got `{msg}`"
        );
        #[cfg(not(feature = "remote"))]
        assert!(
            msg.contains("`remote` feature"),
            "error without `remote` feature should point at enabling it; got `{msg}`"
        );
    }

    #[test]
    fn resolve_url_or_path_rejects_file_scheme() {
        let cfg = EngineConfig::default();
        let e = resolve_url_or_path("file:///models/x.gguf", None, &cfg)
            .expect_err("file:// URIs aren't supported yet");
        let msg = format!("{e}");
        assert!(
            msg.contains("file://") && msg.contains("wick doesn't parse file URIs"),
            "error should point at the file:// limitation; got `{msg}`"
        );
    }

    #[test]
    fn strip_file_scheme_preserves_case() {
        assert_eq!(
            strip_file_scheme("FILE:///Models/Foo.gguf"),
            Some("/Models/Foo.gguf")
        );
        assert_eq!(strip_file_scheme("file://./rel"), Some("./rel"));
        assert_eq!(strip_file_scheme("https://x/y"), None);
        assert_eq!(strip_file_scheme("/abs/path"), None);
    }

    #[test]
    fn resolve_url_or_path_joins_relative_against_base() {
        let cfg = EngineConfig::default();
        let base = PathBuf::from("/models/bundles");
        let got = resolve_url_or_path("LFM2-1.2B-Q4_0.gguf", Some(&base), &cfg).unwrap();
        assert_eq!(got, PathBuf::from("/models/bundles/LFM2-1.2B-Q4_0.gguf"));
    }

    #[test]
    fn resolve_url_or_path_keeps_absolute_unchanged() {
        let cfg = EngineConfig::default();
        let base = PathBuf::from("/models/bundles");
        let got = resolve_url_or_path("/opt/foo.gguf", Some(&base), &cfg).unwrap();
        assert_eq!(got, PathBuf::from("/opt/foo.gguf"));
    }

    /// Regression guard: the resolver must touch every file field, not
    /// just `files.model`. Previously `resolve_primary_model_path` only
    /// handled the primary, silently leaving audio / VL / extras
    /// fields as raw URLs — which then broke downstream consumers.
    #[test]
    fn resolve_all_manifest_files_walks_every_field() {
        use crate::manifest::{GenerationDefaults, InferenceType, Manifest, ManifestFiles};

        let base = PathBuf::from("/models/bundles");
        let mut extras = std::collections::HashMap::new();
        extras.insert("cover_art".to_string(), "cover.png".to_string());
        extras.insert("config".to_string(), "/abs/config.toml".to_string());

        let mut manifest = Manifest {
            inference_type: InferenceType::LlamaCppLfm2AudioV1,
            schema_version: "1.0.0".to_string(),
            files: ManifestFiles {
                model: "model.gguf".to_string(),
                multimodal_projector: Some("mmproj.gguf".to_string()),
                audio_decoder: Some("decoder.gguf".to_string()),
                audio_tokenizer: Some("tokenizer.safetensors".to_string()),
                extras,
            },
            chat_template: None,
            generation_defaults: GenerationDefaults::Other {
                raw: serde_json::Value::Null,
            },
            raw: serde_json::Value::Null,
        };

        let cfg = EngineConfig::default();
        resolve_all_manifest_files(&mut manifest, Some(&base), &cfg).unwrap();

        assert_eq!(manifest.files.model, "/models/bundles/model.gguf");
        assert_eq!(
            manifest.files.multimodal_projector.as_deref(),
            Some("/models/bundles/mmproj.gguf")
        );
        assert_eq!(
            manifest.files.audio_decoder.as_deref(),
            Some("/models/bundles/decoder.gguf")
        );
        assert_eq!(
            manifest.files.audio_tokenizer.as_deref(),
            Some("/models/bundles/tokenizer.safetensors")
        );
        assert_eq!(
            manifest.files.extras.get("cover_art").map(String::as_str),
            Some("/models/bundles/cover.png")
        );
        // Absolute extras stay absolute.
        assert_eq!(
            manifest.files.extras.get("config").map(String::as_str),
            Some("/abs/config.toml")
        );
    }

    #[test]
    fn resolve_all_manifest_files_none_optionals_stay_none() {
        use crate::manifest::{GenerationDefaults, InferenceType, Manifest, ManifestFiles};
        let mut manifest = Manifest {
            inference_type: InferenceType::LlamaCppTextToText,
            schema_version: "1.0.0".to_string(),
            files: ManifestFiles {
                model: "/abs/model.gguf".to_string(),
                multimodal_projector: None,
                audio_decoder: None,
                audio_tokenizer: None,
                extras: std::collections::HashMap::new(),
            },
            chat_template: None,
            generation_defaults: GenerationDefaults::Other {
                raw: serde_json::Value::Null,
            },
            raw: serde_json::Value::Null,
        };
        let cfg = EngineConfig::default();
        resolve_all_manifest_files(&mut manifest, None, &cfg).unwrap();
        assert!(manifest.files.multimodal_projector.is_none());
        assert!(manifest.files.audio_decoder.is_none());
        assert!(manifest.files.audio_tokenizer.is_none());
    }

    #[test]
    fn find_single_manifest_zero_and_many() {
        let dir = tempfile::tempdir().unwrap();
        let e0 = find_single_manifest(dir.path()).expect_err("empty dir must error");
        assert!(format!("{e0}").contains("no .json manifest"));

        std::fs::write(dir.path().join("a.json"), b"{}").unwrap();
        let got = find_single_manifest(dir.path()).unwrap();
        assert_eq!(got.file_name().unwrap(), "a.json");

        std::fs::write(dir.path().join("b.json"), b"{}").unwrap();
        let e2 =
            find_single_manifest(dir.path()).expect_err("two manifests must error (ambiguous)");
        let msg = format!("{e2}");
        assert!(msg.contains("2 .json manifests"), "{msg}");
        assert!(msg.contains("a.json") && msg.contains("b.json"), "{msg}");
    }

    #[test]
    fn synthesize_manifest_from_files_preserves_aux() {
        let files = ModelFiles {
            model: PathBuf::from("/m/model.gguf"),
            multimodal_projector: Some(PathBuf::from("/m/mmproj.gguf")),
            audio_decoder: Some(PathBuf::from("/m/ad.gguf")),
            audio_tokenizer: Some(PathBuf::from("/m/at.safetensors")),
            extras: std::collections::HashMap::new(),
            inference_type: Some(InferenceType::LlamaCppLfm2AudioV1),
            chat_template: None,
        };
        let m = synthesize_manifest_from_files(&files).unwrap();
        assert_eq!(m.inference_type, InferenceType::LlamaCppLfm2AudioV1);
        assert_eq!(m.files.model, "/m/model.gguf");
        assert_eq!(
            m.files.multimodal_projector.as_deref(),
            Some("/m/mmproj.gguf")
        );
        assert_eq!(m.files.audio_decoder.as_deref(), Some("/m/ad.gguf"));
        assert_eq!(
            m.files.audio_tokenizer.as_deref(),
            Some("/m/at.safetensors")
        );
        assert!(matches!(
            m.generation_defaults,
            crate::manifest::GenerationDefaults::Audio { .. }
        ));
    }

    #[test]
    fn model_files_text_helper_is_text_only() {
        let f = ModelFiles::text("/x/y.gguf");
        assert_eq!(f.model, PathBuf::from("/x/y.gguf"));
        assert!(f.multimodal_projector.is_none());
        assert_eq!(f.inference_type, Some(InferenceType::LlamaCppTextToText));
    }
}
