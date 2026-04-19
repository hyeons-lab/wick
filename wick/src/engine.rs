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

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::gguf::GgufFile;
use crate::kv_cache::KvCacheConfig;
use crate::manifest::{InferenceType, Manifest, ManifestFiles};
use crate::model::{self, Model};
use crate::session::{Session, SessionConfig, WickError};
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
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            context_size: 4096,
            backend: BackendPreference::Auto,
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
pub struct WickEngine {
    manifest: Manifest,
    model: Box<dyn Model>,
    tokenizer: BpeTokenizer,
    metadata: ModelMetadata,
    config: EngineConfig,
}

impl WickEngine {
    /// Load from a path that may be:
    /// - a bare `.gguf` file → internally synthesizes a text manifest,
    /// - a `.json` LeapBundles manifest → parsed + dispatched on `inference_type`,
    /// - a directory → scanned for exactly one `.json` manifest.
    pub fn from_path<P: AsRef<Path>>(path: P, cfg: EngineConfig) -> Result<Self, WickError> {
        let path = path.as_ref();
        if path.is_dir() {
            let manifest_path = find_single_manifest(path)?;
            Self::from_manifest_file(&manifest_path, cfg)
        } else if has_extension(path, "json") {
            Self::from_manifest_file(path, cfg)
        } else if has_extension(path, "gguf") || path.is_file() {
            // Bare file (with or without .gguf suffix) → synthesize a text manifest.
            let manifest = Manifest::synthetic_text(path);
            Self::from_manifest(manifest, cfg)
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
    pub fn from_bytes(bytes: impl Into<Arc<[u8]>>, cfg: EngineConfig) -> Result<Self, WickError> {
        let arc_bytes: Arc<[u8]> = bytes.into();
        let gguf = GgufFile::from_bytes(arc_bytes)
            .map_err(|e| WickError::Backend(format!("parsing GGUF bytes: {e}")))?;
        // Synthesize a minimal text manifest; model path is a placeholder
        // since we don't have one. Consumers who care about the manifest
        // should use `from_path`.
        let manifest = Manifest::synthetic_text(Path::new("<bytes>"));
        let tokenizer = BpeTokenizer::from_gguf(&gguf)
            .map_err(|e| WickError::Backend(format!("loading tokenizer: {e}")))?;
        let add_bos_token = gguf
            .get_bool("tokenizer.ggml.add_bos_token")
            .unwrap_or(false);
        let model = load_text_model(gguf, None, &cfg)?;
        let metadata = build_metadata(model.as_ref(), &manifest, add_bos_token);
        Ok(Self {
            manifest,
            model,
            tokenizer,
            metadata,
            config: cfg,
        })
    }

    /// Load from explicit file paths — skips manifest JSON parsing.
    /// `files.inference_type` decides the loader; `None` auto-detects
    /// from the GGUF header.
    pub fn from_files(files: ModelFiles, cfg: EngineConfig) -> Result<Self, WickError> {
        let manifest = synthesize_manifest_from_files(&files)?;
        // If the caller overrode the chat template, apply it by threading
        // it through the manifest; the text loader doesn't need to know.
        // (The tokenizer will still be built from the GGUF; template
        // precedence lives on the manifest for downstream consumers.)
        Self::from_manifest_with_primary(manifest, files.model.as_path(), cfg)
    }

    // --- internal constructors ---

    fn from_manifest_file(path: &Path, cfg: EngineConfig) -> Result<Self, WickError> {
        let manifest = Manifest::from_file(path).map_err(|e| {
            WickError::Backend(format!("parsing manifest `{}`: {e}", path.display()))
        })?;
        let manifest_dir = path.parent().map(Path::to_path_buf);
        let primary = resolve_primary_model_path(&manifest, manifest_dir.as_deref())?;
        Self::from_manifest_with_primary(manifest, &primary, cfg)
    }

    /// Dispatch helper that assumes the primary model path has already
    /// been resolved to a local file. Called by `from_manifest_file` and
    /// `from_files`; `from_path(.gguf)` goes through [`Self::from_manifest`]
    /// which re-resolves the primary from the synthetic manifest.
    fn from_manifest_with_primary(
        manifest: Manifest,
        primary: &Path,
        cfg: EngineConfig,
    ) -> Result<Self, WickError> {
        match &manifest.inference_type {
            InferenceType::LlamaCppTextToText | InferenceType::LlamaCppLfm2AudioV1 => {
                // Text LLMs AND LFM2-audio models both load the primary
                // GGUF through the same path. Audio aux files (decoder,
                // mmproj, safetensors tokenizer) stay on the manifest
                // for the audio pipeline to pick up separately.
            }
            InferenceType::LlamaCppImageToText => {
                return Err(WickError::UnsupportedInferenceType(
                    manifest.inference_type.as_str().to_string(),
                ));
            }
            InferenceType::Unknown(s) => {
                return Err(WickError::UnsupportedInferenceType(s.clone()));
            }
        }

        let gguf = GgufFile::open(primary)
            .map_err(|e| WickError::Backend(format!("opening `{}`: {e}", primary.display())))?;
        let tokenizer = BpeTokenizer::from_gguf(&gguf)
            .map_err(|e| WickError::Backend(format!("loading tokenizer: {e}")))?;
        // Peek at add_bos before `gguf` is moved into the loader.
        let add_bos_token = gguf
            .get_bool("tokenizer.ggml.add_bos_token")
            .unwrap_or(false);
        let model = load_text_model(gguf, Some(primary), &cfg)?;
        let metadata = build_metadata(model.as_ref(), &manifest, add_bos_token);

        Ok(Self {
            manifest,
            model,
            tokenizer,
            metadata,
            config: cfg,
        })
    }

    /// Convergence point for `from_path(.gguf)` and `from_bytes` (after
    /// their respective synthetic manifests). Re-resolves the primary
    /// from the manifest and dispatches through
    /// [`Self::from_manifest_with_primary`].
    fn from_manifest(manifest: Manifest, cfg: EngineConfig) -> Result<Self, WickError> {
        let primary = resolve_primary_model_path(&manifest, None)?;
        Self::from_manifest_with_primary(manifest, &primary, cfg)
    }

    // --- accessors ---

    /// Create a new `Session` borrowing the engine's model + tokenizer.
    /// The returned session's lifetime is tied to `&self`.
    pub fn new_session(&self, cfg: SessionConfig) -> Session<'_> {
        Session::new(self.model.as_ref(), &self.tokenizer, cfg)
    }

    /// Borrow the loaded model. Used by the audio pipeline today;
    /// unified `Session::append_audio` will subsume this in a follow-up.
    pub fn model(&self) -> &dyn Model {
        self.model.as_ref()
    }

    /// Borrow the tokenizer.
    pub fn tokenizer(&self) -> &BpeTokenizer {
        &self.tokenizer
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

fn has_extension(p: &Path, ext: &str) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case(ext))
}

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

/// Resolve the manifest's primary model reference to a local filesystem
/// path. HTTP(S) URLs are rejected with a pointer at Phase 1.6's
/// downloader; relative paths resolve against `manifest_dir` when
/// provided (for on-disk manifests loaded via `from_manifest_file`).
fn resolve_primary_model_path(
    manifest: &Manifest,
    manifest_dir: Option<&Path>,
) -> Result<PathBuf, WickError> {
    resolve_url_or_path(&manifest.files.model, manifest_dir)
}

fn resolve_url_or_path(value: &str, base_dir: Option<&Path>) -> Result<PathBuf, WickError> {
    if is_remote_url(value) {
        return Err(WickError::Backend(format!(
            "manifest references remote URL `{value}` — bundle downloads land in Phase 1.6 (`BundleRepo`). \
             Pass a local file path or pre-download the bundle."
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

fn is_remote_url(s: &str) -> bool {
    let lower = s.to_ascii_lowercase();
    lower.starts_with("http://") || lower.starts_with("https://")
}

/// Build a minimal `Manifest` from an explicit `ModelFiles`.
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
        generation_defaults: match inference_type_kind_default(&files.inference_type) {
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

enum DefaultsShape {
    Text,
    Audio,
    Other,
}

fn inference_type_kind_default(it: &Option<InferenceType>) -> DefaultsShape {
    match it {
        Some(InferenceType::LlamaCppLfm2AudioV1) => DefaultsShape::Audio,
        Some(InferenceType::LlamaCppTextToText) | Some(InferenceType::LlamaCppImageToText) => {
            DefaultsShape::Text
        }
        Some(InferenceType::Unknown(_)) => DefaultsShape::Other,
        // Auto-detect will have resolved to a concrete variant; if it
        // somehow reaches here with `None`, prefer the text shape.
        None => DefaultsShape::Text,
    }
}

/// Peek at the GGUF header and guess an inference type. Minimal mapping
/// for v1 — only `lfm2` is actually loadable today; the other arches
/// are listed so auto-detect doesn't silently confuse a future non-text
/// model for text.
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
        let gguf_for_gpu = if let Some(p) = path {
            clone_gguf_like(&gguf, p)?
        } else {
            // Without a path we can't re-open; consume `gguf` for wgpu.
            gguf
        };
        match model::load_model_gpu(gguf_for_gpu, context_size) {
            Ok(m) => {
                tracing::debug!("wick::engine: using wgpu GPU backend (auto)");
                return Ok(m);
            }
            Err(e) => {
                tracing::debug!("wick::engine: wgpu unavailable ({e}); falling back to CPU");
            }
        }
        // If we took this branch and fell through, we've already consumed
        // `gguf` — need to re-open from path for CPU.
        let gguf_for_cpu = if let Some(p) = path {
            GgufFile::open(p).map_err(|e| {
                WickError::Backend(format!("reopening `{}` for CPU fallback: {e}", p.display()))
            })?
        } else {
            return Err(WickError::Backend(
                "wgpu failed and no path available to re-open for CPU fallback".into(),
            ));
        };
        return model::load_model(gguf_for_cpu, context_size)
            .map_err(|e| WickError::Backend(format!("CPU model load failed: {e}")));
    }

    #[cfg(not(feature = "gpu"))]
    {
        // `path` is threaded so `metal` / `gpu` builds can re-open the
        // file after a fallback; the CPU-only build consumes `gguf`
        // directly.
        let _ = path;
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

fn build_metadata(model: &dyn Model, manifest: &Manifest, add_bos_token: bool) -> ModelMetadata {
    let cfg = model.config();
    let has_chat_template = manifest.chat_template.is_some();
    ModelMetadata {
        architecture: cfg.architecture.clone(),
        max_seq_len: cfg.max_seq_len as u32,
        vocab_size: cfg.vocab_size as u32,
        has_chat_template,
        // `quantization` field follow-up: GGUF's `general.file_type`
        // encodes this but it's not wired through today. Leave
        // "unknown" for v1; UniFFI bindings don't gate on it.
        quantization: "unknown".into(),
        add_bos_token,
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
    fn resolve_url_or_path_rejects_remote() {
        let e =
            resolve_url_or_path("https://hf.co/x.gguf", None).expect_err("remote URL must error");
        let msg = format!("{e}");
        assert!(
            msg.contains("remote URL") && msg.contains("Phase 1.6"),
            "error should point at the downloader follow-up; got `{msg}`"
        );
    }

    #[test]
    fn resolve_url_or_path_joins_relative_against_base() {
        let base = PathBuf::from("/models/bundles");
        let got = resolve_url_or_path("LFM2-1.2B-Q4_0.gguf", Some(&base)).unwrap();
        assert_eq!(got, PathBuf::from("/models/bundles/LFM2-1.2B-Q4_0.gguf"));
    }

    #[test]
    fn resolve_url_or_path_keeps_absolute_unchanged() {
        let base = PathBuf::from("/models/bundles");
        let got = resolve_url_or_path("/opt/foo.gguf", Some(&base)).unwrap();
        assert_eq!(got, PathBuf::from("/opt/foo.gguf"));
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
