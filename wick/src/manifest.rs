//! LeapBundles manifest parser.
//!
//! Each model in `LiquidAI/LeapBundles` on Hugging Face ships a per-quant
//! `.json` file describing how to load it: the inference backend, the file
//! URLs (model + optional aux components like `multimodal_projector`,
//! `audio_decoder`, `audio_tokenizer`), and advisory generation defaults.
//! This module is a `serde` front-end over that schema + some ergonomic
//! enum-typed views.
//!
//! We preserve the raw `serde_json::Value` in `Manifest::raw` so
//! forward-compat extensions (new `load_time_parameters` fields, new
//! inference types, new sampling knobs) are not lost — the typed view
//! only covers the 1.0.0 fields we handle today.
//!
//! Three known inference-type shapes are recognized and given typed
//! variants: `llama.cpp/text-to-text`, `llama.cpp/image-to-text` (VL),
//! and `llama.cpp/lfm2-audio-v1`. Anything else falls into
//! `InferenceType::Unknown(raw)`, letting consumers display a
//! diagnostic error without the parser panicking.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};

/// Top-level manifest view. `raw` is retained for forward-compat; typed
/// accessors cover the 1.0.0 fields wick currently knows how to load.
#[derive(Debug, Clone)]
pub struct Manifest {
    pub inference_type: InferenceType,
    pub schema_version: String,
    pub files: ManifestFiles,
    /// Jinja chat template override. When `Some`, supersedes the template
    /// embedded in the primary GGUF's metadata; when `None`, the GGUF's
    /// own template is used.
    pub chat_template: Option<String>,
    pub generation_defaults: GenerationDefaults,
    /// Pristine JSON for consumers that need fields wick hasn't typed yet.
    pub raw: serde_json::Value,
}

/// Inference backend + modality marker from `inference_type`. The
/// recognised variants drive loader dispatch in `WickEngine::from_manifest`;
/// unrecognised values round-trip through `Unknown(String)` so callers
/// can print the actual string in their error message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InferenceType {
    /// `llama.cpp/text-to-text` — text-in, text-out.
    LlamaCppTextToText,
    /// `llama.cpp/image-to-text` — VL; manifest parsed, loader deferred to v2.
    LlamaCppImageToText,
    /// `llama.cpp/lfm2-audio-v1` — text+audio in/out via wick's existing
    /// audio pipeline.
    LlamaCppLfm2AudioV1,
    Unknown(String),
}

impl InferenceType {
    /// Parse a raw `inference_type` string. Named `parse_str` (not
    /// `from_str`) to avoid shadowing the `std::str::FromStr` trait's
    /// fallible signature — this one is infallible, always returning
    /// `Unknown(raw)` for values we don't recognize.
    pub fn parse_str(s: &str) -> Self {
        match s {
            "llama.cpp/text-to-text" => Self::LlamaCppTextToText,
            "llama.cpp/image-to-text" => Self::LlamaCppImageToText,
            "llama.cpp/lfm2-audio-v1" => Self::LlamaCppLfm2AudioV1,
            other => Self::Unknown(other.to_string()),
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::LlamaCppTextToText => "llama.cpp/text-to-text",
            Self::LlamaCppImageToText => "llama.cpp/image-to-text",
            Self::LlamaCppLfm2AudioV1 => "llama.cpp/lfm2-audio-v1",
            Self::Unknown(s) => s,
        }
    }
}

/// File references from `load_time_parameters`. All values are the raw
/// URLs (or file:// or local path strings) as they appear in the manifest.
/// Callers are responsible for fetching / resolving them — this module
/// only parses.
#[derive(Debug, Clone)]
pub struct ManifestFiles {
    /// Required: primary model GGUF.
    pub model: String,
    /// Optional: multimodal projector GGUF (VL, audio).
    pub multimodal_projector: Option<String>,
    /// Optional: audio-decoder GGUF (audio-out models).
    pub audio_decoder: Option<String>,
    /// Optional: audio tokenizer (usually a `.safetensors` checkpoint).
    pub audio_tokenizer: Option<String>,
    /// Any other `load_time_parameters` key whose value is a string,
    /// preserved verbatim. Forward-compat: new aux roles land here
    /// without a parser change.
    pub extras: HashMap<String, String>,
}

/// Advisory sampling / decoding defaults from `generation_time_parameters`.
/// Different inference types use different shapes; the parser splits them
/// into typed variants rather than a single flat struct.
#[derive(Debug, Clone)]
pub enum GenerationDefaults {
    /// `sampling_parameters` block — text and VL models.
    Text {
        temperature: Option<f32>,
        min_p: Option<f32>,
        top_p: Option<f32>,
        top_k: Option<u32>,
        repetition_penalty: Option<f32>,
    },
    /// `number_of_decoding_threads` style — audio models.
    Audio {
        number_of_decoding_threads: Option<u32>,
    },
    /// Fallback variant — used when the manifest's `inference_type` is
    /// `Unknown(...)`, whether the `generation_time_parameters` block
    /// is present or absent. When it's present, `raw` holds the verbatim
    /// JSON; when it's missing, `raw` is `Value::Null`. Text and audio
    /// manifests always land in the typed `Text`/`Audio` variants, even
    /// when `generation_time_parameters` is absent (the fields just
    /// default to `None`).
    Other { raw: serde_json::Value },
}

impl Manifest {
    /// Parse a manifest from a file on disk. Thin convenience over
    /// `from_str` that also captures a nicer error context.
    ///
    /// Requires the `std-fs` feature (default-on). WASM / embedded
    /// consumers without filesystem access should read the manifest
    /// bytes externally and call [`Self::from_bytes`].
    #[cfg(feature = "std-fs")]
    pub fn from_file(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("reading manifest file: {}", path.display()))?;
        Self::from_bytes(&bytes)
            .with_context(|| format!("parsing manifest file: {}", path.display()))
    }

    /// Parse a manifest from raw JSON bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let raw: serde_json::Value =
            serde_json::from_slice(bytes).context("manifest is not valid JSON")?;
        Self::from_value(raw)
    }

    /// Parse a manifest from an already-deserialized `serde_json::Value`.
    /// Keeps the original value around in `Manifest::raw` for forward-compat.
    pub fn from_value(raw: serde_json::Value) -> Result<Self> {
        // Use the typed shadow struct for validation, but keep `raw` for
        // anything we don't model (future fields, inference-specific
        // extras, etc.).
        let shadow: RawManifest = serde_json::from_value(raw.clone())
            .context("manifest doesn't match the expected LeapBundles schema")?;

        let files = ManifestFiles::from_raw(&shadow.load_time_parameters)?;
        let chat_template = shadow.load_time_parameters.chat_template.clone();
        let inference_type = InferenceType::parse_str(&shadow.inference_type);
        let generation_defaults = GenerationDefaults::from_raw(
            &inference_type,
            shadow.generation_time_parameters.as_ref(),
        );

        Ok(Self {
            inference_type,
            schema_version: shadow.schema_version,
            files,
            chat_template,
            generation_defaults,
            raw,
        })
    }

    /// Synthesize a minimal text-only manifest for a bare `.gguf` file.
    /// Callers who hand `WickEngine::from_path` a `.gguf` path will go
    /// through this — keeps downstream loader dispatch uniform regardless
    /// of how the session was initiated.
    pub fn synthetic_text(model_path: &Path) -> Self {
        let model_path_str = model_path.to_string_lossy().into_owned();
        let mut raw_map = serde_json::Map::new();
        raw_map.insert(
            "inference_type".into(),
            serde_json::Value::String("llama.cpp/text-to-text".into()),
        );
        raw_map.insert(
            "schema_version".into(),
            serde_json::Value::String("1.0.0".into()),
        );
        let mut load_params = serde_json::Map::new();
        load_params.insert(
            "model".into(),
            serde_json::Value::String(model_path_str.clone()),
        );
        raw_map.insert(
            "load_time_parameters".into(),
            serde_json::Value::Object(load_params),
        );
        let raw = serde_json::Value::Object(raw_map);

        Self {
            inference_type: InferenceType::LlamaCppTextToText,
            schema_version: "1.0.0".into(),
            files: ManifestFiles {
                model: model_path_str,
                multimodal_projector: None,
                audio_decoder: None,
                audio_tokenizer: None,
                extras: HashMap::new(),
            },
            chat_template: None,
            generation_defaults: GenerationDefaults::Text {
                temperature: None,
                min_p: None,
                top_p: None,
                top_k: None,
                repetition_penalty: None,
            },
            raw,
        }
    }

    /// Is this manifest's inference type currently supported for loading?
    /// `true` for text + audio; `false` for VL (parser OK, loader v2) +
    /// Unknown. Consumers can probe before constructing a `WickEngine`.
    pub fn is_loadable(&self) -> bool {
        matches!(
            self.inference_type,
            InferenceType::LlamaCppTextToText | InferenceType::LlamaCppLfm2AudioV1
        )
    }

    /// Resolve every file URL / path the manifest references to a
    /// (role_name, value) pair, in a stable order (primary first, then
    /// typed aux in mmproj/decoder/tokenizer order, then extras
    /// alphabetically). Used by the downloader (Phase 1.6) and by
    /// `from_files` callers that want to round-trip through manifest
    /// form.
    pub fn files_in_order(&self) -> Vec<(&str, &str)> {
        // Capacity: 1 (model) + up to 3 typed aux slots + extras.
        let mut out: Vec<(&str, &str)> = Vec::with_capacity(1 + 3 + self.files.extras.len());
        out.push(("model", self.files.model.as_str()));
        if let Some(v) = &self.files.multimodal_projector {
            out.push(("multimodal_projector", v.as_str()));
        }
        if let Some(v) = &self.files.audio_decoder {
            out.push(("audio_decoder", v.as_str()));
        }
        if let Some(v) = &self.files.audio_tokenizer {
            out.push(("audio_tokenizer", v.as_str()));
        }
        let mut extras: Vec<(&String, &String)> = self.files.extras.iter().collect();
        extras.sort_by(|a, b| a.0.cmp(b.0));
        for (k, v) in extras {
            out.push((k.as_str(), v.as_str()));
        }
        out
    }
}

impl ManifestFiles {
    fn from_raw(raw: &RawLoadTimeParameters) -> Result<Self> {
        let model = raw
            .other
            .get("model")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("manifest load_time_parameters missing required `model` field"))?
            .to_string();

        // Pull the typed aux fields out of the `other` map so `extras`
        // ends up with only genuinely novel roles.
        let take_str = |key: &str| -> Option<String> {
            raw.other
                .get(key)
                .and_then(|v| v.as_str())
                .map(str::to_string)
        };

        let multimodal_projector = take_str("multimodal_projector");
        let audio_decoder = take_str("audio_decoder");
        let audio_tokenizer = take_str("audio_tokenizer");

        const KNOWN_KEYS: &[&str] = &[
            "model",
            "multimodal_projector",
            "audio_decoder",
            "audio_tokenizer",
        ];
        // `chat_template` is an explicit field on `RawLoadTimeParameters`,
        // so serde places it there directly; `#[serde(flatten)]` only
        // sends the remaining unrecognized keys into `other`. No need to
        // filter for it here.
        //
        // `extras` is a subset of `other` — same upper bound on size.
        let mut extras = HashMap::with_capacity(raw.other.len());
        for (k, v) in &raw.other {
            if KNOWN_KEYS.contains(&k.as_str()) {
                continue;
            }
            if let Some(s) = v.as_str() {
                extras.insert(k.clone(), s.to_string());
            }
            // Non-string extras are silently dropped from the typed view
            // (they still live in `Manifest::raw` for consumers that
            // want them).
        }

        Ok(Self {
            model,
            multimodal_projector,
            audio_decoder,
            audio_tokenizer,
            extras,
        })
    }

    /// Convert every referenced file URL or path to a local `PathBuf`,
    /// assuming the URL-to-path mapping scheme used by the future
    /// `BundleRepo` (`<cache_root>/huggingface.co/<owner>/<repo>/<file>`).
    /// `local_root_for_url` gets called for each URL; the caller decides
    /// the policy. Returns role → path pairs in the same order as
    /// [`Manifest::files_in_order`].
    pub fn resolve_local<F>(&self, mut local_root_for_url: F) -> Vec<(String, PathBuf)>
    where
        F: FnMut(&str) -> PathBuf,
    {
        // Capacity: 1 (model) + up to 3 typed aux slots + extras.
        let mut out: Vec<(String, PathBuf)> = Vec::with_capacity(1 + 3 + self.extras.len());
        out.push(("model".into(), local_root_for_url(&self.model)));
        if let Some(v) = &self.multimodal_projector {
            out.push(("multimodal_projector".into(), local_root_for_url(v)));
        }
        if let Some(v) = &self.audio_decoder {
            out.push(("audio_decoder".into(), local_root_for_url(v)));
        }
        if let Some(v) = &self.audio_tokenizer {
            out.push(("audio_tokenizer".into(), local_root_for_url(v)));
        }
        let mut extras: Vec<(&String, &String)> = self.extras.iter().collect();
        extras.sort_by(|a, b| a.0.cmp(b.0));
        for (k, v) in extras {
            out.push((k.clone(), local_root_for_url(v)));
        }
        out
    }
}

impl GenerationDefaults {
    fn from_raw(inference_type: &InferenceType, raw: Option<&serde_json::Value>) -> Self {
        let Some(raw) = raw else {
            // Missing block: pick the typed-but-empty variant matching
            // the inference type, so consumers can write uniform code.
            return match inference_type {
                InferenceType::LlamaCppLfm2AudioV1 => Self::Audio {
                    number_of_decoding_threads: None,
                },
                InferenceType::LlamaCppTextToText | InferenceType::LlamaCppImageToText => {
                    Self::Text {
                        temperature: None,
                        min_p: None,
                        top_p: None,
                        top_k: None,
                        repetition_penalty: None,
                    }
                }
                // Unknown inference type + missing params → Other with
                // a Null raw. `Text` defaults would be misleading.
                InferenceType::Unknown(_) => Self::Other {
                    raw: serde_json::Value::Null,
                },
            };
        };

        match inference_type {
            InferenceType::LlamaCppLfm2AudioV1 => {
                let ndt = raw
                    .get("number_of_decoding_threads")
                    .and_then(|v| v.as_u64())
                    .and_then(|n| u32::try_from(n).ok());
                Self::Audio {
                    number_of_decoding_threads: ndt,
                }
            }
            InferenceType::LlamaCppTextToText | InferenceType::LlamaCppImageToText => {
                let sp = raw.get("sampling_parameters");
                let f32_at = |key: &str| -> Option<f32> {
                    sp.and_then(|o| o.get(key))
                        .and_then(|v| v.as_f64())
                        .map(|v| v as f32)
                };
                let u32_at = |key: &str| -> Option<u32> {
                    sp.and_then(|o| o.get(key))
                        .and_then(|v| v.as_u64())
                        .and_then(|n| u32::try_from(n).ok())
                };
                Self::Text {
                    temperature: f32_at("temperature"),
                    min_p: f32_at("min_p"),
                    top_p: f32_at("top_p"),
                    top_k: u32_at("top_k"),
                    repetition_penalty: f32_at("repetition_penalty"),
                }
            }
            InferenceType::Unknown(_) => Self::Other { raw: raw.clone() },
        }
    }
}

// ---------------------------------------------------------------------------
// Serde shadow types — kept private so the `Manifest` public shape can
// evolve without breaking serde integrations.
// ---------------------------------------------------------------------------

#[derive(Deserialize, Serialize, Debug)]
struct RawManifest {
    inference_type: String,
    schema_version: String,
    load_time_parameters: RawLoadTimeParameters,
    #[serde(default)]
    generation_time_parameters: Option<serde_json::Value>,
}

#[derive(Deserialize, Serialize, Debug)]
struct RawLoadTimeParameters {
    #[serde(default)]
    chat_template: Option<String>,
    /// Every other key — `model`, aux file roles, future extensions.
    #[serde(flatten)]
    other: HashMap<String, serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inference_type_roundtrip() {
        for s in [
            "llama.cpp/text-to-text",
            "llama.cpp/image-to-text",
            "llama.cpp/lfm2-audio-v1",
            "llama.cpp/some-new-modality",
        ] {
            let t = InferenceType::parse_str(s);
            assert_eq!(t.as_str(), s, "round-trip failed for {s}");
        }
    }

    #[test]
    fn synthetic_text_shape() {
        let p = std::path::Path::new("/tmp/model.gguf");
        let m = Manifest::synthetic_text(p);
        assert_eq!(m.inference_type, InferenceType::LlamaCppTextToText);
        assert_eq!(m.files.model, "/tmp/model.gguf");
        assert!(m.files.multimodal_projector.is_none());
        assert!(m.chat_template.is_none());
        assert!(m.is_loadable());
    }

    #[test]
    fn files_in_order_stable() {
        let m = ManifestFiles {
            model: "m.gguf".into(),
            multimodal_projector: Some("mm.gguf".into()),
            audio_decoder: Some("ad.gguf".into()),
            audio_tokenizer: Some("at.safetensors".into()),
            extras: {
                let mut e = HashMap::new();
                e.insert("zzz_future".into(), "z.bin".into());
                e.insert("aaa_novel".into(), "a.bin".into());
                e
            },
        };
        let manifest = Manifest {
            inference_type: InferenceType::LlamaCppLfm2AudioV1,
            schema_version: "1.0.0".into(),
            files: m,
            chat_template: None,
            generation_defaults: GenerationDefaults::Audio {
                number_of_decoding_threads: Some(4),
            },
            raw: serde_json::Value::Null,
        };
        let out: Vec<_> = manifest
            .files_in_order()
            .iter()
            .map(|(k, _)| k.to_string())
            .collect();
        assert_eq!(
            out,
            vec![
                "model",
                "multimodal_projector",
                "audio_decoder",
                "audio_tokenizer",
                "aaa_novel",
                "zzz_future",
            ]
        );
    }

    #[test]
    fn parse_fails_on_missing_model() {
        let bad = br#"{
            "inference_type": "llama.cpp/text-to-text",
            "schema_version": "1.0.0",
            "load_time_parameters": {}
        }"#;
        let err = Manifest::from_bytes(bad).unwrap_err();
        assert!(
            err.to_string().contains("missing required `model`"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn parse_fails_on_non_json() {
        let err = Manifest::from_bytes(b"not json").unwrap_err();
        assert!(
            err.to_string().to_lowercase().contains("not valid json"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn parse_defaults_missing_generation_params() {
        let min = br#"{
            "inference_type": "llama.cpp/text-to-text",
            "schema_version": "1.0.0",
            "load_time_parameters": { "model": "m.gguf" }
        }"#;
        let m = Manifest::from_bytes(min).unwrap();
        match &m.generation_defaults {
            GenerationDefaults::Text {
                temperature,
                min_p,
                top_p,
                top_k,
                repetition_penalty,
            } => {
                assert!(temperature.is_none());
                assert!(min_p.is_none());
                assert!(top_p.is_none());
                assert!(top_k.is_none());
                assert!(repetition_penalty.is_none());
            }
            other => panic!("expected Text defaults, got {other:?}"),
        }
    }
}
