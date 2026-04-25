//! wasm-bindgen wrapper around the `wick` core inference engine.
//!
//! This crate produces the `.wasm` cdylib that browser / Node consumers
//! drive via the JS glue emitted by `wasm-bindgen-cli`. Native consumers
//! should use the `wick` crate directly — wick-wasm exists purely to
//! map `wick`'s Rust API onto the JS interop boundary.
//!
//! The lib body is `cfg(target_arch = "wasm32")`-gated: on native
//! targets this crate compiles to an empty cdylib, which keeps
//! workspace-wide commands (`cargo check --workspace`,
//! `cargo clippy --workspace`) honest without needing to special-case
//! the wasm wrapper.

#![cfg(target_arch = "wasm32")]

use wasm_bindgen::prelude::*;

/// Returns the version of the `wick` core library this binding wraps.
///
/// Note this is **`wick`'s** version, not `wick-wasm`'s — JS callers
/// usually want to know what core lib is driving the engine, since
/// the wrapper crate version may evolve independently.
#[wasm_bindgen(js_name = wickVersion)]
pub fn wick_version() -> String {
    wick::VERSION.to_string()
}

/// Map an `anyhow::Error` into a `JsError` preserving the full
/// `{:#}` chain. Centralised so every wrapper surface throws the
/// same shape.
fn map_err(err: anyhow::Error) -> JsError {
    JsError::new(&format!("{err:#}"))
}

/// Parsed view of a LeapBundles `*.json` manifest.
///
/// JS callers fetch the manifest bytes (e.g. via `fetch().arrayBuffer()`)
/// and pass them to `Manifest.parse`. The wrapper exposes the typed
/// fields wick already understands; the raw `serde_json::Value`
/// retained on the inner `wick::manifest::Manifest` is intentionally
/// **not** exposed here — JS callers can re-parse the JSON themselves
/// for forward-compat fields, and we don't want to commit to a
/// `serde-wasm-bindgen` round-trip on every getter.
#[wasm_bindgen]
pub struct Manifest {
    inner: wick::manifest::Manifest,
}

#[wasm_bindgen]
impl Manifest {
    /// Parse a JSON manifest from raw bytes. Throws a `JsError` on
    /// malformed JSON or when required fields are missing or wrongly
    /// typed (e.g. no `load_time_parameters.model`). Unknown
    /// `inference_type` values are **not** an error — they round-trip
    /// through `wick::manifest::InferenceType::Unknown(String)` and
    /// surface verbatim via the `inferenceType` getter, so JS callers
    /// can decide how to react instead of catching here.
    #[wasm_bindgen]
    pub fn parse(json_bytes: &[u8]) -> Result<Manifest, JsError> {
        wick::manifest::Manifest::from_bytes(json_bytes)
            .map(|inner| Manifest { inner })
            .map_err(map_err)
    }

    /// Raw `inference_type` string (e.g. `llama.cpp/text-to-text`).
    /// Round-trips through wick's enum, so unknown variants come back
    /// as their original string — no information loss.
    #[wasm_bindgen(getter, js_name = inferenceType)]
    pub fn inference_type(&self) -> String {
        self.inner.inference_type.as_str().to_string()
    }

    #[wasm_bindgen(getter, js_name = schemaVersion)]
    pub fn schema_version(&self) -> String {
        self.inner.schema_version.clone()
    }

    /// URL (or local path string) for the primary model GGUF.
    #[wasm_bindgen(getter, js_name = modelUrl)]
    pub fn model_url(&self) -> String {
        self.inner.files.model.clone()
    }

    /// URL of the multimodal projector GGUF if the manifest declares
    /// one (VL / audio models). `undefined` for plain text models.
    #[wasm_bindgen(getter, js_name = multimodalProjectorUrl)]
    pub fn multimodal_projector_url(&self) -> Option<String> {
        self.inner.files.multimodal_projector.clone()
    }

    /// URL of the audio-decoder GGUF for audio-out models.
    #[wasm_bindgen(getter, js_name = audioDecoderUrl)]
    pub fn audio_decoder_url(&self) -> Option<String> {
        self.inner.files.audio_decoder.clone()
    }

    /// URL of the audio-tokenizer checkpoint (typically `.safetensors`).
    #[wasm_bindgen(getter, js_name = audioTokenizerUrl)]
    pub fn audio_tokenizer_url(&self) -> Option<String> {
        self.inner.files.audio_tokenizer.clone()
    }

    /// Jinja chat template override from the manifest, if present.
    /// `undefined` means "use the template embedded in the GGUF
    /// metadata" (wick's standard fallback).
    #[wasm_bindgen(getter, js_name = chatTemplate)]
    pub fn chat_template(&self) -> Option<String> {
        self.inner.chat_template.clone()
    }
}
