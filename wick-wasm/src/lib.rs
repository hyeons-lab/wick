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

/// Map a `wick::WickError` into a `JsError`. Uses `Display` (not `Debug`)
/// so JS callers see the same message a wick CLI consumer would. Kept
/// distinct from `map_err` (which handles `anyhow::Error`) so the call
/// sites stay readable — both helpers throw the same `JsError` shape on
/// the JS side.
fn map_wick_err(err: wick::WickError) -> JsError {
    JsError::new(&err.to_string())
}

/// Loaded inference engine — wraps `wick::WickEngine` with sync access
/// to model metadata and the tokenizer.
///
/// JS callers fetch the GGUF (e.g. via `fetch().arrayBuffer()`), pass
/// the bytes to [`WickEngine.fromGgufBytes`], and use the returned
/// handle to read model info or pull a `Tokenizer`. Session-based
/// inference (`generate`, streaming) is intentionally not exposed yet
/// — that shape needs an async/streaming design that lives in a
/// follow-up PR.
///
/// **Memory:** the loaded GGUF stays resident in wasm linear memory
/// for the lifetime of this object. Call `.free()` (auto-emitted by
/// wasm-bindgen) when done to release it; without that, the entire
/// model lives until the page unloads.
#[wasm_bindgen]
pub struct WickEngine {
    inner: wick::WickEngine,
}

#[wasm_bindgen]
impl WickEngine {
    /// Load a model from in-memory GGUF bytes. `contextSize` defaults
    /// to 4096 if omitted; the actual KV-cache cap is the smaller of
    /// the requested size and the model's own `max_seq_len`.
    ///
    /// The backend is forced to CPU — wasm has no native GPU/Metal
    /// backend. Throws on parse failure, unsupported quantization,
    /// or unrecognized architecture.
    #[wasm_bindgen(js_name = fromGgufBytes)]
    pub fn from_gguf_bytes(
        bytes: Vec<u8>,
        context_size: Option<u32>,
    ) -> Result<WickEngine, JsError> {
        // Spread `..Default::default()` so the wrapper picks up any
        // future EngineConfig fields (e.g. `bundle_repo` when the
        // `remote` feature is on) without a compile break — only the
        // two we actually want to override are spelled out.
        let cfg = wick::EngineConfig {
            context_size: context_size.unwrap_or(4096) as usize,
            backend: wick::BackendPreference::Cpu,
            ..wick::EngineConfig::default()
        };
        wick::WickEngine::from_bytes(bytes, cfg)
            .map(|inner| WickEngine { inner })
            .map_err(map_wick_err)
    }

    /// Model architecture string from the GGUF metadata
    /// (e.g. `"lfm2"`, `"llama"`).
    #[wasm_bindgen(getter)]
    pub fn architecture(&self) -> String {
        self.inner.metadata().architecture.clone()
    }

    /// Maximum sequence length the model was trained for. Independent
    /// of the engine's `contextSize` config — that one is the KV
    /// cache cap, this is the model's positional encoding ceiling.
    #[wasm_bindgen(getter, js_name = maxSeqLen)]
    pub fn max_seq_len(&self) -> u32 {
        self.inner.metadata().max_seq_len
    }

    #[wasm_bindgen(getter, js_name = vocabSize)]
    pub fn vocab_size(&self) -> u32 {
        self.inner.metadata().vocab_size
    }

    /// Quantization label from the GGUF (e.g. `"Q4_0"`, `"Q4_K_M"`).
    /// Useful for telling users what they actually loaded when the
    /// download URL doesn't make it obvious.
    #[wasm_bindgen(getter)]
    pub fn quantization(&self) -> String {
        self.inner.metadata().quantization.clone()
    }

    /// `true` when the loaded GGUF carries an embedded Jinja chat
    /// template. JS callers can use this to decide whether to render
    /// `Tokenizer.chatTemplate` themselves vs falling back to a
    /// hard-coded prompt format.
    #[wasm_bindgen(getter, js_name = hasChatTemplate)]
    pub fn has_chat_template(&self) -> bool {
        self.inner.metadata().has_chat_template
    }

    /// `true` when the GGUF declares `tokenizer.ggml.add_bos_token`.
    /// Callers that hand-build a token sequence from `Tokenizer.encode`
    /// should prepend `Tokenizer.bosToken` when this is `true` (and
    /// the model has a BOS) — wick's encoder returns the raw tokens
    /// without that prefix.
    #[wasm_bindgen(getter, js_name = addBosToken)]
    pub fn add_bos_token(&self) -> bool {
        self.inner.metadata().add_bos_token
    }

    /// Returns a `Tokenizer` handle bound to this engine's vocab.
    /// Each call allocates a fresh JS object but the underlying
    /// tokenizer state is shared via `Arc` — cheap to call, JS
    /// callers can cache the result if they prefer one handle.
    #[wasm_bindgen(getter)]
    pub fn tokenizer(&self) -> Tokenizer {
        Tokenizer {
            inner: self.inner.tokenizer_arc(),
        }
    }
}

/// BPE tokenizer wrapper. Constructed via `WickEngine.tokenizer`;
/// no standalone `from*` constructor (the GGUF metadata required to
/// build one is reachable only through the engine).
///
/// Round-trip note: `decode(encode(text))` is **not** guaranteed to
/// be byte-identical to `text` for inputs containing tokens that
/// don't survive BPE merge replay (rare in practice — BOS/EOS,
/// some byte-level edge cases). When you need exact reproduction,
/// keep the original string around.
#[wasm_bindgen]
pub struct Tokenizer {
    inner: std::sync::Arc<wick::tokenizer::BpeTokenizer>,
}

#[wasm_bindgen]
impl Tokenizer {
    /// Tokenize a UTF-8 string. Returns the token IDs as a
    /// `Uint32Array`. No BOS/EOS prefix — callers that want them
    /// should prepend `bosToken` / append `eosToken` manually.
    #[wasm_bindgen]
    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.inner.encode(text)
    }

    /// Detokenize back to a UTF-8 string. Lossy for tokens whose
    /// byte sequences don't decode to valid UTF-8 — those are
    /// replaced with U+FFFD per `String::from_utf8_lossy`.
    #[wasm_bindgen]
    pub fn decode(&self, tokens: &[u32]) -> String {
        self.inner.decode(tokens)
    }

    #[wasm_bindgen(getter, js_name = vocabSize)]
    pub fn vocab_size(&self) -> u32 {
        self.inner.vocab_size() as u32
    }

    /// BOS token ID, if the GGUF metadata declares one.
    #[wasm_bindgen(getter, js_name = bosToken)]
    pub fn bos_token(&self) -> Option<u32> {
        self.inner.bos_token()
    }

    /// EOS token ID, if the GGUF metadata declares one.
    #[wasm_bindgen(getter, js_name = eosToken)]
    pub fn eos_token(&self) -> Option<u32> {
        self.inner.eos_token()
    }

    /// Embedded Jinja chat template, if the GGUF metadata carries
    /// one. Apply it yourself with a Jinja runtime (e.g.
    /// `nunjucks`) to convert a chat-message list into the
    /// model's expected prompt string.
    #[wasm_bindgen(getter, js_name = chatTemplate)]
    pub fn chat_template(&self) -> Option<String> {
        self.inner.chat_template().map(str::to_owned)
    }
}
