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

use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;

// Custom TypeScript declarations injected into the generated
// `wick_wasm.d.ts`. wasm-bindgen would otherwise type wasm-side
// `JsValue` parameters as `any`, losing IDE completion + type
// checking for the structured shapes we expect from JS callers.
//
// Each `extern "C" { #[wasm_bindgen(typescript_type = "...")]
// pub type T; }` block below declares a Rust-side opaque handle
// whose only purpose is to carry a custom TS type label. At
// runtime these are still plain `JsValue`s.
#[wasm_bindgen(typescript_custom_section)]
const TS_APPEND: &'static str = r#"
/**
 * One entry in the chat-message array passed to
 * `Tokenizer.applyChatTemplate`. Mirrors the OpenAI / Anthropic
 * SDK shape — wick currently models only `role` and `content`;
 * tool-calls / function-calls are not yet supported.
 */
export interface ChatMessage {
    role: string;
    content: string;
}
"#;

#[wasm_bindgen]
extern "C" {
    /// Opaque type-label wrapper for `ChatMessage[]` — the
    /// argument shape `Tokenizer.applyChatTemplate` accepts.
    /// At the wasm boundary this is just a JsValue array; the
    /// generated .d.ts surfaces it as `ChatMessage[]` so JS/TS
    /// callers get IDE completion + type checking.
    #[wasm_bindgen(typescript_type = "ChatMessage[]")]
    pub type ChatMessageArray;
}

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

    /// Construct a new `Session` for this engine. The `config`
    /// freezes per-session knobs — sampler `seed`, `nKeep`
    /// pinned-prefix size, `ubatchSize` chunked-prefill batch,
    /// `maxSeqLen` KV cap. For the wick defaults
    /// (`maxSeqLen = null` → engine's effective cap, i.e.
    /// `min(engine.contextSize, model.maxSeqLen)`; `nKeep = 0`,
    /// `seed = null`, `ubatchSize = 512`), pass a freshly-
    /// constructed `new SessionConfig()`.
    ///
    /// `config` is **borrowed**, not consumed — JS callers can
    /// reuse the same `SessionConfig` across multiple `newSession`
    /// calls. Inner state is cloned per-session at the boundary.
    /// This mirrors how `Session.generate` borrows `GenerateOpts`.
    /// (wasm-bindgen doesn't support `Option<&T>` for wrapper
    /// types, so a default-config caller passes
    /// `new SessionConfig()` rather than omitting the arg.)
    ///
    /// The returned `Session` keeps its own `Arc` clones of the
    /// engine's model and tokenizer, so freeing the engine doesn't
    /// invalidate any in-flight sessions.
    #[wasm_bindgen(js_name = newSession)]
    pub fn new_session(&self, config: &SessionConfig) -> Session {
        Session {
            inner: self.inner.new_session(config.inner.clone()),
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

    /// Raw embedded Jinja chat template from the GGUF metadata, if
    /// any. Most callers should use [`Self::apply_chat_template`]
    /// (`applyChatTemplate` in JS) instead — this getter is for
    /// inspection or for callers who want to render with a
    /// different Jinja runtime.
    #[wasm_bindgen(getter, js_name = chatTemplate)]
    pub fn chat_template(&self) -> Option<String> {
        self.inner.chat_template().map(str::to_owned)
    }

    /// Render the model's embedded Jinja chat template against a
    /// `[{ role, content }, ...]` array, returning the prompt
    /// string ready for `Tokenizer.encode` + `Session.appendTokens`.
    ///
    /// `addGenerationPrompt` defaults to `true` (the common case
    /// when sending to the model expecting a response). Set to
    /// `false` when you only want the conversation rendered without
    /// the trailing assistant-prompt suffix.
    ///
    /// Throws `JsError` on:
    /// - the model not carrying a chat template
    ///   (`engine.hasChatTemplate === false`),
    /// - malformed `messages` (not an array, or entries missing
    ///   `role`/`content` strings),
    /// - a Jinja render failure (template references an undefined
    ///   variable, etc.).
    #[wasm_bindgen(js_name = applyChatTemplate)]
    pub fn apply_chat_template(
        &self,
        messages: ChatMessageArray,
        add_generation_prompt: Option<bool>,
    ) -> Result<String, JsError> {
        // `ChatMessageArray` is a wasm-bindgen opaque type-label
        // wrapper around `JsValue` — the runtime check + parse
        // still happens in `parse_chat_messages`. The TS-side
        // win is purely surface (callers get `ChatMessage[]`
        // instead of `any`).
        let msgs = parse_chat_messages(messages.as_ref())?;
        wick::tokenizer::apply_chat_template(
            &self.inner,
            &msgs,
            add_generation_prompt.unwrap_or(true),
        )
        .map_err(map_err)
    }
}

/// Parse a JS-side `[{ role, content }, ...]` array into the wick
/// core type, using `js_sys::Reflect` directly rather than going
/// through `serde-wasm-bindgen`. Both approaches were measured —
/// they produce **the same wasm size** (the size growth from
/// `apply_chat_template` is dominated by minijinja's render path,
/// not the deserialiser). The manual `Reflect` walk is preferred
/// here because it keeps the dep graph smaller (one less crate to
/// audit + faster cold builds) for two flat string fields. If a
/// future surface needs rich nested deserialisation, revisit and
/// add `serde-wasm-bindgen` then.
fn parse_chat_messages(value: &JsValue) -> Result<Vec<wick::tokenizer::ChatMessage>, JsError> {
    let array = value
        .dyn_ref::<js_sys::Array>()
        .ok_or_else(|| JsError::new("messages must be an array"))?;
    // `js_sys::Array::length` returns `u32` and that's the index
    // type `Array::get` takes — keep `len` in `u32` for the loop
    // and only widen to `usize` at the `Vec::with_capacity` call.
    let len = array.length();
    let mut msgs = Vec::with_capacity(len as usize);
    let role_key = JsValue::from_str("role");
    let content_key = JsValue::from_str("content");
    for i in 0..len {
        let entry = array.get(i);
        msgs.push(wick::tokenizer::ChatMessage {
            role: read_string_field(&entry, &role_key, "role", i)?,
            content: read_string_field(&entry, &content_key, "content", i)?,
        });
    }
    Ok(msgs)
}

/// Read a string-typed field off a JS object, distinguishing the
/// three failure modes a JS caller will commonly hit so the thrown
/// `JsError` actually points at the bug:
///   - `entry` is not an object (`Reflect::get` errors)
///   - the field is missing (`Reflect::get` returns `undefined`)
///   - the field is present but not a string
///
/// `js_sys::Reflect::get` only `Err`s on the first case (proxy
/// throws, target not Object); missing-property-on-an-Object
/// returns `Ok(JsValue::UNDEFINED)`, which would otherwise
/// silently fall through to a misleading "must be a string"
/// message. Splitting the cases keeps `messages[i].role missing`
/// distinguishable from `messages[i].role must be a string`.
fn read_string_field(
    entry: &JsValue,
    key: &JsValue,
    field_name: &str,
    index: u32,
) -> Result<String, JsError> {
    let value = js_sys::Reflect::get(entry, key)
        .map_err(|_| JsError::new(&format!("messages[{index}] is not an object")))?;
    if value.is_undefined() {
        return Err(JsError::new(&format!(
            "messages[{index}] missing '{field_name}' field"
        )));
    }
    value
        .as_string()
        .ok_or_else(|| JsError::new(&format!("messages[{index}].{field_name} must be a string")))
}

// ---------------------------------------------------------------------------
// Session + generate
// ---------------------------------------------------------------------------

/// Per-session knobs frozen at `WickEngine.newSession(config)` time.
/// Constructed via `new SessionConfig()` in JS (returns the wick
/// defaults: `maxSeqLen=null` → engine's effective max, `nKeep=0`,
/// `seed=null`, `ubatchSize=512`, `kvCompression=null`).
///
/// Set `kvCompression` to a [`TurboQuantConfig`] to compress the
/// KV cache (~3 bits/elem for keys, ~2 bits/elem for values).
/// See the per-property doc for trade-offs.
#[wasm_bindgen]
#[derive(Default, Clone)]
pub struct SessionConfig {
    inner: wick::SessionConfig,
}

#[wasm_bindgen]
impl SessionConfig {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Cap on total tokens held in KV. `null` (the common case)
    /// defers to the engine's effective max — i.e.
    /// `min(engine.contextSize, model.maxSeqLen)`. Set to a
    /// smaller value here to further lower the cap; values larger
    /// than the engine's effective max are still capped at it.
    #[wasm_bindgen(getter, js_name = maxSeqLen)]
    pub fn max_seq_len(&self) -> Option<u32> {
        self.inner.max_seq_len
    }
    #[wasm_bindgen(setter, js_name = maxSeqLen)]
    pub fn set_max_seq_len(&mut self, v: Option<u32>) {
        self.inner.max_seq_len = v;
    }

    /// Number of leading tokens pinned in KV across context shifts —
    /// a system prompt or persistent prefix that should survive
    /// when the cache fills. `0` (default) disables the pin.
    #[wasm_bindgen(getter, js_name = nKeep)]
    pub fn n_keep(&self) -> u32 {
        self.inner.n_keep
    }
    #[wasm_bindgen(setter, js_name = nKeep)]
    pub fn set_n_keep(&mut self, v: u32) {
        self.inner.n_keep = v;
    }

    /// Deterministic sampler seed. `null` (default) uses a fresh
    /// random seed per session — set this to make a session's
    /// outputs reproducible across runs (useful for testing /
    /// demos / regression checks).
    #[wasm_bindgen(getter)]
    pub fn seed(&self) -> Option<u64> {
        self.inner.seed
    }
    #[wasm_bindgen(setter)]
    pub fn set_seed(&mut self, v: Option<u64>) {
        self.inner.seed = v;
    }

    /// Chunked-prefill batch size (tokens per micro-batch during
    /// the prefill pass). Smaller values give finer-grained
    /// `Session.cancel()` checkpoints during long prompt eval at
    /// some perf cost. wick's default is `512`.
    #[wasm_bindgen(getter, js_name = ubatchSize)]
    pub fn ubatch_size(&self) -> u32 {
        self.inner.ubatch_size
    }
    #[wasm_bindgen(setter, js_name = ubatchSize)]
    pub fn set_ubatch_size(&mut self, v: u32) {
        self.inner.ubatch_size = v;
    }

    /// KV cache compression configuration. `null` (default) stores
    /// keys and values as f32 — best fidelity, biggest memory
    /// footprint. Set to a [`TurboQuantConfig`] to enable
    /// TurboQuant compression — keys to ~3 bits/elem, values to
    /// ~2 bits/elem (plus f16 norms per block); the same `seed`
    /// reproduces the same per-layer Hadamard rotations
    /// deterministically.
    ///
    /// Setting this consumes the JS-side `TurboQuantConfig`
    /// handle (wasm-bindgen's `Option<T>` parameter shape). Read
    /// back via the getter — which returns a fresh handle — if
    /// you need to inspect the current config.
    #[wasm_bindgen(getter, js_name = kvCompression)]
    pub fn kv_compression(&self) -> Option<TurboQuantConfig> {
        match &self.inner.kv_compression {
            wick::kv_cache::KvCompression::None => None,
            wick::kv_cache::KvCompression::TurboQuant { seed, keys, values } => {
                Some(TurboQuantConfig {
                    seed: *seed,
                    keys: *keys,
                    values: *values,
                })
            }
        }
    }
    #[wasm_bindgen(setter, js_name = kvCompression)]
    pub fn set_kv_compression(&mut self, v: Option<TurboQuantConfig>) {
        self.inner.kv_compression = match v {
            None => wick::kv_cache::KvCompression::None,
            Some(tqc) => wick::kv_cache::KvCompression::TurboQuant {
                seed: tqc.seed,
                keys: tqc.keys,
                values: tqc.values,
            },
        };
    }
}

/// TurboQuant KV-cache compression configuration. Construct via
/// `new TurboQuantConfig(seed)` for the common production setup
/// (both `keys` and `values` compressed); flip the per-side
/// toggles for debugging (e.g. to isolate how much drift each
/// side contributes).
///
/// - **Keys**: 2-bit PolarQuant + 1-bit QJL residual
///   (3 bits/elem + f16 norms per block).
/// - **Values**: 2-bit PolarQuant only (2 bits/elem + f16 norms
///   per block).
///
/// `seed` drives the per-layer randomized Hadamard rotations —
/// the same seed produces the same rotations deterministically,
/// so a seeded session with TurboQuant on stays bitwise-
/// reproducible across runs.
#[wasm_bindgen]
#[derive(Clone)]
pub struct TurboQuantConfig {
    seed: u64,
    keys: bool,
    values: bool,
}

#[wasm_bindgen]
impl TurboQuantConfig {
    /// Construct with the common production setup: both keys and
    /// values compressed. Pass an explicit `seed` so the per-layer
    /// rotations are reproducible.
    #[wasm_bindgen(constructor)]
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            keys: true,
            values: true,
        }
    }

    /// Hadamard-rotation seed. Same seed → same rotations →
    /// reproducible KV cache contents (necessary for bitwise-
    /// identical replay across sessions).
    #[wasm_bindgen(getter)]
    pub fn seed(&self) -> u64 {
        self.seed
    }
    #[wasm_bindgen(setter)]
    pub fn set_seed(&mut self, v: u64) {
        self.seed = v;
    }

    /// Compress the K side of the KV cache. Default `true`.
    /// Useful to flip off when debugging quality regressions to
    /// isolate K-side vs V-side contribution.
    #[wasm_bindgen(getter)]
    pub fn keys(&self) -> bool {
        self.keys
    }
    #[wasm_bindgen(setter)]
    pub fn set_keys(&mut self, v: bool) {
        self.keys = v;
    }

    /// Compress the V side of the KV cache. Default `true`.
    #[wasm_bindgen(getter)]
    pub fn values(&self) -> bool {
        self.values
    }
    #[wasm_bindgen(setter)]
    pub fn set_values(&mut self, v: bool) {
        self.values = v;
    }
}

/// Per-call generation options. Constructed via `new GenerateOpts()`
/// in JS (returns the wick defaults: `maxTokens=256`,
/// `temperature=0.7`, `topP=0.9`, `topK=40`, no stop tokens, flush
/// every 16 tokens or 50 ms).
///
/// `repetitionPenalty` is read-only — wick's sampler does not yet
/// honor it (deferred); exposing the setter would let JS callers pass
/// values that silently no-op.
#[wasm_bindgen]
#[derive(Default)]
pub struct GenerateOpts {
    inner: wick::GenerateOpts,
}

#[wasm_bindgen]
impl GenerateOpts {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::default()
    }

    #[wasm_bindgen(getter, js_name = maxTokens)]
    pub fn max_tokens(&self) -> u32 {
        self.inner.max_tokens
    }
    #[wasm_bindgen(setter, js_name = maxTokens)]
    pub fn set_max_tokens(&mut self, v: u32) {
        self.inner.max_tokens = v;
    }

    #[wasm_bindgen(getter)]
    pub fn temperature(&self) -> f32 {
        self.inner.temperature
    }
    #[wasm_bindgen(setter)]
    pub fn set_temperature(&mut self, v: f32) {
        self.inner.temperature = v;
    }

    #[wasm_bindgen(getter, js_name = topP)]
    pub fn top_p(&self) -> f32 {
        self.inner.top_p
    }
    #[wasm_bindgen(setter, js_name = topP)]
    pub fn set_top_p(&mut self, v: f32) {
        self.inner.top_p = v;
    }

    #[wasm_bindgen(getter, js_name = topK)]
    pub fn top_k(&self) -> u32 {
        self.inner.top_k
    }
    #[wasm_bindgen(setter, js_name = topK)]
    pub fn set_top_k(&mut self, v: u32) {
        self.inner.top_k = v;
    }

    /// Read-only — wick's sampler doesn't yet honor this field.
    /// Surfaced as a getter so JS callers can read the default
    /// (`1.0`); the setter is intentionally absent so callers don't
    /// pass values that silently no-op.
    #[wasm_bindgen(getter, js_name = repetitionPenalty)]
    pub fn repetition_penalty(&self) -> f32 {
        self.inner.repetition_penalty
    }

    /// Token IDs that, if produced, end decoding with
    /// `finishReason = "Stop"`. Empty by default.
    #[wasm_bindgen(getter, js_name = stopTokens)]
    pub fn stop_tokens(&self) -> Vec<u32> {
        self.inner.stop_tokens.clone()
    }
    #[wasm_bindgen(setter, js_name = stopTokens)]
    pub fn set_stop_tokens(&mut self, v: Vec<u32>) {
        self.inner.stop_tokens = v;
    }

    #[wasm_bindgen(getter, js_name = flushEveryTokens)]
    pub fn flush_every_tokens(&self) -> u32 {
        self.inner.flush_every_tokens
    }
    #[wasm_bindgen(setter, js_name = flushEveryTokens)]
    pub fn set_flush_every_tokens(&mut self, v: u32) {
        self.inner.flush_every_tokens = v;
    }

    #[wasm_bindgen(getter, js_name = flushEveryMs)]
    pub fn flush_every_ms(&self) -> u32 {
        self.inner.flush_every_ms
    }
    #[wasm_bindgen(setter, js_name = flushEveryMs)]
    pub fn set_flush_every_ms(&mut self, v: u32) {
        self.inner.flush_every_ms = v;
    }
}

/// Summary returned from a completed `Session.generate` call.
#[wasm_bindgen]
pub struct GenerateSummary {
    inner: wick::GenerateSummary,
}

#[wasm_bindgen]
impl GenerateSummary {
    #[wasm_bindgen(getter, js_name = tokensGenerated)]
    pub fn tokens_generated(&self) -> u32 {
        self.inner.tokens_generated
    }

    #[wasm_bindgen(getter, js_name = promptEvalTokens)]
    pub fn prompt_eval_tokens(&self) -> u32 {
        self.inner.prompt_eval_tokens
    }

    #[wasm_bindgen(getter, js_name = promptEvalMs)]
    pub fn prompt_eval_ms(&self) -> u32 {
        self.inner.prompt_eval_ms
    }

    #[wasm_bindgen(getter, js_name = decodeMs)]
    pub fn decode_ms(&self) -> u32 {
        self.inner.decode_ms
    }

    /// Why decode ended. One of `"MaxTokens"`, `"Stop"`,
    /// `"Cancelled"`, `"ContextFull"`, or `"Error(<message>)"` —
    /// the `Error(...)` form preserves the inner string verbatim
    /// (no surrounding quotes), so JS callers can log it directly.
    #[wasm_bindgen(getter, js_name = finishReason)]
    pub fn finish_reason(&self) -> String {
        // `format!("{:?}", reason)` would render `Error(String)` as
        // `Error("...")` (with the Debug-quoted inner string).
        // Match each variant explicitly so the public shape matches
        // the doc comment: `Error(plain inner message)` and bare
        // names for the payload-free variants.
        match &self.inner.finish_reason {
            wick::FinishReason::MaxTokens => "MaxTokens".to_string(),
            wick::FinishReason::Stop => "Stop".to_string(),
            wick::FinishReason::Cancelled => "Cancelled".to_string(),
            wick::FinishReason::ContextFull => "ContextFull".to_string(),
            wick::FinishReason::Error(msg) => format!("Error({msg})"),
        }
    }
}

/// Stateful generation handle. Built via `WickEngine.newSession(config)`.
///
/// JS callers seed the conversation by calling `appendText` /
/// `appendTokens` and then drive decode with `generate(opts, cb)`.
/// The callback fires once per flush boundary (every
/// `flushEveryTokens` decoded tokens, or `flushEveryMs` ms,
/// whichever comes first) with the new tokens.
///
/// **Worker note:** `generate` is synchronous and will block the
/// thread it runs on for the duration of decode (potentially
/// seconds). On the browser main thread that freezes the page —
/// always call from a Web Worker. On Node it also blocks the JS
/// event loop (libuv's background I/O thread pool keeps running,
/// but JS callbacks queue): use `worker_threads` for server
/// processes that need to handle other requests during inference;
/// one-off scripts are fine to run sync.
///
/// **Cancellation:** since the worker thread is blocked inside
/// `generate`, the worker's own `onmessage` handler can't run —
/// incoming `postMessage({kind:'cancel'})` queues but doesn't
/// dispatch until `generate` returns, so a flag set by that
/// handler can't be updated mid-decode. To cancel during a
/// running `generate` call, either call `session.cancel()` from inside
/// the token callback based on state it can observe directly
/// (elapsed time, token budget, accumulated content), or use
/// cross-thread shared memory signalling (`SharedArrayBuffer` +
/// `Atomics`) — see `wick-wasm/README.md` for the full
/// `SharedArrayBuffer` pattern, which requires cross-origin
/// isolation in browsers.
#[wasm_bindgen]
pub struct Session {
    inner: wick::Session,
}

#[wasm_bindgen]
impl Session {
    /// Tokenize `text` using the session's tokenizer and append the
    /// result to the KV cache. Equivalent to
    /// `appendTokens(tokenizer.encode(text))` but avoids the round
    /// trip through JS for the encoded buffer.
    #[wasm_bindgen(js_name = appendText)]
    pub fn append_text(&mut self, text: &str) -> Result<(), JsError> {
        self.inner.append_text(text).map_err(map_wick_err)
    }

    /// Append already-tokenized IDs to the KV cache. Use when you
    /// need control over BOS/EOS framing or you've cached tokens
    /// from a previous encode.
    #[wasm_bindgen(js_name = appendTokens)]
    pub fn append_tokens(&mut self, tokens: &[u32]) -> Result<(), JsError> {
        self.inner.append_tokens(tokens).map_err(map_wick_err)
    }

    /// Current KV cache position (number of tokens currently held).
    #[wasm_bindgen(getter)]
    pub fn position(&self) -> u32 {
        self.inner.position()
    }

    /// Flip the cancel atomic, requesting that any in-flight
    /// `generate` call exit at its next checkpoint with
    /// `finishReason = "Cancelled"`. Safe to call from any thread
    /// (including a Worker that owns this session — though wasm
    /// without SharedArrayBuffer makes cross-thread sharing
    /// unusual).
    #[wasm_bindgen]
    pub fn cancel(&self) {
        self.inner.cancel()
    }

    /// Decode tokens until `opts.maxTokens`, a stop token, EOS, or
    /// `cancel()` fires. The `onTextTokens` callback is invoked once
    /// per flush boundary with a `Uint32Array` of the latest tokens
    /// (*not* the cumulative buffer — concatenate yourself if you
    /// want the full sequence).
    ///
    /// Returns the `GenerateSummary` once decode finishes. Throws
    /// `JsError` on backend failure (the summary's `finishReason`
    /// already covers logical end conditions like `"Stop"` or
    /// `"ContextFull"`).
    #[wasm_bindgen]
    pub fn generate(
        &mut self,
        opts: &GenerateOpts,
        on_text_tokens: &js_sys::Function,
    ) -> Result<GenerateSummary, JsError> {
        let mut sink = JsTextSink {
            on_text: on_text_tokens,
        };
        self.inner
            .generate(&opts.inner, &mut sink)
            .map(|inner| GenerateSummary { inner })
            .map_err(map_wick_err)
    }
}

/// Internal `ModalitySink` implementation that trampolines text
/// tokens to a JS callback. Audio frames are dropped (text-only
/// flow); a separate `JsAudioSink` will land alongside the audio
/// engine wrapper in a future PR.
struct JsTextSink<'a> {
    on_text: &'a js_sys::Function,
}

impl<'a> wick::ModalitySink for JsTextSink<'a> {
    fn on_text_tokens(&mut self, tokens: &[u32]) {
        // `Uint32Array::from(&[u32])` allocates JS-owned memory and
        // copies the slice in. We *could* use `Uint32Array::view`
        // for zero-copy, but the resulting view becomes invalid the
        // moment Rust grows linear memory mid-call (a footgun JS
        // callers would hit randomly). Per-flush copy cost is
        // trivial relative to a forward pass.
        let array = js_sys::Uint32Array::from(tokens);
        // Treat any exception thrown by the JS callback as fatal:
        // re-throw it across the wasm boundary so it lands in the
        // JS caller's `try { ... } catch` around `session.generate`.
        // `wasm_bindgen::throw_val` aborts the current Rust call
        // immediately — wick's generate loop has no defined
        // recovery path for sink errors anyway, so unwinding mid-
        // decode is no worse than a `cancel()` (the KV cache is
        // left in whatever state the partial decode produced).
        if let Err(err) = self.on_text.call1(&JsValue::null(), &array) {
            wasm_bindgen::throw_val(err);
        }
    }

    fn on_done(&mut self, _reason: wick::FinishReason) {
        // The `GenerateSummary` already carries the finish reason;
        // no need to re-emit it through the sink. JS callers see it
        // via `summary.finishReason` after `generate` returns.
    }
}
