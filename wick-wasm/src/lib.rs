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

    /// Construct a new `Session` for this engine, using the default
    /// `SessionConfig` (no `n_keep`, no KV compression, model's own
    /// `max_seq_len`). Customizable per-field session config will land
    /// when concrete consumers ask.
    ///
    /// The returned `Session` keeps its own `Arc` clones of the
    /// engine's model and tokenizer, so freeing the engine doesn't
    /// invalidate any in-flight sessions.
    #[wasm_bindgen(js_name = newSession)]
    pub fn new_session(&self) -> Session {
        Session {
            inner: self.inner.new_session(wick::SessionConfig::default()),
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

// ---------------------------------------------------------------------------
// Session + generate
// ---------------------------------------------------------------------------

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

/// Stateful generation handle. Built via `WickEngine.newSession()`.
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
/// always call from a Web Worker. On Node it's fine to run
/// directly since the generate call only blocks the main script,
/// not the libuv event loop's I/O.
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
