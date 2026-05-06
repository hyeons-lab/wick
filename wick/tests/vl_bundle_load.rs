//! End-to-end smoke test for the Phase-1 VL loader against a real
//! `LiquidAI/LeapBundles` VL bundle.
//!
//! Asserts:
//! 1. Both GGUFs (main LLM + mmproj) download via `BundleRepo`.
//! 2. `WickEngine::from_files` accepts the VL pair (the gate is
//!    open) and constructs an engine without error.
//! 3. The mmproj is mmaped and exposed via
//!    `WickEngine::vision_encoder_gguf()`.
//! 4. The metadata's `max_seq_len` is non-zero so we know the
//!    underlying LFM2 LLM parsed correctly. Greedy decode is
//!    deferred to PR 2+ (vision-encoder forward pass) and to the
//!    larger-coverage `bundle_download.rs`.
//!
//! Gating: `#[ignore]` + `WICK_TEST_DOWNLOAD=1`. Same shared cache
//! path (`target/tmp/wick-test-models`) as the other gated tests, so
//! CI runs amortise the download.
//!
//! ```sh
//! WICK_TEST_DOWNLOAD=1 cargo test -p wick --features remote \
//!     --test vl_bundle_load -- --ignored
//! ```

#![cfg(feature = "remote")]

mod common;

use wick::engine::{BackendPreference, EngineConfig, ModelFiles, WickEngine};
use wick::manifest::InferenceType;
use wick::tokenizer::ChatMessage;
use wick::{FinishReason, GenerateOpts, ModalitySink};

const MAIN_URL: &str =
    "https://huggingface.co/LiquidAI/LFM2.5-VL-450M-GGUF/resolve/main/LFM2.5-VL-450M-Q4_0.gguf";
const MAIN_FILE: &str = "LFM2.5-VL-450M-Q4_0.gguf";
const MMPROJ_URL: &str = "https://huggingface.co/LiquidAI/LFM2.5-VL-450M-GGUF/resolve/main/mmproj-LFM2.5-VL-450m-Q8_0.gguf";
const MMPROJ_FILE: &str = "mmproj-LFM2.5-VL-450m-Q8_0.gguf";

#[test]
#[ignore = "downloads ~310 MB across two GGUFs; set WICK_TEST_DOWNLOAD=1 and pass --ignored"]
fn vl_bundle_loads_text_only() {
    if std::env::var("WICK_TEST_DOWNLOAD").is_err() {
        eprintln!("skipping: WICK_TEST_DOWNLOAD not set");
        return;
    }

    let main = common::download::ensure_cached(MAIN_URL, MAIN_FILE);
    let mmproj = common::download::ensure_cached(MMPROJ_URL, MMPROJ_FILE);
    assert!(main.exists(), "main GGUF missing at {}", main.display());
    assert!(
        mmproj.exists(),
        "mmproj GGUF missing at {}",
        mmproj.display()
    );

    // Construct the engine via `from_files` with the VL pair. The
    // explicit `inference_type` is necessary because auto-detect would
    // see the main GGUF's `architecture = "lfm2"` and pick text
    // (correct for the LLM half but skips the eager mmproj load).
    // Real callers reach this path through a manifest;
    // `from_bundle_id` populates these fields automatically.
    let mut files = ModelFiles::text(&main);
    files.multimodal_projector = Some(mmproj.clone());
    files.inference_type = Some(InferenceType::LlamaCppImageToText);

    let engine = WickEngine::from_files(
        files,
        EngineConfig {
            context_size: 256,
            backend: BackendPreference::Cpu,
            ..Default::default()
        },
    )
    .expect("VL bundle should load with the Phase-1 gate open");

    // Sanity: the LFM2 LLM half parsed cleanly.
    let meta = engine.metadata();
    assert!(
        meta.max_seq_len > 0,
        "engine metadata missing max_seq_len — main GGUF parse failed silently"
    );
    assert_eq!(
        meta.architecture, "lfm2",
        "main GGUF arch should be plain `lfm2`; got `{}`",
        meta.architecture
    );

    // The mmproj must have been mmapped and exposed for Phase 2+.
    let mmproj_gguf = engine
        .vision_encoder_gguf()
        .expect("VL bundles must expose the mmproj GGUF through vision_encoder_gguf()");
    // `clip` arch with `clip.has_vision_encoder = true` is the
    // shape every published VL mmproj uses. Asserting both lets a
    // future schema change surface here rather than silently
    // misbehave in PR 2.
    let arch = mmproj_gguf
        .architecture()
        .expect("mmproj should expose general.architecture");
    assert_eq!(arch, "clip", "mmproj arch should be `clip`; got `{arch}`");
    let has_vision = mmproj_gguf
        .get_bool("clip.has_vision_encoder")
        .unwrap_or(false);
    assert!(
        has_vision,
        "mmproj should set `clip.has_vision_encoder = true`"
    );

    // End-to-end: render the chat template (exercises the
    // `{% generation %}` strip in tokenizer.rs), tokenize, prefill,
    // and greedy-decode a few tokens. Catches regressions in either
    // the template fix or the LFM2 forward pass on a VL bundle —
    // the load-only path above doesn't exercise those.
    let tokenizer = engine.tokenizer();
    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: "Hi".to_string(),
    }];
    let formatted = wick::tokenizer::apply_chat_template(tokenizer, &messages, true)
        .expect("LFM2.5-VL chat template should render after generation-block strip");
    let prompt_tokens = tokenizer.encode(&formatted);
    assert!(
        !prompt_tokens.is_empty(),
        "rendered chat template tokenized to nothing — encoder is broken"
    );

    let mut session = engine.new_session(Default::default());
    session
        .append_tokens(&prompt_tokens)
        .expect("prefill should succeed against a VL bundle's LFM2 LLM");

    struct Collect(Vec<u32>);
    impl ModalitySink for Collect {
        fn on_text_tokens(&mut self, t: &[u32]) {
            self.0.extend_from_slice(t);
        }
        fn on_done(&mut self, _: FinishReason) {}
    }
    let mut sink = Collect(Vec::new());
    let opts = GenerateOpts {
        max_tokens: 8,
        temperature: 0.0,
        ..Default::default()
    };
    session
        .generate(&opts, &mut sink)
        .expect("greedy decode should succeed against a VL bundle");
    assert!(
        !sink.0.is_empty(),
        "VL bundle produced zero tokens — chat template / forward / sink wiring broken"
    );
}
