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

    // Phase-2 typed loader. The mmproj must have been parsed into
    // `VisionEncoderWeights`; spec from `project_vl_architecture.md`
    // (LFM2.5-VL-450M ViT-12-768).
    let ve = engine
        .vision_encoder()
        .expect("VL bundles must expose typed VisionEncoderWeights via vision_encoder()");
    assert_eq!(ve.config.n_layer, 12, "ViT block count");
    assert_eq!(ve.config.n_embd, 768, "ViT hidden dim");
    assert_eq!(ve.config.n_head, 12, "ViT head count");
    assert_eq!(ve.config.n_ff, 3072, "ViT FFN dim");
    assert_eq!(ve.config.image_size, 256);
    assert_eq!(ve.config.patch_size, 16);
    assert_eq!(ve.config.n_patches, 256, "16×16 patch grid");
    assert_eq!(ve.config.projection_dim, 1024, "matches LFM2 embed dim");
    assert_eq!(ve.config.scale_factor, 2, "pixel-shuffle factor");
    assert_eq!(ve.blocks.len(), 12);

    // Sanity-check the preprocessing constants. The loader reads
    // `clip.vision.image_{mean,std}` from f32-array metadata; if
    // the key got renamed or the array length drifted, the
    // loader bails — but a wrong-but-loadable replacement (e.g.
    // a key returning all-zeros) would slip through. CLIP-family
    // mean/std fall in (0, 1); std must be non-zero to avoid
    // divide-by-zero in the future preprocessor.
    for (i, m) in ve.config.image_mean.iter().enumerate() {
        assert!(*m > 0.0 && *m < 1.0, "image_mean[{i}] = {m} outside (0, 1)");
    }
    for (i, s) in ve.config.image_std.iter().enumerate() {
        assert!(*s > 0.0 && *s < 1.0, "image_std[{i}] = {s} outside (0, 1)");
    }
    assert!(
        ve.config.eps > 0.0 && ve.config.eps < 1e-3,
        "layer_norm_epsilon = {} outside (0, 1e-3)",
        ve.config.eps
    );

    // Per-block shapes — every block must have the same
    // ViT-12-768 layout. Looping catches a hypothetical
    // off-by-one or partial-load bug that would leave a later
    // block in a degenerate state. Loader's `anyhow::ensure!`
    // already runs at parse time but asserting here encodes the
    // contract.
    for (i, blk) in ve.blocks.iter().enumerate() {
        assert_eq!(blk.q_w.rows, 768, "block {i} q_w rows");
        assert_eq!(blk.q_w.cols, 768, "block {i} q_w cols");
        assert_eq!(blk.k_w.rows, 768, "block {i} k_w rows");
        assert_eq!(blk.v_w.rows, 768, "block {i} v_w rows");
        assert_eq!(blk.o_w.rows, 768, "block {i} o_w rows");
        assert_eq!(blk.ffn_up_w.rows, 3072, "block {i} ffn_up_w rows");
        assert_eq!(blk.ffn_up_w.cols, 768, "block {i} ffn_up_w cols");
        assert_eq!(blk.ffn_down_w.rows, 768, "block {i} ffn_down_w rows");
        assert_eq!(blk.ffn_down_w.cols, 3072, "block {i} ffn_down_w cols");
        assert_eq!(blk.ln1_w.len(), 768, "block {i} ln1_w len");
        assert_eq!(blk.ln2_w.len(), 768, "block {i} ln2_w len");
    }

    // Position embeddings cover every patch.
    assert_eq!(ve.position_embed.len(), 256 * 768);
    // Projector dims: mm.1 is [3072 → 2048], mm.2 is [2048 → 1024].
    assert_eq!(ve.projector.mm1_w.rows, 2048);
    assert_eq!(ve.projector.mm1_w.cols, 3072);
    assert_eq!(ve.projector.mm2_w.rows, 1024);
    assert_eq!(ve.projector.mm2_w.cols, 2048);

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

    // ── Phase-2 slice 2+3: ViT + projector forward smoke. ──
    //
    // Feed a synthetic constant image (`[3, 256, 256]` of 0.0 —
    // exercises every stage without needing real preprocessor
    // output) through the full vision encoder and assert the
    // image-token output is well-formed:
    //   * length = n_image_tokens × projection_dim = 64 × 1024
    //   * all values finite (catches NaN/Inf from broken
    //     softmax / norm / projector arithmetic)
    //   * not all zero (catches a forward path that short-
    //     circuits without doing any real work)
    //   * magnitudes in a sane range (catches numerical
    //     blow-up that would still be finite)
    //
    // **TODO(parity):** these checks are deliberately weak —
    // they would pass even on a forward pass that's wrong-but-
    // plausible (e.g. a kernel-stride bug producing scrambled
    // values). Strong correctness gate (parity vs llama.cpp's
    // `clip.cpp` on a real image, with strict numerical
    // tolerance) lands in Phase 3 alongside the image
    // preprocessor + real fixture. Two known assumptions also
    // need verification then: (1) GELU variant — wick uses
    // `cpu::gelu_erf_inplace` (erf-form), llama.cpp's
    // `ggml_gelu` is the tanh approximation; ~1e-5
    // per-element drift if mismatched. (2) Pixel-shuffle 2×2
    // ordering — wick concatenates source patches in
    // `(sr·sf + sc)` row-major order; clip.cpp's traversal
    // direction needs a reference vector to confirm.
    let ve = engine
        .vision_encoder()
        .expect("vision_encoder still attached");
    let n_pix = 3 * ve.config.image_size * ve.config.image_size;
    let zeros = vec![0.0f32; n_pix];
    let img_tokens = ve
        .encode_image(&zeros)
        .expect("encode_image should succeed on a zero-input image");
    let expected_n = (ve.config.n_patches / (ve.config.scale_factor * ve.config.scale_factor))
        * ve.config.projection_dim;
    assert_eq!(img_tokens.len(), expected_n, "image-token output length");
    assert!(
        img_tokens.iter().all(|v| v.is_finite()),
        "encode_image produced non-finite values"
    );
    let max_abs = img_tokens.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    assert!(
        max_abs > 0.0,
        "encode_image returned all zeros — forward likely short-circuited"
    );
    assert!(
        max_abs < 1e3,
        "encode_image returned implausibly large values (max abs = {max_abs}) — \
         numerical blow-up somewhere in the pipeline"
    );
}

/// **Phase 3 slice 1 end-to-end smoke.** Synthesises an in-memory
/// PNG, drives `Session::append_image` end-to-end (preprocess →
/// ViT → projector → splice into LLM prefill via
/// `append_embeddings`), generates a few tokens, and asserts the
/// LLM produced non-degenerate text. Catches "completely broken"
/// — pipeline runs, image embeddings get into the LLM stream, the
/// LLM doesn't crash on them. Strong correctness gate (parity vs
/// `clip.cpp` on a real image with strict numerical tolerance) is
/// deferred; the LLM-output check rules out gross structural
/// issues like NaN propagation or pixel-shuffle row/col swaps
/// causing the LLM to produce gibberish.
#[test]
#[ignore = "downloads ~310 MB across two GGUFs; set WICK_TEST_DOWNLOAD=1 and pass --ignored"]
fn vl_bundle_appends_synthetic_image() {
    if std::env::var("WICK_TEST_DOWNLOAD").is_err() {
        eprintln!("skipping: WICK_TEST_DOWNLOAD not set");
        return;
    }

    let main = common::download::ensure_cached(MAIN_URL, MAIN_FILE);
    let mmproj = common::download::ensure_cached(MMPROJ_URL, MMPROJ_FILE);
    let mut files = ModelFiles::text(&main);
    files.multimodal_projector = Some(mmproj);
    files.inference_type = Some(InferenceType::LlamaCppImageToText);
    let engine = WickEngine::from_files(
        files,
        EngineConfig {
            // The LLM context needs room for prefix tokens + 64
            // image tokens + suffix tokens + a few generation
            // tokens. 256 is plenty for the smoke.
            context_size: 256,
            backend: BackendPreference::Cpu,
            ..Default::default()
        },
    )
    .expect("VL bundle load");

    // Image input: prefer a real fixture at
    // `wick/tests/fixtures/pug.jpg` (committed for end-to-end
    // demos against an actual recognisable subject). Fall back
    // to a synthesised solid-red 256² PNG when the fixture is
    // missing — keeps the smoke runnable in environments that
    // haven't checked out the fixture (also avoids hard-failing
    // when this test runs through `--test-threads=1` on a
    // shallow clone). The smoke only asserts the LLM produces
    // non-degenerate text, so either input is fine for that
    // gate; a real image just gives nicer manual output.
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("pug.jpg");
    let img_bytes: Vec<u8> = if fixture.exists() {
        std::fs::read(&fixture).expect("read pug.jpg fixture")
    } else {
        eprintln!(
            "note: {} not found — falling back to a synthetic red PNG. \
             Save a small image fixture at that path for richer manual output.",
            fixture.display()
        );
        use image::{ImageBuffer, Rgb};
        let img = ImageBuffer::<Rgb<u8>, _>::from_fn(256, 256, |_, _| Rgb([255u8, 0, 0]));
        let mut png = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .expect("encode synthetic png");
        png
    };

    // Capabilities should now report image_in for VL bundles.
    let caps = engine.capabilities();
    assert!(caps.image_in, "VL capabilities must report image_in=true");

    // Drive the LFM2-VL chat template manually with the correct
    // image-marker placement. The model was trained on inputs of
    // the form
    //   <bos><|im_start|>user\n<|image_start|>[N image embeds]<|image_end|>TEXT<|im_end|>\n<|im_start|>assistant\n
    // Splicing image embeddings outside that envelope (e.g.
    // before the BOS) leaves the LLM unable to interpret them as
    // visual content, so the smoke covers the proper wrapping.
    // A higher-level helper that walks `<image>` markers in the
    // chat-template output is still slice 2/3 work.
    let tokenizer = engine.tokenizer();
    // Look the marker token ids up at runtime so this test stays
    // robust against vocab shifts across LFM2-VL variants
    // (e.g. 1.6B vs 450M) — hardcoded ids would silently produce
    // valid English but the wrong wrapping.
    let img_start = tokenizer
        .special_token_id("<|image_start|>")
        .expect("LFM2-VL tokenizer must define <|image_start|>");
    let img_end = tokenizer
        .special_token_id("<|image_end|>")
        .expect("LFM2-VL tokenizer must define <|image_end|>");
    // Render the same chat template the public API would, then
    // splice tokens around the user-text portion. Easier than
    // hand-building the full prefix string.
    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: "Describe what you see.".to_string(),
    }];
    let formatted = wick::tokenizer::apply_chat_template(tokenizer, &messages, true)
        .expect("chat template render");
    // The rendered string ends with "<|im_start|>assistant\n".
    // Find "<|im_start|>user\n" — the position right after that
    // marker is where the image goes.
    let user_marker = "<|im_start|>user\n";
    let user_pos = formatted
        .find(user_marker)
        .expect("chat template missing user marker");
    let split = user_pos + user_marker.len();
    let prefix_text = &formatted[..split];
    let suffix_text = &formatted[split..];
    let prefix_tokens = tokenizer.encode(prefix_text);
    let suffix_tokens = tokenizer.encode(suffix_text);
    assert!(!prefix_tokens.is_empty(), "tokenized prefix is empty");
    assert!(!suffix_tokens.is_empty(), "tokenized suffix is empty");

    let mut session = engine.new_session(Default::default());
    session
        .append_tokens(&prefix_tokens)
        .expect("append_tokens prefix");
    session
        .append_tokens(&[img_start])
        .expect("append <|image_start|>");
    session
        .append_image(&img_bytes)
        .expect("append_image should succeed end-to-end");
    session
        .append_tokens(&[img_end])
        .expect("append <|image_end|>");
    session
        .append_tokens(&suffix_tokens)
        .expect("append_tokens suffix");

    struct Collect(Vec<u32>);
    impl ModalitySink for Collect {
        fn on_text_tokens(&mut self, t: &[u32]) {
            self.0.extend_from_slice(t);
        }
        fn on_done(&mut self, _: FinishReason) {}
    }
    let mut sink = Collect(Vec::new());
    let opts = GenerateOpts {
        max_tokens: 16,
        temperature: 0.0,
        ..Default::default()
    };
    session
        .generate(&opts, &mut sink)
        .expect("generate after image+prompt");
    assert!(
        !sink.0.is_empty(),
        "generated zero tokens — append_image / forward / decode wiring is broken"
    );
    let decoded = tokenizer.decode(&sink.0);
    eprintln!("vl smoke decoded output: {decoded:?}");
    // Post-image output must look like English: at least 60% ASCII
    // letters/spaces (catches "lots of unicode + punctuation"
    // garbage from a broken encoder), and at least one space
    // (catches a single long unbroken token sequence). These bars
    // would have failed the pre-fix output ("complex and abstract
    // scene" passed because it *was* English — the fix that
    // mattered was switching from generic to scene-specific). The
    // live test's main correctness gate remains the eprintln
    // dump + manual review on first run; the assertion only
    // catches gross structural regressions.
    let total = decoded.chars().count() as f32;
    let alpha_or_space = decoded
        .chars()
        .filter(|c| c.is_ascii_alphabetic() || *c == ' ')
        .count() as f32;
    assert!(total > 0.0, "decoded text is empty");
    assert!(
        alpha_or_space / total >= 0.6,
        "post-image output mostly non-letters: {decoded:?}"
    );
    assert!(
        decoded.contains(' '),
        "post-image output has no spaces — single unbroken token: {decoded:?}"
    );
    let alpha_count = decoded.chars().filter(char::is_ascii_alphabetic).count();
    assert!(
        alpha_count >= 4,
        "post-image generation looks degenerate (got {decoded:?}); image embeddings \
         likely poisoned the LLM stream"
    );
}
