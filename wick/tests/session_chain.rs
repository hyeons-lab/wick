//! Integration tests for the fixes landed on fix/session-kv-gap (PR #27 follow-up).
//!
//! These tests verify behaviour that requires a real model — unit tests in
//! `src/session.rs` are dep-free and can't observe the `InferenceState`
//! vs. `Session::current_pos` alignment, the Model trait dispatch, or the
//! sampling path that these regression tests exercise.
//!
//! Run with:
//!   WICK_MODEL=$HOME/.leap/models/LFM2-VL-450M-Q4_0/LFM2-VL-450M-Q4_0.gguf \
//!   cargo test -p wick --test session_chain -- --ignored --nocapture
//!
//! If `WICK_MODEL` is unset or the file is missing, each test skips with a
//! message rather than failing.

use std::path::PathBuf;
use std::sync::Arc;

use wick::kv_cache::KvCompression;
use wick::model::Model;
use wick::tokenizer::BpeTokenizer;
use wick::{
    FinishReason, GenerateOpts, ModalityCapabilities, ModalitySink, Session, SessionConfig,
};

/// Test-local helper: wrap a freshly-loaded `(model, tokenizer)` pair
/// in the `Arc`s that `Session::new` now requires post-lifetime-refactor.
/// Collapses what would otherwise be three `Arc::from`/`Arc::new` calls
/// at every test site.
///
/// Every test in this file exercises the text-in / text-out path only
/// (`find_model()`'s default falls back to the LFM2-VL GGUF because
/// that's what the host has handy, but these tests never feed it an
/// image). Pinning capabilities to `text_only()` reflects what the
/// tests exercise, not the underlying model's full capabilities — a VL
/// or audio test would pass the matching constructor.
fn make_session(model: Box<dyn Model>, tokenizer: BpeTokenizer, config: SessionConfig) -> Session {
    let model: Arc<dyn Model> = Arc::from(model);
    let tokenizer = Arc::new(tokenizer);
    Session::new(model, tokenizer, ModalityCapabilities::text_only(), config)
}

/// Locate a text-path GGUF for the tests. Prefers the `WICK_MODEL` env var;
/// falls back to the LFM2-VL-450M copy that the session-api smoke test used.
fn find_model() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("WICK_MODEL") {
        let pb = PathBuf::from(p);
        return pb.exists().then_some(pb);
    }
    let home = std::env::var("HOME").ok()?;
    let fallback =
        PathBuf::from(&home).join(".leap/models/LFM2-VL-450M-Q4_0/LFM2-VL-450M-Q4_0.gguf");
    fallback.exists().then_some(fallback)
}

struct CollectSink(Vec<u32>);
impl ModalitySink for CollectSink {
    fn on_text_tokens(&mut self, tokens: &[u32]) {
        self.0.extend_from_slice(tokens);
    }
    fn on_done(&mut self, _reason: FinishReason) {}
}

fn greedy_opts(max_tokens: u32) -> GenerateOpts {
    GenerateOpts {
        max_tokens,
        temperature: 0.0,
        ..Default::default()
    }
}

/// Regression test for PR #27 review comment #4 (stochastic chaining):
/// After a stochastic `generate()` returns, a second `generate()` call on
/// the same session must continue from the saved logits without requiring
/// an intervening `append_tokens`. The pre-fix bug was that
/// `last_logits.take()` left the field `None`; the chained call hit
/// `WickError::EmptyInput`.
///
/// Greedy mode's chainability was relaxed when `forward_greedy()` returned
/// to the decode loop (GPU perf optimization, PR #29): `forward_greedy`
/// doesn't produce a readable vocab distribution, so `last_logits` stays
/// `None` across greedy calls. `greedy_chain_requires_append_tokens`
/// covers that contract explicitly.
#[test]
#[ignore]
fn stochastic_generate_is_chainable_without_append() {
    let Some(model_path) = find_model() else {
        eprintln!("no model available — skipping");
        return;
    };

    let gguf = wick::gguf::GgufFile::open(&model_path).unwrap();
    let tokenizer = wick::tokenizer::BpeTokenizer::from_gguf(&gguf).unwrap();
    let model = wick::model::load_model(gguf, 4096).unwrap();
    let prompt_toks = tokenizer.encode("The capital of France is");

    let mut session = make_session(
        model,
        tokenizer,
        SessionConfig {
            kv_compression: KvCompression::None,
            seed: Some(42),
            ..Default::default()
        },
    );
    session.append_tokens(&prompt_toks).unwrap();

    let opts = GenerateOpts {
        max_tokens: 4,
        temperature: 0.8,
        top_k: 40,
        top_p: 0.9,
        ..Default::default()
    };

    let mut sink_a = CollectSink(Vec::new());
    session.generate(&opts, &mut sink_a).unwrap();
    assert_eq!(sink_a.0.len(), 4, "first call should generate 4 tokens");

    let mut sink_b = CollectSink(Vec::new());
    let summary = session.generate(&opts, &mut sink_b).unwrap();
    assert_eq!(
        summary.tokens_generated, 4,
        "chained stochastic call should generate 4 tokens"
    );
    assert_eq!(sink_b.0.len(), 4);
    assert_eq!(session.position() as usize, prompt_toks.len() + 8);
}

/// Explicit coverage for greedy mode's relaxed chaining contract: after a
/// greedy `generate()`, `last_logits` is cleared because `forward_greedy`
/// never produced a readable distribution. A second greedy call without an
/// intermediate `append_tokens` returns `WickError::EmptyInput`. Adding an
/// `append_tokens` step between calls makes it work. This is the
/// documented chat-loop flow.
#[test]
#[ignore]
fn greedy_chain_requires_append_tokens() {
    let Some(model_path) = find_model() else {
        eprintln!("no model available — skipping");
        return;
    };

    let gguf = wick::gguf::GgufFile::open(&model_path).unwrap();
    let tokenizer = wick::tokenizer::BpeTokenizer::from_gguf(&gguf).unwrap();
    let model = wick::model::load_model(gguf, 4096).unwrap();
    let prompt_toks = tokenizer.encode("The capital of France is");

    let mut session = make_session(
        model,
        tokenizer,
        SessionConfig {
            seed: Some(42),
            ..Default::default()
        },
    );
    session.append_tokens(&prompt_toks).unwrap();

    let mut sink_a = CollectSink(Vec::new());
    session.generate(&greedy_opts(4), &mut sink_a).unwrap();
    assert_eq!(sink_a.0.len(), 4);

    // Second greedy generate WITHOUT append_tokens must fail with EmptyInput.
    let mut sink_err = CollectSink(Vec::new());
    let err = session
        .generate(&greedy_opts(4), &mut sink_err)
        .unwrap_err();
    assert!(
        matches!(err, wick::WickError::EmptyInput),
        "expected EmptyInput on chained greedy generate without append, got: {err:?}"
    );

    // After appending a FRESH user-side token suffix (the real chat loop —
    // new user turn appended between generates), greedy can continue.
    let follow_up = session.tokenizer().encode(" and the currency is");
    session.append_tokens(&follow_up).unwrap();
    let mut sink_b = CollectSink(Vec::new());
    let summary = session.generate(&greedy_opts(4), &mut sink_b).unwrap();
    assert_eq!(summary.tokens_generated, 4);
}

/// Regression test for PR #27 review comment #3 (KV gap after bounded
/// greedy generate): after `generate()` hits `max_tokens`, `state.seq_len`
/// and `current_pos` must stay aligned. A subsequent `append_tokens`
/// feeds to the correct KV slot; a further greedy generate then produces
/// a sensible continuation (non-EOS, non-degenerate).
///
/// The direct "split == single" equivalence test is split across:
/// - Greedy path: `greedy_chain_requires_append_tokens` (state consistency
///   via positive-path append + generate after bounded call).
/// - Stochastic path: `stochastic_split_matches_single_call_under_seed`
///   (byte-exact split == single under a fixed seed).
#[test]
#[ignore]
fn no_kv_gap_after_bounded_generate_greedy() {
    let Some(model_path) = find_model() else {
        eprintln!("no model available — skipping");
        return;
    };

    let gguf = wick::gguf::GgufFile::open(&model_path).unwrap();
    let tokenizer = wick::tokenizer::BpeTokenizer::from_gguf(&gguf).unwrap();
    let model = wick::model::load_model(gguf, 4096).unwrap();
    let prompt_toks = tokenizer.encode("The capital of France is");

    let mut session = make_session(
        model,
        tokenizer,
        SessionConfig {
            seed: Some(7),
            ..Default::default()
        },
    );
    session.append_tokens(&prompt_toks).unwrap();

    let mut sink = CollectSink(Vec::new());
    session.generate(&greedy_opts(4), &mut sink).unwrap();
    assert_eq!(sink.0.len(), 4);
    let pos_after_gen = session.position() as usize;
    assert_eq!(pos_after_gen, prompt_toks.len() + 4);

    // A further append_tokens must land without error and advance the
    // position cleanly. If `state.seq_len` lagged `current_pos` (the
    // pre-fix bug), this would write to a stale KV slot — no panic would
    // fire but subsequent generate output would be garbage. We use a
    // fresh user-side token suffix (mirrors the real chat loop) rather
    // than re-feeding an already-emitted token.
    let follow_up = session.tokenizer().encode(" and the language is");
    session.append_tokens(&follow_up).unwrap();
    assert_eq!(
        session.position() as usize,
        pos_after_gen + follow_up.len(),
        "append_tokens after bounded generate should advance position by the appended length"
    );

    // And a further greedy generate succeeds.
    let mut sink2 = CollectSink(Vec::new());
    let summary = session.generate(&greedy_opts(4), &mut sink2).unwrap();
    assert_eq!(summary.tokens_generated, 4);
    assert_eq!(sink2.0.len(), 4);
    // Sanity: each emitted token is a valid vocab id (not the EOS, not
    // out of range). Catches a broken KV producing nonsense logits.
    let vocab_size = session.model().config().vocab_size as u32;
    let eos = session.tokenizer().eos_token();
    for t in &sink2.0 {
        assert!(
            *t < vocab_size,
            "emitted token {t} outside vocab {vocab_size}"
        );
        assert!(
            Some(*t) != eos,
            "greedy continuation shouldn't immediately hit EOS on 'France is ... '"
        );
    }
}

/// Regression test for PR #27 review comment #5:
/// `reset()` must re-seed the sampler so a seeded session is reproducible
/// from a clean state. Running twice with the same seed + same prompt must
/// produce identical tokens under stochastic sampling (temperature > 0).
#[test]
#[ignore]
fn reset_reseeds_sampler_for_reproducibility() {
    let Some(model_path) = find_model() else {
        eprintln!("no model available — skipping");
        return;
    };

    let gguf = wick::gguf::GgufFile::open(&model_path).unwrap();
    let tokenizer = wick::tokenizer::BpeTokenizer::from_gguf(&gguf).unwrap();
    let model = wick::model::load_model(gguf, 4096).unwrap();
    let prompt_toks = tokenizer.encode("Tell me a story about");

    let mut session = make_session(
        model,
        tokenizer,
        SessionConfig {
            seed: Some(123),
            ..Default::default()
        },
    );

    let opts = GenerateOpts {
        max_tokens: 8,
        temperature: 0.8,
        top_k: 40,
        top_p: 0.9,
        ..Default::default()
    };

    session.append_tokens(&prompt_toks).unwrap();
    let mut sink_a = CollectSink(Vec::new());
    session.generate(&opts, &mut sink_a).unwrap();

    session.reset();
    session.append_tokens(&prompt_toks).unwrap();
    let mut sink_b = CollectSink(Vec::new());
    session.generate(&opts, &mut sink_b).unwrap();

    assert_eq!(
        sink_a.0, sink_b.0,
        "same-seed run after reset() must produce identical stochastic output.\nrun A: {:?}\nrun B: {:?}",
        sink_a.0, sink_b.0
    );
}

/// Exercises `position_handle()` returning an Arc — watcher in another
/// context sees monotonically-increasing position during generation.
#[test]
#[ignore]
fn position_handle_observes_progress() {
    let Some(model_path) = find_model() else {
        eprintln!("no model available — skipping");
        return;
    };

    let gguf = wick::gguf::GgufFile::open(&model_path).unwrap();
    let tokenizer = wick::tokenizer::BpeTokenizer::from_gguf(&gguf).unwrap();
    let model = wick::model::load_model(gguf, 4096).unwrap();
    let prompt_toks = tokenizer.encode("The capital of France is");

    let mut session = make_session(
        model,
        tokenizer,
        SessionConfig {
            seed: Some(42),
            ..Default::default()
        },
    );
    let pos_handle = session.position_handle();
    assert_eq!(pos_handle.load(std::sync::atomic::Ordering::Relaxed), 0);

    session.append_tokens(&prompt_toks).unwrap();
    let after_append = pos_handle.load(std::sync::atomic::Ordering::Relaxed);
    assert_eq!(after_append as usize, prompt_toks.len());

    let mut sink = CollectSink(Vec::new());
    session.generate(&greedy_opts(4), &mut sink).unwrap();
    let after_gen = pos_handle.load(std::sync::atomic::Ordering::Relaxed);
    assert_eq!(after_gen as usize, prompt_toks.len() + 4);
    assert!(after_gen > after_append);
}

/// Regression test for PR #28 review comment (github-actions @ session.rs:453):
/// Stochastic split-generation reproducibility. A single `generate(8)` with
/// a fixed seed must produce the same token stream as two back-to-back
/// `generate(4)` calls with the same seed, because the decode loop must
/// advance the RNG exactly once per emitted token. The pre-restructure
/// implementation leaked an extra RNG step at the end of each `generate()`
/// call (the unused "seed for next iter" sample), which meant split calls
/// diverged from a single call starting with the same seed.
#[test]
#[ignore]
fn stochastic_split_matches_single_call_under_seed() {
    let Some(model_path) = find_model() else {
        eprintln!("no model available — skipping");
        return;
    };

    let gguf = wick::gguf::GgufFile::open(&model_path).unwrap();
    // Tokenizer is reused by two session blocks below — wrap once,
    // clone the `Arc` into each `Session::new` call.
    let tokenizer = Arc::new(wick::tokenizer::BpeTokenizer::from_gguf(&gguf).unwrap());
    let prompt_toks = tokenizer.encode("Tell me a story about");

    let stochastic_opts = |n: u32| GenerateOpts {
        max_tokens: n,
        temperature: 0.8,
        top_k: 40,
        top_p: 0.9,
        ..Default::default()
    };

    let baseline = {
        let gguf = wick::gguf::GgufFile::open(&model_path).unwrap();
        let model: Arc<dyn Model> = Arc::from(wick::model::load_model(gguf, 4096).unwrap());
        let mut session = Session::new(
            model,
            Arc::clone(&tokenizer),
            ModalityCapabilities::text_only(),
            SessionConfig {
                seed: Some(999),
                ..Default::default()
            },
        );
        session.append_tokens(&prompt_toks).unwrap();
        let mut sink = CollectSink(Vec::new());
        session.generate(&stochastic_opts(8), &mut sink).unwrap();
        sink.0
    };

    let split = {
        let gguf = wick::gguf::GgufFile::open(&model_path).unwrap();
        let model: Arc<dyn Model> = Arc::from(wick::model::load_model(gguf, 4096).unwrap());
        let mut session = Session::new(
            model,
            Arc::clone(&tokenizer),
            ModalityCapabilities::text_only(),
            SessionConfig {
                seed: Some(999),
                ..Default::default()
            },
        );
        session.append_tokens(&prompt_toks).unwrap();
        let mut sink1 = CollectSink(Vec::new());
        session.generate(&stochastic_opts(4), &mut sink1).unwrap();
        let mut sink2 = CollectSink(Vec::new());
        session.generate(&stochastic_opts(4), &mut sink2).unwrap();
        let mut all = sink1.0;
        all.extend(sink2.0);
        all
    };

    assert_eq!(
        baseline, split,
        "split stochastic generation must match single call with the same seed.\nbaseline: {baseline:?}\nsplit:    {split:?}"
    );
}

/// Regression test for new PR #27 review comment (Copilot @ session.rs:385):
/// `position_atomic` must update DURING decode (per-token), not just at
/// call boundaries, so an external watcher thread observes progress in real
/// time. This test snoops via a sink that polls `position_handle()` on each
/// flush and records monotonic non-trivial progress.
#[test]
#[ignore]
fn position_updates_per_token_during_decode() {
    let Some(model_path) = find_model() else {
        eprintln!("no model available — skipping");
        return;
    };

    let gguf = wick::gguf::GgufFile::open(&model_path).unwrap();
    let tokenizer = wick::tokenizer::BpeTokenizer::from_gguf(&gguf).unwrap();
    let model = wick::model::load_model(gguf, 4096).unwrap();
    let prompt_toks = tokenizer.encode("The capital of France is");

    let mut session = make_session(
        model,
        tokenizer,
        SessionConfig {
            seed: Some(42),
            ..Default::default()
        },
    );
    let pos_handle = session.position_handle();
    session.append_tokens(&prompt_toks).unwrap();
    let prefill_pos = pos_handle.load(std::sync::atomic::Ordering::Relaxed);

    struct ProgressSink {
        pos_handle: std::sync::Arc<std::sync::atomic::AtomicU32>,
        observed: Vec<u32>,
    }
    impl ModalitySink for ProgressSink {
        fn on_text_tokens(&mut self, _: &[u32]) {
            self.observed
                .push(self.pos_handle.load(std::sync::atomic::Ordering::Relaxed));
        }
        fn on_done(&mut self, _: FinishReason) {}
    }

    // flush_every_tokens=1 forces a sink call every token so we get
    // per-step position snapshots.
    let opts = GenerateOpts {
        max_tokens: 6,
        temperature: 0.0,
        flush_every_tokens: 1,
        flush_every_ms: 0,
        ..Default::default()
    };
    let mut sink = ProgressSink {
        pos_handle: pos_handle.clone(),
        observed: Vec::new(),
    };
    session.generate(&opts, &mut sink).unwrap();

    // Each observation must be monotonically non-decreasing AND must have
    // advanced past the prefill position by the second observation at latest.
    assert!(sink.observed.len() >= 2);
    assert!(
        sink.observed.last().copied().unwrap() > prefill_pos,
        "position should advance during decode, but stayed at prefill pos {prefill_pos}. observed: {:?}",
        sink.observed
    );
    for w in sink.observed.windows(2) {
        assert!(w[0] <= w[1], "position went backwards: {:?}", sink.observed);
    }
}

/// Smoke test for `Session::append_embeddings` — the soft-token
/// analog of `append_tokens` used by audio / VL input paths.
/// Feeds a small deterministic `[n × hidden_size]` buffer and
/// asserts position advances by `n` and `last_logits` is set
/// (= a follow-up `generate()` could decode without an
/// intervening `append_tokens`).
#[test]
#[ignore]
fn append_embeddings_advances_position_and_sets_logits() {
    let Some(model_path) = find_model() else {
        eprintln!("no model available — skipping");
        return;
    };

    let gguf = wick::gguf::GgufFile::open(&model_path).unwrap();
    let tokenizer = wick::tokenizer::BpeTokenizer::from_gguf(&gguf).unwrap();
    let model = wick::model::load_model(gguf, 4096).unwrap();
    let hidden_size = model.config().hidden_size;

    // append_embeddings is the soft-token path used by audio /
    // VL encoders, so capabilities reflect that — even though
    // the underlying model fixture (text-path GGUF) doesn't
    // technically expose audio in. This test cares about the
    // soft-token wiring, not the modality contract.
    let model: Arc<dyn Model> = Arc::from(model);
    let tokenizer = Arc::new(tokenizer);
    let mut session = Session::new(
        model,
        tokenizer,
        ModalityCapabilities {
            text_in: true,
            text_out: true,
            audio_in: true,
            audio_out: false,
            image_in: false,
        },
        SessionConfig::default(),
    );

    // Tiny deterministic stand-in for a real encoder output. The
    // numbers are arbitrary — we're not asserting numerical
    // semantics, only that the loop runs cleanly and the
    // session bookkeeping advances correctly.
    let n = 4;
    let embeddings: Vec<f32> = (0..n * hidden_size)
        .map(|i| ((i % 13) as f32) * 0.01 - 0.05)
        .collect();

    let pos_before = session.position();
    session
        .append_embeddings(&embeddings, n)
        .expect("append_embeddings");
    assert_eq!(
        session.position(),
        pos_before + n as u32,
        "position must advance by exactly n_tokens"
    );

    // After append_embeddings with last_logits set, generate()
    // should be able to produce a token without an intervening
    // append_tokens (mirrors the post-append_tokens contract).
    let opts = greedy_opts(1);
    let mut sink = CollectSink(Vec::new());
    session.generate(&opts, &mut sink).expect("generate");
    assert_eq!(sink.0.len(), 1, "should emit exactly one token");
}

/// Parity test: the batched
/// `Model::forward_prefill_from_embeddings` override on
/// `Lfm2Model` must produce the same final-frame logits as the
/// per-frame `forward_from_embedding` loop the trait default
/// performs. Reduction-order differences between GEMV and GEMM
/// allow a small epsilon, but the two paths must agree on every
/// vocab entry — proves the column-major transpose + shared
/// `prefill_layers_and_logits` helper match the sequential
/// forward path.
#[test]
#[ignore]
fn forward_prefill_from_embeddings_matches_per_frame_loop() {
    let Some(model_path) = find_model() else {
        eprintln!("no model available — skipping");
        return;
    };

    let gguf_a = wick::gguf::GgufFile::open(&model_path).unwrap();
    let gguf_b = wick::gguf::GgufFile::open(&model_path).unwrap();
    let model_a = wick::model::load_model(gguf_a, 4096).unwrap();
    let model_b = wick::model::load_model(gguf_b, 4096).unwrap();
    let cfg = model_a.config().clone();
    let hidden_size = cfg.hidden_size;

    // Small but non-trivial frame count. Big enough to exercise the
    // batched per-layer GEMM path; small enough that the test is
    // fast even on cold caches.
    let n: usize = 6;
    let embeddings: Vec<f32> = (0..n * hidden_size)
        .map(|i| (((i * 31 + 7) % 257) as f32) * 0.001 - 0.1)
        .collect();

    // Path A: loop forward_from_embedding per frame (the trait
    // default's behavior, exercised here directly).
    let mut state_a = wick::kv_cache::InferenceState::from_config(&cfg);
    let mut last_a: Vec<f32> = Vec::new();
    for j in 0..n {
        let frame = &embeddings[j * hidden_size..(j + 1) * hidden_size];
        last_a = model_a.forward_from_embedding(frame, j, &mut state_a);
    }

    // Path B: single batched call.
    let mut state_b = wick::kv_cache::InferenceState::from_config(&cfg);
    let last_b = model_b.forward_prefill_from_embeddings(&embeddings, n, 0, &mut state_b);

    assert_eq!(
        last_a.len(),
        last_b.len(),
        "logit vector length mismatch between loop and batched paths"
    );

    let mut max_abs_diff = 0.0f32;
    for (i, (a, b)) in last_a.iter().zip(last_b.iter()).enumerate() {
        let d = (a - b).abs();
        if d > max_abs_diff {
            max_abs_diff = d;
        }
        assert!(
            d < 1e-2,
            "logit {i}: loop={a} batched={b} diff={d} (max so far {max_abs_diff})"
        );
    }
    eprintln!("forward_prefill_from_embeddings parity: max |Δlogit| = {max_abs_diff:.4e}");

    // Final state seq_len must match — both paths processed n
    // frames starting from pos 0.
    assert_eq!(state_a.seq_len, n);
    assert_eq!(state_b.seq_len, n);
}

// ---------------------------------------------------------------------------
// append_audio (PR 4d): end-to-end audio-input pipeline.
//
// The text-only-capability test below uses `make_session` (which
// pins capabilities to `text_only`) — that's exactly what its
// `UnsupportedModality` assertion needs. Every other test in this
// section constructs the session directly with `audio_in: true` so
// the audio pipeline can run.
// ---------------------------------------------------------------------------

/// Audio-in capability disabled → `UnsupportedModality` regardless of
/// encoder attachment / samples / sample rate.
#[test]
#[ignore]
fn append_audio_text_only_returns_unsupported_modality() {
    let Some(model_path) = find_model() else {
        eprintln!("no model available — skipping");
        return;
    };
    let gguf = wick::gguf::GgufFile::open(&model_path).unwrap();
    let tokenizer = wick::tokenizer::BpeTokenizer::from_gguf(&gguf).unwrap();
    let model = wick::model::load_model(gguf, 1024).unwrap();
    let mut session = make_session(model, tokenizer, SessionConfig::default());

    let pcm = vec![0.0f32; 16_000];
    let err = session.append_audio(&pcm, 16_000).unwrap_err();
    assert!(
        matches!(err, wick::WickError::UnsupportedModality),
        "expected UnsupportedModality for text-only session, got {err:?}"
    );
}

/// Audio-in capability enabled but no encoder attached → typed
/// `Backend` error pointing at `attach_audio_encoder`.
#[test]
#[ignore]
fn append_audio_without_encoder_returns_backend_error() {
    let Some(model_path) = find_model() else {
        eprintln!("no model available — skipping");
        return;
    };
    let gguf = wick::gguf::GgufFile::open(&model_path).unwrap();
    let tokenizer = wick::tokenizer::BpeTokenizer::from_gguf(&gguf).unwrap();
    let model = wick::model::load_model(gguf, 1024).unwrap();
    let model: Arc<dyn Model> = Arc::from(model);
    let tokenizer = Arc::new(tokenizer);
    let mut session = Session::new(
        model,
        tokenizer,
        ModalityCapabilities {
            text_in: true,
            text_out: true,
            audio_in: true,
            audio_out: false,
            image_in: false,
        },
        SessionConfig::default(),
    );

    let pcm = vec![0.0f32; 16_000];
    let err = session.append_audio(&pcm, 16_000).unwrap_err();
    let wick::WickError::Backend(msg) = err else {
        panic!("expected Backend error for missing encoder, got {err:?}");
    };
    assert!(
        msg.contains("attach_audio_encoder"),
        "Backend error message should reference attach_audio_encoder; got: {msg}"
    );
}

/// Wrong sample rate → typed `Backend` error mentioning the expected
/// rate. Tests the guard before any encoder work.
#[test]
#[ignore]
fn append_audio_wrong_sample_rate_returns_backend_error() {
    use wick::model::audio_encoder::AudioEncoderWeights;

    let Some(model_path) = find_model() else {
        eprintln!("no model available — skipping");
        return;
    };
    let Ok(home) = std::env::var("HOME") else {
        eprintln!("no HOME env — skipping");
        return;
    };
    let mmproj_path = std::path::PathBuf::from(&home)
        .join(".leap/models/LFM2.5-Audio-1.5B-Q4_0/mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf");
    if !mmproj_path.exists() {
        eprintln!(
            "no mmproj available at {} — skipping",
            mmproj_path.display()
        );
        return;
    }
    let mmproj = wick::gguf::GgufFile::open(&mmproj_path).unwrap();
    let mut weights = AudioEncoderWeights::from_gguf(&mmproj).unwrap();

    let gguf = wick::gguf::GgufFile::open(&model_path).unwrap();
    let tokenizer = wick::tokenizer::BpeTokenizer::from_gguf(&gguf).unwrap();
    let model = wick::model::load_model(gguf, 1024).unwrap();
    // The LFM2.5-Audio encoder targets 2048-dim hidden, but our
    // fallback model is LFM2-VL-450M (1024-dim). Patch the encoder's
    // config to match the LLM so the dim-mismatch guard doesn't
    // fire first — this test is specifically about the sample-rate
    // check, which is the next gate in line.
    weights.config.llm_hidden_size = model.config().hidden_size;
    let encoder = Arc::new(weights);

    let model: Arc<dyn Model> = Arc::from(model);
    let tokenizer = Arc::new(tokenizer);
    let mut session = Session::new(
        model,
        tokenizer,
        ModalityCapabilities {
            text_in: true,
            text_out: true,
            audio_in: true,
            audio_out: false,
            image_in: false,
        },
        SessionConfig::default(),
    );
    session.attach_audio_encoder(encoder);

    let pcm = vec![0.0f32; 16_000];
    let err = session.append_audio(&pcm, 24_000).unwrap_err();
    let wick::WickError::Backend(msg) = err else {
        panic!("expected Backend error for sample-rate mismatch, got {err:?}");
    };
    assert!(
        msg.contains("sample_rate") && msg.contains("16000"),
        "Backend error should mention sample_rate + expected 16000; got: {msg}"
    );
}

/// End-to-end smoke: load LFM2.5-Audio + its mmproj, attach the
/// encoder, feed 0.5s of synthetic PCM, verify position advances by
/// exactly the encoder's frame count and `last_logits` is set so a
/// follow-up `generate()` succeeds without an intervening
/// `append_tokens`.
#[test]
#[ignore]
fn append_audio_end_to_end() {
    use wick::model::audio_encoder::{AudioEncoderWeights, SAMPLE_RATE};

    let Ok(home) = std::env::var("HOME") else {
        eprintln!("no HOME env — skipping");
        return;
    };
    let bundle = std::path::PathBuf::from(&home).join(".leap/models/LFM2.5-Audio-1.5B-Q4_0");
    let primary = bundle.join("LFM2.5-Audio-1.5B-Q4_0.gguf");
    let mmproj_path = bundle.join("mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf");
    if !primary.exists() || !mmproj_path.exists() {
        eprintln!(
            "no LFM2.5-Audio bundle (need {} and {}) — skipping",
            primary.display(),
            mmproj_path.display()
        );
        return;
    }

    let primary_gguf = wick::gguf::GgufFile::open(&primary).unwrap();
    let tokenizer = wick::tokenizer::BpeTokenizer::from_gguf(&primary_gguf).unwrap();
    let model = wick::model::load_model(primary_gguf, 2048).unwrap();
    let model: Arc<dyn Model> = Arc::from(model);
    let tokenizer = Arc::new(tokenizer);
    let mut session = Session::new(
        model,
        tokenizer,
        ModalityCapabilities {
            text_in: true,
            text_out: true,
            audio_in: true,
            audio_out: false,
            image_in: false,
        },
        SessionConfig::default(),
    );

    let mmproj = wick::gguf::GgufFile::open(&mmproj_path).unwrap();
    let encoder = Arc::new(AudioEncoderWeights::from_gguf(&mmproj).unwrap());
    session.attach_audio_encoder(encoder);

    // 0.5 s of low-amplitude pink-ish noise (deterministic). Pure
    // silence works but produces nearly-zero activations end-to-end;
    // a small varying signal exercises the FFT path more honestly.
    let n_samples = (SAMPLE_RATE as usize) / 2;
    let pcm: Vec<f32> = (0..n_samples)
        .map(|i| ((i as f32 * 0.07).sin() + (i as f32 * 0.013).sin()) * 0.05)
        .collect();

    let pos_before = session.position();
    session
        .append_audio(&pcm, SAMPLE_RATE)
        .expect("append_audio");
    let n_frames = session.position() - pos_before;
    assert!(n_frames > 0, "encoder must produce at least one frame");
    eprintln!(
        "append_audio: 0.5 s @ {} Hz → {} encoder frames",
        SAMPLE_RATE, n_frames
    );

    // Follow-up greedy generate(1) must succeed without an
    // intervening append_tokens — same contract as
    // append_embeddings's smoke test. We don't assert on token
    // count: synthetic noise can produce EOS as the argmax of the
    // first sampling step, which is a valid `Stop` outcome
    // (greedy breaks before emitting anything). The Ok() return
    // alone proves `last_logits` was populated by append_audio.
    let opts = greedy_opts(1);
    let mut sink = CollectSink(Vec::new());
    let summary = session.generate(&opts, &mut sink).expect("generate");
    eprintln!(
        "post-audio generate: emitted {} tokens, finish_reason = {:?}",
        sink.0.len(),
        summary.finish_reason
    );
    assert!(
        sink.0.len() <= 1,
        "greedy(1) should emit 0 or 1 tokens, got {}",
        sink.0.len()
    );
}

/// `Session::reset` clears KV state + sampler but preserves the
/// attached audio encoder — the encoder is independent of session
/// state and rebuilding it would force callers into an awkward
/// "re-attach after every reset" dance. This test exercises that
/// contract directly: attach, reset, then call `append_audio` and
/// verify it doesn't fail with the "no encoder attached" backend
/// error.
#[test]
#[ignore]
fn reset_preserves_attached_audio_encoder() {
    use wick::model::audio_encoder::AudioEncoderWeights;

    let Some(model_path) = find_model() else {
        eprintln!("no model available — skipping");
        return;
    };
    let Ok(home) = std::env::var("HOME") else {
        eprintln!("no HOME env — skipping");
        return;
    };
    let mmproj_path = std::path::PathBuf::from(&home)
        .join(".leap/models/LFM2.5-Audio-1.5B-Q4_0/mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf");
    if !mmproj_path.exists() {
        eprintln!(
            "no mmproj available at {} — skipping",
            mmproj_path.display()
        );
        return;
    }
    let mmproj = wick::gguf::GgufFile::open(&mmproj_path).unwrap();
    let encoder = Arc::new(AudioEncoderWeights::from_gguf(&mmproj).unwrap());

    let gguf = wick::gguf::GgufFile::open(&model_path).unwrap();
    let tokenizer = wick::tokenizer::BpeTokenizer::from_gguf(&gguf).unwrap();
    let model = wick::model::load_model(gguf, 1024).unwrap();
    let model: Arc<dyn Model> = Arc::from(model);
    let tokenizer = Arc::new(tokenizer);
    let mut session = Session::new(
        model,
        tokenizer,
        ModalityCapabilities {
            text_in: true,
            text_out: true,
            audio_in: true,
            audio_out: false,
            image_in: false,
        },
        SessionConfig::default(),
    );
    session.attach_audio_encoder(encoder);
    session.reset();

    // After reset, calling append_audio should hit the
    // dimension-mismatch path (encoder is for LFM2.5-Audio, model
    // is the LFM2-VL fallback), NOT the "no encoder" path.
    // Either Backend variant is fine here; what we're asserting is
    // that the encoder was preserved.
    let pcm = vec![0.0f32; 16_000];
    match session.append_audio(&pcm, 16_000) {
        Err(wick::WickError::Backend(msg)) => {
            assert!(
                !msg.contains("no audio encoder attached"),
                "encoder should have been preserved across reset; got: {msg}"
            );
        }
        Ok(()) => {
            // Encoder is preserved AND dimensions matched (encoder
            // bundle was loaded for the same LLM). Either way, the
            // contract holds.
        }
        Err(other) => panic!("unexpected error after reset: {other:?}"),
    }
}

/// Wrong-bundle encoder: `llm_hidden_size` mismatch must surface
/// as a typed `Backend` error mentioning both sides, not as a
/// generic shape error from `append_embeddings` downstream.
#[test]
#[ignore]
fn append_audio_dimension_mismatch_returns_backend_error() {
    use wick::model::audio_encoder::AudioEncoderWeights;

    let Some(model_path) = find_model() else {
        eprintln!("no model available — skipping");
        return;
    };
    let Ok(home) = std::env::var("HOME") else {
        eprintln!("no HOME env — skipping");
        return;
    };
    let mmproj_path = std::path::PathBuf::from(&home)
        .join(".leap/models/LFM2.5-Audio-1.5B-Q4_0/mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf");
    if !mmproj_path.exists() {
        eprintln!(
            "no mmproj available at {} — skipping",
            mmproj_path.display()
        );
        return;
    }
    // Encoder is for LFM2.5-Audio (llm_hidden_size = 1024), but the
    // LLM here is LFM2-VL-450M (hidden_size = 1024 too — they
    // happen to match). To force a mismatch, swap the encoder's
    // config llm_hidden_size to something deliberately wrong.
    let mmproj = wick::gguf::GgufFile::open(&mmproj_path).unwrap();
    let mut weights = AudioEncoderWeights::from_gguf(&mmproj).unwrap();
    let actual = weights.config.llm_hidden_size;
    weights.config.llm_hidden_size = actual + 1; // poison
    let encoder = Arc::new(weights);

    let gguf = wick::gguf::GgufFile::open(&model_path).unwrap();
    let tokenizer = wick::tokenizer::BpeTokenizer::from_gguf(&gguf).unwrap();
    let model = wick::model::load_model(gguf, 1024).unwrap();
    let model: Arc<dyn Model> = Arc::from(model);
    let tokenizer = Arc::new(tokenizer);
    let mut session = Session::new(
        model,
        tokenizer,
        ModalityCapabilities {
            text_in: true,
            text_out: true,
            audio_in: true,
            audio_out: false,
            image_in: false,
        },
        SessionConfig::default(),
    );
    session.attach_audio_encoder(encoder);

    let pcm = vec![0.0f32; 16_000];
    let err = session.append_audio(&pcm, 16_000).unwrap_err();
    let wick::WickError::Backend(msg) = err else {
        panic!("expected Backend error for dim mismatch, got {err:?}");
    };
    assert!(
        msg.contains("llm_hidden_size") && msg.contains("hidden_size"),
        "Backend error should mention both encoder llm_hidden_size and LLM hidden_size; got: {msg}"
    );
}

/// End-to-end engine auto-attach: build a `WickEngine` from the
/// LFM2.5-Audio bundle directory, create a session via
/// `new_session`, and call `append_audio` directly without manual
/// `attach_audio_encoder`. The engine must have eagerly loaded the
/// mmproj from `manifest.files.multimodal_projector` and pre-
/// attached it to the new session.
#[test]
#[ignore]
fn engine_auto_attaches_audio_encoder_from_bundle() {
    use wick::model::audio_encoder::SAMPLE_RATE;
    use wick::{EngineConfig, WickEngine};

    let Ok(home) = std::env::var("HOME") else {
        eprintln!("no HOME env — skipping");
        return;
    };
    let bundle_dir = std::path::PathBuf::from(&home).join(".leap/models/LFM2.5-Audio-1.5B-Q4_0");
    if !bundle_dir.is_dir() {
        eprintln!(
            "no LFM2.5-Audio bundle dir at {} — skipping",
            bundle_dir.display()
        );
        return;
    }

    let engine = WickEngine::from_path(&bundle_dir, EngineConfig::default()).unwrap();
    assert!(
        engine.audio_encoder().is_some(),
        "engine should eagerly load mmproj for audio bundles"
    );

    let mut session = engine.new_session(SessionConfig::default());

    // 0.5 s of synthetic deterministic noise — same fixture the
    // direct-attach test uses; not asserting on token quality.
    let n_samples = (SAMPLE_RATE as usize) / 2;
    let pcm: Vec<f32> = (0..n_samples)
        .map(|i| ((i as f32 * 0.07).sin() + (i as f32 * 0.013).sin()) * 0.05)
        .collect();

    let pos_before = session.position();
    session
        .append_audio(&pcm, SAMPLE_RATE)
        .expect("append_audio");
    assert!(
        session.position() > pos_before,
        "append_audio must advance position when engine pre-attached the encoder"
    );

    // Sanity: a follow-up generate succeeds, proving the chain
    // (engine load → auto-attach → encode → append_embeddings →
    // last_logits chain) works end-to-end.
    let opts = greedy_opts(1);
    let mut sink = CollectSink(Vec::new());
    let _summary = session.generate(&opts, &mut sink).expect("generate");
}

/// Negative path: text-only bundle (the LFM2-VL fallback) must not
/// have an audio encoder pre-loaded. Confirms the inference_type
/// gate in `try_load_audio_encoder`.
#[test]
#[ignore]
fn engine_does_not_load_encoder_for_text_bundle() {
    use wick::{EngineConfig, WickEngine};

    let Some(model_path) = find_model() else {
        eprintln!("no model available — skipping");
        return;
    };
    // Direct .gguf load synthesizes a Text manifest — no mmproj
    // even if the LFM2-VL bundle has one in its real manifest.
    let engine = WickEngine::from_path(&model_path, EngineConfig::default()).unwrap();
    assert!(
        engine.audio_encoder().is_none(),
        "text-path engine must not eagerly load any audio encoder"
    );
}
