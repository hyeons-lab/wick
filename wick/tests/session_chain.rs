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
