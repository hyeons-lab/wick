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

use wick::kv_cache::KvCompression;
use wick::{FinishReason, GenerateOpts, ModalitySink, Session, SessionConfig};

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

/// Regression test for PR #27 review comment #4:
/// After `generate()` returns, a second `generate()` call on the same
/// session must continue from the saved logits without requiring an
/// intervening `append_tokens`. Before the fix, the second call returned
/// `WickError::EmptyInput` because `last_logits.take()` left `None`.
#[test]
#[ignore]
fn generate_is_chainable_without_append() {
    let Some(model_path) = find_model() else {
        eprintln!("no model available — skipping");
        return;
    };

    let gguf = wick::gguf::GgufFile::open(&model_path).unwrap();
    let tokenizer = wick::tokenizer::BpeTokenizer::from_gguf(&gguf).unwrap();
    let model = wick::model::load_model(gguf, 4096).unwrap();
    let prompt_toks = tokenizer.encode("The capital of France is");

    let mut session = Session::new(
        model.as_ref(),
        &tokenizer,
        SessionConfig {
            kv_compression: KvCompression::None,
            seed: Some(42),
            ..Default::default()
        },
    );
    session.append_tokens(&prompt_toks).unwrap();

    // First call — 4 tokens.
    let mut sink_a = CollectSink(Vec::new());
    session.generate(&greedy_opts(4), &mut sink_a).unwrap();
    assert_eq!(sink_a.0.len(), 4, "first call should generate 4 tokens");

    // Second call on the SAME session — must succeed (was EmptyInput before).
    let mut sink_b = CollectSink(Vec::new());
    let summary = session.generate(&greedy_opts(4), &mut sink_b).unwrap();
    assert_eq!(
        summary.tokens_generated, 4,
        "chained call should generate 4 tokens"
    );
    assert_eq!(sink_b.0.len(), 4);

    // Position advanced by the full 8 tokens across both calls.
    assert_eq!(session.position() as usize, prompt_toks.len() + 8);
}

/// Regression test for PR #27 review comment #3:
/// After a bounded `generate()` (hit `max_tokens`), the KV state must not
/// lag `current_pos`. A subsequent `append_tokens` + `generate()` must
/// produce coherent output — not gibberish caused by writing to a stale
/// KV slot. Verifies by comparing split generation vs. single-call
/// generation over the same total token budget.
#[test]
#[ignore]
fn no_kv_gap_across_bounded_generate() {
    let Some(model_path) = find_model() else {
        eprintln!("no model available — skipping");
        return;
    };

    let gguf = wick::gguf::GgufFile::open(&model_path).unwrap();
    let tokenizer = wick::tokenizer::BpeTokenizer::from_gguf(&gguf).unwrap();
    let prompt_toks = tokenizer.encode("The capital of France is");

    // Baseline: one session, single generate(8).
    let baseline = {
        let gguf = wick::gguf::GgufFile::open(&model_path).unwrap();
        let model = wick::model::load_model(gguf, 4096).unwrap();
        let mut session = Session::new(
            model.as_ref(),
            &tokenizer,
            SessionConfig {
                seed: Some(7),
                ..Default::default()
            },
        );
        session.append_tokens(&prompt_toks).unwrap();
        let mut sink = CollectSink(Vec::new());
        session.generate(&greedy_opts(8), &mut sink).unwrap();
        sink.0
    };

    // Split: same session, two generate(4) calls back-to-back.
    let split = {
        let gguf = wick::gguf::GgufFile::open(&model_path).unwrap();
        let model = wick::model::load_model(gguf, 4096).unwrap();
        let mut session = Session::new(
            model.as_ref(),
            &tokenizer,
            SessionConfig {
                seed: Some(7),
                ..Default::default()
            },
        );
        session.append_tokens(&prompt_toks).unwrap();
        let mut sink1 = CollectSink(Vec::new());
        session.generate(&greedy_opts(4), &mut sink1).unwrap();
        let mut sink2 = CollectSink(Vec::new());
        session.generate(&greedy_opts(4), &mut sink2).unwrap();
        let mut all = sink1.0;
        all.extend(sink2.0);
        all
    };

    assert_eq!(
        baseline, split,
        "split greedy generation must match single call — KV gap would diverge them.\nbaseline: {baseline:?}\nsplit:    {split:?}"
    );
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

    let mut session = Session::new(
        model.as_ref(),
        &tokenizer,
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

    let mut session = Session::new(
        model.as_ref(),
        &tokenizer,
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
    let tokenizer = wick::tokenizer::BpeTokenizer::from_gguf(&gguf).unwrap();
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
        let model = wick::model::load_model(gguf, 4096).unwrap();
        let mut session = Session::new(
            model.as_ref(),
            &tokenizer,
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
        let model = wick::model::load_model(gguf, 4096).unwrap();
        let mut session = Session::new(
            model.as_ref(),
            &tokenizer,
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

    let mut session = Session::new(
        model.as_ref(),
        &tokenizer,
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
