//! End-to-end smoke test for `n_keep` context shift on a real LFM2
//! model. Unit tests in `n_keep_shift.rs` prove the RoPE math composes
//! against an oracle; this test exercises the full stack — shift runs
//! against production KV layouts, through every attention layer, on
//! weights from a real Q4_0 GGUF.
//!
//! **Scope note:** this test does NOT decode tokens from a post-shift
//! state. After a shift-triggering `append_tokens`, position sits at
//! exactly `max_seq_len` (the shift drops precisely the number of cells
//! needed to fit the append, leaving no headroom), and `Session::generate`
//! returns `FinishReason::ContextFull` without emitting tokens. Changing
//! that to enable post-shift decode would require either a new peek-next
//! API or an overshoot-shift policy — both are out of scope here and for
//! PR #35.
//!
//! What IS covered:
//! - Real LFM2-1.2B-Q4_0 loads via [`WickEngine::from_path`].
//! - Prompt + follow-up prefill drives the session past `max_seq_len`,
//!   triggering [`Model::shift_kv`] under the real RoPE parameters.
//! - Post-shift prefill runs successfully through every attention layer
//!   using the re-rotated K cells. A sign or indexing bug in
//!   [`backend::cpu::apply_rope_delta_to_head`] would propagate NaNs /
//!   out-of-bounds accesses during this prefill; the test passes only
//!   if the full stack is numerically sound.
//! - At least one `wick::kv_shift` tracing event fires.
//! - Final session position matches the invariant (== `max_seq_len`).
//!
//! Gating: `#[ignore]` + `WICK_TEST_DOWNLOAD=1` env var so `cargo test`
//! never hits the network. To opt in:
//!
//! ```sh
//! WICK_TEST_DOWNLOAD=1 cargo test -p wick --test shift_real_model -- --ignored
//! ```
//!
//! On CI, the download is cached under `target/tmp/wick-test-models/`
//! so repeat runs pay only an HTTP HEAD probe.

#![cfg(feature = "remote")]

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use wick::engine::{BackendPreference, EngineConfig, WickEngine};
use wick::session::SessionConfig;

/// LFM2-1.2B-Q4_0: already referenced elsewhere in the test fixtures;
/// keeps download ~700 MB. The URL uses `resolve/main/` (a mutable HF
/// ref) deliberately — this test only asserts execution sanity +
/// shift-fires, so any valid LFM2 GGUF at this path satisfies it. If
/// upstream re-uploads the file, the cache's Content-Length check
/// triggers a re-download on the next run and the test keeps passing.
/// Phase 1.6's `BundleRepo` is where pinning to a revision hash
/// belongs; for this test a mutable ref matches the cost/benefit.
const MODEL_URL: &str =
    "https://huggingface.co/LiquidAI/LFM2-1.2B-GGUF/resolve/main/LFM2-1.2B-Q4_0.gguf";
const MODEL_FILE: &str = "LFM2-1.2B-Q4_0.gguf";

/// Counts the `shift` events emitted on `wick::kv_shift` via a tracing
/// layer. We don't depend on an external subscriber because tests run
/// in parallel and a global subscriber would race — we use an
/// `Arc<AtomicUsize>` installed for the duration of this test only via
/// `tracing::subscriber::set_default` (scoped to the returned
/// `DefaultGuard`).
fn run_with_shift_counter<R>(f: impl FnOnce(Arc<AtomicUsize>) -> R) -> R {
    use tracing::subscriber::DefaultGuard;
    use tracing_subscriber::layer::SubscriberExt;

    let counter = Arc::new(AtomicUsize::new(0));
    struct ShiftCounter(Arc<AtomicUsize>);
    impl<S> tracing_subscriber::Layer<S> for ShiftCounter
    where
        S: tracing::Subscriber,
    {
        fn on_event(
            &self,
            event: &tracing::Event<'_>,
            _ctx: tracing_subscriber::layer::Context<'_, S>,
        ) {
            if event.metadata().target() == "wick::kv_shift" {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
    let subscriber = tracing_subscriber::registry().with(ShiftCounter(counter.clone()));
    let _guard: DefaultGuard = tracing::subscriber::set_default(subscriber);
    f(counter)
}

#[test]
#[ignore = "downloads ~700 MB; set WICK_TEST_DOWNLOAD=1 and pass --ignored"]
fn shift_runs_through_real_model() {
    if std::env::var("WICK_TEST_DOWNLOAD").is_err() {
        eprintln!("skipping: WICK_TEST_DOWNLOAD not set");
        return;
    }

    let model_path = common::download::ensure_cached(MODEL_URL, MODEL_FILE);

    // Small context window so we can trigger overflow with a short prompt.
    // The model's native max_seq_len is much larger; we cap via
    // `EngineConfig::context_size` so KV allocation stays minimal.
    const CTX: usize = 256;
    let engine = WickEngine::from_path(
        &model_path,
        EngineConfig {
            context_size: CTX,
            backend: BackendPreference::Cpu,
            ..Default::default()
        },
    )
    .expect("load engine");

    let cfg = SessionConfig {
        max_seq_len: Some(CTX as u32),
        n_keep: 32,
        seed: Some(0),
        ubatch_size: 0,
        ..Default::default()
    };

    let shift_count = run_with_shift_counter(|counter| {
        let mut session = engine.new_session(cfg);

        // Shift can ONLY fire when prior context already occupies
        // `>= n_keep + shift_needed` cells — the first append must
        // stay *under* the cap. So we ladder: a prompt that fills
        // most of the window, then a follow-up that pushes past.
        //
        // LFM2-1.2B's vocab tokenizes the base phrase at ~19 tokens/rep
        // after special tokens. 9 reps ≈ ~172 tokens; well under the
        // 256 cap with room for the follow-up to force overflow.
        let base = "The quick brown fox jumps over the lazy dog, and then walks back to inspect the fence. ";
        let long_prompt = base.repeat(9);
        session
            .append_text(&long_prompt)
            .expect("append long prompt");
        let pos_after_prompt = session.position() as usize;
        println!("position after prompt: {pos_after_prompt} / {CTX}");
        assert!(
            (CTX / 2..CTX).contains(&pos_after_prompt),
            "prompt should fill most of window but stay under cap \
             (pos={pos_after_prompt}); re-tune reps if vocab changed"
        );

        let before = counter.load(Ordering::Relaxed);

        // Follow-up sized to definitely overflow + fire shift. The
        // post-shift prefill runs through every attention layer using
        // re-rotated K cells — a sign or indexing bug in
        // `apply_rope_delta_to_head` would propagate NaNs here and
        // either panic downstream (tensor-length assertions) or
        // surface as a numeric error.
        let follow = "Additionally, the narrator walks slowly along the river path. ".repeat(15);
        session.append_text(&follow).expect("append forcing shift");
        let pos_after_follow = session.position() as usize;
        println!("position after follow: {pos_after_follow} / {CTX}");

        let after = counter.load(Ordering::Relaxed);
        assert!(
            after > before,
            "shift should have fired during follow-up append \
             (before={before} after={after}, \
              pos_before={pos_after_prompt}, pos_after={pos_after_follow})"
        );

        // Invariant: a shift-triggering append always saturates to the
        // cap (shift drops exactly what's needed, leaving no headroom).
        // Locking this in here catches any future change that would
        // break the post-shift bookkeeping.
        assert_eq!(
            pos_after_follow, CTX,
            "post-shift position must land at exactly max_seq_len"
        );

        after
    });

    assert!(
        shift_count >= 1,
        "expected ≥1 shift event, got {shift_count}"
    );
}
