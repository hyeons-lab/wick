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
//! - Real LFM2-350M-Extract-Q4_0 loads via [`WickEngine::from_bundle_id`].
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
//! so repeat runs pay only an HTTP HEAD probe. The fixture is the same
//! `LFM2-350M-Extract-GGUF/Q4_0` shared with `bundle_download.rs` and
//! `bundle_from_id.rs` — one ~210 MB cache entry covers all three
//! gated integration tests. We picked the 350M-Extract over the 1.2B
//! base because the 1.2B took ~13 minutes to run on the Ubuntu CI
//! runner (no BLAS, ~3500 tokens of prefill across 8 layers) and was
//! hitting the job's 15-minute cap; the 350M brings that down to a
//! few minutes with the same shift-firing coverage.

#![cfg(feature = "remote")]

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use wick::bundle::BundleRepo;
use wick::engine::{BackendPreference, EngineConfig, WickEngine};
use wick::session::SessionConfig;

/// LeapBundles bundle id + quant for the test fixture. Resolves to
/// `LFM2-350M-Extract-Q4_0.gguf` (~209 MB) via the manifest at
/// `LiquidAI/LeapBundles/.../LFM2-350M-Extract-GGUF/Q4_0.json`. Same
/// underlying GGUF as `bundle_from_id.rs` exercises so the test cache
/// is shared. LFM2-Extract is the same architecture + tokenizer family
/// as the LFM2 base models — the shift mechanism doesn't depend on the
/// model being a chat model, only that it has RoPE'd attention
/// (which all current LFM2s do).
const BUNDLE_ID: &str = "LFM2-350M-Extract-GGUF";
const QUANT: &str = "Q4_0";

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
#[ignore = "downloads ~210 MB; set WICK_TEST_DOWNLOAD=1 and pass --ignored"]
fn shift_runs_through_real_model() {
    if std::env::var("WICK_TEST_DOWNLOAD").is_err() {
        eprintln!("skipping: WICK_TEST_DOWNLOAD not set");
        return;
    }

    // Small context window so we can trigger overflow with a short prompt.
    // The model's native max_seq_len is much larger; we cap via
    // `EngineConfig::context_size` so KV allocation stays minimal.
    const CTX: usize = 256;
    // Bundle-based loading dogfoods the same `from_bundle_id` path that
    // `wick-parity` and `bundle_from_id.rs` exercise; the shared cache
    // under `wick-test-models/` means at most one HTTP fetch per run.
    let repo = BundleRepo::new(common::download::cache_dir());
    let engine = WickEngine::from_bundle_id(
        BUNDLE_ID,
        QUANT,
        EngineConfig {
            context_size: CTX,
            backend: BackendPreference::Cpu,
            bundle_repo: Some(repo),
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
        // LFM2's vocab tokenizes the base phrase at ~19 tokens/rep
        // after special tokens (consistent across LFM2 sizes — same
        // tokenizer + vocab regardless of base vs Extract). 9 reps
        // ≈ ~172 tokens; well under the 256 cap with room for the
        // follow-up to force overflow.
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
