//! End-to-end test for `WickEngine::from_bundle_id` against the real
//! LiquidAI/LeapBundles catalog on HuggingFace.
//!
//! Proves the full chain: bundle ID + quant → canonical LeapBundles
//! manifest URL → fetch + cache manifest → parse → resolve primary-
//! model URL (which points back at the model's own HF repo) → mmap-
//! open → load engine. If any link silently produces junk (truncated
//! download, wrong URL, manifest-schema drift), the engine load fails
//! loudly.
//!
//! Gating: `#[ignore]` + `WICK_TEST_DOWNLOAD=1`. To opt in:
//!
//! ```sh
//! WICK_TEST_DOWNLOAD=1 cargo test -p wick --features remote \
//!     --test bundle_from_id -- --ignored
//! ```
//!
//! Shares the `target/tmp/wick-test-models/` cache with
//! `shift_real_model` and `bundle_download` — the GGUF URL resolves to
//! the same file across all three, so repeat runs pay only a HEAD probe.

#![cfg(all(feature = "remote", feature = "mmap"))]

mod common;

use wick::bundle::BundleRepo;
use wick::engine::{BackendPreference, EngineConfig, WickEngine};

#[test]
#[ignore = "downloads ~210 MB; set WICK_TEST_DOWNLOAD=1 and pass --ignored"]
fn from_bundle_id_loads_lfm2_q4_0() {
    if std::env::var("WICK_TEST_DOWNLOAD").is_err() {
        eprintln!("skipping: WICK_TEST_DOWNLOAD not set");
        return;
    }

    let repo = BundleRepo::new(common::download::cache_dir());
    let engine = WickEngine::from_bundle_id(
        "LFM2-350M-Extract-GGUF",
        "Q4_0",
        EngineConfig {
            context_size: 128,
            backend: BackendPreference::Cpu,
            bundle_repo: Some(repo),
            ..Default::default()
        },
    )
    .expect("load engine from bundle id");

    let meta = engine.metadata();
    assert!(
        meta.max_seq_len > 0,
        "engine metadata missing max_seq_len — model parse failed silently"
    );
    assert!(
        !meta.architecture.is_empty(),
        "engine metadata missing architecture"
    );
}

#[test]
fn from_bundle_id_fails_without_bundle_repo() {
    // Fast negative test — no network, no feature gate needed at
    // runtime. Catches the "did we remember to require bundle_repo?"
    // regression cheaply.
    // Can't use `.expect_err()` because `WickEngine` holds a
    // `Box<dyn Model>` and therefore doesn't derive `Debug`.
    let result = WickEngine::from_bundle_id(
        "LFM2-350M-Extract-GGUF",
        "Q4_0",
        EngineConfig {
            context_size: 128,
            backend: BackendPreference::Cpu,
            bundle_repo: None,
            ..Default::default()
        },
    );
    match result {
        Ok(_) => panic!("missing bundle_repo must be an error, but got Ok"),
        Err(e) => {
            let msg = format!("{e}");
            assert!(
                msg.contains("bundle_repo"),
                "error should name the missing config field; got `{msg}`"
            );
        }
    }
}
