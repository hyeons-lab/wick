//! End-to-end smoke test for `BundleRepo` against a real HuggingFace
//! URL. Proves the downloader + integrity check + cache-hit codepath
//! actually work against the upstream CDN, on a real GGUF.
//!
//! Two assertions:
//! 1. A fresh `BundleRepo` resolves an HF URL, downloads the file, and
//!    the resulting path loads into a `WickEngine` without error —
//!    this exercises SHA-256-on-the-fly hashing + `X-Linked-Etag`
//!    verification against the live HF CDN.
//! 2. A second `resolve_url` call for the same URL returns the same
//!    path without re-downloading (proved by comparing mtimes).
//!
//! Gating: `#[ignore]` + `WICK_TEST_DOWNLOAD=1` env var so `cargo test`
//! never hits the network. To opt in:
//!
//! ```sh
//! WICK_TEST_DOWNLOAD=1 cargo test -p wick --features remote \
//!     --test bundle_download -- --ignored
//! ```
//!
//! On CI, the download is cached under `target/tmp/wick-test-models/`
//! so repeat runs pay only an HTTP HEAD probe.

#![cfg(feature = "remote")]

mod common;

use std::time::SystemTime;

use wick::bundle::BundleRepo;
use wick::engine::{BackendPreference, EngineConfig, ModelFiles, WickEngine};

/// Smallest LFM2 GGUF we already pull elsewhere (see
/// `shift_real_model.rs`). Deliberately the same file so CI's
/// `target/tmp/wick-test-models` cache is shared — the second test to
/// run pays only a HEAD probe. Mutable `resolve/main/` ref is
/// intentional for the same reasons documented in `shift_real_model.rs`;
/// PR B will pin to a revision hash once `BundleRepo` grows an ID-
/// indexed lookup layer.
const MODEL_URL: &str =
    "https://huggingface.co/LiquidAI/LFM2-1.2B-GGUF/resolve/main/LFM2-1.2B-Q4_0.gguf";

#[test]
#[ignore = "downloads ~700 MB; set WICK_TEST_DOWNLOAD=1 and pass --ignored"]
fn bundle_repo_resolves_and_loads_from_hf() {
    if std::env::var("WICK_TEST_DOWNLOAD").is_err() {
        eprintln!("skipping: WICK_TEST_DOWNLOAD not set");
        return;
    }

    // Use the shared test cache via the `common` helper — exercises the
    // same BundleRepo codepath but rooted in the CI-cached location so
    // repeat runs don't redownload.
    let path = common::download::ensure_cached(MODEL_URL, "LFM2-1.2B-Q4_0.gguf");
    assert!(
        path.exists(),
        "BundleRepo.resolve_url did not produce an existing file at {}",
        path.display()
    );

    // Sanity: the file is large enough to be a real GGUF (>500 MB).
    let size = std::fs::metadata(&path)
        .expect("stat downloaded file")
        .len();
    assert!(
        size > 500 * 1024 * 1024,
        "downloaded file only {size} bytes — HF returned an error page?"
    );

    // End-to-end: feed the resolved path into WickEngine. A bogus
    // download (truncated / wrong content-type page) would fail GGUF
    // header parse here, so this doubles as an integrity smoke test.
    let engine = WickEngine::from_files(
        ModelFiles::text(&path),
        EngineConfig {
            context_size: 128,
            backend: BackendPreference::Cpu,
            ..Default::default()
        },
    )
    .expect("load engine from resolved bundle file");
    let meta = engine.metadata();
    assert!(
        meta.max_seq_len > 0,
        "engine metadata missing max_seq_len — model parse failed silently"
    );
}

#[test]
#[ignore = "downloads ~700 MB; set WICK_TEST_DOWNLOAD=1 and pass --ignored"]
fn bundle_repo_cache_hit_does_not_redownload() {
    if std::env::var("WICK_TEST_DOWNLOAD").is_err() {
        eprintln!("skipping: WICK_TEST_DOWNLOAD not set");
        return;
    }

    // Pre-populate via the shared helper so we're measuring a cache hit
    // on the *second* call regardless of whether the first test ran.
    let first = common::download::ensure_cached(MODEL_URL, "LFM2-1.2B-Q4_0.gguf");
    let mtime_before = first
        .metadata()
        .and_then(|m| m.modified())
        .unwrap_or(SystemTime::UNIX_EPOCH);

    // A fresh BundleRepo pointed at the same store_dir must see the
    // cached file via the HEAD-probe policy — no rewrite.
    let repo = BundleRepo::new(common::download::cache_dir());
    let second = repo
        .resolve_url(MODEL_URL, None)
        .expect("resolve on cache hit");
    assert_eq!(first, second, "cache-hit path should match the first call");

    let mtime_after = second
        .metadata()
        .and_then(|m| m.modified())
        .unwrap_or(SystemTime::UNIX_EPOCH);
    assert_eq!(
        mtime_before, mtime_after,
        "cache hit should not rewrite the file (mtime changed)"
    );
}
