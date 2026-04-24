//! Cached HTTP download helper for gated integration tests.
//!
//! Thin shim over `wick::bundle::BundleRepo` — the download logic used
//! to live in this file, but the lib version (behind the `remote`
//! feature) now owns it. Tests that call `ensure_cached` must compile
//! with `--features remote`; the module is `#[cfg]`'d accordingly so a
//! plain `cargo test` still builds cleanly (it just won't see these
//! helpers).
//!
//! Cache root: `$WICK_TEST_MODELS_DIR` if set, else
//! `target/tmp/wick-test-models` under the workspace root. A
//! `target/tmp/` subdir is chosen (rather than `target/debug/tmp/` or
//! similar) so the CI cache stanza in `.github/` can key on a single
//! stable path across debug/release invocations.
//!
//! Used only by tests gated on `WICK_TEST_DOWNLOAD=1` — never runs in a
//! default `cargo test` invocation.

use std::fs;
use std::path::PathBuf;

use wick::bundle::BundleRepo;

/// Returns the resolved cache directory, creating it if missing.
///
/// Public so tests that need to build their own `BundleRepo` rooted at
/// the same location (e.g. for cache-hit assertions) can share the
/// directory with `ensure_cached`.
pub fn cache_dir() -> PathBuf {
    if let Ok(override_path) = std::env::var("WICK_TEST_MODELS_DIR") {
        let p = PathBuf::from(override_path);
        fs::create_dir_all(&p).expect("create WICK_TEST_MODELS_DIR");
        return p;
    }
    // Workspace-root-relative. `CARGO_MANIFEST_DIR` points at the crate
    // (`wick/`), so one dir up is the workspace root.
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest
        .parent()
        .expect("CARGO_MANIFEST_DIR has no parent")
        .join("target")
        .join("tmp")
        .join("wick-test-models");
    fs::create_dir_all(&root).expect("create cache_dir");
    root
}

/// Ensure `url` is present under the test cache, returning the local
/// path. Delegates to `BundleRepo` so the caching + integrity policy
/// matches production (etag-preferred, size-fallback, atomic rename).
///
/// `filename` is kept for API compatibility with earlier test code but
/// is no longer used — `BundleRepo` derives the on-disk path from the
/// URL's host+path.
///
/// Panics on unrecoverable errors (no file + download failed). Tests
/// that call this are already gated on `WICK_TEST_DOWNLOAD=1`, so a
/// panic here is the correct failure mode.
pub fn ensure_cached(url: &str, _filename: &str) -> PathBuf {
    let repo = BundleRepo::new(cache_dir());
    repo.resolve_url(url, None)
        .unwrap_or_else(|e| panic!("resolve {url}: {e}"))
}
