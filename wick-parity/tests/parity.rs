//! Real-model parity check.
//!
//! Downloads `LFM2-350M-Extract-GGUF/Q4_0` (~200 MB) on first run via
//! `BundleRepo::from_bundle_id`, caches it under
//! `target/tmp/wick-parity-cache/`, and asserts that running the same
//! prompt through `wick::WickEngine` (the rust leg) and through
//! `wick_ffi::WickEngine` (the ffi leg) yields byte-identical greedy
//! token streams.
//!
//! Gating mirrors `wick/tests/shift_real_model.rs`:
//!   - `#[ignore]` so default `cargo test` doesn't trigger a 200 MB
//!     download.
//!   - `WICK_PARITY_RUN=1` env var must be set; otherwise the test
//!     prints a skip line and returns. Belt-and-suspenders for CI
//!     where `--ignored` is passed but the gate isn't intentionally
//!     opted into.
//!
//! Manual invocation:
//!   `WICK_PARITY_RUN=1 cargo test -p wick-parity -- --ignored`
//!
//! CI invocation: see the `Parity Harness (gated)` job in
//! `.github/workflows/ci.yml`.

#[test]
#[ignore = "downloads ~200 MB; set WICK_PARITY_RUN=1 and pass --ignored"]
fn rust_and_ffi_produce_identical_tokens() {
    if std::env::var("WICK_PARITY_RUN").is_err() {
        eprintln!("skipping: WICK_PARITY_RUN not set");
        return;
    }

    let cache = wick_parity::default_cache_dir().expect("cache dir");
    let args = wick_parity::RunArgs {
        bundle: "LFM2-350M-Extract-GGUF",
        quant: "Q4_0",
        prompt: "The capital of France is",
        max_tokens: 16,
        seed: 0,
        cache_dir: &cache,
    };

    let rust = wick_parity::run_rust(&args).expect("rust path");
    let ffi = wick_parity::run_ffi(&args).expect("ffi path");

    // Greedy decoding is deterministic across runs; the rust and ffi
    // legs share the same library underneath so they MUST agree
    // token-for-token. Divergence here means the FFI wrapper's
    // `RecordType -> wick::Type` adapter is dropping or reordering
    // something that influences sampling (seed, temperature, top_k).
    if let Some(idx) = wick_parity::first_divergence(&rust, &ffi) {
        let window = idx.saturating_sub(2)..idx.saturating_add(3);
        panic!(
            "rust ↔ ffi diverged at index {idx}\n  rust[{window:?}] = {:?}\n  ffi [{window:?}] = {:?}\n  rust.len() = {}, ffi.len() = {}",
            rust.get(window.clone()),
            ffi.get(window.clone()),
            rust.len(),
            ffi.len(),
        );
    }
    assert!(
        rust.len() <= 16,
        "honored max_tokens cap (got {} tokens)",
        rust.len()
    );
    assert!(
        !rust.is_empty(),
        "expected at least one decoded token from a non-empty prompt"
    );
}
