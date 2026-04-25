//! Real-FFI-boundary parity check: rust ↔ kotlin-via-JNA.
//!
//! Drives the same prompt through `wick::WickEngine` (Rust reference)
//! and the vendored Kotlin runner under `wick-parity/legs/kotlin/`,
//! then asserts byte-identical greedy-decoded token streams. Unlike
//! the rust ↔ ffi check in `parity.rs`, this one actually crosses the
//! UniFFI Kotlin binding boundary — a marshalling bug in any
//! `Record` / `Object` / scalar conversion would surface as token
//! divergence.
//!
//! Two env vars gate this test (belt-and-suspenders, mirrors the
//! `WICK_TEST_DOWNLOAD` style):
//!   - `WICK_PARITY_RUN=1` — opt into the ~210 MB model download.
//!   - `WICK_PARITY_KOTLIN_RUNNER=<path/to/wick-parity-kotlin-all.jar>` —
//!     points at the fat jar produced by
//!     `cd wick-parity/legs/kotlin && ./gradlew shadowJar`.
//!
//! Optional:
//!   - `WICK_PARITY_LIB_DIR=<path/to/dir/containing/libwick_ffi>` —
//!     where JNA should look for the cdylib. Defaults to
//!     `<workspace>/target/debug` (the `cargo build -p wick-ffi`
//!     output dir). Set explicitly when running against a release
//!     build.
//!
//! Manual invocation:
//!   ```sh
//!   cd wick-parity/legs/kotlin && ./gradlew shadowJar
//!   cargo build -p wick-ffi
//!   WICK_PARITY_RUN=1 \
//!     WICK_PARITY_KOTLIN_RUNNER=$(pwd)/wick-parity/legs/kotlin/build/libs/wick-parity-kotlin-all.jar \
//!     cargo test -p wick-parity --test parity_kotlin -- --ignored
//!   ```

use std::path::PathBuf;

#[test]
#[ignore = "downloads ~210 MB + needs WICK_PARITY_KOTLIN_RUNNER set"]
fn rust_and_kotlin_jna_produce_identical_tokens() {
    if std::env::var("WICK_PARITY_RUN").is_err() {
        eprintln!("skipping: WICK_PARITY_RUN not set");
        return;
    }
    let runner = match std::env::var("WICK_PARITY_KOTLIN_RUNNER") {
        Ok(p) => PathBuf::from(p),
        Err(_) => {
            eprintln!("skipping: WICK_PARITY_KOTLIN_RUNNER not set");
            return;
        }
    };
    assert!(
        runner.exists(),
        "WICK_PARITY_KOTLIN_RUNNER points at {} which does not exist",
        runner.display()
    );

    // Default lib_dir = <workspace>/target/debug. CARGO_MANIFEST_DIR
    // points at this crate (wick-parity); workspace root is one up.
    let lib_dir = std::env::var("WICK_PARITY_LIB_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            manifest
                .parent()
                .expect("CARGO_MANIFEST_DIR has no parent")
                .join("target")
                .join("debug")
        });

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
    let kotlin = wick_parity::run_kotlin_jna(&args, &runner, &lib_dir).expect("kotlin-jna path");

    if let Some(idx) = wick_parity::first_divergence(&rust, &kotlin) {
        let start = idx.saturating_sub(2);
        let end = idx.saturating_add(3);
        let rust_window = start..end.min(rust.len());
        let kotlin_window = start..end.min(kotlin.len());
        let rust_dump = format!("{:?}", &rust[rust_window.clone()]);
        let kotlin_dump = format!("{:?}", &kotlin[kotlin_window.clone()]);
        panic!(
            "rust ↔ kotlin-jna diverged at index {idx}\n  rust  [{rust_window:?}] = {rust_dump}\n  kotlin[{kotlin_window:?}] = {kotlin_dump}\n  rust.len() = {}, kotlin.len() = {}",
            rust.len(),
            kotlin.len(),
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
