//! `uniffi-bindgen` — binding generator CLI for `wick-ffi`.
//!
//! Thin wrapper around `uniffi::uniffi_bindgen_main()` so the standard
//! `uniffi-bindgen generate ...` command is available in-repo without
//! installing `uniffi_bindgen` globally. Invoked from `just bindings`
//! and from CI (`.github/workflows/rust.yml`'s `uniffi-bindings` job).
//!
//! Typical use:
//!
//! ```text
//! cargo run -p wick-ffi --bin uniffi-bindgen -- \
//!     generate --library target/debug/libwick_ffi.dylib \
//!     --language kotlin --out-dir wick-ffi/bindings/kotlin
//! ```
//!
//! See `wick-ffi/README.md` for the full workflow.

fn main() {
    uniffi::uniffi_bindgen_main()
}
