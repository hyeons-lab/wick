// UniFFI build-time hook. In proc-macro mode (our choice — see
// `src/lib.rs` for rationale) the crate doesn't need a UDL-based
// scaffolding generator; `uniffi::setup_scaffolding!()` handles it at
// macro expansion. This file is retained as an explicit marker that
// UniFFI is involved in the build so the build.rs presence flags this
// crate to tooling (e.g. `uniffi-bindgen`) that expects the hook.

fn main() {
    // Rebuild only when library source changes. `src` (directory)
    // over `src/lib.rs` so the rerun trigger catches any source file
    // added to the crate without needing this list to grow. Without
    // this, a touch in any workspace dep would force wick-ffi to
    // rerun its (trivial) build script.
    println!("cargo:rerun-if-changed=src");
}
