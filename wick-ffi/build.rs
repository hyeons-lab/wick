// UniFFI build-time hook. In proc-macro mode (our choice — see
// `src/lib.rs` for rationale) the crate doesn't need a UDL-based
// scaffolding generator; `uniffi::setup_scaffolding!()` handles it at
// macro expansion. This file is retained as an explicit marker that
// UniFFI is involved in the build so the build.rs presence flags this
// crate to tooling (e.g. `uniffi-bindgen`) that expects the hook.

fn main() {
    // Rebuild only when the library source actually changes. Without
    // this, any dep touch forces wick-ffi to rerun its (trivial) build
    // script unnecessarily.
    println!("cargo:rerun-if-changed=src/lib.rs");
}
