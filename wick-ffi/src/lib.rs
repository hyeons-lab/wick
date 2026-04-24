//! `wick-ffi` — foreign-language bindings to [`wick`] via UniFFI.
//!
//! This crate exposes a subset of the `wick` inference engine to
//! Kotlin, Swift, Python, and any other language UniFFI supports. It
//! is structured around the **proc-macro** path (rather than a UDL
//! file) so the Rust types we expose are the source of truth and
//! annotations stay colocated with the code they describe.
//!
//! ## Scope of the current bootstrap
//!
//! This PR ships the crate shell + validates the UniFFI build. The
//! exposed surface is deliberately minimal — a single version accessor
//! — to keep the first review small. Follow-up PRs grow the surface
//! per the Phase 2 roadmap:
//!
//! 1. `WickEngine::from_path`, `EngineConfig`, `ModelMetadata`,
//!    `ModalityCapabilities`.
//! 2. `Session`, `SessionConfig`, `GenerateOpts`, `GenerateSummary`,
//!    synchronous `generate`.
//! 3. `ModalitySink` as a UniFFI foreign-trait callback.
//! 4. `async` `generate` via `#[uniffi::export(async_runtime = "tokio")]`.
//! 5. Kotlin + Swift binding generation + vendored outputs.
//! 6. Error-type marshalling, parity harness, Android / iOS builds.
//!
//! ## Why proc-macro (not UDL)
//!
//! UniFFI 0.28 supports both. The proc-macro path is lighter for a
//! small surface, lets the FFI annotations live next to the Rust
//! types, and doesn't require a separate grammar. UDL wins for very
//! large surfaces where a single-file schema is easier to review as a
//! block — we can migrate then if the annotation density becomes
//! unmanageable, but the current plan is comfortably proc-macro-sized.

uniffi::setup_scaffolding!();

/// Version string of the underlying `wick-ffi` crate. Smoke test for
/// the UniFFI build pipeline — if this compiles, links as `cdylib`,
/// and is callable from a generated binding, the foundational FFI
/// wiring works end-to-end.
#[uniffi::export]
pub fn wick_ffi_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_non_empty_semver_ish() {
        let v = wick_ffi_version();
        assert!(!v.is_empty());
        // Loose shape check — three dot-separated chunks.
        assert_eq!(
            v.split('.').count(),
            3,
            "expected semver-shaped version: {v}"
        );
    }
}
