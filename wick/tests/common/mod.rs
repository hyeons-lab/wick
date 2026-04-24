//! Shared helpers for integration tests.
//!
//! Lives under `tests/common/` (not `tests/helpers/`) to follow Cargo's
//! convention for non-test files inside the integration-test directory
//! — Cargo skips the `common/` subdir when picking up test binaries.
//!
//! Currently just [`download::ensure_cached`] for tests that need a real
//! GGUF but don't want to bake multi-hundred-MB fixtures into the repo.
//! The download helper compiles only when the `remote` feature is
//! active; callers are `#[cfg(feature = "remote")]`'d accordingly.

#![allow(dead_code)]

#[cfg(feature = "remote")]
pub mod download;
