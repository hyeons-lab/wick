//! Shared helpers for integration tests.
//!
//! Lives under `tests/common/` (not `tests/helpers/`) to follow Cargo's
//! convention for non-test files inside the integration-test directory
//! — Cargo skips the `common/` subdir when picking up test binaries.
//!
//! Currently just [`download::ensure_cached`] for tests that need a real
//! GGUF but don't want to bake multi-hundred-MB fixtures into the repo.

#![allow(dead_code)]

pub mod download;
