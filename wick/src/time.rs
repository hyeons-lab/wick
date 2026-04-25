//! Cross-target time primitives.
//!
//! `std::time::Instant` compiles on `wasm32-unknown-unknown` since
//! Rust 1.74 but **panics at runtime** on the first call (the wasm
//! spec has no monotonic clock; the std stub is a placeholder that
//! aborts when actually invoked). The `web-time` shim provides a
//! `wasm-bindgen`-backed `Instant` on wasm32 (using
//! `performance.now()` in browsers and `process.hrtime()` in Node)
//! and re-exports `std::time::Instant` everywhere else.
//!
//! wick lib code uses `crate::time::Instant` instead of
//! `std::time::Instant` directly so the same code runs without
//! runtime panics on both native and browser targets. The `Wasm
//! Build` CI job validates compile-time only; this module is what
//! makes the runtime work too once Phase 3.2's `wick-wasm` crate
//! actually drives `Session::generate` from JS.

#[cfg(target_arch = "wasm32")]
pub use web_time::{Duration, Instant};

#[cfg(not(target_arch = "wasm32"))]
pub use std::time::{Duration, Instant};
