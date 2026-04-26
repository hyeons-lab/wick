#![cfg_attr(
    target_arch = "aarch64",
    feature(stdarch_neon_dotprod, stdarch_aarch64_prefetch)
)]

/// Crate version, sourced from `Cargo.toml` at compile time. Useful
/// for FFI / wrapper crates that want to surface the core lib version
/// to their consumers (e.g. `wick-wasm::wickVersion()`) without
/// re-reading the manifest themselves.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod audio_engine;
pub mod backend;
#[cfg(feature = "remote")]
pub mod bundle;
pub mod engine;
/// Auto-generated FlatBuffers code for KV cache serialization.
/// Regenerate with: `flatc --rust -o src/generated schema/kv_cache.fbs`
#[allow(warnings)]
mod generated {
    include!("generated/kv_cache_generated.rs");
}
pub mod gguf;
pub mod kv_cache;
pub mod manifest;
pub mod model;
pub mod par;
pub mod quant;
pub mod safetensors;
pub mod sampler;
pub mod session;
pub mod tensor;
pub mod time;
pub mod tokenizer;
pub mod turboquant;

// Canonical public re-exports for the stateful API. Consumers should
// `use wick::{Session, ModalitySink, ...}` rather than reaching into
// `wick::session::*`.
pub use engine::{BackendPreference, EngineConfig, ModelFiles, ModelMetadata, WickEngine};
pub use session::{
    FinishReason, GenerateOpts, GenerateSummary, ImageFormat, ModalityCapabilities, ModalitySink,
    Session, SessionConfig, WickError,
};
