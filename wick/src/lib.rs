#![cfg_attr(
    target_arch = "aarch64",
    feature(stdarch_neon_dotprod, stdarch_aarch64_prefetch)
)]

pub mod audio_engine;
pub mod backend;
pub mod engine;
/// Auto-generated FlatBuffers code for KV cache serialization.
/// Regenerate with: `flatc --rust -o src/generated schema/kv_cache.fbs`
#[allow(warnings)]
mod generated {
    include!("generated/kv_cache_generated.rs");
}
pub mod gguf;
pub mod kv_cache;
pub mod model;
pub mod quant;
pub mod sampler;
pub mod tensor;
pub mod tokenizer;
