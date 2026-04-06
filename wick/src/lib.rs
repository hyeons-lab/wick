#![cfg_attr(
    target_arch = "aarch64",
    feature(stdarch_neon_dotprod, stdarch_aarch64_prefetch)
)]

pub mod audio_engine;
pub mod backend;
pub mod engine;
pub mod gguf;
pub mod kv_cache;
pub mod model;
pub mod quant;
pub mod sampler;
pub mod tensor;
pub mod tokenizer;
