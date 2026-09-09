// `stdarch_neon_dotprod` stabilized in 1.99.0-nightly, so it's no longer gated
// here (keeping it would trip the `stable_features` lint). `stdarch_neon_i8mm`
// and `stdarch_aarch64_prefetch` are still unstable — remove them from this
// list as they stabilize.
#![cfg_attr(
    target_arch = "aarch64",
    feature(stdarch_aarch64_prefetch, stdarch_neon_i8mm)
)]

/// Crate version, sourced from `Cargo.toml` at compile time. Useful
/// for FFI / wrapper crates that want to surface the core lib version
/// to their consumers (e.g. `cera-wasm::ceraVersion()`) without
/// re-reading the manifest themselves.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Short git SHA of this build, embedded by `build.rs`. Best-effort: `"unknown"`
/// when git was unavailable at build time (a packaged source build). Can be
/// pinned via the `CERA_GIT_SHA` build-env override.
pub const GIT_SHA: &str = env!("CERA_GIT_SHA");

/// Build provenance for telemetry — `"<version>+<git-sha>"`, e.g.
/// `"0.4.0+1a2b3c4d5e6f"`. This is the analog of the llama.cpp build commit that
/// a benchmark harness records alongside results to identify exactly which
/// engine build produced them.
pub fn build_info() -> String {
    format!("{VERSION}+{GIT_SHA}")
}

pub mod audio_engine;
pub mod backend;
pub mod bundle;
pub mod classifier;
pub mod convert;
pub mod engine;
/// Auto-generated FlatBuffers code for KV cache serialization.
/// Regenerate with: `flatc --rust -o src/generated schema/kv_cache.fbs`
#[allow(warnings)]
mod generated {
    include!("generated/kv_cache_generated.rs");
}
pub mod gguf;
pub mod grammar;
pub mod hotword;
pub mod kv_cache;
pub mod lora;
pub mod manifest;
pub mod model;
pub mod par;
pub mod quant;
pub mod sampler;
pub mod session;
pub mod spec;
pub mod sysmem;
pub mod tensor;
pub mod time;
pub mod tokenizer;
pub mod tools;
pub mod turboquant;
pub mod vad;

// Canonical public re-exports for the stateful API. Consumers should
// `use cera::{Session, ModalitySink, ...}` rather than reaching into
// `cera::session::*`.
pub use backend::cpu_features::{CpuFeatures, CpuTier, cpu_features, cpu_tier};
pub use classifier::{
    BioesPrefix, EntitySpan, detect_pii, extract_spans, parse_bioes, viterbi_decode,
};
pub use engine::{
    BackendPreference, CeraEngine, EngineConfig, ModelBytes, ModelFiles, ModelMetadata,
};
pub use hotword::{
    HotwordConfig, HotwordDetector, HotwordEvent, HotwordIterator, HotwordScore, LogMelFrontEnd,
};
pub use model::whisper::{
    Conv1dWeights, WHISPER_CATALOG, WhisperConfig, WhisperDecoderBlockWeights,
    WhisperDecoderWeights, WhisperEncoderBlockWeights, WhisperEncoderWeights, WhisperModel,
    WhisperModelCatalogEntry, WhisperSpecialTokens, WhisperTranscribeOpts, WhisperWeights,
    find_whisper_catalog_entry, is_whisper_gguf, transcribe_pcm, transcribe_pcm_with_tokens,
};
pub use session::{
    CeraError, FinishReason, GenerateOpts, GenerateSummary, ModalityCapabilities, ModalitySink,
    Session, SessionConfig, SpecDecode,
};
pub use sysmem::{available_memory_bytes, fits_in_available_memory};
pub use vad::{SileroVad, SpeechTimestamp, VadConfig, VadEvent, VadIterator, VadSampleRate};

#[cfg(test)]
mod build_info_tests {
    use super::*;

    #[test]
    fn build_info_is_version_plus_sha() {
        let info = build_info();
        assert_eq!(info, format!("{VERSION}+{GIT_SHA}"));
        // version prefix, single '+' separator, non-empty sha segment.
        let (ver, sha) = info.split_once('+').expect("build_info has a '+'");
        assert_eq!(ver, VERSION);
        assert!(!sha.is_empty());
    }
}
