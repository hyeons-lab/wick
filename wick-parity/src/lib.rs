//! Parity harness — shared library surface.
//!
//! The harness runs the same prompt through every binding leg
//! (Rust direct, Rust through wick-ffi, Kotlin via JNA, Swift via
//! UniFFI, …) and byte-compares the resulting greedy-decoded token
//! streams. Divergence points to a marshalling bug at the FFI layer.
//!
//! This crate ships **two Rust legs** for the first PR: `run_rust`
//! (calls `wick::WickEngine` directly) and `run_ffi` (calls
//! `wick_ffi::WickEngine` through its Rust public surface). The two
//! paths produce identical output by construction — the FFI wrapper
//! is a thin Rust adapter over the same library — so this leg's
//! signal is "the construction-path adapters round-trip cleanly". The
//! real bug-detection bar arrives once a Kotlin/Swift leg lands and
//! traffic actually crosses the FFI boundary.
//!
//! Public surface kept narrow on purpose: a future Kotlin or Swift
//! leg only needs to produce a `Vec<u32>` from the same `RunArgs` —
//! the harness binary does the diffing.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Arguments shared across every leg. Constructed once by the harness
/// binary (or the integration test) and handed to each `run_*` fn.
///
/// Lifetime parameter keeps the borrow ergonomic for the binary —
/// every field comes from a `clap`-parsed string that outlives the
/// run. Future legs that spawn subprocesses will turn this into JSON
/// (see `RunArgsOwned` below) rather than holding borrows across the
/// boundary.
#[derive(Debug, Clone)]
pub struct RunArgs<'a> {
    /// LeapBundles bundle id, e.g. `"LFM2-350M-Extract-GGUF"`.
    pub bundle: &'a str,
    /// Manifest quant key, e.g. `"Q4_0"`.
    pub quant: &'a str,
    /// Prompt text. Tokenized via the model's own GGUF tokenizer.
    pub prompt: &'a str,
    /// Hard cap on emitted tokens. `0` is allowed — short-circuits
    /// to an empty token list.
    pub max_tokens: u32,
    /// Sampler seed. Greedy decoding (temperature=0) is
    /// seed-insensitive; field is plumbed for forward compat with
    /// sampling-based legs.
    pub seed: u64,
    /// Cache root for the underlying `BundleRepo`. Both legs share
    /// the same root so the model is downloaded at most once per
    /// harness run.
    pub cache_dir: &'a Path,
}

/// JSON-friendly mirror of [`RunArgs`]. Use this when serializing a
/// run record to disk or across a process boundary (subprocess legs).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunArgsOwned {
    pub bundle: String,
    pub quant: String,
    pub prompt: String,
    pub max_tokens: u32,
    pub seed: u64,
    pub cache_dir: PathBuf,
}

impl RunArgsOwned {
    pub fn as_ref(&self) -> RunArgs<'_> {
        RunArgs {
            bundle: &self.bundle,
            quant: &self.quant,
            prompt: &self.prompt,
            max_tokens: self.max_tokens,
            seed: self.seed,
            cache_dir: &self.cache_dir,
        }
    }
}

/// Output of a single leg run. Includes the token stream + enough
/// provenance to render a meaningful diff when two legs disagree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunOutput {
    /// Which leg produced this output. Free-form so non-Rust legs
    /// can label themselves (`"kotlin-jna"`, `"swift-uniffi"`, …).
    pub via: String,
    pub bundle: String,
    pub quant: String,
    pub prompt: String,
    pub max_tokens: u32,
    pub seed: u64,
    /// Greedy-decoded token IDs in emission order. Excludes prompt
    /// tokens and the `<bos>` (if added by the tokenizer).
    pub tokens: Vec<u32>,
}

/// Default cache root for harness runs. Resolved as
/// `$WICK_PARITY_CACHE_DIR` if set, else
/// `<workspace-root>/target/tmp/wick-parity-cache/`. Distinct from
/// `wick-test-models/` (used by `wick`'s integration tests) so
/// developers can wipe one cache without nuking the other.
///
/// Always creates the directory on return so callers don't have to.
pub fn default_cache_dir() -> Result<PathBuf> {
    if let Ok(override_path) = std::env::var("WICK_PARITY_CACHE_DIR") {
        let p = PathBuf::from(override_path);
        std::fs::create_dir_all(&p)
            .with_context(|| format!("create cache dir at {}", p.display()))?;
        return Ok(p);
    }
    // CARGO_MANIFEST_DIR points at this crate (wick-parity); workspace
    // root is one level up. Same convention as wick's
    // `tests/common/download.rs`.
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest
        .parent()
        .context("CARGO_MANIFEST_DIR has no parent")?
        .join("target")
        .join("tmp")
        .join("wick-parity-cache");
    std::fs::create_dir_all(&root)
        .with_context(|| format!("create cache dir at {}", root.display()))?;
    Ok(root)
}

/// Greedy-decoding generation options. Fixed across both Rust legs so
/// the only thing varying between runs is the construction path. CPU
/// backend, temperature 0, top_k 1 — fully deterministic regardless
/// of the underlying sampler's RNG state.
fn greedy_opts(max_tokens: u32) -> wick::GenerateOpts {
    wick::GenerateOpts {
        max_tokens,
        temperature: 0.0,
        top_p: 1.0,
        top_k: 1,
        repetition_penalty: 1.0,
        stop_tokens: Vec::new(),
        flush_every_tokens: 1,
        flush_every_ms: 0,
    }
}

fn engine_config_with_repo(cache_dir: &Path) -> wick::EngineConfig {
    wick::EngineConfig {
        // 256-token KV cap keeps the parity test cheap. Greedy decode
        // of 16 tokens off a short prompt fits in well under this.
        context_size: 256,
        backend: wick::BackendPreference::Cpu,
        bundle_repo: Some(wick::bundle::BundleRepo::new(cache_dir)),
    }
}

/// Run the prompt through `wick::WickEngine` directly. Reference leg —
/// every other leg's output is diffed against this.
pub fn run_rust(args: &RunArgs<'_>) -> Result<Vec<u32>> {
    let cfg = engine_config_with_repo(args.cache_dir);
    let engine = wick::WickEngine::from_bundle_id(args.bundle, args.quant, cfg)
        .with_context(|| format!("load bundle {}/{} (rust)", args.bundle, args.quant))?;

    let session_cfg = wick::SessionConfig {
        seed: Some(args.seed),
        ..Default::default()
    };
    let mut session = engine.new_session(session_cfg);

    if !args.prompt.is_empty() {
        session
            .append_text(args.prompt)
            .context("append prompt (rust)")?;
    }

    struct CollectSink(Vec<u32>);
    impl wick::ModalitySink for CollectSink {
        fn on_text_tokens(&mut self, tokens: &[u32]) {
            self.0.extend_from_slice(tokens);
        }
        fn on_done(&mut self, _reason: wick::FinishReason) {}
    }
    let mut sink = CollectSink(Vec::new());
    session
        .generate(&greedy_opts(args.max_tokens), &mut sink)
        .context("generate (rust)")?;
    Ok(sink.0)
}

/// Run the prompt through `wick_ffi::WickEngine`'s Rust public
/// surface. Same library underneath — this leg verifies the
/// `RecordType <-> wick::Type` adapters round-trip without dropping
/// or reordering anything that affects token output.
pub fn run_ffi(args: &RunArgs<'_>) -> Result<Vec<u32>> {
    let repo = wick_ffi::BundleRepo::new(args.cache_dir.to_string_lossy().into_owned());
    let cfg = wick_ffi::EngineConfig {
        context_size: 256,
        backend: wick_ffi::BackendPreference::Cpu,
        bundle_repo: Some(repo),
    };
    let engine =
        wick_ffi::WickEngine::from_bundle_id(args.bundle.to_string(), args.quant.to_string(), cfg)
            .with_context(|| format!("load bundle {}/{} (ffi)", args.bundle, args.quant))?;

    let session_cfg = wick_ffi::SessionConfig {
        seed: Some(args.seed),
        ..Default::default()
    };
    let session = engine.new_session(session_cfg);

    if !args.prompt.is_empty() {
        session
            .append_text(args.prompt.to_string())
            .context("append prompt (ffi)")?;
    }

    let opts = wick_ffi::GenerateOpts {
        max_tokens: args.max_tokens,
        temperature: 0.0,
        top_p: 1.0,
        top_k: 1,
        repetition_penalty: 1.0,
        stop_tokens: Vec::new(),
        flush_every_tokens: 1,
        flush_every_ms: 0,
    };
    let output = session.generate(opts).context("generate (ffi)")?;
    Ok(output.tokens)
}

/// First index where `a` and `b` differ, or `None` if they're equal.
/// Pure helper for the `check` subcommand and tests; doesn't allocate.
pub fn first_divergence(a: &[u32], b: &[u32]) -> Option<usize> {
    a.iter()
        .zip(b.iter())
        .position(|(x, y)| x != y)
        .or_else(|| {
            if a.len() != b.len() {
                Some(a.len().min(b.len()))
            } else {
                None
            }
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_divergence_finds_index_in_common_prefix() {
        assert_eq!(first_divergence(&[1, 2, 3, 4], &[1, 2, 9, 4]), Some(2));
    }

    #[test]
    fn first_divergence_finds_length_mismatch_after_common_prefix() {
        assert_eq!(first_divergence(&[1, 2, 3], &[1, 2, 3, 4]), Some(3));
        assert_eq!(first_divergence(&[1, 2, 3, 4], &[1, 2, 3]), Some(3));
    }

    #[test]
    fn first_divergence_returns_none_for_equal_streams() {
        assert_eq!(first_divergence(&[1, 2, 3], &[1, 2, 3]), None);
        assert_eq!(first_divergence(&[], &[]), None);
    }
}
