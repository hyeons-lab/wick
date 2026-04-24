//! `wick-ffi` — foreign-language bindings to [`wick`] via UniFFI.
//!
//! This crate exposes a subset of the `wick` inference engine to
//! Kotlin, Swift, Python, and any other language UniFFI supports. It
//! is structured around the **proc-macro** path (rather than a UDL
//! file) so the Rust types we expose are the source of truth and
//! annotations stay colocated with the code they describe.
//!
//! ## Current surface
//!
//! - [`WickEngine`] — load a model from a local path; introspect
//!   metadata + capabilities.
//! - [`EngineConfig`] — construction-time config (context size,
//!   backend preference).
//! - [`BackendPreference`] — CPU / GPU / Metal / Auto selector.
//! - [`ModelMetadata`] — architecture, vocab size, max context, etc.
//! - [`ModalityCapabilities`] — what the loaded model accepts / emits.
//! - [`FfiError`] — minimal error shell (string-based). The typed
//!   variants (`ContextOverflow { max_seq_len, by }` etc.) will land
//!   as additional variants on this enum in a later PR; for now the
//!   string form preserves the message without locking in a premature
//!   schema.
//!
//! ## Not exposed yet
//!
//! Future PRs grow the surface per the roadmap in
//! `wick-ffi/README.md`: `Session` + `generate` (sync → async → with
//! callbacks), `BundleRepo` + remote URL loading (gated on the
//! `remote` feature), binding generation for Kotlin / Swift / Python,
//! and typed error variants.
//!
//! ## Design notes
//!
//! - **Wrapper types, not annotations on `wick` core.** Every
//!   UniFFI-exposed type is a wrapper defined in this crate with
//!   `From` conversions to/from the `wick` equivalent. The core crate
//!   stays UniFFI-agnostic.
//! - **`u64` on the wire, `usize` internally.** UniFFI records can't
//!   marshal `usize` (pointer-sized). Convert at the boundary.

use std::sync::Arc;

uniffi::setup_scaffolding!();

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Minimal error type for v1 of the FFI surface. Wraps a human-readable
/// message; the typed variants (`ContextOverflow`, `UnsupportedModality`,
/// etc.) will land as additional variants on this enum in a later PR
/// so callers can pattern-match on error class rather than string-sniff.
#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum FfiError {
    #[error("{message}")]
    Backend { message: String },
}

impl From<wick::WickError> for FfiError {
    fn from(e: wick::WickError) -> Self {
        FfiError::Backend {
            message: e.to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Config + enums
// ---------------------------------------------------------------------------

/// Compute-backend selector. Mirrors [`wick::BackendPreference`];
/// kept as a separate type so the `wick` crate doesn't carry UniFFI
/// annotations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum BackendPreference {
    /// Probe Metal → GPU → CPU at load time.
    Auto,
    Cpu,
    /// `wgpu` (Vulkan / Metal / DX12). Requires the `gpu` feature.
    Gpu,
    /// Native Metal. Requires the `metal` feature + macOS.
    Metal,
}

impl From<BackendPreference> for wick::BackendPreference {
    fn from(b: BackendPreference) -> Self {
        match b {
            BackendPreference::Auto => wick::BackendPreference::Auto,
            BackendPreference::Cpu => wick::BackendPreference::Cpu,
            BackendPreference::Gpu => wick::BackendPreference::Gpu,
            BackendPreference::Metal => wick::BackendPreference::Metal,
        }
    }
}

impl From<wick::BackendPreference> for BackendPreference {
    fn from(b: wick::BackendPreference) -> Self {
        match b {
            wick::BackendPreference::Auto => BackendPreference::Auto,
            wick::BackendPreference::Cpu => BackendPreference::Cpu,
            wick::BackendPreference::Gpu => BackendPreference::Gpu,
            wick::BackendPreference::Metal => BackendPreference::Metal,
        }
    }
}

/// Per-engine configuration at load time. Mirrors [`wick::EngineConfig`]
/// with `u64` fields (UniFFI doesn't marshal `usize`).
#[derive(Debug, Clone, uniffi::Record)]
pub struct EngineConfig {
    /// KV-cache capacity in tokens. Capped by the model's own
    /// `max_seq_len`. Pass `0` to use the model's full declared
    /// `max_seq_len` (translated to `usize::MAX` internally, then
    /// capped by the loader).
    pub context_size: u64,
    pub backend: BackendPreference,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            context_size: 4096,
            backend: BackendPreference::Auto,
        }
    }
}

impl TryFrom<EngineConfig> for wick::EngineConfig {
    type Error = FfiError;

    fn try_from(c: EngineConfig) -> Result<Self, FfiError> {
        // Checked `u64 → usize` conversion. On 32-bit targets (Android
        // armv7 is still a supported ABI) `u64` can exceed `usize::MAX`
        // and a bare `as usize` would silently truncate — producing a
        // much smaller KV cache than the caller intended. Surface the
        // overflow as a typed error instead.
        let context_size = if c.context_size == 0 {
            // Sentinel for "use model default" — wick caps at model.max_seq_len.
            usize::MAX
        } else {
            usize::try_from(c.context_size).map_err(|_| FfiError::Backend {
                message: format!(
                    "context_size {} exceeds usize::MAX on this target",
                    c.context_size
                ),
            })?
        };
        // `..Default::default()` is intentional forward-compat — under
        // the `remote` feature `wick::EngineConfig` gains a
        // `bundle_repo: Option<BundleRepo>` field, and the spread
        // supplies `None` without this file needing to know. Without
        // the feature clippy sees two specified fields + a redundant
        // update; the #[allow] keeps the spread in place so a future
        // remote-feature activation doesn't require a diff here.
        #[allow(clippy::needless_update)]
        Ok(wick::EngineConfig {
            context_size,
            backend: c.backend.into(),
            ..Default::default()
        })
    }
}

// ---------------------------------------------------------------------------
// Metadata + capabilities
// ---------------------------------------------------------------------------

/// Short summary of a loaded model. Mirrors [`wick::ModelMetadata`].
#[derive(Debug, Clone, uniffi::Record)]
pub struct ModelMetadata {
    pub architecture: String,
    pub max_seq_len: u32,
    pub vocab_size: u32,
    pub has_chat_template: bool,
    pub quantization: String,
    /// Mirror of GGUF `tokenizer.ggml.add_bos_token`. Consumers that
    /// want to insert a BOS at the head of a raw prompt should honor it.
    pub add_bos_token: bool,
}

impl From<&wick::ModelMetadata> for ModelMetadata {
    fn from(m: &wick::ModelMetadata) -> Self {
        ModelMetadata {
            architecture: m.architecture.clone(),
            max_seq_len: m.max_seq_len,
            vocab_size: m.vocab_size,
            has_chat_template: m.has_chat_template,
            quantization: m.quantization.clone(),
            add_bos_token: m.add_bos_token,
        }
    }
}

/// Modality support flags for a loaded model. Mirrors
/// [`wick::ModalityCapabilities`].
#[derive(Debug, Clone, Copy, uniffi::Record)]
pub struct ModalityCapabilities {
    pub text_in: bool,
    pub text_out: bool,
    pub image_in: bool,
    pub audio_in: bool,
    pub audio_out: bool,
}

impl From<wick::ModalityCapabilities> for ModalityCapabilities {
    fn from(c: wick::ModalityCapabilities) -> Self {
        ModalityCapabilities {
            text_in: c.text_in,
            text_out: c.text_out,
            image_in: c.image_in,
            audio_in: c.audio_in,
            audio_out: c.audio_out,
        }
    }
}

// ---------------------------------------------------------------------------
// WickEngine
// ---------------------------------------------------------------------------

/// Owning handle to a loaded model. Mirrors [`wick::WickEngine`];
/// `#[uniffi::Object]` requires `Arc<Self>` wrapping which matches how
/// the underlying engine is already used internally.
#[derive(uniffi::Object)]
pub struct WickEngine {
    inner: wick::WickEngine,
}

#[uniffi::export]
impl WickEngine {
    /// Load a model from a local filesystem path. Accepts the same
    /// inputs as the native [`wick::WickEngine::from_path`]:
    /// a bare `.gguf`, a LeapBundles `.json` manifest, or a directory
    /// containing exactly one `.json` manifest.
    ///
    /// Remote URLs in manifests are rejected in this PR — `BundleRepo`
    /// wiring lands when the `remote` feature activates here.
    #[uniffi::constructor]
    pub fn from_path(path: String, config: EngineConfig) -> Result<Arc<Self>, FfiError> {
        let inner = wick::WickEngine::from_path(&path, config.try_into()?)?;
        Ok(Arc::new(Self { inner }))
    }

    /// Short summary of the loaded model (architecture, vocab size,
    /// max context, etc.). Returns a `Clone` of the stored metadata.
    pub fn metadata(&self) -> ModelMetadata {
        ModelMetadata::from(self.inner.metadata())
    }

    /// What this model accepts as input / emits as output. Derived at
    /// load time from the manifest's `inference_type`.
    pub fn capabilities(&self) -> ModalityCapabilities {
        self.inner.capabilities().into()
    }
}

// ---------------------------------------------------------------------------
// Smoke test (from PR 1)
// ---------------------------------------------------------------------------

/// Version string of the `wick-ffi` crate. Useful as a smoke test
/// from the foreign-language side — if this is callable, the binding
/// pipeline works end-to-end.
#[uniffi::export]
pub fn wick_ffi_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_non_empty() {
        // Smoke test: proves the proc-macro expanded and the export is
        // callable. No shape check on the string — SemVer allows
        // pre-release + build-metadata suffixes (`0.1.0-alpha.1+deadbe`)
        // that a strict `x.y.z` split would reject.
        let v = wick_ffi_version();
        assert!(!v.is_empty(), "version string must not be empty");
    }

    #[test]
    fn engine_config_default_roundtrips_to_wick() {
        let ffi = EngineConfig::default();
        let core: wick::EngineConfig = ffi.try_into().unwrap();
        assert_eq!(core.context_size, 4096);
        assert_eq!(core.backend, wick::BackendPreference::Auto);
    }

    #[test]
    fn engine_config_zero_context_size_means_max() {
        // `0` on the wire is the FFI's "use model default" signal;
        // translate to `usize::MAX` so wick caps at model.max_seq_len.
        let ffi = EngineConfig {
            context_size: 0,
            backend: BackendPreference::Cpu,
        };
        let core: wick::EngineConfig = ffi.try_into().unwrap();
        assert_eq!(core.context_size, usize::MAX);
    }

    #[test]
    fn engine_config_oversize_context_errors_on_32bit_targets() {
        // On 32-bit targets, `u64::MAX` exceeds `usize::MAX` and the
        // checked conversion must surface an error rather than
        // silently truncating. On 64-bit targets `usize::MAX ==
        // u64::MAX`, so the conversion succeeds — skip the assert
        // there. This test proves the error path compiles + is
        // reachable under the narrow condition where it matters.
        let ffi = EngineConfig {
            context_size: u64::MAX,
            backend: BackendPreference::Cpu,
        };
        let result: Result<wick::EngineConfig, FfiError> = ffi.try_into();
        #[cfg(target_pointer_width = "32")]
        {
            let err = result.expect_err("u64::MAX must fail on 32-bit");
            match err {
                FfiError::Backend { message } => {
                    assert!(
                        message.contains("exceeds usize::MAX"),
                        "unexpected: {message}"
                    );
                }
            }
        }
        #[cfg(target_pointer_width = "64")]
        {
            // On 64-bit `u64::MAX == usize::MAX`; the sentinel check
            // has already rejected `0`, so `u64::MAX` converts cleanly.
            let core = result.expect("u64::MAX fits usize::MAX on 64-bit");
            assert_eq!(core.context_size, usize::MAX);
        }
    }

    #[test]
    fn backend_preference_roundtrips() {
        for ffi in [
            BackendPreference::Auto,
            BackendPreference::Cpu,
            BackendPreference::Gpu,
            BackendPreference::Metal,
        ] {
            let core: wick::BackendPreference = ffi.into();
            let back: BackendPreference = core.into();
            assert_eq!(ffi, back, "{ffi:?} didn't round-trip");
        }
    }

    #[test]
    fn ffi_error_wraps_wick_error_message() {
        let we = wick::WickError::EmptyInput;
        let expected = we.to_string();
        let fe: FfiError = we.into();
        match fe {
            FfiError::Backend { message } => assert_eq!(message, expected),
        }
    }
}
