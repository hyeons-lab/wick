//! Integration coverage for `WickEngine`'s loader dispatch.
//!
//! The tests here focus on **control flow** — manifest shape → the
//! right `WickError` variant or the right load path — not on actually
//! loading a model. Tests that need a real GGUF are marked `#[ignore]`
//! and gated on the `WICK_TEST_MODEL` env var so CI and offline
//! developers aren't blocked. This mirrors `session_chain.rs`.

use std::path::{Path, PathBuf};

use wick::manifest::InferenceType;
use wick::{BackendPreference, EngineConfig, ModelFiles, WickEngine, WickError};

// ---------------------------------------------------------------------------
// Test-only helpers (no real model required)
// ---------------------------------------------------------------------------

fn write_manifest(dir: &Path, name: &str, json: &str) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, json).unwrap();
    path
}

fn cpu_cfg() -> EngineConfig {
    EngineConfig {
        context_size: 256,
        backend: BackendPreference::Cpu,
        ..Default::default()
    }
}

/// `WickEngine` can't derive `Debug` (holds `Box<dyn Model>`). Shim
/// around `.expect_err` that does the same job: pattern-match the
/// `Err` and panic with `msg` otherwise.
#[track_caller]
fn expect_err(r: Result<WickEngine, WickError>, msg: &str) -> WickError {
    match r {
        Ok(_) => panic!("{msg}"),
        Err(e) => e,
    }
}

// ---------------------------------------------------------------------------
// Control-flow tests — no real GGUF needed
// ---------------------------------------------------------------------------

#[test]
fn from_path_rejects_remote_model_url() {
    let dir = tempfile::tempdir().unwrap();
    let json = r#"{
        "inference_type": "llama.cpp/text-to-text",
        "schema_version": "1.0.0",
        "load_time_parameters": {
            "model": "https://huggingface.co/LiquidAI/LFM2-1.2B-GGUF/resolve/main/LFM2-1.2B-Q4_0.gguf"
        }
    }"#;
    let manifest = write_manifest(dir.path(), "remote.json", json);

    let err = expect_err(
        WickEngine::from_path(&manifest, cpu_cfg()),
        "remote URL must be rejected without a BundleRepo",
    );
    let msg = format!("{err}");
    // Error must name the problem (remote URL) and steer the caller at
    // the fix. Without `remote` feature → point at enabling it; with it
    // on → point at the `bundle_repo` config field.
    assert!(
        msg.contains("remote URL"),
        "error should name the remote URL; got `{msg}`"
    );
    #[cfg(feature = "remote")]
    assert!(
        msg.contains("bundle_repo"),
        "error under `remote` feature should name the config field; got `{msg}`"
    );
    #[cfg(not(feature = "remote"))]
    assert!(
        msg.contains("`remote` feature"),
        "error without `remote` feature should point at enabling it; got `{msg}`"
    );
    // The typed variant: Backend(_) with the remote-URL message lives
    // behind Backend; not a dedicated variant.
    assert!(matches!(err, WickError::Backend(_)));
}

#[test]
fn from_path_vl_manifest_errors_with_unsupported_inference_type() {
    let dir = tempfile::tempdir().unwrap();
    let gguf = dir.path().join("placeholder.gguf");
    std::fs::write(&gguf, b"placeholder").unwrap();
    let mmproj = dir.path().join("placeholder-mmproj.gguf");
    std::fs::write(&mmproj, b"placeholder").unwrap();

    let json = format!(
        r#"{{
            "inference_type": "llama.cpp/image-to-text",
            "schema_version": "1.0.0",
            "load_time_parameters": {{
                "model": {model:?},
                "multimodal_projector": {mmproj:?}
            }}
        }}"#,
        model = gguf.to_string_lossy(),
        mmproj = mmproj.to_string_lossy(),
    );
    let manifest = write_manifest(dir.path(), "vl.json", &json);

    let err = expect_err(
        WickEngine::from_path(&manifest, cpu_cfg()),
        "VL models must error until the loader lands",
    );
    match err {
        WickError::UnsupportedInferenceType(s) => {
            assert_eq!(s, "llama.cpp/image-to-text", "error carries the raw type");
        }
        other => panic!("expected UnsupportedInferenceType, got {other:?}"),
    }
}

#[test]
fn from_path_unknown_inference_type_errors_with_raw_string() {
    let dir = tempfile::tempdir().unwrap();
    let gguf = dir.path().join("placeholder.gguf");
    std::fs::write(&gguf, b"placeholder").unwrap();

    let json = format!(
        r#"{{
            "inference_type": "llama.cpp/some-future-modality",
            "schema_version": "1.0.0",
            "load_time_parameters": {{ "model": {model:?} }}
        }}"#,
        model = gguf.to_string_lossy(),
    );
    let manifest = write_manifest(dir.path(), "unknown.json", &json);

    let err = expect_err(
        WickEngine::from_path(&manifest, cpu_cfg()),
        "unknown inference types must error",
    );
    match err {
        WickError::UnsupportedInferenceType(s) => {
            assert_eq!(s, "llama.cpp/some-future-modality");
        }
        other => panic!("expected UnsupportedInferenceType, got {other:?}"),
    }
}

#[test]
fn from_path_directory_with_zero_manifests_errors() {
    let dir = tempfile::tempdir().unwrap();
    let err = expect_err(
        WickEngine::from_path(dir.path(), cpu_cfg()),
        "empty directory must error",
    );
    let msg = format!("{err}");
    assert!(
        msg.contains("no .json manifest"),
        "error should mention missing manifest; got `{msg}`"
    );
}

#[test]
fn from_path_directory_with_multiple_manifests_errors() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("a.json"), b"{}").unwrap();
    std::fs::write(dir.path().join("b.json"), b"{}").unwrap();

    let err = expect_err(
        WickEngine::from_path(dir.path(), cpu_cfg()),
        "ambiguous directory must error",
    );
    let msg = format!("{err}");
    assert!(
        msg.contains("2 .json manifests") && msg.contains("a.json") && msg.contains("b.json"),
        "error should list the conflicting manifests; got `{msg}`"
    );
}

#[test]
fn from_path_directory_dispatches_single_manifest() {
    // One .json manifest with a remote URL → dispatch succeeds, then
    // the URL check fires. Proves the directory path reaches manifest
    // parsing + dispatch without needing a real GGUF.
    let dir = tempfile::tempdir().unwrap();
    let json = r#"{
        "inference_type": "llama.cpp/text-to-text",
        "schema_version": "1.0.0",
        "load_time_parameters": {
            "model": "https://hf.co/x/y.gguf"
        }
    }"#;
    write_manifest(dir.path(), "only.json", json);

    let err = expect_err(
        WickEngine::from_path(dir.path(), cpu_cfg()),
        "directory-with-remote-manifest must still reject the remote URL",
    );
    // After PR A lands BundleRepo, the error still fires when no
    // `bundle_repo` is configured; the message no longer name-checks
    // "Phase 1.6" (the feature shipped), but it names "remote URL" and
    // the fix path. Assert the essentials.
    let msg = format!("{err}");
    assert!(
        msg.contains("remote URL"),
        "directory-with-remote error should name the remote URL; got `{msg}`"
    );
}

#[test]
fn from_path_rejects_arbitrary_extension() {
    let dir = tempfile::tempdir().unwrap();
    // File exists and has an extension that's neither .gguf nor .json;
    // must error rather than being silently fed to the GGUF parser.
    let odd = dir.path().join("model.weirdext");
    std::fs::write(&odd, b"not a gguf").unwrap();
    let err = expect_err(
        WickEngine::from_path(&odd, cpu_cfg()),
        "unknown extension must error",
    );
    let msg = format!("{err}");
    assert!(
        msg.contains("don't know how to load"),
        "got `{msg}` from {odd:?}"
    );
}

// Note on bare-`.gguf` VL detection: `auto_detect_inference_type`
// dispatches on `general.architecture`. Current LFM2 VL bundles ship the
// primary GGUF with `architecture = "lfm2"` (the VL-ness lives in the
// mmproj aux + manifest), so bare-loading that file yields a text
// session — which is the correct text-only subset. The
// `LlamaCppImageToText` arm in `from_path(.gguf)` is defense-in-depth
// for any future GGUF that does declare a VL-specific arch (`lfm2vl`
// etc.); no current fixture exercises it.

// ---------------------------------------------------------------------------
// Smoke tests that need a real model on disk. Gated.
// ---------------------------------------------------------------------------

fn find_model() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("WICK_TEST_MODEL") {
        let path = PathBuf::from(p);
        if path.exists() {
            return Some(path);
        }
    }
    None
}

#[test]
#[ignore = "requires WICK_TEST_MODEL pointing at a local .gguf"]
fn from_path_on_gguf_loads_text_engine() {
    let Some(model_path) = find_model() else {
        return;
    };
    let engine = WickEngine::from_path(&model_path, cpu_cfg()).expect("load from .gguf");
    assert_eq!(
        engine.manifest().inference_type,
        InferenceType::LlamaCppTextToText,
        "bare .gguf should synthesize a text manifest"
    );
    let md = engine.metadata();
    assert!(md.max_seq_len > 0);
    assert!(md.vocab_size > 0);
    assert!(!md.architecture.is_empty());
}

#[test]
#[ignore = "requires WICK_TEST_MODEL pointing at a local .gguf"]
fn from_files_equivalent_to_from_path_for_text_gguf() {
    let Some(model_path) = find_model() else {
        return;
    };

    let via_path = WickEngine::from_path(&model_path, cpu_cfg()).expect("from_path load");
    let via_files =
        WickEngine::from_files(ModelFiles::text(&model_path), cpu_cfg()).expect("from_files load");

    let a = via_path.metadata();
    let b = via_files.metadata();
    assert_eq!(a.architecture, b.architecture);
    assert_eq!(a.max_seq_len, b.max_seq_len);
    assert_eq!(a.vocab_size, b.vocab_size);
}

#[test]
#[ignore = "requires WICK_TEST_MODEL pointing at a local .gguf"]
fn engine_hands_out_session_that_can_append() {
    let Some(model_path) = find_model() else {
        return;
    };
    let engine = WickEngine::from_path(&model_path, cpu_cfg()).expect("load");
    let mut session = engine.new_session(Default::default());
    // Just one BOS-ish token to confirm the session shape works.
    let toks = engine.tokenizer().encode("hi");
    assert!(!toks.is_empty(), "tokenizer produced tokens");
    // Keep under the configured context window.
    session.append_tokens(&toks).expect("append_tokens");
    assert_eq!(session.position(), toks.len() as u32);
}
