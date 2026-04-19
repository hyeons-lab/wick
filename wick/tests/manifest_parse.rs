//! Fixture-driven tests for `wick::manifest::Manifest`.
//!
//! The fixtures under `tests/fixtures/manifests/` are trimmed copies of
//! the canonical `LiquidAI/LeapBundles` manifest shapes at schema 1.0.0
//! (text, VL, audio) plus a synthesized "unknown inference type" case
//! for forward-compat testing. Updating them to track the upstream
//! schema should require zero parser changes for the shapes already
//! recognized — any new required field is a red flag.

use std::path::PathBuf;

use wick::manifest::{GenerationDefaults, InferenceType, Manifest};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/manifests")
        .join(name)
}

#[test]
fn parses_text_manifest() {
    let m = Manifest::from_file(&fixture("text.json")).unwrap();
    assert_eq!(m.inference_type, InferenceType::LlamaCppTextToText);
    assert_eq!(m.schema_version, "1.0.0");
    assert!(m.files.model.ends_with("LFM2-1.2B-Q4_0.gguf"));
    assert!(m.files.multimodal_projector.is_none());
    assert!(m.files.audio_decoder.is_none());
    assert!(m.files.audio_tokenizer.is_none());
    assert!(m.files.extras.is_empty());
    assert!(m.chat_template.is_some());
    assert!(m.is_loadable(), "text models are loadable in v1");

    match m.generation_defaults {
        GenerationDefaults::Text {
            temperature,
            min_p,
            repetition_penalty,
            ..
        } => {
            assert!((temperature.unwrap() - 0.3).abs() < 1e-5);
            assert!((min_p.unwrap() - 0.15).abs() < 1e-5);
            assert!((repetition_penalty.unwrap() - 1.05).abs() < 1e-5);
        }
        other => panic!("expected Text defaults, got {other:?}"),
    }
}

#[test]
fn parses_vl_manifest_with_aux_mmproj() {
    let m = Manifest::from_file(&fixture("vl.json")).unwrap();
    assert_eq!(m.inference_type, InferenceType::LlamaCppImageToText);
    assert!(m.files.model.ends_with("LFM2-VL-1.6B-F16.gguf"));
    assert!(
        m.files
            .multimodal_projector
            .as_deref()
            .unwrap()
            .ends_with("mmproj-LFM2-VL-1.6B-F16.gguf")
    );
    assert!(m.files.audio_decoder.is_none());
    assert!(m.files.audio_tokenizer.is_none());

    // VL is parsed OK but NOT loadable in v1.
    assert!(
        !m.is_loadable(),
        "VL should parse but fail at load time until the VL loader lands"
    );
}

#[test]
fn parses_audio_manifest_with_all_aux() {
    let m = Manifest::from_file(&fixture("audio.json")).unwrap();
    assert_eq!(m.inference_type, InferenceType::LlamaCppLfm2AudioV1);
    assert!(m.files.model.ends_with("LFM2-Audio-1.5B-F16.gguf"));
    assert!(m.files.multimodal_projector.is_some());
    assert!(
        m.files
            .audio_decoder
            .as_deref()
            .unwrap()
            .ends_with("audiodecoder-LFM2-Audio-1.5B-F16.gguf")
    );
    assert!(
        m.files
            .audio_tokenizer
            .as_deref()
            .unwrap()
            .ends_with(".safetensors")
    );
    assert!(
        m.is_loadable(),
        "audio models load via wick's audio pipeline in v1"
    );

    match m.generation_defaults {
        GenerationDefaults::Audio {
            number_of_decoding_threads,
        } => {
            assert_eq!(number_of_decoding_threads, Some(4));
        }
        other => panic!("expected Audio defaults, got {other:?}"),
    }
}

#[test]
fn files_in_order_matches_known_shape() {
    let audio = Manifest::from_file(&fixture("audio.json")).unwrap();
    let names: Vec<_> = audio.files_in_order().iter().map(|(k, _)| *k).collect();
    assert_eq!(
        names,
        [
            "model",
            "multimodal_projector",
            "audio_decoder",
            "audio_tokenizer"
        ]
    );

    let text = Manifest::from_file(&fixture("text.json")).unwrap();
    let names: Vec<_> = text.files_in_order().iter().map(|(k, _)| *k).collect();
    assert_eq!(names, ["model"]);
}

#[test]
fn unknown_inference_type_survives_round_trip() {
    let m = Manifest::from_file(&fixture("unknown.json")).unwrap();
    match &m.inference_type {
        InferenceType::Unknown(s) => assert_eq!(s, "llama.cpp/some-future-modality"),
        other => panic!("expected Unknown variant, got {other:?}"),
    }
    assert!(!m.is_loadable());
    // Novel aux file lands in extras instead of being dropped.
    assert!(
        m.files.extras.contains_key("novel_aux_component"),
        "unknown aux keys must land in `extras`; got {:?}",
        m.files.extras
    );
    // `generation_time_parameters` with an unrecognized shape falls into
    // the `Other` bucket rather than crashing the parser.
    match &m.generation_defaults {
        GenerationDefaults::Other { raw } => {
            assert!(raw.get("mystery_knob").is_some());
        }
        other => panic!("expected Other defaults for unknown inference type, got {other:?}"),
    }
}

#[test]
fn raw_value_preserves_round_trip_fidelity() {
    // Any field we don't model is still accessible via `Manifest::raw`.
    let m = Manifest::from_file(&fixture("unknown.json")).unwrap();
    let raw_str = serde_json::to_string(&m.raw).unwrap();
    assert!(
        raw_str.contains("llama.cpp/some-future-modality"),
        "raw should contain the verbatim inference_type"
    );
    assert!(raw_str.contains("mystery_knob"));
}

#[test]
fn synthetic_text_from_gguf_path() {
    let p = PathBuf::from("/tmp/fake-model.gguf");
    let m = Manifest::synthetic_text(&p);
    assert_eq!(m.inference_type, InferenceType::LlamaCppTextToText);
    assert_eq!(m.files.model, "/tmp/fake-model.gguf");
    assert!(m.is_loadable());
}

/// Regression test for PR #30 review: when `inference_type` is `Unknown`
/// and `generation_time_parameters` is missing, the parser must return
/// `GenerationDefaults::Other { raw: Value::Null }` rather than an empty
/// `Text` variant (which would imply text-sampling defaults for a model
/// whose inference type we can't identify).
#[test]
fn unknown_inference_type_without_gen_params_returns_other_not_text() {
    let bytes = br#"{
        "inference_type": "llama.cpp/some-future-modality",
        "schema_version": "1.0.0",
        "load_time_parameters": { "model": "future.gguf" }
    }"#;
    let m = wick::manifest::Manifest::from_bytes(bytes).unwrap();
    assert!(matches!(
        m.inference_type,
        InferenceType::Unknown(ref s) if s == "llama.cpp/some-future-modality"
    ));
    match &m.generation_defaults {
        GenerationDefaults::Other { raw } => assert!(raw.is_null()),
        other => panic!(
            "expected Other with null raw for unknown inference + missing params, got {other:?}"
        ),
    }
}
