#![cfg(not(target_arch = "wasm32"))]

use std::path::PathBuf;

use cera_ffi::{FfiWhisperModel, FfiWhisperTranscribeOpts, whisper_default_transcribe_opts};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn find_whisper_model() -> Option<PathBuf> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        manifest_dir.join("../models/whisper.gguf"),
        manifest_dir.join("../models/whisper-tiny.en-Q5_K_M.gguf"),
        manifest_dir.join("../models/whisper-base.en-Q5_K_M.gguf"),
        PathBuf::from("models/whisper.gguf"),
    ];
    candidates.into_iter().find(|p| p.exists())
}

#[test]
fn test_ffi_whisper_opts_and_defaults() {
    let def = whisper_default_transcribe_opts();
    assert_eq!(def.language, None);
    assert!(!def.translate);
    assert!(!def.timestamps);
    assert_eq!(def.max_tokens, Some(448));
    assert_eq!(def.temperature, Some(0.0));

    let custom = FfiWhisperTranscribeOpts {
        language: Some("fr".to_string()),
        translate: true,
        timestamps: true,
        max_tokens: Some(128),
        temperature: Some(0.5),
    };

    let core: cera::WhisperTranscribeOpts = custom.clone().into();
    assert_eq!(core.language.as_deref(), Some("fr"));
    assert!(core.translate);
    assert!(core.timestamps);
    assert_eq!(core.max_tokens, 128);
    assert_eq!(core.temperature, 0.5);

    let roundtrip: FfiWhisperTranscribeOpts = core.into();
    assert_eq!(roundtrip, custom);
}

#[test]
fn test_ffi_whisper_error_handling_no_panic() {
    // 1. Loading non-existent file returns error without panicking
    let err_file = FfiWhisperModel::from_file("/path/to/nonexistent/whisper.gguf".to_string());
    assert!(err_file.is_err());

    // 2. Loading garbage bytes returns error without panicking
    let err_bytes = FfiWhisperModel::from_bytes(vec![0x47, 0x47, 0x55, 0x46, 0x00, 0x00]);
    assert!(err_bytes.is_err());
}

#[test]
fn test_ffi_whisper_model_transcribe_when_available() -> Result<()> {
    let Some(model_path) = find_whisper_model() else {
        eprintln!("Skipping live model test: whisper GGUF not found");
        return Ok(());
    };

    let model_str = model_path.to_str().unwrap().to_string();
    let whisper = FfiWhisperModel::from_file(model_str)?;

    let languages = whisper.languages();
    assert_eq!(languages.len(), 100);
    assert_eq!(languages[0], "en");
    assert_eq!(languages[1], "zh");

    // Empty audio transcription must succeed with empty text
    let empty_text = whisper.transcribe(Vec::new(), None)?;
    assert_eq!(empty_text, "");

    Ok(())
}
