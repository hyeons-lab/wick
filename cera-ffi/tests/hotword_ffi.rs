#![cfg(not(target_arch = "wasm32"))]

use std::path::PathBuf;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
use cera_ffi::{FfiHotwordConfig, FfiHotwordDetector, FfiHotwordIterator};

fn find_hotword_model() -> Option<PathBuf> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        manifest_dir.join("../models/hey_liquid.gguf"),
        manifest_dir.join("models/hey_liquid.gguf"),
        PathBuf::from("models/hey_liquid.gguf"),
    ];
    candidates.into_iter().find(|p| p.exists())
}

#[test]
fn test_ffi_hotword_detector_and_iterator() -> Result<()> {
    let Some(model_path) = find_hotword_model() else {
        eprintln!("Skipping test: models/hey_liquid.gguf not found");
        return Ok(());
    };

    let model_str = model_path.to_str().unwrap().to_string();

    // 1. Test FfiHotwordDetector
    let detector = FfiHotwordDetector::from_file(model_str.clone())?;
    let keywords = detector.keywords()?;
    assert_eq!(keywords, vec!["Hey Liquid"]);

    let config = detector.default_config()?;
    assert_eq!(config.threshold, 0.75);
    assert_eq!(config.cooldown_ms, 2000);
    assert_eq!(config.step_ms, 80);
    assert_eq!(config.window_ms, 1200);

    // 2. Synthesize 19,200 samples of 16 kHz audio
    let mut synthetic_audio = vec![0.0f32; 19200];
    for (i, sample) in synthetic_audio.iter_mut().enumerate() {
        let t = i as f64 / 16000.0;
        let s = 0.3 * (2.0 * std::f64::consts::PI * 440.0 * t).sin()
            + 0.2 * (2.0 * std::f64::consts::PI * 880.0 * t).sin()
            + 0.1 * (2.0 * std::f64::consts::PI * 1760.0 * t).sin();
        *sample = s as f32;
    }

    let scores = detector.process_window(synthetic_audio.clone())?;
    assert_eq!(scores.len(), 1);
    assert!(scores[0] >= 0.0 && scores[0] <= 1.0);

    // 3. Test FfiHotwordIterator
    let mut custom_config: FfiHotwordConfig = config;
    custom_config.threshold = 0.40;

    let iterator = FfiHotwordIterator::from_files(model_str, None, Some(custom_config))?;

    let mut detected_events = Vec::new();
    for chunk in synthetic_audio.chunks(640) {
        if let Some(event) = iterator.process_chunk(chunk.to_vec())? {
            detected_events.push(event);
        }
    }

    assert!(
        !detected_events.is_empty(),
        "expected at least one detection event"
    );
    let event = &detected_events[0];
    assert_eq!(event.keyword, "Hey Liquid");
    assert!(event.confidence >= 0.40);
    assert!(event.sample_offset > 0);

    // 4. Test reset
    iterator.reset()?;
    let zero_chunk = vec![0.0f32; 640];
    let next_event = iterator.process_chunk(zero_chunk)?;
    assert!(next_event.is_none());

    Ok(())
}

#[test]
fn test_ffi_hotword_large_chunk_processing() -> Result<()> {
    let Some(model_path) = find_hotword_model() else {
        return Ok(());
    };
    let model_str = model_path.to_str().unwrap().to_string();

    let custom_config = FfiHotwordConfig {
        threshold: 0.40,
        cooldown_ms: 2000,
        step_ms: 80,
        window_ms: 1200,
        pre_roll_ms: 150,
        vad_threshold: 0.5,
    };

    let iterator = FfiHotwordIterator::from_files(model_str, None, Some(custom_config))?;

    // Synthesize 3 seconds (48,000 samples)
    let mut synthetic_audio = vec![0.0f32; 48000];
    for (i, sample) in synthetic_audio.iter_mut().enumerate() {
        let t = i as f64 / 16000.0;
        let s = 0.3 * (2.0 * std::f64::consts::PI * 440.0 * t).sin()
            + 0.2 * (2.0 * std::f64::consts::PI * 880.0 * t).sin()
            + 0.1 * (2.0 * std::f64::consts::PI * 1760.0 * t).sin();
        *sample = s as f32;
    }

    // Pass the entire 48,000 samples in a single call
    let event = iterator.process_chunk(synthetic_audio)?;
    assert!(
        event.is_some(),
        "expected detection from large single-chunk call"
    );
    let event = event.unwrap();
    assert_eq!(event.keyword, "Hey Liquid");
    assert!(event.confidence >= 0.40);

    Ok(())
}
