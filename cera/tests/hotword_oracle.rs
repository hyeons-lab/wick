#![cfg(all(feature = "mmap", not(target_arch = "wasm32")))]

use std::path::PathBuf;

use anyhow::Result;
use cera::hotword::{HotwordDetector, HotwordIterator};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct FixtureData {
    keywords: Vec<String>,
    sample_rate: usize,
    window_samples: usize,
    mel_bins: usize,
    embedding_dim: usize,
    audio_slice: Vec<f32>,
    mel_slice: Vec<Vec<f32>>,
    embedding: Vec<f32>,
    probabilities: Vec<f32>,
}

fn find_hotword_model() -> Option<PathBuf> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        manifest_dir.join("models/hey_liquid.gguf"),
        manifest_dir.join("../models/hey_liquid.gguf"),
        PathBuf::from("models/hey_liquid.gguf"),
    ];
    candidates.into_iter().find(|p| p.exists())
}

fn find_hotword_fixture() -> Option<PathBuf> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        manifest_dir.join("models/hey_liquid_fixture.json"),
        manifest_dir.join("../models/hey_liquid_fixture.json"),
        PathBuf::from("models/hey_liquid_fixture.json"),
    ];
    candidates.into_iter().find(|p| p.exists())
}

#[test]
fn test_hotword_oracle_parity() -> Result<()> {
    let Some(model_path) = find_hotword_model() else {
        eprintln!("Skipping test: models/hey_liquid.gguf not found");
        return Ok(());
    };
    let Some(fixture_path) = find_hotword_fixture() else {
        eprintln!("Skipping test: models/hey_liquid_fixture.json not found");
        return Ok(());
    };

    let fixture_bytes = std::fs::read(&fixture_path)?;
    let fixture: FixtureData = serde_json::from_slice(&fixture_bytes)?;

    let mut detector = HotwordDetector::from_file(&model_path)?;
    assert_eq!(detector.keywords(), &fixture.keywords);

    // 1. Synthesize 19,200 samples audio using reference formula
    let mut synthetic_audio = vec![0.0f32; fixture.window_samples];
    for (i, sample) in synthetic_audio.iter_mut().enumerate() {
        let t = i as f64 / fixture.sample_rate as f64;
        let s = 0.3 * (2.0 * std::f64::consts::PI * 440.0 * t).sin()
            + 0.2 * (2.0 * std::f64::consts::PI * 880.0 * t).sin()
            + 0.1 * (2.0 * std::f64::consts::PI * 1760.0 * t).sin();
        *sample = s as f32;
    }

    // Verify audio synthesis matches fixture
    for (i, &expected) in fixture.audio_slice.iter().enumerate() {
        let diff = (synthetic_audio[i] - expected).abs();
        assert!(
            diff < 1e-4,
            "synthetic audio sample {i} mismatch: got {}, expected {}",
            synthetic_audio[i],
            expected
        );
    }

    // 2. Verify mel-spectrogram extraction parity against Python reference
    let num_frames = detector.front_end().num_frames(synthetic_audio.len());
    let mut mel_extracted = vec![0.0f32; fixture.mel_bins * num_frames];
    detector
        .front_end()
        .extract(&synthetic_audio, &mut mel_extracted);

    for (m, row) in fixture.mel_slice.iter().enumerate() {
        for (f, &expected) in row.iter().enumerate() {
            let actual = mel_extracted[m * num_frames + f];
            let diff = (actual - expected).abs();
            assert!(
                diff < 1e-3,
                "mel spectrogram bin {m} frame {f} mismatch: got {actual:.5}, expected {expected:.5} (diff: {diff:.2e})"
            );
        }
    }

    // 3. Run detector forward pass
    let scores = detector.process_window(&synthetic_audio)?;
    assert_eq!(scores.len(), fixture.probabilities.len());

    for (k, (&score, &expected)) in scores.iter().zip(&fixture.probabilities).enumerate() {
        let diff = (score - expected).abs();
        assert!(
            diff < 1e-3,
            "score for keyword '{k}' mismatch: got {score:.5}, expected {expected:.5} (diff: {diff:.2e})"
        );
    }

    // 3. Streaming evaluation with HotwordIterator
    let mut config = detector.default_config();
    config.threshold = 0.40; // Allow fixture probability (~0.499) to trigger
    let mut iterator = HotwordIterator::new(detector, None, config);

    let chunk_size = 640; // 40 ms @ 16 kHz
    let mut detected_count = 0;

    for chunk in synthetic_audio.chunks(chunk_size) {
        if let Some(event) = iterator.process_chunk(chunk)? {
            assert_eq!(event.keyword, "Hey Liquid");
            assert!(event.confidence >= 0.40);
            assert!(event.sample_offset > 0);
            detected_count += 1;
        }
    }

    assert!(
        detected_count >= 1,
        "expected at least one detection event during synthetic audio stream"
    );

    // 4. Test reset semantics
    iterator.reset();
    let zero_chunk = vec![0.0f32; chunk_size];
    let event = iterator.process_chunk(&zero_chunk)?;
    assert!(event.is_none());

    Ok(())
}
