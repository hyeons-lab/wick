#[test]
fn detokenize_smoke() {
    let path = std::path::PathBuf::from(std::env::var("HOME").expect("HOME not set"))
        .join(".leap/models/LFM2.5-Audio-1.5B-Q4_0/vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf");
    if !path.exists() {
        eprintln!("skipping — vocoder not found");
        return;
    }

    let gguf = wick::gguf::GgufFile::open(&path).unwrap();
    let decoder_weights =
        wick::model::audio_decoder::AudioDecoderWeights::from_gguf(&gguf).unwrap();
    let detok_weights = wick::model::audio_decoder::DetokenizerWeights::from_gguf(&gguf).unwrap();
    let mut detok_state = wick::model::audio_decoder::DetokenizerState::new(&detok_weights.config);

    // Use fake codes (won't produce meaningful audio but tests the pipeline).
    let codes = [100i32, 200, 300, 400, 500, 600, 700, 800];

    let t0 = std::time::Instant::now();
    let spectrum = wick::model::audio_decoder::detokenize_to_spectrum(
        &detok_weights,
        &decoder_weights,
        &mut detok_state,
        &codes,
    );
    let spec_ms = t0.elapsed().as_millis();

    let n_fft_bins = detok_weights.config.n_fft / 2 + 1; // 641
    let frame_size = n_fft_bins * 2; // 1282
    let n_frames = spectrum.len() / frame_size;
    eprintln!(
        "spectrum: {} frames, {} total values, {spec_ms} ms",
        n_frames,
        spectrum.len()
    );
    assert!(n_frames > 0);
    assert!(
        spectrum.iter().all(|x| x.is_finite()),
        "spectrum has NaN/Inf"
    );

    // ISTFT.
    let t1 = std::time::Instant::now();
    let pcm = wick::model::audio_decoder::istft_to_pcm(
        &spectrum,
        detok_weights.config.n_fft,
        detok_weights.config.hop_length,
    );
    let istft_ms = t1.elapsed().as_millis();
    eprintln!(
        "PCM: {} samples ({:.1} ms of audio at {} Hz), ISTFT took {istft_ms} ms",
        pcm.len(),
        pcm.len() as f64 / detok_weights.config.sample_rate as f64 * 1000.0,
        detok_weights.config.sample_rate,
    );
    assert!(!pcm.is_empty());
    assert!(pcm.iter().all(|x| x.is_finite()), "PCM has NaN/Inf");
    eprintln!("detokenize_smoke OK");
}
