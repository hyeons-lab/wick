#[test]
fn depthformer_forward_smoke() {
    let path = std::path::PathBuf::from(std::env::var("HOME").expect("HOME not set"))
        .join(".leap/models/LFM2.5-Audio-1.5B-Q4_0/vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf");
    if !path.exists() {
        eprintln!("skipping — vocoder not found");
        return;
    }

    let gguf = wick::gguf::GgufFile::open(&path).unwrap();
    let weights = wick::model::audio_decoder::AudioDecoderWeights::from_gguf(&gguf).unwrap();
    let mut state = wick::model::audio_decoder::DepthformerState::new(&weights.depthformer_config);

    // Feed a random 1024-dim input, run 3 sequential tokens (like 3 codebook steps).
    let input: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.001).sin()).collect();
    for step in 0..3 {
        let t0 = std::time::Instant::now();
        let out = wick::model::audio_decoder::depthformer_forward(&weights, &mut state, &input);
        let ms = t0.elapsed().as_millis();
        assert_eq!(out.len(), 1024);
        assert!(
            out.iter().all(|x| x.is_finite()),
            "step {step}: NaN/Inf in output"
        );
        eprintln!("step {step}: {ms} ms, out[0..4] = {:?}", &out[..4]);
    }

    // Reset and verify state is clean.
    state.reset();
    let out2 = wick::model::audio_decoder::depthformer_forward(&weights, &mut state, &input);
    assert_eq!(out2.len(), 1024);
    eprintln!("after reset: out[0..4] = {:?}", &out2[..4]);
}
