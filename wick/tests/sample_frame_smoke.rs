#[test]
fn sample_audio_frame_smoke() {
    let path = std::path::PathBuf::from(std::env::var("HOME").expect("HOME not set"))
        .join(".leap/models/LFM2.5-Audio-1.5B-Q4_0/vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf");
    if !path.exists() {
        eprintln!("skipping — vocoder not found");
        return;
    }

    let gguf = wick::gguf::GgufFile::open(&path).unwrap();
    let weights = wick::model::audio_decoder::AudioDecoderWeights::from_gguf(&gguf).unwrap();
    let mut state = wick::model::audio_decoder::DepthformerState::new(&weights.depthformer_config);

    // Fake LLM embedding (2048-dim).
    let embedding: Vec<f32> = (0..2048).map(|i| (i as f32 * 0.001).sin()).collect();

    let t0 = std::time::Instant::now();
    let codes =
        wick::model::audio_decoder::sample_audio_frame(&weights, &mut state, &embedding, 0.0, 1);
    let ms = t0.elapsed().as_millis();
    eprintln!("audio frame sampled in {ms} ms: {:?}", codes);

    // Verify: 8 codes, each in [0, 2049).
    assert_eq!(codes.len(), 8);
    for (j, &c) in codes.iter().enumerate() {
        assert!(c >= 0 && c < 2049, "code {j} = {c} out of range");
    }

    // Embed codes back.
    let emb = wick::model::audio_decoder::embed_audio_token(&weights, &codes);
    assert_eq!(emb.len(), 2048);
    assert!(emb.iter().all(|x| x.is_finite()));
    eprintln!("embed_audio_token OK: dim={}", emb.len());
}
