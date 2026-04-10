#![cfg(all(feature = "metal", target_os = "macos"))]

#[test]
fn frame2_codes_from_ref_embedding() {
    let vocoder_path = std::path::PathBuf::from(std::env::var("HOME").expect("HOME not set"))
        .join(".leap/models/LFM2.5-Audio-1.5B-Q4_0/vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf");

    let load_emb = |path: &str| -> Vec<f32> {
        std::fs::read(path)
            .unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };

    if !vocoder_path.exists() {
        return;
    }
    // These reference files are created by external comparison scripts.
    let ref_files = [
        "/tmp/ref_frame0_emb.bin",
        "/tmp/ref_frame1_emb.bin",
        "/tmp/ref_frame2_emb.bin",
        "/tmp/wick_frame2_emb.bin",
    ];
    if ref_files.iter().any(|f| !std::path::Path::new(f).exists()) {
        eprintln!("skipping — reference embedding files not found in /tmp/");
        return;
    }
    let ref_f0 = load_emb("/tmp/ref_frame0_emb.bin");
    let ref_f1 = load_emb("/tmp/ref_frame1_emb.bin");
    let ref_f2 = load_emb("/tmp/ref_frame2_emb.bin");
    let wick_f2 = load_emb("/tmp/wick_frame2_emb.bin");

    let voc_gguf = wick::gguf::GgufFile::open(&vocoder_path).unwrap();
    let dw = wick::model::audio_decoder::AudioDecoderWeights::from_gguf(&voc_gguf).unwrap();

    // The depthformer has KV cache state from previous frames.
    // We need to replay frames 0 and 1 to build up the state, then check frame 2.

    // Test 1: ref embeddings for all 3 frames
    {
        let mut df = wick::model::audio_decoder::DepthformerState::new(&dw.depthformer_config);
        let c0 = wick::model::audio_decoder::sample_audio_frame(&dw, &mut df, &ref_f0, 0.0, 1);
        let c1 = wick::model::audio_decoder::sample_audio_frame(&dw, &mut df, &ref_f1, 0.0, 1);
        let c2 = wick::model::audio_decoder::sample_audio_frame(&dw, &mut df, &ref_f2, 0.0, 1);
        eprintln!("All ref embs → c0={c0:?} c1={c1:?} c2={c2:?}");
    }

    // Test 2: ref for frames 0-1, then WICK embedding for frame 2
    {
        let mut df = wick::model::audio_decoder::DepthformerState::new(&dw.depthformer_config);
        let c0 = wick::model::audio_decoder::sample_audio_frame(&dw, &mut df, &ref_f0, 0.0, 1);
        let c1 = wick::model::audio_decoder::sample_audio_frame(&dw, &mut df, &ref_f1, 0.0, 1);
        let c2 = wick::model::audio_decoder::sample_audio_frame(&dw, &mut df, &wick_f2, 0.0, 1);
        eprintln!("Ref f0+f1, wick f2 → c0={c0:?} c1={c1:?} c2={c2:?}");
    }

    // Test 3: all WICK embeddings (to see what wick actually does)
    // We only have wick's frame 1 and 2, not frame 0. Use ref's frame 0.
    {
        let wick_f1 = load_emb("/tmp/wick_frame1_emb.bin");
        let mut df = wick::model::audio_decoder::DepthformerState::new(&dw.depthformer_config);
        let c0 = wick::model::audio_decoder::sample_audio_frame(&dw, &mut df, &ref_f0, 0.0, 1);
        let c1 = wick::model::audio_decoder::sample_audio_frame(&dw, &mut df, &wick_f1, 0.0, 1);
        let c2 = wick::model::audio_decoder::sample_audio_frame(&dw, &mut df, &wick_f2, 0.0, 1);
        eprintln!("Ref f0, wick f1+f2 → c0={c0:?} c1={c1:?} c2={c2:?}");
    }
}
