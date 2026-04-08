#![cfg(all(feature = "metal", target_os = "macos"))]

/// Check depthformer sensitivity: feed wick's frame 1 embedding and ref's frame 1
/// embedding, compare cb2 logits to see why argmax flips.
#[test]
fn depthformer_cb2_sensitivity() {
    let vocoder_path = std::path::Path::new(env!("HOME"))
        .join(".leap/models/LFM2.5-Audio-1.5B-Q4_0/vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf");
    let wick_emb_path = std::path::Path::new("/tmp/wick_frame1_emb.bin");
    let ref_emb_path = std::path::Path::new("/tmp/ref_frame1_emb.bin");
    if !vocoder_path.exists() || !wick_emb_path.exists() || !ref_emb_path.exists() {
        eprintln!("Skipping: files not found");
        return;
    }

    let load_emb = |path: &std::path::Path| -> Vec<f32> {
        std::fs::read(path)
            .unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };
    let wick_emb = load_emb(wick_emb_path);
    let ref_emb = load_emb(ref_emb_path);

    let voc_gguf = wick::gguf::GgufFile::open(&vocoder_path).unwrap();
    let dw = wick::model::audio_decoder::AudioDecoderWeights::from_gguf(&voc_gguf).unwrap();

    // Both should produce frame 1 codes [127, 1470, 457, 1422, 481, 1509, 976, 2008]
    // Frame 2 codes diverge at cb2: wick=1697, ref=1400

    // Run depthformer for frame 1 (to populate KV cache), then frame 2
    for (label, emb) in [("wick", &wick_emb), ("ref", &ref_emb)] {
        let mut df_state =
            wick::model::audio_decoder::DepthformerState::new(&dw.depthformer_config);

        // Frame 1 codes (both produce the same)
        let codes1 =
            wick::model::audio_decoder::sample_audio_frame(&dw, &mut df_state, emb, 0.0, 1);
        eprintln!("{label} frame 1: {codes1:?}");

        // Now feed the frame 1 codes through embed_audio_token and run depthformer for frame 2
        // Actually, sample_audio_frame already ran the depthformer for 8 codebooks.
        // For frame 2, we need a NEW embedding (from the LLM feedback). But we don't
        // have the frame 2 embedding from ggml.
        //
        // Instead, let's just check: do both embeddings produce the same frame 1 codes?
    }

    // More detailed: manually run the depthformer for codebook 2 of frame 2
    // This requires the frame 2 embedding, which we don't have.
    // Instead, run frame 1 with both embeddings and compare frame 1's cb2 logits margin.
    for (label, emb) in [("wick", &wick_emb), ("ref", &ref_emb)] {
        let mut df_state =
            wick::model::audio_decoder::DepthformerState::new(&dw.depthformer_config);
        let codes = wick::model::audio_decoder::sample_audio_frame(&dw, &mut df_state, emb, 0.0, 1);
        eprintln!("{label} frame 1 codes: {codes:?}");

        // Check if frame 1 codes match between wick and ref embeddings
    }
}
