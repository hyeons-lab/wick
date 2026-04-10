#![cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn detok_bypass_backbone() {
    let path = std::path::PathBuf::from(std::env::var("HOME").expect("HOME not set"))
        .join(".leap/models/LFM2.5-Audio-1.5B-Q4_0/vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf");
    if !path.exists() {
        return;
    }
    let gguf = wick::gguf::GgufFile::open(&path).unwrap();
    let tw = wick::model::audio_decoder::DetokenizerWeights::from_gguf(&gguf).unwrap();
    let n = tw.config.n_embd;

    // Embed codes.
    let emb = &tw.emb_weight;
    let n_vocab_per_cb = emb.rows / tw.config.n_codes;
    let codes = [500i32; 8];
    let mut embedding = vec![0.0f32; n];
    for (j, &c) in codes.iter().enumerate() {
        let idx = j * n_vocab_per_cb + c as usize;
        for (r, e) in embedding.iter_mut().zip(&emb.data[idx * n..(idx + 1) * n]) {
            *r += e;
        }
    }
    for r in &mut embedding {
        *r /= 8.0;
    }

    // Skip backbone — apply output norm + linear head directly to embedding.
    let mut normed = embedding.clone();
    wick::backend::cpu::rmsnorm(&mut normed, &tw.output_norm, tw.config.rms_norm_eps);

    let mut lin_out = vec![0.0; tw.lin_w.rows];
    tw.lin_w.gemv(&normed, &mut lin_out);
    for (l, b) in lin_out.iter_mut().zip(&tw.lin_b) {
        *l += b;
    }

    let n_bins = tw.config.n_fft / 2 + 1;
    let log_abs = &lin_out[..n_bins];
    let angles = &lin_out[n_bins..];
    eprintln!("BYPASS (no backbone):");
    eprintln!(
        "  log_abs: min={:.2} max={:.2}",
        log_abs.iter().cloned().fold(f32::INFINITY, f32::min),
        log_abs.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );
    eprintln!(
        "  angles: min={:.2} max={:.2}",
        angles.iter().cloned().fold(f32::INFINITY, f32::min),
        angles.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );

    // ISTFT just the bypass spectrum (6 identical frames).
    let mut spectrum = Vec::new();
    for _ in 0..6 {
        spectrum.extend_from_slice(&lin_out);
    }
    let pcm =
        wick::model::audio_decoder::istft_to_pcm(&spectrum, tw.config.n_fft, tw.config.hop_length);
    let rms = (pcm.iter().map(|x| x * x).sum::<f32>() / pcm.len().max(1) as f32).sqrt();
    let peak = pcm.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    eprintln!("  PCM: rms={rms:.4}, peak={peak:.4}");

    // Compare: WITH backbone (use existing detokenize_to_spectrum).
    let dw = wick::model::audio_decoder::AudioDecoderWeights::from_gguf(&gguf).unwrap();
    let mut ts = wick::model::audio_decoder::DetokenizerState::new(&tw.config);
    let full_spectrum =
        wick::model::audio_decoder::detokenize_to_spectrum(&tw, &dw, &mut ts, &codes);
    let full_pcm = wick::model::audio_decoder::istft_to_pcm(
        &full_spectrum,
        tw.config.n_fft,
        tw.config.hop_length,
    );
    let full_rms =
        (full_pcm.iter().map(|x| x * x).sum::<f32>() / full_pcm.len().max(1) as f32).sqrt();
    let full_peak = full_pcm.iter().map(|x| x.abs()).fold(0.0f32, f32::max);

    let n_frames_full = full_spectrum.len() / (n_bins * 2);
    let full_la = &full_spectrum[..n_bins];
    let full_ang = &full_spectrum[n_bins..n_bins * 2];
    eprintln!("WITH backbone ({n_frames_full} frames):");
    eprintln!(
        "  log_abs: min={:.2} max={:.2}",
        full_la.iter().cloned().fold(f32::INFINITY, f32::min),
        full_la.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );
    eprintln!(
        "  angles: min={:.2} max={:.2}",
        full_ang.iter().cloned().fold(f32::INFINITY, f32::min),
        full_ang.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );
    eprintln!("  PCM: rms={full_rms:.4}, peak={full_peak:.4}");
}
