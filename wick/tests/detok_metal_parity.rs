#![cfg(all(feature = "metal", target_os = "macos"))]

//! GPU↔CPU parity tests for the Metal detokenizer.
//!
//! These tests verify that the Metal GPU detokenizer produces spectrums
//! matching the CPU path. They serve as precision regression tests —
//! if a performance optimization (Q4_0 GEMV, f16 KV cache, fused shaders)
//! degrades audio quality, these tests fail.
//!
//! Thresholds assume both paths use F32 dequantized weights. If the GPU
//! path switches to quantized weights, thresholds must be re-evaluated.

use std::path::Path;

// ── Helpers ─────────────────────────────────────────────────────────────────

fn cosine_sim(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let dot: f64 = a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum();
    let na: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na * nb)
}

fn rms(x: &[f32]) -> f64 {
    (x.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / x.len() as f64).sqrt()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn band_energy(log_abs: &[f32], start: usize, end: usize) -> f64 {
    log_abs[start..end].iter().map(|&v| (v as f64).exp()).sum()
}

fn load_vocoder() -> Option<(wick::gguf::GgufFile, std::path::PathBuf)> {
    let path = Path::new(env!("HOME"))
        .join(".leap/models/LFM2.5-Audio-1.5B-Q4_0/vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf");
    if !path.exists() {
        eprintln!("vocoder not found at {}, skipping", path.display());
        return None;
    }
    let gguf = wick::gguf::GgufFile::open(&path).unwrap();
    Some((gguf, path))
}

// ── Test 1: Spectrum parity ─────────────────────────────────────────────────

#[test]
fn spectrum_parity() {
    let Some((gguf, path)) = load_vocoder() else {
        return;
    };

    let detok_w = wick::model::audio_decoder::DetokenizerWeights::from_gguf(&gguf).unwrap();
    let dec_w = wick::model::audio_decoder::AudioDecoderWeights::from_gguf(&gguf).unwrap();
    let gpu = wick::model::metal_audio_decoder::MetalAudioDecoder::from_gguf(&gguf, &path).unwrap();

    let code_sets: &[&[i32]] = &[
        &[100, 200, 300, 400, 500, 600, 700, 800],
        &[1, 1000, 2000, 500, 1500, 750, 1250, 2047],
    ];

    let n_fft_bins = 641;
    let spectrum_per_frame = n_fft_bins * 2; // 1282

    for (ci, codes) in code_sets.iter().enumerate() {
        // Reset both
        let mut cpu_state = wick::model::audio_decoder::DetokenizerState::new(&detok_w.config);
        gpu.reset();

        // Run CPU
        let cpu_spec = wick::model::audio_decoder::detokenize_to_spectrum(
            &detok_w,
            &dec_w,
            &mut cpu_state,
            codes,
        );

        // Run GPU
        let gpu_spec = gpu.detokenize_to_spectrum(&detok_w, codes);

        assert_eq!(
            cpu_spec.len(),
            gpu_spec.len(),
            "code set {ci}: spectrum length mismatch"
        );
        assert_eq!(
            cpu_spec.len(),
            6 * spectrum_per_frame,
            "code set {ci}: expected 6 frames × 1282"
        );

        // Per-frame checks
        for f in 0..6 {
            let off = f * spectrum_per_frame;
            let cpu_frame = &cpu_spec[off..off + spectrum_per_frame];
            let gpu_frame = &gpu_spec[off..off + spectrum_per_frame];

            // No NaN/Inf
            assert!(
                cpu_frame.iter().all(|v| v.is_finite()),
                "code set {ci}, frame {f}: CPU has NaN/Inf"
            );
            assert!(
                gpu_frame.iter().all(|v| v.is_finite()),
                "code set {ci}, frame {f}: GPU has NaN/Inf"
            );

            // Cosine similarity
            let cos = cosine_sim(cpu_frame, gpu_frame);
            assert!(
                cos > 0.99,
                "code set {ci}, frame {f}: cosine {cos:.6} < 0.99"
            );

            // Log-magnitude max diff
            let log_abs_cpu = &cpu_frame[..n_fft_bins];
            let log_abs_gpu = &gpu_frame[..n_fft_bins];
            let max_diff = max_abs_diff(log_abs_cpu, log_abs_gpu);
            assert!(
                max_diff < 0.5,
                "code set {ci}, frame {f}: log_abs max_diff {max_diff:.4} >= 0.5"
            );

            eprintln!("  code_set={ci} frame={f}: cos={cos:.6} max_diff={max_diff:.4}");
        }
    }
    eprintln!("spectrum_parity: PASSED");
}

// ── Test 2: PCM parity ──────────────────────────────────────────────────────

#[test]
fn pcm_parity() {
    let Some((gguf, path)) = load_vocoder() else {
        return;
    };

    let detok_w = wick::model::audio_decoder::DetokenizerWeights::from_gguf(&gguf).unwrap();
    let dec_w = wick::model::audio_decoder::AudioDecoderWeights::from_gguf(&gguf).unwrap();
    let gpu = wick::model::metal_audio_decoder::MetalAudioDecoder::from_gguf(&gguf, &path).unwrap();

    let codes: &[i32] = &[100, 200, 300, 400, 500, 600, 700, 800];
    let n_fft_bins = 641;

    let mut cpu_state = wick::model::audio_decoder::DetokenizerState::new(&detok_w.config);
    gpu.reset();

    let cpu_spec =
        wick::model::audio_decoder::detokenize_to_spectrum(&detok_w, &dec_w, &mut cpu_state, codes);
    let gpu_spec = gpu.detokenize_to_spectrum(&detok_w, codes);

    // ISTFT both
    let cpu_pcm = wick::model::audio_decoder::istft_to_pcm(
        &cpu_spec,
        detok_w.config.n_fft,
        detok_w.config.hop_length,
    );
    let gpu_pcm = wick::model::audio_decoder::istft_to_pcm(
        &gpu_spec,
        detok_w.config.n_fft,
        detok_w.config.hop_length,
    );

    let cpu_rms = rms(&cpu_pcm);
    let gpu_rms = rms(&gpu_pcm);

    assert!(cpu_rms > 0.0, "CPU PCM is silent");
    assert!(gpu_rms > 0.0, "GPU PCM is silent");
    assert!(cpu_pcm.iter().all(|v| v.is_finite()), "CPU PCM has NaN/Inf");
    assert!(gpu_pcm.iter().all(|v| v.is_finite()), "GPU PCM has NaN/Inf");

    let rms_ratio = gpu_rms / cpu_rms;
    eprintln!("  cpu_rms={cpu_rms:.2} gpu_rms={gpu_rms:.2} ratio={rms_ratio:.4}");
    assert!(
        (0.8..=1.25).contains(&rms_ratio),
        "PCM RMS ratio {rms_ratio:.4} outside [0.8, 1.25]"
    );

    // Per-band energy comparison (catches hollow/tinny sound)
    let bands = [(0, 160), (160, 320), (320, 480), (480, n_fft_bins)];
    let cpu_log_abs = &cpu_spec[..n_fft_bins];
    let gpu_log_abs = &gpu_spec[..n_fft_bins];

    for &(start, end) in &bands {
        let cpu_e = band_energy(cpu_log_abs, start, end);
        let gpu_e = band_energy(gpu_log_abs, start, end);
        if cpu_e > 1e-10 {
            let ratio = gpu_e / cpu_e;
            eprintln!("  band [{start}-{end}]: cpu={cpu_e:.2} gpu={gpu_e:.2} ratio={ratio:.4}");
            assert!(
                (0.5..=2.0).contains(&ratio),
                "Band [{start}-{end}] energy ratio {ratio:.4} outside [0.5, 2.0]"
            );
        }
    }
    eprintln!("pcm_parity: PASSED");
}

// ── Test 3: Multi-frame stability ───────────────────────────────────────────

#[test]
fn multi_frame_stability() {
    let Some((gguf, path)) = load_vocoder() else {
        return;
    };

    let detok_w = wick::model::audio_decoder::DetokenizerWeights::from_gguf(&gguf).unwrap();
    let dec_w = wick::model::audio_decoder::AudioDecoderWeights::from_gguf(&gguf).unwrap();
    let gpu = wick::model::metal_audio_decoder::MetalAudioDecoder::from_gguf(&gguf, &path).unwrap();

    let mut cpu_state = wick::model::audio_decoder::DetokenizerState::new(&detok_w.config);
    gpu.reset();

    // 8 frames = 48 tokens → wraps the SWA window (30) 1.6 times
    let all_codes: Vec<[i32; 8]> = (0..8)
        .map(|i| {
            let base = (i * 100 + 50) as i32;
            [
                base,
                base + 100,
                base + 200,
                base + 300,
                base + 400,
                base + 500,
                base + 600,
                base + 700,
            ]
        })
        .collect();

    let n_fft_bins = 641;
    let spectrum_per_frame = n_fft_bins * 2;
    let mut first_rms_cpu = 0.0f64;
    let mut first_rms_gpu = 0.0f64;

    for (fi, codes) in all_codes.iter().enumerate() {
        let cpu_spec = wick::model::audio_decoder::detokenize_to_spectrum(
            &detok_w,
            &dec_w,
            &mut cpu_state,
            codes,
        );
        let gpu_spec = gpu.detokenize_to_spectrum(&detok_w, codes);

        // Check first frame of this call
        let cpu_f0 = &cpu_spec[..spectrum_per_frame];
        let gpu_f0 = &gpu_spec[..spectrum_per_frame];

        assert!(
            cpu_f0.iter().all(|v| v.is_finite()),
            "frame {fi}: CPU has NaN/Inf"
        );
        assert!(
            gpu_f0.iter().all(|v| v.is_finite()),
            "frame {fi}: GPU has NaN/Inf"
        );

        let cos = cosine_sim(cpu_f0, gpu_f0);
        let cpu_r = rms(cpu_f0);
        let gpu_r = rms(gpu_f0);

        if fi == 0 {
            first_rms_cpu = cpu_r;
            first_rms_gpu = gpu_r;
        }

        eprintln!("  frame {fi}: cos={cos:.6} cpu_rms={cpu_r:.4} gpu_rms={gpu_r:.4}");

        // Later frames may diverge more due to accumulated state
        let threshold = if fi < 3 { 0.99 } else { 0.95 };
        assert!(cos > threshold, "frame {fi}: cosine {cos:.6} < {threshold}");

        // RMS shouldn't explode
        if first_rms_cpu > 1e-10 {
            assert!(
                cpu_r / first_rms_cpu < 100.0,
                "frame {fi}: CPU RMS exploded ({cpu_r:.2} / {first_rms_cpu:.2})"
            );
        }
        if first_rms_gpu > 1e-10 {
            assert!(
                gpu_r / first_rms_gpu < 100.0,
                "frame {fi}: GPU RMS exploded ({gpu_r:.2} / {first_rms_gpu:.2})"
            );
        }
    }
    eprintln!("multi_frame_stability: PASSED");
}
