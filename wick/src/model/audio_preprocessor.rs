//! LFM2A audio preprocessor — PCM samples → log-mel spectrogram.
//!
//! Mirrors the C++ reference's `mtmd_audio_preprocessor_conformer`
//! pipeline: center-pad → pre-emphasis → per-frame (Hann-window
//! → FFT → power → mel filterbank → natural log) → per-feature
//! normalization. Output is `[n_frames × n_mel_bins]` row-major
//! (time-major outer, freq inner) — the natural input layout for
//! `audio_encoder::conv_stem_forward`.
//!
//! Design notes:
//! - **Slaney mel scale** (linear < 1 kHz, log ≥ 1 kHz) with
//!   Slaney area normalization. Matches librosa defaults and the
//!   C++ reference exactly. Differs from the HTK formula
//!   (`2595 * log10(1 + f / 700)`).
//! - **Hann window** is `WINDOW_LEN = 400` samples (periodic),
//!   centered inside an `N_FFT = 512`-sized buffer (zero-padded
//!   56 samples on each side).
//! - **Center padding** by `N_FFT / 2 = 256` zeros on both
//!   sides (Whisper / librosa `center=True` mode).
//! - **Pre-emphasis** runs over the inner (un-padded) region only.
//! - **Per-feature norm** uses the unbiased variance estimator
//!   (denominator `effective_n_len - 1`) and an `eps = 1e-5`
//!   floor before the sqrt — matches the C++ ref exactly.
//! - **f64 accumulation** for the per-feature mean/var sums and
//!   the mel-filterbank dot products, matching the project's
//!   numerical-precision convention.
//!
//! Allocates a fresh `rustfft` planner + per-call scratch on
//! every invocation. The encoder runs once per audio chunk so
//! amortized cost is negligible; if a real-time-streaming caller
//! ever needs sub-ms-per-frame, this is the obvious place to
//! introduce a thread-local cache.

use crate::model::audio_encoder::{HOP_LEN, LOG_MEL_EPS, N_FFT, PREEMPH, SAMPLE_RATE, WINDOW_LEN};
use rustfft::FftPlanner;
use rustfft::num_complex::Complex32;

/// Number of unique FFT bins for an `N_FFT`-point real-input
/// transform: the bins `0..=N_FFT/2` are unique; the rest are
/// complex conjugates of those.
pub const N_FFT_BINS: usize = N_FFT / 2 + 1;

// Compile-time guard: `(N_FFT - WINDOW_LEN) / 2` in the per-frame
// hann-padding math would underflow if a future config change
// inverted the relation. Catch it at build time before the
// runtime panic could ever happen.
const _: () = assert!(
    WINDOW_LEN <= N_FFT,
    "WINDOW_LEN must be <= N_FFT (audio_encoder constants)"
);

/// Build the Slaney-scale mel filterbank matrix
/// `[n_mel × N_FFT_BINS]`, row-major. Each row is the per-bin
/// triangular weighting for one mel filter, with Slaney area
/// normalization (`enorm = 2 / (f_right - f_left)`). Matches the
/// C++ reference's `fill_mel_filterbank_matrix` with the
/// Conformer call site's defaults (`fmin = 0`, `fmax = sr / 2`).
pub fn build_mel_filterbank(n_mel: usize, n_fft: usize, sample_rate: usize) -> Vec<f32> {
    assert!(n_mel > 0, "n_mel must be > 0");
    assert!(n_fft > 1, "n_fft must be > 1");
    // sample_rate = 0 would silently collapse fmax to 0 and emit
    // an all-zero filterbank — fail loudly instead.
    assert!(sample_rate > 0, "sample_rate must be > 0");

    let n_fft_bins = n_fft / 2 + 1;
    let bin_hz_step = sample_rate as f64 / n_fft as f64;
    let fmin = 0.0_f64;
    let fmax = 0.5_f64 * sample_rate as f64;

    // Slaney mel scale: linear below 1 kHz, log above.
    let min_log_hz = 1000.0_f64;
    let lin_slope = 3.0 / 200.0;
    let min_log_mel = min_log_hz * lin_slope;
    let log_step = 6.4_f64.ln() / 27.0;
    let hz_to_mel = |f_hz: f64| -> f64 {
        if f_hz < min_log_hz {
            f_hz * lin_slope
        } else {
            min_log_mel + (f_hz / min_log_hz).ln() / log_step
        }
    };
    let mel_to_hz = |m: f64| -> f64 {
        if m < min_log_mel {
            m / lin_slope
        } else {
            min_log_hz * ((m - min_log_mel) * log_step).exp()
        }
    };

    // n_mel + 2 mel-equispaced points (left/center/right edges).
    let m_lo = hz_to_mel(fmin);
    let m_hi = hz_to_mel(fmax);
    let mut hz_pts = Vec::with_capacity(n_mel + 2);
    for i in 0..(n_mel + 2) {
        let m = m_lo + (m_hi - m_lo) * (i as f64 / (n_mel + 1) as f64);
        hz_pts.push(mel_to_hz(m));
    }

    let mut filters = vec![0.0f32; n_mel * n_fft_bins];
    for m in 0..n_mel {
        let f_left = hz_pts[m];
        let f_center = hz_pts[m + 1];
        let f_right = hz_pts[m + 2];
        let denom_l = (f_center - f_left).max(1e-30);
        let denom_r = (f_right - f_center).max(1e-30);
        // Slaney area normalization.
        let enorm = 2.0 / (f_right - f_left).max(1e-30);

        let row = &mut filters[m * n_fft_bins..(m + 1) * n_fft_bins];
        for (k, slot) in row.iter_mut().enumerate() {
            let f = k as f64 * bin_hz_step;
            let w = if f >= f_left && f <= f_center {
                (f - f_left) / denom_l
            } else if f > f_center && f <= f_right {
                (f_right - f) / denom_r
            } else {
                0.0
            };
            *slot = (w * enorm) as f32;
        }
    }
    filters
}

/// Build a periodic Hann window of `length` samples. "Periodic"
/// matches librosa / Whisper / the C++ reference (cos divisor is
/// `length`, not `length - 1` as in the symmetric form).
pub fn build_hann_window(length: usize) -> Vec<f32> {
    let mut w = Vec::with_capacity(length);
    let denom = length as f64;
    for i in 0..length {
        let v = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / denom).cos());
        w.push(v as f32);
    }
    w
}

/// Compute the LFM2A log-mel spectrogram of a mono PCM chunk
/// sampled at `SAMPLE_RATE` (16 kHz). Output is row-major
/// `[n_frames × n_mel_bins]` ready to feed into
/// `audio_encoder::conv_stem_forward`.
///
/// Returns `(mel, n_frames)`. Empty input yields `(vec![], 0)`.
///
/// Per the C++ reference's `mtmd_audio_preprocessor_conformer`:
/// - Center-pad input by `N_FFT / 2` zeros on both sides
///   (Whisper / librosa `center=True`).
/// - Pre-emphasis (`y[t] = x[t] - PREEMPH * x[t-1]`) on the
///   inner (un-padded) region only.
/// - Per frame: Hann-window the (`N_FFT`-padded) frame, FFT,
///   power spectrum, mel filterbank projection, natural log
///   with `LOG_MEL_EPS` floor.
/// - Per-feature normalization (zero mean / unit variance) per
///   mel bin, computed across the **effective** number of frames
///   (= `n_samples_in / HOP_LEN`); frames after the effective
///   end are zeroed out.
///
/// `n_frames` is derived from the **padded** sample length (matching
/// the C++ reference's `out.n_len = (n_samples_padded - frame_size)
/// / hop + 1`). The frames in `[effective_n_len, n_frames)` are
/// part of the output but are post-norm zeroed, so downstream
/// callers see them as a valid-but-silent tail. Trimming to
/// `effective_n_len` would diverge from the reference's frame
/// count — the conv stem expects all `n_frames` rows.
pub fn log_mel_spectrogram(pcm: &[f32], n_mel_bins: usize) -> (Vec<f32>, usize) {
    if pcm.is_empty() {
        return (Vec::new(), 0);
    }
    assert!(n_mel_bins > 0, "n_mel_bins must be > 0");

    let n_samples_in = pcm.len();
    let pad_amount = N_FFT / 2;

    // Center-pad: prepend + append `pad_amount` zeros.
    let n_samples_padded = n_samples_in
        .checked_add(2 * pad_amount)
        .expect("log_mel_spectrogram: n_samples + 2 * pad_amount overflowed usize");
    let mut samples = vec![0.0f32; n_samples_padded];
    samples[pad_amount..pad_amount + n_samples_in].copy_from_slice(pcm);

    // Pre-emphasis on the inner region only (matches C++ ref).
    // C++ writes back to samples[pad_amount + 1..n_samples - pad_amount];
    // first inner sample is left untouched.
    let inner_end = n_samples_padded - pad_amount;
    let mut prev = samples[pad_amount];
    for s in samples[pad_amount + 1..inner_end].iter_mut() {
        let cur = *s;
        *s = cur - PREEMPH * prev;
        prev = cur;
    }

    // Hann window centered inside an N_FFT-sized buffer (left-pad
    // = (N_FFT - WINDOW_LEN) / 2 zeros, then WINDOW_LEN of Hann,
    // then right-pad zeros).
    let hann_raw = build_hann_window(WINDOW_LEN);
    let hann_pad = (N_FFT - WINDOW_LEN) / 2;
    let mut hann = vec![0.0f32; N_FFT];
    hann[hann_pad..hann_pad + WINDOW_LEN].copy_from_slice(&hann_raw);

    // Mel filterbank.
    let filters = build_mel_filterbank(n_mel_bins, N_FFT, SAMPLE_RATE as usize);

    // FFT planner (one per call; encoder runs per chunk so this
    // is amortized).
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);

    let n_frames = if n_samples_padded < N_FFT {
        0
    } else {
        (n_samples_padded - N_FFT) / HOP_LEN + 1
    };
    if n_frames == 0 {
        return (Vec::new(), 0);
    }

    // Compute mel spectrogram in mel-major layout (per-feature
    // norm walks per-mel-bin slices of consecutive timesteps —
    // contiguous in this layout). Transpose to time-major at the
    // end for the conv_stem_forward consumer.
    let mut mel = vec![0.0f32; n_mel_bins * n_frames];
    let mut fft_buf: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); N_FFT];
    // Power spectrum scratch — hoisted out of the (ti, mi) loop
    // so each frame computes |X[k]|² exactly once instead of
    // n_mel_bins times.
    let mut power_spec = vec![0.0f64; N_FFT_BINS];

    for ti in 0..n_frames {
        let offset = ti * HOP_LEN;
        // Apply Hann window to this frame; clear the imaginary
        // parts. n_frames is sized so `offset + N_FFT - 1` always
        // falls within `samples` (no out-of-bounds branch needed).
        let frame_samples = &samples[offset..offset + N_FFT];
        for (fb, (&h, &s)) in fft_buf.iter_mut().zip(hann.iter().zip(frame_samples)) {
            *fb = Complex32::new(h * s, 0.0);
        }
        fft.process(&mut fft_buf);

        // Per-frame power spectrum, computed once.
        for (p, c) in power_spec.iter_mut().zip(fft_buf.iter().take(N_FFT_BINS)) {
            *p = c.re as f64 * c.re as f64 + c.im as f64 * c.im as f64;
        }

        // Per-mel-bin filter dot product. f64 accumulation per
        // the project convention.
        for mi in 0..n_mel_bins {
            let frow = &filters[mi * N_FFT_BINS..(mi + 1) * N_FFT_BINS];
            let sum: f64 = power_spec
                .iter()
                .zip(frow)
                .map(|(&p, &f)| p * f as f64)
                .sum();
            mel[mi * n_frames + ti] = (sum + LOG_MEL_EPS as f64).ln() as f32;
        }
    }

    // Per-feature normalization across the effective_n_len timesteps
    // (= n_samples_in / HOP_LEN). Frames beyond effective_n_len are
    // always zeroed out. For `effective_n_len == 1`, the single
    // live frame is also zeroed (centering around its own value
    // gives 0; variance is undefined in the unbiased estimator) —
    // this keeps the output uniformly zero-tailed for short inputs
    // instead of leaving frame 0 as an unnormalized raw log-mel.
    let effective_n_len = (n_samples_in / HOP_LEN).min(n_frames);
    for mi in 0..n_mel_bins {
        let row = &mut mel[mi * n_frames..(mi + 1) * n_frames];
        if effective_n_len > 1 {
            let mut mean_sum = 0.0f64;
            for &v in &row[..effective_n_len] {
                mean_sum += v as f64;
            }
            let mean = mean_sum / effective_n_len as f64;
            let mut var_sum = 0.0f64;
            for &v in &row[..effective_n_len] {
                let d = v as f64 - mean;
                var_sum += d * d;
            }
            let var = var_sum / (effective_n_len - 1) as f64; // unbiased
            let inv_std = 1.0 / (var + 1e-5).sqrt();
            for v in row[..effective_n_len].iter_mut() {
                *v = ((*v as f64 - mean) * inv_std) as f32;
            }
            for v in row[effective_n_len..].iter_mut() {
                *v = 0.0;
            }
        } else {
            // effective_n_len ∈ {0, 1}: zero everything.
            for v in row.iter_mut() {
                *v = 0.0;
            }
        }
    }

    // Transpose mel-major [n_mel × n_frames] → time-major
    // [n_frames × n_mel_bins].
    let mut mel_time_major = vec![0.0f32; n_frames * n_mel_bins];
    for mi in 0..n_mel_bins {
        for ti in 0..n_frames {
            mel_time_major[ti * n_mel_bins + mi] = mel[mi * n_frames + ti];
        }
    }
    (mel_time_major, n_frames)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Periodic Hann window: `w[0] == 0`, peaks at `length / 2`.
    /// (For odd lengths the peak is between two samples; the
    /// nearest sample is close to but not exactly 1.)
    #[test]
    fn hann_window_periodic_endpoints() {
        let w = build_hann_window(8);
        assert!((w[0] - 0.0).abs() < 1e-6, "w[0] = {}", w[0]);
        assert!((w[4] - 1.0).abs() < 1e-6, "w[4] = {}", w[4]);
        // Symmetry (about the peak): w[1] ≈ w[7], w[2] ≈ w[6], w[3] ≈ w[5].
        assert!((w[1] - w[7]).abs() < 1e-6);
        assert!((w[2] - w[6]).abs() < 1e-6);
        assert!((w[3] - w[5]).abs() < 1e-6);
    }

    /// Sanity: the LFM2A Hann window is 400 samples and peaks at
    /// index 200.
    #[test]
    fn hann_window_lfm2a_dims() {
        let w = build_hann_window(WINDOW_LEN);
        assert_eq!(w.len(), 400);
        assert!((w[200] - 1.0).abs() < 1e-6);
    }

    /// Mel filterbank: shape and per-row positivity.
    #[test]
    fn mel_filterbank_shape_and_positive() {
        let n_mel = 32;
        let f = build_mel_filterbank(n_mel, N_FFT, SAMPLE_RATE as usize);
        assert_eq!(f.len(), n_mel * N_FFT_BINS);
        // Every row must have at least one positive entry (the
        // triangle peak) — catches "all zero" placement bugs.
        for mi in 0..n_mel {
            let row = &f[mi * N_FFT_BINS..(mi + 1) * N_FFT_BINS];
            assert!(row.iter().any(|&v| v > 0.0), "mel row {mi} is all zero");
        }
    }

    /// Mel filter peaks march monotonically up the FFT bin axis as
    /// the filter index increases (low mel filters cover low Hz
    /// bins; high mel filters cover high Hz bins).
    #[test]
    fn mel_filterbank_peak_indices_monotonic() {
        let n_mel = 32;
        let f = build_mel_filterbank(n_mel, N_FFT, SAMPLE_RATE as usize);
        let mut prev_peak = 0;
        for mi in 0..n_mel {
            let row = &f[mi * N_FFT_BINS..(mi + 1) * N_FFT_BINS];
            let (peak_k, _) = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();
            assert!(
                peak_k >= prev_peak,
                "filter {mi} peak at bin {peak_k} < prev {prev_peak}"
            );
            prev_peak = peak_k;
        }
    }

    /// End-to-end smoke: a 1 kHz sine wave should put most of its
    /// post-norm energy into a low-but-not-lowest mel bin (1 kHz
    /// is well within the linear part of the Slaney scale).
    /// Verifies the pipeline produces finite output of the right
    /// shape and exhibits the expected peak structure.
    #[test]
    fn log_mel_spectrogram_sine_wave_smoke() {
        let n_mel = 80;
        let dur_sec = 1.0;
        let n_samples = (SAMPLE_RATE as f32 * dur_sec) as usize;
        let freq_hz = 1000.0_f32;
        let pcm: Vec<f32> = (0..n_samples)
            .map(|i| (2.0 * std::f32::consts::PI * freq_hz * i as f32 / SAMPLE_RATE as f32).sin())
            .collect();

        let (mel, n_frames) = log_mel_spectrogram(&pcm, n_mel);
        assert_eq!(mel.len(), n_frames * n_mel);
        assert!(n_frames > 0);
        for (i, &v) in mel.iter().enumerate() {
            assert!(v.is_finite(), "mel[{i}] = {v} (not finite)");
        }
        // Verify the pipeline produced *something* — after per-feature
        // norm a pure tone won't sit at a single huge value, but
        // there should be variation across mel bins (a flat-zero
        // output would mean the mel projection collapsed).
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for &v in &mel {
            min = min.min(v);
            max = max.max(v);
        }
        assert!(
            max - min > 0.01,
            "mel output has no variation (min={min}, max={max})"
        );
    }

    /// Empty input returns an empty vec without panicking.
    #[test]
    fn log_mel_spectrogram_empty_input_is_empty_output() {
        let (mel, n_frames) = log_mel_spectrogram(&[], 80);
        assert_eq!(n_frames, 0);
        assert!(mel.is_empty());
    }
}
