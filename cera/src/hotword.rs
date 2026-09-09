//! Generic native Keyword Spotting (KWS) engine.
//!
//! Provides real-time streaming and windowed keyword detection using GGUF-packaged
//! KWS models in pure Rust with zero external runtime dependencies.
//!
//! # Example
//! ```ignore
//! use cera::hotword::{HotwordConfig, HotwordDetector, HotwordIterator};
//!
//! let detector = HotwordDetector::from_file("models/hey_liquid.gguf")?;
//! let mut iterator = HotwordIterator::new(detector, None, HotwordConfig::default());
//!
//! // Audio chunk from microphone (16 kHz mono PCM)
//! let chunk = [0.0f32; 640];
//! if let Some(event) = iterator.process_chunk(&chunk)? {
//!     println!("Detected keyword: {}", event.keyword);
//! }
//! # Ok::<(), anyhow::Error>(())
//! ```

#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use rustfft::FftPlanner;
use rustfft::num_complex::Complex32;
use serde::{Deserialize, Serialize};

use crate::gguf::GgufFile;
use crate::tensor::{DType, Tensor};
use crate::vad::SileroVad;

/// Configuration options for keyword spotting.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HotwordConfig {
    /// Activation probability threshold (default: 0.75).
    pub threshold: f32,
    /// Post-detection debounce cooldown in milliseconds (default: 2000 ms).
    pub cooldown_ms: usize,
    /// Evaluation step interval in milliseconds (default: 80 ms).
    pub step_ms: usize,
    /// Sliding window length in milliseconds (default: 1200 ms).
    pub window_ms: usize,
    /// Audio pre-roll margin in milliseconds to preserve before command (default: 150 ms).
    pub pre_roll_ms: usize,
    /// VAD speech probability threshold for gating KWS (default: 0.5).
    pub vad_threshold: f32,
}

impl Default for HotwordConfig {
    fn default() -> Self {
        Self {
            threshold: 0.75,
            cooldown_ms: 2000,
            step_ms: 80,
            window_ms: 1200,
            pre_roll_ms: 150,
            vad_threshold: 0.5,
        }
    }
}

/// Confidence score for a specific keyword candidate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HotwordScore {
    /// Target keyword string.
    pub keyword: String,
    /// Model activation probability between 0.0 and 1.0.
    pub score: f32,
}

/// Event emitted when a keyword spotting threshold is crossed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HotwordEvent {
    /// The matched keyword string.
    pub keyword: String,
    /// Exact audio stream sample index where the keyword completed.
    pub sample_offset: u64,
    /// Audio stream sample index including pre-roll safety margin for downstream ASR.
    pub command_start_sample: u64,
    /// Timestamp in milliseconds from stream origin where keyword completed.
    pub timestamp_ms: f32,
    /// Model confidence probability (0.0 to 1.0).
    pub confidence: f32,
}

// ── Mel-Spectrogram Front-End ─────────────────────────────────────────────────

fn hz_to_mel(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f64) -> f64 {
    700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0)
}

/// Acoustic feature extractor computing log-mel spectrogram frames.
pub struct LogMelFrontEnd {
    fft: Arc<dyn rustfft::Fft<f32>>,
    hann_window: Vec<f32>,
    mel_filterbank: Vec<f32>,
    fft_scratch: Vec<Complex32>,
    sample_rate: usize,
    window_samples: usize,
    hop_samples: usize,
    fft_size: usize,
    mel_bins: usize,
}

impl LogMelFrontEnd {
    /// Construct a new log-mel front-end with HTK filterbank.
    pub fn new(
        sample_rate: usize,
        window_samples: usize,
        hop_samples: usize,
        fft_size: usize,
        mel_bins: usize,
    ) -> Result<Self> {
        ensure!(mel_bins > 0, "mel_bins must be > 0");
        ensure!(fft_size > 0, "fft_size must be > 0");
        ensure!(
            window_samples <= fft_size,
            "window_samples must be <= fft_size"
        );
        ensure!(hop_samples > 0, "hop_samples must be > 0");
        ensure!(sample_rate > 0, "sample_rate must be > 0");

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(fft_size);

        // Symmetric Hann window
        let mut hann_window = Vec::with_capacity(window_samples);
        for n in 0..window_samples {
            let val =
                0.5 - 0.5 * ((2.0 * std::f64::consts::PI * n as f64) / window_samples as f64).cos();
            hann_window.push(val as f32);
        }

        // Triangular HTK Mel filterbank: 32 bins from 80 Hz to 7600 Hz
        let n_fft_bins = fft_size / 2 + 1;
        let min_mel = hz_to_mel(80.0);
        let max_mel = hz_to_mel(7600.0);
        let num_points = mel_bins + 2;

        let mut mel_points = Vec::with_capacity(num_points);
        for i in 0..num_points {
            let mel = min_mel + (max_mel - min_mel) * (i as f64 / (num_points - 1) as f64);
            mel_points.push(mel);
        }

        let mut bin_points = Vec::with_capacity(num_points);
        for &m in &mel_points {
            let hz = mel_to_hz(m);
            let b = ((fft_size as f64 + 1.0) * hz / sample_rate as f64).floor() as usize;
            bin_points.push(b);
        }

        let mut mel_filterbank = vec![0.0f32; mel_bins * n_fft_bins];
        for m in 1..=mel_bins {
            let f_m_minus = bin_points[m - 1];
            let f_m = bin_points[m];
            let f_m_plus = bin_points[m + 1];

            let denom_left = (f_m.saturating_sub(f_m_minus)).max(1) as f32;
            for k in f_m_minus..f_m.min(n_fft_bins) {
                mel_filterbank[(m - 1) * n_fft_bins + k] = (k - f_m_minus) as f32 / denom_left;
            }

            let denom_right = (f_m_plus.saturating_sub(f_m)).max(1) as f32;
            for k in f_m..f_m_plus.min(n_fft_bins) {
                mel_filterbank[(m - 1) * n_fft_bins + k] = (f_m_plus - k) as f32 / denom_right;
            }
        }

        Ok(Self {
            fft,
            hann_window,
            mel_filterbank,
            fft_scratch: vec![Complex32::new(0.0, 0.0); fft_size],
            sample_rate,
            window_samples,
            hop_samples,
            fft_size,
            mel_bins,
        })
    }

    /// Expected audio sample rate in Hz.
    pub fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    /// Number of output mel frames for an audio window of `audio_len` samples.
    pub fn num_frames(&self, audio_len: usize) -> usize {
        if audio_len < self.window_samples {
            0
        } else {
            (audio_len - self.window_samples) / self.hop_samples + 1
        }
    }

    /// Extract log-mel frames into a flat buffer of shape `[mel_bins, num_frames]`.
    pub fn extract(&mut self, audio: &[f32], output: &mut [f32]) {
        let num_frames = self.num_frames(audio.len());
        if output.len() < self.mel_bins * num_frames {
            debug_assert!(false, "output buffer too small for extracted mel frames");
            return;
        }

        let n_fft_bins = self.fft_size / 2 + 1;

        for i in 0..num_frames {
            let start = i * self.hop_samples;
            for j in 0..self.window_samples {
                self.fft_scratch[j] = Complex32::new(audio[start + j] * self.hann_window[j], 0.0);
            }
            self.fft_scratch[self.window_samples..self.fft_size].fill(Complex32::new(0.0, 0.0));

            self.fft.process(&mut self.fft_scratch);

            for m in 0..self.mel_bins {
                let filter_row = &self.mel_filterbank[m * n_fft_bins..(m + 1) * n_fft_bins];
                let mut energy = 0.0f32;
                for (&filter_coeff, complex) in
                    filter_row.iter().zip(&self.fft_scratch[..n_fft_bins])
                {
                    energy += filter_coeff * complex.norm_sqr();
                }
                output[m * num_frames + i] = (1.0 + energy.max(0.0)).ln();
            }
        }
    }
}

// ── Model Weights & Tensor Representation ─────────────────────────────────────

struct HotwordWeights {
    conv0_w: Tensor,
    conv0_b: Tensor,
    conv1_w: Tensor,
    conv1_b: Tensor,
    conv2_w: Tensor,
    conv2_b: Tensor,
    conv3_w: Tensor,
    conv3_b: Tensor,
    dense1_w: Tensor,
    dense1_b: Tensor,
    dense2_w: Tensor,
    dense2_b: Tensor,

    keywords: Vec<String>,
    sample_rate: usize,
    window_samples: usize,
    hop_samples: usize,
    mel_bins: usize,
    mel_window_samples: usize,
    mel_hop_samples: usize,
    fft_size: usize,
    embedding_dim: usize,
    default_threshold: f32,
    cooldown_ms: usize,
    pre_roll_ms: usize,
}

impl HotwordWeights {
    fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let arch = gguf.architecture().unwrap_or_default();
        ensure!(
            arch == "kws",
            "unsupported GGUF architecture '{arch}', expected 'kws'"
        );

        fn get_f32_tensor(gguf: &GgufFile, name: &str, expected_len: usize) -> Result<Tensor> {
            let t = gguf
                .get_tensor(name)
                .with_context(|| format!("missing tensor '{name}'"))?;
            ensure!(
                t.dtype() == DType::F32,
                "tensor '{name}' expected F32 dtype, found {:?}",
                t.dtype()
            );
            ensure!(
                t.numel() == expected_len,
                "tensor '{name}' expected {} elements, found {}",
                expected_len,
                t.numel()
            );
            Ok(t)
        }

        let keywords: Vec<String> = gguf
            .get_string_array("kws.keywords")
            .map(|arr| arr.into_iter().map(ToString::to_string).collect())
            .unwrap_or_else(|| vec!["Hey Liquid".to_string()]);
        ensure!(
            !keywords.is_empty(),
            "model must define at least one keyword"
        );

        let sample_rate = gguf.get_u32("kws.sample_rate").unwrap_or(16000) as usize;
        let window_samples = gguf.get_u32("kws.window_samples").unwrap_or(19200) as usize;
        let hop_samples = gguf.get_u32("kws.hop_samples").unwrap_or(1280) as usize;
        let mel_bins = gguf.get_u32("kws.mel_bins").unwrap_or(32) as usize;
        let mel_window_samples = gguf.get_u32("kws.mel_window_samples").unwrap_or(400) as usize;
        let mel_hop_samples = gguf.get_u32("kws.mel_hop_samples").unwrap_or(160) as usize;
        let fft_size = gguf.get_u32("kws.fft_size").unwrap_or(512) as usize;
        let embedding_dim = gguf.get_u32("kws.embedding_dim").unwrap_or(64) as usize;
        let default_threshold = gguf.get_f32("kws.default_threshold").unwrap_or(0.75);
        let cooldown_ms = gguf.get_u32("kws.cooldown_ms").unwrap_or(2000) as usize;
        let pre_roll_ms = gguf.get_u32("kws.pre_roll_ms").unwrap_or(150) as usize;

        ensure!(sample_rate > 0, "kws.sample_rate must be > 0");
        ensure!(window_samples > 0, "kws.window_samples must be > 0");
        ensure!(hop_samples > 0, "kws.hop_samples must be > 0");
        ensure!(mel_bins > 0, "kws.mel_bins must be > 0");
        ensure!(mel_window_samples > 0, "kws.mel_window_samples must be > 0");
        ensure!(mel_hop_samples > 0, "kws.mel_hop_samples must be > 0");
        ensure!(fft_size > 0, "kws.fft_size must be > 0");
        ensure!(
            mel_window_samples <= fft_size,
            "kws.mel_window_samples must be <= kws.fft_size"
        );
        ensure!(
            window_samples >= mel_window_samples,
            "kws.window_samples must be >= kws.mel_window_samples"
        );
        ensure!(embedding_dim > 0, "kws.embedding_dim must be > 0");

        let num_keywords = keywords.len();

        Ok(Self {
            conv0_w: get_f32_tensor(gguf, "kws.backbone.conv0.weight", 64 * mel_bins * 3)?,
            conv0_b: get_f32_tensor(gguf, "kws.backbone.conv0.bias", 64)?,
            conv1_w: get_f32_tensor(gguf, "kws.backbone.conv1.weight", 64 * 64 * 3)?,
            conv1_b: get_f32_tensor(gguf, "kws.backbone.conv1.bias", 64)?,
            conv2_w: get_f32_tensor(gguf, "kws.backbone.conv2.weight", 64 * 64 * 3)?,
            conv2_b: get_f32_tensor(gguf, "kws.backbone.conv2.bias", 64)?,
            conv3_w: get_f32_tensor(gguf, "kws.backbone.conv3.weight", embedding_dim * 64 * 3)?,
            conv3_b: get_f32_tensor(gguf, "kws.backbone.conv3.bias", embedding_dim)?,
            dense1_w: get_f32_tensor(gguf, "kws.head.dense1.weight", 32 * embedding_dim)?,
            dense1_b: get_f32_tensor(gguf, "kws.head.dense1.bias", 32)?,
            dense2_w: get_f32_tensor(gguf, "kws.head.dense2.weight", num_keywords * 32)?,
            dense2_b: get_f32_tensor(gguf, "kws.head.dense2.bias", num_keywords)?,

            keywords,
            sample_rate,
            window_samples,
            hop_samples,
            mel_bins,
            mel_window_samples,
            mel_hop_samples,
            fft_size,
            embedding_dim,
            default_threshold,
            cooldown_ms,
            pre_roll_ms,
        })
    }
}

// ── Math & Neural Operations ──────────────────────────────────────────────────

#[inline(always)]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// 1D Convolution with stride 2, padding 1, and SiLU activation.
///
/// Input layout: `[in_c, in_len]` row-major.
/// Output layout: `[out_c, out_len]` row-major.
/// Weight layout: `[out_c, in_c, 3]` row-major.
#[inline]
fn conv1d_silu_s2(
    input: &[f32],
    output: &mut [f32],
    in_channels: usize,
    in_len: usize,
    out_channels: usize,
    weights: &[f32],
    bias: &[f32],
) {
    let out_len = (in_len.saturating_sub(1)) / 2 + 1;
    if in_len == 0
        || input.len() < in_channels * in_len
        || output.len() < out_channels * out_len
        || weights.len() < out_channels * in_channels * 3
        || bias.len() < out_channels
    {
        debug_assert!(false, "buffer dimension mismatch in conv1d_silu_s2");
        return;
    }

    for (out_c, &b) in bias.iter().enumerate().take(out_channels) {
        let w_out = &weights[out_c * in_channels * 3..(out_c + 1) * in_channels * 3];
        let out_row_start = out_c * out_len;
        let out_row = &mut output[out_row_start..out_row_start + out_len];
        out_row.fill(b);

        for in_c in 0..in_channels {
            let w_c = &w_out[in_c * 3..(in_c + 1) * 3];
            let in_row = &input[in_c * in_len..(in_c + 1) * in_len];

            for (out_idx, out_val) in out_row.iter_mut().enumerate() {
                let mut acc = 0.0f32;
                for (k, &w) in w_c.iter().enumerate() {
                    let in_pos = out_idx * 2 + k;
                    if in_pos > 0 && in_pos <= in_len {
                        acc += w * in_row[in_pos - 1];
                    }
                }
                *out_val += acc;
            }
        }

        for val in out_row.iter_mut() {
            *val = silu(*val);
        }
    }
}

// ── Hotword Detector ──────────────────────────────────────────────────────────

/// Stateful Keyword Spotting detector executing pure-Rust forward inference.
pub struct HotwordDetector {
    weights: HotwordWeights,
    front_end: LogMelFrontEnd,
    mel_scratch: Vec<f32>,
    conv0_scratch: Vec<f32>,
    conv1_scratch: Vec<f32>,
    conv2_scratch: Vec<f32>,
    conv3_scratch: Vec<f32>,
    emb_scratch: Vec<f32>,
    dense1_scratch: [f32; 32],
    scores_scratch: Vec<f32>,
}

impl HotwordDetector {
    /// Load a detector from a GGUF file path using memory mapping.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let gguf = GgufFile::open(path.as_ref())?;
        Self::from_gguf(&gguf)
    }

    /// Load a detector from in-memory GGUF bytes.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        let gguf = GgufFile::from_bytes(Arc::from(bytes.into_boxed_slice()))?;
        Self::from_gguf(&gguf)
    }

    /// Load from parsed `GgufFile`.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let weights = HotwordWeights::from_gguf(gguf)?;
        let front_end = LogMelFrontEnd::new(
            weights.sample_rate,
            weights.mel_window_samples,
            weights.mel_hop_samples,
            weights.fft_size,
            weights.mel_bins,
        )?;

        let num_frames = front_end.num_frames(weights.window_samples);
        ensure!(
            num_frames > 0,
            "window_samples {} too short for mel extraction",
            weights.window_samples
        );
        let l0 = (num_frames - 1) / 2 + 1;
        let l1 = (l0 - 1) / 2 + 1;
        let l2 = (l1 - 1) / 2 + 1;
        let l3 = (l2 - 1) / 2 + 1;
        ensure!(l3 > 0, "downsampled temporal dimension l3 must be > 0");

        let mel_scratch = vec![0.0f32; weights.mel_bins * num_frames];
        let conv0_scratch = vec![0.0f32; 64 * l0];
        let conv1_scratch = vec![0.0f32; 64 * l1];
        let conv2_scratch = vec![0.0f32; 64 * l2];
        let conv3_scratch = vec![0.0f32; weights.embedding_dim * l3];
        let emb_scratch = vec![0.0f32; weights.embedding_dim];
        let num_keywords = weights.keywords.len();

        Ok(Self {
            weights,
            front_end,
            mel_scratch,
            conv0_scratch,
            conv1_scratch,
            conv2_scratch,
            conv3_scratch,
            emb_scratch,
            dense1_scratch: [0.0f32; 32],
            scores_scratch: Vec::with_capacity(num_keywords),
        })
    }

    /// List of target keywords supported by this model.
    pub fn keywords(&self) -> &[String] {
        &self.weights.keywords
    }

    /// Access the underlying acoustic log-mel front-end.
    pub fn front_end(&self) -> &LogMelFrontEnd {
        &self.front_end
    }

    /// Access the underlying acoustic log-mel front-end mutably.
    pub fn front_end_mut(&mut self) -> &mut LogMelFrontEnd {
        &mut self.front_end
    }

    /// Get default configuration suggested by model metadata.
    pub fn default_config(&self) -> HotwordConfig {
        HotwordConfig {
            threshold: self.weights.default_threshold,
            cooldown_ms: self.weights.cooldown_ms,
            step_ms: (self.weights.hop_samples * 1000) / self.weights.sample_rate,
            window_ms: (self.weights.window_samples * 1000) / self.weights.sample_rate,
            pre_roll_ms: self.weights.pre_roll_ms,
            vad_threshold: 0.5,
        }
    }

    /// Process a full audio window and return probability scores for each keyword.
    pub fn process_window(&mut self, window: &[f32]) -> Result<&[f32]> {
        ensure!(
            window.len() == self.weights.window_samples,
            "expected window of {} samples, got {}",
            self.weights.window_samples,
            window.len()
        );

        let num_frames = self.front_end.num_frames(window.len());

        // 1. Extract log-mel spectrogram
        self.front_end.extract(window, &mut self.mel_scratch);

        // 2. Convolutional Backbone Forward Pass
        let l0 = (num_frames - 1) / 2 + 1;
        conv1d_silu_s2(
            &self.mel_scratch,
            &mut self.conv0_scratch,
            self.weights.mel_bins,
            num_frames,
            64,
            self.weights.conv0_w.as_f32_slice(),
            self.weights.conv0_b.as_f32_slice(),
        );

        let l1 = (l0 - 1) / 2 + 1;
        conv1d_silu_s2(
            &self.conv0_scratch,
            &mut self.conv1_scratch,
            64,
            l0,
            64,
            self.weights.conv1_w.as_f32_slice(),
            self.weights.conv1_b.as_f32_slice(),
        );

        let l2 = (l1 - 1) / 2 + 1;
        conv1d_silu_s2(
            &self.conv1_scratch,
            &mut self.conv2_scratch,
            64,
            l1,
            64,
            self.weights.conv2_w.as_f32_slice(),
            self.weights.conv2_b.as_f32_slice(),
        );

        let l3 = (l2 - 1) / 2 + 1;
        conv1d_silu_s2(
            &self.conv2_scratch,
            &mut self.conv3_scratch,
            64,
            l2,
            self.weights.embedding_dim,
            self.weights.conv3_w.as_f32_slice(),
            self.weights.conv3_b.as_f32_slice(),
        );

        // 3. Temporal Mean Pooling -> 64-dim embedding
        let inv_l3 = 1.0 / (l3 as f32);
        for c in 0..self.weights.embedding_dim {
            let row = &self.conv3_scratch[c * l3..(c + 1) * l3];
            let sum: f32 = row.iter().sum();
            self.emb_scratch[c] = sum * inv_l3;
        }

        // 4. Dense Head Layer 1: Linear 64 -> 32 + SiLU
        let d1_w = self.weights.dense1_w.as_f32_slice();
        let d1_b = self.weights.dense1_b.as_f32_slice();
        for j in 0..32 {
            let mut acc = d1_b[j];
            let row = &d1_w[j * self.weights.embedding_dim..(j + 1) * self.weights.embedding_dim];
            for (&w, &emb) in row.iter().zip(&self.emb_scratch) {
                acc += w * emb;
            }
            self.dense1_scratch[j] = silu(acc);
        }

        // 5. Dense Head Layer 2: Linear 32 -> num_keywords + Sigmoid
        let num_keywords = self.weights.keywords.len();
        let d2_w = self.weights.dense2_w.as_f32_slice();
        let d2_b = self.weights.dense2_b.as_f32_slice();
        self.scores_scratch.clear();

        for k in 0..num_keywords {
            let mut acc = d2_b[k];
            let row = &d2_w[k * 32..(k + 1) * 32];
            for (&w, &h) in row.iter().zip(&self.dense1_scratch) {
                acc += w * h;
            }
            self.scores_scratch.push(sigmoid(acc));
        }

        Ok(&self.scores_scratch)
    }
}

// ── Circular Audio Ring Buffer ────────────────────────────────────────────────

struct CircularBuffer {
    buffer: Vec<f32>,
    write_pos: usize,
    count: usize,
    capacity: usize,
}

impl CircularBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            buffer: vec![0.0f32; capacity],
            write_pos: 0,
            count: 0,
            capacity,
        }
    }

    fn push_slice(&mut self, slice: &[f32]) {
        let mut remain = slice;
        while !remain.is_empty() {
            let space = self.capacity - self.write_pos;
            let take = remain.len().min(space);
            let (head, tail) = remain.split_at(take);

            for (dst, &src) in self.buffer[self.write_pos..self.write_pos + take]
                .iter_mut()
                .zip(head)
            {
                *dst = if src.is_finite() { src } else { 0.0 };
            }

            self.write_pos = (self.write_pos + take) % self.capacity;
            remain = tail;
        }
        self.count = (self.count + slice.len()).min(self.capacity);
    }

    fn read_last(&self, n: usize, out: &mut [f32]) -> bool {
        if self.count < n || out.len() < n {
            return false;
        }

        let start = (self.write_pos + self.capacity - n) % self.capacity;
        if start + n <= self.capacity {
            out[..n].copy_from_slice(&self.buffer[start..start + n]);
        } else {
            let first = self.capacity - start;
            out[..first].copy_from_slice(&self.buffer[start..]);
            out[first..n].copy_from_slice(&self.buffer[..n - first]);
        }
        true
    }

    fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.write_pos = 0;
        self.count = 0;
    }
}

// ── Stateful Streaming Hotword Iterator ───────────────────────────────────────

const VAD_FRAME_SIZE: usize = 512;

/// Streaming Keyword Spotting manager with VAD gating and debounce state.
pub struct HotwordIterator {
    detector: HotwordDetector,
    vad: Option<SileroVad>,
    ring_buffer: CircularBuffer,
    window_buffer: Vec<f32>,
    vad_buffer: Vec<f32>,
    vad_processed_samples: u64,
    config: HotwordConfig,

    current_sample: u64,
    last_eval_sample: u64,
    last_vad_speech_sample: u64,
    cooldown_until_sample: u64,
}

impl HotwordIterator {
    /// Create a new streaming `HotwordIterator`.
    pub fn new(detector: HotwordDetector, vad: Option<SileroVad>, config: HotwordConfig) -> Self {
        let window_samples = detector.weights.window_samples;
        let ring_capacity = (window_samples * 2).max(24_000);

        Self {
            ring_buffer: CircularBuffer::new(ring_capacity),
            window_buffer: vec![0.0f32; window_samples],
            vad_buffer: Vec::with_capacity(1024),
            vad_processed_samples: 0,
            detector,
            vad,
            config,
            current_sample: 0,
            last_eval_sample: 0,
            last_vad_speech_sample: 0,
            cooldown_until_sample: 0,
        }
    }

    /// Reset stream state, circular ring buffer, and debounce timers.
    pub fn reset(&mut self) {
        self.ring_buffer.reset();
        self.window_buffer.fill(0.0);
        self.vad_buffer.clear();
        self.vad_processed_samples = 0;
        if let Some(vad) = &mut self.vad {
            vad.reset();
        }
        self.current_sample = 0;
        self.last_eval_sample = 0;
        self.last_vad_speech_sample = 0;
        self.cooldown_until_sample = 0;
    }

    /// Process a streaming chunk of audio samples and return a detection event if triggered.
    pub fn process_chunk(&mut self, chunk: &[f32]) -> Result<Option<HotwordEvent>> {
        if chunk.is_empty() {
            return Ok(None);
        }

        let hop_samples = self.detector.weights.hop_samples as u64;
        let mut remain = chunk;
        let mut detected_event = None;

        while !remain.is_empty() {
            let samples_since_last_eval = self.current_sample.saturating_sub(self.last_eval_sample);
            let needed_for_hop = if samples_since_last_eval >= hop_samples {
                hop_samples
            } else {
                hop_samples - samples_since_last_eval
            };
            let take_len = (remain.len() as u64).min(needed_for_hop) as usize;

            let (sub_chunk, rest) = remain.split_at(take_len);
            remain = rest;

            self.ring_buffer.push_slice(sub_chunk);
            self.current_sample += sub_chunk.len() as u64;

            // 1. Step Silero VAD if attached (VAD_FRAME_SIZE increments)
            if let Some(vad) = &mut self.vad {
                self.vad_buffer.extend_from_slice(sub_chunk);
                while self.vad_buffer.len() >= VAD_FRAME_SIZE {
                    let prob = vad.process_chunk(
                        &self.vad_buffer[..VAD_FRAME_SIZE],
                        crate::vad::VadSampleRate::Rate16kHz,
                    )?;
                    self.vad_processed_samples += VAD_FRAME_SIZE as u64;
                    if prob >= self.config.vad_threshold {
                        self.last_vad_speech_sample = self.vad_processed_samples;
                    }
                    self.vad_buffer.drain(..VAD_FRAME_SIZE);
                }
            }

            // 2. Evaluate KWS if hop interval has elapsed
            if self.current_sample.saturating_sub(self.last_eval_sample) < hop_samples {
                continue;
            }
            self.last_eval_sample = self.current_sample;

            // Gating: if VAD is active and reported no speech within the window, skip KWS
            let window_samples = self.detector.weights.window_samples as u64;
            if self.vad.is_some()
                && self
                    .vad_processed_samples
                    .saturating_sub(self.last_vad_speech_sample)
                    > window_samples
            {
                continue;
            }

            // 3. Extract 1200 ms audio window from ring buffer
            if !self.ring_buffer.read_last(
                self.detector.weights.window_samples,
                &mut self.window_buffer,
            ) {
                continue;
            }

            // 4. Run KWS inference
            let scores = self.detector.process_window(&self.window_buffer)?;

            // 5. Threshold & Debounce checks
            let triggered = if self.current_sample >= self.cooldown_until_sample {
                scores
                    .iter()
                    .enumerate()
                    .find(|&(_, score)| *score >= self.config.threshold)
                    .map(|(idx, &score)| (idx, score))
            } else {
                None
            };

            if let Some((idx, score)) = triggered {
                let sample_rate = self.detector.weights.sample_rate as u64;
                let cooldown_samples = (self.config.cooldown_ms as u64 * sample_rate) / 1000;
                let pre_roll_samples = (self.config.pre_roll_ms as u64 * sample_rate) / 1000;

                self.cooldown_until_sample = self.current_sample + cooldown_samples;
                let keyword = self.detector.weights.keywords[idx].clone();
                let sample_offset = self.current_sample;
                let command_start_sample = sample_offset.saturating_sub(pre_roll_samples);
                let timestamp_ms = (sample_offset as f64 * 1000.0 / sample_rate as f64) as f32;

                if detected_event.is_none() {
                    detected_event = Some(HotwordEvent {
                        keyword,
                        sample_offset,
                        command_start_sample,
                        timestamp_ms,
                        confidence: score,
                    });
                }
            }
        }

        Ok(detected_event)
    }
}

// ── Unit Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circular_buffer_wrap_and_read() {
        let mut cb = CircularBuffer::new(10);
        assert!(!cb.read_last(5, &mut [0.0; 5]));

        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        cb.push_slice(&data);

        let mut read = [0.0f32; 5];
        assert!(cb.read_last(5, &mut read));
        assert_eq!(read, [3.0, 4.0, 5.0, 6.0, 7.0]);

        let wrap_data = [8.0, 9.0, 10.0, 11.0, 12.0];
        cb.push_slice(&wrap_data);
        assert!(cb.read_last(5, &mut read));
        assert_eq!(read, [8.0, 9.0, 10.0, 11.0, 12.0]);
    }

    #[test]
    fn test_log_mel_front_end_dimensions() -> Result<()> {
        let mut front_end = LogMelFrontEnd::new(16000, 400, 160, 512, 32)?;
        let num_frames = front_end.num_frames(19200);
        assert_eq!(num_frames, 118);

        let audio = vec![0.0f32; 19200];
        let mut mel = vec![0.0f32; 32 * num_frames];
        front_end.extract(&audio, &mut mel);

        assert_eq!(mel.len(), 32 * 118);
        assert!(mel.iter().all(|&v| v == 0.0));
        Ok(())
    }

    #[test]
    fn test_empty_chunk_handling() -> Result<()> {
        let mut cb = CircularBuffer::new(10);
        cb.push_slice(&[]);
        assert_eq!(cb.count, 0);
        Ok(())
    }
}
