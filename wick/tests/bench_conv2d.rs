//! Per-layer timing for the LFM2A conv subsampling stem on the
//! `wick::backend::cpu::conv2d` kernel. Shapes match the real
//! `mmproj-LFM2.5-Audio-1.5B` GGUF tensors; mel-input dims match
//! the standard Whisper-style 80-bin × N-frame layout.
//!
//! Run with:
//!
//! ```bash
//! cargo test -p wick --release --test bench_conv2d -- --ignored --nocapture
//! ```
//!
//! Gated `#[ignore]` so the suite stays fast in default `cargo test`
//! runs. Prints median ms per layer + total stem ms across `RUNS`
//! iterations.
//!
//! Measured baseline on Apple M1 Max (release build), median of 5
//! runs after one warmup:
//!
//! | Build                       | 5s    | 30s    | vs original |
//! |-----------------------------|------:|-------:|------------:|
//! | naive 7-loop (original)     | 832 ms| 5439 ms|         1×  |
//! | + 1×1 pointwise → matmul    |  74 ms|  451 ms|        12×  |
//! | + im2col + matmul (regular) |  33 ms|  204 ms|        27×  |
//! | + depthwise im2col          |  27 ms|  163 ms|        33×  |
//! | + rayon on depthwise        |  21 ms|  129 ms|        42×  |
//! | + BLAS (`--features blas`)  | 2.7 ms|   17 ms|       320×  |
//!
//! Default-feature bench (`cargo test --release ...`) lands at
//! ~129 ms / 30s; the BLAS build (`--features blas`) drops that
//! to ~17 ms via Apple Accelerate's AMX gemm.

use std::time::Instant;
use wick::backend::cpu::conv2d;

const RUNS: usize = 5;

/// One conv2d layer fixture: descriptor, weight buffer, bias, and
/// the precomputed output dims. Initialized once and reused across
/// timing runs.
struct Layer {
    name: &'static str,
    in_ch: usize,
    out_ch: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    groups: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl Layer {
    #[allow(clippy::too_many_arguments)]
    fn new(
        name: &'static str,
        in_ch: usize,
        out_ch: usize,
        kh: usize,
        kw: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
        groups: usize,
    ) -> Self {
        let in_per_group = in_ch / groups;
        let weight_len = out_ch * in_per_group * kh * kw;
        // Deterministic small-magnitude weights — keeps the
        // multiplications in normal-number range without depending
        // on a rand crate.
        let weight: Vec<f32> = (0..weight_len)
            .map(|i| 0.01 + ((i % 17) as f32) * 0.005)
            .collect();
        let bias: Vec<f32> = (0..out_ch).map(|i| 0.001 * i as f32).collect();
        Self {
            name,
            in_ch,
            out_ch,
            kh,
            kw,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            groups,
            weight,
            bias,
        }
    }

    fn out_dims(&self, h_in: usize, w_in: usize) -> (usize, usize) {
        let h_out = (h_in + 2 * self.pad_h - self.kh) / self.stride_h + 1;
        let w_out = (w_in + 2 * self.pad_w - self.kw) / self.stride_w + 1;
        (h_out, w_out)
    }

    fn run(&self, input: &[f32], output: &mut [f32], h_in: usize, w_in: usize) {
        conv2d(
            input,
            &self.weight,
            Some(&self.bias),
            output,
            self.in_ch,
            self.out_ch,
            h_in,
            w_in,
            self.kh,
            self.kw,
            self.stride_h,
            self.stride_w,
            self.pad_h,
            self.pad_w,
            self.groups,
        );
    }
}

fn median_ms(samples: &mut [f64]) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

fn time_layer(layer: &Layer, input: &[f32], output: &mut [f32], h_in: usize, w_in: usize) -> f64 {
    let mut samples = Vec::with_capacity(RUNS);
    // Warmup pass — populates caches, JIT-warms the branch
    // predictor on the rather branchy padding logic.
    layer.run(input, output, h_in, w_in);
    for _ in 0..RUNS {
        let t0 = Instant::now();
        layer.run(input, output, h_in, w_in);
        samples.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    median_ms(&mut samples)
}

/// Walk one chunk through the full stem. Returns per-layer median
/// ms and the total stem latency. Mirrors `clip.cpp`'s LFM2A stem
/// (5 weighted conv layers; ReLUs between are excluded since they're
/// not the kernel under test).
fn bench_stem(label: &str, h_in_mel_frames: usize) -> (Vec<(&'static str, f64)>, f64) {
    let w_in_mel_bins: usize = 80;

    // Layer descriptors per `mmproj-LFM2.5-Audio-1.5B-Q4_0`.
    let layer0 = Layer::new("layer.0 (3x3 s2 p1, 1→256)", 1, 256, 3, 3, 2, 2, 1, 1, 1);
    let layer2 = Layer::new(
        "layer.2 (3x3 dw s2 p1, 256ch)",
        256,
        256,
        3,
        3,
        2,
        2,
        1,
        1,
        256,
    );
    let layer3 = Layer::new("layer.3 (1x1 pw, 256→256)", 256, 256, 1, 1, 1, 1, 0, 0, 1);
    let layer5 = Layer::new(
        "layer.5 (3x3 dw s2 p1, 256ch)",
        256,
        256,
        3,
        3,
        2,
        2,
        1,
        1,
        256,
    );
    let layer6 = Layer::new("layer.6 (1x1 pw, 256→256)", 256, 256, 1, 1, 1, 1, 0, 0, 1);

    // Flow input shapes through the stem.
    let (h0, w0) = (h_in_mel_frames, w_in_mel_bins);
    let in0: Vec<f32> = (0..h0 * w0).map(|i| (i % 13) as f32 * 0.1).collect();
    let (h1, w1) = layer0.out_dims(h0, w0);
    let mut out0 = vec![0.0f32; 256 * h1 * w1];

    let (h2, w2) = layer2.out_dims(h1, w1);
    let mut out2 = vec![0.0f32; 256 * h2 * w2];

    let (h3, w3) = layer3.out_dims(h2, w2);
    let mut out3 = vec![0.0f32; 256 * h3 * w3];

    let (h5, w5) = layer5.out_dims(h3, w3);
    let mut out5 = vec![0.0f32; 256 * h5 * w5];

    let (h6, w6) = layer6.out_dims(h5, w5);
    let mut out6 = vec![0.0f32; 256 * h6 * w6];

    eprintln!("\n── {label} (input {h0} × {w0}) ──");
    let mut per_layer = Vec::new();

    // Layer 0 — regular conv on 1-channel mel-spec.
    let t = time_layer(&layer0, &in0, &mut out0, h0, w0);
    eprintln!("  {} → {h1}×{w1}: {t:.3} ms", layer0.name);
    per_layer.push((layer0.name, t));

    // Layer 2 — depthwise.
    let t = time_layer(&layer2, &out0, &mut out2, h1, w1);
    eprintln!("  {} → {h2}×{w2}: {t:.3} ms", layer2.name);
    per_layer.push((layer2.name, t));

    // Layer 3 — pointwise.
    let t = time_layer(&layer3, &out2, &mut out3, h2, w2);
    eprintln!("  {} → {h3}×{w3}: {t:.3} ms", layer3.name);
    per_layer.push((layer3.name, t));

    // Layer 5 — depthwise.
    let t = time_layer(&layer5, &out3, &mut out5, h3, w3);
    eprintln!("  {} → {h5}×{w5}: {t:.3} ms", layer5.name);
    per_layer.push((layer5.name, t));

    // Layer 6 — pointwise.
    let t = time_layer(&layer6, &out5, &mut out6, h5, w5);
    eprintln!("  {} → {h6}×{w6}: {t:.3} ms", layer6.name);
    per_layer.push((layer6.name, t));

    let total: f64 = per_layer.iter().map(|(_, t)| *t).sum();
    eprintln!("  ── total stem: {total:.3} ms ──");
    (per_layer, total)
}

#[test]
#[ignore = "perf bench; run with: cargo test -p wick --release --test bench_conv2d -- --ignored --nocapture"]
fn conv2d_stem_bench() {
    eprintln!("\nLFM2A conv subsampling stem — conv2d kernel timing");
    eprintln!("  build: {} runs/layer (median reported)", RUNS);

    let (_, t_5s) = bench_stem("5s audio: 500 mel frames", 500);
    let (_, t_30s) = bench_stem("30s audio: 3000 mel frames", 3000);

    eprintln!("\n── Summary ──");
    eprintln!("  5s  audio: {t_5s:.3} ms / chunk");
    eprintln!("  30s audio: {t_30s:.3} ms / chunk");
}
