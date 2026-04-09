#![cfg(all(feature = "metal", target_os = "macos"))]

//! Performance regression tests for Metal decode and prefill.
//!
//! Run with: cargo test -p wick --release --features metal --test bench_perf -- --ignored --nocapture
//!
//! These tests print actual throughput and assert minimum floors.
//! Thresholds are set conservatively below measured baselines to avoid
//! flaky failures on slower machines while still catching major regressions.

use std::path::Path;
use std::time::Instant;

fn find_model(name: &str) -> Option<std::path::PathBuf> {
    let p = Path::new(env!("HOME"))
        .join(".leap/models")
        .join(name)
        .join(format!("{name}.gguf"));
    if p.exists() {
        Some(p)
    } else {
        eprintln!("model not found: {}, skipping", p.display());
        None
    }
}

fn bench_decode(model_path: &Path, n_tokens: usize, runs: usize) -> f64 {
    use wick::model::Model;
    use wick::model::metal_lfm2::MetalLfm2Model;

    let gguf = wick::gguf::GgufFile::open(model_path).unwrap();
    let model = MetalLfm2Model::from_gguf(gguf, model_path, 8192).unwrap();
    let cfg = model.config();
    let mut state = wick::kv_cache::InferenceState::from_config(cfg);

    // Warmup
    let _ = model.forward(&[1], 0, &mut state);

    let mut tok_per_sec = Vec::new();
    for _ in 0..runs {
        state = wick::kv_cache::InferenceState::from_config(cfg);
        // Prefill a short prompt
        let _ = model.forward(&[1], 0, &mut state);

        let t0 = Instant::now();
        for pos in 1..n_tokens {
            let _ = model.forward(&[1], pos, &mut state);
        }
        let elapsed = t0.elapsed().as_secs_f64();
        let tps = (n_tokens - 1) as f64 / elapsed;
        tok_per_sec.push(tps);
    }

    tok_per_sec.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = tok_per_sec[tok_per_sec.len() / 2];
    eprintln!(
        "  decode n={n_tokens}: {median:.1} tok/s (runs: {:?})",
        tok_per_sec
            .iter()
            .map(|v| format!("{v:.1}"))
            .collect::<Vec<_>>()
    );
    median
}

fn bench_prefill(model_path: &Path, n_tokens: usize, runs: usize) -> f64 {
    use wick::model::Model;
    use wick::model::metal_lfm2::MetalLfm2Model;

    let gguf = wick::gguf::GgufFile::open(model_path).unwrap();
    let model = MetalLfm2Model::from_gguf(gguf, model_path, 8192).unwrap();
    let cfg = model.config();

    let tokens: Vec<u32> = (0..n_tokens as u32).map(|i| i % 1000 + 1).collect();

    // Warmup
    let mut state = wick::kv_cache::InferenceState::from_config(cfg);
    let _ = model.forward_prefill(&tokens, 0, &mut state);

    let mut tok_per_sec = Vec::new();
    for _ in 0..runs {
        let mut state = wick::kv_cache::InferenceState::from_config(cfg);
        let t0 = Instant::now();
        let _ = model.forward_prefill(&tokens, 0, &mut state);
        let elapsed = t0.elapsed().as_secs_f64();
        let tps = n_tokens as f64 / elapsed;
        tok_per_sec.push(tps);
    }

    tok_per_sec.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = tok_per_sec[tok_per_sec.len() / 2];
    eprintln!(
        "  prefill n={n_tokens}: {median:.1} tok/s (runs: {:?})",
        tok_per_sec
            .iter()
            .map(|v| format!("{v:.1}"))
            .collect::<Vec<_>>()
    );
    median
}

#[test]
#[ignore]
fn test_metal_decode_throughput() {
    let Some(path) = find_model("LFM2.5-VL-1.6B-Q4_0") else {
        return;
    };
    eprintln!("=== Metal decode throughput ===");
    let tps = bench_decode(&path, 64, 3);
    // Floor set conservatively below measured baseline (~240 standalone,
    // ~230 sequential) to account for thermal throttling in sequential runs.
    assert!(tps > 150.0, "Metal decode n=64: {tps:.1} tok/s < 150 floor");
}

#[test]
#[ignore]
fn test_metal_decode_long_context() {
    let Some(path) = find_model("LFM2.5-VL-1.6B-Q4_0") else {
        return;
    };
    eprintln!("=== Metal decode long context ===");
    let tps = bench_decode(&path, 512, 2);
    assert!(
        tps > 120.0,
        "Metal decode n=512: {tps:.1} tok/s < 120 floor"
    );
}

#[test]
#[ignore]
fn test_metal_prefill_throughput() {
    let Some(path) = find_model("LFM2.5-VL-1.6B-Q4_0") else {
        return;
    };
    eprintln!("=== Metal prefill throughput ===");
    let tps = bench_prefill(&path, 128, 3);
    // Floor set conservatively below measured baseline (~2900 standalone,
    // ~2400 sequential) to catch major regressions while allowing thermal variance.
    assert!(
        tps > 1500.0,
        "Metal prefill n=128: {tps:.1} tok/s < 1500 floor"
    );
}

#[test]
#[ignore]
fn test_metal_prefill_long() {
    let Some(path) = find_model("LFM2.5-VL-1.6B-Q4_0") else {
        return;
    };
    eprintln!("=== Metal prefill long ===");
    let tps = bench_prefill(&path, 512, 5);
    assert!(
        tps > 1000.0,
        "Metal prefill n=512: {tps:.1} tok/s < 1000 floor"
    );
}

/// Profile prefill scaling across token counts. Not a regression test —
/// prints timing breakdown for analysis.
#[test]
#[ignore]
fn test_prefill_scaling_profile() {
    use wick::model::Model;
    use wick::model::metal_lfm2::MetalLfm2Model;

    let Some(path) = find_model("LFM2.5-VL-1.6B-Q4_0") else {
        return;
    };
    eprintln!("=== Prefill scaling profile ===");
    let gguf = wick::gguf::GgufFile::open(&path).unwrap();
    let model = MetalLfm2Model::from_gguf(gguf, &path, 8192).unwrap();
    let cfg = model.config();

    for &n in &[1, 4, 8, 16, 32, 64, 128, 256, 512] {
        let tokens: Vec<u32> = (0..n as u32).map(|i| i % 1000 + 1).collect();
        // warmup
        let mut state = wick::kv_cache::InferenceState::from_config(cfg);
        let _ = model.forward_prefill(&tokens, 0, &mut state);
        // measure best of 3
        let mut best = f64::MAX;
        for _ in 0..3 {
            let mut state = wick::kv_cache::InferenceState::from_config(cfg);
            let t0 = Instant::now();
            let _ = model.forward_prefill(&tokens, 0, &mut state);
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            if ms < best {
                best = ms;
            }
        }
        let tps = n as f64 / (best / 1000.0);
        let per_tok_us = best * 1000.0 / n as f64;
        eprintln!("  n={n:>4}: {best:>7.2} ms  {tps:>7.0} tok/s  {per_tok_us:>6.1} µs/tok");
    }
}

/// Assert GPU memory stays within budget per model size.
/// Catches accidental buffer over-allocation.
#[test]
#[ignore]
fn test_gpu_memory_budget() {
    use wick::model::Model;
    use wick::model::metal_lfm2::MetalLfm2Model;

    // 1.6B Q4_0 budget.
    if let Some(path) = find_model("LFM2.5-VL-1.6B-Q4_0") {
        let gguf = wick::gguf::GgufFile::open(&path).unwrap();
        let model = MetalLfm2Model::from_gguf(gguf, &path, 4096).unwrap();
        let gpu_mb = model.gpu_memory_bytes() as f64 / 1_048_576.0;
        eprintln!("1.6B Q4_0 GPU memory: {gpu_mb:.0} MB");
        assert!(
            gpu_mb < 800.0,
            "1.6B Q4_0 GPU memory {gpu_mb:.0} MB > 800 MB budget"
        );
    }

    // 450M Q4_0 budget.
    if let Some(path) = find_model("LFM2.5-VL-450M-Q4_0") {
        let gguf = wick::gguf::GgufFile::open(&path).unwrap();
        let model = MetalLfm2Model::from_gguf(gguf, &path, 4096).unwrap();
        let gpu_mb = model.gpu_memory_bytes() as f64 / 1_048_576.0;
        eprintln!("450M Q4_0 GPU memory: {gpu_mb:.0} MB");
        assert!(
            gpu_mb < 350.0,
            "450M Q4_0 GPU memory {gpu_mb:.0} MB > 350 MB budget"
        );
    }
}

/// Verify that forward_prefill produces the same last-token logits as
/// sequential forward() calls. This catches any offset or accumulation bug
/// in the batched prefill path.
#[test]
#[ignore]
fn test_batched_prefill_logits_match_sequential() {
    use wick::model::Model;
    use wick::model::metal_lfm2::MetalLfm2Model;

    let Some(path) = find_model("LFM2.5-VL-1.6B-Q4_0") else {
        return;
    };
    eprintln!("=== Prefill logit parity vs sequential ===");

    let n_tokens = 32;
    let tokens: Vec<u32> = (0..n_tokens as u32).map(|i| i % 1000 + 1).collect();

    // Sequential: forward() one token at a time.
    let gguf_seq = wick::gguf::GgufFile::open(&path).unwrap();
    let model_seq = MetalLfm2Model::from_gguf(gguf_seq, &path, 8192).unwrap();
    let cfg = model_seq.config();
    let mut state_seq = wick::kv_cache::InferenceState::from_config(cfg);
    let mut logits_seq = Vec::new();
    for (i, &tok) in tokens.iter().enumerate() {
        logits_seq = model_seq.forward(&[tok], i, &mut state_seq);
    }

    // Prefill: forward_prefill() all tokens at once.
    let gguf_pf = wick::gguf::GgufFile::open(&path).unwrap();
    let model_pf = MetalLfm2Model::from_gguf(gguf_pf, &path, 8192).unwrap();
    let mut state_pf = wick::kv_cache::InferenceState::from_config(cfg);
    let logits_pf = model_pf.forward_prefill(&tokens, 0, &mut state_pf);

    // Compare: cosine similarity and max abs diff.
    assert_eq!(
        logits_seq.len(),
        logits_pf.len(),
        "logit vector length mismatch"
    );
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    let mut max_abs = 0.0f32;
    for i in 0..logits_seq.len() {
        let a = logits_seq[i] as f64;
        let b = logits_pf[i] as f64;
        dot += a * b;
        norm_a += a * a;
        norm_b += b * b;
        let d = (logits_seq[i] - logits_pf[i]).abs();
        if d > max_abs {
            max_abs = d;
        }
    }
    let cosine = dot / (norm_a.sqrt() * norm_b.sqrt());
    eprintln!("  cosine: {cosine:.6}, max_abs_diff: {max_abs:.6}");

    // Top-5 comparison.
    let mut idx_seq: Vec<usize> = (0..logits_seq.len()).collect();
    idx_seq.sort_by(|&a, &b| logits_seq[b].partial_cmp(&logits_seq[a]).unwrap());
    let mut idx_pf: Vec<usize> = (0..logits_pf.len()).collect();
    idx_pf.sort_by(|&a, &b| logits_pf[b].partial_cmp(&logits_pf[a]).unwrap());
    eprintln!("  seq top5: {:?}", &idx_seq[..5]);
    eprintln!("  pf  top5: {:?}", &idx_pf[..5]);

    assert!(
        cosine > 0.999,
        "prefill vs sequential cosine {cosine:.6} < 0.999"
    );
    assert!(
        max_abs < 0.05,
        "prefill vs sequential max_abs_diff {max_abs:.6} > 0.05"
    );
}

/// Compare GEMM vs batch GEMV crossover point.
#[test]
#[ignore]
fn test_gemm_crossover() {
    use wick::model::Model;
    use wick::model::metal_lfm2::MetalLfm2Model;

    let Some(path) = find_model("LFM2.5-VL-1.6B-Q4_0") else {
        return;
    };
    eprintln!("=== GEMM crossover ===");
    let gguf = wick::gguf::GgufFile::open(&path).unwrap();
    let model = MetalLfm2Model::from_gguf(gguf, &path, 8192).unwrap();
    let cfg = model.config();

    for &n in &[4, 8, 12, 16, 24, 32] {
        let tokens: Vec<u32> = (0..n as u32).map(|i| i % 1000 + 1).collect();
        let mut state = wick::kv_cache::InferenceState::from_config(cfg);
        let _ = model.forward_prefill(&tokens, 0, &mut state);
        let mut best = f64::MAX;
        for _ in 0..5 {
            let mut state = wick::kv_cache::InferenceState::from_config(cfg);
            let t0 = Instant::now();
            let _ = model.forward_prefill(&tokens, 0, &mut state);
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            if ms < best {
                best = ms;
            }
        }
        eprintln!(
            "  n={n:>3}: {best:>7.2} ms  {:>7.0} tok/s",
            n as f64 / (best / 1000.0)
        );
    }
}

/// Measure raw GEMM throughput for a single weight matrix.
#[test]
#[ignore]
fn test_gemm_microbench() {
    use wick::model::Model;
    use wick::model::metal_lfm2::MetalLfm2Model;

    let Some(path) = find_model("LFM2.5-VL-1.6B-Q4_0") else {
        return;
    };
    let gguf = wick::gguf::GgufFile::open(&path).unwrap();
    let model = MetalLfm2Model::from_gguf(gguf, &path, 8192).unwrap();
    let cfg = model.config();

    // Run full prefill to warm up, then measure with different n
    for &n in &[32, 64, 128, 256] {
        let tokens: Vec<u32> = (0..n as u32).map(|i| i % 1000 + 1).collect();
        let mut state = wick::kv_cache::InferenceState::from_config(cfg);
        let _ = model.forward_prefill(&tokens, 0, &mut state);

        let mut best = f64::MAX;
        for _ in 0..5 {
            let mut state = wick::kv_cache::InferenceState::from_config(cfg);
            let t0 = Instant::now();
            let _ = model.forward_prefill(&tokens, 0, &mut state);
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            if ms < best {
                best = ms;
            }
        }
        // Estimate GEMM-only time: total - fixed overhead (~6ms)
        let gemm_est = best - 6.0;
        // Total weight bytes: sum all Q4_0 weight matrices
        // 10 conv layers: in_proj(2048×6144) + out_proj(2048×2048) + FFN(2048×8192×2 + 8192×2048)
        // 6 attn layers: Q(2048×2048) + K(2048×512) + V(2048×512) + O(2048×2048) + FFN same
        // Approximate: 593 MB total weights
        let weight_mb = 593.0;
        let bw = weight_mb / gemm_est * 1000.0;
        eprintln!(
            "  n={n:>3}: {best:>6.1}ms (est GEMM: {gemm_est:>5.1}ms) eff BW: {bw:.0} MB/s  {:.0} tok/s",
            n as f64 / (best / 1000.0)
        );
    }
}

/// Isolate GEMM kernel performance by running just the prefill's GEMM
/// dispatches (skip conv1d, attention, etc.) to measure raw GEMM throughput.
#[test]
#[ignore]
fn test_gemm_isolation() {
    use wick::model::Model;
    use wick::model::metal_lfm2::MetalLfm2Model;

    let Some(path) = find_model("LFM2.5-VL-1.6B-Q4_0") else {
        return;
    };
    let gguf = wick::gguf::GgufFile::open(&path).unwrap();
    let model = MetalLfm2Model::from_gguf(gguf, &path, 8192).unwrap();
    let cfg = model.config();

    // Measure full forward_prefill
    let n = 128usize;
    let tokens: Vec<u32> = (0..n as u32).map(|i| i % 1000 + 1).collect();

    // warmup
    let mut state = wick::kv_cache::InferenceState::from_config(cfg);
    let _ = model.forward_prefill(&tokens, 0, &mut state);

    // measure
    let mut times = Vec::new();
    for _ in 0..5 {
        let mut state = wick::kv_cache::InferenceState::from_config(cfg);
        let t0 = Instant::now();
        let _ = model.forward_prefill(&tokens, 0, &mut state);
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let best = times[0];
    let median = times[times.len() / 2];

    // Compute total Q4_0 weight bytes
    let mut total_weight_bytes = 0u64;
    for tensor_line in [
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
        "blk.0.shortconv.in_proj.weight",
        "blk.0.shortconv.out_proj.weight",
    ] {
        // Approximate: 16 layers, each has these weights
        // FFN: gate(2048×8192) + up(2048×8192) + down(8192×2048) = 3 × 2048×8192/32×18
        // Conv: in_proj(2048×6144) + out_proj(2048×2048)
    }
    // Manual calculation for 1.6B:
    // 10 conv layers: (2048*6144 + 2048*2048)/32*18 = (12.6M + 4.2M)/32*18 = 9.45M bytes per layer
    // 6 attn layers: (2048*2048*3 + 2048*512*2 + 2048*2048)/32*18 per layer
    // 16 FFN: (2048*8192*2 + 8192*2048)/32*18 per layer
    let conv_weight = (2048 * 6144 + 2048 * 2048) / 32 * 18;
    let attn_weight = (2048 * 2048 * 3 + 2048 * 512 * 2 + 2048 * 2048) / 32 * 18;
    let ffn_weight = (2048 * 8192 * 2 + 8192 * 2048) / 32 * 18;
    total_weight_bytes = (10 * conv_weight + 6 * attn_weight + 16 * ffn_weight) as u64;
    let weight_mb = total_weight_bytes as f64 / 1_048_576.0;

    // Effective bandwidth = weight_bytes / time (weights read once for all n tokens)
    let eff_bw_gbs = weight_mb / 1024.0 / (best / 1000.0);
    // Compute throughput = total_flops / time
    // Each Q4_0 element = 2 FLOPs (mul + add) per token
    let total_elements = total_weight_bytes as f64 * 32.0 / 18.0;
    let total_gflops = total_elements * n as f64 * 2.0 / 1e9;
    let tflops = total_gflops / (best / 1000.0) / 1000.0;

    eprintln!("=== GEMM Isolation ===");
    eprintln!(
        "  n=128 prefill: best={best:.1}ms median={median:.1}ms ({:.0} tok/s)",
        n as f64 / (best / 1000.0)
    );
    eprintln!("  Weight data: {weight_mb:.0} MB");
    eprintln!("  Effective BW: {eff_bw_gbs:.1} GB/s (weight read once)");
    eprintln!("  Compute: {total_gflops:.0} GFLOPS in {best:.1}ms = {tflops:.2} TFLOPS");
    eprintln!("  Fixed overhead est: ~6ms → GEMM est: {:.1}ms", best - 6.0);

    // Compare with llama.cpp
    let llama_ms = 128.0 / 3331.0 * 1000.0;
    eprintln!(
        "  llama.cpp total: {llama_ms:.1}ms → GEMM est: {:.1}ms",
        llama_ms - 6.0
    );
    eprintln!("  GEMM ratio: {:.2}×", (best - 6.0) / (llama_ms - 6.0));
}

/// Per-phase GPU timing breakdown of prefill. Commits/waits after each phase
/// to measure wall-clock time. Much slower than production — for analysis only.
#[test]
#[ignore]
fn test_prefill_phase_profile() {
    use std::collections::HashMap;
    use wick::model::Model;
    use wick::model::metal_lfm2::MetalLfm2Model;

    let Some(path) = find_model("LFM2.5-VL-1.6B-Q4_0") else {
        return;
    };
    let gguf = wick::gguf::GgufFile::open(&path).unwrap();
    let model = MetalLfm2Model::from_gguf(gguf, &path, 8192).unwrap();
    let cfg = model.config();

    let n = 128;
    let tokens: Vec<u32> = (0..n as u32).map(|i| i % 1000 + 1).collect();

    // Warmup.
    let mut state = wick::kv_cache::InferenceState::from_config(cfg);
    let _ = model.forward_prefill_profiled(&tokens, 0, &mut state);

    // Profiled run.
    let mut state = wick::kv_cache::InferenceState::from_config(cfg);
    let timings = model.forward_prefill_profiled(&tokens, 0, &mut state);

    // Aggregate by category (strip layer number).
    let mut by_cat: HashMap<String, (f64, usize)> = HashMap::new();
    for (name, us) in &timings {
        // "L0_conv_inproj" → "conv_inproj"
        let cat = name.split('_').skip(1).collect::<Vec<_>>().join("_");
        let entry = by_cat.entry(cat).or_insert((0.0, 0));
        entry.0 += us;
        entry.1 += 1;
    }

    let total_us: f64 = timings.iter().map(|(_, us)| us).sum();
    eprintln!("=== Prefill Phase Profile (n={n}) ===");
    eprintln!(
        "  Total: {:.1} ms ({:.0} tok/s)",
        total_us / 1000.0,
        n as f64 / (total_us / 1e6)
    );
    eprintln!();

    // Sort by total time descending.
    let mut cats: Vec<_> = by_cat.into_iter().collect();
    cats.sort_by(|a, b| b.1.0.partial_cmp(&a.1.0).unwrap());

    eprintln!(
        "  {:30} {:>8} {:>6} {:>6}",
        "Phase", "Total µs", "Count", "%"
    );
    eprintln!(
        "  {:30} {:>8} {:>6} {:>6}",
        "-----", "--------", "-----", "--"
    );
    for (cat, (total, count)) in &cats {
        let pct = total / total_us * 100.0;
        eprintln!("  {:30} {:>8.0} {:>6} {:>5.1}%", cat, total, count, pct);
    }
}
