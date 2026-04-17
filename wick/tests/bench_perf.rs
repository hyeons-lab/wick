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
    let p = std::path::PathBuf::from(std::env::var("HOME").expect("HOME not set"))
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

    // Warmup with unique tokens (offset 9999 to avoid cache collisions with runs).
    let warmup_tokens: Vec<u32> = (0..n_tokens as u32).map(|i| i % 1000 + 9999).collect();
    let mut state = wick::kv_cache::InferenceState::from_config(cfg);
    let _ = model.forward_prefill(&warmup_tokens, 0, &mut state);

    let mut tok_per_sec = Vec::new();
    for run in 0..runs {
        // Unique tokens per run to avoid KV prefix cache hits.
        // Use run index in a way that changes the first token (cache key).
        let tokens: Vec<u32> = (0..n_tokens as u32)
            .map(|i| (i.wrapping_mul(7) + run as u32 * 3571 + 1) % 50000 + 1)
            .collect();
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
    // Manual calculation for 1.6B:
    // 10 conv layers: (2048*6144 + 2048*2048)/32*18 = (12.6M + 4.2M)/32*18 = 9.45M bytes per layer
    // 6 attn layers: (2048*2048*3 + 2048*512*2 + 2048*2048)/32*18 per layer
    // 16 FFN: (2048*8192*2 + 8192*2048)/32*18 per layer
    let conv_weight = (2048 * 6144 + 2048 * 2048) / 32 * 18;
    let attn_weight = (2048 * 2048 * 3 + 2048 * 512 * 2 + 2048 * 2048) / 32 * 18;
    let ffn_weight = (2048 * 8192 * 2 + 8192 * 2048) / 32 * 18;
    let total_weight_bytes = (10 * conv_weight + 6 * attn_weight + 16 * ffn_weight) as u64;
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

    let (total_us, cats) = aggregate_prefill_phases(&timings);
    eprintln!("=== Prefill Phase Profile (n={n}) ===");
    eprintln!(
        "  Total: {:.1} ms ({:.0} tok/s)",
        total_us / 1000.0,
        n as f64 / (total_us / 1e6)
    );
    eprintln!();
    eprintln!(
        "  {:30} {:>8} {:>6} {:>6}",
        "Phase", "Total µs", "Count", "%"
    );
    eprintln!(
        "  {:30} {:>8} {:>6} {:>6}",
        "-----", "--------", "-----", "--"
    );
    for (cat, total, count) in &cats {
        let pct = total / total_us * 100.0;
        eprintln!("  {:30} {:>8.0} {:>6} {:>5.1}%", cat, total, count, pct);
    }
}

/// Group per-layer-per-phase timings from `forward_prefill_profiled` into
/// per-category totals, stripping the `L{layer}_` prefix. Returns
/// `(total_us, Vec<(category, total_us, count)>)` sorted by total
/// descending.
///
/// Phase names without a `_` (e.g. the whole-model `"out"` epilogue)
/// are preserved as-is rather than collapsing to an empty string.
fn aggregate_prefill_phases(timings: &[(String, f64)]) -> (f64, Vec<(String, f64, usize)>) {
    use std::collections::HashMap;
    let mut by_cat: HashMap<String, (f64, usize)> = HashMap::new();
    for (name, us) in timings {
        let cat = match name.split_once('_') {
            Some((_prefix, rest)) => rest.to_string(),
            None => name.to_string(),
        };
        let entry = by_cat.entry(cat).or_insert((0.0, 0));
        entry.0 += us;
        entry.1 += 1;
    }
    let total_us: f64 = timings.iter().map(|(_, us)| us).sum();
    let mut cats: Vec<_> = by_cat
        .into_iter()
        .map(|(cat, (total, count))| (cat, total, count))
        .collect();
    cats.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    (total_us, cats)
}

/// Verify that KV prefix cache produces identical results on cache hit.
#[test]
#[ignore]
fn test_prefix_cache_correctness() {
    use wick::model::Model;
    use wick::model::metal_lfm2::MetalLfm2Model;

    let Some(path) = find_model("LFM2.5-VL-1.6B-Q4_0") else {
        return;
    };
    let gguf = wick::gguf::GgufFile::open(&path).unwrap();
    let model = MetalLfm2Model::from_gguf(gguf, &path, 8192).unwrap();
    let cfg = model.config();

    let tokens: Vec<u32> = (0..64u32).map(|i| i % 1000 + 1).collect();

    // First call: cache miss → full prefill.
    let mut state1 = wick::kv_cache::InferenceState::from_config(cfg);
    let logits1 = model.forward_prefill(&tokens, 0, &mut state1);

    // Second call: cache hit → should restore and produce identical logits.
    let mut state2 = wick::kv_cache::InferenceState::from_config(cfg);
    let t0 = Instant::now();
    let logits2 = model.forward_prefill(&tokens, 0, &mut state2);
    let hit_ms = t0.elapsed().as_secs_f64() * 1000.0;

    assert_eq!(logits1.len(), logits2.len());
    let mut max_diff = 0.0f32;
    for i in 0..logits1.len() {
        max_diff = max_diff.max((logits1[i] - logits2[i]).abs());
    }

    eprintln!("=== Prefix Cache Correctness ===");
    eprintln!("  First call (miss): full prefill");
    eprintln!("  Second call (hit): {hit_ms:.2} ms");
    eprintln!("  Max logit diff: {max_diff:.6}");
    assert!(
        max_diff < 0.05,
        "Cache hit logits differ: max_diff={max_diff}"
    );

    // Third call: prefix match with extra tokens.
    let mut extended = tokens.clone();
    extended.extend_from_slice(&[42, 43, 44, 45]);
    let mut state3 = wick::kv_cache::InferenceState::from_config(cfg);
    let logits3 = model.forward_prefill(&extended, 0, &mut state3);
    eprintln!(
        "  Extended prefill (64+4 tokens): logits[0]={:.4}",
        logits3[0]
    );
    // Just check it doesn't crash and produces valid logits.
    assert!(logits3[0].is_finite());
}

/// Verify cold-tier (disk) cache roundtrip: save → load → verify logits match.
#[test]
#[ignore]
fn test_prefix_cache_cold_roundtrip() {
    use wick::model::Model;
    use wick::model::metal_lfm2::MetalLfm2Model;

    let Some(path) = find_model("LFM2.5-VL-1.6B-Q4_0") else {
        return;
    };

    let cache_dir = std::env::temp_dir().join("wick_test_cold_cache");
    let _ = std::fs::remove_dir_all(&cache_dir);

    // First model: prefill and cache to disk.
    let gguf1 = wick::gguf::GgufFile::open(&path).unwrap();
    let model1 = MetalLfm2Model::from_gguf(gguf1, &path, 8192).unwrap();
    let cfg = model1.config();

    let tokens: Vec<u32> = (0..64u32).map(|i| i % 1000 + 1).collect();

    // Configure cache with disk.
    {
        let mut cache = model1.prefix_cache.borrow_mut();
        cache.config.cache_dir = Some(cache_dir.clone());
    }

    // Prefill — triggers auto-cache (warm + cold).
    let mut state1 = wick::kv_cache::InferenceState::from_config(cfg);
    let logits1 = model1.forward_prefill(&tokens, 0, &mut state1);

    // Verify cold file exists.
    let cold_files: Vec<_> = std::fs::read_dir(&cache_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "kvcache"))
        .collect();
    eprintln!("Cold cache files: {}", cold_files.len());
    assert!(!cold_files.is_empty(), "No cold cache files created");

    // Second model: fresh instance, load from cold cache.
    let gguf2 = wick::gguf::GgufFile::open(&path).unwrap();
    let model2 = MetalLfm2Model::from_gguf(gguf2, &path, 8192).unwrap();
    {
        let mut cache = model2.prefix_cache.borrow_mut();
        cache.config.cache_dir = Some(cache_dir.clone());
    }

    // Prefill with same tokens — should hit cold cache.
    let mut state2 = wick::kv_cache::InferenceState::from_config(cfg);
    let t0 = Instant::now();
    let logits2 = model2.forward_prefill(&tokens, 0, &mut state2);
    let ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Compare logits.
    let mut max_diff = 0.0f32;
    for i in 0..logits1.len() {
        max_diff = max_diff.max((logits1[i] - logits2[i]).abs());
    }

    eprintln!("=== Cold Cache Roundtrip ===");
    eprintln!("  Cold restore + prefill: {ms:.1} ms");
    eprintln!("  Max logit diff: {max_diff:.6}");
    assert!(
        max_diff < 0.05,
        "Cold cache logits differ: max_diff={max_diff}"
    );

    // Cleanup.
    let _ = std::fs::remove_dir_all(&cache_dir);
}

/// Debug Q8_0 GEMV: compare a single weight matrix × vector on GPU vs CPU.
#[test]
#[ignore]
fn test_q8_0_gemv_parity() {
    use wick::model::Model;
    use wick::model::metal_lfm2::MetalLfm2Model;

    let Some(path) = find_model("LFM2.5-VL-1.6B-Q8_0") else {
        return;
    };
    let gguf = wick::gguf::GgufFile::open(&path).unwrap();
    let model = MetalLfm2Model::from_gguf(gguf, &path, 8192).unwrap();
    let cfg = model.config();

    // Run a single forward pass.
    let mut state_metal = wick::kv_cache::InferenceState::from_config(cfg);
    let logits_metal = model.forward(&[1], 0, &mut state_metal);

    // Also run on CPU for reference.
    let gguf2 = wick::gguf::GgufFile::open(&path).unwrap();
    let cpu_model = wick::model::load_model(gguf2, 8192).unwrap();
    let mut state_cpu = wick::kv_cache::InferenceState::from_config(cpu_model.config());
    let logits_cpu = cpu_model.forward(&[1], 0, &mut state_cpu);

    // Compare.
    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f64;
    for i in 0..logits_metal.len().min(logits_cpu.len()) {
        let d = (logits_metal[i] - logits_cpu[i]).abs();
        max_diff = max_diff.max(d);
        sum_diff += d as f64;
    }
    let avg_diff = sum_diff / logits_metal.len() as f64;

    // Top-5 comparison.
    let mut idx_metal: Vec<usize> = (0..logits_metal.len()).collect();
    idx_metal.sort_by(|&a, &b| logits_metal[b].partial_cmp(&logits_metal[a]).unwrap());
    let mut idx_cpu: Vec<usize> = (0..logits_cpu.len()).collect();
    idx_cpu.sort_by(|&a, &b| logits_cpu[b].partial_cmp(&logits_cpu[a]).unwrap());

    eprintln!("=== Q8_0 GEMV Parity (single token forward) ===");
    eprintln!(
        "  Metal logits[0..3]: {:.4} {:.4} {:.4}",
        logits_metal[0], logits_metal[1], logits_metal[2]
    );
    eprintln!(
        "  CPU   logits[0..3]: {:.4} {:.4} {:.4}",
        logits_cpu[0], logits_cpu[1], logits_cpu[2]
    );
    eprintln!("  max_diff: {max_diff:.4}, avg_diff: {avg_diff:.6}");
    eprintln!("  Metal top-5: {:?}", &idx_metal[..5]);
    eprintln!("  CPU   top-5: {:?}", &idx_cpu[..5]);
    eprintln!("  Metal top-1 logit: {:.4}", logits_metal[idx_metal[0]]);
    eprintln!("  CPU   top-1 logit: {:.4}", logits_cpu[idx_cpu[0]]);
}

/// Q8_0 prefill parity: compare 6-token prefill logits Metal vs CPU.
#[test]
#[ignore]
fn test_q8_0_prefill_parity() {
    use wick::model::Model;
    use wick::model::metal_lfm2::MetalLfm2Model;

    let Some(path) = find_model("LFM2.5-VL-1.6B-Q8_0") else {
        return;
    };

    let tokens: Vec<u32> = vec![1, 422, 3871, 315, 5765, 338]; // "The capital of France is"

    // Metal prefill
    let gguf = wick::gguf::GgufFile::open(&path).unwrap();
    let model = MetalLfm2Model::from_gguf(gguf, &path, 8192).unwrap();
    model.configure_cache(wick::kv_cache::KvCacheConfig {
        cache_dir: None,
        max_warm_entries: 0,
        max_warm_bytes: 0,
        max_cold_bytes: 0,
    });
    let cfg = model.config();
    let mut state = wick::kv_cache::InferenceState::from_config(cfg);
    let logits_metal = model.forward_prefill(&tokens, 0, &mut state);

    // CPU reference
    let gguf2 = wick::gguf::GgufFile::open(&path).unwrap();
    let cpu_model = wick::model::load_model(gguf2, 8192).unwrap();
    let mut state_cpu = wick::kv_cache::InferenceState::from_config(cpu_model.config());
    let logits_cpu = cpu_model.forward_prefill(&tokens, 0, &mut state_cpu);

    let mut max_diff = 0.0f32;
    for i in 0..logits_metal.len().min(logits_cpu.len()) {
        max_diff = max_diff.max((logits_metal[i] - logits_cpu[i]).abs());
    }

    let mut idx_m: Vec<usize> = (0..logits_metal.len()).collect();
    idx_m.sort_by(|&a, &b| logits_metal[b].partial_cmp(&logits_metal[a]).unwrap());
    let mut idx_c: Vec<usize> = (0..logits_cpu.len()).collect();
    idx_c.sort_by(|&a, &b| logits_cpu[b].partial_cmp(&logits_cpu[a]).unwrap());

    eprintln!("=== Q8_0 Prefill Parity (6 tokens) ===");
    eprintln!(
        "  Metal logits[0..3]: {:.4} {:.4} {:.4}",
        logits_metal[0], logits_metal[1], logits_metal[2]
    );
    eprintln!(
        "  CPU   logits[0..3]: {:.4} {:.4} {:.4}",
        logits_cpu[0], logits_cpu[1], logits_cpu[2]
    );
    eprintln!("  max_diff: {max_diff:.4}");
    eprintln!("  Metal top-5: {:?}", &idx_m[..5]);
    eprintln!("  CPU   top-5: {:?}", &idx_c[..5]);
}

/// Standalone Q8_0 GEMM parity: create a tiny matrix, run GPU GEMM, compare with CPU.
#[test]
#[ignore]
fn test_q8_0_gemm_standalone() {
    use wick::backend::metal::MetalContext;

    let ctx = MetalContext::new().unwrap();

    // Create a small Q8_0 weight: m=64 rows, k=32 cols (1 block per row).
    // Each row is one Q8_0 block: 2 bytes (f16 scale) + 32 bytes (int8 quants) = 34 bytes.
    let m = 64u32;
    let k = 32u32;
    let n = 4u32; // 4 input vectors
    let _nb = k / 32; // 1 block per row

    // Build Q8_0 weight data: scale=1.0, quants=[1,2,3,...,32] for each row.
    let mut weight_data = Vec::new();
    for row in 0..m {
        let scale: u16 = half::f16::from_f32(1.0 / (row as f32 + 1.0)).to_bits();
        weight_data.extend_from_slice(&scale.to_le_bytes());
        for j in 0..32u8 {
            weight_data.push(((j as i8) - 16) as u8); // quants: -16..15
        }
    }
    assert_eq!(weight_data.len(), m as usize * 34);

    // Build input: n vectors of k=32 floats, all 1.0.
    let input_data: Vec<f32> = vec![1.0; (n * k) as usize];

    // Expected output: for each row, dot(scale * quants, input) = scale * sum(quants)
    // quants = [-16, -15, ..., 15], sum = sum(-16..15) = -16+(-15)+...+15 = -16
    // (16 negative values from -16..-1 sum to -136, 16 values 0..15 sum to 120, total = -16)
    let quant_sum = -16.0f32;
    let mut expected = vec![0.0f32; (m * n) as usize];
    for row in 0..m {
        let scale = 1.0 / (row as f32 + 1.0);
        for col in 0..n {
            expected[(col * m + row) as usize] = scale * quant_sum;
        }
    }

    // Upload to GPU.
    let weight_buf = ctx.device.new_buffer_with_data(
        weight_data.as_ptr() as *const _,
        weight_data.len() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let input_buf = ctx.device.new_buffer_with_data(
        input_data.as_ptr() as *const _,
        (input_data.len() * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let output_buf = ctx.create_buffer((m * n * 4) as u64);

    // Create pipeline.
    let pipeline = ctx
        .create_pipeline(wick::backend::metal::shaders::GEMM_Q8_0, "gemm_q8_0")
        .unwrap();

    // Dispatch.
    let params: [u32; 6] = [m, k, n, k, m, 0]; // x_stride=k, y_stride=m
    let cb = ctx.queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&weight_buf), 0);
    enc.set_buffer(1, Some(&input_buf), 0);
    enc.set_buffer(2, Some(&output_buf), 0);
    enc.set_bytes(
        3,
        std::mem::size_of_val(&params) as u64,
        params.as_ptr() as *const _,
    );
    enc.set_threadgroup_memory_length(0, 8192);
    let tg_rows = (m + 63) / 64;
    let tg_cols = (n + 31) / 32;
    enc.dispatch_thread_groups(
        metal::MTLSize {
            width: tg_cols as u64,
            height: tg_rows as u64,
            depth: 1,
        },
        metal::MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    // Read back.
    let gpu_output = ctx.read_f32(&output_buf, (m * n) as usize);

    // Compare.
    let mut max_diff = 0.0f32;
    let mut nan_count = 0;
    for i in 0..(m * n) as usize {
        if gpu_output[i].is_nan() {
            nan_count += 1;
            continue;
        }
        max_diff = max_diff.max((gpu_output[i] - expected[i]).abs());
    }

    eprintln!("=== Q8_0 GEMM Standalone Test ===");
    eprintln!("  m={m}, k={k}, n={n}");
    eprintln!("  NaN count: {nan_count}/{}", m * n);
    eprintln!("  max_diff: {max_diff:.6}");
    eprintln!(
        "  GPU[0..4]: {:.4} {:.4} {:.4} {:.4}",
        gpu_output[0], gpu_output[1], gpu_output[2], gpu_output[3]
    );
    eprintln!(
        "  Expected[0..4]: {:.4} {:.4} {:.4} {:.4}",
        expected[0], expected[1], expected[2], expected[3]
    );

    assert_eq!(nan_count, 0, "GEMM produced NaN values");
    assert!(max_diff < 1.0, "GEMM max_diff {max_diff} > 1.0");
}

/// Profile 450M prefill phase breakdown.
#[test]
#[ignore]
fn test_450m_prefill_phase_profile() {
    use wick::model::Model;
    use wick::model::metal_lfm2::MetalLfm2Model;

    let Some(path) = find_model("LFM2.5-VL-450M-Q4_0") else {
        return;
    };
    let gguf = wick::gguf::GgufFile::open(&path).unwrap();
    let model = MetalLfm2Model::from_gguf(gguf, &path, 8192).unwrap();
    model.configure_cache(wick::kv_cache::KvCacheConfig {
        cache_dir: None,
        max_warm_entries: 0,
        max_warm_bytes: 0,
        max_cold_bytes: 0,
    });
    let cfg = model.config();
    let n = 128;
    let tokens: Vec<u32> = (0..n as u32).map(|i| i % 1000 + 1).collect();

    let mut state = wick::kv_cache::InferenceState::from_config(cfg);
    let _ = model.forward_prefill_profiled(&tokens, 0, &mut state);

    let mut state = wick::kv_cache::InferenceState::from_config(cfg);
    let timings = model.forward_prefill_profiled(&tokens, 0, &mut state);

    let (total_us, cats) = aggregate_prefill_phases(&timings);
    eprintln!("=== 450M Prefill Phase Profile (n={n}) ===");
    eprintln!(
        "  Total: {:.1} ms ({:.0} tok/s)",
        total_us / 1000.0,
        n as f64 / (total_us / 1e6)
    );
    eprintln!(
        "  {:30} {:>8} {:>6} {:>6}",
        "Phase", "Total µs", "Count", "%"
    );
    for (cat, total, count) in &cats {
        eprintln!(
            "  {:30} {:>8.0} {:>6} {:>5.1}%",
            cat,
            total,
            count,
            total / total_us * 100.0
        );
    }
}

#[test]
#[ignore]
fn test_cpu_gemv_microbench() {
    use wick::backend::cpu;

    fn quantize_q4_0(weights_f32: &[f32], m: usize, k: usize) -> Vec<u8> {
        assert_eq!(k % 32, 0);
        let nb = k / 32;
        let mut out = Vec::with_capacity(m * nb * 18);
        for row in 0..m {
            for b in 0..nb {
                let start = row * k + b * 32;
                let block = &weights_f32[start..start + 32];
                let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                let d = amax / 7.0;
                let d_f16 = half::f16::from_f32(d);
                out.extend_from_slice(&d_f16.to_bits().to_le_bytes());
                let id = if d != 0.0 { 1.0 / d } else { 0.0 };
                for qi in 0..16 {
                    let lo = ((block[qi] * id + 8.5) as u8).min(15);
                    let hi = ((block[16 + qi] * id + 8.5) as u8).min(15);
                    out.push(lo | (hi << 4));
                }
            }
        }
        out
    }

    let shapes = [
        (1024, 1024, "attn-q"),
        (4096, 1024, "mid-4096"),
        (1024, 2048, "ffn-down"),
    ];

    println!("\n=== CPU Microbenchmarks (GEMV) ===");
    for (m, k, label) in shapes {
        let weights_f32: Vec<f32> = (0..m * k).map(|i| (i as f32).sin()).collect();
        let q4_bytes = quantize_q4_0(&weights_f32, m, k);
        let x: Vec<f32> = (0..k).map(|i| (i as f32).cos()).collect();
        let mut y = vec![0.0f32; m];

        // f32 gemv
        let t0 = Instant::now();
        let iters = 100;
        for _ in 0..iters {
            cpu::gemv_f32(bytemuck::cast_slice(&weights_f32), &x, &mut y, m, k);
        }
        let elapsed_f32 = t0.elapsed().as_secs_f64() / iters as f64;

        // Q4_0 gemv
        let mut q8_scales = vec![0.0f32; k / 32];
        let mut q8_quants = vec![0i8; k];
        let t0 = Instant::now();
        for _ in 0..iters {
            cpu::gemv_q4_0_f32(&q4_bytes, &x, &mut y, m, k, &mut q8_scales, &mut q8_quants);
        }
        let elapsed_q4 = t0.elapsed().as_secs_f64() / iters as f64;

        // Q4_0 with Q8 pre-quantized x
        let (q8_scales_pre, q8_qs_pre) = cpu::quantize_f32_to_q8_0(&x);
        let t0 = Instant::now();
        for _ in 0..iters {
            cpu::gemv_q4_0_with_q8(&q4_bytes, &q8_scales_pre, &q8_qs_pre, &mut y, m, k);
        }
        let elapsed_q4_q8 = t0.elapsed().as_secs_f64() / iters as f64;

        println!(
            "Shape {:<12}: f32={:7.1}µs, q4_0={:7.1}µs, q4_0+q8={:7.1}µs (x{:.1} vs f32)",
            label,
            elapsed_f32 * 1e6,
            elapsed_q4 * 1e6,
            elapsed_q4_q8 * 1e6,
            elapsed_f32 / elapsed_q4_q8
        );
    }
}

/// Long-context profiling: run `forward_prefill_profiled` for a (model, n)
/// cell and emit a structured, grep-able block to stderr.
///
/// Output format (one block per invocation):
///   === PROFILE_LONGCTX BEGIN ===
///   model=<name>
///   n=<tokens>
///   total_ms=<f>
///   tok_per_sec=<f>
///   category<TAB>total_us<TAB>count<TAB>pct
///   <cat><TAB><us><TAB><count><TAB><pct>
///   ...
///   === PROFILE_LONGCTX END ===
fn profile_longctx_run(model_name: &str, n: usize) {
    use wick::model::Model;
    use wick::model::metal_lfm2::MetalLfm2Model;

    let Some(path) = find_model(model_name) else {
        return;
    };
    let gguf = wick::gguf::GgufFile::open(&path).unwrap();
    let ctx = 8192usize.max(2 * n);
    let model = MetalLfm2Model::from_gguf(gguf, &path, ctx).unwrap();
    let cfg = model.config();

    let tokens: Vec<u32> = (0..n as u32).map(|i| i % 1000 + 1).collect();

    // Warmup.
    let mut state = wick::kv_cache::InferenceState::from_config(cfg);
    let _ = model.forward_prefill_profiled(&tokens, 0, &mut state);

    // Measured.
    let mut state = wick::kv_cache::InferenceState::from_config(cfg);
    let timings = model.forward_prefill_profiled(&tokens, 0, &mut state);

    let (total_us, cats) = aggregate_prefill_phases(&timings);

    eprintln!("=== PROFILE_LONGCTX BEGIN ===");
    eprintln!("model={model_name}");
    eprintln!("n={n}");
    eprintln!("total_ms={:.3}", total_us / 1000.0);
    eprintln!("tok_per_sec={:.1}", n as f64 / (total_us / 1e6));
    eprintln!("category\ttotal_us\tcount\tpct");
    for (cat, total, count) in &cats {
        let pct = total / total_us * 100.0;
        eprintln!("{cat}\t{total:.0}\t{count}\t{pct:.2}");
    }
    eprintln!("=== PROFILE_LONGCTX END ===");
}

#[test]
#[ignore]
fn test_profile_longctx_2_5_450m_n128() {
    profile_longctx_run("LFM2.5-VL-450M-Q4_0", 128);
}

#[test]
#[ignore]
fn test_profile_longctx_2_5_450m_n1024() {
    profile_longctx_run("LFM2.5-VL-450M-Q4_0", 1024);
}

#[test]
#[ignore]
fn test_profile_longctx_2_5_450m_n4096() {
    profile_longctx_run("LFM2.5-VL-450M-Q4_0", 4096);
}

#[test]
#[ignore]
fn test_profile_longctx_1_6b_n128() {
    profile_longctx_run("LFM2.5-VL-1.6B-Q4_0", 128);
}

#[test]
#[ignore]
fn test_profile_longctx_1_6b_n4096() {
    profile_longctx_run("LFM2.5-VL-1.6B-Q4_0", 4096);
}
