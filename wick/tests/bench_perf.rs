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
    // Floor set below measured baseline (259 tok/s in test, 384 in bench)
    // to account for test framework overhead + cold model load.
    assert!(tps > 200.0, "Metal decode n=64: {tps:.1} tok/s < 200 floor");
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
        tps > 200.0,
        "Metal decode n=512: {tps:.1} tok/s < 200 floor"
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
    // Floor set below measured baseline (299 tok/s in test, 344 in bench)
    assert!(
        tps > 250.0,
        "Metal prefill n=128: {tps:.1} tok/s < 250 floor"
    );
}

#[test]
#[ignore]
fn test_metal_prefill_long() {
    let Some(path) = find_model("LFM2.5-VL-1.6B-Q4_0") else {
        return;
    };
    eprintln!("=== Metal prefill long ===");
    let tps = bench_prefill(&path, 512, 2);
    assert!(
        tps > 200.0,
        "Metal prefill n=512: {tps:.1} tok/s < 200 floor"
    );
}
