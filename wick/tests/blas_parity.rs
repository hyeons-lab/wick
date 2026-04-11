//! Parity test: prefill (BLAS path when feature is on, NEON path otherwise)
//! versus sequential per-token forward.
//!
//! `forward_prefill` runs the batched aarch64 path — under `feature = "blas"`
//! every GEMM site goes through `try_blas_prefill_gemm` and Accelerate SGEMM.
//! `forward` called token-by-token runs the sequential GEMV path that never
//! touches the BLAS helper. If both paths agree on the last-token logits to
//! within numerical noise the BLAS rewrite is correct.
//!
//! Two scenarios:
//! 1. **Single token** — exercises the prefill GEMM call sites once with no
//!    cross-token compounding. Tight tolerance (cosine > 0.9999), this catches
//!    layout / dim / transpose bugs immediately.
//! 2. **Nine tokens** — same operations but with KV-cache feedback so any
//!    drift compounds across positions. Looser tolerance (cosine > 0.99) but
//!    still requires top-1 agreement.
//!
//! The single-token check is the tight one. The integer-NEON path is
//! bit-identical to sequential (verified locally with cosine = 1.0,
//! max_diff = 0.0), so any drift on the BLAS build is purely from the f32
//! SGEMM accumulation order vs the NEON int8-dot + f32-scale order — both
//! correct, both expected.
//!
//! Marked `#[ignore]` because it depends on a real GGUF model in
//! `~/.leap/models`. Run with:
//!
//! ```
//! cargo test -p wick --release --test blas_parity -- --ignored --nocapture
//! ```

#![cfg(target_arch = "aarch64")]

use std::path::PathBuf;

fn find_model(name: &str) -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var("HOME").ok()?)
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

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

fn run_parity(model_name: &str, tokens: &[u32]) -> Option<(f32, f32, usize, usize)> {
    // Bring the `Model` trait into scope so the boxed trait object's methods
    // (`forward_prefill`, `forward`, `config`) resolve via the vtable.
    #[allow(unused_imports)]
    use wick::model::Model;

    let path = find_model(model_name)?;

    let gguf_a = wick::gguf::GgufFile::open(&path).unwrap();
    let model_a = wick::model::load_model(gguf_a).unwrap();
    let cfg = model_a.config();
    let mut state_a = wick::kv_cache::InferenceState::from_config(cfg);
    let logits_prefill = model_a.forward_prefill(tokens, 0, &mut state_a);

    let gguf_b = wick::gguf::GgufFile::open(&path).unwrap();
    let model_b = wick::model::load_model(gguf_b).unwrap();
    let mut state_b = wick::kv_cache::InferenceState::from_config(cfg);
    let mut logits_seq = Vec::new();
    for (i, &tok) in tokens.iter().enumerate() {
        logits_seq = model_b.forward(&[tok], i, &mut state_b);
    }

    assert_eq!(logits_prefill.len(), logits_seq.len());
    Some((
        cosine(&logits_prefill, &logits_seq),
        max_abs_diff(&logits_prefill, &logits_seq),
        argmax(&logits_prefill),
        argmax(&logits_seq),
    ))
}

#[test]
#[ignore]
fn test_prefill_single_token_parity() {
    let Some((cos, max_diff, top_prefill, top_seq)) = run_parity("LFM2.5-VL-1.6B-Q4_0", &[1])
    else {
        return;
    };

    eprintln!("=== Prefill vs sequential (1 token) ===");
    eprintln!("  cosine:   {cos:.6}");
    eprintln!("  max_diff: {max_diff:.4}");
    eprintln!("  top-1:    prefill={top_prefill}  seq={top_seq}");

    // Tight bound: a single forward step has no cross-token compounding,
    // so even f32-vs-int reduction-order noise should leave cosine very near 1.
    // A real layout / dim / transpose bug shows up here as cosine < 0.99 or
    // a top-1 mismatch.
    assert!(
        cos > 0.9999,
        "single-token prefill vs sequential cosine = {cos} (< 0.9999) — likely a layout/dim/transpose bug"
    );
    assert_eq!(top_prefill, top_seq, "top-1 mismatch on single token");
}

#[test]
#[ignore]
fn test_prefill_multi_token_parity() {
    // Nine tokens — exercises KV-cache feedback so drift can compound across
    // positions through attention.
    let Some((cos, max_diff, top_prefill, top_seq)) = run_parity(
        "LFM2.5-VL-1.6B-Q4_0",
        &[1, 422, 3871, 315, 5765, 338, 891, 27, 14],
    ) else {
        return;
    };

    eprintln!("=== Prefill vs sequential (9 tokens) ===");
    eprintln!("  cosine:   {cos:.6}");
    eprintln!("  max_diff: {max_diff:.4}");
    eprintln!("  top-1:    prefill={top_prefill}  seq={top_seq}");

    // Looser bound: KV-cache feedback compounds reduction-order drift across
    // positions. Empirically NEON gives cosine = 1.0 here (bit-identical) and
    // BLAS gives ~0.996 (legitimate f32 reordering). Anything below 0.99 is
    // a real bug.
    assert!(
        cos > 0.99,
        "9-token prefill vs sequential cosine = {cos} (< 0.99) — likely a real correctness bug"
    );
    assert_eq!(top_prefill, top_seq, "top-1 mismatch on 9-token prefill");
}
