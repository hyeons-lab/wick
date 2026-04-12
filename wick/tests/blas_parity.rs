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

/// Verify the flash attention path produces correct results by comparing
/// a 300-token `forward_prefill` (n >= 256, triggers flash attention) against
/// sequential `forward()` calls over the same tokens. Also confirms that
/// two runs of the naive path (9 tokens, below the threshold) are
/// bit-identical as a baseline sanity check.
///
/// Flash attention uses online softmax (different reduction order from the
/// naive full-vector softmax), so some drift is expected. The bar is
/// cosine > 0.99 and matching top-1.
#[test]
#[ignore]
fn test_flash_vs_naive_prefill_parity() {
    #[allow(unused_imports)]
    use wick::model::Model;

    let Some(path) = find_model("LFM2.5-VL-1.6B-Q4_0") else {
        return;
    };

    // 300 tokens — above the FLASH_ATTN_THRESHOLD (256), so attention
    // layers use the flash path. Use simple sequential token IDs.
    let tokens_long: Vec<u32> = (1..=300).collect();
    // 9 tokens — below the threshold, so attention layers use naive.
    let tokens_short: Vec<u32> = tokens_long[..9].to_vec();

    // Run forward_prefill on the SHORT prompt (naive path).
    let gguf_a = wick::gguf::GgufFile::open(&path).unwrap();
    let model_a = wick::model::load_model(gguf_a).unwrap();
    let cfg = model_a.config();
    let mut state_a = wick::kv_cache::InferenceState::from_config(cfg);
    let logits_naive = model_a.forward_prefill(&tokens_short, 0, &mut state_a);

    // Run forward_prefill on the LONG prompt (flash path), but only
    // compare the last-token logits from the first 9 tokens' perspective.
    // Since the long prompt has MORE context, the logits won't match the
    // short-prompt logits exactly — they're conditioned on different inputs.
    //
    // Instead, run a SECOND short-prompt prefill using forward_prefill to
    // confirm it produces the same result as the first (both use naive).
    let gguf_b = wick::gguf::GgufFile::open(&path).unwrap();
    let model_b = wick::model::load_model(gguf_b).unwrap();
    let mut state_b = wick::kv_cache::InferenceState::from_config(cfg);
    let logits_naive2 = model_b.forward_prefill(&tokens_short, 0, &mut state_b);

    // Naive vs naive should be bit-identical.
    let cos_nn = cosine(&logits_naive, &logits_naive2);
    assert!(
        cos_nn > 0.9999,
        "naive vs naive cosine = {cos_nn} — should be near-identical"
    );

    // Now run the LONG prompt (flash path) and compare its last-token
    // logits to a sequential forward() over the same 300 tokens.
    let gguf_c = wick::gguf::GgufFile::open(&path).unwrap();
    let model_c = wick::model::load_model(gguf_c).unwrap();
    let mut state_c = wick::kv_cache::InferenceState::from_config(cfg);
    let logits_flash = model_c.forward_prefill(&tokens_long, 0, &mut state_c);

    let gguf_d = wick::gguf::GgufFile::open(&path).unwrap();
    let model_d = wick::model::load_model(gguf_d).unwrap();
    let mut state_d = wick::kv_cache::InferenceState::from_config(cfg);
    let mut logits_seq = Vec::new();
    for (i, &tok) in tokens_long.iter().enumerate() {
        logits_seq = model_d.forward(&[tok], i, &mut state_d);
    }

    let cos_fs = cosine(&logits_flash, &logits_seq);
    let max_diff = max_abs_diff(&logits_flash, &logits_seq);
    let top_flash = argmax(&logits_flash);
    let top_seq = argmax(&logits_seq);

    eprintln!("=== Flash prefill (300 tok) vs sequential ===");
    eprintln!("  cosine:   {cos_fs:.6}");
    eprintln!("  max_diff: {max_diff:.4}");
    eprintln!("  top-1:    flash={top_flash}  seq={top_seq}");

    // Flash uses online softmax (different reduction order), so some
    // drift is expected. With 300 tokens the drift compounds more than
    // with 9. Cosine > 0.99 and matching top-1 is the bar.
    assert!(
        cos_fs > 0.99,
        "flash vs sequential cosine = {cos_fs} (< 0.99)"
    );
    assert_eq!(
        top_flash, top_seq,
        "top-1 mismatch: flash={top_flash} seq={top_seq}"
    );
}
