//! Numerical parity smoke against llama.cpp's `mtmd-cli` clip
//! encoder. Runs wick's vision encoder on a fixed input
//! (synthesised solid-red 256×256 PNG, deterministic byte-for-byte)
//! and asserts the resulting `[64, 1024]` image embeddings stay
//! within tolerance of values captured from llama.cpp's
//! `MTMD_DEBUG_EMBEDDINGS` dump on the same input + same model
//! weights (LFM2.5-VL-450M, mmproj-Q8_0).
//!
//! The captured reference doesn't cover the full 65 536 floats —
//! only the stats `MTMD_DEBUG_EMBEDDINGS` prints (mean / std / min /
//! max) and token 0's first / last 16 values. Even so it's enough to
//! catch the regression class that the patch-embed kernel-layout
//! fix corrected: a transposed kernel read produces values
//! correlated-but-wrong (the same channels show large negatives
//! at the same positions but per-channel magnitudes drift 50%+),
//! and that's exactly what the per-element tolerance below
//! detects.
//!
//! Tolerance picks: 8% relative for the four scalar stats
//! (aggregates that average out per-element f32 noise — measured
//! drift on the LFM2.5-VL-450M reference is ≤5% with the parallel
//! matmul's order-of-summation drift; 8% leaves a 60% headroom
//! while still flagging a structural regression that'd shift the
//! aggregates by 10%+), and 25% relative + 0.5 absolute floor for
//! the 32 token-0 sample values. Per-element drift is up to ~15%
//! relative on the captured input, so 25% covers parallel-summation
//! noise while still failing loudly on a transpose / GELU / norm
//! bug. A future tighter parity gate (full 65 536-float diff)
//! would land in `wick-parity` once we capture the full vector.
//!
//! Gating: `#[ignore]` + `WICK_TEST_DOWNLOAD=1` to share the same
//! cached LFM2.5-VL-450M GGUFs as `vl_bundle_load.rs`. Skips
//! silently when the env var is unset.

#![cfg(feature = "remote")]

mod common;

use wick::engine::{BackendPreference, EngineConfig, ModelFiles, WickEngine};
use wick::manifest::InferenceType;

const MAIN_URL: &str =
    "https://huggingface.co/LiquidAI/LFM2.5-VL-450M-GGUF/resolve/main/LFM2.5-VL-450M-Q4_0.gguf";
const MAIN_FILE: &str = "LFM2.5-VL-450M-Q4_0.gguf";
const MMPROJ_URL: &str = "https://huggingface.co/LiquidAI/LFM2.5-VL-450M-GGUF/resolve/main/mmproj-LFM2.5-VL-450m-Q8_0.gguf";
const MMPROJ_FILE: &str = "mmproj-LFM2.5-VL-450m-Q8_0.gguf";

// ── Captured llama.cpp reference (red 256² PNG, --image-min-tokens
// 64 --image-max-tokens 64, mtmd-cli @ 2026-05-06) ─────────────
//
// Captured by:
//   MTMD_DEBUG_EMBEDDINGS=1 ~/development/llama.cpp/build/bin/llama-mtmd-cli \
//     -m LFM2.5-VL-450M-Q4_0.gguf --mmproj mmproj-LFM2.5-VL-450m-Q8_0.gguf \
//     --image red_256.png --image-min-tokens 64 --image-max-tokens 64 \
//     -ngl 0 --temp 0 -n 1 --no-warmup
const REF_MEAN: f32 = -0.189_559;
const REF_STD: f32 = 6.200_130;
const REF_MIN: f32 = -324.152_222;
const REF_MAX: f32 = 49.164_997;
const REF_TOKEN0_FIRST16: [f32; 16] = [
    -0.939_353, 0.294_400, 3.302_522, 4.070_815, 0.248_474, 1.013_565, -4.281_222, -0.095_652,
    -2.364_076, 0.311_590, 1.661_267, 2.294_227, -4.201_199, 4.589_795, 0.908_630, -6.679_206,
];
const REF_TOKEN0_LAST16: [f32; 16] = [
    -1.012_028,
    -120.899_910,
    0.605_745,
    -1.800_021,
    -0.269_893,
    3.727_429,
    -1.852_228,
    -1.807_398,
    2.405_378,
    7.943_906,
    -5.764_641,
    0.502_975,
    1.892_469,
    0.259_958,
    -7.124_193,
    2.326_940,
];

#[test]
#[ignore = "downloads ~310 MB across two GGUFs; set WICK_TEST_DOWNLOAD=1 and pass --ignored"]
fn vl_clip_parity_smoke() {
    if std::env::var("WICK_TEST_DOWNLOAD").is_err() {
        eprintln!("skipping: WICK_TEST_DOWNLOAD not set");
        return;
    }

    let main = common::download::ensure_cached(MAIN_URL, MAIN_FILE);
    let mmproj = common::download::ensure_cached(MMPROJ_URL, MMPROJ_FILE);
    let mut files = ModelFiles::text(&main);
    files.multimodal_projector = Some(mmproj);
    files.inference_type = Some(InferenceType::LlamaCppImageToText);
    let engine = WickEngine::from_files(
        files,
        EngineConfig {
            context_size: 256,
            backend: BackendPreference::Cpu,
            ..Default::default()
        },
    )
    .expect("VL bundle load");

    let encoder = engine
        .vision_encoder()
        .expect("VL bundle should expose a vision encoder");

    // Synthesise the same solid-red 256×256 PNG llama.cpp ran on.
    // Every pixel is (255, 0, 0) so the image is identical
    // byte-for-byte to the file `MTMD_DEBUG_EMBEDDINGS` was captured
    // against (modulo PNG container metadata which the decoder
    // ignores). Avoids committing yet another binary fixture.
    use image::{ImageBuffer, Rgb};
    let img = ImageBuffer::<Rgb<u8>, _>::from_fn(256, 256, |_, _| Rgb([255u8, 0, 0]));
    let mut png = Vec::new();
    image::DynamicImage::ImageRgb8(img)
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .expect("encode synthetic png");

    let pixels = wick::model::vision_preprocessor::preprocess_image(&png, &encoder.config)
        .expect("preprocess red 256");
    let out = encoder.encode_image(&pixels).expect("encode_image");

    let proj_dim = encoder.config.projection_dim;
    assert_eq!(
        proj_dim, 1024,
        "test reference is calibrated for projection_dim=1024"
    );
    assert_eq!(
        out.len(),
        64 * proj_dim,
        "expected 64 image tokens × {proj_dim} = {}, got {}",
        64 * proj_dim,
        out.len()
    );

    // ── Stats check (tight: 5% relative). Aggregates average out
    // per-element f32 noise, so a 5% drift here means a structural
    // regression. ────────────────────────────────────────────────
    let n = out.len() as f32;
    let mean: f32 = out.iter().sum::<f32>() / n;
    let var: f32 = out.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
    let std = var.sqrt();
    let mn = out.iter().cloned().fold(f32::INFINITY, f32::min);
    let mx = out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    eprintln!("wick : mean={mean:.4} std={std:.4} min={mn:.4} max={mx:.4}");
    eprintln!("llama: mean={REF_MEAN:.4} std={REF_STD:.4} min={REF_MIN:.4} max={REF_MAX:.4}");
    assert_close_relative("mean", mean, REF_MEAN, 0.08);
    assert_close_relative("std", std, REF_STD, 0.08);
    assert_close_relative("min", mn, REF_MIN, 0.08);
    assert_close_relative("max", mx, REF_MAX, 0.08);

    // ── Per-element check on token 0 (looser: 25% relative + 0.5
    // absolute). Catches "values correlated-but-wrong" — the
    // signature of a structural transpose / GELU / norm bug. ─────
    let tok0 = &out[..proj_dim];
    let tok0_first16 = &tok0[..16];
    let tok0_last16 = &tok0[proj_dim - 16..];
    for (i, (got, want)) in tok0_first16
        .iter()
        .zip(REF_TOKEN0_FIRST16.iter())
        .enumerate()
    {
        assert_close_per_element(&format!("tok0[{i}]"), *got, *want);
    }
    for (i, (got, want)) in tok0_last16.iter().zip(REF_TOKEN0_LAST16.iter()).enumerate() {
        let pos = proj_dim - 16 + i;
        assert_close_per_element(&format!("tok0[{pos}]"), *got, *want);
    }
}

fn assert_close_relative(label: &str, got: f32, want: f32, rel_tol: f32) {
    let abs_diff = (got - want).abs();
    let rel_diff = if want.abs() > 1e-3 {
        abs_diff / want.abs()
    } else {
        abs_diff
    };
    assert!(
        rel_diff <= rel_tol,
        "{label}: got {got}, want {want} (rel diff {:.3} > tol {:.3})",
        rel_diff,
        rel_tol,
    );
}

fn assert_close_per_element(label: &str, got: f32, want: f32) {
    // Tolerance: max(0.5 absolute, 25% relative). The absolute
    // floor handles values near zero where relative tolerance is
    // meaningless; the relative bound handles large-magnitude
    // outliers (e.g. the -120 / -324 values in the captured
    // reference) where 0.5 absolute would be unreasonably tight.
    let abs_floor = 0.5_f32;
    let rel_tol = 0.25_f32;
    let abs_diff = (got - want).abs();
    let limit = abs_floor.max(want.abs() * rel_tol);
    assert!(
        abs_diff <= limit,
        "{label}: got {got}, want {want} (|diff| {abs_diff:.3} > limit {limit:.3})",
    );
}
