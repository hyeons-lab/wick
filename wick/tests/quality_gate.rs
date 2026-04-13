//! End-to-end quality gate for TurboQuant KV cache compression.
//!
//! Run with:
//! ```text
//! WICK_QUALITY_GATE=1 cargo test -p wick --release --test quality_gate -- --ignored --nocapture
//! ```
//!
//! This test is REQUIRED to pass before merging changes that touch the
//! TurboQuant encode/decode paths. It loads a real LFM2 model and measures
//! the logit similarity between the uncompressed and TurboQuant-compressed
//! paths after a fixed prompt. Lossy compression WILL cause some drift —
//! the thresholds here are calibrated to catch math bugs (sign flips,
//! indexing errors, off-by-one) while tolerating the legitimate quantization
//! noise that the PolarQuant + QJL estimators produce.
//!
//! Gated behind `WICK_QUALITY_GATE=1` so it only runs intentionally (it
//! requires the LFM2 model file on disk and takes a few seconds).

use wick::gguf::GgufFile;
use wick::kv_cache::{InferenceState, KvCompression};
use wick::model::Model;
use wick::model::lfm2::Lfm2Model;
use wick::tokenizer::BpeTokenizer;

fn find_lfm2_model() -> Option<std::path::PathBuf> {
    let home = std::env::var("HOME").ok()?;
    for candidate in [
        ".leap/models/LFM2.5-VL-450M-Q4_0/LFM2.5-VL-450M-Q4_0.gguf",
        ".leap/models/LFM2-VL-450M-Q4_0/LFM2-VL-450M-Q4_0.gguf",
        ".leap/models/LFM2.5-VL-1.6B-Q4_0/LFM2.5-VL-1.6B-Q4_0.gguf",
    ] {
        let p = std::path::PathBuf::from(&home).join(candidate);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

/// Cosine similarity between two f32 vectors.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na * nb)
}

/// Indices of the top-k elements in a logits vector, sorted descending.
fn top_k(logits: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    idx.truncate(k);
    idx
}

/// Runs the full prompt through a model and returns the logits for the
/// token AFTER the last prompt position — i.e. what would normally be
/// sampled as the first generated token.
fn prefill_and_next_logits(model: &dyn Model, tokens: &[u32], cm: KvCompression) -> Vec<f32> {
    let cfg = model.config();
    let mut state = InferenceState::from_config_with_compression(cfg, &cm);
    model.forward_prefill(tokens, 0, &mut state)
}

#[test]
#[ignore] // run with --ignored + WICK_QUALITY_GATE=1
fn test_kv_compression_quality_gate() {
    if std::env::var("WICK_QUALITY_GATE").as_deref() != Ok("1") {
        eprintln!("skipping: WICK_QUALITY_GATE=1 not set");
        return;
    }

    let model_path = match find_lfm2_model() {
        Some(p) => p,
        None => {
            eprintln!("skipping: no LFM2 model found in ~/.leap/models/");
            return;
        }
    };
    eprintln!("quality gate model: {}", model_path.display());

    let tok_gguf = GgufFile::open(&model_path).expect("open gguf");
    let tokenizer = BpeTokenizer::from_gguf(&tok_gguf).expect("tokenizer");

    // One model instance is enough — TurboQuant state now lives entirely on
    // the InferenceState, so the same model can be reused across modes.
    let gguf = GgufFile::open(&model_path).expect("open gguf");
    let model = Lfm2Model::from_gguf(gguf, 8192).expect("load model");
    assert!(model.turboquant_supported(), "TurboQuant not supported");

    // Deterministic prompt.
    let prompt = "The capital of France is";
    let mut tokens = Vec::new();
    if tokenizer.bos_token().is_some()
        && tok_gguf
            .get_bool("tokenizer.ggml.add_bos_token")
            .unwrap_or(false)
    {
        tokens.push(tokenizer.bos_token().unwrap());
    }
    tokens.extend_from_slice(&tokenizer.encode(prompt));
    eprintln!("prompt tokens: {} ({:?})", tokens.len(), tokens);

    // Compare logits at the next-token position after prefill.
    let logits_a = prefill_and_next_logits(&model as &dyn Model, &tokens, KvCompression::None);
    let logits_b =
        prefill_and_next_logits(&model as &dyn Model, &tokens, KvCompression::turboquant(42));
    assert_eq!(logits_a.len(), logits_b.len(), "vocab sizes differ");

    let cos = cosine(&logits_a, &logits_b);
    let argmax_a = logits_a
        .iter()
        .enumerate()
        .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    let argmax_b = logits_b
        .iter()
        .enumerate()
        .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    let top5_a = top_k(&logits_a, 5);
    let top5_b = top_k(&logits_b, 5);
    let top5_overlap = top5_a.iter().filter(|t| top5_b.contains(t)).count();

    // Does the uncompressed-path argmax still appear in the TurboQuant top-5?
    // This is the most important semantic check — it means TurboQuant still
    // "knows about" the right answer even if it ranks it slightly lower.
    let expected_in_top5 = top5_b.contains(&argmax_a);

    eprintln!("  logit cosine similarity: {cos:.4}");
    eprintln!(
        "  argmax A: {argmax_a} ({:?})",
        tokenizer.decode(&[argmax_a as u32])
    );
    eprintln!(
        "  argmax B: {argmax_b} ({:?})",
        tokenizer.decode(&[argmax_b as u32])
    );
    eprintln!("  top-5 A: {top5_a:?}");
    eprintln!("  top-5 B: {top5_b:?}");
    eprintln!("  top-5 overlap: {top5_overlap}/5");
    eprintln!("  expected token (argmax A) in top-5 B: {expected_in_top5}");

    // Gate 1: logit cosine similarity. TurboQuant compresses both keys
    // (3-bit) and values (2-bit); 16 layers of compound quantization noise
    // produces real drift. This threshold is calibrated against a known-good
    // baseline: cosine ~0.90 is the current achievable level, so 0.85 catches
    // regressions that worsen drift by 50%+ without being artificially tight.
    //
    // A cosine below 0.85 would indicate a real math bug (sign flip, wrong
    // index, bad rotation) — random logits have cosine ~0.
    assert!(
        cos > 0.85,
        "TurboQuant quality gate FAILED: logit cosine {cos:.4} < 0.85 — suggests a math bug"
    );

    // Gate 2: top-5 overlap. At minimum 2 of the top 5 argmaxes should
    // still agree. Catches cases where cosine is passable but the ordering
    // has been scrambled in a way that would affect any sampler.
    assert!(
        top5_overlap >= 2,
        "TurboQuant quality gate FAILED: top-5 overlap {top5_overlap}/5 < 2"
    );

    // Gate 3: the correct answer stays in the top-5 of the compressed path.
    // This is the semantic check — even if argmax drifts, the model must
    // still rank the right token highly.
    assert!(
        expected_in_top5,
        "TurboQuant quality gate FAILED: expected token {argmax_a} ({:?}) not in top-5 of compressed path {top5_b:?}",
        tokenizer.decode(&[argmax_a as u32])
    );

    eprintln!("quality gate passed");
}
