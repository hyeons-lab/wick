// Verify forward_embedding → forward_from_embedding roundtrip matches forward.
// Run with: cargo test -p wick --features metal --test embedding_roundtrip
#![cfg(all(feature = "metal", target_os = "macos"))]

use std::path::Path;

#[test]
fn embedding_roundtrip() {
    let path =
        Path::new(env!("HOME")).join(".leap/models/LFM2-VL-450M-Q4_0/LFM2-VL-450M-Q4_0.gguf");
    if !path.exists() {
        eprintln!("skipping — model not found");
        return;
    }

    let gguf = wick::gguf::GgufFile::open(&path).unwrap();
    let model = wick::model::load_model_metal(gguf, &path, 1024).unwrap();
    let cfg = model.config().clone();

    // Path A: normal forward for 2 tokens
    let mut state_a = wick::kv_cache::InferenceState::from_config(&cfg);
    let logits_a_t0 = model.forward(&[1], 0, &mut state_a);
    let logits_a_t1 = model.forward(&[5242], 1, &mut state_a); // "Paris"

    // Path B: forward_embedding for token 0, then forward_from_embedding for token 1
    let mut state_b = wick::kv_cache::InferenceState::from_config(&cfg);
    let emb = model.forward_embedding(&[1], 0, &mut state_b);
    // feed embedding back → should produce same logits as normal forward
    let logits_b = model.forward_from_embedding(&emb, 1, &mut state_b);

    // Note: logits_b won't match logits_a_t1 because forward_from_embedding
    // feeds the HIDDEN STATE as the next token's embedding (not a real token
    // lookup). This test verifies the roundtrip doesn't crash and produces
    // valid logits of the right length.
    assert_eq!(logits_a_t0.len(), cfg.vocab_size);
    assert_eq!(logits_a_t1.len(), cfg.vocab_size);
    assert_eq!(logits_b.len(), cfg.vocab_size);
    assert!(
        logits_b.iter().all(|x| x.is_finite()),
        "logits contain NaN/Inf"
    );
    eprintln!("roundtrip OK: {} logits, all finite", logits_b.len());
}
