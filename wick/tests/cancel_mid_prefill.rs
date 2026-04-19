//! Cancellation during chunked prefill (Phase 1.4 regression coverage).
//!
//! These tests drive the default `Model::forward_prefill_chunked`
//! directly, NOT through `Session::append_tokens`. The chunking + cancel
//! logic lives on the trait method, so this exercises the real business
//! rule (stop between chunks when cancel fires, always run at least one
//! chunk, `ubatch == 0` → one chunk).
//!
//! **What is NOT covered here:** `Session`'s wiring — `ubatch_size`
//! threading from `SessionConfig`, the `Cancelled` → `Err` mapping,
//! `current_pos` advance by `consumed` rather than `tokens.len()`, and
//! the "clear `last_logits` on cancel" contract. That surface can't be
//! unit-tested without constructing a `BpeTokenizer` (currently only
//! buildable from a real GGUF). End-to-end coverage lives in the
//! `#[ignore]`d `session_chain.rs` integration tests that load a local
//! model; adding proper Session-level unit tests is a follow-up that
//! requires loosening the tokenizer constructor.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use wick::kv_cache::InferenceState;
use wick::model::{Model, ModelConfig};

/// Minimal fake model: no real inference, just enough trait surface
/// to call `Model::forward_prefill_chunked` directly. Records chunk
/// counts and flips the shared cancel flag on a configurable chunk
/// index so tests can observe the between-chunks cancel check.
struct MockModel {
    config: ModelConfig,
    chunks_seen: AtomicUsize,
    cancel_after_chunks: usize,
    cancel_flag: Arc<AtomicBool>,
}

impl Model for MockModel {
    fn forward(&self, _: &[u32], _: usize, _: &mut InferenceState) -> Vec<f32> {
        vec![0.0; self.config.vocab_size]
    }

    fn forward_prefill(
        &self,
        _tokens: &[u32],
        _start_pos: usize,
        _state: &mut InferenceState,
    ) -> Vec<f32> {
        let seen = self.chunks_seen.fetch_add(1, Ordering::Relaxed) + 1;
        if seen >= self.cancel_after_chunks {
            // Flip the session's cancel flag; the default
            // `forward_prefill_chunked` will observe it between chunks.
            self.cancel_flag.store(true, Ordering::Relaxed);
        }
        vec![0.0; self.config.vocab_size]
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }
}

fn mock_config(vocab_size: usize, max_seq_len: usize) -> ModelConfig {
    ModelConfig {
        architecture: "mock".into(),
        n_layers: 0,
        hidden_size: 0,
        intermediate_size: 0,
        n_heads: 0,
        n_kv_heads: 0,
        vocab_size,
        max_seq_len,
        rope_theta: 0.0,
        rms_norm_eps: 0.0,
        block_types: Vec::new(),
        conv_kernel_size: None,
        kv_heads_per_layer: Vec::new(),
    }
}

#[test]
fn cancel_after_first_chunk_returns_cancelled_and_advances_one_ubatch() {
    let cfg = mock_config(8, 1024);
    let cancel = Arc::new(AtomicBool::new(false));
    let model = MockModel {
        config: cfg.clone(),
        chunks_seen: AtomicUsize::new(0),
        cancel_after_chunks: 1,
        cancel_flag: Arc::clone(&cancel),
    };
    let mut state = InferenceState::new(0);

    let tokens: Vec<u32> = (0..64u32).collect();
    let ubatch = 16;
    let (consumed, last_logits) =
        model.forward_prefill_chunked(&tokens, 0, &mut state, ubatch, &cancel);

    assert_eq!(
        consumed, ubatch,
        "should stop after exactly one ubatch once cancel fires after the first chunk"
    );
    assert_eq!(
        model.chunks_seen.load(Ordering::Relaxed),
        1,
        "exactly one forward_prefill should have run"
    );
    assert!(
        last_logits.is_some(),
        "last chunk's logits should be returned"
    );
}

#[test]
fn chunked_default_impl_does_all_chunks_when_not_cancelled() {
    let cfg = mock_config(8, 1024);
    let cancel = Arc::new(AtomicBool::new(false));
    let model = MockModel {
        config: cfg.clone(),
        chunks_seen: AtomicUsize::new(0),
        // Never trip cancel — chunk index > any chunk count we'll hit.
        cancel_after_chunks: 100,
        cancel_flag: Arc::clone(&cancel),
    };
    let mut state = InferenceState::new(0);

    let tokens: Vec<u32> = (0..40u32).collect();
    let ubatch = 16;
    let (consumed, _) = model.forward_prefill_chunked(&tokens, 0, &mut state, ubatch, &cancel);

    assert_eq!(consumed, tokens.len());
    // ceil(40 / 16) = 3 chunks (16, 16, 8).
    assert_eq!(model.chunks_seen.load(Ordering::Relaxed), 3);
}

#[test]
fn ubatch_zero_means_no_chunking() {
    let cfg = mock_config(8, 1024);
    let cancel = Arc::new(AtomicBool::new(false));
    let model = MockModel {
        config: cfg.clone(),
        chunks_seen: AtomicUsize::new(0),
        cancel_after_chunks: 100,
        cancel_flag: Arc::clone(&cancel),
    };
    let mut state = InferenceState::new(0);

    // `ubatch = 0` is the CLI's "disable chunking" convention: do the
    // whole input in a single `forward_prefill` call. Must not
    // infinite-loop (older builds clamped to 1 → chunks.len() chunks,
    // which defeats the opt-out).
    let tokens = vec![0u32; 4];
    let (consumed, _) = model.forward_prefill_chunked(&tokens, 0, &mut state, 0, &cancel);
    assert_eq!(consumed, 4);
    assert_eq!(
        model.chunks_seen.load(Ordering::Relaxed),
        1,
        "ubatch=0 must produce exactly one chunk"
    );
}
