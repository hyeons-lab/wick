//! `n_keep` context shift (Phase 1.5) — correctness coverage.
//!
//! - `shift_attention_kv_drains_middle_range` drives the KV-cache
//!   method directly on a hand-built `InferenceState` with a
//!   deterministic fill, so the memmove behavior (head preserved,
//!   tail slid down) is auditable cell-by-cell.
//! - `shift_frees_capacity_when_n_keep_set` simulates the exact
//!   sequence `Session::append_tokens` runs on overflow (prefill →
//!   `shift_attention_kv` → more prefill) using a `MockModel`. We
//!   don't drive `Session` directly because it holds a
//!   `&BpeTokenizer` that isn't buildable without a real GGUF
//!   (same reason as `cancel_mid_prefill.rs`); the end-to-end
//!   Session wiring — `ubatch_size` threading, position atomic,
//!   `last_logits` clearing, `tracing::info!` emission — is covered
//!   by the `#[ignore]`-gated `session_chain.rs` integration tests
//!   plus manual CLI smoke.
//! - `is_compressed_false_on_fresh_state` locks in the fast-path for
//!   the overflow arm's compression gate.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use wick::kv_cache::{InferenceState, LayerState};
use wick::model::{Model, ModelConfig};

// ---------------------------------------------------------------------------
// Unit: shift_attention_kv directly
// ---------------------------------------------------------------------------

fn build_attention_state(seq_len: usize, kv_dim: usize, n_layers: usize) -> InferenceState {
    let mut state = InferenceState::new(n_layers);
    state.seq_len = seq_len;
    for layer_idx in 0..n_layers {
        if let LayerState::Attention {
            key_cache,
            value_cache,
            ..
        } = &mut state.layers[layer_idx]
        {
            key_cache.reserve(seq_len * kv_dim);
            value_cache.reserve(seq_len * kv_dim);
            for t in 0..seq_len {
                for d in 0..kv_dim {
                    // Deterministic pattern: position + dim + layer.
                    let v = (t * 1000 + d * 10 + layer_idx) as f32;
                    key_cache.push(v);
                    value_cache.push(v + 0.5);
                }
            }
        }
    }
    state
}

#[test]
fn shift_attention_kv_drains_middle_range() {
    let seq_len = 16;
    let kv_dim = 4;
    let n_layers = 2;
    let mut state = build_attention_state(seq_len, kv_dim, n_layers);

    let n_keep = 3;
    let shift = 5;
    state.shift_attention_kv(n_keep, shift);

    assert_eq!(state.seq_len, seq_len - shift);
    for layer_idx in 0..n_layers {
        if let LayerState::Attention {
            key_cache,
            value_cache,
            ..
        } = &state.layers[layer_idx]
        {
            assert_eq!(key_cache.len(), (seq_len - shift) * kv_dim);
            assert_eq!(value_cache.len(), (seq_len - shift) * kv_dim);

            // Head cells (0..n_keep) byte-identical to pre-shift values.
            for t in 0..n_keep {
                for d in 0..kv_dim {
                    let expected = (t * 1000 + d * 10 + layer_idx) as f32;
                    assert_eq!(
                        key_cache[t * kv_dim + d],
                        expected,
                        "layer {layer_idx} head cell t={t} d={d}"
                    );
                }
            }
            // Tail cells (n_keep..new_seq_len) hold the values that were
            // at [n_keep+shift..seq_len) pre-shift.
            for new_t in n_keep..(seq_len - shift) {
                let old_t = new_t + shift;
                for d in 0..kv_dim {
                    let expected = (old_t * 1000 + d * 10 + layer_idx) as f32;
                    assert_eq!(
                        key_cache[new_t * kv_dim + d],
                        expected,
                        "layer {layer_idx} tail cell new_t={new_t} old_t={old_t} d={d}"
                    );
                    assert_eq!(
                        value_cache[new_t * kv_dim + d],
                        expected + 0.5,
                        "value tail new_t={new_t}"
                    );
                }
            }
        }
    }
}

#[test]
fn is_compressed_false_on_fresh_state() {
    let state = InferenceState::new(4);
    assert!(!state.is_compressed());
}

// ---------------------------------------------------------------------------
// Integration via MockModel — Session::append_tokens overflow path
// ---------------------------------------------------------------------------

struct MockModel {
    config: ModelConfig,
    prefill_calls: AtomicUsize,
}

impl Model for MockModel {
    fn forward(&self, _: &[u32], _: usize, _: &mut InferenceState) -> Vec<f32> {
        vec![0.0; self.config.vocab_size]
    }
    fn forward_prefill(
        &self,
        tokens: &[u32],
        _start_pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        self.prefill_calls.fetch_add(1, Ordering::Relaxed);
        // Advance seq_len + append placeholder KV per-token so the
        // state actually grows (so subsequent shifts have something
        // to move). KV width is `n_kv_heads * head_dim` — for GQA/MQA
        // models that's smaller than `n_heads * head_dim`, so don't
        // use `n_heads` here even though our mock config happens to
        // have them equal.
        let head_dim = self.config.hidden_size / self.config.n_heads.max(1);
        let kv_dim = self.config.n_kv_heads * head_dim;
        for _ in tokens {
            if let LayerState::Attention {
                key_cache,
                value_cache,
                ..
            } = &mut state.layers[0]
            {
                key_cache.extend(std::iter::repeat_n(0.0f32, kv_dim));
                value_cache.extend(std::iter::repeat_n(0.0f32, kv_dim));
            }
            state.seq_len += 1;
        }
        vec![0.0; self.config.vocab_size]
    }
    fn config(&self) -> &ModelConfig {
        &self.config
    }
}

fn mock_attention_config(max_seq_len: usize) -> ModelConfig {
    ModelConfig {
        architecture: "mock".into(),
        n_layers: 1,
        hidden_size: 8,
        intermediate_size: 16,
        n_heads: 4,
        n_kv_heads: 4,
        vocab_size: 8,
        max_seq_len,
        rope_theta: 0.0,
        rms_norm_eps: 0.0,
        block_types: vec![wick::model::BlockType::Attention],
        conv_kernel_size: None,
        kv_heads_per_layer: vec![4],
    }
}

/// Drive the chunked-prefill path on a fresh MockModel state until
/// we've eaten `tokens_total` tokens. Returns (consumed, final_state).
fn run_prefill_until_full(
    model: &MockModel,
    state: &mut InferenceState,
    ubatch: usize,
    tokens: &[u32],
) -> usize {
    let cancel = Arc::new(AtomicBool::new(false));
    let (consumed, _) =
        model.forward_prefill_chunked(tokens, state.seq_len, state, ubatch, &cancel);
    consumed
}

#[test]
fn shift_frees_capacity_when_n_keep_set() {
    // Simulate Session::append_tokens's overflow arm by composing the
    // same primitives. We don't construct Session directly (needs a
    // BpeTokenizer; see cancel_mid_prefill.rs for the rationale).
    let max_seq_len = 32;
    let n_keep = 4;
    let cfg = mock_attention_config(max_seq_len);
    let model = MockModel {
        config: cfg.clone(),
        prefill_calls: AtomicUsize::new(0),
    };

    let mut state = InferenceState::new(cfg.block_types.len());
    // Fill close to capacity.
    let first_batch: Vec<u32> = (0..28u32).collect();
    let consumed = run_prefill_until_full(&model, &mut state, 64, &first_batch);
    assert_eq!(consumed, 28);
    assert_eq!(state.seq_len, 28);

    // Simulate the Session overflow-arm: we want to add 8 more but
    // 28 + 8 = 36 > 32. Shift by 4 → seq_len = 24, then prefill 8 → 32.
    let shift_needed = 28 + 8 - max_seq_len;
    assert_eq!(shift_needed, 4);
    assert!(state.seq_len >= n_keep + shift_needed);
    state.shift_attention_kv(n_keep, shift_needed);
    assert_eq!(state.seq_len, 24);

    // Add the new 8 tokens after the shift.
    let second_batch: Vec<u32> = (28..36u32).collect();
    let consumed = run_prefill_until_full(&model, &mut state, 64, &second_batch);
    assert_eq!(consumed, 8);
    assert_eq!(state.seq_len, 32);
    assert_eq!(state.seq_len, max_seq_len);

    // Attention layer's KV now stores 32 cells (head + shifted tail + new).
    if let LayerState::Attention { key_cache, .. } = &state.layers[0] {
        let head_dim = cfg.hidden_size / cfg.n_heads;
        let kv_dim = cfg.n_kv_heads * head_dim;
        assert_eq!(key_cache.len(), 32 * kv_dim);
    }
}
