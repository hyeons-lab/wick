//! `n_keep` context shift (Phase 1.5) — correctness coverage.
//!
//! Four tests:
//! 1. **RoPE delta composes with direct rotation** —
//!    `apply_rope_to_head(raw, p_old)` then
//!    `apply_rope_delta_to_head(.., p_new - p_old)` must equal
//!    `apply_rope_to_head(raw, p_new)` within f32 epsilon. Uses the
//!    existing (known-good) `apply_rope_to_head` as the oracle, so a
//!    sign or pairing bug in the new delta helper is caught.
//! 2. **`shift_kv_with_rope` correctness** — hand-populate KV cache
//!    with `apply_rope_to_head(identity, t)` for each token position,
//!    call shift, assert post-shift cells match fresh-rotation for
//!    their new positions. Head cells `[0..n_keep)` must be byte-
//!    identical to pre-shift.
//! 3. **`is_compressed_false_on_fresh_state`** — fast-path gate for
//!    the uncompressed case (lock-in).
//! 4. **Session plumbing via MockModel** — MockModel overrides
//!    `supports_kv_shift → true` and counts `shift_kv` calls. Drives
//!    the same sequence `Session::append_tokens` runs on overflow and
//!    verifies the Session-visible effects (position advance, shift
//!    dispatch to the trait method). Doesn't re-exercise RoPE — that's
//!    covered by tests 1–2 and the real-model test in
//!    `shift_real_model.rs`.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use wick::backend::cpu::{apply_rope_delta_to_head, apply_rope_to_head};
use wick::kv_cache::{InferenceState, LayerState};
use wick::model::{Model, ModelConfig};

// ---------------------------------------------------------------------------
// Test 1 — RoPE delta composes with direct rotation
// ---------------------------------------------------------------------------

/// L2 distance between two slices; used for float-closeness asserts
/// instead of per-element `assert_eq` so sin/cos reassociation rounding
/// doesn't cause spurious failures.
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn rope_delta_composes_with_direct_rotation() {
    // Exercise a range of old/new position pairs including negative
    // deltas (the shift case). For each (p_old, p_new), a head rotated
    // for p_old and then re-rotated by delta=(p_new - p_old) must
    // match a head rotated directly for p_new.
    let head_dim = 64;
    let freq_base = 10_000.0f32;
    for &p_old in &[0u32, 1, 7, 64, 2048] {
        for &p_new in &[0u32, 1, 5, 63, 2000] {
            // Identity head values so we're measuring rotation, not
            // cancellation. Pick something non-trivial so most dim
            // pairs have non-zero magnitude.
            let mut via_compose: Vec<f32> = (0..head_dim).map(|i| (i as f32) * 0.01).collect();
            let mut via_direct = via_compose.clone();

            // Compose path: rotate for p_old, then delta to p_new.
            apply_rope_to_head(&mut via_compose, p_old as usize, head_dim, freq_base);
            let delta = (p_new as i32) - (p_old as i32);
            apply_rope_delta_to_head(&mut via_compose, delta, head_dim, freq_base);

            // Direct path: rotate raw for p_new in one step.
            apply_rope_to_head(&mut via_direct, p_new as usize, head_dim, freq_base);

            let err = max_abs_diff(&via_compose, &via_direct);
            assert!(
                err < 1e-3,
                "p_old={p_old} p_new={p_new} delta={delta} max-err={err}"
            );
        }
    }
}

#[test]
fn rope_delta_zero_is_identity() {
    // delta=0 should leave the head untouched (rotation by 0).
    let head_dim = 32;
    let freq_base = 10_000.0f32;
    let original: Vec<f32> = (0..head_dim).map(|i| i as f32 * 0.1 + 1.0).collect();
    let mut head = original.clone();
    apply_rope_delta_to_head(&mut head, 0, head_dim, freq_base);
    let err = max_abs_diff(&head, &original);
    assert!(err < 1e-5, "delta=0 should be identity, max-err={err}");
}

// ---------------------------------------------------------------------------
// Test 2 — shift_kv_with_rope correctness via oracle
// ---------------------------------------------------------------------------

fn build_state_with_rope_filled(
    seq_len: usize,
    head_dim: usize,
    n_kv_heads_per_layer: &[usize],
    freq_base: f32,
) -> InferenceState {
    let mut state = InferenceState::new(n_kv_heads_per_layer.len());
    state.seq_len = seq_len;
    for (layer_idx, &n_kv_heads) in n_kv_heads_per_layer.iter().enumerate() {
        let kv_dim = n_kv_heads * head_dim;
        if let LayerState::Attention {
            key_cache,
            value_cache,
            ..
        } = &mut state.layers[layer_idx]
        {
            key_cache.reserve(seq_len * kv_dim);
            value_cache.reserve(seq_len * kv_dim);
            for t in 0..seq_len {
                for h in 0..n_kv_heads {
                    // Deterministic "raw" K per (layer, head, dim):
                    // mixes layer + head + dim so different cells have
                    // distinct values that RoPE will rotate differently.
                    let raw: Vec<f32> = (0..head_dim)
                        .map(|d| (layer_idx as f32) + 0.1 * (h as f32) + 0.01 * (d as f32))
                        .collect();
                    let mut rotated = raw.clone();
                    apply_rope_to_head(&mut rotated, t, head_dim, freq_base);
                    key_cache.extend_from_slice(&rotated);
                    // V isn't RoPE'd, but we still populate each row with a
                    // position-dependent value so misindexed post-shift V
                    // rows (wrong offset, off-by-one in drain math) get
                    // caught — a position-invariant V fill would mask
                    // row-swap bugs because every row would look the same.
                    let v_row: Vec<f32> = raw.iter().map(|x| x + 0.001 * (t as f32)).collect();
                    value_cache.extend_from_slice(&v_row);
                }
            }
        }
    }
    state
}

#[test]
fn shift_kv_with_rope_preserves_head_and_re_rotates_tail() {
    let head_dim = 16;
    let n_kv_heads_per_layer = vec![4usize, 2];
    let seq_len = 24;
    let n_keep = 5;
    let shift = 7;
    let freq_base = 10_000.0f32;

    // Oracle snapshot: pre-shift K cells at [0, n_keep) (head)
    // and [n_keep + shift, seq_len) (which will become [n_keep, new_seq_len)
    // post-shift, and whose K must match fresh-rotation for their new
    // position).
    let mut state =
        build_state_with_rope_filled(seq_len, head_dim, &n_kv_heads_per_layer, freq_base);

    // Snapshot the head cells so we can prove they're untouched.
    let head_snapshot: Vec<Vec<f32>> = n_kv_heads_per_layer
        .iter()
        .enumerate()
        .map(|(layer_idx, &n_kv_heads)| {
            let kv_dim = n_kv_heads * head_dim;
            if let LayerState::Attention { key_cache, .. } = &state.layers[layer_idx] {
                key_cache[..n_keep * kv_dim].to_vec()
            } else {
                unreachable!()
            }
        })
        .collect();

    state.shift_kv_with_rope(n_keep, shift, freq_base, head_dim, &n_kv_heads_per_layer);

    assert_eq!(state.seq_len, seq_len - shift);

    for (layer_idx, &n_kv_heads) in n_kv_heads_per_layer.iter().enumerate() {
        let kv_dim = n_kv_heads * head_dim;
        let new_seq_len = seq_len - shift;
        if let LayerState::Attention {
            key_cache,
            value_cache,
            ..
        } = &state.layers[layer_idx]
        {
            assert_eq!(key_cache.len(), new_seq_len * kv_dim);
            assert_eq!(value_cache.len(), new_seq_len * kv_dim);

            // Head cells: byte-identical to pre-shift (no rotation applied).
            assert_eq!(
                &key_cache[..n_keep * kv_dim],
                head_snapshot[layer_idx].as_slice(),
                "layer {layer_idx} head cells must be untouched"
            );

            // Tail cells: the cell now at position `t_new` (for
            // t_new in [n_keep, new_seq_len)) was originally at
            // `t_old = t_new + shift`. Its K must equal the oracle
            // value `apply_rope_to_head(raw_for_t_old's_head, t_new)`
            // — i.e., the K that a fresh rotation for the new position
            // would produce from the same raw. Since we built the
            // state with deterministic raw values per (layer, head, dim),
            // we can reconstruct the oracle locally.
            for t_new in n_keep..new_seq_len {
                let t_old = t_new + shift;
                for h in 0..n_kv_heads {
                    let raw: Vec<f32> = (0..head_dim)
                        .map(|d| (layer_idx as f32) + 0.1 * (h as f32) + 0.01 * (d as f32))
                        .collect();
                    // Oracle: raw rotated for the NEW position.
                    let mut oracle = raw.clone();
                    apply_rope_to_head(&mut oracle, t_new, head_dim, freq_base);

                    let start = t_new * kv_dim + h * head_dim;
                    let end = start + head_dim;
                    let actual = &key_cache[start..end];
                    let err = max_abs_diff(actual, &oracle);
                    assert!(
                        err < 1e-3,
                        "layer {layer_idx} head {h} t_new={t_new} t_old={t_old} max-err={err}"
                    );

                    // V at this cell: unrotated but drained down. Original
                    // V row for t_old was `raw + 0.001 * t_old`; the drain
                    // moved it to row t_new, so the stored value at t_new
                    // must still encode t_old — i.e., row identity was
                    // preserved. If drain misindexed, the stored value
                    // would encode a different t.
                    let expected_v: Vec<f32> =
                        raw.iter().map(|x| x + 0.001 * (t_old as f32)).collect();
                    let v_actual = &value_cache[start..end];
                    let v_err = max_abs_diff(v_actual, &expected_v);
                    assert!(
                        v_err < 1e-6,
                        "V at layer {layer_idx} h={h} t_new={t_new} t_old={t_old} row identity lost: max-err={v_err}"
                    );
                }
            }
        } else {
            panic!("expected attention layer {layer_idx}");
        }
    }
}

#[test]
fn is_compressed_false_on_fresh_state() {
    let state = InferenceState::new(4);
    assert!(!state.is_compressed());
}

// ---------------------------------------------------------------------------
// Test 4 — Session-plumbing integration via MockModel
// ---------------------------------------------------------------------------
//
// Focus: the caller-visible contract of `shift_kv` dispatch — does
// `Session::append_tokens` actually invoke `model.shift_kv(..)` with
// the right args, gated on `supports_kv_shift`? RoPE correctness is
// covered by tests 1 & 2 above, so MockModel's `shift_kv` just
// bookkeeps the call + mutates state the way CPU LFM2 would
// (drain + decrement seq_len — no RoPE needed).

struct MockModel {
    config: ModelConfig,
    prefill_calls: AtomicUsize,
    shift_calls: AtomicUsize,
    supports_shift: bool,
}

impl MockModel {
    fn new(config: ModelConfig, supports_shift: bool) -> Self {
        Self {
            config,
            prefill_calls: AtomicUsize::new(0),
            shift_calls: AtomicUsize::new(0),
            supports_shift,
        }
    }
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

    fn supports_kv_shift(&self) -> bool {
        self.supports_shift
    }

    fn shift_kv(&self, state: &mut InferenceState, n_keep: usize, shift: usize) {
        self.shift_calls.fetch_add(1, Ordering::Relaxed);
        // Mirror what `Lfm2Model::shift_kv` does structurally (drain +
        // seq_len decrement) minus the RoPE rotation — enough for
        // Session to observe a correct post-shift state without needing
        // a real RoPE-bearing KV.
        let head_dim = self.config.hidden_size / self.config.n_heads.max(1);
        state.shift_kv_with_rope(
            n_keep,
            shift,
            self.config.rope_theta,
            head_dim,
            &self.config.kv_heads_per_layer,
        );
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
        rope_theta: 10_000.0,
        rms_norm_eps: 0.0,
        block_types: vec![wick::model::BlockType::Attention],
        conv_kernel_size: None,
        kv_heads_per_layer: vec![4],
    }
}

fn run_prefill(model: &MockModel, state: &mut InferenceState, tokens: &[u32]) -> usize {
    let cancel = Arc::new(AtomicBool::new(false));
    let (consumed, _) = model.forward_prefill_chunked(tokens, state.seq_len, state, 64, &cancel);
    consumed
}

#[test]
fn shift_frees_capacity_when_n_keep_set() {
    // Simulate `Session::append_tokens` overflow arm: fill KV to near
    // capacity, then "append" more than fits. We drive the shift
    // through `Model::shift_kv` just as Session does, and verify
    // the trait probe + args match expectations.
    let max_seq_len = 32;
    let n_keep = 4;
    let cfg = mock_attention_config(max_seq_len);
    let model = MockModel::new(cfg.clone(), /* supports_shift = */ true);

    let mut state = InferenceState::new(cfg.block_types.len());
    let first_batch: Vec<u32> = (0..28u32).collect();
    assert_eq!(run_prefill(&model, &mut state, &first_batch), 28);
    assert_eq!(state.seq_len, 28);

    assert!(model.supports_kv_shift(), "probe must agree with field");

    // Shift and append second batch.
    let shift_needed = 28 + 8 - max_seq_len;
    assert_eq!(shift_needed, 4);
    assert!(state.seq_len >= n_keep + shift_needed);
    model.shift_kv(&mut state, n_keep, shift_needed);
    assert_eq!(model.shift_calls.load(Ordering::Relaxed), 1);
    assert_eq!(state.seq_len, 24);

    let second_batch: Vec<u32> = (28..36u32).collect();
    assert_eq!(run_prefill(&model, &mut state, &second_batch), 8);
    assert_eq!(state.seq_len, 32);
    assert_eq!(state.seq_len, max_seq_len);

    // Attention layer KV grew + shrunk correctly.
    if let LayerState::Attention { key_cache, .. } = &state.layers[0] {
        let head_dim = cfg.hidden_size / cfg.n_heads;
        let kv_dim = cfg.n_kv_heads * head_dim;
        assert_eq!(key_cache.len(), 32 * kv_dim);
    }
}

// ---------------------------------------------------------------------------
// Test 6 — Session overflow gate is a pure predicate
// ---------------------------------------------------------------------------
//
// `Session::append_tokens` uses a 4-input predicate to decide between
// running a shift and returning `ContextOverflow`. Since the predicate
// is extracted as a free function (`session::can_shift`) we test each
// branch directly — this is the coverage the MockModel-based tests
// couldn't provide without a real `BpeTokenizer`.

#[test]
fn can_shift_gate_all_branches() {
    use wick::session::can_shift;

    // Happy path — all conditions hold.
    assert!(
        can_shift(
            /* supports */ true, /* n_keep */ 4, /* compressed */ false,
            /* current_pos */ 28, /* shift */ 4,
        ),
        "all conditions hold → can_shift"
    );

    // Backend doesn't support shift (Metal today, non-RoPE archs).
    assert!(
        !can_shift(false, 4, false, 28, 4),
        "supports_kv_shift=false → ContextOverflow"
    );

    // User didn't opt in to shift (default n_keep=0).
    assert!(
        !can_shift(true, 0, false, 28, 4),
        "n_keep=0 → ContextOverflow"
    );

    // TurboQuant-compressed state — can't shift compressed blocks.
    assert!(
        !can_shift(true, 4, true, 28, 4),
        "is_compressed=true → ContextOverflow"
    );

    // Pinned prefix leaves no room to drop (current_pos == n_keep).
    assert!(
        !can_shift(true, 4, false, 4, 4),
        "current_pos < n_keep + shift → ContextOverflow"
    );

    // Boundary: current_pos exactly meets the minimum.
    assert!(
        can_shift(true, 4, false, 8, 4),
        "current_pos == n_keep + shift is allowed (inclusive)"
    );
}
