# Plan 000015-01: KV cache pre-allocation via `context_size` API

## Thinking

The CPU `Lfm2Model::from_gguf` doesn't take a `context_size` parameter — it always uses the GGUF metadata's `context_length` as `max_seq_len` (128000 for LFM2.5-VL). Metal and wgpu loaders both accept `context_size` and cap `max_seq_len = context_size.min(gguf_max)`.

This asymmetry has a real cost. `InferenceState::from_config_with_compression` constructs each per-layer KV vec as `Vec::new()` and grows it via `extend_from_slice` during decode/prefill. Vec doubles capacity each grow, which means:
1. **Wasted RSS** — capacity overshoots actual length by up to 2×.
2. **Memcpy on every doubling** — full cache copied to a new allocation. ~13 doublings during a long decode, each touching MBs of f32 data.

The fix matches what Metal already does: accept `context_size`, cap `max_seq_len`, and pre-allocate the f32 KV vecs to exactly `max_seq_len * kv_dim` per layer. `extend_from_slice` is now amortized O(1), no realloc.

I considered a fixed `INITIAL_KV_CAPACITY` constant (e.g. 16384) instead of plumbing `context_size`, but that overshoots typical workloads — a 4k-token chat would always pay 16k worth of f32. The proper fix is to plumb the actual cap through the API.

## Plan

### API changes

1. `wick/src/model/lfm2.rs` — `from_gguf(gguf)` → `from_gguf(gguf, context_size: usize)`. Cap `max_seq_len = context_size.min(gguf_max)`.
2. `wick/src/model/mod.rs` — `load_model(gguf)` → `load_model(gguf, context_size: usize)`. Pass through.
3. `wick/src/model/gpu_lfm2.rs`, `wick/src/model/metal_lfm2.rs` — pass `context_size` to inner `Lfm2Model::from_gguf`.

### Pre-allocation

In `InferenceState::from_config_with_compression`, attention layers:
```rust
let kv_dim = n_kv_heads * head_dim;
let cap = config.max_seq_len * kv_dim;
let key_cache = if compress_keys && n_kv_heads > 0 {
    Vec::new()  // TurboQuant path: f32 vec stays empty
} else {
    Vec::with_capacity(cap)
};
// same for value_cache
```

### Call site updates

- `wick-cli/src/main.rs` (2 sites) — pass `context_size` (already in scope).
- Tests (10 sites across 4 files) — default to 8192.

## Verification

1. `cargo fmt && cargo clippy --workspace -- -D warnings`
2. `cargo clippy --workspace --features blas -- -D warnings`
3. `cargo test --workspace`
4. `cargo check --target x86_64-unknown-linux-gnu`
5. Coherence: "The capital of France is" → "Paris."
6. RSS measurement at 4k prompt + 8k context, before vs after.
7. Throughput at long context, before vs after.
