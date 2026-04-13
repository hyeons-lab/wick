# 000015 — perf: KV cache pre-allocation

**Agent:** Claude (claude-opus-4-6) @ wick branch perf/kv-ring

## Intent

Eliminate Vec-doubling reallocations in the CPU KV cache by plumbing `context_size` through the loader API and pre-allocating f32 KV vecs to exactly `max_seq_len * kv_dim` per attention layer. Mirrors the pattern already used by the Metal/wgpu loaders.

## What Changed

2026-04-13T13:31-0700 `wick/src/model/lfm2.rs` — `Lfm2Model::from_gguf` now takes `context_size: usize` and caps `max_seq_len = context_size.min(gguf_max)`.
2026-04-13T13:31-0700 `wick/src/model/mod.rs` — `load_model` takes `context_size`, passes through to LFM2 loader.
2026-04-13T13:31-0700 `wick/src/model/gpu_lfm2.rs`, `wick/src/model/metal_lfm2.rs` — pass `context_size` to inner `Lfm2Model::from_gguf` (already had it in scope).
2026-04-13T13:31-0700 `wick/src/kv_cache.rs` — `InferenceState::from_config_with_compression` pre-allocates each attention layer's f32 key/value vec with `Vec::with_capacity(max_seq_len * kv_dim)`. Skipped when TurboQuant compression is active for that side (the f32 vec stays empty and the compressed cache owns storage).
2026-04-13T13:31-0700 `wick-cli/src/main.rs` — 2 `load_model` call sites pass `context_size`.
2026-04-13T13:31-0700 `wick/tests/{llm_layer0_compare,quality_gate,blas_parity,bench_perf}.rs` — 10 test call sites default to `context_size = 8192`.

## Decisions

2026-04-13T13:31-0700 **Plumb `context_size` through the loader rather than picking a fixed `INITIAL_KV_CAPACITY` constant.** A fixed cap (e.g. 16384) overshoots typical short-conversation workloads — a 4k chat would always pay for 16k of f32 storage. Plumbing the real cap matches Metal's existing pattern and keeps RSS tight to actual workload.

2026-04-13T13:31-0700 **Default test `context_size` = 8192.** Matches the CLI's default-ish range and is large enough that tests don't hit cache caps but small enough that test RSS stays modest.

## Verification

| Check | Result |
|---|---|
| `cargo fmt && cargo clippy --workspace -- -D warnings` | clean |
| `cargo clippy --workspace --features blas -- -D warnings` | clean |
| `cargo test --workspace` | all pass |
| `cargo check --target x86_64-unknown-linux-gnu` | clean |
| Coherence: "The capital of France is" | → " Paris." |

### RSS + throughput, 4001-token prompt, `--context-size 8192`, 32 decode tokens (LFM2.5-VL-1.6B-Q4_0)

| Metric | Before | After | Δ |
|---|---|---|---|
| Decode tok/s | 23.8 | 26.9 | **+13.0%** |
| Prefill tok/s | 150.9 | 145.4 | -3.6% (within noise) |
| Peak memory footprint | 858 MB | 832 MB | **-26 MB** |
| Max RSS | 1553 MB | 1527 MB | **-26 MB** |

The 13% decode improvement is the headline. With ~13 Vec doublings during a long decode and several MB per memcpy each time, that overhead dominated the steady-state decode loop more than expected. The 26 MB RSS drop is the doubled-capacity slack going away.

## Commits

- HEAD — perf(kv-cache): pre-allocate f32 KV vecs via context_size loader API
