# TurboQuant KV Cache Compression

**Agent:** Claude Code (claude-opus-4-6) @ repository branch feat/turboquant-kv-cache

**Intent:** Implement TurboQuant (arXiv:2504.19874) for KV cache key compression — 3-bit per element via PolarQuant (2-bit) + QJL (1-bit). Targets ~10x key cache memory reduction with near-lossless accuracy, no calibration required.

## What Changed

- 2026-04-09T17:57-0700 wick/src/turboquant.rs — NEW: core TurboQuant module with RHT, PolarQuant, QJL, Lloyd-Max solver, CompressedKeyCache, attention score estimator, 9 unit tests
- 2026-04-09T17:57-0700 wick/src/kv_cache.rs — extended LayerState::Attention with compressed_keys field, added KeyCompression enum, from_config_with_compression()
- 2026-04-09T17:57-0700 wick/src/model/lfm2.rs — wired TurboQuant into forward_attn_block() and forward_prefill(), added enable_turboquant()/tq_rotations/tq_config to model
- 2026-04-09T17:57-0700 wick/src/model/mod.rs — added enable_turboquant() and turboquant_enabled() to Model trait
- 2026-04-09T17:57-0700 wick/src/engine.rs — thread KeyCompression through GenerateConfig → InferenceState
- 2026-04-09T17:57-0700 wick-cli/src/main.rs — added --kv-cache-keys flag (f32 | tq3) to Run and Bench commands
- 2026-04-09T17:57-0700 wick/src/lib.rs — added pub mod turboquant
- 2026-04-09T18:50-0700 wick/src/backend/cpu.rs — added NEON attn_scores_turboquant_neon kernel with centroid-select FMA and branchless QJL bit-to-f32 mask

## Decisions

- 2026-04-09T16:11-0700 Use RHT (Randomized Hadamard Transform) instead of paper's dense QR-of-Gaussian rotation — O(d log d) vs O(d^2), O(d) storage. Empirically equivalent for d >= 64.
- 2026-04-09T16:11-0700 Keys-only compression (values stay f32) — keys benefit much more from compression due to K/V norm asymmetry. Value Q8_0 is Phase 6.
- 2026-04-09T16:11-0700 CPU path only initially — GPU model (GpuLfm2Model) has separate wgpu buffer KV management, deferred to future phase.
- 2026-04-09T16:11-0700 Stack-allocated scratch for rotated queries — avoids ScratchBuffers borrow conflicts during attention loop.
- 2026-04-09T16:11-0700 Lloyd-Max centroids computed at init time — blog-post constants may lack precision for specific head_dim values.
- 2026-04-09T16:11-0700 QJL residual computed in rotated space — avoids extra inverse+forward rotation. JL projection applied to rotated query accordingly.

## Issues

- 2026-04-09T18:16-0700 Perf review found 8 issues: redundant RHT per GQA head, heap allocs per call (5+ per token per layer), f16 conversion in inner loop, suboptimal PolarQuant dot product, branchy QJL signed sum, 3-pass RHT, 2-pass quantize+pack, no GQA batching. All fixed.

## Commits

33890ab — feat: implement TurboQuant KV cache key compression
d588d5f — perf: optimize TurboQuant hot paths (8 fixes)
b066957 — perf: add NEON SIMD kernel for TurboQuant attention
HEAD — fix: address PR review comments (7 items)

## Next Steps

- Value cache Q8_0 compression (Phase 6)
- GPU backend support (Phase 6)
