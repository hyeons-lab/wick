# 000008 — perf/neon-attention

**Agent:** Claude Code (claude-opus-4-6) @ wick branch perf/neon-attention

## Intent

Vectorize the attention score computation (Q×K dot products) and weighted value summation (softmax×V) using NEON SIMD. These scalar loops are the dominant cost in the per-token sequential attention core, especially at longer sequence lengths.

## What Changed

- 2026-04-03T21:34-0700 `wick/src/backend/cpu.rs` — Added `attn_scores` and `attn_values` dispatch functions with NEON implementations (`attn_scores_neon`, `attn_values_neon`). Uses vfmaq_f32 for fused multiply-add, processes head_dim in chunks of 8 (two float32x4). Scalar fallback on non-aarch64.
- 2026-04-03T21:34-0700 `wick/src/model/lfm2.rs` — Replaced scalar attention loops in both `forward_attn_block` (decode) and `forward_prefill` (prefill attention core) with calls to `cpu::attn_scores` and `cpu::attn_values`.

## Decisions

- 2026-04-03T21:34-0700 Put NEON attention in cpu.rs (not simd.rs) — these operate on f32 slices, not quantized data. simd.rs is for quantized kernels.
- 2026-04-03T21:34-0700 Process 8 elements per iteration (two float32x4) for ILP — head_dim=64 gives exactly 8 iterations with zero scalar tail.

## Commits

- HEAD — perf: NEON-vectorized attention scores and weighted values

## Research & Discoveries

- For 450M (head_dim=64), NEON attention is a ~2x speedup on the attention loops
- 117-token prefill: 260 → 497 tok/s (1.9x) — attention is a larger fraction at longer sequences  
- 32-token prefill: 455 → 461 tok/s (marginal) — GEMM dominates at short sequences
- Decode: 115 → 115 tok/s (flat) — GEMV dominates decode for small models
- The win will be more visible on larger models with more attention layers and longer contexts
