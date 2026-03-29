# 000002 — feature/tensor-cpu-compute

**Agent:** Claude Code (claude-opus-4-6) @ repository branch feature/tensor-cpu-compute

**Intent:** Implement Phase 1 of Wick — tensor types, Q4_K_M/Q8_0 dequantization, all CPU compute ops (naive + SIMD), and tests proving correctness.

## What Changed

- 2026-03-29T16:26-0700 Cargo.toml (root) — added bytemuck workspace dependency
- 2026-03-29T16:26-0700 wick/Cargo.toml — added bytemuck dep
- 2026-03-29T16:26-0700 wick/src/tensor.rs — added from_f32, zeros_f32, as_f32_slice, as_f32_slice_mut, to_f32_vec (with auto-dequant for Q8_0/Q4_K_M/F16/BF16)
- 2026-03-29T16:30-0700 wick/src/quant.rs — full Q8_0 and Q4_K_M dequantization, scalar vec_dot, scale decoding. Ported from llama.cpp's ggml-quants.c layout.
- 2026-03-29T16:32-0700 wick/src/backend/cpu.rs — all naive CPU ops: matmul_f32, matmul_q8_0_f32, matmul_q4km_f32, rmsnorm, silu, softmax, rope, conv1d_depthwise, add_inplace, mul_inplace
- 2026-03-29T16:34-0700 wick/src/backend/simd.rs — NEON (aarch64) and AVX2 (x86_64) optimized vec_dot for Q8_0 and Q4_K_M with runtime dispatch
- 2026-03-29T16:34-0700 wick/src/backend/mod.rs — added simd module

## Decisions

- 2026-03-29T16:26-0700 Used bytemuck for f32 ↔ byte slice casting — zero-copy, well-maintained, no unsafe in our code
- 2026-03-29T16:30-0700 Q4_K_M scale decoding follows llama.cpp's get_scale_min_k4 exactly — 6-bit scales packed into 12 bytes with split high bits
- 2026-03-29T16:32-0700 Quantized matmul extracts column slices into temp Vec for vec_dot — correct but not fast. OK for Phase 1 correctness; will optimize in Phase 5 when weights are transposed.
- 2026-03-29T16:34-0700 SIMD dispatch: compile-time cfg for aarch64 (NEON always available), runtime feature detection for x86_64 AVX2+FMA
- 2026-03-29T16:34-0700 Rust 2024 edition requires unsafe blocks inside unsafe fn — wrapped all SIMD intrinsic calls in explicit unsafe blocks

## Issues

- Rust 2024 `unsafe_op_in_unsafe_fn` lint: first SIMD build had ~12 warnings because intrinsics inside `unsafe fn` need explicit `unsafe {}` blocks. Fixed by wrapping function bodies.
- Clippy `needless_range_loop`: refactored Q8_0 dequant/dot to use iterators.

## Commits

- HEAD — feat: implement tensor ops, quantization, CPU compute, and SIMD kernels (Phase 1)
