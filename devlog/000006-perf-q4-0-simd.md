# Devlog: perf/q4-0-simd

**Agent:** Claude Code (claude-opus-4-6) @ wick branch perf/q4-0-simd

## Intent

Add NEON (aarch64) and AVX2 (x86_64) SIMD kernels for Q4_0 vec_dot to close the performance gap vs Q8_0. Also fix a decode bug with raw byte tokens from PR #4 review.

## What Changed

- 2026-04-02T23:02-0700 wick/src/backend/simd.rs — Added vec_dot_q4_0_f32_neon and vec_dot_q4_0_f32_avx2 + dispatch
- 2026-04-02T23:02-0700 wick/src/quant.rs — Wired vec_dot_q4_0_f32 dispatch to SIMD path
- 2026-04-02T23:02-0700 wick/src/tokenizer.rs — Fixed decode() to handle non-UTF8 byte tokens

## Performance

LFM2-VL-450M Q4_0 decode: 5.7 → 9.3 tok/s (1.6x speedup). Q8_0 reference: 30.5 tok/s.

## Commits

HEAD — perf: add Q4_0 SIMD kernels (NEON/AVX2) and fix byte token decode
