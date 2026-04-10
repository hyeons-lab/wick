# 000009 — perf/gemm-tuning

**Agent:** Claude Code (claude-opus-4-6) @ wick branch perf/gemm-tuning

## Intent

Tune GEMM and GEMV micro-kernels: software prefetch, 8-column grouping, par_rows_n threshold.

## What Changed

- 2026-04-03T22:54-0700 `wick/src/backend/simd.rs` — 8-column grouped Q4_0 GEMM (was 4-column), software prefetch for Q4_0/Q8_0 GEMV weight blocks
- 2026-04-03T22:54-0700 `wick/src/lib.rs` — Enable `stdarch_aarch64_prefetch` feature

## Decisions

- 2026-04-03T22:30-0700 Software prefetch: marginal benefit on Apple Silicon (~1-3%). Hardware prefetcher is already effective. Kept it since it doesn't hurt and may help on other ARM chips.
- 2026-04-03T22:40-0700 par_rows_n min_rows: tested 8 vs 64, no difference. Kept at 64.
- 2026-04-03T22:50-0700 8-column Q4_0 GEMM: ~8% improvement for 32 tokens, larger for 117 tokens. Halves weight decode count. 16 accumulator registers + 4 weight vectors = 20 NEON regs, within budget. Falls back to 4-column for n%8 remainder.

## Commits

- HEAD — perf: 8-column Q4_0 GEMM + software prefetch

## Research & Discoveries

- Apple Silicon hardware prefetch is excellent — software prefetch barely helps
- 8-column grouping: 450M 32tok 440→475 (+8%), 450M 117tok 500→539 (+8%)
- Diminishing returns: the per-thread GEMM compute is already efficient, bottleneck shifts to rayon dispatch and sequential attention at longer sequences
