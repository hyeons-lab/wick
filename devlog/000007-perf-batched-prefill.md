# 000007 — perf/batched-prefill

**Agent:** Claude Code (claude-opus-4-6) @ wick branch perf/batched-prefill

## Intent

Improve prefill (prompt processing) throughput by using batched GEMM instead of sequential per-token GEMV. Each weight matrix is read from memory once and dotted against all N token vectors, reducing memory bandwidth by ~Nx.

## What Changed

- 2026-04-03T15:00-0700 `wick/src/backend/simd.rs` — Added `gemm_q4_0_q8_0_neon`: batched Q4_0×Q8_0 GEMM kernel with rayon row parallelism
- 2026-04-03T15:00-0700 `wick/src/backend/cpu.rs` — Added `par_rows_n` helper for GEMM row parallelism (rows of width n)
- 2026-04-03T15:00-0700 `wick/src/model/lfm2.rs` — Wired batched GEMM into `forward_prefill()` FFN for Q4_0 weights; hoisted all GEMM buffers outside per-layer loop

## Decisions

- 2026-04-03T17:30-0700 Column-first GEMM loop order — iterating over columns (tokens) in the inner loop and blocks in the outer loop was 2x slower due to cache thrashing of Q8_0 data across N scattered column locations. Column-first (process full dot product per column) keeps each column's Q8_0 data sequential in cache while weight data stays warm in L2. Despite not amortizing weight decode, the cache behavior dominates.
- 2026-04-03T17:30-0700 Block-first loop would be better with column-interleaved Q8_0 layout (q8[block][column]) but would require changing the quantization format — not worth the complexity.

## Issues

- Block-first GEMM loop (decode weight blocks once, dot all N columns): 109 tok/s — SLOWER than per-token GEMV because Q8_0 column data is laid out as col0_all_blocks, col1_all_blocks, etc. Inner loop over columns creates random access.
- Partially applied edit from previous session left unclosed delimiter in lfm2.rs — fixed by reverting and reapplying cleanly.

## Commits

- 978646a — perf: batched prefill — 2x faster prompt processing
- 13f3234 — chore: remove patch artifact files
- 5a322d4 — perf: hoist all allocations outside per-layer/per-token loops in prefill
- HEAD — perf: batched FFN GEMM — 1.6-2.6x faster prefill

## Research & Discoveries

- For small N (6 tokens), GEMM overhead dominates and there's no speedup
- For N=32: 450M sees 1.6x speedup, 1.6B sees 2.6x speedup (larger matrices = more weight reuse)
- After GEMM FFN optimization, the bottleneck shifts to sequential conv/attention layers
- Theoretical FFN-only limit for 450M at 30 GB/s: ~14k tok/s — conv/attn now dominate

## Next Steps

- Batch attention Q/K/V projections as GEMM (biggest remaining linear projection win)
- Batch conv in_proj/out_proj as GEMM
- Consider interleaved Q8_0 layout for block-first GEMM (more complex, bigger payoff at large N)
