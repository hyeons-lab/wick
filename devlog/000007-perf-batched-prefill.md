# 000007 — perf/batched-prefill

**Agent:** Claude Code (claude-opus-4-6) @ wick branch perf/batched-prefill

## Intent

Improve prefill (prompt processing) throughput by using batched GEMM instead of sequential per-token GEMV. Each weight matrix is read from memory once and dotted against all N token vectors, reducing memory bandwidth by ~Nx.

## What Changed

- 2026-04-03T15:00-0700 `wick/src/backend/simd.rs` — Added `gemm_q4_0_q8_0_neon`: batched Q4_0×Q8_0 GEMM kernel with rayon row parallelism
- 2026-04-03T15:00-0700 `wick/src/backend/cpu.rs` — Added `par_rows_n` helper for GEMM row parallelism (rows of width n)
- 2026-04-03T15:00-0700 `wick/src/model/lfm2.rs` — Wired batched GEMM into `forward_prefill()` FFN for Q4_0 weights; hoisted all GEMM buffers outside per-layer loop
- 2026-04-03T17:54-0700 `wick/src/model/lfm2.rs` — Batched conv in_proj/out_proj and attn Q/K/V/output projections as GEMM in forward_prefill; restructured per-token loop into 3 phases (batch input proj → sequential core → batch output proj); hoisted q_mat/k_mat/v_mat outside layer loop

## Decisions

- 2026-04-03T17:30-0700 Column-first GEMM loop order — iterating over columns (tokens) in the inner loop and blocks in the outer loop was 2x slower due to cache thrashing of Q8_0 data across N scattered column locations. Column-first (process full dot product per column) keeps each column's Q8_0 data sequential in cache while weight data stays warm in L2.
- 2026-04-03T17:54-0700 Three-phase prefill for conv/attn blocks — batch input projections via GEMM, run sequential core (conv rolling buffer / attention scores) per-token, batch output projections via GEMM. This batches all batchable linear projections while keeping inherently sequential ops per-token.

## Issues

- Block-first GEMM loop (decode weight blocks once, dot all N columns): 109 tok/s — SLOWER than per-token GEMV because Q8_0 column data is laid out as col0_all_blocks, col1_all_blocks. Inner loop over columns creates random access.
- Partially applied edit from previous session left unclosed delimiter in lfm2.rs — fixed by reverting and reapplying cleanly.
- Borrow checker: split_at_mut needed for conv_proj[b, c, x] non-overlapping slices.

## Commits

- 978646a — perf: batched prefill — 2x faster prompt processing
- 13f3234 — chore: remove patch artifact files
- 5a322d4 — perf: hoist all allocations outside per-layer/per-token loops in prefill
- 3ace668 — perf: batched FFN GEMM — 1.6-2.6x faster prefill
- HEAD — perf: batched conv/attn projections via GEMM — 2.5x total prefill speedup

## Research & Discoveries

- For small N (6 tokens), GEMM overhead dominates and there's no speedup
- For N=32: 450M sees 1.6x FFN-only speedup, 1.6B sees 2.6x
- Batching conv/attn projections adds another 1.6x on top of FFN GEMM
- Total speedup from all GEMM: 450M 2.5x (148→365), 1.6B 3.4x (37→125)
- Q8_0 model doesn't benefit yet — no Q8_0 GEMM kernel implemented

## Next Steps

- Add Q8_0×Q8_0 GEMM kernel for Q8_0 model prefill
- Add Q6_K×Q8_0 GEMM kernel for output projection batching
- Consider interleaved Q8_0 layout for block-first GEMM at large N
