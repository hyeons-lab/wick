# 000013 — perf: CPU flash attention for prefill

## Agent
Claude (claude-opus-4-6[1m]) @ wick branch perf/flash-attention

## Intent
Close the remaining 2.4× CPU prefill gap vs llama.cpp (280 vs 662 tok/s at pp2000 with BLAS). The BLAS PR (#13) optimized the GEMM portion; the remaining bottleneck is the naive O(n²) attention loop in `forward_prefill_inner` which has zero cache tiling and zero parallelism. Replace with tiled flash attention (online softmax, TILE_KV=32) and rayon-parallel dispatch across KV heads.

## What Changed
- 2026-04-12T08:26-0700 Timing diagnostic: attention is 50% of BLAS prefill time (6 attention layers × ~600 ms each = 3,600 ms / 7,180 ms total at pp2000). Validated the plan's estimate.
- 2026-04-12T08:45-0700 `wick/src/backend/cpu.rs` — added `flash_attention_gqa_cpu()` kernel (scalar + NEON) with online softmax tiled over TILE_KV=32. Reads Q from stride-n layout (local copy per query), writes contiguous `[group_size, n, head_dim]` output (caller scatter-copies back to stride-n). GQA-aware: processes group_size query heads per call sharing KV tiles. Unit test `test_flash_attention_matches_naive` verifies against the existing `attn_scores + softmax + attn_values` pipeline (max_diff < 1e-4).

## Decisions
- 2026-04-12T08:26-0700 Two-pass decomposition: Pass A (RoPE + cache append, sequential O(n)) then Pass B (flash attention, parallel O(n²)). Separates KV cache construction from attention computation so the full cache is available for tiled access.
- 2026-04-12T08:26-0700 No explicit Q transpose. The flash attention kernel copies each query's head_dim values from the stride-n q_mat into a local contiguous array — cheaper than a full-matrix transpose and avoids extra buffer allocation.
- 2026-04-12T08:26-0700 TurboQuant fallback: keep the existing per-token attention loop for compressed KV. Flash attention for compressed KV is a separate follow-up.

## Issues
(none yet)

## Commits
- HEAD — feat(cpu): add flash_attention_gqa_cpu scalar kernel + parity test

## Next Steps
1. ~~Timing diagnostic~~ ✓ (50% attention fraction)
2. ~~Scalar flash attention kernel + test~~ ✓
3. Smoke test: wire flash attention into one attention layer, benchmark pp2000
4. NEON-optimized inner loops
5. rayon parallel dispatch over KV heads
6. Full rollout + benchmark matrix + docs
