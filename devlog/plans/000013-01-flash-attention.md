# 000013-01 — Flash attention CPU prefill plan

## Thinking

Post-BLAS prefill at pp2000: 280 tok/s (wick BLAS) vs 662 tok/s (llama.cpp). GEMM is optimized; attention is 50% of remaining time. The attention loop in `forward_prefill_inner` processes tokens sequentially with no tiling and no parallelism.

Two independent wins: tiled flash attention (~2-3× via L1-resident KV tiles) and rayon parallelism across 8 KV heads (~4-6×). Combined: ~8-12× on the 50% attention portion → end-to-end ~1.7-2× → target 480-620 tok/s.

## Plan

Scalar kernel → NEON → rayon → full rollout → threshold calibration.

Two-pass decomposition: Pass A (RoPE + cache append, sequential O(n)) then
Pass B (flash attention, rayon-parallel O(n²)). TILE_KV=32, online softmax
with ggml_expf + f64 sum. Contiguous output per KV head group, scatter-copy
back to stride-n layout. Threshold at 256 tokens (measured crossover on M1 Max).
