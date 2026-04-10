## Thinking

Phase 1 implements the compute foundation. The critical path is:
1. Tensor needs f32 conversion methods for CPU ops to work on
2. Quant dequantization must be byte-exact with llama.cpp's ggml-quants.c
3. CPU ops all work on raw f32 slices — no abstraction in the hot path
4. SIMD: on this Mac (aarch64) we target NEON. AVX2 for x86_64.
5. Tests verify quant round-trip, op correctness, and SIMD-vs-naive parity

Q4_K_M is the complex one — 256 values packed into 144 bytes with nested
sub-block scales. Must match llama.cpp's `dequantize_row_q4_K` exactly.

## Plan

1. **tensor.rs** — Add `from_f32_vec`, `as_f32_slice`, `as_f32_slice_mut` methods
2. **quant.rs** — Implement:
   - `dequantize_q8_0_block(block) -> [f32; 32]`
   - `dequantize_q4_k_m_block(block) -> [f32; 256]`
   - `dequantize_q8_0_row(src, dst)` — batch over blocks
   - `dequantize_q4_k_m_row(src, dst)` — batch over blocks
   - `vec_dot_q8_0_f32(block, y) -> f32`
   - `vec_dot_q4_k_m_f32(block, y) -> f32`
3. **cpu.rs** — Naive scalar implementations:
   - `matmul_f32(a, b, c, m, n, k)`
   - `matmul_q8_0_f32(a_quant, b, c, m, n, k)`
   - `matmul_q4km_f32(a_quant, b, c, m, n, k)`
   - `rmsnorm(x, weight, eps)`
   - `silu_inplace(x)`
   - `softmax_inplace(x)`
   - `rope(q, k, pos, head_dim, freq_base)`
   - `conv1d_depthwise(input, weight, bias, output, channels, kernel_size, seq_len)`
   - `add_inplace(a, b)`, `mul_inplace(a, b)`
4. **SIMD** — aarch64 NEON + x86_64 AVX2:
   - `vec_dot_q8_0_f32_neon` / `vec_dot_q8_0_f32_avx2`
   - `vec_dot_q4_k_m_f32_neon` / `vec_dot_q4_k_m_f32_avx2`
   - Runtime feature detection with dispatch functions
5. **Tests** in each module:
   - Quant: dequant correctness, vec_dot vs naive
   - CPU ops: matmul, rmsnorm, softmax, silu, rope, conv1d against reference
   - SIMD vs scalar parity
