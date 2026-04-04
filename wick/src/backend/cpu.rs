// CPU compute backend — naive scalar implementations.
//
// All functions operate on raw f32 slices. No Tensor abstraction in the hot path.

use crate::quant::{
    BlockQ4_0, BlockQ4KM, BlockQ8_0, vec_dot_q4_0_f32, vec_dot_q4_k_m_f32, vec_dot_q8_0_f32,
};
#[cfg(not(target_arch = "aarch64"))]
use crate::quant::{BlockQ6K, vec_dot_q6_k_f32};
use crate::tensor::DType;
use std::mem::size_of;

// ── Matrix multiplication ───────────────────────────────────────────────────

/// Dense f32 matrix multiply: C[m,n] = A[m,k] * B[k,n] (row-major).
///
/// `c` must be pre-zeroed.
pub fn matmul_f32(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);

    for i in 0..m {
        for p in 0..k {
            let a_val = a[i * k + p];
            for j in 0..n {
                c[i * n + j] += a_val * b[p * n + j];
            }
        }
    }
}

/// Quantized Q4_0 × f32 matmul: C[m,n] = dequant(A_q4_0)[m,k] * B[k,n].
///
/// `a_quant` is raw Q4_0 bytes, row-major with `m` rows of `k` elements each.
/// Each row is k/32 blocks of 18 bytes.
pub fn matmul_q4_0_f32(a_quant: &[u8], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    debug_assert_eq!(k % 32, 0);
    let blocks_per_row = k / 32;
    let bytes_per_row = blocks_per_row * size_of::<BlockQ4_0>();
    debug_assert_eq!(a_quant.len(), m * bytes_per_row);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);

    for i in 0..m {
        let row_start = i * bytes_per_row;
        for j in 0..n {
            let mut sum = 0.0f32;
            for bi in 0..blocks_per_row {
                let block_offset = row_start + bi * size_of::<BlockQ4_0>();
                let block = unsafe { &*(a_quant.as_ptr().add(block_offset) as *const BlockQ4_0) };
                let col_start = bi * 32;
                let b_slice: Vec<f32> = (0..32).map(|l| b[(col_start + l) * n + j]).collect();
                sum += vec_dot_q4_0_f32(block, &b_slice);
            }
            c[i * n + j] = sum;
        }
    }
}

/// Quantized Q8_0 × f32 matmul: C[m,n] = dequant(A_q8)[m,k] * B[k,n].
///
/// `a_quant` is raw Q8_0 bytes, row-major with `m` rows of `k` elements each.
/// Each row is k/32 blocks of 34 bytes.
pub fn matmul_q8_0_f32(a_quant: &[u8], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    debug_assert_eq!(k % 32, 0);
    let blocks_per_row = k / 32;
    let bytes_per_row = blocks_per_row * size_of::<BlockQ8_0>();
    debug_assert_eq!(a_quant.len(), m * bytes_per_row);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);

    for i in 0..m {
        let row_start = i * bytes_per_row;
        for j in 0..n {
            let mut sum = 0.0f32;
            for bi in 0..blocks_per_row {
                let block_offset = row_start + bi * size_of::<BlockQ8_0>();
                let block = unsafe { &*(a_quant.as_ptr().add(block_offset) as *const BlockQ8_0) };
                // Extract the 32-element column slice from B
                let col_start = bi * 32;
                let b_slice: Vec<f32> = (0..32).map(|l| b[(col_start + l) * n + j]).collect();
                sum += vec_dot_q8_0_f32(block, &b_slice);
            }
            c[i * n + j] = sum;
        }
    }
}

/// Quantized Q4_K_M × f32 matmul: C[m,n] = dequant(A_q4km)[m,k] * B[k,n].
///
/// `a_quant` is raw Q4_K_M bytes, row-major with `m` rows of `k` elements each.
/// Each row is k/256 blocks of 144 bytes.
pub fn matmul_q4km_f32(a_quant: &[u8], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    debug_assert_eq!(k % 256, 0);
    let blocks_per_row = k / 256;
    let bytes_per_row = blocks_per_row * size_of::<BlockQ4KM>();
    debug_assert_eq!(a_quant.len(), m * bytes_per_row);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);

    for i in 0..m {
        let row_start = i * bytes_per_row;
        for j in 0..n {
            let mut sum = 0.0f32;
            for bi in 0..blocks_per_row {
                let block_offset = row_start + bi * size_of::<BlockQ4KM>();
                let block = unsafe { &*(a_quant.as_ptr().add(block_offset) as *const BlockQ4KM) };
                let col_start = bi * 256;
                let b_slice: Vec<f32> = (0..256).map(|l| b[(col_start + l) * n + j]).collect();
                sum += vec_dot_q4_k_m_f32(block, &b_slice);
            }
            c[i * n + j] = sum;
        }
    }
}

// ── GEMV (matrix-vector multiply) ──────────────────────────────────────────

/// Parallel for_each with chunking to prevent over-splitting.
/// Each thread gets at least `min_rows` rows to amortize rayon dispatch overhead.
pub fn par_rows(y: &mut [f32], min_rows: usize, f: impl Fn((usize, &mut f32)) + Sync + Send) {
    use rayon::iter::{IndexedParallelIterator, ParallelIterator};
    use rayon::slice::ParallelSliceMut;
    let chunk_size = (y.len() / rayon::current_num_threads()).max(min_rows);
    y.par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(ci, chunk)| {
            let base = ci * chunk_size;
            for (j, yi) in chunk.iter_mut().enumerate() {
                f((base + j, yi));
            }
        });
}

/// Like `par_rows` but for GEMM output where each "row" is `n` contiguous f32 elements.
/// `f` receives (row_index, &mut [f32; n]).
pub fn par_rows_n(
    y: &mut [f32],
    n: usize,
    min_rows: usize,
    f: impl Fn((usize, &mut [f32])) + Sync + Send,
) {
    debug_assert_ne!(n, 0, "par_rows_n: n must be > 0");
    if n == 0 || y.is_empty() {
        return;
    }
    use rayon::iter::{IndexedParallelIterator, ParallelIterator};
    use rayon::slice::ParallelSliceMut;
    let m = y.len() / n;
    let rows_per_chunk = (m / rayon::current_num_threads()).max(min_rows);
    let elems_per_chunk = rows_per_chunk * n;
    y.par_chunks_mut(elems_per_chunk)
        .enumerate()
        .for_each(|(ci, chunk)| {
            let base_row = ci * rows_per_chunk;
            for (j, row) in chunk.chunks_mut(n).enumerate() {
                f((base_row + j, row));
            }
        });
}

#[allow(clippy::ptr_arg)]
/// Q4_0 GEMV: y[m] = A_q4_0[m,k] @ x[k].
///
/// On aarch64, uses integer dot product with caller-provided Q8_0 scratch buffers
/// to avoid per-call heap allocation. The scratch buffers are resized as needed.
pub fn gemv_q4_0_f32(
    a_quant: &[u8],
    x: &[f32],
    y: &mut [f32],
    m: usize,
    k: usize,
    q8_scales: &mut Vec<f32>,
    q8_quants: &mut Vec<i8>,
) {
    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(y.len(), m);
    debug_assert_eq!(k % 32, 0, "Q4_0 GEMV: k must be divisible by 32");
    let blocks_per_row = k / 32;
    let row_bytes = blocks_per_row * size_of::<BlockQ4_0>();
    debug_assert_eq!(a_quant.len(), m * row_bytes);

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            crate::backend::simd::neon::gemv_q4_0_f32_neon(
                a_quant, x, y, m, k, q8_scales, q8_quants,
            );
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = (q8_scales, q8_quants);
        for (i, yi) in y.iter_mut().enumerate() {
            let row_start = i * row_bytes;
            let mut sum = 0.0f32;
            for bi in 0..blocks_per_row {
                let offset = row_start + bi * size_of::<BlockQ4_0>();
                let block = unsafe { &*(a_quant.as_ptr().add(offset) as *const BlockQ4_0) };
                sum += vec_dot_q4_0_f32(block, &x[bi * 32..(bi + 1) * 32]);
            }
            *yi = sum;
        }
    }
}

/// Minimum output dimension to use parallel GEMV (avoid rayon overhead for small ops).
pub const GEMV_PAR_THRESHOLD: usize = 256;

/// Quantize f32 vector to Q8_0 format for use with `gemv_q4_0_with_q8`.
/// Returns (scales, quants). On aarch64, uses NEON-vectorized quantization.
#[cfg(target_arch = "aarch64")]
pub fn quantize_f32_to_q8_0(x: &[f32]) -> (Vec<f32>, Vec<i8>) {
    assert_eq!(
        x.len() % 32,
        0,
        "quantize_f32_to_q8_0: x.len() must be divisible by 32"
    );
    let n_blocks = x.len() / 32;
    let mut scales = vec![0.0f32; n_blocks];
    let mut quants = vec![0i8; x.len()];
    unsafe {
        crate::backend::simd::neon::quantize_f32_to_q8_0_neon(x, &mut scales, &mut quants);
    }
    (scales, quants)
}

/// GEMV with pre-quantized Q8_0 input. Dispatches to Q4_0 or Q8_0 integer path.
/// For other dtypes, falls back to the regular f32 path.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
pub fn gemv_with_preq(
    dtype: DType,
    a_quant: &[u8],
    x_scales: &[f32],
    x_quants: &[i8],
    x_f32: &[f32],
    y: &mut [f32],
    m: usize,
    k: usize,
) {
    match dtype {
        DType::Q4_0 => gemv_q4_0_with_q8(a_quant, x_scales, x_quants, y, m, k),
        DType::Q8_0 => unsafe {
            crate::backend::simd::neon::gemv_q8_0_q8_0_neon(a_quant, x_scales, x_quants, y, m, k)
        },
        DType::Q6K => unsafe {
            crate::backend::simd::neon::gemv_q6k_q8_0_neon(a_quant, x_scales, x_quants, y, m, k)
        },
        _ => gemv_dispatch(dtype, a_quant, x_f32, y, m, k, None),
    }
}

/// Q4_0 GEMV with pre-quantized Q8_0 input. Avoids re-quantizing x when
/// the same input is used for multiple weight matrices (e.g., ffn_gate + ffn_up).
#[cfg(target_arch = "aarch64")]
pub fn gemv_q4_0_with_q8(
    a_quant: &[u8],
    x_scales: &[f32],
    x_quants: &[i8],
    y: &mut [f32],
    m: usize,
    k: usize,
) {
    unsafe {
        crate::backend::simd::neon::gemv_q4_0_q8_0_neon(a_quant, x_scales, x_quants, y, m, k);
    }
}

#[allow(clippy::ptr_arg)]
/// Q8_0 GEMV: y[m] = A_q8_0[m,k] @ x[k].
/// On aarch64, uses integer dot product (quantize x to Q8_0, then Q8_0 × Q8_0
/// with vdotq_s32 — ~4x fewer instructions than f32 widening path).
pub fn gemv_q8_0_f32(
    a_quant: &[u8],
    x: &[f32],
    y: &mut [f32],
    m: usize,
    k: usize,
    q8_scales: &mut Vec<f32>,
    q8_quants: &mut Vec<i8>,
) {
    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(y.len(), m);
    debug_assert_eq!(k % 32, 0, "Q8_0 GEMV: k must be divisible by 32");
    let blocks_per_row = k / 32;
    let row_bytes = blocks_per_row * size_of::<BlockQ8_0>();
    debug_assert_eq!(a_quant.len(), m * row_bytes);

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            crate::backend::simd::neon::gemv_q8_0_f32_neon(
                a_quant, x, y, m, k, q8_scales, q8_quants,
            );
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = (q8_scales, q8_quants);
        let compute_row = |(i, yi): (usize, &mut f32)| {
            let row_start = i * row_bytes;
            let mut sum = 0.0f32;
            for bi in 0..blocks_per_row {
                let offset = row_start + bi * size_of::<BlockQ8_0>();
                let block = unsafe { &*(a_quant.as_ptr().add(offset) as *const BlockQ8_0) };
                sum += vec_dot_q8_0_f32(block, &x[bi * 32..(bi + 1) * 32]);
            }
            *yi = sum;
        };

        if m >= GEMV_PAR_THRESHOLD {
            par_rows(y, 512, compute_row);
        } else {
            y.iter_mut().enumerate().for_each(compute_row);
        }
    }
}

/// Q6_K GEMV: y[m] = A_q6k[m,k] @ x[k]. Parallelized across rows.
/// On aarch64, quantizes x to Q8_0 then uses integer Q6_K × Q8_0 dot product with vdotq_s32.
#[allow(clippy::ptr_arg)]
#[allow(unused_variables)]
pub fn gemv_q6k_f32(
    a_quant: &[u8],
    x: &[f32],
    y: &mut [f32],
    m: usize,
    k: usize,
    q8_scales: &mut Vec<f32>,
    q8_quants: &mut Vec<i8>,
) {
    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(y.len(), m);
    debug_assert_eq!(k % 256, 0, "Q6_K GEMV: k must be divisible by 256");

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            crate::backend::simd::neon::gemv_q6k_f32_neon(
                a_quant, x, y, m, k, q8_scales, q8_quants,
            );
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let blocks_per_row = k / 256;
        let row_bytes = blocks_per_row * size_of::<BlockQ6K>();
        debug_assert_eq!(a_quant.len(), m * row_bytes);

        let compute_row = |(i, yi): (usize, &mut f32)| {
            let row_start = i * row_bytes;
            let mut sum = 0.0f32;
            for bi in 0..blocks_per_row {
                let offset = row_start + bi * size_of::<BlockQ6K>();
                let block = unsafe { &*(a_quant.as_ptr().add(offset) as *const BlockQ6K) };
                sum += vec_dot_q6_k_f32(block, &x[bi * 256..(bi + 1) * 256]);
            }
            *yi = sum;
        };

        if m >= GEMV_PAR_THRESHOLD {
            par_rows(y, 512, compute_row);
        } else {
            y.iter_mut().enumerate().for_each(compute_row);
        }
    }
}

/// Q4_K_M GEMV: y[m] = A_q4km[m,k] @ x[k]. Parallelized across rows.
pub fn gemv_q4km_f32(a_quant: &[u8], x: &[f32], y: &mut [f32], m: usize, k: usize) {
    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(y.len(), m);
    debug_assert_eq!(k % 256, 0, "Q4_K_M GEMV: k must be divisible by 256");
    let blocks_per_row = k / 256;
    let row_bytes = blocks_per_row * size_of::<BlockQ4KM>();
    debug_assert_eq!(a_quant.len(), m * row_bytes);

    let compute_row = |(i, yi): (usize, &mut f32)| {
        let row_start = i * row_bytes;
        let mut sum = 0.0f32;
        for bi in 0..blocks_per_row {
            let offset = row_start + bi * size_of::<BlockQ4KM>();
            let block = unsafe { &*(a_quant.as_ptr().add(offset) as *const BlockQ4KM) };
            sum += vec_dot_q4_k_m_f32(block, &x[bi * 256..(bi + 1) * 256]);
        }
        *yi = sum;
    };

    if m >= GEMV_PAR_THRESHOLD {
        par_rows(y, 512, compute_row);
    } else {
        y.iter_mut().enumerate().for_each(compute_row);
    }
}

/// F32 GEMV: y[m] = A_f32[m,k] @ x[k].
pub fn gemv_f32(a: &[u8], x: &[f32], y: &mut [f32], m: usize, k: usize) {
    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(y.len(), m);
    let a_f32: &[f32] = bytemuck::cast_slice(a);
    debug_assert_eq!(a_f32.len(), m * k);

    for i in 0..m {
        let row = &a_f32[i * k..(i + 1) * k];
        let mut sum = 0.0f32;
        for j in 0..k {
            sum += row[j] * x[j];
        }
        y[i] = sum;
    }
}

/// Dispatch GEMV based on dtype: y[m] = W[m,k] @ x[k].
/// For Q4_0, pass scratch buffers to avoid per-call allocation.
pub fn gemv_dispatch(
    dtype: DType,
    data: &[u8],
    x: &[f32],
    y: &mut [f32],
    m: usize,
    k: usize,
    q8_scratch: Option<(&mut Vec<f32>, &mut Vec<i8>)>,
) {
    match dtype {
        DType::Q4_0 => {
            if let Some((scales, quants)) = q8_scratch {
                gemv_q4_0_f32(data, x, y, m, k, scales, quants);
            } else {
                let mut s = Vec::new();
                let mut q = Vec::new();
                gemv_q4_0_f32(data, x, y, m, k, &mut s, &mut q);
            }
        }
        DType::Q8_0 => {
            if let Some((scales, quants)) = q8_scratch {
                gemv_q8_0_f32(data, x, y, m, k, scales, quants);
            } else {
                let mut s = Vec::new();
                let mut q = Vec::new();
                gemv_q8_0_f32(data, x, y, m, k, &mut s, &mut q);
            }
        }
        DType::F32 => gemv_f32(data, x, y, m, k),
        DType::Q6K => {
            #[cfg(target_arch = "aarch64")]
            if let Some((scales, quants)) = q8_scratch {
                unsafe {
                    crate::backend::simd::neon::gemv_q6k_f32_neon(data, x, y, m, k, scales, quants);
                }
            } else {
                let mut s = Vec::new();
                let mut q = Vec::new();
                unsafe {
                    crate::backend::simd::neon::gemv_q6k_f32_neon(data, x, y, m, k, &mut s, &mut q);
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                let mut s = Vec::new();
                let mut q = Vec::new();
                gemv_q6k_f32(data, x, y, m, k, &mut s, &mut q);
            }
        }
        DType::Q4KM => gemv_q4km_f32(data, x, y, m, k),
        _ => panic!("gemv_dispatch: unsupported dtype {:?}", dtype),
    }
}

// ── Normalization ───────────────────────────────────────────────────────────

/// RMS normalization in-place: x = x / rms(x) * weight.
pub fn rmsnorm(x: &mut [f32], weight: &[f32], eps: f32) {
    debug_assert_eq!(x.len(), weight.len());
    let n = x.len();

    // Compute mean of squares
    let mut sum_sq = 0.0f32;
    for &v in x.iter() {
        sum_sq += v * v;
    }
    let rms = (sum_sq / n as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;

    for i in 0..n {
        x[i] = x[i] * inv_rms * weight[i];
    }
}

// ── Activation functions ────────────────────────────────────────────────────

/// SiLU (Swish) activation in-place: x = x * sigmoid(x).
pub fn silu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = *v / (1.0 + (-*v).exp());
    }
}

/// Fused SiLU activation + element-wise multiply: gate = silu(gate) * up.
/// Single pass instead of separate silu_inplace + mul_inplace.
pub fn silu_mul_inplace(gate: &mut [f32], up: &[f32]) {
    debug_assert_eq!(gate.len(), up.len());
    for (g, &u) in gate.iter_mut().zip(up.iter()) {
        *g = *g / (1.0 + (-*g).exp()) * u;
    }
}

// ── Softmax ─────────────────────────────────────────────────────────────────

/// Softmax in-place over a 1D slice.
pub fn softmax_inplace(x: &mut [f32]) {
    // Find max for numerical stability
    let max = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    // Exponentiate and sum
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }

    // Normalize
    let inv_sum = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv_sum;
    }
}

// ── Attention score/value computation ───────────────────────────────────────

/// Compute attention scores for one head: scores[t] = dot(q_head, k_cache_row_t) * scale.
/// `k_cache` has stride `kv_dim` between timesteps; each key starts at offset `kv_h_offset`.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn attn_scores(
    q_head: &[f32],
    k_cache: &[f32],
    scores: &mut [f32],
    kv_dim: usize,
    kv_h_offset: usize,
    head_dim: usize,
    scale: f32,
    seq_len: usize,
) {
    debug_assert!(q_head.len() >= head_dim);
    debug_assert!(scores.len() >= seq_len);
    if seq_len > 0 {
        debug_assert!(k_cache.len() >= (seq_len - 1) * kv_dim + kv_h_offset + head_dim);
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        attn_scores_neon(
            q_head,
            k_cache,
            scores,
            kv_dim,
            kv_h_offset,
            head_dim,
            scale,
            seq_len,
        );
    }
    #[cfg(not(target_arch = "aarch64"))]
    for t in 0..seq_len {
        let mut dot = 0.0f32;
        let k_off = t * kv_dim + kv_h_offset;
        for d in 0..head_dim {
            dot += q_head[d] * k_cache[k_off + d];
        }
        scores[t] = dot * scale;
    }
}

/// Compute weighted sum of V cache for one head: attn_out[d] = sum_t(scores[t] * v[t,d]).
/// `v_cache` has stride `kv_dim` between timesteps; each value starts at offset `kv_h_offset`.
#[allow(clippy::needless_range_loop)]
pub fn attn_values(
    scores: &[f32],
    v_cache: &[f32],
    attn_out: &mut [f32],
    kv_dim: usize,
    kv_h_offset: usize,
    head_dim: usize,
    seq_len: usize,
) {
    debug_assert!(scores.len() >= seq_len);
    debug_assert!(attn_out.len() >= head_dim);
    if seq_len > 0 {
        debug_assert!(v_cache.len() >= (seq_len - 1) * kv_dim + kv_h_offset + head_dim);
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        attn_values_neon(
            scores,
            v_cache,
            attn_out,
            kv_dim,
            kv_h_offset,
            head_dim,
            seq_len,
        );
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        attn_out[..head_dim].fill(0.0);
        for t in 0..seq_len {
            let s = scores[t];
            let v_base = t * kv_dim + kv_h_offset;
            for d in 0..head_dim {
                attn_out[d] += s * v_cache[v_base + d];
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
#[target_feature(enable = "neon")]
unsafe fn attn_scores_neon(
    q_head: &[f32],
    k_cache: &[f32],
    scores: &mut [f32],
    kv_dim: usize,
    kv_h_offset: usize,
    head_dim: usize,
    scale: f32,
    seq_len: usize,
) {
    use std::arch::aarch64::*;
    // Safety: caller ensures buffer bounds; intrinsics require unsafe in Edition 2024.
    unsafe {
        let q_ptr = q_head.as_ptr();
        let k_ptr = k_cache.as_ptr();

        for t in 0..seq_len {
            let k_off = t * kv_dim + kv_h_offset;
            let mut sum0 = vdupq_n_f32(0.0);
            let mut sum1 = vdupq_n_f32(0.0);

            let mut d = 0usize;
            while d + 8 <= head_dim {
                let q0 = vld1q_f32(q_ptr.add(d));
                let q1 = vld1q_f32(q_ptr.add(d + 4));
                let k0 = vld1q_f32(k_ptr.add(k_off + d));
                let k1 = vld1q_f32(k_ptr.add(k_off + d + 4));
                sum0 = vfmaq_f32(sum0, q0, k0);
                sum1 = vfmaq_f32(sum1, q1, k1);
                d += 8;
            }
            if d + 4 <= head_dim {
                let q0 = vld1q_f32(q_ptr.add(d));
                let k0 = vld1q_f32(k_ptr.add(k_off + d));
                sum0 = vfmaq_f32(sum0, q0, k0);
                d += 4;
            }
            let mut total = vaddvq_f32(vaddq_f32(sum0, sum1));
            while d < head_dim {
                total += *q_ptr.add(d) * *k_ptr.add(k_off + d);
                d += 1;
            }
            scores[t] = total * scale;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::needless_range_loop)]
#[target_feature(enable = "neon")]
unsafe fn attn_values_neon(
    scores: &[f32],
    v_cache: &[f32],
    attn_out: &mut [f32],
    kv_dim: usize,
    kv_h_offset: usize,
    head_dim: usize,
    seq_len: usize,
) {
    use std::arch::aarch64::*;
    // Safety: caller ensures buffer bounds; intrinsics require unsafe in Edition 2024.
    unsafe {
        let v_ptr = v_cache.as_ptr();
        let out_ptr = attn_out.as_mut_ptr();

        // T-outer loop: load scores[t] once, FMA across all d chunks.
        let n_vec = head_dim / 4;
        let n_tail = head_dim % 4;

        // Zero output first (we accumulate into it)
        let mut d = 0usize;
        while d + 4 <= head_dim {
            vst1q_f32(out_ptr.add(d), vdupq_n_f32(0.0));
            d += 4;
        }
        for dd in d..head_dim {
            *out_ptr.add(dd) = 0.0;
        }

        for t in 0..seq_len {
            let s = vdupq_n_f32(scores[t]);
            let v_base = t * kv_dim + kv_h_offset;

            // Process all head_dim in chunks of 4
            let mut d = 0usize;
            for _ in 0..n_vec {
                let acc = vld1q_f32(out_ptr.add(d));
                let v = vld1q_f32(v_ptr.add(v_base + d));
                vst1q_f32(out_ptr.add(d), vfmaq_f32(acc, s, v));
                d += 4;
            }
            // Scalar tail
            for dd in 0..n_tail {
                *out_ptr.add(d + dd) += scores[t] * *v_ptr.add(v_base + d + dd);
            }
        }
    }
}

// ── Positional encoding ─────────────────────────────────────────────────────

/// Apply Rotary Position Embedding (RoPE) to Q and K vectors.
///
/// `q` and `k` are [n_heads * head_dim] and [n_kv_heads * head_dim] respectively.
/// Applies rotation based on position `pos` and frequency base `freq_base`.
pub fn rope(
    q: &mut [f32],
    k: &mut [f32],
    pos: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    freq_base: f32,
) {
    debug_assert_eq!(q.len(), n_heads * head_dim);
    debug_assert_eq!(k.len(), n_kv_heads * head_dim);

    // Apply to Q heads
    for h in 0..n_heads {
        let offset = h * head_dim;
        apply_rope_to_head(&mut q[offset..offset + head_dim], pos, head_dim, freq_base);
    }

    // Apply to K heads
    for h in 0..n_kv_heads {
        let offset = h * head_dim;
        apply_rope_to_head(&mut k[offset..offset + head_dim], pos, head_dim, freq_base);
    }
}

/// Apply RoPE rotation to a single head vector.
fn apply_rope_to_head(head: &mut [f32], pos: usize, head_dim: usize, freq_base: f32) {
    let half_dim = head_dim / 2;
    for i in 0..half_dim {
        let freq = 1.0 / freq_base.powf(2.0 * i as f32 / head_dim as f32);
        let theta = pos as f32 * freq;
        let cos_t = theta.cos();
        let sin_t = theta.sin();

        let x0 = head[i];
        let x1 = head[i + half_dim];
        head[i] = x0 * cos_t - x1 * sin_t;
        head[i + half_dim] = x0 * sin_t + x1 * cos_t;
    }
}

// ── Convolution ─────────────────────────────────────────────────────────────

/// Depthwise 1D convolution.
///
/// `input`:  [seq_len, channels]
/// `weight`: [channels, kernel_size] (one kernel per channel)
/// `bias`:   optional [channels]
/// `output`: [seq_len, channels] (same padding via zero-pad)
pub fn conv1d_depthwise(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    channels: usize,
    kernel_size: usize,
    seq_len: usize,
) {
    debug_assert_eq!(input.len(), seq_len * channels);
    debug_assert_eq!(weight.len(), channels * kernel_size);
    debug_assert_eq!(output.len(), seq_len * channels);

    let pad = kernel_size / 2; // causal or symmetric padding

    for t in 0..seq_len {
        for c in 0..channels {
            let mut sum = if let Some(b) = bias { b[c] } else { 0.0 };

            for ki in 0..kernel_size {
                let input_t = t as isize + ki as isize - pad as isize;
                if input_t >= 0 && (input_t as usize) < seq_len {
                    sum += input[input_t as usize * channels + c] * weight[c * kernel_size + ki];
                }
            }
            output[t * channels + c] = sum;
        }
    }
}

// ── Element-wise operations ─────────────────────────────────────────────────

/// Element-wise addition: a += b.
pub fn add_inplace(a: &mut [f32], b: &[f32]) {
    debug_assert_eq!(a.len(), b.len());
    for (a, b) in a.iter_mut().zip(b.iter()) {
        *a += *b;
    }
}

/// Element-wise multiplication: a *= b.
pub fn mul_inplace(a: &mut [f32], b: &[f32]) {
    debug_assert_eq!(a.len(), b.len());
    for (a, b) in a.iter_mut().zip(b.iter()) {
        *a *= *b;
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_f32_identity() {
        // 2x2 identity matrix × [1,2; 3,4] = [1,2; 3,4]
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let mut c = vec![0.0; 4];
        matmul_f32(&a, &b, &mut c, 2, 2, 2);
        assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_matmul_f32_3x2_times_2x4() {
        // A = [[1,2],[3,4],[5,6]], B = [[1,2,3,4],[5,6,7,8]]
        // C = [[11,14,17,20],[23,30,37,44],[35,46,57,68]]
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0; 12];
        matmul_f32(&a, &b, &mut c, 3, 4, 2);
        assert_eq!(
            c,
            vec![
                11.0, 14.0, 17.0, 20.0, 23.0, 30.0, 37.0, 44.0, 35.0, 46.0, 57.0, 68.0
            ]
        );
    }

    #[test]
    fn test_rmsnorm() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let eps = 1e-5;

        // rms = sqrt((1+4+9+16)/4 + eps) = sqrt(7.5 + eps)
        let rms = (7.5f32 + eps).sqrt();
        let expected: Vec<f32> = vec![1.0 / rms, 2.0 / rms, 3.0 / rms, 4.0 / rms];

        rmsnorm(&mut x, &weight, eps);

        for (i, (&got, &exp)) in x.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "rmsnorm[{i}]: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn test_rmsnorm_with_weight() {
        let mut x = vec![2.0, 2.0];
        let weight = vec![3.0, 0.5];
        let eps = 1e-5;

        let rms = (4.0f32 + eps).sqrt(); // sqrt((4+4)/2 + eps)
        let inv_rms = 1.0 / rms;
        let expected = vec![2.0 * inv_rms * 3.0, 2.0 * inv_rms * 0.5];

        rmsnorm(&mut x, &weight, eps);

        for (i, (&got, &exp)) in x.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "rmsnorm[{i}]: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn test_silu() {
        let mut x = vec![0.0, 1.0, -1.0, 5.0];
        silu_inplace(&mut x);

        // silu(0) = 0, silu(1) = 1/(1+e^-1) ≈ 0.7311, silu(-1) ≈ -0.2689, silu(5) ≈ 4.9665
        assert!((x[0] - 0.0).abs() < 1e-5);
        assert!((x[1] - 0.7311).abs() < 1e-3);
        assert!((x[2] - (-0.2689)).abs() < 1e-3);
        assert!((x[3] - 4.9665).abs() < 1e-3);
    }

    #[test]
    fn test_silu_mul_inplace() {
        let mut gate = vec![0.0, 1.0, -1.0, 5.0];
        let up = vec![2.0, 3.0, 0.5, 1.0];

        // Reference: silu(gate) * up
        let mut gate_ref = gate.clone();
        silu_inplace(&mut gate_ref);
        mul_inplace(&mut gate_ref, &up);

        silu_mul_inplace(&mut gate, &up);

        for (i, (&got, &expected)) in gate.iter().zip(gate_ref.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-6,
                "silu_mul mismatch at {i}: got {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_softmax() {
        let mut x = vec![1.0, 2.0, 3.0];
        softmax_inplace(&mut x);

        // Sum should be 1.0
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // Values should be monotonically increasing
        assert!(x[0] < x[1]);
        assert!(x[1] < x[2]);

        // Check known values: softmax([1,2,3]) = [0.0900, 0.2447, 0.6652]
        assert!((x[0] - 0.0900).abs() < 1e-3);
        assert!((x[1] - 0.2447).abs() < 1e-3);
        assert!((x[2] - 0.6652).abs() < 1e-3);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Large values should not overflow
        let mut x = vec![1000.0, 1001.0, 1002.0];
        softmax_inplace(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(x.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_rope_basic() {
        // Basic test: pos=0 should not rotate (cos(0)=1, sin(0)=0)
        let mut q = vec![1.0, 2.0, 3.0, 4.0]; // 1 head, dim=4
        let mut k = vec![5.0, 6.0, 7.0, 8.0];
        let q_orig = q.clone();
        let k_orig = k.clone();

        rope(&mut q, &mut k, 0, 1, 1, 4, 10000.0);

        // At pos=0, theta=0 for all dims, so cos=1, sin=0 → no change
        for i in 0..4 {
            assert!((q[i] - q_orig[i]).abs() < 1e-5, "q[{i}] changed at pos=0");
            assert!((k[i] - k_orig[i]).abs() < 1e-5, "k[{i}] changed at pos=0");
        }
    }

    #[test]
    fn test_rope_rotates() {
        // At pos > 0, values should change
        let mut q = vec![1.0, 0.0, 0.0, 0.0]; // 1 head, dim=4
        let mut k = vec![1.0, 0.0, 0.0, 0.0];

        rope(&mut q, &mut k, 10, 1, 1, 4, 10000.0);

        // q should have been rotated — not identical anymore
        assert!((q[0] - 1.0).abs() > 1e-3 || (q[2]).abs() > 1e-3);
    }

    #[test]
    fn test_conv1d_depthwise_identity() {
        // Kernel [0, 1, 0] is identity
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // seq=3, channels=2
        let weight = vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0]; // 2 channels, kernel=3
        let mut output = vec![0.0; 6];

        conv1d_depthwise(&input, &weight, None, &mut output, 2, 3, 3);

        assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_conv1d_depthwise_with_bias() {
        let input = vec![1.0, 2.0]; // seq=1, channels=2
        let weight = vec![1.0, 1.0]; // 2 channels, kernel=1
        let bias = vec![10.0, 20.0];
        let mut output = vec![0.0; 2];

        conv1d_depthwise(&input, &weight, Some(&bias), &mut output, 2, 1, 1);

        assert_eq!(output, vec![11.0, 22.0]);
    }

    #[test]
    fn test_add_inplace() {
        let mut a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        add_inplace(&mut a, &b);
        assert_eq!(a, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_mul_inplace() {
        let mut a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        mul_inplace(&mut a, &b);
        assert_eq!(a, vec![4.0, 10.0, 18.0]);
    }

    #[test]
    fn test_par_rows_n_basic() {
        // 3 rows × 2 columns, each row doubles its index
        let mut out = vec![0.0f32; 6];
        par_rows_n(&mut out, 2, 1, |(i, row)| {
            row[0] = i as f32;
            row[1] = i as f32 * 2.0;
        });
        assert_eq!(out, vec![0.0, 0.0, 1.0, 2.0, 2.0, 4.0]);
    }

    #[test]
    fn test_par_rows_n_empty() {
        let mut out: Vec<f32> = vec![];
        par_rows_n(&mut out, 3, 1, |(_i, _row)| {
            panic!("should not be called");
        });
    }

    /// Reference scalar attention scores for testing.
    fn attn_scores_scalar(
        q: &[f32],
        k_cache: &[f32],
        scores: &mut [f32],
        kv_dim: usize,
        kv_h_off: usize,
        head_dim: usize,
        scale: f32,
        seq_len: usize,
    ) {
        for t in 0..seq_len {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[d] * k_cache[t * kv_dim + kv_h_off + d];
            }
            scores[t] = dot * scale;
        }
    }

    /// Reference scalar attention values for testing.
    fn attn_values_scalar(
        scores: &[f32],
        v_cache: &[f32],
        out: &mut [f32],
        kv_dim: usize,
        kv_h_off: usize,
        head_dim: usize,
        seq_len: usize,
    ) {
        for d in 0..head_dim {
            let mut val = 0.0f32;
            for t in 0..seq_len {
                val += scores[t] * v_cache[t * kv_dim + kv_h_off + d];
            }
            out[d] = val;
        }
    }

    #[test]
    fn test_attn_scores_matches_scalar() {
        let head_dim = 64;
        let kv_dim = 128; // 2 KV heads × 64
        let kv_h_off = 64; // second KV head
        let seq_len = 10;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q: Vec<f32> = (0..head_dim).map(|i| (i as f32 - 32.0) * 0.05).collect();
        let k_cache: Vec<f32> = (0..seq_len * kv_dim)
            .map(|i| ((i * 7 + 3) % 31) as f32 * 0.04 - 0.6)
            .collect();

        let mut expected = vec![0.0f32; seq_len];
        attn_scores_scalar(
            &q,
            &k_cache,
            &mut expected,
            kv_dim,
            kv_h_off,
            head_dim,
            scale,
            seq_len,
        );

        let mut actual = vec![0.0f32; seq_len];
        attn_scores(
            &q,
            &k_cache,
            &mut actual,
            kv_dim,
            kv_h_off,
            head_dim,
            scale,
            seq_len,
        );

        for t in 0..seq_len {
            let diff = (expected[t] - actual[t]).abs();
            assert!(
                diff < 1e-5,
                "attn_scores mismatch at t={t}: expected={}, actual={}, diff={diff}",
                expected[t],
                actual[t]
            );
        }
    }

    #[test]
    fn test_attn_values_matches_scalar() {
        let head_dim = 64;
        let kv_dim = 128;
        let kv_h_off = 0;
        let seq_len = 10;

        let scores: Vec<f32> = (0..seq_len)
            .map(|i| (i as f32 + 1.0) / seq_len as f32)
            .collect();
        let v_cache: Vec<f32> = (0..seq_len * kv_dim)
            .map(|i| ((i * 11 + 5) % 29) as f32 * 0.03 - 0.4)
            .collect();

        let mut expected = vec![0.0f32; head_dim];
        attn_values_scalar(
            &scores,
            &v_cache,
            &mut expected,
            kv_dim,
            kv_h_off,
            head_dim,
            seq_len,
        );

        let mut actual = vec![0.0f32; head_dim];
        attn_values(
            &scores,
            &v_cache,
            &mut actual,
            kv_dim,
            kv_h_off,
            head_dim,
            seq_len,
        );

        for d in 0..head_dim {
            let diff = (expected[d] - actual[d]).abs();
            assert!(
                diff < 1e-4,
                "attn_values mismatch at d={d}: expected={}, actual={}, diff={diff}",
                expected[d],
                actual[d]
            );
        }
    }

    #[test]
    fn test_attn_scores_seq_len_zero() {
        let mut scores = vec![];
        attn_scores(&[0.0; 64], &[], &mut scores, 64, 0, 64, 0.125, 0);
        assert!(scores.is_empty());
    }
}
