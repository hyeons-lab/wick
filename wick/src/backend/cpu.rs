// CPU compute backend — naive scalar implementations.
//
// All functions operate on raw f32 slices. No Tensor abstraction in the hot path.

// ── Thread pool configuration ──────────────────────────────────────────────

/// Configure rayon's global thread pool to use only performance cores.
///
/// On Apple Silicon, the efficiency cores (E-cores) have lower clock speed
/// and share memory bandwidth. Including them in rayon's pool creates
/// straggler threads that slow down synchronized `par_chunks_mut` dispatches
/// — measured as a 12% decode regression on M1 Max (58.6 vs 66.4 tok/s).
///
/// This function queries `hw.perflevel0.logicalcpu` (the P-core count) and
/// configures rayon to use at most that many threads. If the user has set
/// `RAYON_NUM_THREADS`, that takes precedence (rayon respects it before our
/// `build_global` call). On non-macOS or if the sysctl fails, rayon's
/// default (all logical cores) is used.
///
/// Must be called once before any rayon work (e.g., early in `main()`).
/// Returns the number of threads configured.
#[cfg(feature = "parallel")]
pub fn configure_thread_pool() -> usize {
    // If the user explicitly set RAYON_NUM_THREADS, respect it.
    if std::env::var("RAYON_NUM_THREADS").is_ok() {
        return rayon::current_num_threads();
    }

    let n = performance_core_count().unwrap_or(0);
    if n == 0 {
        return rayon::current_num_threads();
    }

    // build_global() can only succeed once per process. If rayon's global
    // pool was already initialized (e.g. by another caller, a test harness,
    // or a dependency), we can't apply the P-core limit here — surface that
    // by warning and returning the actual current thread count so callers
    // don't get a misleading value.
    match rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build_global()
    {
        Ok(()) => n,
        Err(err) => {
            tracing::warn!("failed to configure rayon global thread pool to {n} threads: {err}");
            rayon::current_num_threads()
        }
    }
}

/// No-op stub for builds without the `parallel` feature. Returns `1`
/// because single-threaded is the only choice.
#[cfg(not(feature = "parallel"))]
pub fn configure_thread_pool() -> usize {
    1
}

/// Returns the number of performance cores on macOS (Apple Silicon).
/// Uses `sysctlbyname("hw.perflevel0.logicalcpu")` directly — no subprocess.
#[cfg(target_os = "macos")]
fn performance_core_count() -> Option<usize> {
    unsafe extern "C" {
        fn sysctlbyname(
            name: *const std::ffi::c_char,
            oldp: *mut std::ffi::c_void,
            oldlenp: *mut usize,
            newp: *const std::ffi::c_void,
            newlen: usize,
        ) -> i32;
    }
    let name = c"hw.perflevel0.logicalcpu";
    let mut value: i32 = 0;
    let mut size = std::mem::size_of::<i32>();
    let ret = unsafe {
        sysctlbyname(
            name.as_ptr(),
            &mut value as *mut _ as *mut std::ffi::c_void,
            &mut size,
            std::ptr::null(),
            0,
        )
    };
    if ret == 0 && value > 0 {
        Some(value as usize)
    } else {
        None
    }
}

#[cfg(not(target_os = "macos"))]
fn performance_core_count() -> Option<usize> {
    None
}

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
    use crate::par::{IndexedParallelIterator, ParallelIterator, ParallelSliceMut};
    let chunk_size = (y.len() / crate::par::current_num_threads()).max(min_rows);
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
    use crate::par::{IndexedParallelIterator, ParallelIterator, ParallelSliceMut};
    let m = y.len() / n;
    let rows_per_chunk = (m / crate::par::current_num_threads()).max(min_rows);
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

    // Accumulate sum of squares in f64 to match ggml's ggml_float (double) precision.
    // This avoids f32 rounding that compounds across layers.
    let mut sum_sq = 0.0f64;
    for &v in x.iter() {
        sum_sq += (v as f64) * (v as f64);
    }
    let mean = sum_sq / n as f64;
    let rms = (mean + eps as f64).sqrt();
    let inv_rms = (1.0 / rms) as f32;

    for i in 0..n {
        x[i] = x[i] * inv_rms * weight[i];
    }
}

// ── Exp approximation ──────────────────────────────────────────────────────

/// Polynomial exp approximation matching ggml's `ggml_v_expf` (ARM optimized routine).
/// Maximum error: 1.45358 + 0.5 ULPs.
/// Inputs above 88.38 flush to infinity, below -103.97 flush to zero.
#[inline(always)]
fn ggml_expf(x: f32) -> f32 {
    // Bit-exact constants from ggml's hex float literals.
    const R: f32 = f32::from_bits(0x4B400000); // 0x1.8p23       = 12582912.0
    const LOG2E: f32 = f32::from_bits(0x3FB8AA3B); // 0x1.715476p+0  = log2(e)
    const LN2_HI: f32 = f32::from_bits(0x3F317200); // 0x1.62e4p-1    = ln(2) high
    const LN2_LO: f32 = f32::from_bits(0x35BFBE8E); // 0x1.7f7d1cp-20 = ln(2) low
    const C1: f32 = f32::from_bits(0x3F7FFFF6); // 0x1.ffffecp-1  ≈ 1/1!
    const C2: f32 = f32::from_bits(0x3EFFFEDB); // 0x1.fffdb6p-2  ≈ 1/2!
    const C3: f32 = f32::from_bits(0x3E2AAF33); // 0x1.555e66p-3  ≈ 1/3!
    const C4: f32 = f32::from_bits(0x3D2B9F17); // 0x1.573e2ep-5  ≈ 1/4!
    const C5: f32 = f32::from_bits(0x3C072010); // 0x1.0e4020p-7  ≈ 1/5!

    // n = round(x / ln2) via magic number trick
    let z = R + x * LOG2E;
    let n = z - R;

    // Cody-Waite range reduction: b = x - n*ln2
    let b = x - n * LN2_HI - n * LN2_LO;

    // 2^n via integer bit manipulation
    let e = z.to_bits().wrapping_shl(23);
    let k = f32::from_bits(e.wrapping_add(1.0f32.to_bits()));

    // Polynomial approximation of exp(b) - 1 (Estrin's scheme)
    let u = b * b;
    let j = C1 * b + (C2 + C3 * b + (C4 + C5 * b) * u) * u;

    // Combine: result = k * (1 + j) = 2^n * exp(b)
    let abs_n = f32::from_bits(n.to_bits() & 0x7FFF_FFFF);

    if abs_n <= 126.0 {
        k + j * k
    } else if abs_n > 192.0 {
        if n > 0.0 { f32::INFINITY } else { 0.0 }
    } else {
        let d = if n <= 0.0 { 0x82000000u32 } else { 0u32 };
        let s1 = f32::from_bits(d.wrapping_add(0x7f000000));
        let s2 = f32::from_bits(e.wrapping_sub(d));
        (s2 + s2 * j) * s1
    }
}

// ── Activation functions ────────────────────────────────────────────────────

/// SiLU (Swish) activation in-place: x = x * sigmoid(x).
/// Uses ggml's polynomial exp approximation to match ggml's NEON silu path.
pub fn silu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = *v / (1.0 + ggml_expf(-*v));
    }
}

/// Fused SiLU activation + element-wise multiply: gate = silu(gate) * up.
/// Single pass instead of separate silu_inplace + mul_inplace.
pub fn silu_mul_inplace(gate: &mut [f32], up: &[f32]) {
    debug_assert_eq!(gate.len(), up.len());
    for (g, &u) in gate.iter_mut().zip(up.iter()) {
        *g = *g / (1.0 + ggml_expf(-*g)) * u;
    }
}

// ── Softmax ─────────────────────────────────────────────────────────────────

/// Softmax in-place over a 1D slice.
/// Uses ggml's polynomial exp approximation and f64 accumulation to match ggml exactly.
pub fn softmax_inplace(x: &mut [f32]) {
    // Find max for numerical stability
    let max = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    // Exponentiate using ggml's polynomial exp and sum with f64 (matches ggml_float)
    let mut sum = 0.0f64;
    for v in x.iter_mut() {
        *v = ggml_expf(*v - max);
        sum += *v as f64;
    }

    // Normalize
    let inv_sum = (1.0 / sum) as f32;
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

        // Pre-load Q vectors once (constant across all timesteps).
        // Max 32 float32x4 = head_dim up to 128. Stack array avoids heap alloc.
        const MAX_Q_VECS: usize = 32;
        let n_q_vecs = head_dim / 4;
        debug_assert!(n_q_vecs <= MAX_Q_VECS, "head_dim > 128 not supported");
        let mut q_vecs = [vdupq_n_f32(0.0); MAX_Q_VECS];
        for i in 0..n_q_vecs {
            q_vecs[i] = vld1q_f32(q_ptr.add(i * 4));
        }

        for t in 0..seq_len {
            let k_off = t * kv_dim + kv_h_offset;
            let mut sum0 = vdupq_n_f32(0.0);
            let mut sum1 = vdupq_n_f32(0.0);

            let mut d = 0usize;
            let mut qi = 0usize;
            while d + 8 <= head_dim {
                let k0 = vld1q_f32(k_ptr.add(k_off + d));
                let k1 = vld1q_f32(k_ptr.add(k_off + d + 4));
                sum0 = vfmaq_f32(sum0, q_vecs[qi], k0);
                sum1 = vfmaq_f32(sum1, q_vecs[qi + 1], k1);
                d += 8;
                qi += 2;
            }
            if d + 4 <= head_dim {
                let k0 = vld1q_f32(k_ptr.add(k_off + d));
                sum0 = vfmaq_f32(sum0, q_vecs[qi], k0);
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

        // Accumulate in registers (not memory) across all timesteps, store once at end.
        // Max 32 float32x4 = head_dim up to 128.
        const MAX_ACC_VECS: usize = 32;
        let n_vec = head_dim / 4;
        let n_tail = head_dim % 4;
        debug_assert!(n_vec <= MAX_ACC_VECS, "head_dim > 128 not supported");
        let mut acc = [vdupq_n_f32(0.0); MAX_ACC_VECS];

        for t in 0..seq_len {
            let s = vdupq_n_f32(scores[t]);
            let v_base = t * kv_dim + kv_h_offset;
            for i in 0..n_vec {
                let v = vld1q_f32(v_ptr.add(v_base + i * 4));
                acc[i] = vfmaq_f32(acc[i], s, v);
            }
        }

        // Store accumulators to output
        for i in 0..n_vec {
            vst1q_f32(out_ptr.add(i * 4), acc[i]);
        }
        // Scalar tail
        let tail_start = n_vec * 4;
        for dd in 0..n_tail {
            let mut val = 0.0f32;
            for t in 0..seq_len {
                val += scores[t] * *v_ptr.add(t * kv_dim + kv_h_offset + tail_start + dd);
            }
            *out_ptr.add(tail_start + dd) = val;
        }
    }
}

// ── Flash attention (tiled, online softmax) ────────────────────────────────

const FLASH_TILE_KV: usize = 32;

/// Tiled flash attention for one KV head group (GQA).
///
/// Processes `group_size` query heads against a single KV head's cache. For
/// each query position, tiles over the KV cache with `FLASH_TILE_KV`-sized
/// chunks, using online softmax (running max + sum) so the full score vector
/// is never materialized.
///
/// **Layouts:**
/// - `q_mat`: `[hs, n]` stride-n (the batched projection output). Q for head
///   h, token j, dim d lives at `q_mat[(h * head_dim + d) * q_stride + j]`.
///   Gathered into a local contiguous array per query.
/// - `k_cache` / `v_cache`: `[total_seq, kv_dim]`, stride `kv_dim`. Position
///   t, dim d of KV head kv_h is at `cache[t * kv_dim + kv_h_offset + d]`.
/// - `out`: contiguous `[group_size, n_queries, head_dim]`. Element
///   `out[(g * n_queries + j) * head_dim + d]` is dim d of query j, group
///   member g. Caller is responsible for scatter-copying back to stride-n
///   layout if needed.
///
/// **Causal masking:** query at position `start_pos + j` attends only to KV
/// positions `0 .. start_pos + j` (inclusive). Tiles beyond the causal limit
/// are skipped entirely; individual positions within a boundary tile are
/// masked to `-INF` before the softmax update.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn flash_attention_gqa_cpu(
    q_mat: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    out: &mut [f32],
    n_heads_start: usize,
    group_size: usize,
    n_queries: usize,
    q_stride: usize,
    kv_dim: usize,
    kv_h_offset: usize,
    head_dim: usize,
    scale: f32,
    start_pos: usize,
) {
    // NEON kernel requires head_dim to be a multiple of 4 and <= 128.
    // Fall back to scalar for unsupported dimensions.
    #[cfg(target_arch = "aarch64")]
    {
        if head_dim % 4 == 0 && head_dim <= 128 {
            unsafe {
                flash_attention_gqa_neon(
                    q_mat,
                    k_cache,
                    v_cache,
                    out,
                    n_heads_start,
                    group_size,
                    n_queries,
                    q_stride,
                    kv_dim,
                    kv_h_offset,
                    head_dim,
                    scale,
                    start_pos,
                );
            }
            return;
        }
    }
    flash_attention_gqa_scalar(
        q_mat,
        k_cache,
        v_cache,
        out,
        n_heads_start,
        group_size,
        n_queries,
        q_stride,
        kv_dim,
        kv_h_offset,
        head_dim,
        scale,
        start_pos,
    );
}

#[allow(dead_code, clippy::too_many_arguments, clippy::needless_range_loop)]
fn flash_attention_gqa_scalar(
    q_mat: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    out: &mut [f32],
    n_heads_start: usize,
    group_size: usize,
    n_queries: usize,
    q_stride: usize,
    kv_dim: usize,
    kv_h_offset: usize,
    head_dim: usize,
    scale: f32,
    start_pos: usize,
) {
    // Stack-allocated scratch to avoid heap alloc contention in parallel
    // dispatch. 256 covers all known model head_dims (64, 128, 160, 256).
    // The NEON kernel falls back to this scalar path for head_dim > 128.
    assert!(
        head_dim <= 256,
        "flash_attention_gqa_scalar: head_dim {head_dim} > 256"
    );
    let mut q_buf = [0.0f32; 256];
    let mut acc_buf = [0.0f32; 256];
    let q_local = &mut q_buf[..head_dim];
    let acc = &mut acc_buf[..head_dim];
    let mut tile_scores = [0.0f32; FLASH_TILE_KV];

    for g in 0..group_size {
        let h = n_heads_start + g;
        let h_off = h * head_dim;

        for j in 0..n_queries {
            let max_kv = start_pos + j + 1;

            for d in 0..head_dim {
                q_local[d] = q_mat[(h_off + d) * q_stride + j];
            }

            let mut running_max = f32::NEG_INFINITY;
            let mut running_sum = 0.0f64;
            acc.fill(0.0);

            for kv_start in (0..max_kv).step_by(FLASH_TILE_KV) {
                let kv_end = (kv_start + FLASH_TILE_KV).min(max_kv);
                let tile_len = kv_end - kv_start;

                for ti in 0..tile_len {
                    let k_off = (kv_start + ti) * kv_dim + kv_h_offset;
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q_local[d] * k_cache[k_off + d];
                    }
                    tile_scores[ti] = dot * scale;
                }

                let tile_max = tile_scores[..tile_len]
                    .iter()
                    .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let new_max = running_max.max(tile_max);

                let rescale = if running_max > f32::NEG_INFINITY {
                    ggml_expf(running_max - new_max)
                } else {
                    0.0
                };

                let mut tile_sum = 0.0f64;
                for ti in 0..tile_len {
                    tile_scores[ti] = ggml_expf(tile_scores[ti] - new_max);
                    tile_sum += tile_scores[ti] as f64;
                }

                for d in 0..head_dim {
                    acc[d] *= rescale;
                }
                for ti in 0..tile_len {
                    let s = tile_scores[ti];
                    let v_off = (kv_start + ti) * kv_dim + kv_h_offset;
                    for d in 0..head_dim {
                        acc[d] += s * v_cache[v_off + d];
                    }
                }

                running_sum = running_sum * rescale as f64 + tile_sum;
                running_max = new_max;
            }

            let inv_sum = (1.0 / running_sum) as f32;
            let out_off = (g * n_queries + j) * head_dim;
            for d in 0..head_dim {
                out[out_off + d] = acc[d] * inv_sum;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
unsafe fn flash_attention_gqa_neon(
    q_mat: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    out: &mut [f32],
    n_heads_start: usize,
    group_size: usize,
    n_queries: usize,
    q_stride: usize,
    kv_dim: usize,
    kv_h_offset: usize,
    head_dim: usize,
    scale: f32,
    start_pos: usize,
) {
    use std::arch::aarch64::*;
    unsafe {
        debug_assert!(
            q_mat.len() >= ((n_heads_start + group_size) * head_dim - 1) * q_stride + n_queries,
            "q_mat too small for the given head range and q_stride"
        );
        debug_assert!(
            (start_pos + n_queries == 0)
                || k_cache.len() >= (start_pos + n_queries - 1) * kv_dim + kv_h_offset + head_dim,
            "k_cache too small"
        );
        debug_assert!(
            (start_pos + n_queries == 0)
                || v_cache.len() >= (start_pos + n_queries - 1) * kv_dim + kv_h_offset + head_dim,
            "v_cache too small"
        );
        debug_assert!(
            out.len() >= group_size * n_queries * head_dim,
            "out buffer too small for contiguous [group_size, n_queries, head_dim] output"
        );

        let q_ptr = q_mat.as_ptr();
        let k_ptr = k_cache.as_ptr();
        let v_ptr = v_cache.as_ptr();
        let out_ptr = out.as_mut_ptr();

        let n_vecs = head_dim / 4;
        debug_assert!(
            head_dim % 4 == 0 && n_vecs <= 32,
            "head_dim must be a multiple of 4 and <= 128"
        );

        const MAX_VECS: usize = 32;
        let mut q_vecs = [vdupq_n_f32(0.0); MAX_VECS];
        let mut acc_vecs = [vdupq_n_f32(0.0); MAX_VECS];
        let mut tile_scores = [0.0f32; FLASH_TILE_KV];

        for g in 0..group_size {
            let h = n_heads_start + g;
            let h_off = h * head_dim;

            for j in 0..n_queries {
                let max_kv = start_pos + j + 1;

                // Gather Q[h, j] from stride-n layout into NEON registers
                for i in 0..n_vecs {
                    let d = i * 4;
                    let q = [
                        *q_ptr.add((h_off + d) * q_stride + j),
                        *q_ptr.add((h_off + d + 1) * q_stride + j),
                        *q_ptr.add((h_off + d + 2) * q_stride + j),
                        *q_ptr.add((h_off + d + 3) * q_stride + j),
                    ];
                    q_vecs[i] = vld1q_f32(q.as_ptr());
                }

                let mut running_max = f32::NEG_INFINITY;
                let mut running_sum = 0.0f64;
                for i in 0..n_vecs {
                    acc_vecs[i] = vdupq_n_f32(0.0);
                }

                for kv_start in (0..max_kv).step_by(FLASH_TILE_KV) {
                    let kv_end = (kv_start + FLASH_TILE_KV).min(max_kv);
                    let tile_len = kv_end - kv_start;

                    // QK dot products for the tile
                    for ti in 0..tile_len {
                        let k_off = (kv_start + ti) * kv_dim + kv_h_offset;
                        let mut sum0 = vdupq_n_f32(0.0);
                        let mut sum1 = vdupq_n_f32(0.0);
                        let mut i = 0;
                        while i + 2 <= n_vecs {
                            let k0 = vld1q_f32(k_ptr.add(k_off + i * 4));
                            let k1 = vld1q_f32(k_ptr.add(k_off + i * 4 + 4));
                            sum0 = vfmaq_f32(sum0, q_vecs[i], k0);
                            sum1 = vfmaq_f32(sum1, q_vecs[i + 1], k1);
                            i += 2;
                        }
                        if i < n_vecs {
                            let k0 = vld1q_f32(k_ptr.add(k_off + i * 4));
                            sum0 = vfmaq_f32(sum0, q_vecs[i], k0);
                        }
                        tile_scores[ti] = vaddvq_f32(vaddq_f32(sum0, sum1)) * scale;
                    }

                    // Online softmax: tile max
                    let mut tile_max = f32::NEG_INFINITY;
                    for ti in 0..tile_len {
                        if tile_scores[ti] > tile_max {
                            tile_max = tile_scores[ti];
                        }
                    }
                    let new_max = running_max.max(tile_max);

                    let rescale = if running_max > f32::NEG_INFINITY {
                        ggml_expf(running_max - new_max)
                    } else {
                        0.0
                    };

                    // Exp scores and sum
                    let mut tile_sum = 0.0f64;
                    for ti in 0..tile_len {
                        tile_scores[ti] = ggml_expf(tile_scores[ti] - new_max);
                        tile_sum += tile_scores[ti] as f64;
                    }

                    // Rescale accumulator
                    let rescale_v = vdupq_n_f32(rescale);
                    for i in 0..n_vecs {
                        acc_vecs[i] = vmulq_f32(acc_vecs[i], rescale_v);
                    }

                    // Accumulate weighted V: acc += score * V
                    for ti in 0..tile_len {
                        let s = vdupq_n_f32(tile_scores[ti]);
                        let v_base = (kv_start + ti) * kv_dim + kv_h_offset;
                        for i in 0..n_vecs {
                            let v = vld1q_f32(v_ptr.add(v_base + i * 4));
                            acc_vecs[i] = vfmaq_f32(acc_vecs[i], s, v);
                        }
                    }

                    running_sum = running_sum * rescale as f64 + tile_sum;
                    running_max = new_max;
                }

                // Normalize and write contiguous output
                let inv_sum = (1.0 / running_sum) as f32;
                let inv_sum_v = vdupq_n_f32(inv_sum);
                let out_off = (g * n_queries + j) * head_dim;
                for i in 0..n_vecs {
                    let result = vmulq_f32(acc_vecs[i], inv_sum_v);
                    vst1q_f32(out_ptr.add(out_off + i * 4), result);
                }
            }
        }
    }
}

// ── TurboQuant NEON attention ───────────────────────────────────────────────

/// NEON-optimized TurboQuant attention scores for one KV head, multiple query heads.
///
/// Replaces the scalar bucket-sum + QJL loops with NEON intrinsics.
/// For head_dim=128: processes 32 polar bytes and 16 JL bytes per timestep.
///
/// # Safety
/// Caller must ensure all buffer lengths match head_dim, seq_len, and group_size.
/// Requires aarch64 NEON.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn attn_scores_turboquant_neon(
    q_rot_all: &[f32], // [n_heads * head_dim] pre-rotated queries
    q_jl_all: &[f32],  // [n_heads * head_dim] pre-JL-projected queries
    polar_data: &[u8], // packed 2-bit data for this KV head
    jl_data: &[u8],    // packed 1-bit data for this KV head
    norms_f32: &[f32], // pre-converted f32 norms
    residual_norms_f32: &[f32],
    q_jl_total_sums: &[f32], // pre-computed sum(q_jl) per head
    group_start: usize,      // first query head index in the group
    group_size: usize,       // number of query heads in the group
    scores_flat: &mut [f32], // [group_size * seq_len] output, row-major by head
    head_dim: usize,
    centroids: &[f32; 4],
    scale: f32,
    qjl_scale: f32,
    seq_len: usize,
) {
    use std::arch::aarch64::*;
    unsafe {
        let polar_bytes = head_dim / 4;
        let jl_bytes = head_dim / 8;
        let c_arr = *centroids;

        // Comment #13: Pre-unpack centroid f32x4 vectors per timestep,
        // shared across all query heads in the GQA group.
        // Max head_dim=128 → 32 polar bytes → 32 float32x4 centroids.
        // Comment #18: Same for QJL masks — 16 jl bytes × 2 halves = 32 float32x4.
        const MAX_VECS: usize = 32;
        let n_cent_vecs = polar_bytes; // one float32x4 per packed byte
        let n_mask_vecs = jl_bytes * 2; // two float32x4 per jl byte (lo/hi)
        debug_assert!(n_cent_vecs <= MAX_VECS);
        debug_assert!(n_mask_vecs <= MAX_VECS);
        let mut cent_vecs = [vdupq_n_f32(0.0); MAX_VECS];
        let mut mask_vecs = [vdupq_n_f32(0.0); MAX_VECS];

        for t in 0..seq_len {
            let p_base = t * polar_bytes;
            let j_base = t * jl_bytes;
            let norm = norms_f32[t];
            let residual_norm = residual_norms_f32[t];

            // Unpack centroids once per timestep (hoisted from head loop)
            for (i, cv) in cent_vecs.iter_mut().enumerate().take(n_cent_vecs) {
                let b = *polar_data.get_unchecked(p_base + i);
                *cv = select_centroids_4(b, &c_arr);
            }

            // Unpack QJL masks once per timestep (hoisted from head loop, Comment #18)
            for i in 0..jl_bytes {
                let b = *jl_data.get_unchecked(j_base + i) as u32;
                mask_vecs[i * 2] = bits_to_f32_mask_lo(b);
                mask_vecs[i * 2 + 1] = bits_to_f32_mask_hi(b);
            }

            // Process each query head in the GQA group
            for g in 0..group_size {
                let h = group_start + g;
                let q_rot = &q_rot_all[h * head_dim..];
                let q_jl = &q_jl_all[h * head_dim..];

                // PolarQuant dot: FMA pre-unpacked centroids with query
                let mut dot_acc0 = vdupq_n_f32(0.0);
                let mut dot_acc1 = vdupq_n_f32(0.0);
                let mut ci = 0usize;
                let mut q_off = 0usize;
                while ci + 4 <= n_cent_vecs {
                    let qv0 = vld1q_f32(q_rot.as_ptr().add(q_off));
                    let qv1 = vld1q_f32(q_rot.as_ptr().add(q_off + 4));
                    let qv2 = vld1q_f32(q_rot.as_ptr().add(q_off + 8));
                    let qv3 = vld1q_f32(q_rot.as_ptr().add(q_off + 12));
                    dot_acc0 = vfmaq_f32(dot_acc0, qv0, cent_vecs[ci]);
                    dot_acc1 = vfmaq_f32(dot_acc1, qv1, cent_vecs[ci + 1]);
                    dot_acc0 = vfmaq_f32(dot_acc0, qv2, cent_vecs[ci + 2]);
                    dot_acc1 = vfmaq_f32(dot_acc1, qv3, cent_vecs[ci + 3]);
                    ci += 4;
                    q_off += 16;
                }
                while ci < n_cent_vecs {
                    let qv = vld1q_f32(q_rot.as_ptr().add(q_off));
                    dot_acc0 = vfmaq_f32(dot_acc0, qv, cent_vecs[ci]);
                    ci += 1;
                    q_off += 4;
                }
                let polar_dot = vaddvq_f32(vaddq_f32(dot_acc0, dot_acc1)) * norm;

                // QJL: pos_sum only, total_sum pre-computed, masks pre-unpacked
                let total_sum = *q_jl_total_sums.get_unchecked(h);
                let mut pos_acc0 = vdupq_n_f32(0.0);
                let mut pos_acc1 = vdupq_n_f32(0.0);
                let mut mi = 0usize;
                let mut jl_q_off = 0usize;
                while mi + 4 <= n_mask_vecs {
                    let q0 = vld1q_f32(q_jl.as_ptr().add(jl_q_off));
                    let q1 = vld1q_f32(q_jl.as_ptr().add(jl_q_off + 4));
                    let q2 = vld1q_f32(q_jl.as_ptr().add(jl_q_off + 8));
                    let q3 = vld1q_f32(q_jl.as_ptr().add(jl_q_off + 12));
                    pos_acc0 = vfmaq_f32(pos_acc0, q0, mask_vecs[mi]);
                    pos_acc1 = vfmaq_f32(pos_acc1, q1, mask_vecs[mi + 1]);
                    pos_acc0 = vfmaq_f32(pos_acc0, q2, mask_vecs[mi + 2]);
                    pos_acc1 = vfmaq_f32(pos_acc1, q3, mask_vecs[mi + 3]);
                    mi += 4;
                    jl_q_off += 16;
                }
                while mi < n_mask_vecs {
                    let q = vld1q_f32(q_jl.as_ptr().add(jl_q_off));
                    pos_acc0 = vfmaq_f32(pos_acc0, q, mask_vecs[mi]);
                    mi += 1;
                    jl_q_off += 4;
                }
                let pos_sum = vaddvq_f32(vaddq_f32(pos_acc0, pos_acc1));
                let signed_sum = 2.0 * pos_sum - total_sum;
                // residual_norm is stored in unit-normalized key space, so
                // the correction must be rescaled by the original key norm
                // to match polar_dot (which was multiplied by norm above).
                let correction = norm * residual_norm * qjl_scale * signed_sum;

                scores_flat[g * seq_len + t] = (polar_dot + correction) * scale;
            }
        }
    }
}

/// NEON weighted sum of compressed values for a GQA group.
///
/// For each query head in `[group_start, group_start + group_size)`:
///   `out[h*head_dim + d] = Σ_t (scores[g*seq_len + t] * norms_f32[t]) * centroid[indices[t, d]]`
///
/// Writes the **rotated-space** accumulator to `attn_out`; the caller is
/// responsible for applying `rht_inverse` to each head after this function
/// returns. Caller must also ensure `head_dim <= 128` (stack accumulator
/// limit) and that all buffer lengths are consistent.
///
/// # Safety
/// All slices must be large enough: `polar_data.len() >= seq_len * head_dim/4`,
/// `norms_f32.len() >= seq_len`, `scores.len() >= group_size * seq_len`,
/// `attn_out.len() >= (group_start + group_size) * head_dim`.
/// Requires aarch64 NEON.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub unsafe fn attn_values_turboquant_neon(
    polar_data: &[u8],    // packed 2-bit data for this KV head
    norms_f32: &[f32],    // pre-converted f32 norms for this KV head
    scores: &[f32],       // [group_size * seq_len], row-major by head
    attn_out: &mut [f32], // [n_heads * head_dim] — writes group_size heads
    group_start: usize,
    group_size: usize,
    head_dim: usize,
    seq_len: usize,
    centroids: &[f32; 4],
) {
    use std::arch::aarch64::*;
    unsafe {
        let polar_bytes = head_dim / 4;
        debug_assert!(
            polar_bytes <= 32,
            "head_dim > 128 not supported by NEON path"
        );

        // Stack-allocated accumulator: up to 32 float32x4 = 128 floats.
        // One set per group member; reused across the group loop.
        const MAX_VECS: usize = 32;

        for g in 0..group_size {
            let h = group_start + g;
            let head_scores = scores.as_ptr().add(g * seq_len);

            // Initialize accumulator to zero.
            let mut acc = [vdupq_n_f32(0.0); MAX_VECS];

            // Accumulate weighted centroid vectors across all timesteps.
            for t in 0..seq_len {
                let w = *head_scores.add(t) * *norms_f32.get_unchecked(t);
                let w_vec = vdupq_n_f32(w);
                let base = t * polar_bytes;
                for i in 0..polar_bytes {
                    let b = *polar_data.get_unchecked(base + i);
                    let c_vec = select_centroids_4(b, centroids);
                    acc[i] = vfmaq_f32(acc[i], w_vec, c_vec);
                }
            }

            // Store the per-head accumulator to attn_out. The caller applies
            // rht_inverse afterwards — it's cheap (O(head_dim log head_dim))
            // and doesn't benefit from being inline here.
            let out_ptr = attn_out.as_mut_ptr().add(h * head_dim);
            for i in 0..polar_bytes {
                vst1q_f32(out_ptr.add(i * 4), acc[i]);
            }
        }
    }
}

/// Select 4 centroid values from a packed 2-bit byte.
/// Returns float32x4 with centroids[idx0], centroids[idx1], centroids[idx2], centroids[idx3].
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn select_centroids_4(byte: u8, c: &[f32; 4]) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    unsafe {
        let vals: [f32; 4] = [
            *c.get_unchecked((byte & 0x03) as usize),
            *c.get_unchecked(((byte >> 2) & 0x03) as usize),
            *c.get_unchecked(((byte >> 4) & 0x03) as usize),
            *c.get_unchecked(((byte >> 6) & 0x03) as usize),
        ];
        vld1q_f32(vals.as_ptr())
    }
}

/// Expand lower 4 bits of a byte to f32 mask: bit i → 0.0 or 1.0.
/// Returns float32x4 for bits 0,1,2,3.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn bits_to_f32_mask_lo(byte: u32) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    unsafe {
        let vals: [f32; 4] = [
            (byte & 1) as f32,
            ((byte >> 1) & 1) as f32,
            ((byte >> 2) & 1) as f32,
            ((byte >> 3) & 1) as f32,
        ];
        vld1q_f32(vals.as_ptr())
    }
}

/// Expand upper 4 bits of a byte to f32 mask: bit i → 0.0 or 1.0.
/// Returns float32x4 for bits 4,5,6,7.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn bits_to_f32_mask_hi(byte: u32) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    unsafe {
        let vals: [f32; 4] = [
            ((byte >> 4) & 1) as f32,
            ((byte >> 5) & 1) as f32,
            ((byte >> 6) & 1) as f32,
            ((byte >> 7) & 1) as f32,
        ];
        vld1q_f32(vals.as_ptr())
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
/// Uses iterative theta multiplication to match ggml's `ggml_rope_cache_init`.
fn apply_rope_to_head(head: &mut [f32], pos: usize, head_dim: usize, freq_base: f32) {
    let half_dim = head_dim / 2;
    let theta_scale = freq_base.powf(-2.0 / head_dim as f32);
    let mut theta = pos as f32;
    for i in 0..half_dim {
        let (sin_t, cos_t) = theta.sin_cos();

        let x0 = head[i];
        let x1 = head[i + half_dim];
        head[i] = x0 * cos_t - x1 * sin_t;
        head[i + half_dim] = x0 * sin_t + x1 * cos_t;
        theta *= theta_scale;
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
    fn test_ggml_expf() {
        // ggml_expf should approximate exp() within ~1.5 ULPs
        let test_vals = [0.0f32, 1.0, -1.0, 2.0, -5.0, -10.0, -50.0, 80.0];
        for &x in &test_vals {
            let got = ggml_expf(x);
            let expected = x.exp();
            let rel_err = if expected.abs() > 1e-10 {
                ((got - expected) / expected).abs()
            } else {
                (got - expected).abs()
            };
            assert!(
                rel_err < 1e-5,
                "ggml_expf({x}) = {got}, expected {expected}, rel_err = {rel_err}"
            );
        }
        // Edge cases
        assert!(ggml_expf(100.0).is_infinite() || ggml_expf(100.0) > 1e30);
        assert!(ggml_expf(-200.0) < 1e-30);
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

    #[test]
    fn test_flash_attention_matches_naive() {
        // Compare flash attention output against the naive
        // attn_scores + softmax_inplace + attn_values pipeline.
        //
        // Setup: 4 query heads, 2 KV heads (group_size=2), head_dim=64,
        // 8 query tokens, start_pos=4 (so total seq_len up to 12).
        let n_heads = 4;
        let n_kv_heads = 2;
        let group_size = n_heads / n_kv_heads;
        let head_dim = 64;
        let hs = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let n = 8;
        let start_pos = 4;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let total_seq = start_pos + n; // 12

        // Random Q in [hs, n] stride-n layout
        let mut q_mat = vec![0.0f32; hs * n];
        let mut seed: u64 = 0xCAFE_BABE;
        for v in q_mat.iter_mut() {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            *v = ((seed >> 33) as i32 as f32) * 1e-9;
        }

        // Random K/V cache in [total_seq, kv_dim] layout
        let mut k_cache = vec![0.0f32; total_seq * kv_dim];
        for v in k_cache.iter_mut() {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            *v = ((seed >> 33) as i32 as f32) * 1e-9;
        }
        let mut v_cache = vec![0.0f32; total_seq * kv_dim];
        for v in v_cache.iter_mut() {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            *v = ((seed >> 33) as i32 as f32) * 1e-9;
        }

        // ── Flash attention ────────────────────────────────────────
        // Kernel writes contiguous [group_size, n, head_dim] per KV head.
        // Scatter-copy back to [hs, n] stride-n for comparison with naive.
        let chunk_size = group_size * n * head_dim;
        let mut flash_raw = vec![0.0f32; n_kv_heads * chunk_size];
        for kv_h in 0..n_kv_heads {
            let chunk = &mut flash_raw[kv_h * chunk_size..(kv_h + 1) * chunk_size];
            flash_attention_gqa_cpu(
                &q_mat,
                &k_cache,
                &v_cache,
                chunk,
                kv_h * group_size,
                group_size,
                n,
                n, // q_stride
                kv_dim,
                kv_h * head_dim,
                head_dim,
                scale,
                start_pos,
            );
        }
        let mut flash_out = vec![0.0f32; hs * n];
        for kv_h in 0..n_kv_heads {
            for g in 0..group_size {
                let h = kv_h * group_size + g;
                let src_base = kv_h * chunk_size + g * n * head_dim;
                for j in 0..n {
                    for d in 0..head_dim {
                        flash_out[(h * head_dim + d) * n + j] =
                            flash_raw[src_base + j * head_dim + d];
                    }
                }
            }
        }

        // ── Naive reference ────────────────────────────────────────
        let mut naive_out = vec![0.0f32; hs * n];
        for j in 0..n {
            let seq_len = start_pos + j + 1; // causal: attend to 0..seq_len
            for h in 0..n_heads {
                let kv_h = h / group_size;
                let kv_h_offset = kv_h * head_dim;

                // Gather Q[h, j] from stride-n layout
                let mut q_head = vec![0.0f32; head_dim];
                for d in 0..head_dim {
                    q_head[d] = q_mat[(h * head_dim + d) * n + j];
                }

                // Scores
                let mut scores = vec![0.0f32; seq_len];
                attn_scores(
                    &q_head,
                    &k_cache,
                    &mut scores,
                    kv_dim,
                    kv_h_offset,
                    head_dim,
                    scale,
                    seq_len,
                );

                // Softmax
                softmax_inplace(&mut scores);

                // Weighted values
                let mut attn_out = vec![0.0f32; head_dim];
                attn_values(
                    &scores,
                    &v_cache,
                    &mut attn_out,
                    kv_dim,
                    kv_h_offset,
                    head_dim,
                    seq_len,
                );

                // Scatter-write to stride-n output
                for d in 0..head_dim {
                    naive_out[(h * head_dim + d) * n + j] = attn_out[d];
                }
            }
        }

        // ── Compare ────────────────────────────────────────────────
        let mut max_diff = 0.0f32;
        for i in 0..hs * n {
            max_diff = max_diff.max((flash_out[i] - naive_out[i]).abs());
        }
        assert!(
            max_diff < 1e-4,
            "flash vs naive max_diff = {max_diff} (expected < 1e-4)"
        );
    }

    /// Microbenchmark: measure GEMV throughput and effective memory bandwidth
    /// for the Q4_0 × Q8_0 pre-quantized kernel at FFN gate shape.
    ///
    /// Run with:
    /// `cargo test -p wick --release --lib backend::cpu::tests::microbench_gemv_q4_0 -- --ignored --nocapture`
    #[cfg(all(target_arch = "aarch64", feature = "parallel"))]
    #[test]
    #[ignore]
    fn microbench_gemv_q4_0() {
        use std::time::Instant;

        // Build a *local* rayon pool sized to P-cores. Using `build_global`
        // here would silently fail because cargo's test harness initializes
        // rayon early — `pool.install` instead applies the P-core limit only
        // for the closure body via rayon's thread-local current-pool.
        let n_threads = performance_core_count().unwrap_or(8);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build()
            .expect("local pool build");

        pool.install(|| {
            let m = 6912; // FFN gate rows
            let k = 2048; // hidden_size
            let iters = 200;

            // Random Q4_0 weight
            let blocks_per_row = k / 32;
            let row_bytes = blocks_per_row * size_of::<crate::quant::BlockQ4_0>();
            let mut weight = vec![0u8; m * row_bytes];
            let mut s: u64 = 0xdead_beef;
            for b in weight.iter_mut() {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                *b = (s >> 33) as u8;
            }

            // Random input, pre-quantized to Q8_0
            let x: Vec<f32> = (0..k)
                .map(|i| ((i * 31) % 127) as f32 * 0.01 - 0.5)
                .collect();
            let (x_scales, x_quants) = quantize_f32_to_q8_0(&x);
            let mut y = vec![0.0f32; m];

            // Warmup
            gemv_q4_0_with_q8(&weight, &x_scales, &x_quants, &mut y, m, k);

            let t0 = Instant::now();
            for _ in 0..iters {
                gemv_q4_0_with_q8(&weight, &x_scales, &x_quants, &mut y, m, k);
            }
            let elapsed = t0.elapsed().as_secs_f64();
            let per_call = elapsed / iters as f64;

            let weight_bytes = m * row_bytes;
            let input_bytes = x_scales.len() * 4 + x_quants.len();
            let total_bytes = weight_bytes + input_bytes;
            let bw_gbps = (total_bytes as f64 / per_call) / 1e9;

            eprintln!("\n=== GEMV Q4_0×Q8_0 microbench (m={m}, k={k}) ===");
            eprintln!("  per-call: {:.1} µs", per_call * 1e6);
            eprintln!("  weight:   {:.2} MB", weight_bytes as f64 / 1e6);
            eprintln!("  bandwidth: {:.1} GB/s", bw_gbps);
            eprintln!("  rayon threads: {n_threads} (local pool, P-cores)");

            // Also measure a large GEMV (output projection shape)
            let m_large = 65536;
            let mut weight_large = vec![0u8; m_large * row_bytes];
            for b in weight_large.iter_mut() {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                *b = (s >> 33) as u8;
            }
            let mut y_large = vec![0.0f32; m_large];
            gemv_q4_0_with_q8(
                &weight_large,
                &x_scales,
                &x_quants,
                &mut y_large,
                m_large,
                k,
            );

            let t0 = Instant::now();
            for _ in 0..20 {
                gemv_q4_0_with_q8(
                    &weight_large,
                    &x_scales,
                    &x_quants,
                    &mut y_large,
                    m_large,
                    k,
                );
            }
            let elapsed = t0.elapsed().as_secs_f64();
            let per_call = elapsed / 20.0;
            let weight_bytes_large = m_large * row_bytes;
            let bw_large = ((weight_bytes_large + input_bytes) as f64 / per_call) / 1e9;

            eprintln!("\n=== GEMV Q4_0×Q8_0 large (m={m_large}, k={k}) ===");
            eprintln!("  per-call: {:.1} µs", per_call * 1e6);
            eprintln!("  weight:   {:.2} MB", weight_bytes_large as f64 / 1e6);
            eprintln!("  bandwidth: {:.1} GB/s", bw_large);
        });
    }
}
