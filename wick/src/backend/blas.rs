//! Thin BLAS wrapper for prefill GEMM.
//!
//! On macOS this dispatches through Apple's Accelerate framework, which
//! routes SGEMM to the AMX (Apple Matrix eXtension) unit — delivering
//! ~1.5-2 TFLOPs f32 vs the ~500-600 GFLOPs/s we get from parallelized
//! NEON integer kernels. On Linux it goes through OpenBLAS.
//!
//! The provider crates (`accelerate-src` on macOS, `openblas-src` elsewhere)
//! provide the `#[link]` attributes — we just need to import them so the
//! linker includes them, and use the C ABI from `cblas-sys`.

// Pull in the provider so its #[link] attribute takes effect at link time.
// We never reference its symbols directly; the `use` alone is enough because
// its lib.rs has `#[link(name = "...", kind = "framework")]`.
#[cfg(target_os = "macos")]
#[allow(unused_imports)]
use accelerate_src as _;

#[cfg(not(target_os = "macos"))]
#[allow(unused_imports)]
use openblas_src as _;

use cblas_sys::{CBLAS_ORDER, CBLAS_TRANSPOSE, cblas_sgemm};

/// Compute `C[m, n] = A[m, k] * B[k, n]` in row-major layout, no transpose on either input.
///
/// `ld_a = k, ld_b = n, ld_c = n`. Alpha=1, beta=0 (output is overwritten, not accumulated).
///
/// # Panics
/// - if `a.len() < m * k`
/// - if `b.len() < k * n`
/// - if `c.len() < m * n`
///
/// # Aliasing
/// `a`, `b`, and `c` must reference non-overlapping memory regions. The CBLAS
/// contract requires distinct input and output buffers — passing the same
/// allocation for two slots is undefined behavior. Rust's `&mut` rules already
/// prevent `c` from aliasing `a` or `b` at the call site (you can't have a
/// shared borrow alive alongside an exclusive borrow), but `a` and `b` could
/// in principle be the same shared slice. We never do that and BLAS would
/// happily compute a meaningless result if we did.
pub fn sgemm_rowmajor_nn(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    assert!(
        a.len() >= m * k,
        "sgemm_rowmajor_nn: A buffer too small: {} < {} * {}",
        a.len(),
        m,
        k
    );
    assert!(
        b.len() >= k * n,
        "sgemm_rowmajor_nn: B buffer too small: {} < {} * {}",
        b.len(),
        k,
        n
    );
    assert!(
        c.len() >= m * n,
        "sgemm_rowmajor_nn: C buffer too small: {} < {} * {}",
        c.len(),
        m,
        n
    );

    // CBLAS integer widths are c_int on Linux, c_int on macOS too — just i32
    // on both supported hosts. cast_int is guarded because rustc complains
    // about potential truncation on 16-bit targets which we don't care about.
    let m_i = i32::try_from(m).expect("m overflow");
    let n_i = i32::try_from(n).expect("n overflow");
    let k_i = i32::try_from(k).expect("k overflow");

    // SAFETY:
    // - lengths verified above (a ≥ m*k, b ≥ k*n, c ≥ m*n).
    // - row-major leading dims match: lda=k, ldb=n, ldc=n with no transpose.
    // - non-aliasing: see the function-level Aliasing note. `&mut c` cannot
    //   alias `&a` or `&b` at the call site due to Rust borrow rules.
    // - cblas_sgemm reads a/b and writes c synchronously and does not retain
    //   the pointers after returning.
    unsafe {
        cblas_sgemm(
            CBLAS_ORDER::CblasRowMajor,
            CBLAS_TRANSPOSE::CblasNoTrans,
            CBLAS_TRANSPOSE::CblasNoTrans,
            m_i,
            n_i,
            k_i,
            1.0, // alpha
            a.as_ptr(),
            k_i, // lda
            b.as_ptr(),
            n_i, // ldb
            0.0, // beta
            c.as_mut_ptr(),
            n_i, // ldc
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgemm_identity() {
        // C = I * B should equal B
        let m = 4;
        let k = 4;
        let n = 3;
        let mut a = vec![0.0f32; m * k];
        for i in 0..m {
            a[i * k + i] = 1.0;
        }
        let b: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.5 + 1.0).collect();
        let mut c = vec![0.0f32; m * n];
        sgemm_rowmajor_nn(m, n, k, &a, &b, &mut c);
        for i in 0..m * n {
            assert_eq!(c[i], b[i], "identity GEMM failed at {i}");
        }
    }

    #[test]
    fn test_sgemm_simple_2x2() {
        // A = [[1,2],[3,4]], B = [[5,6],[7,8]]
        // C = A*B = [[19,22],[43,50]]
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![5.0f32, 6.0, 7.0, 8.0];
        let mut c = vec![0.0f32; 4];
        sgemm_rowmajor_nn(2, 2, 2, &a, &b, &mut c);
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    /// Microbenchmark: ffn_up shape (m=6912, n=2002, k=2048) — compare the
    /// NEON integer kernel (quantize input + q4_0×q8_0 GEMM) against the
    /// dequant + cblas_sgemm path the smoke test is wiring up. Ignored by
    /// default; run with:
    /// `cargo test -p wick --release --lib backend::blas::tests::microbench_ffn_up_gemm -- --ignored --nocapture`
    #[cfg(target_arch = "aarch64")]
    #[test]
    #[ignore]
    fn microbench_ffn_up_gemm() {
        use crate::backend::simd::neon;
        use crate::quant::{BlockQ4_0, dequantize_q4_0_matrix};
        use std::time::Instant;

        fn gflops(m: usize, n: usize, k: usize, seconds: f64) -> f64 {
            (2.0 * m as f64 * n as f64 * k as f64) / (seconds * 1e9)
        }

        let m = 6912; // is
        let k = 2048; // hs
        let n = 2002; // prompt length
        let iters = 4;

        // Random Q4_0 weight buffer — contents aren't statistically realistic
        // but the kernel work is identical regardless of byte content.
        let blocks_per_row = k / 32;
        let row_bytes = blocks_per_row * size_of::<BlockQ4_0>();
        let mut weight = vec![0u8; m * row_bytes];
        let mut s: u64 = 0xdead_beef;
        for byte in weight.iter_mut() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *byte = (s >> 33) as u8;
        }

        let input: Vec<f32> = (0..k * n)
            .map(|i| ((i * 31) % 127) as f32 * 0.01 - 0.5)
            .collect();

        // ── Path A: BLAS (dequant + cblas_sgemm) ──────────────────────
        let mut dequant = vec![0.0f32; m * k];
        let mut out_blas = vec![0.0f32; m * n];

        // Warmup
        dequantize_q4_0_matrix(&weight, m, k, &mut dequant);
        sgemm_rowmajor_nn(m, n, k, &dequant, &input, &mut out_blas);

        let t0 = Instant::now();
        for _ in 0..iters {
            dequantize_q4_0_matrix(&weight, m, k, &mut dequant);
        }
        let dequant_per = t0.elapsed().as_secs_f64() / iters as f64;

        let t0 = Instant::now();
        for _ in 0..iters {
            sgemm_rowmajor_nn(m, n, k, &dequant, &input, &mut out_blas);
        }
        let sgemm_per = t0.elapsed().as_secs_f64() / iters as f64;
        let blas_total_per = dequant_per + sgemm_per;

        // ── Path B: NEON integer GEMM ─────────────────────────────────
        let nb_k = k / 32;
        let mut b_scales = vec![0.0f32; n * nb_k];
        let mut b_quants = vec![0i8; n * k];
        let mut col = vec![0.0f32; k];

        let t0 = Instant::now();
        for _ in 0..iters {
            for j in 0..n {
                for i in 0..k {
                    col[i] = input[i * n + j];
                }
                unsafe {
                    neon::quantize_f32_to_q8_0_neon(
                        &col,
                        &mut b_scales[j * nb_k..(j + 1) * nb_k],
                        &mut b_quants[j * k..(j + 1) * k],
                    );
                }
            }
        }
        let quantize_per = t0.elapsed().as_secs_f64() / iters as f64;

        let mut out_neon = vec![0.0f32; m * n];
        // Warmup
        unsafe {
            neon::gemm_q4_0_q8_0_neon(&weight, &b_scales, &b_quants, &mut out_neon, m, n, k);
        }
        let t0 = Instant::now();
        for _ in 0..iters {
            unsafe {
                neon::gemm_q4_0_q8_0_neon(&weight, &b_scales, &b_quants, &mut out_neon, m, n, k);
            }
        }
        let neon_gemm_per = t0.elapsed().as_secs_f64() / iters as f64;
        let neon_total_per = quantize_per + neon_gemm_per;

        eprintln!("\n=== ffn_up GEMM microbench ({m} × {n} × {k}) ===");
        eprintln!("BLAS (dequant + sgemm):");
        eprintln!("  dequant:  {:>7.1} ms", dequant_per * 1000.0);
        eprintln!(
            "  sgemm:    {:>7.1} ms   ({:.1} GFLOPs/s)",
            sgemm_per * 1000.0,
            gflops(m, n, k, sgemm_per)
        );
        eprintln!(
            "  total:    {:>7.1} ms   ({:.1} GFLOPs/s effective)",
            blas_total_per * 1000.0,
            gflops(m, n, k, blas_total_per)
        );
        eprintln!("NEON (quantize + q4_0×q8_0 gemm):");
        eprintln!("  quantize: {:>7.1} ms", quantize_per * 1000.0);
        eprintln!(
            "  gemm:     {:>7.1} ms   ({:.1} GFLOPs/s)",
            neon_gemm_per * 1000.0,
            gflops(m, n, k, neon_gemm_per)
        );
        eprintln!(
            "  total:    {:>7.1} ms   ({:.1} GFLOPs/s effective)",
            neon_total_per * 1000.0,
            gflops(m, n, k, neon_total_per)
        );
        eprintln!(
            "\nNEON / BLAS total: {:.2}×   (>1 means NEON wins)",
            neon_total_per / blas_total_per
        );
        eprintln!(
            "NEON gemm / BLAS sgemm only: {:.2}×   (isolates kernel, excludes dequant/quantize)",
            neon_gemm_per / sgemm_per
        );
    }
}
