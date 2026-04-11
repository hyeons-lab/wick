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

    // SAFETY: we verified lengths above. The pointers are valid for the sizes
    // passed to cblas_sgemm, and cblas_sgemm reads a/b and writes c without
    // retaining them.
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
}
