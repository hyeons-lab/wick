// SIMD-optimized kernels for quantized operations.
//
// Platform-specific implementations behind cfg gates.
// The dispatch functions select the best available implementation at compile time.

#[cfg(target_arch = "aarch64")]
use crate::quant::BlockQ6K;
use crate::quant::{BlockQ4_0, BlockQ4KM, BlockQ8_0};
use half::f16;

// ── aarch64 NEON ────────────────────────────────────────────────────────────

/// Send+Sync pointer wrapper for parallel GEMV closures.
/// Stores pointers as usize to satisfy Send+Sync (raw pointers don't implement them).
/// Safety: callers ensure non-overlapping row access and immutable source data.
#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy)]
struct GemvPtrs {
    a: usize,
    xq: usize,
    xs: usize,
}
#[cfg(target_arch = "aarch64")]
impl GemvPtrs {
    fn a(&self) -> *const u8 {
        self.a as *const u8
    }
    fn xq(&self) -> *const i8 {
        self.xq as *const i8
    }
    fn xs(&self) -> *const f32 {
        self.xs as *const f32
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::needless_range_loop, unused_unsafe)]
pub(crate) mod neon {
    use super::*;
    use std::arch::aarch64::*;
    use std::mem::size_of;

    // ── Shared GEMM dot-product macros ─────────────────────────────────────
    // Used by both Q4_0 and Q8_0 GEMM kernels to avoid duplicating the
    // Q8_0 input loading + vdotq_s32 + scale accumulation pattern.

    /// Accumulate dot products for a pair of decoded weight blocks against one Q8_0 column.
    /// `$w0_lo/$w0_hi`: decoded weight int8x16 for block bi (lo/hi halves)
    /// `$w1_lo/$w1_hi`: decoded weight int8x16 for block bi+1
    /// `$d0_w/$d1_w`: weight-side f32 scales for blocks bi and bi+1
    macro_rules! gemm_dot_pair {
        ($w0_lo:expr, $w0_hi:expr, $w1_lo:expr, $w1_hi:expr,
         $d0_w:expr, $d1_w:expr,
         $xq:expr, $xs:expr, $bi:expr,
         $sumv0:expr, $sumv1:expr) => {{
            let y0_lo = vld1q_s8($xq.add($bi * 32));
            let y0_hi = vld1q_s8($xq.add($bi * 32 + 16));
            let y1_lo = vld1q_s8($xq.add(($bi + 1) * 32));
            let y1_hi = vld1q_s8($xq.add(($bi + 1) * 32 + 16));
            let z = vdupq_n_s32(0);
            let p_0 = vdotq_s32(vdotq_s32(z, $w0_lo, y0_lo), $w0_hi, y0_hi);
            let p_1 = vdotq_s32(vdotq_s32(z, $w1_lo, y1_lo), $w1_hi, y1_hi);
            $sumv0 = vmlaq_n_f32($sumv0, vcvtq_f32_s32(p_0), $d0_w * *$xs.add($bi));
            $sumv1 = vmlaq_n_f32($sumv1, vcvtq_f32_s32(p_1), $d1_w * *$xs.add($bi + 1));
        }};
    }

    /// Accumulate dot product for a single decoded weight block against one Q8_0 column.
    macro_rules! gemm_dot_single {
        ($w_lo:expr, $w_hi:expr, $d_w:expr,
         $xq:expr, $xs:expr, $bi:expr, $sumv:expr) => {{
            let y_lo = vld1q_s8($xq.add($bi * 32));
            let y_hi = vld1q_s8($xq.add($bi * 32 + 16));
            let z = vdupq_n_s32(0);
            let p = vdotq_s32(vdotq_s32(z, $w_lo, y_lo), $w_hi, y_hi);
            $sumv = vmlaq_n_f32($sumv, vcvtq_f32_s32(p), $d_w * *$xs.add($bi));
        }};
    }

    /// NEON-optimized Q8_0 dot product with f32 vector.
    #[target_feature(enable = "neon")]
    pub unsafe fn vec_dot_q8_0_f32_neon(block: &BlockQ8_0, y: &[f32]) -> f32 {
        unsafe {
            debug_assert_eq!(y.len(), 32);
            let d = f16::from_bits(block.delta).to_f32();

            let mut sumv = vdupq_n_f32(0.0);
            let quants_ptr = block.quants.as_ptr();
            let y_ptr = y.as_ptr();

            for i in (0..32).step_by(8) {
                // Load 8 i8 values, sign-extend to i16, then split to i32, convert to f32
                let q_bytes = vld1_s8(quants_ptr.add(i));
                let q_i16 = vmovl_s8(q_bytes);

                let q_lo_f32 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q_i16)));
                let q_hi_f32 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q_i16)));

                let y_lo = vld1q_f32(y_ptr.add(i));
                let y_hi = vld1q_f32(y_ptr.add(i + 4));

                sumv = vfmaq_f32(sumv, q_lo_f32, y_lo);
                sumv = vfmaq_f32(sumv, q_hi_f32, y_hi);
            }

            d * vaddvq_f32(sumv)
        }
    }

    /// NEON-optimized Q4_0 dot product with f32 vector.
    ///
    /// Q4_0 block: 16 bytes `qs` holding 32 4-bit unsigned values (low nibble first,
    /// then high nibble). Values are offset by -8: value = (nibble - 8) * d.
    ///
    /// Uses vector nibble extraction (vand/vshr on uint8x8) then widens to f32
    /// without scalar code in the inner loop.
    #[target_feature(enable = "neon")]
    pub unsafe fn vec_dot_q4_0_f32_neon(block: &BlockQ4_0, y: &[f32]) -> f32 {
        unsafe {
            debug_assert_eq!(y.len(), 32);
            let d = f16::from_bits(block.d).to_f32();
            let offset = vdupq_n_f32(8.0);
            let mask_lo = vdup_n_u8(0x0F);

            let mut sumv = vdupq_n_f32(0.0);
            let qs_ptr = block.qs.as_ptr();
            let y_ptr = y.as_ptr();

            // Process 8 bytes at a time → 8 low nibbles + 8 high nibbles = 16 values.
            // Two iterations cover all 16 bytes (32 values).
            for i in (0..16).step_by(8) {
                // Load 8 bytes of quantized data
                let qbytes = vld1_u8(qs_ptr.add(i));

                // Extract low and high nibbles as u8 vectors
                let lo_u8 = vand_u8(qbytes, mask_lo);
                let hi_u8 = vshr_n_u8::<4>(qbytes);

                // Widen low nibbles: u8x8 → u16x8 → split → u32x4 → f32x4
                let lo_u16 = vmovl_u8(lo_u8);
                let lo_f32_0 = vsubq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo_u16))), offset);
                let lo_f32_1 = vsubq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo_u16))), offset);

                // Widen high nibbles similarly
                let hi_u16 = vmovl_u8(hi_u8);
                let hi_f32_0 = vsubq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi_u16))), offset);
                let hi_f32_1 = vsubq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi_u16))), offset);

                // FMA with corresponding y values
                // Low nibbles: y[i..i+4], y[i+4..i+8]
                sumv = vfmaq_f32(sumv, lo_f32_0, vld1q_f32(y_ptr.add(i)));
                sumv = vfmaq_f32(sumv, lo_f32_1, vld1q_f32(y_ptr.add(i + 4)));
                // High nibbles: y[i+16..i+20], y[i+20..i+24]
                sumv = vfmaq_f32(sumv, hi_f32_0, vld1q_f32(y_ptr.add(i + 16)));
                sumv = vfmaq_f32(sumv, hi_f32_1, vld1q_f32(y_ptr.add(i + 16 + 4)));
            }

            d * vaddvq_f32(sumv)
        }
    }

    /// NEON-optimized Q4_K_M dot product with f32 vector.
    #[target_feature(enable = "neon")]
    pub unsafe fn vec_dot_q4_k_m_f32_neon(block: &BlockQ4KM, y: &[f32]) -> f32 {
        unsafe {
            let d = f16::from_bits(block.d).to_f32();
            let dmin = f16::from_bits(block.dmin).to_f32();

            let scales = &block.scales;
            let mut sc = [0u8; 8];
            let mut mn = [0u8; 8];
            for j in 0..4 {
                sc[j] = scales[j] & 63;
                mn[j] = scales[j + 4] & 63;
            }
            for j in 4..8 {
                sc[j] = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
                mn[j] = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
            }

            let qs = &block.qs;
            let y_ptr = y.as_ptr();
            let mut sumf = 0.0f32;
            let mut qi = 0usize;
            let mut yi = 0usize;

            for j in 0..4 {
                let sc1 = d * sc[j * 2] as f32;
                let mn1 = dmin * mn[j * 2] as f32;
                let sc2 = d * sc[j * 2 + 1] as f32;
                let mn2 = dmin * mn[j * 2 + 1] as f32;

                let mut sum1v = vdupq_n_f32(0.0);
                let mut sum2v = vdupq_n_f32(0.0);
                let mut sum_mn1v = vdupq_n_f32(0.0);
                let mut sum_mn2v = vdupq_n_f32(0.0);

                for l in (0..32).step_by(4) {
                    let q0 = qs[qi + l] as u32;
                    let q1 = qs[qi + l + 1] as u32;
                    let q2 = qs[qi + l + 2] as u32;
                    let q3 = qs[qi + l + 3] as u32;

                    let lo = [
                        (q0 & 0xF) as f32,
                        (q1 & 0xF) as f32,
                        (q2 & 0xF) as f32,
                        (q3 & 0xF) as f32,
                    ];
                    let lo_v = vld1q_f32(lo.as_ptr());

                    let hi = [
                        (q0 >> 4) as f32,
                        (q1 >> 4) as f32,
                        (q2 >> 4) as f32,
                        (q3 >> 4) as f32,
                    ];
                    let hi_v = vld1q_f32(hi.as_ptr());

                    let y1 = vld1q_f32(y_ptr.add(yi + l));
                    let y2 = vld1q_f32(y_ptr.add(yi + l + 32));

                    sum1v = vfmaq_f32(sum1v, lo_v, y1);
                    sum2v = vfmaq_f32(sum2v, hi_v, y2);
                    sum_mn1v = vaddq_f32(sum_mn1v, y1);
                    sum_mn2v = vaddq_f32(sum_mn2v, y2);
                }

                let sum1 = vaddvq_f32(sum1v);
                let sum2 = vaddvq_f32(sum2v);
                let sum_mn1 = vaddvq_f32(sum_mn1v);
                let sum_mn2 = vaddvq_f32(sum_mn2v);

                sumf += sc1 * sum1 + sc2 * sum2 - mn1 * sum_mn1 - mn2 * sum_mn2;
                qi += 32;
                yi += 64;
            }

            sumf
        }
    }

    /// Quantize f32 vector to Q8_0 format (NEON-vectorized, f16 scale roundtrip).
    /// Stores scales and quants into caller-provided buffers.
    /// Returns the number of blocks written.
    #[target_feature(enable = "neon")]
    pub unsafe fn quantize_f32_to_q8_0_neon(
        x: &[f32],
        scales: &mut [f32],
        quants: &mut [i8],
    ) -> usize {
        unsafe {
            let k = x.len();
            debug_assert_eq!(
                k % 32,
                0,
                "quantize_f32_to_q8_0: x.len() must be divisible by 32"
            );
            debug_assert!(scales.len() >= k / 32);
            debug_assert!(quants.len() >= k);
            let n_blocks = k / 32;

            for bi in 0..n_blocks {
                let base = bi * 32;
                let x_ptr = x.as_ptr().add(base);

                let s0 = vld1q_f32(x_ptr);
                let s1 = vld1q_f32(x_ptr.add(4));
                let s2 = vld1q_f32(x_ptr.add(8));
                let s3 = vld1q_f32(x_ptr.add(12));
                let s4 = vld1q_f32(x_ptr.add(16));
                let s5 = vld1q_f32(x_ptr.add(20));
                let s6 = vld1q_f32(x_ptr.add(24));
                let s7 = vld1q_f32(x_ptr.add(28));

                let a0 = vmaxq_f32(vabsq_f32(s0), vabsq_f32(s1));
                let a1 = vmaxq_f32(vabsq_f32(s2), vabsq_f32(s3));
                let a2 = vmaxq_f32(vabsq_f32(s4), vabsq_f32(s5));
                let a3 = vmaxq_f32(vabsq_f32(s6), vabsq_f32(s7));
                let a4 = vmaxq_f32(a0, a1);
                let a5 = vmaxq_f32(a2, a3);
                let a6 = vmaxq_f32(a4, a5);
                let amax = vmaxvq_f32(a6);

                let d = amax / 127.0;
                let d = f16::from_f32(d).to_f32(); // f16 roundtrip
                let id = if d != 0.0 { 1.0 / d } else { 0.0 };
                scales[bi] = d;

                // Quantize 32 f32 → 32 i8 using NEON vector narrowing.
                // f32→i32 (vcvtnq), then i32→i16→i8 via vqmovn (saturating narrow).
                // Process 8 values at a time → 4 iterations for 32 values.
                let qp = quants.as_mut_ptr().add(base);
                let vi0 = vcvtnq_s32_f32(vmulq_n_f32(s0, id));
                let vi1 = vcvtnq_s32_f32(vmulq_n_f32(s1, id));
                let vi2 = vcvtnq_s32_f32(vmulq_n_f32(s2, id));
                let vi3 = vcvtnq_s32_f32(vmulq_n_f32(s3, id));
                let vi4 = vcvtnq_s32_f32(vmulq_n_f32(s4, id));
                let vi5 = vcvtnq_s32_f32(vmulq_n_f32(s5, id));
                let vi6 = vcvtnq_s32_f32(vmulq_n_f32(s6, id));
                let vi7 = vcvtnq_s32_f32(vmulq_n_f32(s7, id));

                // i32x4 pairs → i16x8 → i8x8, then store 8 bytes at a time
                let n16_01 = vcombine_s16(vqmovn_s32(vi0), vqmovn_s32(vi1));
                let n16_23 = vcombine_s16(vqmovn_s32(vi2), vqmovn_s32(vi3));
                let n16_45 = vcombine_s16(vqmovn_s32(vi4), vqmovn_s32(vi5));
                let n16_67 = vcombine_s16(vqmovn_s32(vi6), vqmovn_s32(vi7));

                vst1_s8(qp, vqmovn_s16(n16_01));
                vst1_s8(qp.add(8), vqmovn_s16(n16_23));
                vst1_s8(qp.add(16), vqmovn_s16(n16_45));
                vst1_s8(qp.add(24), vqmovn_s16(n16_67));
            }
            n_blocks
        }
    }

    /// NEON integer GEMV using pre-quantized Q8_0 input.
    /// Call `quantize_f32_to_q8_0_neon` first, then call this for each weight matrix.
    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn gemv_q4_0_q8_0_neon(
        a_quant: &[u8],
        x_scales: &[f32],
        x_quants: &[i8],
        y: &mut [f32],
        _m: usize,
        k: usize,
    ) {
        unsafe {
            let blocks_per_row = k / 32;
            let row_bytes = blocks_per_row * size_of::<BlockQ4_0>();

            let ptrs = GemvPtrs {
                a: a_quant.as_ptr() as usize,
                xq: x_quants.as_ptr() as usize,
                xs: x_scales.as_ptr() as usize,
            };

            let compute_row = move |(i, yi): (usize, &mut f32)| unsafe {
                let mask_lo = vdupq_n_u8(0x0F);
                let offset_8 = vdupq_n_s8(0x8);
                let row_start = i * row_bytes;
                let mut sumv0 = vdupq_n_f32(0.0);
                let mut sumv1 = vdupq_n_f32(0.0);

                let mut bi = 0usize;
                while bi + 1 < blocks_per_row {
                    // Prefetch next weight block pair
                    if bi + 3 < blocks_per_row {
                        _prefetch(
                            ptrs.a().add(row_start + (bi + 2) * size_of::<BlockQ4_0>())
                                as *const i8,
                            _PREFETCH_READ,
                            _PREFETCH_LOCALITY2,
                        );
                    }
                    let b0 = &*(ptrs.a().add(row_start + bi * size_of::<BlockQ4_0>())
                        as *const BlockQ4_0);
                    let b1 = &*(ptrs.a().add(row_start + (bi + 1) * size_of::<BlockQ4_0>())
                        as *const BlockQ4_0);

                    let v0 = vld1q_u8(b0.qs.as_ptr());
                    let v1 = vld1q_u8(b1.qs.as_ptr());
                    let v0_lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v0, mask_lo)), offset_8);
                    let v0_hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(v0)), offset_8);
                    let v1_lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v1, mask_lo)), offset_8);
                    let v1_hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(v1)), offset_8);

                    let y0_lo = vld1q_s8(ptrs.xq().add(bi * 32));
                    let y0_hi = vld1q_s8(ptrs.xq().add(bi * 32 + 16));
                    let y1_lo = vld1q_s8(ptrs.xq().add((bi + 1) * 32));
                    let y1_hi = vld1q_s8(ptrs.xq().add((bi + 1) * 32 + 16));

                    let z = vdupq_n_s32(0);
                    let p_0 = vdotq_s32(vdotq_s32(z, v0_lo, y0_lo), v0_hi, y0_hi);
                    let p_1 = vdotq_s32(vdotq_s32(z, v1_lo, y1_lo), v1_hi, y1_hi);

                    let d0 = f16::from_bits(b0.d).to_f32() * *ptrs.xs().add(bi);
                    let d1 = f16::from_bits(b1.d).to_f32() * *ptrs.xs().add(bi + 1);
                    sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p_0), d0);
                    sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p_1), d1);
                    bi += 2;
                }

                if bi < blocks_per_row {
                    let b = &*(ptrs.a().add(row_start + bi * size_of::<BlockQ4_0>())
                        as *const BlockQ4_0);
                    let v = vld1q_u8(b.qs.as_ptr());
                    let v_lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v, mask_lo)), offset_8);
                    let v_hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(v)), offset_8);
                    let y_lo = vld1q_s8(ptrs.xq().add(bi * 32));
                    let y_hi = vld1q_s8(ptrs.xq().add(bi * 32 + 16));
                    let z = vdupq_n_s32(0);
                    let p = vdotq_s32(vdotq_s32(z, v_lo, y_lo), v_hi, y_hi);
                    let d = f16::from_bits(b.d).to_f32() * *ptrs.xs().add(bi);
                    sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p), d);
                }

                *yi = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
            };

            if y.len() >= super::super::cpu::GEMV_PAR_THRESHOLD {
                crate::backend::cpu::par_rows(y, 512, compute_row);
            } else {
                y.iter_mut().enumerate().for_each(compute_row);
            }
        }
    }

    /// NEON integer GEMV: y[m] = A_q4_0[m,k] @ x_f32[k].
    /// Uses caller-provided scratch buffers to avoid per-call heap allocation.
    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn gemv_q4_0_f32_neon(
        a_quant: &[u8],
        x: &[f32],
        y: &mut [f32],
        _m: usize,
        k: usize,
        q8_scales: &mut Vec<f32>,
        q8_quants: &mut Vec<i8>,
    ) {
        unsafe {
            let n_blocks = k / 32;
            q8_scales.resize(n_blocks, 0.0);
            q8_quants.resize(k, 0);
            quantize_f32_to_q8_0_neon(x, q8_scales, q8_quants);
            gemv_q4_0_q8_0_neon(a_quant, q8_scales, q8_quants, y, _m, k);
        }
    }

    /// NEON Q8_0 × Q8_0 GEMV with pre-quantized input (no quantization step).
    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn gemv_q8_0_q8_0_neon(
        a_quant: &[u8],
        x_scales: &[f32],
        x_quants: &[i8],
        y: &mut [f32],
        _m: usize,
        k: usize,
    ) {
        unsafe {
            let n_blocks = k / 32;
            let row_bytes = n_blocks * size_of::<BlockQ8_0>();

            let ptrs = GemvPtrs {
                a: a_quant.as_ptr() as usize,
                xq: x_quants.as_ptr() as usize,
                xs: x_scales.as_ptr() as usize,
            };

            let compute_row = move |(i, yi): (usize, &mut f32)| unsafe {
                let row_start = i * row_bytes;
                let mut sumv0 = vdupq_n_f32(0.0);
                let mut sumv1 = vdupq_n_f32(0.0);

                let mut bi = 0usize;
                while bi + 1 < n_blocks {
                    if bi + 3 < n_blocks {
                        _prefetch(
                            ptrs.a().add(row_start + (bi + 2) * size_of::<BlockQ8_0>())
                                as *const i8,
                            _PREFETCH_READ,
                            _PREFETCH_LOCALITY2,
                        );
                    }
                    let wb0 = &*(ptrs.a().add(row_start + bi * size_of::<BlockQ8_0>())
                        as *const BlockQ8_0);
                    let wb1 = &*(ptrs.a().add(row_start + (bi + 1) * size_of::<BlockQ8_0>())
                        as *const BlockQ8_0);

                    let w0_lo = vld1q_s8(wb0.quants.as_ptr());
                    let w0_hi = vld1q_s8(wb0.quants.as_ptr().add(16));
                    let w1_lo = vld1q_s8(wb1.quants.as_ptr());
                    let w1_hi = vld1q_s8(wb1.quants.as_ptr().add(16));

                    // Load input quants
                    let x0_lo = vld1q_s8(ptrs.xq().add(bi * 32));
                    let x0_hi = vld1q_s8(ptrs.xq().add(bi * 32 + 16));
                    let x1_lo = vld1q_s8(ptrs.xq().add((bi + 1) * 32));
                    let x1_hi = vld1q_s8(ptrs.xq().add((bi + 1) * 32 + 16));

                    // Integer dot product: 2 × vdotq_s32 per block
                    let z = vdupq_n_s32(0);
                    let p_0 = vdotq_s32(vdotq_s32(z, w0_lo, x0_lo), w0_hi, x0_hi);
                    let p_1 = vdotq_s32(vdotq_s32(z, w1_lo, x1_lo), w1_hi, x1_hi);

                    // Scale: d_weight × d_input
                    let d0 = f16::from_bits(wb0.delta).to_f32() * *ptrs.xs().add(bi);
                    let d1 = f16::from_bits(wb1.delta).to_f32() * *ptrs.xs().add(bi + 1);
                    sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p_0), d0);
                    sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p_1), d1);
                    bi += 2;
                }

                if bi < n_blocks {
                    let wb = &*(ptrs.a().add(row_start + bi * size_of::<BlockQ8_0>())
                        as *const BlockQ8_0);
                    let w_lo = vld1q_s8(wb.quants.as_ptr());
                    let w_hi = vld1q_s8(wb.quants.as_ptr().add(16));
                    let x_lo = vld1q_s8(ptrs.xq().add(bi * 32));
                    let x_hi = vld1q_s8(ptrs.xq().add(bi * 32 + 16));
                    let z = vdupq_n_s32(0);
                    let p = vdotq_s32(vdotq_s32(z, w_lo, x_lo), w_hi, x_hi);
                    let d = f16::from_bits(wb.delta).to_f32() * *ptrs.xs().add(bi);
                    sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p), d);
                }

                *yi = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
            };

            if y.len() >= super::super::cpu::GEMV_PAR_THRESHOLD {
                crate::backend::cpu::par_rows(y, 512, compute_row);
            } else {
                y.iter_mut().enumerate().for_each(compute_row);
            }
        }
    }

    /// NEON integer GEMV: y[m] = A_q8_0[m,k] @ x_f32[k].
    /// Convenience wrapper: quantizes x then calls gemv_q8_0_q8_0_neon.
    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn gemv_q8_0_f32_neon(
        a_quant: &[u8],
        x: &[f32],
        y: &mut [f32],
        _m: usize,
        k: usize,
        q8_scales: &mut Vec<f32>,
        q8_quants: &mut Vec<i8>,
    ) {
        unsafe {
            let n_blocks = k / 32;
            q8_scales.resize(n_blocks, 0.0);
            q8_quants.resize(k, 0);
            quantize_f32_to_q8_0_neon(x, q8_scales, q8_quants);
            gemv_q8_0_q8_0_neon(a_quant, q8_scales, q8_quants, y, _m, k);
        }
    }

    /// NEON Q6_K × Q8_0 integer GEMV with pre-quantized input.
    ///
    /// Extracts 6-bit quants as i8, dots with Q8_0 input using vdotq_s32.
    /// 16 sub-blocks of 16 values per Q6_K block, each with its own scale.
    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn gemv_q6k_q8_0_neon(
        a_quant: &[u8],
        x_scales: &[f32],
        x_quants: &[i8],
        y: &mut [f32],
        _m: usize,
        k: usize,
    ) {
        unsafe {
            let blocks_per_row = k / 256;
            let row_bytes = blocks_per_row * size_of::<BlockQ6K>();
            let a_base = a_quant.as_ptr() as usize;
            let xq_base = x_quants.as_ptr() as usize;
            let xs_base = x_scales.as_ptr() as usize;

            let compute_row = move |(i, yi): (usize, &mut f32)| unsafe {
                let row_start = i * row_bytes;
                let mut sumf = 0.0f32;
                let mask_0f = vdupq_n_u8(0x0F);
                let mask_03 = vdupq_n_u8(0x03);
                let offset_32 = vdupq_n_s8(32);
                let z = vdupq_n_s32(0);

                for bi in 0..blocks_per_row {
                    let blk =
                        &*((a_base + row_start + bi * size_of::<BlockQ6K>()) as *const BlockQ6K);
                    let d = f16::from_bits(blk.d).to_f32();
                    let ql = blk.ql.as_ptr();
                    let qh = blk.qh.as_ptr();
                    let sc = blk.scales.as_ptr();
                    let xq_off = bi * 256;

                    // Fused extraction + dot product: extract 16 6-bit quants, immediately
                    // dot with Q8_0 input. No intermediate buffer — stays in registers.
                    // Scale index tracks which of the 16 sub-block scales to use.
                    let mut sc_idx = 0usize;
                    let mut ql_p = 0usize;
                    let mut qh_p = 0usize;
                    let mut y_p = 0usize;

                    for _pass in 0..2 {
                        for half in 0..2 {
                            let l_off = half * 16;
                            let ql_lo_v = vld1q_u8(ql.add(ql_p + l_off));
                            let ql_hi_v = vld1q_u8(ql.add(ql_p + l_off + 32));
                            let qh_v = vld1q_u8(qh.add(qh_p + l_off));

                            // q1: values at y_p + l_off (16 values, sc_idx)
                            let q1 = vsubq_s8(
                                vreinterpretq_s8_u8(vorrq_u8(
                                    vandq_u8(ql_lo_v, mask_0f),
                                    vshlq_n_u8::<4>(vandq_u8(qh_v, mask_03)),
                                )),
                                offset_32,
                            );
                            let xv1 = vld1q_s8((xq_base as *const i8).add(xq_off + y_p + l_off));
                            let q8_bi1 = (xq_off + y_p + l_off) / 32;
                            let d1 =
                                d * (*sc.add(sc_idx) as f32) * *(xs_base as *const f32).add(q8_bi1);
                            sumf += d1 * vaddvq_s32(vdotq_s32(z, q1, xv1)) as f32;

                            // q2: values at y_p + l_off + 32 (16 values, sc_idx + 2)
                            let q2 = vsubq_s8(
                                vreinterpretq_s8_u8(vorrq_u8(
                                    vandq_u8(ql_hi_v, mask_0f),
                                    vshlq_n_u8::<4>(vandq_u8(vshrq_n_u8::<2>(qh_v), mask_03)),
                                )),
                                offset_32,
                            );
                            let xv2 =
                                vld1q_s8((xq_base as *const i8).add(xq_off + y_p + l_off + 32));
                            let q8_bi2 = (xq_off + y_p + l_off + 32) / 32;
                            let d2 = d
                                * (*sc.add(sc_idx + 2) as f32)
                                * *(xs_base as *const f32).add(q8_bi2);
                            sumf += d2 * vaddvq_s32(vdotq_s32(z, q2, xv2)) as f32;

                            // q3: values at y_p + l_off + 64 (16 values, sc_idx + 4)
                            let q3 = vsubq_s8(
                                vreinterpretq_s8_u8(vorrq_u8(
                                    vshrq_n_u8::<4>(ql_lo_v),
                                    vshlq_n_u8::<4>(vandq_u8(vshrq_n_u8::<4>(qh_v), mask_03)),
                                )),
                                offset_32,
                            );
                            let xv3 =
                                vld1q_s8((xq_base as *const i8).add(xq_off + y_p + l_off + 64));
                            let q8_bi3 = (xq_off + y_p + l_off + 64) / 32;
                            let d3 = d
                                * (*sc.add(sc_idx + 4) as f32)
                                * *(xs_base as *const f32).add(q8_bi3);
                            sumf += d3 * vaddvq_s32(vdotq_s32(z, q3, xv3)) as f32;

                            // q4: values at y_p + l_off + 96 (16 values, sc_idx + 6)
                            let q4 = vsubq_s8(
                                vreinterpretq_s8_u8(vorrq_u8(
                                    vshrq_n_u8::<4>(ql_hi_v),
                                    vshlq_n_u8::<4>(vshrq_n_u8::<6>(qh_v)),
                                )),
                                offset_32,
                            );
                            let xv4 =
                                vld1q_s8((xq_base as *const i8).add(xq_off + y_p + l_off + 96));
                            let q8_bi4 = (xq_off + y_p + l_off + 96) / 32;
                            let d4 = d
                                * (*sc.add(sc_idx + 6) as f32)
                                * *(xs_base as *const f32).add(q8_bi4);
                            sumf += d4 * vaddvq_s32(vdotq_s32(z, q4, xv4)) as f32;

                            sc_idx += 1; // advance by 1 per half (is = l/16 = half)
                        }
                        y_p += 128;
                        ql_p += 64;
                        qh_p += 32;
                        sc_idx = 8; // second pass uses scales 8..15
                    }
                }

                *yi = sumf;
            };

            if y.len() >= super::super::cpu::GEMV_PAR_THRESHOLD {
                crate::backend::cpu::par_rows(y, 512, compute_row);
            } else {
                y.iter_mut().enumerate().for_each(compute_row);
            }
        }
    }

    /// Batched GEMM: C[m, n] = A_q4_0[m, k] @ B_q8_0[k, n].
    ///
    /// `b_scales` layout: n columns × blocks_per_col, i.e. b_scales[j * nb + b]
    /// `b_quants` layout: n columns × k elements, i.e. b_quants[j * k + b*32..]
    /// `out` layout: row-major m × n, i.e. out[i * n + j]
    ///
    /// Reads each weight row once and dots against all n Q8_0 columns.
    /// Uses 4-column grouping to amortize Q4_0 nibble extraction.
    /// Parallelized across output rows with rayon.
    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn gemm_q4_0_q8_0_neon(
        a_quant: &[u8],
        b_scales: &[f32],
        b_quants: &[i8],
        out: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) {
        debug_assert_eq!(k % 32, 0, "GEMM: k must be divisible by 32");
        debug_assert_eq!(a_quant.len(), m * (k / 32) * size_of::<BlockQ4_0>());
        debug_assert_eq!(b_quants.len(), n * k);
        debug_assert_eq!(b_scales.len(), n * (k / 32));
        debug_assert_eq!(out.len(), m * n);
        unsafe {
            let nb = k / 32;
            let row_bytes = nb * size_of::<BlockQ4_0>();
            let a_ptr = a_quant.as_ptr() as usize;
            let bq_ptr = b_quants.as_ptr() as usize;
            let bs_ptr = b_scales.as_ptr() as usize;

            // Decode helpers — Q4_0 nibble extraction is the only difference vs Q8_0 GEMM.
            macro_rules! decode_q4_pair {
                ($a_base:expr, $off0:expr, $off1:expr, $mask_lo:expr, $offset_8:expr) => {{
                    let b0 = &*($a_base.add($off0) as *const BlockQ4_0);
                    let b1 = &*($a_base.add($off1) as *const BlockQ4_0);
                    let v0 = vld1q_u8(b0.qs.as_ptr());
                    let v1 = vld1q_u8(b1.qs.as_ptr());
                    (
                        vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v0, $mask_lo)), $offset_8),
                        vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(v0)), $offset_8),
                        vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v1, $mask_lo)), $offset_8),
                        vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(v1)), $offset_8),
                        f16::from_bits(b0.d).to_f32(),
                        f16::from_bits(b1.d).to_f32(),
                    )
                }};
            }
            macro_rules! decode_q4_single {
                ($a_base:expr, $off:expr, $mask_lo:expr, $offset_8:expr) => {{
                    let b = &*($a_base.add($off) as *const BlockQ4_0);
                    let v = vld1q_u8(b.qs.as_ptr());
                    (
                        vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v, $mask_lo)), $offset_8),
                        vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(v)), $offset_8),
                        f16::from_bits(b.d).to_f32(),
                    )
                }};
            }

            let compute_row = move |(i, row_out): (usize, &mut [f32])| unsafe {
                let mask_lo = vdupq_n_u8(0x0F);
                let offset_8 = vdupq_n_s8(0x8);
                let rs = i * row_bytes;
                let a = a_ptr as *const u8;
                let bq = bq_ptr as *const i8;
                let bs = bs_ptr as *const f32;
                let bsz = size_of::<BlockQ4_0>();

                // Process 8 columns at a time (halves Q4_0 decode count vs 4-col)
                let mut j = 0usize;
                while j + 8 <= n {
                    let mut s0a = vdupq_n_f32(0.0);
                    let mut s0b = vdupq_n_f32(0.0);
                    let mut s1a = vdupq_n_f32(0.0);
                    let mut s1b = vdupq_n_f32(0.0);
                    let mut s2a = vdupq_n_f32(0.0);
                    let mut s2b = vdupq_n_f32(0.0);
                    let mut s3a = vdupq_n_f32(0.0);
                    let mut s3b = vdupq_n_f32(0.0);
                    let mut s4a = vdupq_n_f32(0.0);
                    let mut s4b = vdupq_n_f32(0.0);
                    let mut s5a = vdupq_n_f32(0.0);
                    let mut s5b = vdupq_n_f32(0.0);
                    let mut s6a = vdupq_n_f32(0.0);
                    let mut s6b = vdupq_n_f32(0.0);
                    let mut s7a = vdupq_n_f32(0.0);
                    let mut s7b = vdupq_n_f32(0.0);
                    let xq = [
                        bq.add(j * k),
                        bq.add((j + 1) * k),
                        bq.add((j + 2) * k),
                        bq.add((j + 3) * k),
                        bq.add((j + 4) * k),
                        bq.add((j + 5) * k),
                        bq.add((j + 6) * k),
                        bq.add((j + 7) * k),
                    ];
                    let xs = [
                        bs.add(j * nb),
                        bs.add((j + 1) * nb),
                        bs.add((j + 2) * nb),
                        bs.add((j + 3) * nb),
                        bs.add((j + 4) * nb),
                        bs.add((j + 5) * nb),
                        bs.add((j + 6) * nb),
                        bs.add((j + 7) * nb),
                    ];

                    let mut bi = 0usize;
                    while bi + 1 < nb {
                        if bi + 3 < nb {
                            _prefetch(
                                a.add(rs + (bi + 2) * bsz) as *const i8,
                                _PREFETCH_READ,
                                _PREFETCH_LOCALITY2,
                            );
                        }
                        let (w0l, w0h, w1l, w1h, d0, d1) = decode_q4_pair!(
                            a,
                            rs + bi * bsz,
                            rs + (bi + 1) * bsz,
                            mask_lo,
                            offset_8
                        );
                        gemm_dot_pair!(w0l, w0h, w1l, w1h, d0, d1, xq[0], xs[0], bi, s0a, s0b);
                        gemm_dot_pair!(w0l, w0h, w1l, w1h, d0, d1, xq[1], xs[1], bi, s1a, s1b);
                        gemm_dot_pair!(w0l, w0h, w1l, w1h, d0, d1, xq[2], xs[2], bi, s2a, s2b);
                        gemm_dot_pair!(w0l, w0h, w1l, w1h, d0, d1, xq[3], xs[3], bi, s3a, s3b);
                        gemm_dot_pair!(w0l, w0h, w1l, w1h, d0, d1, xq[4], xs[4], bi, s4a, s4b);
                        gemm_dot_pair!(w0l, w0h, w1l, w1h, d0, d1, xq[5], xs[5], bi, s5a, s5b);
                        gemm_dot_pair!(w0l, w0h, w1l, w1h, d0, d1, xq[6], xs[6], bi, s6a, s6b);
                        gemm_dot_pair!(w0l, w0h, w1l, w1h, d0, d1, xq[7], xs[7], bi, s7a, s7b);
                        bi += 2;
                    }
                    if bi < nb {
                        let (wl, wh, d) = decode_q4_single!(a, rs + bi * bsz, mask_lo, offset_8);
                        gemm_dot_single!(wl, wh, d, xq[0], xs[0], bi, s0a);
                        gemm_dot_single!(wl, wh, d, xq[1], xs[1], bi, s1a);
                        gemm_dot_single!(wl, wh, d, xq[2], xs[2], bi, s2a);
                        gemm_dot_single!(wl, wh, d, xq[3], xs[3], bi, s3a);
                        gemm_dot_single!(wl, wh, d, xq[4], xs[4], bi, s4a);
                        gemm_dot_single!(wl, wh, d, xq[5], xs[5], bi, s5a);
                        gemm_dot_single!(wl, wh, d, xq[6], xs[6], bi, s6a);
                        gemm_dot_single!(wl, wh, d, xq[7], xs[7], bi, s7a);
                    }
                    row_out[j] = vaddvq_f32(s0a) + vaddvq_f32(s0b);
                    row_out[j + 1] = vaddvq_f32(s1a) + vaddvq_f32(s1b);
                    row_out[j + 2] = vaddvq_f32(s2a) + vaddvq_f32(s2b);
                    row_out[j + 3] = vaddvq_f32(s3a) + vaddvq_f32(s3b);
                    row_out[j + 4] = vaddvq_f32(s4a) + vaddvq_f32(s4b);
                    row_out[j + 5] = vaddvq_f32(s5a) + vaddvq_f32(s5b);
                    row_out[j + 6] = vaddvq_f32(s6a) + vaddvq_f32(s6b);
                    row_out[j + 7] = vaddvq_f32(s7a) + vaddvq_f32(s7b);
                    j += 8;
                }
                // 4-column remainder
                while j + 4 <= n {
                    let mut s0a = vdupq_n_f32(0.0);
                    let mut s0b = vdupq_n_f32(0.0);
                    let mut s1a = vdupq_n_f32(0.0);
                    let mut s1b = vdupq_n_f32(0.0);
                    let mut s2a = vdupq_n_f32(0.0);
                    let mut s2b = vdupq_n_f32(0.0);
                    let mut s3a = vdupq_n_f32(0.0);
                    let mut s3b = vdupq_n_f32(0.0);
                    let (xq0, xq1, xq2, xq3) = (
                        bq.add(j * k),
                        bq.add((j + 1) * k),
                        bq.add((j + 2) * k),
                        bq.add((j + 3) * k),
                    );
                    let (xs0, xs1, xs2, xs3) = (
                        bs.add(j * nb),
                        bs.add((j + 1) * nb),
                        bs.add((j + 2) * nb),
                        bs.add((j + 3) * nb),
                    );
                    let mut bi = 0usize;
                    while bi + 1 < nb {
                        let (w0l, w0h, w1l, w1h, d0, d1) = decode_q4_pair!(
                            a,
                            rs + bi * bsz,
                            rs + (bi + 1) * bsz,
                            mask_lo,
                            offset_8
                        );
                        gemm_dot_pair!(w0l, w0h, w1l, w1h, d0, d1, xq0, xs0, bi, s0a, s0b);
                        gemm_dot_pair!(w0l, w0h, w1l, w1h, d0, d1, xq1, xs1, bi, s1a, s1b);
                        gemm_dot_pair!(w0l, w0h, w1l, w1h, d0, d1, xq2, xs2, bi, s2a, s2b);
                        gemm_dot_pair!(w0l, w0h, w1l, w1h, d0, d1, xq3, xs3, bi, s3a, s3b);
                        bi += 2;
                    }
                    if bi < nb {
                        let (wl, wh, d) = decode_q4_single!(a, rs + bi * bsz, mask_lo, offset_8);
                        gemm_dot_single!(wl, wh, d, xq0, xs0, bi, s0a);
                        gemm_dot_single!(wl, wh, d, xq1, xs1, bi, s1a);
                        gemm_dot_single!(wl, wh, d, xq2, xs2, bi, s2a);
                        gemm_dot_single!(wl, wh, d, xq3, xs3, bi, s3a);
                    }
                    row_out[j] = vaddvq_f32(s0a) + vaddvq_f32(s0b);
                    row_out[j + 1] = vaddvq_f32(s1a) + vaddvq_f32(s1b);
                    row_out[j + 2] = vaddvq_f32(s2a) + vaddvq_f32(s2b);
                    row_out[j + 3] = vaddvq_f32(s3a) + vaddvq_f32(s3b);
                    j += 4;
                }
                // Remaining columns (< 4)
                while j < n {
                    let mut sumv0 = vdupq_n_f32(0.0);
                    let mut sumv1 = vdupq_n_f32(0.0);
                    let xq = bq.add(j * k);
                    let xs = bs.add(j * nb);
                    let mut bi = 0usize;
                    while bi + 1 < nb {
                        let (w0l, w0h, w1l, w1h, d0, d1) = decode_q4_pair!(
                            a,
                            rs + bi * bsz,
                            rs + (bi + 1) * bsz,
                            mask_lo,
                            offset_8
                        );
                        gemm_dot_pair!(w0l, w0h, w1l, w1h, d0, d1, xq, xs, bi, sumv0, sumv1);
                        bi += 2;
                    }
                    if bi < nb {
                        let (wl, wh, d) = decode_q4_single!(a, rs + bi * bsz, mask_lo, offset_8);
                        gemm_dot_single!(wl, wh, d, xq, xs, bi, sumv0);
                    }
                    row_out[j] = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
                    j += 1;
                }
            };

            if m >= super::super::cpu::GEMV_PAR_THRESHOLD {
                crate::backend::cpu::par_rows_n(out, n, 64, compute_row);
            } else {
                out.chunks_mut(n).enumerate().for_each(compute_row);
            }
        }
    }

    /// Batched GEMM: C[m, n] = A_q8_0[m, k] @ B_q8_0[k, n].
    ///
    /// Same layout and shared dot-product macros as Q4_0 GEMM, but with
    /// Q8_0 weight blocks (direct i8 load, no nibble extraction).
    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn gemm_q8_0_q8_0_neon(
        a_quant: &[u8],
        b_scales: &[f32],
        b_quants: &[i8],
        out: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) {
        debug_assert_eq!(k % 32, 0, "GEMM: k must be divisible by 32");
        debug_assert_eq!(a_quant.len(), m * (k / 32) * size_of::<BlockQ8_0>());
        debug_assert_eq!(b_quants.len(), n * k);
        debug_assert_eq!(b_scales.len(), n * (k / 32));
        debug_assert_eq!(out.len(), m * n);
        unsafe {
            let nb = k / 32;
            let row_bytes = nb * size_of::<BlockQ8_0>();
            let a_ptr = a_quant.as_ptr() as usize;
            let bq_ptr = b_quants.as_ptr() as usize;
            let bs_ptr = b_scales.as_ptr() as usize;

            // Q8_0 decode: direct i8 load (no nibble extraction needed).
            macro_rules! decode_q8_pair {
                ($a_base:expr, $off0:expr, $off1:expr) => {{
                    let b0 = &*($a_base.add($off0) as *const BlockQ8_0);
                    let b1 = &*($a_base.add($off1) as *const BlockQ8_0);
                    (
                        vld1q_s8(b0.quants.as_ptr()),
                        vld1q_s8(b0.quants.as_ptr().add(16)),
                        vld1q_s8(b1.quants.as_ptr()),
                        vld1q_s8(b1.quants.as_ptr().add(16)),
                        f16::from_bits(b0.delta).to_f32(),
                        f16::from_bits(b1.delta).to_f32(),
                    )
                }};
            }
            macro_rules! decode_q8_single {
                ($a_base:expr, $off:expr) => {{
                    let b = &*($a_base.add($off) as *const BlockQ8_0);
                    (
                        vld1q_s8(b.quants.as_ptr()),
                        vld1q_s8(b.quants.as_ptr().add(16)),
                        f16::from_bits(b.delta).to_f32(),
                    )
                }};
            }

            let compute_row = move |(i, row_out): (usize, &mut [f32])| unsafe {
                let rs = i * row_bytes;
                let a = a_ptr as *const u8;
                let bq = bq_ptr as *const i8;
                let bs = bs_ptr as *const f32;
                let bsz = size_of::<BlockQ8_0>();

                // Single-column loop (Q8_0 has no nibble decode to amortize,
                // so 4-column grouping provides minimal benefit)
                for j in 0..n {
                    let mut sumv0 = vdupq_n_f32(0.0);
                    let mut sumv1 = vdupq_n_f32(0.0);
                    let xq = bq.add(j * k);
                    let xs = bs.add(j * nb);
                    let mut bi = 0usize;
                    while bi + 1 < nb {
                        let (w0l, w0h, w1l, w1h, d0, d1) =
                            decode_q8_pair!(a, rs + bi * bsz, rs + (bi + 1) * bsz);
                        gemm_dot_pair!(w0l, w0h, w1l, w1h, d0, d1, xq, xs, bi, sumv0, sumv1);
                        bi += 2;
                    }
                    if bi < nb {
                        let (wl, wh, d) = decode_q8_single!(a, rs + bi * bsz);
                        gemm_dot_single!(wl, wh, d, xq, xs, bi, sumv0);
                    }
                    row_out[j] = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
                }
            };

            if m >= super::super::cpu::GEMV_PAR_THRESHOLD {
                crate::backend::cpu::par_rows_n(out, n, 64, compute_row);
            } else {
                out.chunks_mut(n).enumerate().for_each(compute_row);
            }
        }
    }

    /// NEON Q6_K GEMV: quantizes x to Q8_0 using scratch, then calls integer path.
    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn gemv_q6k_f32_neon(
        a_quant: &[u8],
        x: &[f32],
        y: &mut [f32],
        _m: usize,
        k: usize,
        q8_scales: &mut Vec<f32>,
        q8_quants: &mut Vec<i8>,
    ) {
        unsafe {
            let n_blocks = k / 32;
            q8_scales.resize(n_blocks, 0.0);
            q8_quants.resize(k, 0);
            quantize_f32_to_q8_0_neon(x, q8_scales, q8_quants);
            gemv_q6k_q8_0_neon(a_quant, q8_scales, q8_quants, y, _m, k);
        }
    }
}

// ── x86_64 AVX2 ─────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::*;
    use std::arch::x86_64::*;

    /// AVX2-optimized Q8_0 dot product with f32 vector.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn vec_dot_q8_0_f32_avx2(block: &BlockQ8_0, y: &[f32]) -> f32 {
        unsafe {
            debug_assert_eq!(y.len(), 32);
            let d = f16::from_bits(block.delta).to_f32();

            let mut sum256 = _mm256_setzero_ps();
            let quants_ptr = block.quants.as_ptr();
            let y_ptr = y.as_ptr();

            for i in (0..32).step_by(8) {
                let q = [
                    *quants_ptr.add(i) as i32,
                    *quants_ptr.add(i + 1) as i32,
                    *quants_ptr.add(i + 2) as i32,
                    *quants_ptr.add(i + 3) as i32,
                    *quants_ptr.add(i + 4) as i32,
                    *quants_ptr.add(i + 5) as i32,
                    *quants_ptr.add(i + 6) as i32,
                    *quants_ptr.add(i + 7) as i32,
                ];
                let qi32 = _mm256_loadu_si256(q.as_ptr() as *const __m256i);
                let qf32 = _mm256_cvtepi32_ps(qi32);
                let yv = _mm256_loadu_ps(y_ptr.add(i));
                sum256 = _mm256_fmadd_ps(qf32, yv, sum256);
            }

            d * hsum_avx(sum256)
        }
    }

    /// AVX2-optimized Q4_0 dot product with f32 vector.
    ///
    /// Loads all 16 qs bytes at once, extracts nibbles with vector AND/SHIFT,
    /// then widens to i32 and converts to f32 for FMA.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn vec_dot_q4_0_f32_avx2(block: &BlockQ4_0, y: &[f32]) -> f32 {
        unsafe {
            debug_assert_eq!(y.len(), 32);
            let d = f16::from_bits(block.d).to_f32();
            let offset = _mm256_set1_ps(8.0);
            let mask_lo = _mm_set1_epi8(0x0F);

            let mut sum256 = _mm256_setzero_ps();
            let y_ptr = y.as_ptr();

            // Load all 16 bytes of qs
            let qbytes = _mm_loadu_si128(block.qs.as_ptr() as *const __m128i);

            // Extract low nibbles (AND with 0x0F) and high nibbles (shift right 4)
            let lo_bytes = _mm_and_si128(qbytes, mask_lo);
            let hi_bytes = _mm_and_si128(_mm_srli_epi16(qbytes, 4), mask_lo);

            // Process low nibbles: 16 u8 values → 2 groups of 8 i32 → f32
            // First 8 low nibbles
            let lo_0_i32 = _mm256_cvtepu8_epi32(lo_bytes); // lower 8 bytes → 8 i32
            let lo_0_f32 = _mm256_sub_ps(_mm256_cvtepi32_ps(lo_0_i32), offset);
            sum256 = _mm256_fmadd_ps(lo_0_f32, _mm256_loadu_ps(y_ptr), sum256);

            // Next 8 low nibbles
            let lo_hi_half = _mm_srli_si128(lo_bytes, 8); // shift right 8 bytes
            let lo_1_i32 = _mm256_cvtepu8_epi32(lo_hi_half);
            let lo_1_f32 = _mm256_sub_ps(_mm256_cvtepi32_ps(lo_1_i32), offset);
            sum256 = _mm256_fmadd_ps(lo_1_f32, _mm256_loadu_ps(y_ptr.add(8)), sum256);

            // Process high nibbles: same pattern, y offset by 16
            let hi_0_i32 = _mm256_cvtepu8_epi32(hi_bytes);
            let hi_0_f32 = _mm256_sub_ps(_mm256_cvtepi32_ps(hi_0_i32), offset);
            sum256 = _mm256_fmadd_ps(hi_0_f32, _mm256_loadu_ps(y_ptr.add(16)), sum256);

            let hi_hi_half = _mm_srli_si128(hi_bytes, 8);
            let hi_1_i32 = _mm256_cvtepu8_epi32(hi_hi_half);
            let hi_1_f32 = _mm256_sub_ps(_mm256_cvtepi32_ps(hi_1_i32), offset);
            sum256 = _mm256_fmadd_ps(hi_1_f32, _mm256_loadu_ps(y_ptr.add(24)), sum256);

            d * hsum_avx(sum256)
        }
    }

    /// AVX2-optimized Q4_K_M dot product with f32 vector.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn vec_dot_q4_k_m_f32_avx2(block: &BlockQ4KM, y: &[f32]) -> f32 {
        unsafe {
            let d = f16::from_bits(block.d).to_f32();
            let dmin = f16::from_bits(block.dmin).to_f32();

            let scales = &block.scales;
            let mut sc = [0u8; 8];
            let mut mn = [0u8; 8];
            for j in 0..4 {
                sc[j] = scales[j] & 63;
                mn[j] = scales[j + 4] & 63;
            }
            for j in 4..8 {
                sc[j] = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
                mn[j] = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
            }

            let qs = &block.qs;
            let y_ptr = y.as_ptr();
            let mut sumf = 0.0f32;
            let mut qi = 0usize;
            let mut yi = 0usize;

            for j in 0..4 {
                let sc1 = d * sc[j * 2] as f32;
                let mn1 = dmin * mn[j * 2] as f32;
                let sc2 = d * sc[j * 2 + 1] as f32;
                let mn2 = dmin * mn[j * 2 + 1] as f32;

                let mut sum1_acc = _mm256_setzero_ps();
                let mut sum2_acc = _mm256_setzero_ps();
                let mut mn1_acc = _mm256_setzero_ps();
                let mut mn2_acc = _mm256_setzero_ps();

                for l in (0..32).step_by(8) {
                    let mut lo_arr = [0i32; 8];
                    let mut hi_arr = [0i32; 8];
                    for k in 0..8 {
                        lo_arr[k] = (qs[qi + l + k] & 0xF) as i32;
                        hi_arr[k] = (qs[qi + l + k] >> 4) as i32;
                    }

                    let lo_f32 =
                        _mm256_cvtepi32_ps(_mm256_loadu_si256(lo_arr.as_ptr() as *const __m256i));
                    let hi_f32 =
                        _mm256_cvtepi32_ps(_mm256_loadu_si256(hi_arr.as_ptr() as *const __m256i));

                    let y1 = _mm256_loadu_ps(y_ptr.add(yi + l));
                    let y2 = _mm256_loadu_ps(y_ptr.add(yi + l + 32));

                    sum1_acc = _mm256_fmadd_ps(lo_f32, y1, sum1_acc);
                    sum2_acc = _mm256_fmadd_ps(hi_f32, y2, sum2_acc);
                    mn1_acc = _mm256_add_ps(mn1_acc, y1);
                    mn2_acc = _mm256_add_ps(mn2_acc, y2);
                }

                sumf += sc1 * hsum_avx(sum1_acc) + sc2 * hsum_avx(sum2_acc)
                    - mn1 * hsum_avx(mn1_acc)
                    - mn2 * hsum_avx(mn2_acc);
                qi += 32;
                yi += 64;
            }

            sumf
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn hsum_avx(v: __m256) -> f32 {
        let hi128 = _mm256_extractf128_ps(v, 1);
        let lo128 = _mm256_castps256_ps128(v);
        let sum128 = _mm_add_ps(lo128, hi128);
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
        _mm_cvtss_f32(sum32)
    }
}

// ── Dispatch ────────────────────────────────────────────────────────────────

/// Best available Q4_0 dot product.
pub fn vec_dot_q4_0_f32(block: &BlockQ4_0, y: &[f32]) -> f32 {
    assert_eq!(y.len(), 32, "Q4_0 vec_dot requires y.len() == 32");

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon::vec_dot_q4_0_f32_neon(block, y) }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { avx2::vec_dot_q4_0_f32_avx2(block, y) }
        } else {
            crate::quant::vec_dot_q4_0_f32_scalar(block, y)
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        crate::quant::vec_dot_q4_0_f32_scalar(block, y)
    }
}

/// Best available Q8_0 dot product.
pub fn vec_dot_q8_0_f32(block: &BlockQ8_0, y: &[f32]) -> f32 {
    assert_eq!(y.len(), 32, "Q8_0 vec_dot requires y.len() == 32");

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon::vec_dot_q8_0_f32_neon(block, y) }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { avx2::vec_dot_q8_0_f32_avx2(block, y) }
        } else {
            crate::quant::vec_dot_q8_0_f32_scalar(block, y)
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        crate::quant::vec_dot_q8_0_f32_scalar(block, y)
    }
}

/// Best available Q4_K_M dot product.
pub fn vec_dot_q4_k_m_f32(block: &BlockQ4KM, y: &[f32]) -> f32 {
    assert_eq!(y.len(), 256, "Q4_K_M vec_dot requires y.len() == 256");
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon::vec_dot_q4_k_m_f32_neon(block, y) }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { avx2::vec_dot_q4_k_m_f32_avx2(block, y) }
        } else {
            crate::quant::vec_dot_q4_k_m_f32_scalar(block, y)
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        crate::quant::vec_dot_q4_k_m_f32_scalar(block, y)
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_q4_0_matches_scalar() {
        let block = BlockQ4_0 {
            d: f16::from_f32(0.5).to_bits(),
            qs: {
                let mut q = [0u8; 16];
                for i in 0..16 {
                    q[i] = ((i % 13) as u8) | (((i % 7) as u8) << 4);
                }
                q
            },
        };
        let y: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();

        let scalar = crate::quant::vec_dot_q4_0_f32_scalar(&block, &y);
        let simd = vec_dot_q4_0_f32(&block, &y);

        assert!(
            (scalar - simd).abs() < 1e-3,
            "SIMD Q4_0 mismatch: scalar={scalar}, simd={simd}"
        );
    }

    #[test]
    fn test_simd_q8_0_matches_scalar() {
        let block = BlockQ8_0 {
            delta: f16::from_f32(0.3).to_bits(),
            quants: {
                let mut q = [0i8; 32];
                for i in 0..32 {
                    q[i] = (i as i8) * 3 - 48;
                }
                q
            },
        };
        let y: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();

        let scalar = crate::quant::vec_dot_q8_0_f32_scalar(&block, &y);
        let simd = vec_dot_q8_0_f32(&block, &y);

        assert!(
            (scalar - simd).abs() < 1e-3,
            "SIMD Q8_0 mismatch: scalar={scalar}, simd={simd}"
        );
    }

    #[test]
    fn test_simd_q4km_matches_scalar() {
        let mut block = BlockQ4KM {
            d: f16::from_f32(0.5).to_bits(),
            dmin: f16::from_f32(0.1).to_bits(),
            scales: [0u8; 12],
            qs: [0u8; 128],
        };

        for i in 0..4 {
            block.scales[i] = 3;
        }
        for i in 4..8 {
            block.scales[i] = 1;
        }
        for i in 8..12 {
            block.scales[i] = 0x12;
        }

        for (i, b) in block.qs.iter_mut().enumerate() {
            *b = ((i % 13) as u8) | (((i % 9) as u8) << 4);
        }

        let y: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();

        let scalar = crate::quant::vec_dot_q4_k_m_f32_scalar(&block, &y);
        let simd = vec_dot_q4_k_m_f32(&block, &y);

        assert!(
            (scalar - simd).abs() < 1e-2,
            "SIMD Q4_K_M mismatch: scalar={scalar}, simd={simd}"
        );
    }

    /// Build a Q4_0-quantized weight matrix (m rows × k cols) from f32 values.
    /// Returns raw bytes suitable for GEMM kernels.
    #[cfg(target_arch = "aarch64")]
    fn build_q4_0_matrix(values: &[f32], m: usize, k: usize) -> Vec<u8> {
        assert_eq!(values.len(), m * k);
        let nb = k / 32;
        let mut bytes = Vec::new();
        for row in 0..m {
            for b in 0..nb {
                let block_start = row * k + b * 32;
                let block = &values[block_start..block_start + 32];
                let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                let d = amax / 7.0;
                let d_f16 = half::f16::from_f32(d);
                bytes.extend_from_slice(&d_f16.to_bits().to_le_bytes());
                let id = if d != 0.0 { 1.0 / d } else { 0.0 };
                let mut qs = [0u8; 16];
                for i in 0..16 {
                    let lo = ((block[i] * id + 8.5) as u8).min(15);
                    let hi = ((block[16 + i] * id + 8.5) as u8).min(15);
                    qs[i] = lo | (hi << 4);
                }
                bytes.extend_from_slice(&qs);
            }
        }
        bytes
    }

    /// Build a Q8_0-quantized weight matrix (m rows × k cols) from f32 values.
    #[cfg(target_arch = "aarch64")]
    fn build_q8_0_matrix(values: &[f32], m: usize, k: usize) -> Vec<u8> {
        assert_eq!(values.len(), m * k);
        let nb = k / 32;
        let mut bytes = Vec::new();
        for row in 0..m {
            for b in 0..nb {
                let block_start = row * k + b * 32;
                let block = &values[block_start..block_start + 32];
                let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                let d = amax / 127.0;
                let d_f16 = half::f16::from_f32(d);
                bytes.extend_from_slice(&d_f16.to_bits().to_le_bytes());
                let id = if d != 0.0 { 1.0 / d } else { 0.0 };
                let mut qs = [0i8; 32];
                for i in 0..32 {
                    qs[i] = (block[i] * id).round().clamp(-128.0, 127.0) as i8;
                }
                bytes.extend_from_slice(bytemuck::cast_slice(&qs));
            }
        }
        bytes
    }

    /// Quantize n columns of f32 input to Q8_0 format (scales + quants).
    #[cfg(target_arch = "aarch64")]
    fn quantize_input_columns(inputs: &[Vec<f32>], k: usize) -> (Vec<f32>, Vec<i8>) {
        let n = inputs.len();
        let nb = k / 32;
        let mut scales = vec![0.0f32; n * nb];
        let mut quants = vec![0i8; n * k];
        for (j, col) in inputs.iter().enumerate() {
            unsafe {
                neon::quantize_f32_to_q8_0_neon(
                    col,
                    &mut scales[j * nb..(j + 1) * nb],
                    &mut quants[j * k..(j + 1) * k],
                );
            }
        }
        (scales, quants)
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_gemm_q4_0_matches_sequential_gemv() {
        let m = 8;
        let k = 64; // 2 Q4_0 blocks per row
        let n = 7; // not divisible by 4, tests remainder path

        // Random-ish weight values
        let weights: Vec<f32> = (0..m * k)
            .map(|i| ((i * 17 + 3) % 29) as f32 * 0.1 - 1.4)
            .collect();
        let a_bytes = build_q4_0_matrix(&weights, m, k);

        // Random-ish input columns
        let inputs: Vec<Vec<f32>> = (0..n)
            .map(|j| {
                (0..k)
                    .map(|i| ((i * 13 + j * 7 + 5) % 23) as f32 * 0.2 - 2.3)
                    .collect()
            })
            .collect();

        let (b_scales, b_quants) = quantize_input_columns(&inputs, k);

        // GEMM
        let mut gemm_out = vec![0.0f32; m * n];
        unsafe {
            neon::gemm_q4_0_q8_0_neon(&a_bytes, &b_scales, &b_quants, &mut gemm_out, m, n, k);
        }

        // Sequential GEMV for each column
        for j in 0..n {
            let col_scales = &b_scales[j * (k / 32)..(j + 1) * (k / 32)];
            let col_quants = &b_quants[j * k..(j + 1) * k];
            let mut gemv_out = vec![0.0f32; m];
            unsafe {
                neon::gemv_q4_0_q8_0_neon(&a_bytes, col_scales, col_quants, &mut gemv_out, m, k);
            }
            for i in 0..m {
                let diff = (gemm_out[i * n + j] - gemv_out[i]).abs();
                assert!(
                    diff < 1e-4,
                    "GEMM/GEMV Q4_0 mismatch at [{i},{j}]: gemm={}, gemv={}, diff={diff}",
                    gemm_out[i * n + j],
                    gemv_out[i]
                );
            }
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_gemm_q8_0_matches_sequential_gemv() {
        let m = 8;
        let k = 64;
        let n = 5;

        let weights: Vec<f32> = (0..m * k)
            .map(|i| ((i * 17 + 3) % 29) as f32 * 0.1 - 1.4)
            .collect();
        let a_bytes = build_q8_0_matrix(&weights, m, k);

        let inputs: Vec<Vec<f32>> = (0..n)
            .map(|j| {
                (0..k)
                    .map(|i| ((i * 13 + j * 7 + 5) % 23) as f32 * 0.2 - 2.3)
                    .collect()
            })
            .collect();

        let (b_scales, b_quants) = quantize_input_columns(&inputs, k);

        let mut gemm_out = vec![0.0f32; m * n];
        unsafe {
            neon::gemm_q8_0_q8_0_neon(&a_bytes, &b_scales, &b_quants, &mut gemm_out, m, n, k);
        }

        for j in 0..n {
            let col_scales = &b_scales[j * (k / 32)..(j + 1) * (k / 32)];
            let col_quants = &b_quants[j * k..(j + 1) * k];
            let mut gemv_out = vec![0.0f32; m];
            unsafe {
                neon::gemv_q8_0_q8_0_neon(&a_bytes, col_scales, col_quants, &mut gemv_out, m, k);
            }
            for i in 0..m {
                let diff = (gemm_out[i * n + j] - gemv_out[i]).abs();
                assert!(
                    diff < 1e-4,
                    "GEMM/GEMV Q8_0 mismatch at [{i},{j}]: gemm={}, gemv={}, diff={diff}",
                    gemm_out[i * n + j],
                    gemv_out[i]
                );
            }
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_gemm_q4_0_single_column() {
        // n=1: tests single-column fallback path
        let m = 4;
        let k = 32;
        let n = 1;

        let weights: Vec<f32> = (0..m * k).map(|i| (i % 11) as f32 * 0.3 - 1.5).collect();
        let a_bytes = build_q4_0_matrix(&weights, m, k);
        let inputs = vec![(0..k).map(|i| (i % 7) as f32 * 0.5 - 1.75).collect()];
        let (b_scales, b_quants) = quantize_input_columns(&inputs, k);

        let mut gemm_out = vec![0.0f32; m];
        unsafe {
            neon::gemm_q4_0_q8_0_neon(&a_bytes, &b_scales, &b_quants, &mut gemm_out, m, n, k);
        }

        let mut gemv_out = vec![0.0f32; m];
        unsafe {
            neon::gemv_q4_0_q8_0_neon(&a_bytes, &b_scales, &b_quants, &mut gemv_out, m, k);
        }

        for i in 0..m {
            let diff = (gemm_out[i] - gemv_out[i]).abs();
            assert!(diff < 1e-4, "n=1 mismatch at row {i}: {diff}");
        }
    }
}
