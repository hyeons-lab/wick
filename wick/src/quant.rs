use half::f16;

// ── Block layouts ────────────────────────────────────────────────────────────

/// Q4_0 quantization block: 32 values in 18 bytes.
///
/// Layout:
///   d: f16 (2 bytes) — scale factor
///   qs: [u8; 16] (16 bytes) — 32 4-bit unsigned quantized values (offset by 8)
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ4_0 {
    pub d: u16, // f16 stored as raw bits
    pub qs: [u8; 16],
}

const _: () = assert!(size_of::<BlockQ4_0>() == 18);

/// Q8_0 quantization block: 32 values in 34 bytes.
///
/// Layout:
///   delta: f16 (2 bytes) — scale factor
///   quants: [i8; 32] (32 bytes) — quantized values
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ8_0 {
    pub delta: u16, // f16 stored as raw bits
    pub quants: [i8; 32],
}

const _: () = assert!(size_of::<BlockQ8_0>() == 34);

/// Q4_K_M quantization block: 256 values in 144 bytes.
///
/// Layout:
///   d: f16 (2 bytes) — super-block scale
///   dmin: f16 (2 bytes) — super-block minimum
///   scales: [u8; 12] (12 bytes) — packed sub-block scales and mins
///   qs: [u8; 128] (128 bytes) — 256 4-bit quantized values
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ4KM {
    pub d: u16,    // f16 stored as raw bits
    pub dmin: u16, // f16 stored as raw bits
    pub scales: [u8; 12],
    pub qs: [u8; 128],
}

const _: () = assert!(size_of::<BlockQ4KM>() == 144);

/// Q6_K quantization block: 256 values in 210 bytes.
///
/// Layout (from ggml-common.h):
///   ql: [u8; 128] — lower 4 bits of 6-bit quants
///   qh: [u8; 64]  — upper 2 bits of 6-bit quants
///   scales: [i8; 16] — per-16-element sub-block scales (8-bit signed)
///   d: f16 (2 bytes) — super-block scale
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ6K {
    pub ql: [u8; 128],
    pub qh: [u8; 64],
    pub scales: [i8; 16],
    pub d: u16, // f16 stored as raw bits
}

const _: () = assert!(size_of::<BlockQ6K>() == 210);

// ── Q4_0 dequantization ─────────────────────────────────────────────────────

/// Dequantize a single Q4_0 block to 32 f32 values.
///
/// Each byte in qs holds two 4-bit unsigned values (low nibble, high nibble).
/// Values are offset by -8 to center around zero: value = (nibble - 8) * d.
pub fn dequantize_q4_0_block(block: &BlockQ4_0) -> [f32; 32] {
    let d = f16::from_bits(block.d).to_f32();
    let mut out = [0.0f32; 32];

    for i in 0..16 {
        let byte = block.qs[i];
        let lo = (byte & 0xF) as i32 - 8;
        let hi = (byte >> 4) as i32 - 8;
        out[i] = lo as f32 * d;
        out[i + 16] = hi as f32 * d;
    }
    out
}

/// Dequantize a row of Q4_0 blocks. `src` is raw bytes, `dst` is f32 output.
pub fn dequantize_q4_0_row(src: &[u8], dst: &mut [f32]) {
    let block_size = size_of::<BlockQ4_0>();
    let n_blocks = src.len() / block_size;
    debug_assert_eq!(src.len() % block_size, 0);
    debug_assert_eq!(dst.len(), n_blocks * 32);

    for i in 0..n_blocks {
        let block_bytes = &src[i * block_size..(i + 1) * block_size];
        let block = unsafe { &*(block_bytes.as_ptr() as *const BlockQ4_0) };
        let values = dequantize_q4_0_block(block);
        dst[i * 32..(i + 1) * 32].copy_from_slice(&values);
    }
}

/// Dot product of a Q4_0 block with an f32 vector of length 32. Scalar version.
pub fn vec_dot_q4_0_f32_scalar(block: &BlockQ4_0, y: &[f32]) -> f32 {
    debug_assert_eq!(y.len(), 32);
    let d = f16::from_bits(block.d).to_f32();
    let mut sum = 0.0f32;

    for i in 0..16 {
        let byte = block.qs[i];
        let lo = (byte & 0xF) as i32 - 8;
        let hi = (byte >> 4) as i32 - 8;
        sum += lo as f32 * y[i];
        sum += hi as f32 * y[i + 16];
    }
    sum * d
}

// ── Q8_0 dequantization ─────────────────────────────────────────────────────

/// Dequantize a single Q8_0 block to 32 f32 values.
pub fn dequantize_q8_0_block(block: &BlockQ8_0) -> [f32; 32] {
    let d = f16::from_bits(block.delta).to_f32();
    let mut out = [0.0f32; 32];
    for (o, &q) in out.iter_mut().zip(block.quants.iter()) {
        *o = q as f32 * d;
    }
    out
}

/// Dequantize a row of Q8_0 blocks. `src` is raw bytes, `dst` is f32 output.
pub fn dequantize_q8_0_row(src: &[u8], dst: &mut [f32]) {
    let block_size = size_of::<BlockQ8_0>();
    let n_blocks = src.len() / block_size;
    debug_assert_eq!(src.len() % block_size, 0);
    debug_assert_eq!(dst.len(), n_blocks * 32);

    for i in 0..n_blocks {
        let block_bytes = &src[i * block_size..(i + 1) * block_size];
        // SAFETY: BlockQ8_0 is repr(C, packed) and we've verified the slice length
        let block = unsafe { &*(block_bytes.as_ptr() as *const BlockQ8_0) };
        let values = dequantize_q8_0_block(block);
        dst[i * 32..(i + 1) * 32].copy_from_slice(&values);
    }
}

/// Dot product of a Q8_0 block with an f32 vector of length 32. Scalar version.
pub fn vec_dot_q8_0_f32_scalar(block: &BlockQ8_0, y: &[f32]) -> f32 {
    debug_assert_eq!(y.len(), 32);
    let d = f16::from_bits(block.delta).to_f32();
    let sum: f32 = block
        .quants
        .iter()
        .zip(y.iter())
        .map(|(&q, &y)| q as f32 * y)
        .sum();
    sum * d
}

// ── Q4_K_M dequantization ───────────────────────────────────────────────────

/// Decode the packed sub-block scales and minimums from Q4_K_M's 12-byte scales array.
///
/// Q4_K_M has 8 sub-blocks of 32 values each. The 12 bytes encode:
/// - 8 6-bit scales and 8 6-bit minimums
///
/// Bytes 0-3: low 4 bits of scales[0..3] and mins[0..3]  (packed as scale|min per byte)
///   Wait — actually llama.cpp packs them differently.
///
/// From ggml-quants.c (get_scale_min_k4):
///   j < 4:  sc = scales[j] & 63,      m = scales[j+4] & 63
///   j >= 4: sc = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4),
///           m  = (scales[j+4] >> 4)   | ((scales[j-0] >> 6) << 4)
///
/// Returns (scales[8], mins[8]).
fn decode_q4km_scales(scales: &[u8; 12]) -> ([u8; 8], [u8; 8]) {
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

    (sc, mn)
}

/// Dequantize a single Q4_K_M block to 256 f32 values.
///
/// Ported from llama.cpp's dequantize_row_q4_K.
pub fn dequantize_q4_k_m_block(block: &BlockQ4KM) -> [f32; 256] {
    let d = f16::from_bits(block.d).to_f32();
    let dmin = f16::from_bits(block.dmin).to_f32();
    let (sc, mn) = decode_q4km_scales(&block.scales);

    let mut out = [0.0f32; 256];
    let qs = &block.qs;

    for j in 0..8 {
        // Each sub-block has 32 values
        let sc_val = d * sc[j] as f32;
        let mn_val = dmin * mn[j] as f32;

        // First 16 values: low nibble of qs[j*16..j*16+16]
        // Second 16 values: high nibble of qs[j*16..j*16+16]
        // But the layout is actually:
        //   sub-blocks 0-3 use qs[0..64], lower nibble for 0-1, upper for 2-3
        //   sub-blocks 4-7 use qs[64..128], lower nibble for 4-5, upper for 6-7
        //
        // Actually from llama.cpp:
        //   for (int l = 0; l < 32; ++l) {
        //     *y++ = d * sc[is] * ((q[l] & 0xF) - (m ? dmin * mn[is] : 0))
        //   but that's not right either.
        //
        // Let me re-read the llama.cpp source carefully.
        // The actual layout from dequantize_row_q4_K:
        //
        //   q = qs (pointer to start of qs array)
        //   for j in 0..QK_K/64:     (QK_K=256, so j in 0..4)
        //     sc1 = get_scale(j*2), mn1 = get_min(j*2)
        //     sc2 = get_scale(j*2+1), mn2 = get_min(j*2+1)
        //     for l in 0..32:
        //       y[l+0]  = d * sc1 * (q[l] & 0xF) - dmin * mn1
        //       y[l+32] = d * sc2 * (q[l] >> 4)   - dmin * mn2
        //     q += 32, y += 64
        //
        // So it processes 64 values at a time using 32 bytes of qs.
        // Each byte holds two 4-bit values: low nibble and high nibble.
        let _ = (sc_val, mn_val); // will use below
    }

    // Re-implement following llama.cpp's actual loop structure
    let mut qi = 0; // index into qs
    let mut yi = 0; // index into output

    for j in 0..4 {
        let d_sc1 = d * sc[j * 2] as f32;
        let d_mn1 = dmin * mn[j * 2] as f32;
        let d_sc2 = d * sc[j * 2 + 1] as f32;
        let d_mn2 = dmin * mn[j * 2 + 1] as f32;

        for l in 0..32 {
            out[yi + l] = d_sc1 * (qs[qi + l] & 0xF) as f32 - d_mn1;
            out[yi + l + 32] = d_sc2 * (qs[qi + l] >> 4) as f32 - d_mn2;
        }
        qi += 32;
        yi += 64;
    }

    out
}

/// Dequantize a row of Q4_K_M blocks. `src` is raw bytes, `dst` is f32 output.
pub fn dequantize_q4_k_m_row(src: &[u8], dst: &mut [f32]) {
    let block_size = size_of::<BlockQ4KM>();
    let n_blocks = src.len() / block_size;
    debug_assert_eq!(src.len() % block_size, 0);
    debug_assert_eq!(dst.len(), n_blocks * 256);

    for i in 0..n_blocks {
        let block_bytes = &src[i * block_size..(i + 1) * block_size];
        let block = unsafe { &*(block_bytes.as_ptr() as *const BlockQ4KM) };
        let values = dequantize_q4_k_m_block(block);
        dst[i * 256..(i + 1) * 256].copy_from_slice(&values);
    }
}

/// Dot product of a Q4_K_M block with an f32 vector of length 256. Scalar version.
///
/// Ported from llama.cpp's ggml_vec_dot_q4_K_q8_K.
pub fn vec_dot_q4_k_m_f32_scalar(block: &BlockQ4KM, y: &[f32]) -> f32 {
    debug_assert_eq!(y.len(), 256);

    let d = f16::from_bits(block.d).to_f32();
    let dmin = f16::from_bits(block.dmin).to_f32();
    let (sc, mn) = decode_q4km_scales(&block.scales);
    let qs = &block.qs;

    let mut sumf = 0.0f32;
    let mut qi = 0usize;
    let mut yi = 0usize;

    for j in 0..4 {
        let sc1 = sc[j * 2] as f32;
        let mn1 = mn[j * 2] as f32;
        let sc2 = sc[j * 2 + 1] as f32;
        let mn2 = mn[j * 2 + 1] as f32;

        let mut sum1 = 0.0f32;
        let mut sum2 = 0.0f32;
        let mut sum_mn1 = 0.0f32;
        let mut sum_mn2 = 0.0f32;

        for l in 0..32 {
            sum1 += (qs[qi + l] & 0xF) as f32 * y[yi + l];
            sum2 += (qs[qi + l] >> 4) as f32 * y[yi + l + 32];
            sum_mn1 += y[yi + l];
            sum_mn2 += y[yi + l + 32];
        }

        sumf += d * (sc1 * sum1 + sc2 * sum2) - dmin * (mn1 * sum_mn1 + mn2 * sum_mn2);
        qi += 32;
        yi += 64;
    }

    sumf
}

// ── Q6_K dequantization ────────────────────────────────────────────────────

/// Dequantize a single Q6_K block to 256 f32 values.
///
/// Ported from llama.cpp's `dequantize_row_q6_K`. The 256 values are processed
/// in two passes of 128 values each. Within each pass, 32 iterations produce
/// 4 values each by reassembling 6-bit quants from ql (low 4 bits) and qh (high 2 bits).
pub fn dequantize_q6_k_block(block: &BlockQ6K) -> [f32; 256] {
    let d = f16::from_bits(block.d).to_f32();
    let ql = &block.ql;
    let qh = &block.qh;
    let sc = &block.scales;

    let mut out = [0.0f32; 256];
    let mut ql_off = 0usize;
    let mut qh_off = 0usize;
    let mut sc_off = 0usize;
    let mut y_off = 0usize;

    // Two passes of 128 values (n = 0 and n = 128)
    for _n in 0..2 {
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[ql_off + l] & 0xF) | ((qh[qh_off + l] & 3) << 4)) as i8 - 32;
            let q2 = ((ql[ql_off + l + 32] & 0xF) | (((qh[qh_off + l] >> 2) & 3) << 4)) as i8 - 32;
            let q3 = ((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) as i8 - 32;
            let q4 = ((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)) as i8 - 32;
            out[y_off + l] = d * sc[sc_off + is] as f32 * q1 as f32;
            out[y_off + l + 32] = d * sc[sc_off + is + 2] as f32 * q2 as f32;
            out[y_off + l + 64] = d * sc[sc_off + is + 4] as f32 * q3 as f32;
            out[y_off + l + 96] = d * sc[sc_off + is + 6] as f32 * q4 as f32;
        }
        y_off += 128;
        ql_off += 64;
        qh_off += 32;
        sc_off += 8;
    }

    out
}

/// Dequantize a row of Q6_K blocks. `src` is raw bytes, `dst` is f32 output.
pub fn dequantize_q6_k_row(src: &[u8], dst: &mut [f32]) {
    let block_size = size_of::<BlockQ6K>();
    let n_blocks = src.len() / block_size;
    debug_assert_eq!(src.len() % block_size, 0);
    debug_assert_eq!(dst.len(), n_blocks * 256);

    for i in 0..n_blocks {
        let block_bytes = &src[i * block_size..(i + 1) * block_size];
        // SAFETY: BlockQ6K is repr(C, packed) and we've verified the slice length
        let block = unsafe { &*(block_bytes.as_ptr() as *const BlockQ6K) };
        let values = dequantize_q6_k_block(block);
        dst[i * 256..(i + 1) * 256].copy_from_slice(&values);
    }
}

/// Dot product of a Q6_K block with an f32 vector of length 256. Scalar version.
pub fn vec_dot_q6_k_f32_scalar(block: &BlockQ6K, y: &[f32]) -> f32 {
    debug_assert_eq!(y.len(), 256);
    let d = f16::from_bits(block.d).to_f32();
    let ql = &block.ql;
    let qh = &block.qh;
    let sc = &block.scales;

    let mut sumf = 0.0f32;
    let mut ql_off = 0usize;
    let mut qh_off = 0usize;
    let mut sc_off = 0usize;
    let mut y_off = 0usize;

    for _n in 0..2 {
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[ql_off + l] & 0xF) | ((qh[qh_off + l] & 3) << 4)) as i8 - 32;
            let q2 = ((ql[ql_off + l + 32] & 0xF) | (((qh[qh_off + l] >> 2) & 3) << 4)) as i8 - 32;
            let q3 = ((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) as i8 - 32;
            let q4 = ((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)) as i8 - 32;
            sumf += sc[sc_off + is] as f32 * q1 as f32 * y[y_off + l];
            sumf += sc[sc_off + is + 2] as f32 * q2 as f32 * y[y_off + l + 32];
            sumf += sc[sc_off + is + 4] as f32 * q3 as f32 * y[y_off + l + 64];
            sumf += sc[sc_off + is + 6] as f32 * q4 as f32 * y[y_off + l + 96];
        }
        y_off += 128;
        ql_off += 64;
        qh_off += 32;
        sc_off += 8;
    }

    sumf * d
}

/// Dot product of a Q6_K block with an f32 vector. Dispatches to best available impl.
pub fn vec_dot_q6_k_f32(block: &BlockQ6K, y: &[f32]) -> f32 {
    vec_dot_q6_k_f32_scalar(block, y)
}

// ── Dispatch functions ──────────────────────────────────────────────────────

/// Dot product of a Q4_0 block with an f32 vector. Dispatches to best available impl.
pub fn vec_dot_q4_0_f32(block: &BlockQ4_0, y: &[f32]) -> f32 {
    vec_dot_q4_0_f32_scalar(block, y)
}

/// Dot product of a Q8_0 block with an f32 vector. Dispatches to best available impl.
pub fn vec_dot_q8_0_f32(block: &BlockQ8_0, y: &[f32]) -> f32 {
    vec_dot_q8_0_f32_scalar(block, y)
}

/// Dot product of a Q4_K_M block with an f32 vector. Dispatches to best available impl.
pub fn vec_dot_q4_k_m_f32(block: &BlockQ4KM, y: &[f32]) -> f32 {
    vec_dot_q4_k_m_f32_scalar(block, y)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a Q8_0 block from known values for testing.
    fn make_q8_0_block(scale: f32, quants: [i8; 32]) -> BlockQ8_0 {
        BlockQ8_0 {
            delta: f16::from_f32(scale).to_bits(),
            quants,
        }
    }

    #[test]
    fn test_dequantize_q4_0_simple() {
        // All nibbles = 8 → offset to 0
        let block = BlockQ4_0 {
            d: f16::from_f32(1.0).to_bits(),
            qs: [0x88; 16], // lo=8, hi=8 → both (8-8)*1.0 = 0.0
        };
        let out = dequantize_q4_0_block(&block);
        for (i, &v) in out.iter().enumerate() {
            assert!(v.abs() < 1e-3, "expected 0.0 at {i}, got {v}");
        }
    }

    #[test]
    fn test_dequantize_q4_0_varied() {
        // lo nibbles: 0..16, hi nibbles: all 15
        let mut qs = [0u8; 16];
        for i in 0..16 {
            qs[i] = (i as u8) | (15 << 4);
        }
        let block = BlockQ4_0 {
            d: f16::from_f32(0.5).to_bits(),
            qs,
        };
        let out = dequantize_q4_0_block(&block);

        // First 16: (i - 8) * 0.5
        for i in 0..16 {
            let expected = (i as f32 - 8.0) * 0.5;
            assert!(
                (out[i] - expected).abs() < 1e-3,
                "lo[{i}]: got {}, expected {expected}",
                out[i]
            );
        }
        // Last 16: (15 - 8) * 0.5 = 3.5
        for i in 16..32 {
            assert!(
                (out[i] - 3.5).abs() < 1e-3,
                "hi[{i}]: got {}, expected 3.5",
                out[i]
            );
        }
    }

    #[test]
    fn test_vec_dot_q4_0_matches_dequantize() {
        let mut qs = [0u8; 16];
        for i in 0..16 {
            qs[i] = ((i % 13) as u8) | (((i % 7) as u8) << 4);
        }
        let block = BlockQ4_0 {
            d: f16::from_f32(0.3).to_bits(),
            qs,
        };
        let y: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();

        let dequantized = dequantize_q4_0_block(&block);
        let expected: f32 = dequantized.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        let got = vec_dot_q4_0_f32(&block, &y);

        assert!(
            (got - expected).abs() < 1e-3,
            "vec_dot Q4_0 mismatch: got {got}, expected {expected}"
        );
    }

    #[test]
    fn test_dequantize_q8_0_simple() {
        let block = make_q8_0_block(0.5, {
            let mut q = [0i8; 32];
            for i in 0..32 {
                q[i] = i as i8;
            }
            q
        });
        let out = dequantize_q8_0_block(&block);
        for i in 0..32 {
            let expected = i as f32 * 0.5;
            assert!(
                (out[i] - expected).abs() < 1e-3,
                "mismatch at {i}: got {}, expected {expected}",
                out[i]
            );
        }
    }

    #[test]
    fn test_dequantize_q8_0_row() {
        // Two blocks
        let block1 = make_q8_0_block(1.0, {
            let mut q = [0i8; 32];
            for i in 0..32 {
                q[i] = (i as i8) - 16;
            }
            q
        });
        let block2 = make_q8_0_block(0.25, [1i8; 32]);

        let mut src = vec![0u8; 68];
        unsafe {
            std::ptr::copy_nonoverlapping(&block1 as *const _ as *const u8, src.as_mut_ptr(), 34);
            std::ptr::copy_nonoverlapping(
                &block2 as *const _ as *const u8,
                src.as_mut_ptr().add(34),
                34,
            );
        }

        let mut dst = vec![0.0f32; 64];
        dequantize_q8_0_row(&src, &mut dst);

        // Check block1 values
        for i in 0..32 {
            let expected = (i as f32 - 16.0) * 1.0;
            assert!(
                (dst[i] - expected).abs() < 1e-3,
                "block1[{i}]: got {}, expected {expected}",
                dst[i]
            );
        }
        // Check block2 values
        for i in 0..32 {
            let expected = 1.0 * 0.25;
            assert!(
                (dst[32 + i] - expected).abs() < 1e-3,
                "block2[{i}]: got {}, expected {expected}",
                dst[32 + i]
            );
        }
    }

    #[test]
    fn test_vec_dot_q8_0() {
        let block = make_q8_0_block(0.1, {
            let mut q = [0i8; 32];
            for i in 0..32 {
                q[i] = (i as i8) * 2 - 31;
            }
            q
        });
        let y: Vec<f32> = (0..32).map(|i| i as f32 * 0.5).collect();

        // Compute expected via dequantize
        let dequantized = dequantize_q8_0_block(&block);
        let expected: f32 = dequantized.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        let got = vec_dot_q8_0_f32(&block, &y);

        assert!(
            (got - expected).abs() < 1e-3,
            "vec_dot mismatch: got {got}, expected {expected}"
        );
    }

    #[test]
    fn test_dequantize_q4_k_m_basic() {
        // Create a Q4_K_M block with known values
        let mut block = BlockQ4KM {
            d: f16::from_f32(1.0).to_bits(),
            dmin: f16::from_f32(0.0).to_bits(), // zero min for simplicity
            scales: [0u8; 12],
            qs: [0u8; 128],
        };
        // Set all sub-block scales to 1 (6-bit value)
        for i in 0..4 {
            block.scales[i] = 1; // sc[i] = 1, bits 6-7 = 0
        }
        for i in 4..8 {
            block.scales[i] = 0; // mn[0..4] = 0
        }
        // sc[4..8] and mn[4..8] come from bytes 8-11
        for i in 8..12 {
            block.scales[i] = 0x01; // sc[j] low nibble = 1, mn[j] high nibble = 0
        }

        // Set qs: all nibbles = 3
        for b in block.qs.iter_mut() {
            *b = 0x33; // low nibble = 3, high nibble = 3
        }

        let out = dequantize_q4_k_m_block(&block);
        // With d=1.0, dmin=0.0, sc=1, all nibbles=3:
        // value = 1.0 * 1 * 3 - 0.0 = 3.0
        for (i, &v) in out.iter().enumerate() {
            assert!(
                (v - 3.0).abs() < 1e-3,
                "mismatch at {i}: got {v}, expected 3.0"
            );
        }
    }

    #[test]
    fn test_vec_dot_q4km_matches_dequantize() {
        // Create a block with varied values
        let mut block = BlockQ4KM {
            d: f16::from_f32(0.5).to_bits(),
            dmin: f16::from_f32(0.1).to_bits(),
            scales: [0u8; 12],
            qs: [0u8; 128],
        };
        // Set scales: sc=2, mn=1 for first 4 sub-blocks
        for i in 0..4 {
            block.scales[i] = 2;
        }
        for i in 4..8 {
            block.scales[i] = 1;
        }
        for i in 8..12 {
            block.scales[i] = 0x21; // sc low=1, mn high=2 -> sc[j]=1|(bits<<4), mn[j]=(2)|(bits<<4)
        }

        // Varied quantized values
        for (i, b) in block.qs.iter_mut().enumerate() {
            *b = ((i % 7) as u8) | (((i % 11) as u8) << 4);
        }

        let y: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();

        // Compute expected via dequantize + dot
        let dequantized = dequantize_q4_k_m_block(&block);
        let expected: f32 = dequantized.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        let got = vec_dot_q4_k_m_f32(&block, &y);

        assert!(
            (got - expected).abs() < 1e-2,
            "vec_dot mismatch: got {got}, expected {expected}"
        );
    }

    #[test]
    fn test_dequantize_q6_k_basic() {
        // Create a Q6_K block where all quants reassemble to 0 (offset 32 → value -32+32=0)
        // and scale d=1.0, sub-block scales=1
        let mut block = BlockQ6K {
            ql: [0u8; 128],
            qh: [0u8; 64],
            scales: [1i8; 16],
            d: f16::from_f32(1.0).to_bits(),
        };
        // Set ql and qh so that all 6-bit values = 32 (which becomes 32-32 = 0)
        // 32 in 6 bits = 0b100000 → low 4 bits = 0, high 2 bits = 0b10 = 2
        // ql stores pairs: ql[l] low nibble for q1, ql[l+32] low nibble for q2
        //                  ql[l] high nibble for q3, ql[l+32] high nibble for q4
        // qh[l] bits 0-1 for q1, bits 2-3 for q2, bits 4-5 for q3, bits 6-7 for q4
        // For value 32: low 4 = 0, high 2 = 2
        // So ql = 0x00 (both nibbles = 0), qh = 0b10_10_10_10 = 0xAA
        for b in block.ql.iter_mut() {
            *b = 0x00;
        }
        for b in block.qh.iter_mut() {
            *b = 0xAA; // bits: 10_10_10_10
        }

        let out = dequantize_q6_k_block(&block);
        for (i, &v) in out.iter().enumerate() {
            assert!(v.abs() < 1e-5, "expected ~0.0 at {i}, got {v}");
        }
    }

    #[test]
    fn test_vec_dot_q6_k_matches_dequantize() {
        // Build a Q6_K block with varied values
        let mut block = BlockQ6K {
            ql: [0u8; 128],
            qh: [0u8; 64],
            scales: [0i8; 16],
            d: f16::from_f32(0.5).to_bits(),
        };
        // Set sub-block scales to small values
        for (i, s) in block.scales.iter_mut().enumerate() {
            *s = (i as i8 % 5) + 1;
        }
        // Set varied ql values
        for (i, b) in block.ql.iter_mut().enumerate() {
            *b = ((i % 13) as u8) | (((i % 9) as u8) << 4);
        }
        // Set varied qh values
        for (i, b) in block.qh.iter_mut().enumerate() {
            *b = (i % 256) as u8;
        }

        let y: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();

        let dequantized = dequantize_q6_k_block(&block);
        let expected: f32 = dequantized.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        let got = vec_dot_q6_k_f32(&block, &y);

        assert!(
            (got - expected).abs() < 1e-2,
            "vec_dot Q6_K mismatch: got {got}, expected {expected}"
        );
    }

    #[test]
    fn test_decode_q4km_scales_roundtrip() {
        // Test that known scale values decode correctly
        let mut scales = [0u8; 12];
        // Set sc[0]=5, sc[1]=10, sc[2]=15, sc[3]=20 (6-bit, low bits in bytes 0-3)
        scales[0] = 5;
        scales[1] = 10;
        scales[2] = 15;
        scales[3] = 20;
        // Set mn[0]=1, mn[1]=2, mn[2]=3, mn[3]=4 (6-bit, low bits in bytes 4-7)
        scales[4] = 1;
        scales[5] = 2;
        scales[6] = 3;
        scales[7] = 4;
        // sc[4..8] and mn[4..8]: bytes 8-11, with high bits from bytes 0-3 bits 6-7
        // For simplicity set bytes 8-11 to 0 and don't use high bits
        scales[8] = 0;
        scales[9] = 0;
        scales[10] = 0;
        scales[11] = 0;

        let (sc, mn) = decode_q4km_scales(&scales);
        assert_eq!(sc[0], 5);
        assert_eq!(sc[1], 10);
        assert_eq!(sc[2], 15);
        assert_eq!(sc[3], 20);
        assert_eq!(mn[0], 1);
        assert_eq!(mn[1], 2);
        assert_eq!(mn[2], 3);
        assert_eq!(mn[3], 4);
    }
}
