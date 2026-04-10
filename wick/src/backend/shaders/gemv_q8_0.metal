#include <metal_stdlib>
using namespace metal;

// Q8_0 GEMV matching llama.cpp's kernel_mul_mv_q8_0_f32 pattern.
// NQ=8 elements per thread, 4 threads per Q8_0 block, 2 rows per TG.
// Dispatch: ceil(m/2) threadgroups × 32 threads.

struct Params {
    uint m;
    uint k;
};

constant constexpr uint QK8_0 = 32;
constant constexpr uint BLOCK_BYTES = 34;
constant constexpr short NQ = 8;       // elements per thread
constant constexpr short NR = 2;       // rows per TG
constant constexpr short NW = 32;      // simdgroup width

kernel void gemv_q8_0(
    const device uchar* src0 [[buffer(0)]],
    const device float* src1 [[buffer(1)]],
    device float* dst [[buffer(2)]],
    constant Params& params [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]]
) {
    const uint m = params.m;
    const uint k = params.k;
    const uint nb = k / QK8_0;
    const uint row_bytes = nb * BLOCK_BYTES;

    const short ix = tiisg / (NW/NQ);  // 0..7: which block in the stride
    const short il = tiisg % (NW/NQ);  // 0..3: which 8-element chunk

    for (short r = 0; r < NR; r++) {
        const uint row = tg_id * NR + r;
        if (row >= m) continue;

        const device uchar* row_ptr = src0 + row * row_bytes;
        const device float* y = src1;

        float sumf = 0.0f;

        for (uint ib = ix; ib < nb; ib += NQ) {
            // Read scale from block header (2-byte f16)
            const device uchar* blk = row_ptr + ib * BLOCK_BYTES;
            float d = float(*(const device half*)blk);

            // Read 8 quants from this block, offset by il*NQ
            const device int8_t* qs = (const device int8_t*)(blk + 2) + il * NQ;

            // Read corresponding 8 input values
            const device float* yb = y + ib * QK8_0 + il * NQ;

            float partial = 0.0f;
            for (short i = 0; i < NQ; i++) {
                partial += float(qs[i]) * yb[i];
            }

            sumf += partial * d;
        }

        float total = simd_sum(sumf);
        if (tiisg == 0) {
            dst[row] = total;
        }
    }
}

kernel void gemv_q8_0_accum(
    const device uchar* src0 [[buffer(0)]],
    const device float* src1 [[buffer(1)]],
    device float* dst [[buffer(2)]],
    constant Params& params [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]]
) {
    const uint m = params.m;
    const uint k = params.k;
    const uint nb = k / QK8_0;
    const uint row_bytes = nb * BLOCK_BYTES;

    const short ix = tiisg / (NW/NQ);
    const short il = tiisg % (NW/NQ);

    for (short r = 0; r < NR; r++) {
        const uint row = tg_id * NR + r;
        if (row >= m) continue;

        const device uchar* row_ptr = src0 + row * row_bytes;
        const device float* y = src1;

        float sumf = 0.0f;

        for (uint ib = ix; ib < nb; ib += NQ) {
            const device uchar* blk = row_ptr + ib * BLOCK_BYTES;
            float d = float(*(const device half*)blk);
            const device int8_t* qs = (const device int8_t*)(blk + 2) + il * NQ;
            const device float* yb = y + ib * QK8_0 + il * NQ;

            float partial = 0.0f;
            for (short i = 0; i < NQ; i++) {
                partial += float(qs[i]) * yb[i];
            }

            sumf += partial * d;
        }

        float total = simd_sum(sumf);
        if (tiisg == 0) {
            dst[row] += total;
        }
    }
}
