#include <metal_stdlib>
using namespace metal;

// Batch Q8_0 mat-vec: same structure as gemv_q4_0_batch but with Q8_0 dequant.
// 2 simdgroups (64 threads), 8 rows/TG, 4 columns/TG.
// Dispatch: (ceil(m/8), ceil(n/4), 1) threadgroups × 64 threads.

struct BatchParams {
    uint m;
    uint k;
    uint n;
    uint x_stride;
    uint y_stride;
    uint accum;
};

constant constexpr uint Q8_BLOCK_BYTES = 34;
constant constexpr short NSG   = 2;
constant constexpr short NXPSG = 8;
constant constexpr short NYPSG = 4;
constant constexpr short R1PTG = 4;
constant constexpr short CHPT  = 4;
constant constexpr short CHPB  = 8;   // 32 elements / 4 per float4 chunk

kernel void gemv_q8_0_batch(
    const device uchar* src0 [[buffer(0)]],
    const device float* src1 [[buffer(1)]],
    device float* dst [[buffer(2)]],
    constant BatchParams& params [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    const uint m = params.m;
    const uint k = params.k;
    const uint n_cols = params.n;
    const uint x_stride = params.x_stride;
    const uint y_stride = params.y_stride;
    const bool do_accum = params.accum != 0;
    const uint nb = k / 32;
    const uint row_bytes = nb * Q8_BLOCK_BYTES;

    const short tx = tiisg % NXPSG;
    const short ty = tiisg / NXPSG;

    const int i01 = tgpig.x * (NYPSG * NSG) + NYPSG * sgitg + ty;
    const int i11 = tgpig.y * R1PTG;

    // Row pointer and initial block/chunk position.
    const device uchar * row_ptr = (i01 < (int)m) ? src0 + i01 * row_bytes : src0;
    uint blk_idx = tx / CHPB;
    short cch = tx % CHPB;

    // Input column pointers.
    device const float4 * y4[R1PTG];
    for (short ir1 = 0; ir1 < R1PTG; ++ir1) {
        y4[ir1] = (i11 + ir1 < (int)n_cols)
            ? (device const float4 *)(src1 + (i11 + ir1) * x_stride) + tx
            : (device const float4 *)src1;
    }

    float sumf[R1PTG] = { 0.0f, 0.0f, 0.0f, 0.0f };

    for (int ich = tx; 4 * ich < (int)k; ich += CHPT * NXPSG) {
        float4 lx[CHPT];

        #pragma unroll(CHPT)
        for (short ch = 0; ch < CHPT; ++ch) {
            const device uchar * blk = row_ptr + blk_idx * Q8_BLOCK_BYTES;
            float d = float(*(const device half*)blk);
            const device int8_t * qs = (const device int8_t*)(blk + 2);
            uint qoff = cch * 4;
            lx[ch] = float4(
                float(qs[qoff]) * d,
                float(qs[qoff + 1]) * d,
                float(qs[qoff + 2]) * d,
                float(qs[qoff + 3]) * d
            );

            cch += NXPSG;
            if (cch >= CHPB) {
                blk_idx += cch / CHPB;
                cch %= CHPB;
            }
        }

        #pragma unroll(CHPT)
        for (short ch = 0; ch < CHPT; ++ch) {
            #pragma unroll(R1PTG)
            for (short ir1 = 0; ir1 < R1PTG; ++ir1) {
                sumf[ir1] += dot(lx[ch], y4[ir1][ch * NXPSG]);
            }
        }

        #pragma unroll(R1PTG)
        for (short ir1 = 0; ir1 < R1PTG; ++ir1) {
            y4[ir1] += CHPT * NXPSG;
        }
    }

    #pragma unroll(R1PTG)
    for (short ir1 = 0; ir1 < R1PTG; ++ir1) {
        sumf[ir1] += simd_shuffle_down(sumf[ir1], 4);
        sumf[ir1] += simd_shuffle_down(sumf[ir1], 2);
        sumf[ir1] += simd_shuffle_down(sumf[ir1], 1);
    }

    if (tx == 0 && i01 < (int)m) {
        for (short ir1 = 0; ir1 < R1PTG; ++ir1) {
            if (i11 + ir1 < (int)n_cols) {
                uint idx = (i11 + ir1) * y_stride + i01;
                if (do_accum) {
                    dst[idx] += sumf[ir1];
                } else {
                    dst[idx] = sumf[ir1];
                }
            }
        }
    }
}
