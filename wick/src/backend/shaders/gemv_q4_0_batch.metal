#include <metal_stdlib>
using namespace metal;

// Batch Q4_0 mat-vec based on llama.cpp's kernel_mul_mv_ext_q4_f32 design.
// 2 simdgroups per TG (64 threads), 8 rows per TG, 4 columns per TG.
// Dispatch: (ceil(m/8), ceil(n/4), 1) threadgroups × 64 threads.

struct BatchParams {
    uint m;
    uint k;
    uint n;
    uint x_stride;
    uint y_stride;
    uint accum;
};

struct block_q4_0 {
    half d;
    uchar qs[16];
};

inline void dequantize_q4_0_t4(device const block_q4_0 * xb, short il, thread float4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 1);
    const float d1 = (il/4) ? (xb->d / 16.h) : xb->d;
    const float d2 = d1 / 256.f;
    const float md = -8.h * xb->d;
    const ushort mask0 = (il/4) ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;

    for (int i = 0; i < 2; i++) {
        reg[2*i + 0] = d1 * (qs[2*(il%4) + i] & mask0) + md;
        reg[2*i + 1] = d2 * (qs[2*(il%4) + i] & mask1) + md;
    }
}

constant constexpr short NSG   = 2;
constant constexpr short NXPSG = 8;
constant constexpr short NYPSG = 4;
constant constexpr short R1PTG = 4;
constant constexpr short CHPT  = 4;
constant constexpr short CHPB  = 8;

kernel void gemv_q4_0_batch(
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
    const uint row_bytes = nb * 18;

    const short tx = tiisg % NXPSG;
    const short ty = tiisg / NXPSG;

    const int i01 = tgpig.x * (NYPSG * NSG) + NYPSG * sgitg + ty;
    const int i11 = tgpig.y * R1PTG;

    device const block_q4_0 * xq = (i01 < (int)m)
        ? (device const block_q4_0 *)(src0 + i01 * row_bytes) + tx / CHPB
        : (device const block_q4_0 *)src0;

    device const float4 * y4[R1PTG];
    for (short ir1 = 0; ir1 < R1PTG; ++ir1) {
        y4[ir1] = (i11 + ir1 < (int)n_cols)
            ? (device const float4 *)(src1 + (i11 + ir1) * x_stride) + tx
            : (device const float4 *)src1;
    }

    float sumf[R1PTG] = { 0.0f, 0.0f, 0.0f, 0.0f };
    short cch = tx % CHPB;

    for (int ich = tx; 4 * ich < (int)k; ich += CHPT * NXPSG) {
        float4 lx[CHPT];

        #pragma unroll(CHPT)
        for (short ch = 0; ch < CHPT; ++ch) {
            dequantize_q4_0_t4(xq, cch, lx[ch]);
            cch += NXPSG;
            if (cch >= CHPB) {
                xq  += cch / CHPB;
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
