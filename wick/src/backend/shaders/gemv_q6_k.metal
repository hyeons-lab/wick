#include <metal_stdlib>
using namespace metal;

// Q6_K GEMV: y[row] = Σ dequant(W_q6k[row, i]) × x[i].
// Ported from llama.cpp's `kernel_mul_mv_q6_K_f32_impl` (N_R0_Q6_K=2).
//
// Q6_K super-block: 256 elements, 210 bytes total:
//   ql[128]     — lower 4 bits of 256 values
//   qh[64]      — upper 2 bits of 256 values
//   scales[16]  — signed int8 per 16-element sub-block
//   d           — f16 super-block scale
//
// Dequantized value: d * scales[sub] * (q_6bit - 32).
//
// Kernel layout: NR=2 rows per simdgroup, NSG=2 simdgroups per TG → 4 rows/TG,
// 64 threads/TG. Dispatch: ceil(m/4) threadgroups.

constant constexpr uint QK_K = 256;
constant constexpr uint Q6K_BYTES = 210;
constant constexpr uint NR = 2;   // rows per simdgroup
constant constexpr uint NSG = 2;  // simdgroups per TG

struct Params { uint m; uint k; };

kernel void gemv_q6_k(
    const device uchar* a [[buffer(0)]],    // raw Q6_K bytes
    const device float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant Params& params [[buffer(3)]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]]
) {
    uint m = params.m;
    uint k = params.k;
    uint nb = k / QK_K;
    uint row_bytes = nb * Q6K_BYTES;
    uint first_row = (tg_id * NSG + sgitg) * NR;

    // Per-row base byte pointers.
    device const uchar* row_base[NR];
    #pragma clang loop unroll(full)
    for (uint r = 0; r < NR; r++) {
        row_base[r] = a + (first_row + r) * row_bytes;
    }

    // Thread-layout constants (llama.cpp's mul_mv_q6_K_f32_impl).
    const uint tid = tiisg / 2;        // 0..15
    const uint ix  = tiisg & 1u;       // 0 or 1
    const uint ip  = tid >> 3;         // 0 or 1
    const uint il  = tid & 7u;         // 0..7
    const uint l0  = 4u * il;
    const uint is  = 8u * ip + l0 / 16u;

    const uint y_offset   = 128u * ip + l0;
    const uint q_offset_l = 64u * ip + l0;
    const uint q_offset_h = 32u * ip + l0;

    const uchar kmask1 = 0x03;
    const uchar kmask2 = 0x0C;
    const uchar kmask3 = 0x30;
    const uchar kmask4 = 0xC0;

    float sumf[NR] = {0.0f, 0.0f};

    // Iterate super-blocks; each pair of threads (ix=0,1) handles consecutive blocks.
    for (uint b = ix; b < nb; b += 2u) {
        device const float* yb = x + b * QK_K + y_offset;

        // Cache 16 y values (scattered across the super-block).
        float yl[16];
        #pragma clang loop unroll(full)
        for (uint l = 0u; l < 4u; l++) {
            yl[4u*l + 0u] = yb[l +  0];
            yl[4u*l + 1u] = yb[l + 32];
            yl[4u*l + 2u] = yb[l + 64];
            yl[4u*l + 3u] = yb[l + 96];
        }

        // Apply to each row with its own weight pointers.
        #pragma clang loop unroll(full)
        for (uint row = 0u; row < NR; row++) {
            device const uchar* blk_base = row_base[row] + b * Q6K_BYTES;
            device const uchar* q1 = blk_base + q_offset_l;
            device const uchar* q2 = q1 + 32;
            device const uchar* qh = blk_base + 128u + q_offset_h;
            device const char*  sc = (device const char*)(blk_base + 128u + 64u) + is;
            device const half*  dh = (device const half*)(blk_base + 128u + 64u + 16u);

            float4 sums = float4(0.0f);
            #pragma clang loop unroll(full)
            for (uint l = 0u; l < 4u; l++) {
                int q6_1 = int((q1[l]        & 0x0Fu) | ((qh[l] & kmask1) << 4u)) - 32;
                int q6_2 = int((q2[l]        & 0x0Fu) | ((qh[l] & kmask2) << 2u)) - 32;
                int q6_3 = int((q1[l] >> 4u)          | ((qh[l] & kmask3)      )) - 32;
                int q6_4 = int((q2[l] >> 4u)          | ((qh[l] & kmask4) >> 2u)) - 32;
                sums[0] += yl[4u*l + 0u] * float(q6_1);
                sums[1] += yl[4u*l + 1u] * float(q6_2);
                sums[2] += yl[4u*l + 2u] * float(q6_3);
                sums[3] += yl[4u*l + 3u] * float(q6_4);
            }

            float dblk = float(*dh);
            sumf[row] += dblk * (sums[0] * float(sc[0])
                               + sums[1] * float(sc[2])
                               + sums[2] * float(sc[4])
                               + sums[3] * float(sc[6]));
        }
    }

    // Reduce each row across the simdgroup.
    #pragma clang loop unroll(full)
    for (uint row = 0u; row < NR; row++) {
        float total = simd_sum(sumf[row]);
        if (tiisg == 0u && first_row + row < m) {
            y[first_row + row] = total;
        }
    }
}
