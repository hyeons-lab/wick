#include <metal_stdlib>
using namespace metal;

// F16 GEMV: y[row] = Σ A[row, k] × x[k], A is stored as half.
//
// 2 rows per simdgroup, 2 simdgroups per TG → 4 rows/TG, 64 threads/TG.
// half4 + float4 vector loads (16 bytes per load) — matches llama.cpp's
// mul_mv_f16_f32_4 kernel layout.
//
// Dispatch: ceil(m / 4) threadgroups × 64 threads.

struct Params { uint m; uint k; };

constant constexpr uint NR0 = 2;   // rows per simdgroup
constant constexpr uint NSG = 2;   // simdgroups per TG
constant constexpr uint ROWS_PER_TG = NR0 * NSG;

kernel void gemv_f16(
    const device half* a [[buffer(0)]],
    const device float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant Params& params [[buffer(3)]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]]
) {
    uint m = params.m;
    uint k = params.k;
    uint r0 = (tg_id * NSG + sgitg) * NR0;

    const device half4* a4 = (const device half4*) a;
    const device float4* x4 = (const device float4*) x;
    uint k4 = k / 4;
    uint row_stride4 = k / 4;

    // Per-row pointer bases (in half4 units).
    uint base[NR0];
    #pragma clang loop unroll(full)
    for (uint r = 0; r < NR0; r++) {
        base[r] = (r0 + r) * row_stride4;
    }

    float sumf[NR0] = {0.0f, 0.0f};
    // 32 threads strided through k4 chunks. Each thread does 1 half4 per iter.
    for (uint c = tiisg; c < k4; c += 32u) {
        float4 xv = x4[c];
        #pragma clang loop unroll(full)
        for (uint r = 0; r < NR0; r++) {
            float4 av = float4(a4[base[r] + c]);
            sumf[r] += dot(av, xv);
        }
    }

    #pragma clang loop unroll(full)
    for (uint r = 0; r < NR0; r++) {
        float tot = simd_sum(sumf[r]);
        if (tiisg == 0 && r0 + r < m) {
            y[r0 + r] = tot;
        }
    }
}
