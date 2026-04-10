#include <metal_stdlib>
using namespace metal;

// F32 GEMV: y[row] = dot(A[row, :], x).
// One threadgroup per row (2D dispatch: row = tg.x + tg.y * 65535).
// 32 threads per threadgroup (one simdgroup).

struct Params { uint m; uint k; };

kernel void gemv_f32(
    const device float* a [[buffer(0)]],
    const device float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    const device Params& params [[buffer(3)]],
    uint3 tid_v [[thread_position_in_threadgroup]],
    uint3 tg_id [[threadgroup_position_in_grid]]
) {
    uint tid = tid_v.x;
    uint row = tg_id.x + tg_id.y * 65535u;
    uint m = params.m;
    uint k = params.k;
    if (row >= m) return;

    uint row_offset = row * k;
    float partial = 0.0f;
    for (uint col = tid; col < k; col += 32u) {
        partial += a[row_offset + col] * x[col];
    }
    float total = simd_sum(partial);
    if (tid == 0u) y[row] = total;
}
