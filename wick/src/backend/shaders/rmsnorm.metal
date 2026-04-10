#include <metal_stdlib>
using namespace metal;

// RMSnorm in-place: x[i] = x[i] / rms(x) * weight[i]
// Single threadgroup (256 threads). Two-stage simd reduction.
// Dispatch: 1 threadgroup of 256 threads.

struct Params { uint n; uint eps_bits; uint _pad0; uint _pad1; };

kernel void rmsnorm(
    const device float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    const device float* w [[buffer(2)]],
    const device Params& params [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]]
) {
    uint n = params.n;
    float eps = as_type<float>(params.eps_bits);

    // Phase 1: partial sum of squares.
    float partial = 0.0f;
    for (uint i = tid; i < n; i += 256u) {
        float v = src[i];
        partial += v * v;
    }

    // Two-stage simd reduction.
    threadgroup float sg_sums[8];
    uint simd_lane = tid & 31u;
    uint simd_id = tid >> 5u;
    float sg_sum = simd_sum(partial);
    if (simd_lane == 0) sg_sums[simd_id] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {
        float v = simd_lane < 8u ? sg_sums[simd_lane] : 0.0f;
        float total = simd_sum(v);
        if (simd_lane == 0) sg_sums[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum_sq = sg_sums[0];

    float inv_rms = 1.0f / sqrt(sum_sq / float(n) + eps);

    // Phase 2: normalize + scale.
    for (uint i = tid; i < n; i += 256u) {
        dst[i] = src[i] * inv_rms * w[i];
    }
}
