#include <metal_stdlib>
using namespace metal;

// Per-head RMSnorm. Input: [n_heads × head_dim]. One threadgroup per head.
// Dispatch: n_heads threadgroups × 256 threads.

struct Params { uint head_dim; uint eps_bits; uint _pad0; uint _pad1; };

kernel void per_head_rmsnorm(
    device float* x [[buffer(0)]],
    const device float* w [[buffer(1)]],
    const device Params& params [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint head [[threadgroup_position_in_grid]]
) {
    uint head_dim = params.head_dim;
    float eps = as_type<float>(params.eps_bits);
    uint offset = head * head_dim;

    float partial = 0.0f;
    for (uint i = tid; i < head_dim; i += 256u) {
        float v = x[offset + i];
        partial += v * v;
    }

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

    float inv_rms = 1.0f / sqrt(sum_sq / float(head_dim) + eps);

    for (uint i = tid; i < head_dim; i += 256u) {
        x[offset + i] = x[offset + i] * inv_rms * w[i];
    }
}
