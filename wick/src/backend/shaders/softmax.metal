#include <metal_stdlib>
using namespace metal;

// Softmax in-place. Single threadgroup (256 threads).

struct Params { uint n; uint _pad; };

kernel void softmax(
    device float* x [[buffer(0)]],
    const device Params& params [[buffer(1)]],
    uint tid [[thread_position_in_threadgroup]]
) {
    uint n = params.n;
    threadgroup float sg_val[8];
    uint simd_lane = tid & 31u;
    uint simd_id = tid >> 5u;

    // Phase 1: max.
    float local_max = -INFINITY;
    for (uint i = tid; i < n; i += 256u) {
        local_max = max(local_max, x[i]);
    }
    float sg_max = simd_max(local_max);
    if (simd_lane == 0) sg_val[simd_id] = sg_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {
        float v = simd_lane < 8u ? sg_val[simd_lane] : -INFINITY;
        float total = simd_max(v);
        if (simd_lane == 0) sg_val[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float max_val = sg_val[0];

    // Phase 2: exp + sum.
    float partial_sum = 0.0f;
    for (uint i = tid; i < n; i += 256u) {
        float e = fast::exp(x[i] - max_val);
        x[i] = e;
        partial_sum += e;
    }
    float sg_sum = simd_sum(partial_sum);
    if (simd_lane == 0) sg_val[simd_id] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {
        float v = simd_lane < 8u ? sg_val[simd_lane] : 0.0f;
        float total = simd_sum(v);
        if (simd_lane == 0) sg_val[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_sum = 1.0f / sg_val[0];

    // Phase 3: normalize.
    for (uint i = tid; i < n; i += 256u) {
        x[i] *= inv_sum;
    }
}
