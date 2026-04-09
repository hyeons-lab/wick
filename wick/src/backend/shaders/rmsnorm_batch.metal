#include <metal_stdlib>
using namespace metal;

// Batch RMSnorm: process N independent vectors in a single dispatch.
// Each threadgroup handles one vector (same algorithm as rmsnorm.metal).
// Dispatch: N threadgroups of 256 threads.

struct Params {
    uint n;           // vector length (hidden_size)
    uint eps_bits;    // f32 epsilon as raw bits
    uint src_stride;  // stride between src vectors (floats)
    uint dst_stride;  // stride between dst vectors (floats)
};

kernel void rmsnorm_batch(
    const device float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    const device float* w [[buffer(2)]],
    constant Params& params [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]]
) {
    uint n = params.n;
    float eps = as_type<float>(params.eps_bits);
    uint src_off = tg_id * params.src_stride;
    uint dst_off = tg_id * params.dst_stride;

    float partial = 0.0f;
    for (uint i = tid; i < n; i += 256u) {
        float v = src[src_off + i];
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

    float inv_rms = 1.0f / sqrt(sum_sq / float(n) + eps);

    for (uint i = tid; i < n; i += 256u) {
        dst[dst_off + i] = src[src_off + i] * inv_rms * w[i];
    }
}

// Fused add + rmsnorm: src[i] += residual[i], then rmsnorm(src) → dst.
// Replaces a separate add_inplace + rmsnorm_batch with one kernel.
// The residual uses the same stride as src (both are batch_buf at stride hs).
kernel void add_rmsnorm_batch(
    device float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    const device float* w [[buffer(2)]],
    constant Params& params [[buffer(3)]],
    const device float* residual [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]]
) {
    uint n = params.n;
    float eps = as_type<float>(params.eps_bits);
    uint src_off = tg_id * params.src_stride;
    uint dst_off = tg_id * params.dst_stride;
    uint res_off = tg_id * params.src_stride;

    // Phase 1: add residual in-place AND compute sum of squares.
    float partial = 0.0f;
    for (uint i = tid; i < n; i += 256u) {
        float v = src[src_off + i] + residual[res_off + i];
        src[src_off + i] = v;  // write back the sum
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

    float inv_rms = 1.0f / sqrt(sum_sq / float(n) + eps);

    for (uint i = tid; i < n; i += 256u) {
        dst[dst_off + i] = src[src_off + i] * inv_rms * w[i];
    }
}
