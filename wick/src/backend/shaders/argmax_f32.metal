#include <metal_stdlib>
using namespace metal;

// argmax over f32 logits. 1 TG × 256 threads.
// Output: writes winning index as u32 to out[0].
//
// NaN handling: `>` returns false for NaN, so NaN values are naturally
// ignored (never selected), matching the CPU total_cmp(NaN) → -inf path.

constant constexpr uint TG_SIZE = 256;

struct Params { uint n; };

kernel void argmax_f32(
    const device float* logits [[buffer(0)]],
    device uint* out [[buffer(1)]],
    const device Params& params [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    uint n = params.n;
    float best_v = -INFINITY;
    uint best_i = 0u;
    for (uint i = tid; i < n; i += TG_SIZE) {
        float v = logits[i];
        if (v > best_v) { best_v = v; best_i = i; }
    }

    // Simdgroup reduction: shuffle partners to find lane with max value.
    for (uint off = 16u; off > 0u; off >>= 1) {
        float ov = simd_shuffle_down(best_v, off);
        uint  oi = simd_shuffle_down(best_i, off);
        if (ov > best_v) { best_v = ov; best_i = oi; }
    }

    // Lane 0 of each simdgroup publishes its winner to TG memory.
    threadgroup float sg_v[8];
    threadgroup uint  sg_i[8];
    if (simd_lane == 0u) { sg_v[simd_id] = best_v; sg_i[simd_id] = best_i; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // First simdgroup reduces the 8 partials.
    if (simd_id == 0u) {
        float v = simd_lane < 8u ? sg_v[simd_lane] : -INFINITY;
        uint  i = simd_lane < 8u ? sg_i[simd_lane] : 0u;
        for (uint off = 4u; off > 0u; off >>= 1) {
            float ov = simd_shuffle_down(v, off);
            uint  oi = simd_shuffle_down(i, off);
            if (ov > v) { v = ov; i = oi; }
        }
        if (simd_lane == 0u) out[0] = i;
    }
}
