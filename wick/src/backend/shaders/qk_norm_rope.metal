#include <metal_stdlib>
using namespace metal;

// Fused: per-head RMSnorm + RoPE for Q and K (in a GQA attention layer).
// Replaces 3 dispatches (per_head_rmsnorm Q, per_head_rmsnorm K, rope) with 1.
//
// Dispatch: max(n_heads, n_kv_heads) threadgroups × 256 threads.
//   TGs 0..n_kv_heads-1 process BOTH a Q head AND the corresponding K head.
//   TGs n_kv_heads..n_heads-1 process only their Q head.
//
// K is accessed via a byte offset into k_cache (typically pos * kv_dim * 4).

struct Params {
    uint pos;
    uint n_heads;
    uint n_kv_heads;
    uint head_dim;
    uint eps_bits;
    uint freq_base_bits;
};

inline void head_rmsnorm(
    device float* buf,          // buf[0..head_dim]
    const device float* w,      // w[0..head_dim]
    threadgroup float* scratch, // at least [head_dim + 8] floats
    uint tid,
    uint head_dim,
    float eps
) {
    // Sum of squares.
    float partial = 0.0f;
    for (uint i = tid; i < head_dim; i += 256u) {
        float v = buf[i];
        scratch[i] = v;  // cache x into threadgroup mem for phase 2
        partial += v * v;
    }
    // Two-stage simd reduction.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint simd_lane = tid & 31u;
    uint simd_id = tid >> 5u;
    threadgroup float* sg_val = scratch + head_dim;
    float sg_sum = simd_sum(partial);
    if (simd_lane == 0u) sg_val[simd_id] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0u) {
        float v = simd_lane < 8u ? sg_val[simd_lane] : 0.0f;
        float total = simd_sum(v);
        if (simd_lane == 0u) sg_val[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = 1.0f / sqrt(sg_val[0] / float(head_dim) + eps);
    // Apply normalization back to buf.
    for (uint i = tid; i < head_dim; i += 256u) {
        buf[i] = scratch[i] * inv_rms * w[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Apply RoPE to buf[0..head_dim] in place.
inline void head_rope(
    device float* buf,
    uint tid,
    uint head_dim,
    uint pos,
    float freq_base
) {
    uint half_dim = head_dim / 2u;
    for (uint d = tid; d < half_dim; d += 256u) {
        float freq = 1.0f / fast::powr(freq_base, float(2u * d) / float(head_dim));
        float angle = float(pos) * freq;
        float cos_a = fast::cos(angle);
        float sin_a = fast::sin(angle);
        float x0 = buf[d];
        float x1 = buf[d + half_dim];
        buf[d] = x0 * cos_a - x1 * sin_a;
        buf[d + half_dim] = x0 * sin_a + x1 * cos_a;
    }
}

kernel void qk_norm_rope(
    device float* q [[buffer(0)]],        // [n_heads × head_dim]
    device float* k_cache [[buffer(1)]],  // [seq_len × kv_dim], offset to current row
    const device float* q_norm_w [[buffer(2)]],
    const device float* k_norm_w [[buffer(3)]],
    const device Params& params [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint head [[threadgroup_position_in_grid]]
) {
    uint n_heads = params.n_heads;
    uint n_kv_heads = params.n_kv_heads;
    uint head_dim = params.head_dim;
    float eps = as_type<float>(params.eps_bits);
    float freq_base = as_type<float>(params.freq_base_bits);
    uint pos = params.pos;

    // Threadgroup scratch: head_dim floats for x-cache + 8 for simd reduction.
    // head_dim ≤ 256 for LFM2/Llama-family models.
    threadgroup float scratch[264];

    // Dispatch: (n_heads + n_kv_heads) TGs. First n_heads handle Q heads;
    // remaining n_kv_heads handle K heads. Gives every TG equal work (no
    // load imbalance from doing 2× work in the first few TGs).
    if (head < n_heads) {
        device float* q_head = q + head * head_dim;
        head_rmsnorm(q_head, q_norm_w, scratch, tid, head_dim, eps);
        head_rope(q_head, tid, head_dim, pos, freq_base);
    } else {
        uint kh = head - n_heads;
        if (kh < n_kv_heads) {
            device float* k_head = k_cache + kh * head_dim;
            head_rmsnorm(k_head, k_norm_w, scratch, tid, head_dim, eps);
            head_rope(k_head, tid, head_dim, pos, freq_base);
        }
    }
}
