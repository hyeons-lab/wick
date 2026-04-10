#include <metal_stdlib>
using namespace metal;

// Batched qk_norm_rope: processes N tokens in one dispatch.
// Dispatch: (n_tokens * (n_heads + n_kv_heads)) TGs × 256 threads.

struct BatchParams {
    uint start_pos;
    uint n_tokens;
    uint n_heads;
    uint n_kv_heads;
    uint head_dim;
    uint eps_bits;
    uint freq_base_bits;
    uint rope_type;
    uint q_stride;  // floats between Q vectors (typically hs)
    uint k_stride;  // floats between K vectors (typically kv_dim)
};

inline void head_rmsnorm(
    device float* buf,
    const device float* w,
    threadgroup float* scratch,
    uint tid,
    uint head_dim,
    float eps
) {
    float partial = 0.0f;
    for (uint i = tid; i < head_dim; i += 256u) {
        float v = buf[i];
        scratch[i] = v;
        partial += v * v;
    }
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
    for (uint i = tid; i < head_dim; i += 256u) {
        buf[i] = scratch[i] * inv_rms * w[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void head_rope(
    device float* buf,
    uint tid,
    uint head_dim,
    uint pos,
    float freq_base,
    uint rope_type
) {
    uint half_dim = head_dim / 2u;
    // theta[d] = pos * freq_base^(-2d/head_dim). Compute with powr for O(1)
    // per thread instead of an O(d) iterative multiplication loop.
    float theta_scale = powr(freq_base, -2.0f / float(head_dim));
    for (uint d = tid; d < half_dim; d += 256u) {
        float theta = float(pos) * powr(theta_scale, float(d));
        float cos_a = cos(theta);
        float sin_a = sin(theta);
        if (rope_type == 0u) {
            float x0 = buf[d];
            float x1 = buf[d + half_dim];
            buf[d] = x0 * cos_a - x1 * sin_a;
            buf[d + half_dim] = x0 * sin_a + x1 * cos_a;
        } else {
            float x0 = buf[2u * d];
            float x1 = buf[2u * d + 1u];
            buf[2u * d] = x0 * cos_a - x1 * sin_a;
            buf[2u * d + 1u] = x0 * sin_a + x1 * cos_a;
        }
    }
}

kernel void qk_norm_rope_batch(
    device float* q_batch [[buffer(0)]],
    device float* k_batch [[buffer(1)]],
    const device float* q_norm_w [[buffer(2)]],
    const device float* k_norm_w [[buffer(3)]],
    constant BatchParams& params [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_idx [[threadgroup_position_in_grid]]
) {
    uint n_heads = params.n_heads;
    uint n_kv_heads = params.n_kv_heads;
    uint head_dim = params.head_dim;
    float eps = as_type<float>(params.eps_bits);
    float freq_base = as_type<float>(params.freq_base_bits);
    uint rope_type = params.rope_type;

    uint heads_per_token = n_heads + n_kv_heads;
    uint token_idx = tg_idx / heads_per_token;
    uint head = tg_idx % heads_per_token;
    uint pos = params.start_pos + token_idx;

    threadgroup float scratch[264];

    if (head < n_heads) {
        device float* q_head = q_batch + token_idx * params.q_stride + head * head_dim;
        head_rmsnorm(q_head, q_norm_w, scratch, tid, head_dim, eps);
        head_rope(q_head, tid, head_dim, pos, freq_base, rope_type);
    } else {
        uint kh = head - n_heads;
        if (kh < n_kv_heads) {
            device float* k_head = k_batch + token_idx * params.k_stride + kh * head_dim;
            head_rmsnorm(k_head, k_norm_w, scratch, tid, head_dim, eps);
            head_rope(k_head, tid, head_dim, pos, freq_base, rope_type);
        }
    }
}
