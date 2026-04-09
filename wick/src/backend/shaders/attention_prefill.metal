#include <metal_stdlib>
using namespace metal;

// Batched causal attention for prefill: all queries in one dispatch.
// Each threadgroup handles one (query_idx, head) pair.
// Dispatch: (n_queries * n_heads) threadgroups × 256 threads.

constant constexpr uint MAX_SEQ_LEN = 4096;
constant constexpr uint MAX_HEAD_DIM = 128;

struct PrefillAttnParams {
    uint n_heads;
    uint n_kv_heads;
    uint head_dim;
    uint kv_dim;
    uint start_pos;
    uint n_queries;
    uint scale_bits;
    uint q_stride;
    uint out_stride;
};

kernel void attention_prefill(
    const device float* q_batch [[buffer(0)]],
    const device half* k_cache [[buffer(1)]],
    const device half* v_cache [[buffer(2)]],
    device float* out_batch [[buffer(3)]],
    constant PrefillAttnParams& params [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_idx [[threadgroup_position_in_grid]]
) {
    uint n_heads = params.n_heads;
    uint n_kv_heads = params.n_kv_heads;
    uint head_dim = params.head_dim;
    uint kv_dim = params.kv_dim;
    uint start_pos = params.start_pos;
    float scale = as_type<float>(params.scale_bits);

    uint query_idx = tg_idx / n_heads;
    uint head = tg_idx % n_heads;
    uint seq_len = start_pos + query_idx + 1;

    if (seq_len > MAX_SEQ_LEN) return;

    uint group_size = n_heads / n_kv_heads;
    uint kv_head = head / group_size;
    uint kv_h_offset = kv_head * head_dim;

    uint q_offset = query_idx * params.q_stride + head * head_dim;
    uint out_offset = query_idx * params.out_stride + head * head_dim;

    threadgroup float scores[MAX_SEQ_LEN];
    threadgroup float q_shared[MAX_HEAD_DIM];
    threadgroup float sg_val[8];
    uint simd_lane = tid & 31u;
    uint simd_id = tid >> 5u;

    if (tid < head_dim) {
        q_shared[tid] = q_batch[q_offset + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 1: Q·K scores with vectorized loads.
    uint hd4 = head_dim / 4u;
    for (uint t = tid; t < seq_len; t += 256u) {
        float acc = 0.0f;
        uint k_base = t * kv_dim + kv_h_offset;
        const device half4* k4 = (device const half4*)(k_cache + k_base);
        for (uint d = 0u; d < hd4; d++) {
            float4 kk = float4(k4[d]);
            float4 qq = float4(q_shared[d * 4u], q_shared[d * 4u + 1u],
                               q_shared[d * 4u + 2u], q_shared[d * 4u + 3u]);
            acc += dot(qq, kk);
        }
        scores[t] = acc * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: softmax.
    float local_max = -INFINITY;
    for (uint t = tid; t < seq_len; t += 256u) {
        local_max = max(local_max, scores[t]);
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

    float partial_sum = 0.0f;
    for (uint t = tid; t < seq_len; t += 256u) {
        float e = exp(scores[t] - max_val);
        scores[t] = e;
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

    for (uint t = tid; t < seq_len; t += 256u) {
        scores[t] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: weighted V sum.
    uint dims_per_lane = head_dim / 32u;
    if (dims_per_lane < 1u) dims_per_lane = 1u;
    if (dims_per_lane > 8u) dims_per_lane = 8u;

    uint chunk = (seq_len + 7u) / 8u;
    uint t_start = simd_id * chunk;
    uint t_end = min(t_start + chunk, seq_len);

    float po[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (dims_per_lane == 2u) {
        for (uint tt = t_start; tt < t_end; tt++) {
            float s = scores[tt];
            uint v_base = tt * kv_dim + kv_h_offset + simd_lane * 2u;
            float2 v2 = float2(*((device const half2*)(v_cache + v_base)));
            po[0] += s * v2.x;
            po[1] += s * v2.y;
        }
    } else if (dims_per_lane == 4u) {
        for (uint tt = t_start; tt < t_end; tt++) {
            float s = scores[tt];
            uint v_base = tt * kv_dim + kv_h_offset + simd_lane * 4u;
            float4 v4 = float4(*((device const half4*)(v_cache + v_base)));
            po[0] += s * v4.x;
            po[1] += s * v4.y;
            po[2] += s * v4.z;
            po[3] += s * v4.w;
        }
    } else {
        for (uint tt = t_start; tt < t_end; tt++) {
            float s = scores[tt];
            for (uint d = 0; d < dims_per_lane; d++) {
                uint v_idx = tt * kv_dim + kv_h_offset + simd_lane * dims_per_lane + d;
                po[d] += s * float(v_cache[v_idx]);
            }
        }
    }

    // Reduce across simdgroups.
    threadgroup float reduce_buf[256 * 8];
    for (uint d = 0; d < dims_per_lane; d++) {
        reduce_buf[simd_id * 32 * dims_per_lane + simd_lane * dims_per_lane + d] = po[d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        for (uint d = 0; d < dims_per_lane; d++) {
            float sum = 0.0f;
            for (uint sg = 0; sg < 8u; sg++) {
                sum += reduce_buf[sg * 32 * dims_per_lane + simd_lane * dims_per_lane + d];
            }
            uint out_dim = simd_lane * dims_per_lane + d;
            if (out_dim < head_dim) {
                out_batch[out_offset + out_dim] = sum;
            }
        }
    }
}
