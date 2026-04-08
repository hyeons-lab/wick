#include <metal_stdlib>
using namespace metal;

// GQA-batched attention: one threadgroup per KV head, handling ALL Q heads in
// its group together. Loads K/V once and reuses across `group_size` Q heads,
// cutting K/V bandwidth by `group_size` (=4 for LFM2).
//
// Assumes group_size ∈ {1, 2, 4} (checked at call site). 256 threads per TG.
// Same binding layout as attention.metal so encode_attention is unchanged.

// scores buffer is MAX_GROUP × MAX_SEQ_LEN floats (4KB per group-slot at seq_len=1024).
// Keep under Apple's typical 32KB threadgroup memory limit.
constant constexpr uint MAX_SEQ_LEN = 1024;
constant constexpr uint MAX_GROUP = 4;

struct Params {
    uint n_heads;
    uint n_kv_heads;
    uint head_dim;
    uint kv_dim;
    uint seq_len;
    uint scale_bits;
    uint _pad0;
    uint _pad1;
};

kernel void attention_gqa(
    const device float* q [[buffer(0)]],
    const device float* k_cache [[buffer(1)]],
    const device float* v_cache [[buffer(2)]],
    device float* out [[buffer(3)]],
    const device Params& params [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint kv_head [[threadgroup_position_in_grid]]
) {
    uint n_heads = params.n_heads;
    uint n_kv_heads = params.n_kv_heads;
    uint head_dim = params.head_dim;
    uint kv_dim = params.kv_dim;
    uint seq_len = params.seq_len;
    float scale = as_type<float>(params.scale_bits);

    uint group_size = n_heads / n_kv_heads;  // e.g. 4
    uint kv_h_offset = kv_head * head_dim;
    uint q_head_base = kv_head * group_size;  // first Q head in this group

    // Per-group scores: [group_size × seq_len] in threadgroup memory.
    // group_size ≤ MAX_GROUP, seq_len ≤ MAX_SEQ_LEN.
    threadgroup float scores[MAX_GROUP * MAX_SEQ_LEN];
    threadgroup float q_shared[MAX_GROUP * 256];  // group_size × head_dim
    threadgroup float sg_val[8];
    uint simd_lane = tid & 31u;
    uint simd_id = tid >> 5u;

    // Load all Q vectors in this group into shared memory (group_size × head_dim).
    uint q_total = group_size * head_dim;
    for (uint i = tid; i < q_total; i += 256u) {
        uint gh = i / head_dim;
        uint d = i % head_dim;
        q_shared[gh * head_dim + d] = q[(q_head_base + gh) * head_dim + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 1: compute scores for ALL group members at each timestep.
    // Each thread handles one (t) and computes group_size scores.
    // K[t] is loaded ONCE per timestep across the group.
    for (uint t = tid; t < seq_len; t += 256u) {
        uint k_base = t * kv_dim + kv_h_offset;
        // Compute group_size dot products using shared K via registers.
        float dots[MAX_GROUP];
        for (uint g = 0; g < group_size; g++) dots[g] = 0.0f;
        for (uint d = 0u; d < head_dim; d++) {
            float k_val = k_cache[k_base + d];  // loaded once
            for (uint g = 0; g < group_size; g++) {
                dots[g] += q_shared[g * head_dim + d] * k_val;
            }
        }
        for (uint g = 0; g < group_size; g++) {
            scores[g * seq_len + t] = dots[g] * scale;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: softmax per group member. Process one head at a time.
    for (uint g = 0; g < group_size; g++) {
        uint s_off = g * seq_len;

        // Find max.
        float local_max = -INFINITY;
        for (uint t = tid; t < seq_len; t += 256u) {
            local_max = max(local_max, scores[s_off + t]);
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

        // exp + sum.
        float partial_sum = 0.0f;
        for (uint t = tid; t < seq_len; t += 256u) {
            float e = exp(scores[s_off + t] - max_val);
            scores[s_off + t] = e;
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
            scores[s_off + t] *= inv_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Phase 3: weighted V sum. For each (g, d), accumulate partials across
    // cooperating threads. `par = 256/head_dim` threads per dim (same as attention.metal).
    uint par = 256u / head_dim;
    if (par < 1u) par = 1u;
    uint d = tid / par;
    uint s_group = tid % par;
    if (d < head_dim) {
        // Compute one (g, d) partial at a time — reads V ONCE per timestep across group.
        for (uint g = 0; g < group_size; g++) {
            float val = 0.0f;
            uint s_off = g * seq_len;
            for (uint tt = s_group; tt < seq_len; tt += par) {
                val += scores[s_off + tt] * v_cache[tt * kv_dim + kv_h_offset + d];
            }
            for (uint offset = par >> 1; offset > 0u; offset >>= 1) {
                val += simd_shuffle_down(val, offset);
            }
            if (s_group == 0u) {
                out[(q_head_base + g) * head_dim + d] = val;
            }
        }
    }
}
