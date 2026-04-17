#include <metal_stdlib>
using namespace metal;

// Split-K attention: N_SPLITS threadgroups per head, each handling 1/N_SPLITS
// of the sequence. Each split emits a partial (out, max, sum) triple; a second
// kernel merges them via numerically-stable online softmax combine.
//
// Increases GPU occupancy at long contexts: instead of 16 TGs (one per head),
// we launch 16 * N_SPLITS TGs, filling more of the 32 M1 Max cores.

constant constexpr uint MAX_CHUNK = 512;

struct SplitParams {
    uint n_heads;
    uint n_kv_heads;
    uint head_dim;
    uint kv_dim;
    uint seq_len;
    uint scale_bits;
    uint n_splits;
    uint _pad;
};

// Phase A: per-split partial computation.
// Dispatch: (n_heads × n_splits) threadgroups, 256 threads each.
// Writes: partials_out[head, split, d], partials_max[head, split], partials_sum[head, split].
// K/V caches are stored as f16 (see `encode_cast_f32_to_f16_offsets` in
// metal_lfm2.rs and the `// f16 bytes` comment). Binding as `float*`
// would reinterpret two adjacent halves as one f32 and produce garbage.
kernel void attention_split_compute(
    const device float* q [[buffer(0)]],
    const device half*  k_cache [[buffer(1)]],
    const device half*  v_cache [[buffer(2)]],
    device float* partials_out [[buffer(3)]],    // [n_heads × n_splits × head_dim]
    device float* partials_max [[buffer(4)]],    // [n_heads × n_splits]
    device float* partials_sum [[buffer(5)]],    // [n_heads × n_splits]
    constant SplitParams& params [[buffer(6)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]]
) {
    uint n_heads = params.n_heads;
    uint n_kv_heads = params.n_kv_heads;
    uint head_dim = params.head_dim;
    uint kv_dim = params.kv_dim;
    uint seq_len = params.seq_len;
    uint n_splits = params.n_splits;
    float scale = as_type<float>(params.scale_bits);

    uint head = tg_id / n_splits;
    uint split = tg_id % n_splits;

    uint group_size = n_heads / n_kv_heads;
    uint kv_head = head / group_size;
    uint kv_h_offset = kv_head * head_dim;
    uint q_offset = head * head_dim;

    // Determine this split's timestep range.
    uint chunk = (seq_len + n_splits - 1) / n_splits;
    uint t_start = split * chunk;
    uint t_end = min(t_start + chunk, seq_len);
    uint t_len = (t_end > t_start) ? (t_end - t_start) : 0u;

    threadgroup float scores[MAX_CHUNK];
    threadgroup float q_shared[256];
    threadgroup float sg_val[8];
    uint simd_lane = tid & 31u;
    uint simd_id = tid >> 5u;

    // Load Q once.
    if (tid < head_dim) {
        q_shared[tid] = q[q_offset + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write empty partials for empty splits (t_len == 0) and return.
    if (t_len == 0) {
        if (tid == 0) {
            partials_max[head * n_splits + split] = -INFINITY;
            partials_sum[head * n_splits + split] = 0.0f;
        }
        uint par = 256u / head_dim;
        if (par < 1u) par = 1u;
        uint d = tid / par;
        uint s_group = tid % par;
        if (d < head_dim && s_group == 0u) {
            partials_out[(head * n_splits + split) * head_dim + d] = 0.0f;
        }
        return;
    }

    // Phase 1: scores over this chunk.
    for (uint i = tid; i < t_len; i += 256u) {
        uint t = t_start + i;
        float dot = 0.0f;
        uint k_base = t * kv_dim + kv_h_offset;
        for (uint d = 0u; d < head_dim; d++) {
            dot += q_shared[d] * float(k_cache[k_base + d]);
        }
        scores[i] = dot * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: chunk-local softmax.
    float local_max = -INFINITY;
    for (uint i = tid; i < t_len; i += 256u) {
        local_max = max(local_max, scores[i]);
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
    for (uint i = tid; i < t_len; i += 256u) {
        float e = exp(scores[i] - max_val);
        scores[i] = e;
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
    float total_sum = sg_val[0];

    // Phase 3: weighted V sum over the chunk (scores are pre-exp values, NOT normalized).
    uint par = 256u / head_dim;
    if (par < 1u) par = 1u;
    uint d = tid / par;
    uint s_group = tid % par;
    if (d < head_dim) {
        float val = 0.0f;
        for (uint ii = s_group; ii < t_len; ii += par) {
            val += scores[ii] * float(v_cache[(t_start + ii) * kv_dim + kv_h_offset + d]);
        }
        for (uint offset = par >> 1; offset > 0u; offset >>= 1) {
            val += simd_shuffle_down(val, offset);
        }
        if (s_group == 0u) {
            partials_out[(head * n_splits + split) * head_dim + d] = val;
        }
    }
    // Write max/sum (one writer per split).
    if (tid == 0) {
        partials_max[head * n_splits + split] = max_val;
        partials_sum[head * n_splits + split] = total_sum;
    }
}

// Phase B: merge partials across splits for one head.
// Dispatch: n_heads threadgroups, head_dim threads per TG (or 256 if head_dim ≤ 256).
kernel void attention_split_merge(
    const device float* partials_out [[buffer(0)]],  // [n_heads × n_splits × head_dim]
    const device float* partials_max [[buffer(1)]],  // [n_heads × n_splits]
    const device float* partials_sum [[buffer(2)]],  // [n_heads × n_splits]
    device float* out [[buffer(3)]],                 // [n_heads × head_dim]
    constant SplitParams& params [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint head [[threadgroup_position_in_grid]]
) {
    uint head_dim = params.head_dim;
    uint n_splits = params.n_splits;

    // Find combined max across splits.
    float combined_max = -INFINITY;
    for (uint s = 0; s < n_splits; s++) {
        combined_max = max(combined_max, partials_max[head * n_splits + s]);
    }

    // Combine sum with corrections.
    float combined_sum = 0.0f;
    for (uint s = 0; s < n_splits; s++) {
        float m = partials_max[head * n_splits + s];
        if (!isinf(m)) {
            combined_sum += partials_sum[head * n_splits + s]
                          * exp(m - combined_max);
        }
    }
    float inv_sum = 1.0f / combined_sum;

    // Combine output.
    if (tid < head_dim) {
        float acc = 0.0f;
        for (uint s = 0; s < n_splits; s++) {
            float m = partials_max[head * n_splits + s];
            if (!isinf(m)) {
                float corr = exp(m - combined_max);
                acc += partials_out[(head * n_splits + s) * head_dim + tid] * corr;
            }
        }
        out[head * head_dim + tid] = acc * inv_sum;
    }
}
