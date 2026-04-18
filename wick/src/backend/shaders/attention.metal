#include <metal_stdlib>
using namespace metal;

// Fused attention per head: Q·K → softmax → weighted V sum.
// One threadgroup per head, 256 threads. Scores are kept in threadgroup
// memory — no round-trips through a device-memory scores buffer.
//
// MAX_SEQ_LEN caps the per-head attention window. If you exceed this,
// raise it (tradeoff: 4 bytes per slot per active threadgroup of shared
// memory, which is plentiful on Apple GPUs).

// Caps per-TG threadgroup memory. 4096 scores = 16 KB; total TG memory
// ≈ 20.6 KB (within Apple Silicon's 32 KB limit). For seq_len > 4096,
// the Rust side auto-switches to flash attention.
constant constexpr uint MAX_SEQ_LEN = 4096;
constant constexpr uint MAX_HEAD_DIM = 128;

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

kernel void attention(
    const device float* q [[buffer(0)]],
    const device half* k_cache [[buffer(1)]],
    const device half* v_cache [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant Params& params [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint head [[threadgroup_position_in_grid]]
) {
    uint n_heads = params.n_heads;
    uint n_kv_heads = params.n_kv_heads;
    uint head_dim = params.head_dim;
    uint kv_dim = params.kv_dim;
    uint seq_len = params.seq_len;
    float scale = as_type<float>(params.scale_bits);

    uint group_size = n_heads / n_kv_heads;
    uint kv_head = head / group_size;
    uint kv_h_offset = kv_head * head_dim;
    uint q_offset = head * head_dim;

    threadgroup float scores[MAX_SEQ_LEN];
    threadgroup float q_shared[MAX_HEAD_DIM];
    threadgroup float sg_val[8];
    uint simd_lane = tid & 31u;
    uint simd_id = tid >> 5u;

    // Defensive early returns. These preconditions are already asserted on
    // the host side (encode_attention / encode_attention_q_offset), so
    // these branches are belt-and-suspenders.
    // head_dim > MAX_HEAD_DIM would overflow the static q_shared array.
    // seq_len > MAX_SEQ_LEN would overflow `scores[MAX_SEQ_LEN]` (classic
    // kernel's cap — dispatch routes to flash above this threshold).
    if (head_dim > MAX_HEAD_DIM || seq_len > MAX_SEQ_LEN) {
        return;
    }

    // Load Q once into threadgroup memory — all 256 threads reuse it in phase 1.
    if (tid < head_dim) {
        q_shared[tid] = q[q_offset + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 1: Q·K scores (kept in threadgroup memory).
    // Vectorized float4 loads for K when head_dim % 4 == 0 — one instruction
    // reads 16 bytes vs four scalar loads. q_shared is packed into float4s lazily.
    uint hd4 = head_dim / 4u;
    for (uint t = tid; t < seq_len; t += 256u) {
        float acc = 0.0f;
        uint k_base = t * kv_dim + kv_h_offset;
        const device half4* k4 = (device const half4*) (k_cache + k_base);
        #pragma clang loop unroll(full)
        for (uint d = 0u; d < hd4; d++) {
            float4 kk = float4(k4[d]);
            float4 qq = float4(q_shared[d * 4u + 0u],
                               q_shared[d * 4u + 1u],
                               q_shared[d * 4u + 2u],
                               q_shared[d * 4u + 3u]);
            acc += dot(qq, kk);
        }
        scores[t] = acc * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: softmax — find max.
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

    // exp + sum.
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

    // Phase 3: weighted V sum. Coalesced across simdgroup lanes.
    // 8 simdgroups each process seq_len/8 timesteps. Within each simdgroup:
    // 32 lanes cooperate to load v_cache[tt, 0..head_dim] coalesced. Each lane
    // holds DIMS_PER_LANE partial_out values for its slice of head_dim.
    // After the loop, reduce partials across simdgroups via threadgroup memory.
    //
    // Works for head_dim ≤ 256 (DIMS_PER_LANE = head_dim/32, capped at 8).
    uint dims_per_lane = head_dim / 32u;
    if (dims_per_lane < 1u) dims_per_lane = 1u;
    if (dims_per_lane > 8u) dims_per_lane = 8u;

    // Each simdgroup owns a timestep range.
    uint chunk = (seq_len + 7u) / 8u;
    uint t_start = simd_id * chunk;
    uint t_end = min(t_start + chunk, seq_len);

    // Per-lane partial accumulators (up to 8 dims).
    float po[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (dims_per_lane == 2u) {
        // head_dim=64: each lane owns 2 contiguous dims → one half2 load per tt.
        for (uint tt = t_start; tt < t_end; tt++) {
            float s = scores[tt];
            uint v_base = tt * kv_dim + kv_h_offset + simd_lane * 2u;
            float2 v2 = float2(*((device const half2*) (v_cache + v_base)));
            po[0] += s * v2.x;
            po[1] += s * v2.y;
        }
    } else if (dims_per_lane == 4u) {
        // head_dim=128: half4 load per tt.
        for (uint tt = t_start; tt < t_end; tt++) {
            float s = scores[tt];
            uint v_base = tt * kv_dim + kv_h_offset + simd_lane * 4u;
            float4 v4 = float4(*((device const half4*) (v_cache + v_base)));
            po[0] += s * v4.x;
            po[1] += s * v4.y;
            po[2] += s * v4.z;
            po[3] += s * v4.w;
        }
    } else {
        for (uint tt = t_start; tt < t_end; tt++) {
            float s = scores[tt];
            uint v_base = tt * kv_dim + kv_h_offset;
            #pragma clang loop unroll(full)
            for (uint i = 0u; i < 8u; i++) {
                if (i < dims_per_lane) {
                    uint d = simd_lane * dims_per_lane + i;
                    if (d < head_dim) {
                        po[i] += s * v_cache[v_base + d];
                    }
                }
            }
        }
    }

    // Reduce partials across simdgroups via threadgroup memory.
    // Layout: partials_tg[sg_id × head_dim + d]
    threadgroup float partials_tg[8 * MAX_HEAD_DIM];
    for (uint i = 0u; i < dims_per_lane; i++) {
        uint d = simd_lane * dims_per_lane + i;
        if (d < head_dim) {
            partials_tg[simd_id * head_dim + d] = po[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // First simdgroup sums the 8 partials per output dim.
    if (simd_id == 0u) {
        for (uint i = 0u; i < dims_per_lane; i++) {
            uint d = simd_lane * dims_per_lane + i;
            if (d < head_dim) {
                float sum = 0.0f;
                #pragma clang loop unroll(full)
                for (uint sg = 0u; sg < 8u; sg++) {
                    sum += partials_tg[sg * head_dim + d];
                }
                out[q_offset + d] = sum;
            }
        }
    }
}
