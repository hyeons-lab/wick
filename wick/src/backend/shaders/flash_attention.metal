#include <metal_stdlib>
using namespace metal;

// FlashAttention (Dao 2022) for a single query vector (decode path).
// Tiled K/V with online softmax — one pass over KV cache, O(TILE) threadgroup
// memory regardless of seq_len. Removes the MAX_SEQ_LEN cap of the original
// attention kernel and cuts KV memory traffic roughly in half.
//
// One threadgroup per head, 256 threads. Binding layout matches attention.metal
// so the Rust-side dispatch is identical.

constant constexpr uint TILE = 32;

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

kernel void flash_attention(
    const device float* q [[buffer(0)]],
    const device float* k_cache [[buffer(1)]],
    const device float* v_cache [[buffer(2)]],
    device float* out [[buffer(3)]],
    const device Params& params [[buffer(4)]],
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

    uint simd_lane = tid & 31u;
    uint simd_id = tid >> 5u;

    // par co-op threads per output dim (matches existing phase 3 pattern).
    uint par = 256u / head_dim;
    if (par < 1u) par = 1u;
    uint d = tid / par;
    uint s_group = tid % par;

    threadgroup float q_shared[256];   // head_dim ≤ 256
    threadgroup float tile_scores[TILE];
    threadgroup float broadcast[2];    // [new_max, correction] then [tile_sum, _]

    // Load Q once into shared memory.
    if (tid < head_dim) {
        q_shared[tid] = q[q_offset + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float partial_out = 0.0f;   // per-thread running accumulator for out[d]
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    for (uint tile_start = 0u; tile_start < seq_len; tile_start += TILE) {
        uint tile_len = min(TILE, seq_len - tile_start);

        // Step 1: first TILE threads compute scores for their timestep.
        if (tid < tile_len) {
            float sc = 0.0f;
            uint k_base = (tile_start + tid) * kv_dim + kv_h_offset;
            for (uint dd = 0u; dd < head_dim; dd++) {
                sc += q_shared[dd] * k_cache[k_base + dd];
            }
            tile_scores[tid] = sc * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 2: simdgroup 0 reduces tile_max → computes new_max, correction.
        if (simd_id == 0u) {
            float v = (simd_lane < tile_len) ? tile_scores[simd_lane] : -INFINITY;
            float tile_max = simd_max(v);
            if (simd_lane == 0u) {
                float new_max = max(running_max, tile_max);
                broadcast[0] = new_max;
                broadcast[1] = fast::exp(running_max - new_max);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float new_max = broadcast[0];
        float correction = broadcast[1];

        // Step 3: exponentiate tile scores in place.
        if (tid < tile_len) {
            tile_scores[tid] = fast::exp(tile_scores[tid] - new_max);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 4: simdgroup 0 reduces tile_sum.
        if (simd_id == 0u) {
            float v = (simd_lane < tile_len) ? tile_scores[simd_lane] : 0.0f;
            float tile_sum = simd_sum(v);
            if (simd_lane == 0u) {
                broadcast[0] = tile_sum;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tile_sum = broadcast[0];

        running_sum = running_sum * correction + tile_sum;
        running_max = new_max;

        // Step 5: update this thread's slice of the running output.
        if (d < head_dim) {
            partial_out *= correction;
            for (uint b = s_group; b < tile_len; b += par) {
                partial_out += tile_scores[b]
                             * v_cache[(tile_start + b) * kv_dim + kv_h_offset + d];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Epilogue: reduce partial_out across the par co-op threads.
    if (d < head_dim) {
        for (uint offset = par >> 1; offset > 0u; offset >>= 1) {
            partial_out += simd_shuffle_down(partial_out, offset);
        }
        if (s_group == 0u) {
            out[q_offset + d] = partial_out / running_sum;
        }
    }
}
