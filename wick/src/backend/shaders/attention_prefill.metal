#include <metal_stdlib>
using namespace metal;

// Multi-query tiled flash attention for prefill.
//
// Processes Q_PER_TG=8 queries per threadgroup with KV tiling (chunk C).
// Each TG loads KV tiles into threadgroup memory once, shared across 8 queries.
// Online softmax with rescaling enables unbounded sequence length.
//
// Dispatch: ceil(n_queries/Q_PER_TG) × n_heads TGs × 128 threads (4 SGs).
// Threadgroup memory: dynamically allocated via set_threadgroup_memory_length.

constant constexpr uint Q_PER_TG = 8;   // queries per threadgroup
constant constexpr uint C = 32;          // KV chunk size
constant constexpr uint NW = 32;         // SIMD width
constant constexpr uint NSG = 4;         // simdgroups per TG
constant constexpr uint N_THREADS = NSG * NW; // 128

struct PrefillAttnParams {
    uint n_heads;
    uint n_kv_heads;
    uint head_dim;
    uint kv_dim;
    uint start_pos;
    uint n_queries;
    uint scale_bits;
    uint q_stride;   // floats between query vectors
    uint out_stride;  // floats between output vectors
};

kernel void attention_prefill(
    const device float* q_batch [[buffer(0)]],
    const device half* k_cache [[buffer(1)]],
    const device half* v_cache [[buffer(2)]],
    device float* out_batch [[buffer(3)]],
    constant PrefillAttnParams& params [[buffer(4)]],
    threadgroup char* shmem [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_idx [[threadgroup_position_in_grid]]
) {
    const uint n_heads = params.n_heads;
    const uint n_kv_heads = params.n_kv_heads;
    const uint hd = params.head_dim;
    const uint kv_dim = params.kv_dim;
    const uint start_pos = params.start_pos;
    const uint n_queries = params.n_queries;
    const float scale = as_type<float>(params.scale_bits);

    const uint head = tg_idx % n_heads;
    const uint q_group = tg_idx / n_heads;
    const uint q_base = q_group * Q_PER_TG;
    const uint group_size = n_heads / n_kv_heads;
    const uint kv_head = head / group_size;
    const uint kv_h_off = kv_head * hd;

    // How many valid queries in this TG (last group may be partial).
    const uint n_q = min(Q_PER_TG, n_queries - q_base);
    // Max seq_len across all queries in this group.
    const uint max_seq = start_pos + q_base + n_q;

    // Dynamic threadgroup memory layout (all floats):
    //   q_tg:    Q_PER_TG × hd       (query vectors)
    //   kv_tile: C × hd               (K or V tile, shared between K/V phases)
    //   scores:  Q_PER_TG × C         (QK scores for current chunk)
    //   out_tg:  Q_PER_TG × hd        (output accumulators)
    //   state:   Q_PER_TG × 2         (running softmax max + sum per query)
    //   sg_val:  8                     (simdgroup reduction scratch)
    threadgroup float* q_tg    = (threadgroup float*)(shmem);
    threadgroup float* kv_tile = q_tg + Q_PER_TG * hd;
    threadgroup float* scores  = kv_tile + C * hd;
    threadgroup float* out_tg  = scores + Q_PER_TG * C;
    threadgroup float* state   = out_tg + Q_PER_TG * hd;
    threadgroup float* sg_val  = state + Q_PER_TG * 2;

    const uint simd_lane = tid & 31u;
    const uint simd_id = tid >> 5u;
    const uint hd4 = hd / 4u;

    // Load Q vectors (cooperative, all threads).
    for (uint idx = tid; idx < n_q * hd; idx += N_THREADS) {
        uint q = idx / hd;
        uint d = idx % hd;
        q_tg[q * hd + d] = q_batch[(q_base + q) * params.q_stride + head * hd + d];
    }
    // Init output accumulators to 0 and softmax state.
    for (uint idx = tid; idx < n_q * hd; idx += N_THREADS) {
        out_tg[idx] = 0.0f;
    }
    if (tid < Q_PER_TG) {
        state[tid * 2 + 0] = -INFINITY; // max
        state[tid * 2 + 1] = 0.0f;       // sum
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Process KV cache in chunks of C.
    for (uint c0 = 0; c0 < max_seq; c0 += C) {
        const uint c_end = min(c0 + C, max_seq);
        const uint c_len = c_end - c0;

        // ---- Load K tile into threadgroup memory ----
        for (uint idx = tid; idx < c_len * hd; idx += N_THREADS) {
            uint t = idx / hd;
            uint d = idx % hd;
            kv_tile[t * hd + d] = float(k_cache[(c0 + t) * kv_dim + kv_h_off + d]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- QK scores: parallelize across (query, kv_pos) pairs ----
        for (uint idx = tid; idx < n_q * c_len; idx += N_THREADS) {
            uint q = idx / c_len;
            uint t = idx % c_len;
            uint seq_len_q = start_pos + q_base + q + 1;

            float s;
            if (c0 + t >= seq_len_q) {
                s = -INFINITY; // causal mask
            } else {
                // Dot product Q[q] · K[t] using float4 vectorization.
                float acc = 0.0f;
                for (uint d4 = 0; d4 < hd4; d4++) {
                    float4 qq = float4(
                        q_tg[q * hd + d4 * 4],
                        q_tg[q * hd + d4 * 4 + 1],
                        q_tg[q * hd + d4 * 4 + 2],
                        q_tg[q * hd + d4 * 4 + 3]);
                    float4 kk = float4(
                        kv_tile[t * hd + d4 * 4],
                        kv_tile[t * hd + d4 * 4 + 1],
                        kv_tile[t * hd + d4 * 4 + 2],
                        kv_tile[t * hd + d4 * 4 + 3]);
                    acc += dot(qq, kk);
                }
                s = acc * scale;
            }
            scores[q * C + t] = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- Per-query online softmax + V accumulation ----
        // Load V tile (reuse kv_tile memory — K is no longer needed).
        for (uint idx = tid; idx < c_len * hd; idx += N_THREADS) {
            uint t = idx / hd;
            uint d = idx % hd;
            kv_tile[t * hd + d] = float(v_cache[(c0 + t) * kv_dim + kv_h_off + d]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Process each query's softmax and V accumulation.
        // Parallelize: 128 threads work on the C positions and hd dimensions.
        for (uint q = 0; q < n_q; q++) {
            // Find chunk max (all threads cooperate).
            float local_max = -INFINITY;
            for (uint t = tid; t < c_len; t += N_THREADS) {
                local_max = max(local_max, scores[q * C + t]);
            }
            float sg_max_v = simd_max(local_max);
            if (simd_lane == 0u) sg_val[simd_id] = sg_max_v;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_id == 0u) {
                float v = simd_lane < NSG ? sg_val[simd_lane] : -INFINITY;
                if (simd_lane == 0u) sg_val[0] = simd_max(v);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float chunk_max = sg_val[0];

            float prev_max = state[q * 2 + 0];
            float new_max = max(prev_max, chunk_max);
            float rescale = (prev_max > -INFINITY) ? exp(prev_max - new_max) : 0.0f;

            // Compute exp(score - new_max) and chunk sum.
            float local_sum = 0.0f;
            for (uint t = tid; t < c_len; t += N_THREADS) {
                float e = exp(scores[q * C + t] - new_max);
                scores[q * C + t] = e;
                local_sum += e;
            }
            float sg_sum_v = simd_sum(local_sum);
            if (simd_lane == 0u) sg_val[simd_id] = sg_sum_v;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_id == 0u) {
                float v = simd_lane < NSG ? sg_val[simd_lane] : 0.0f;
                if (simd_lane == 0u) sg_val[0] = simd_sum(v);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float chunk_sum = sg_val[0];

            // Rescale previous output + accumulate weighted V.
            for (uint d = tid; d < hd; d += N_THREADS) {
                float prev = out_tg[q * hd + d] * rescale;
                float v_sum = 0.0f;
                for (uint t = 0; t < c_len; t++) {
                    v_sum += scores[q * C + t] * kv_tile[t * hd + d];
                }
                out_tg[q * hd + d] = prev + v_sum;
            }

            // Update softmax state.
            if (tid == 0u) {
                state[q * 2 + 0] = new_max;
                state[q * 2 + 1] = state[q * 2 + 1] * rescale + chunk_sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Final: divide by softmax sum and write output.
    for (uint q = 0; q < n_q; q++) {
        float inv_sum = 1.0f / state[q * 2 + 1];
        for (uint d = tid; d < hd; d += N_THREADS) {
            uint out_idx = (q_base + q) * params.out_stride + head * hd + d;
            out_batch[out_idx] = out_tg[q * hd + d] * inv_sum;
        }
    }
}
