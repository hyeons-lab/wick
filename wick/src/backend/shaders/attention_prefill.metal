#include <metal_stdlib>
using namespace metal;

// Multi-query tiled flash attention for prefill.
//
// Layout: one simdgroup per query within a threadgroup. NSG=8 matches
// Q_PER_TG=8 so each SG owns a query's softmax + V accumulation
// independently — all reductions inside the per-query block use
// simdgroup collectives (simd_max / simd_sum), with NO threadgroup
// barriers between them. Barriers remain only between the cooperative
// K-load, score, V-load phases (3 per chunk) — vs the previous design's
// ~5 per query × n_q per chunk.
//
// C=32 matches simd_width so each chunk's softmax reduction fits in a
// single simd_max / simd_sum call per SG. Growing C beyond simd_width
// would require per-SG loops with partial sums and defeat the point.
//
// Dispatch: ceil(n_queries/Q_PER_TG) × n_heads TGs × 256 threads.
// Threadgroup memory: dynamically allocated via set_threadgroup_memory_length.

constant constexpr uint Q_PER_TG = 8;   // queries per threadgroup
constant constexpr uint C = 32;         // KV chunk size (must equal NW)
constant constexpr uint NW = 32;        // SIMD width
constant constexpr uint NSG = 8;        // simdgroups per TG (must equal Q_PER_TG)
constant constexpr uint N_THREADS = NSG * NW; // 256

struct PrefillAttnParams {
    uint n_heads;
    uint n_kv_heads;
    uint head_dim;
    uint kv_dim;
    uint start_pos;
    uint n_queries;
    uint scale_bits;
    uint q_stride;   // floats between query vectors
    uint out_stride; // floats between output vectors
};

kernel void attention_prefill(
    const device float* q_batch [[buffer(0)]],
    const device half*  k_cache [[buffer(1)]],
    const device half*  v_cache [[buffer(2)]],
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

    const uint n_q = min(Q_PER_TG, n_queries - q_base);
    const uint max_seq = start_pos + q_base + n_q;

    // Dynamic threadgroup memory layout (all floats):
    //   q_tg:    Q_PER_TG × hd       (query vectors)
    //   kv_tile: C × hd               (K or V tile, reused)
    //   scores:  Q_PER_TG × C         (per-chunk scores)
    //   out_tg:  Q_PER_TG × hd        (running output accumulators)
    //   state:   Q_PER_TG × 2         (running max + sum per query)
    threadgroup float* q_tg    = (threadgroup float*)(shmem);
    threadgroup float* kv_tile = q_tg + Q_PER_TG * hd;
    threadgroup float* scores  = kv_tile + C * hd;
    threadgroup float* out_tg  = scores + Q_PER_TG * C;
    threadgroup float* state   = out_tg + Q_PER_TG * hd;

    const uint simd_lane = tid & 31u;
    const uint simd_id = tid >> 5u;
    const uint hd4 = hd / 4u;

    // Lane owns a contiguous slice of head_dim. Ceil-div so a trailing
    // partial slice (hd not a multiple of NW) is still covered — the
    // `if (d < hd)` guards inside the V loops prevent out-of-bounds
    // reads/writes. Host code asserts `hd <= NW * 8` and `hd % 4 == 0`
    // so `po[8]` and the float4 scoring loop are safe.
    const uint dims_per_lane = (hd + NW - 1u) / NW;

    // Load Q vectors (cooperative, all threads).
    for (uint idx = tid; idx < n_q * hd; idx += N_THREADS) {
        uint q = idx / hd;
        uint d = idx % hd;
        q_tg[q * hd + d] = q_batch[(q_base + q) * params.q_stride + head * hd + d];
    }
    // Init output accumulators + softmax state.
    for (uint idx = tid; idx < n_q * hd; idx += N_THREADS) {
        out_tg[idx] = 0.0f;
    }
    if (tid < Q_PER_TG) {
        state[tid * 2 + 0] = -INFINITY;
        state[tid * 2 + 1] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Iterate KV chunks.
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
                s = -INFINITY;
            } else {
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

        // ---- Overwrite kv_tile with V values for this chunk ----
        for (uint idx = tid; idx < c_len * hd; idx += N_THREADS) {
            uint t = idx / hd;
            uint d = idx % hd;
            kv_tile[t * hd + d] = float(v_cache[(c0 + t) * kv_dim + kv_h_off + d]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- Per-query softmax + V accumulation ----
        // Each SG handles its own query. No inter-SG coordination in here
        // until the next iteration's barriers.
        const uint q = simd_id;
        if (q < n_q) {
            // Softmax max: lane L holds score[q][L] (since C == simd_width).
            float score_lane = (simd_lane < c_len)
                ? scores[q * C + simd_lane]
                : -INFINITY;
            float chunk_max = simd_max(score_lane);

            float prev_max = state[q * 2 + 0];
            float new_max = max(prev_max, chunk_max);
            float rescale = (prev_max > -INFINITY) ? exp(prev_max - new_max) : 0.0f;

            // Exponentiate — store back into the per-SG row of `scores` so
            // the V loop can read timestep tt's value on every lane.
            // `simdgroup_barrier` provides the threadgroup-memory visibility
            // between the write (this lane's slot) and the read (all lanes'
            // slots) that a bare `simd_sum` wouldn't.
            float e = 0.0f;
            if (simd_lane < c_len) {
                e = exp(score_lane - new_max);
                scores[q * C + simd_lane] = e;
            }
            float chunk_sum = simd_sum(e);
            simdgroup_barrier(mem_flags::mem_threadgroup);

            // V accumulation: each lane owns dims_per_lane dims, iterates
            // all c_len timesteps. scores are in threadgroup memory, made
            // visible by the simdgroup_barrier above.
            float po[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            for (uint tt = 0u; tt < c_len; tt++) {
                float s = scores[q * C + tt];
                uint v_base = tt * hd + simd_lane * dims_per_lane;
                #pragma clang loop unroll(full)
                for (uint i = 0u; i < 8u; i++) {
                    if (i < dims_per_lane) {
                        uint d = simd_lane * dims_per_lane + i;
                        if (d < hd) {
                            po[i] += s * kv_tile[v_base + i];
                        }
                    }
                }
            }

            // Rescale running output + write chunk contribution.
            for (uint i = 0u; i < dims_per_lane; i++) {
                uint d = simd_lane * dims_per_lane + i;
                if (d < hd) {
                    out_tg[q * hd + d] = out_tg[q * hd + d] * rescale + po[i];
                }
            }

            // Update softmax state (single lane).
            if (simd_lane == 0u) {
                state[q * 2 + 0] = new_max;
                state[q * 2 + 1] = state[q * 2 + 1] * rescale + chunk_sum;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
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
