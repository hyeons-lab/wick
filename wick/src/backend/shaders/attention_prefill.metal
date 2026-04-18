#include <metal_stdlib>
using namespace metal;

// Iter 3 — MMA-based prefill attention.
//
// Forked from `attention_prefill.metal` (#20 Iter 2). Both inner
// multiplies — QK^T scoring (Step A) and softmax×V accumulation
// (Step B) — now use `simdgroup_matrix_float8x8` +
// `simdgroup_multiply_accumulate`. The outer online-softmax loop and
// per-query softmax reduction retain Iter 2's one-SG-per-query pattern.
//
// Tile layout:
//   - Score matrix [Q_PER_TG × C] = [8 × 32] = 1 row-tile × 4 col-tiles.
//     SGs 0..3 each own one 8×8 output tile during QK^T; SGs 4..7 idle
//     during scoring.
//   - Per-chunk V-output matrix [Q_PER_TG × hd] = [8 × hd].
//     hd/8 output tiles, distributed round-robin across all 8 SGs.
//     For hd=64: exactly one dim-tile per SG. For hd=128: two per SG.
//
// Constraint: hd must be a multiple of 8 (MMA requires 8x8 tile
// alignment on the head_dim axis). Host asserts this.

constant constexpr uint Q_PER_TG = 8;
constant constexpr uint C = 32;
constant constexpr uint NW = 32;
constant constexpr uint NSG = 8;
constant constexpr uint N_THREADS = NSG * NW; // 256

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

    // TG memory layout (floats). Adds `rescales[Q_PER_TG]` after `state`
    // so Step B's V-MMA can pick up per-query rescale values written by
    // the softmax pass.
    threadgroup float* q_tg    = (threadgroup float*)(shmem);
    threadgroup float* kv_tile = q_tg + Q_PER_TG * hd;
    threadgroup float* scores  = kv_tile + C * hd;
    threadgroup float* out_tg  = scores + Q_PER_TG * C;
    threadgroup float* state   = out_tg + Q_PER_TG * hd;
    threadgroup float* rescales = state + Q_PER_TG * 2;

    const uint simd_lane = tid & 31u;
    const uint simd_id = tid >> 5u;
    const uint hd4 = hd / 4u;
    const uint dims_per_lane = (hd + NW - 1u) / NW;

    // --- Load Q + init output accumulators (cooperative) ---
    for (uint idx = tid; idx < n_q * hd; idx += N_THREADS) {
        uint q = idx / hd;
        uint d = idx % hd;
        q_tg[q * hd + d] = q_batch[(q_base + q) * params.q_stride + head * hd + d];
    }
    for (uint idx = tid; idx < n_q * hd; idx += N_THREADS) {
        out_tg[idx] = 0.0f;
    }
    if (tid < Q_PER_TG) {
        state[tid * 2 + 0] = -INFINITY;
        state[tid * 2 + 1] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Outer chunk loop (online softmax) ---
    for (uint c0 = 0; c0 < max_seq; c0 += C) {
        const uint c_end = min(c0 + C, max_seq);
        const uint c_len = c_end - c0;

        // --- Load K tile into TG memory (cooperative) ---
        //
        // MMA reads the full C×hd tile regardless of c_len, so the tail
        // rows (t >= c_len, only on the last chunk) must be zeroed —
        // otherwise 0 × uninitialized = NaN propagates through MMA.
        // The scalar kernel was safe because its inner loop stopped at c_len.
        for (uint idx = tid; idx < c_len * hd; idx += N_THREADS) {
            uint t = idx / hd;
            uint d = idx % hd;
            kv_tile[t * hd + d] = float(k_cache[(c0 + t) * kv_dim + kv_h_off + d]);
        }
        for (uint idx = tid + c_len * hd; idx < C * hd; idx += N_THREADS) {
            kv_tile[idx] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- MMA QK scoring (Step A) ---
        //
        // Score matrix is 8×32 = 1 row-tile × 4 col-tiles.
        // Simdgroups 0..3 each own one 8×8 output tile (simd_id == t_tile).
        // Inner dim is head_dim; we accumulate via hd/8 MMAs per tile.
        //
        // simd_id 4..7 skip the scoring MMAs — they participate in V
        // accumulation later, and for scoring there's only 4 output tiles.
        if (simd_id < 4u) {
            const uint t_tile = simd_id;          // which 8-wide col of the score matrix
            const uint hd_tiles = hd / 8u;         // must divide cleanly

            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8>(0.0f);
            simdgroup_float8x8 q_mat;
            simdgroup_float8x8 k_mat;

            // Accumulate over head_dim.
            // Q is q_tg[query][d], K is kv_tile[t][d]. Score[q][t] = sum_d Q[q][d] * K[t][d].
            // MMA computes C = A * B where A is [8×8], B is [8×8]. We want S[q][t] = Q[q][d] * K[t][d]^T
            // so A = Q (rows: q, cols: d-slice), B = K^T (rows: d-slice, cols: t).
            // K^T is produced by loading K with transpose=true.
            for (uint d_tile = 0u; d_tile < hd_tiles; d_tile++) {
                // Load Q[0..8, d_tile*8..d_tile*8+8] — stride hd, no transpose.
                // (Only 8 queries exist per TG, and we always have 8 here if n_q==Q_PER_TG.
                //  When n_q < Q_PER_TG, the extra q_tg rows are garbage but their score
                //  outputs are masked by the causal-mask step below and by the per-query
                //  `q < n_q` guard downstream.)
                simdgroup_load(q_mat, q_tg + d_tile * 8u, hd);

                // Load K^T[d_tile*8..+8, t_tile*8..+8] — i.e. K with transpose=true
                // starting at row t_tile*8, col d_tile*8.
                simdgroup_load(k_mat,
                               kv_tile + t_tile * 8u * hd + d_tile * 8u,
                               hd,
                               /*origin*/ ulong2(0, 0),
                               /*transpose*/ true);

                simdgroup_multiply_accumulate(acc, q_mat, k_mat, acc);
            }

            // Store the 8×8 score tile to TG memory. `scores` is Q_PER_TG × C row-major,
            // so we write rows [0..8), cols [t_tile*8 .. t_tile*8+8) at stride C.
            simdgroup_store(acc, scores + t_tile * 8u, C);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Scale + causal mask (per-lane fix-up of the score matrix) ---
        //
        // The MMA produced raw QK. We still need `* scale` and the
        // triangular mask. Parallelize over all n_q × c_len entries like
        // Iter 2 did — cheap relative to MMA.
        for (uint idx = tid; idx < n_q * c_len; idx += N_THREADS) {
            uint q = idx / c_len;
            uint t = idx % c_len;
            uint seq_len_q = start_pos + q_base + q + 1;
            float s = scores[q * C + t];
            if (c0 + t >= seq_len_q) {
                s = -INFINITY;
            } else {
                s = s * scale;
            }
            scores[q * C + t] = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Overwrite kv_tile with V values ---
        // Same tail-zero fix as the K load: MMA reads the full C×hd tile.
        for (uint idx = tid; idx < c_len * hd; idx += N_THREADS) {
            uint t = idx / hd;
            uint d = idx % hd;
            kv_tile[t * hd + d] = float(v_cache[(c0 + t) * kv_dim + kv_h_off + d]);
        }
        for (uint idx = tid + c_len * hd; idx < C * hd; idx += N_THREADS) {
            kv_tile[idx] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Per-query softmax (Iter 2's one-SG-per-query pattern) ---
        //
        // Each SG owns one query. Writes: exp-scored row of `scores`,
        // state[q], rescales[q]. For Step B's MMA below, scores lanes
        // beyond c_len and rows beyond n_q must be zero (MMA reads the
        // full 8×32 tile). Zero-out the tail explicitly.
        {
            const uint q = simd_id;
            if (q < n_q) {
                float score_lane = (simd_lane < c_len)
                    ? scores[q * C + simd_lane]
                    : -INFINITY;
                float chunk_max = simd_max(score_lane);

                float prev_max = state[q * 2 + 0];
                float new_max = max(prev_max, chunk_max);
                float rescale = (prev_max > -INFINITY) ? exp(prev_max - new_max) : 0.0f;

                float e = 0.0f;
                if (simd_lane < c_len) {
                    e = exp(score_lane - new_max);
                    scores[q * C + simd_lane] = e;
                } else {
                    // Zero out inactive lanes so MMA sees clean inputs.
                    scores[q * C + simd_lane] = 0.0f;
                }
                float chunk_sum = simd_sum(e);

                if (simd_lane == 0u) {
                    state[q * 2 + 0] = new_max;
                    state[q * 2 + 1] = state[q * 2 + 1] * rescale + chunk_sum;
                    rescales[q] = rescale;
                }
            } else {
                // Unused query row: zero its score row so MMA doesn't
                // pollute adjacent valid rows via ordinary load. Set
                // rescale to 0 so its out_tg row (garbage) is zeroed
                // in the pre-MMA rescale step.
                scores[q * C + simd_lane] = 0.0f;
                if (simd_lane == 0u) {
                    rescales[q] = 0.0f;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Step B: MMA V accumulation ---
        //
        // Pre-rescale out_tg by per-query `rescales[q]` (cooperative),
        // then for each 8-wide dim tile:
        //   po (loaded from out_tg) += scores × V_tile
        // The `simdgroup_multiply_accumulate(po, A, B, po)` call computes
        // po = A*B + po, so loading `po` with the rescaled out_tg before
        // the MMA yields the correct fused update in a single operation.
        for (uint idx = tid; idx < Q_PER_TG * hd; idx += N_THREADS) {
            uint q = idx / hd;
            out_tg[idx] = out_tg[idx] * rescales[q];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Each SG owns a dim-tile (8 output columns); round-robin across
        // the hd/8 tiles. For hd=64 → 8 tiles → 1 per SG.
        // For hd=128 → 16 tiles → 2 per SG (d_off advances by NSG=8).
        const uint dim_tiles = hd / 8u;
        const uint inner_tiles = C / 8u;  // C=32 → 4 MMA inner iterations
        for (uint d_off = simd_id; d_off < dim_tiles; d_off += NSG) {
            simdgroup_float8x8 po;
            simdgroup_load(po, out_tg + d_off * 8u, hd);

            simdgroup_float8x8 s_mat;
            simdgroup_float8x8 v_mat;
            for (uint t_in = 0u; t_in < inner_tiles; t_in++) {
                // scores is [Q_PER_TG × C] row-major, stride=C.
                simdgroup_load(s_mat, scores + t_in * 8u, C);
                // kv_tile is [C × hd] row-major. Load an 8×8 tile at
                // rows [t_in*8..+8], cols [d_off*8..+8].
                simdgroup_load(v_mat,
                               kv_tile + t_in * 8u * hd + d_off * 8u,
                               hd);
                simdgroup_multiply_accumulate(po, s_mat, v_mat, po);
            }
            simdgroup_store(po, out_tg + d_off * 8u, hd);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // --- Final normalization + write-out ---
    for (uint q = 0; q < n_q; q++) {
        float inv_sum = 1.0f / state[q * 2 + 1];
        for (uint d = tid; d < hd; d += N_THREADS) {
            uint out_idx = (q_base + q) * params.out_stride + head * hd + d;
            out_batch[out_idx] = out_tg[q * hd + d] * inv_sum;
        }
    }
}
