#include <metal_stdlib>
using namespace metal;

// Iter 4 — MMA prefill attention with C=64 + fp16 Q/K/V.
//
// Builds on Iter 3 (#22). Two coordinated changes vs Iter 3:
//
// 1. `C` bumped 32 → 64. Score matrix becomes [Q_PER_TG × C] = [8 × 64]
//    = 8 tiles of 8×8, one per simdgroup — all 8 SGs are busy during
//    QK^T scoring (Iter 3 had only 4 SGs active).
//
// 2. Q, K, V staged in threadgroup memory as `half` instead of `float`
//    and loaded into `simdgroup_half8x8` matrices. Scores matrix and
//    output accumulator remain fp32 (standard flash-attention precision
//    shape). This halves TG memory for q_tg and kv_tile, keeping total
//    shmem at ~13.1 KB for hd=64 (unchanged from Iter 3 despite C
//    doubling) and ~24.1 KB for hd=128.
//
// Empirical results vs Iter 3 at p=4096 on M1 Max (5-run median):
//   450M-Q4_0 (hd=64):  prefill 5663 → 6184 tok/s (+9.2%)
//                       attn_kernel 338k → 281k µs (−17%)
//   1.6B-Q4_0 (hd=128): attn_kernel 640k → 549k µs (−14%)
// The end-to-end gain on 1.6B is within measurement noise because
// non-attention phases dominate its total wall time.
//
// Mixed-precision MMA overloads in use (confirmed supported on M1+):
//   - QK^T:  simdgroup_multiply_accumulate(float8x8, half8x8, half8x8, float8x8)
//   - AV:    simdgroup_multiply_accumulate(float8x8, float8x8, half8x8, float8x8)
//
// Softmax reduces over 64 cells per query but the simdgroup is only 32
// lanes wide, so each lane owns two cells (lane l handles cells l and
// l+32) and we take `max`/`simd_max` / `simd_sum` on the lane-local pair
// before the cross-lane reduction.
//
// Constraint (unchanged): head_dim % 8 == 0, head_dim <= 256.

constant constexpr uint Q_PER_TG = 8;
constant constexpr uint C = 64;
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

    // TG memory layout:
    //   q_tg    : half  [Q_PER_TG × hd]
    //   kv_tile : half  [C × hd]            (K first, overwritten by V)
    //   scores  : float [Q_PER_TG × C]
    //   out_tg  : float [Q_PER_TG × hd]     (running softmax-weighted V sum)
    //   state   : float [Q_PER_TG × 2]      (per-query max, sum)
    //   rescales: float [Q_PER_TG]
    // The `half*` → `float*` pointer cast is safe because
    // `(Q_PER_TG + C) * hd * 2` bytes is always a multiple of 4
    // (Q_PER_TG + C = 72, times hd ≥ 8 = minimum 1152 bytes, and hd is
    // guaranteed divisible by 8 by the host-side assertion).
    threadgroup half*  q_tg     = (threadgroup half*)(shmem);
    threadgroup half*  kv_tile  = q_tg + Q_PER_TG * hd;
    threadgroup float* scores   = (threadgroup float*)(kv_tile + C * hd);
    threadgroup float* out_tg   = scores + Q_PER_TG * C;
    threadgroup float* state    = out_tg + Q_PER_TG * hd;
    threadgroup float* rescales = state + Q_PER_TG * 2;

    const uint simd_lane = tid & 31u;
    const uint simd_id = tid >> 5u;

    // --- Load Q + init output accumulators (cooperative) ---
    //
    // out_tg is zeroed for the full Q_PER_TG × hd (not just n_q × hd) —
    // Step B's pre-rescale step reads all Q_PER_TG rows and threadgroup
    // memory is not guaranteed zero on entry. Even though
    // rescales[q >= n_q] = 0 (set in the softmax else branch below) would
    // logically zero those rows, NaN × 0 = NaN leaves unused rows as NaN.
    // Per-row-independent MMA semantics mean NaN in unused rows doesn't
    // leak into valid rows, but we avoid the hazard entirely by
    // initializing the full tile.
    //
    // q_tg for q >= n_q remains uninitialized; any garbage produced by
    // the scoring MMA on those rows is scrubbed by the softmax else
    // branch (which writes 0 to every lane of rows q >= n_q) before the
    // V MMA reads `scores`.
    for (uint idx = tid; idx < n_q * hd; idx += N_THREADS) {
        uint q = idx / hd;
        uint d = idx % hd;
        q_tg[q * hd + d] = half(q_batch[(q_base + q) * params.q_stride + head * hd + d]);
    }
    for (uint idx = tid; idx < Q_PER_TG * hd; idx += N_THREADS) {
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

        // --- Load K tile into TG memory (cooperative, half-precision) ---
        //
        // MMA reads the full C×hd tile regardless of c_len, so tail rows
        // (t >= c_len on the last chunk) must be zeroed to avoid
        // 0 × uninitialized = NaN propagation.
        for (uint idx = tid; idx < c_len * hd; idx += N_THREADS) {
            uint t = idx / hd;
            uint d = idx % hd;
            kv_tile[t * hd + d] = k_cache[(c0 + t) * kv_dim + kv_h_off + d];
        }
        for (uint idx = tid + c_len * hd; idx < C * hd; idx += N_THREADS) {
            kv_tile[idx] = half(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- MMA QK scoring (all 8 SGs) ---
        //
        // Score matrix [Q_PER_TG × C] = [8 × 64] = 1 row-tile × 8 col-tiles.
        // SG `simd_id` owns col-tile `simd_id`.
        {
            const uint t_tile = simd_id;
            const uint hd_tiles = hd / 8u;

            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8>(0.0f);
            simdgroup_half8x8  q_mat;
            simdgroup_half8x8  k_mat;

            for (uint d_tile = 0u; d_tile < hd_tiles; d_tile++) {
                // Q[0..8, d_tile*8..+8], stride = hd, no transpose.
                simdgroup_load(q_mat, q_tg + d_tile * 8u, hd);

                // K with transpose=true loads K^T[d_tile*8..+8, t_tile*8..+8].
                // `origin` is `ulong2` per MSL's simdgroup_load signature on
                // this toolchain (not uint2 as a review tool once suggested
                // — using uint2 fails compilation).
                simdgroup_load(k_mat,
                               kv_tile + t_tile * 8u * hd + d_tile * 8u,
                               hd,
                               /*origin*/ ulong2(0, 0),
                               /*transpose*/ true);

                simdgroup_multiply_accumulate(acc, q_mat, k_mat, acc);
            }

            simdgroup_store(acc, scores + t_tile * 8u, C);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Scale + causal mask (per-lane fix-up of the score matrix) ---
        //
        // The MMA produced raw QK. Apply `* scale` and the triangular mask
        // in one cooperative pass over n_q × c_len entries.
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

        // --- Overwrite kv_tile with V values (half-precision) ---
        for (uint idx = tid; idx < c_len * hd; idx += N_THREADS) {
            uint t = idx / hd;
            uint d = idx % hd;
            kv_tile[t * hd + d] = v_cache[(c0 + t) * kv_dim + kv_h_off + d];
        }
        for (uint idx = tid + c_len * hd; idx < C * hd; idx += N_THREADS) {
            kv_tile[idx] = half(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Per-query softmax (one SG per query, two cells per lane) ---
        //
        // Simdgroup has 32 lanes but C=64 cells per query. Lane `l` owns
        // cells `l` and `l + 32`. Lane-local max/sum folds both cells
        // before the cross-lane simd_max / simd_sum.
        {
            const uint q = simd_id;
            if (q < n_q) {
                const uint idx0 = simd_lane;
                const uint idx1 = simd_lane + NW;

                float s0 = (idx0 < c_len) ? scores[q * C + idx0] : -INFINITY;
                float s1 = (idx1 < c_len) ? scores[q * C + idx1] : -INFINITY;

                float chunk_max = simd_max(max(s0, s1));

                float prev_max = state[q * 2 + 0];
                float new_max = max(prev_max, chunk_max);
                float rescale = (prev_max > -INFINITY) ? exp(prev_max - new_max) : 0.0f;

                float e0 = 0.0f;
                float e1 = 0.0f;
                if (idx0 < c_len) {
                    e0 = exp(s0 - new_max);
                    scores[q * C + idx0] = e0;
                } else {
                    scores[q * C + idx0] = 0.0f;
                }
                if (idx1 < c_len) {
                    e1 = exp(s1 - new_max);
                    scores[q * C + idx1] = e1;
                } else {
                    scores[q * C + idx1] = 0.0f;
                }
                float chunk_sum = simd_sum(e0 + e1);

                if (simd_lane == 0u) {
                    state[q * 2 + 0] = new_max;
                    state[q * 2 + 1] = state[q * 2 + 1] * rescale + chunk_sum;
                    rescales[q] = rescale;
                }
            } else {
                // Unused query row (q >= n_q): zero both cells per lane so
                // the scores row is fully zero (MMA then produces 0 × V = 0),
                // and zero rescales[q] so the pre-MMA rescale of out_tg is
                // also 0. Two independent guards — see Iter 3 commentary
                // for the full argument.
                scores[q * C + simd_lane]        = 0.0f;
                scores[q * C + simd_lane + NW]   = 0.0f;
                if (simd_lane == 0u) {
                    rescales[q] = 0.0f;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- MMA V accumulation ---
        //
        // Pre-rescale out_tg by per-query rescales[q] (cooperative), then
        // fuse V MMA with the add via po pre-loaded with rescaled out_tg:
        //   po_new = scores × V + po   where po = rescales[q] · out_tg
        for (uint idx = tid; idx < Q_PER_TG * hd; idx += N_THREADS) {
            uint q = idx / hd;
            out_tg[idx] = out_tg[idx] * rescales[q];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Each SG owns a dim-tile (8 output cols); round-robin across the
        // hd/8 tiles. For hd=64 → 8 tiles → 1 per SG.
        // For hd=128 → 16 tiles → 2 per SG.
        const uint dim_tiles = hd / 8u;
        const uint inner_tiles = C / 8u;  // C=64 → 8 MMA inner iterations
        for (uint d_off = simd_id; d_off < dim_tiles; d_off += NSG) {
            simdgroup_float8x8 po;
            simdgroup_load(po, out_tg + d_off * 8u, hd);

            simdgroup_float8x8 s_mat;
            simdgroup_half8x8  v_mat;
            for (uint t_in = 0u; t_in < inner_tiles; t_in++) {
                simdgroup_load(s_mat, scores + t_in * 8u, C);
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
