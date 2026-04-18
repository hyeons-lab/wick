#include <metal_stdlib>
using namespace metal;

// FlashAttention (Dao 2022) for a single query vector (decode path).
// Tiled K/V with online softmax — one pass over KV cache, bounded
// threadgroup memory regardless of seq_len.
//
// Iter 2 rewrite of the previous flash kernel (TILE=32, scalar QK inner
// loop, 224 idle threads during scoring, per-thread per-dim serial V
// accumulation). Ports the proven optimizations from the classic
// `attention.metal` into the tile loop:
//   - TILE=256: scoring uses all 256 threads (one dot product per thread
//     per tile iteration) — no idle threads during Phase 1.
//   - QK uses float4-from-half4 vectorized loads.
//   - Phase 4 V accumulation uses 8 SGs × 32 lanes with specialized
//     half2/half4 fast paths for hd=64/128 (same pattern as classic).
//   - Outer max/sum reductions use the SG-tree pattern (simd_max in each
//     SG → cross-SG via sg_val[8]), matching classic.
//
// Still flash (not classic) because:
//   - `tile_scores` shmem is bounded at TILE=256 elements (1 KB), not
//     seq_len. This is the property that lets flash handle seq_len
//     beyond classic's MAX_SEQ_LEN=4096 cap.
//   - Online softmax with running_max / running_sum / correction keeps
//     the O(1)-shmem-in-seq_len invariant.
//
// Constraints:
//   - head_dim ≤ MAX_HEAD_DIM = 128 (bounds q_shared and partials_tg).
//     Host caller asserts this in the Rust dispatch site.
//   - head_dim % 4 == 0 (Phase 1 uses float4-from-half4 loads with no
//     scalar tail; matches classic attention.metal which has the same
//     unstated requirement). All current LFM2 models satisfy this
//     (hd ∈ {64, 128}).
//
// Fast paths for V accumulation are gated on the exact head_dim value
// (not on dims_per_lane) so they never read past head_dim into the
// next kv-head's slot in v_cache. head_dim ∈ {64, 128} hit fast paths;
// everything else falls into the bounds-checked generic loop.
//
// One threadgroup per head, 256 threads. Binding layout matches
// attention.metal so the Rust dispatch is identical (drop-in).

constant constexpr uint TILE = 256;
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

kernel void flash_attention(
    const device float* q [[buffer(0)]],
    const device half*  k_cache [[buffer(1)]],
    const device half*  v_cache [[buffer(2)]],
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

    uint simd_lane = tid & 31u;
    uint simd_id   = tid >> 5u;

    threadgroup float q_shared[MAX_HEAD_DIM];
    threadgroup float tile_scores[TILE];
    threadgroup float sg_val[8];                     // cross-SG reduction buffer
    threadgroup float partials_tg[8 * MAX_HEAD_DIM]; // cross-SG po reduction

    // seq_len=0 would leave running_sum=0 and produce inv_sum=inf → NaN
    // output. Write zeros and bail defensively. (Not expected at runtime —
    // decode always has at least one KV slot — but cheap insurance.)
    if (seq_len == 0u) {
        if (tid < head_dim) {
            out[q_offset + tid] = 0.0f;
        }
        return;
    }

    // Load Q once into shared memory.
    // head_dim ≤ MAX_HEAD_DIM=128 is asserted on the host side in
    // encode_attention(), so this q_shared write and the partials_tg
    // indexing in the epilogue are always within the static bounds.
    if (tid < head_dim) {
        q_shared[tid] = q[q_offset + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Per-thread running softmax state + V accumulator (register).
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float po[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // Ceil-div so head_dim not a multiple of 32 is still fully covered
    // by the generic V-accum path (the `if (d < head_dim)` bounds check
    // masks the tail lanes). Classic attention.metal uses floor-div, which
    // silently drops dims 32..hd-1 for hd like 48 or 80 — not triggered by
    // any current LFM2 model but worth fixing here.
    uint dims_per_lane = (head_dim + 31u) / 32u;
    if (dims_per_lane > 8u) dims_per_lane = 8u;
    uint hd4 = head_dim / 4u;

    for (uint tile_start = 0u; tile_start < seq_len; tile_start += TILE) {
        uint tile_len = min(TILE, seq_len - tile_start);

        // --- Phase 1: QK scoring (all 256 threads). ---
        // Each active thread scores one timestep of this tile.
        if (tid < tile_len) {
            float acc = 0.0f;
            uint k_base = (tile_start + tid) * kv_dim + kv_h_offset;
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
            tile_scores[tid] = acc * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Phase 2: tile max (cross-SG tree). ---
        float local_max = (tid < tile_len) ? tile_scores[tid] : -INFINITY;
        float sg_max = simd_max(local_max);
        if (simd_lane == 0u) sg_val[simd_id] = sg_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_id == 0u) {
            float v = simd_lane < 8u ? sg_val[simd_lane] : -INFINITY;
            float total = simd_max(v);
            if (simd_lane == 0u) sg_val[0] = total;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tile_max = sg_val[0];

        float new_max = max(running_max, tile_max);
        // On first iter running_max == -INFINITY → correction = 0, so the
        // existing po (zero-initialized above) stays 0 after the
        // `po[i] *= correction` step below. Using the ternary guards
        // against `exp(-INFINITY - finite)` returning NaN on some drivers
        // (IEEE says 0, but historically inconsistent).
        float correction = (running_max > -INFINITY) ? exp(running_max - new_max) : 0.0f;
        running_max = new_max;

        // --- Phase 3: exp (in place) + tile sum (cross-SG tree). ---
        if (tid < tile_len) {
            tile_scores[tid] = exp(tile_scores[tid] - new_max);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float local_sum = (tid < tile_len) ? tile_scores[tid] : 0.0f;
        float sg_sum = simd_sum(local_sum);
        if (simd_lane == 0u) sg_val[simd_id] = sg_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_id == 0u) {
            float v = simd_lane < 8u ? sg_val[simd_lane] : 0.0f;
            float total = simd_sum(v);
            if (simd_lane == 0u) sg_val[0] = total;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tile_sum = sg_val[0];
        running_sum = running_sum * correction + tile_sum;

        // --- Phase 4: V accumulation for this tile. ---
        // Mirrors classic's Phase 3: 8 SGs split the tile timesteps;
        // each lane owns dims_per_lane contiguous output dims. Apply
        // per-tile `correction` to running po first, then add the tile
        // contribution (scores × V).
        #pragma clang loop unroll(full)
        for (uint i = 0u; i < 8u; i++) {
            if (i < dims_per_lane) {
                po[i] *= correction;
            }
        }

        uint chunk = (tile_len + 7u) / 8u;
        uint tt_start = simd_id * chunk;
        uint tt_end   = min(tt_start + chunk, tile_len);

        // Gate fast paths on exact head_dim so they never read past the
        // head boundary into the next kv-head's slot in v_cache.
        // Matching `dims_per_lane == 2/4` alone would let hd=48 or hd=80
        // (with the new ceil-div) take the half2 path and read invalid V.
        if (head_dim == 64u) {
            // hd=64 fast path: half2 load per timestep. 32 lanes × 2 dims = 64.
            for (uint tt = tt_start; tt < tt_end; tt++) {
                float s = tile_scores[tt];
                uint v_base = (tile_start + tt) * kv_dim + kv_h_offset + simd_lane * 2u;
                float2 v2 = float2(*((device const half2*) (v_cache + v_base)));
                po[0] += s * v2.x;
                po[1] += s * v2.y;
            }
        } else if (head_dim == 128u) {
            // hd=128 fast path: half4 load per timestep. 32 lanes × 4 dims = 128.
            for (uint tt = tt_start; tt < tt_end; tt++) {
                float s = tile_scores[tt];
                uint v_base = (tile_start + tt) * kv_dim + kv_h_offset + simd_lane * 4u;
                float4 v4 = float4(*((device const half4*) (v_cache + v_base)));
                po[0] += s * v4.x;
                po[1] += s * v4.y;
                po[2] += s * v4.z;
                po[3] += s * v4.w;
            }
        } else {
            for (uint tt = tt_start; tt < tt_end; tt++) {
                float s = tile_scores[tt];
                uint v_base = (tile_start + tt) * kv_dim + kv_h_offset;
                #pragma clang loop unroll(full)
                for (uint i = 0u; i < 8u; i++) {
                    if (i < dims_per_lane) {
                        uint d = simd_lane * dims_per_lane + i;
                        if (d < head_dim) {
                            po[i] += s * float(v_cache[v_base + d]);
                        }
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // --- Epilogue: cross-SG reduction of po, divide by running_sum, write. ---
    for (uint i = 0u; i < dims_per_lane; i++) {
        uint d = simd_lane * dims_per_lane + i;
        if (d < head_dim) {
            partials_tg[simd_id * head_dim + d] = po[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0u) {
        // running_sum > 0 is guaranteed here: seq_len == 0 is handled by
        // the early return near the kernel entry, and for seq_len >= 1
        // each tile contributes at least one exp() > 0 to running_sum.
        // So `1.0 / running_sum` is well-defined.
        float inv_sum = 1.0f / running_sum;
        for (uint i = 0u; i < dims_per_lane; i++) {
            uint d = simd_lane * dims_per_lane + i;
            if (d < head_dim) {
                float sum = 0.0f;
                #pragma clang loop unroll(full)
                for (uint sg = 0u; sg < 8u; sg++) {
                    sum += partials_tg[sg * head_dim + d];
                }
                out[q_offset + d] = sum * inv_sum;
            }
        }
    }
}
