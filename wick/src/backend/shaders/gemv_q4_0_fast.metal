#include <metal_stdlib>
using namespace metal;

// Fast Q4_0 GEMV based on llama.cpp's mul_mv_q4_0_f32 kernel.
//
// Key optimizations vs gemv_q4_0.metal:
//  - 4 rows per simdgroup × 2 simdgroups per TG = 8 rows per threadgroup
//    (amortizes x loads 4× across rows compared to our 2-row kernel).
//  - 2 threads per block: each processes 16 elements (half block). Threads
//    (tiisg, tiisg+1) cooperate on one block.
//  - uint16 nibble loads (not byte-by-byte scalar).
//  - Pre-scaled y values: yl[i+1] /= 256, yl[i+8] /= 16, yl[i+9] /= 4096 so
//    raw nibble masks multiply directly — no per-element bit shifts in the
//    inner loop.
//  - Sumy bias hoisting: d * (sumy * -8 + acc) replaces N per-element -8 subs.
//
// Dispatch: ceil(m/8) threadgroups × 64 threads (2 simdgroups).
//
// Weight layout: standard Q4_0 blocks, 18 bytes each. Caller MUST pad the
// buffer by at least 2 bytes (uint16 load alignment).

struct Params {
    uint m;
    uint k;
};

constant constexpr uint ROWS_PER_SIMD = 4;
constant constexpr uint NSG = 2;             // simdgroups per TG
constant constexpr uint ROWS_PER_TG = ROWS_PER_SIMD * NSG;  // 8
constant constexpr uint BLOCK_BYTES = 18;
constant constexpr uint NQ = 16;             // threads per half-pair stride

// One Q4_0 block viewed as raw bytes.
struct block_q4_0 {
    half d;
    uchar qs[16];
};

// Compute the contribution of half a Q4_0 block (16 elements) to one row's dot
// product. `yl` holds the 16 pre-scaled y values for this block-half. `sumy`
// is the raw y-sum (so the -8 bias can be hoisted out: result += d*(sumy*-8 + acc)).
inline float half_block_dot(
    device const block_q4_0* blk,
    float sumy,
    thread const float* yl,
    uint il       // 0 or 8 (which half)
) {
    float d = float(blk->d);
    // nibble stream as uint16: 4 uint16s per half.
    device const uint16_t* qs = ((device const uint16_t*) &blk->d) + 1 + (il / 2);
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 8; i += 2) {
        uint16_t q = qs[i / 2];
        acc[0] += yl[i + 0] * float(q & 0x000Fu);
        acc[1] += yl[i + 1] * float(q & 0x0F00u);
        acc[2] += yl[i + 8] * float(q & 0x00F0u);
        acc[3] += yl[i + 9] * float(q & 0xF000u);
    }
    return d * (sumy * -8.0f + acc[0] + acc[1] + acc[2] + acc[3]);
}

kernel void gemv_q4_0_fast(
    const device uchar* a [[buffer(0)]],
    const device float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant Params& params [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]]
) {
    uint m = params.m;
    uint k = params.k;
    uint nb = k / 32;
    uint row_bytes = nb * BLOCK_BYTES;

    // First of this simdgroup's 4 rows.
    uint r0 = (tg_id * NSG + sgitg) * ROWS_PER_SIMD;

    // Per-row block-pointer base.
    device const block_q4_0* ax[ROWS_PER_SIMD];
    #pragma clang loop unroll(full)
    for (uint r = 0; r < ROWS_PER_SIMD; r++) {
        ax[r] = (device const block_q4_0*) (a + (r0 + r) * row_bytes);
    }

    // Thread position inside block group:
    // ix = tiisg / 2 (0..15): which block in the stride group
    // il = (tiisg % 2) * 8: which half of the block (0 = low, 8 = high)
    uint ix = tiisg / 2;
    uint il = (tiisg & 1u) * 8;

    float sumf[ROWS_PER_SIMD] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Each thread iterates blocks at stride 16, processing half of each block.
    float yl[16];
    device const float* yb = x + ix * 32 + il;

    for (uint ib = ix; ib < nb; ib += NQ) {
        // Cache 16 y values for this half-block, pre-scaling as we go so the
        // inner loop can multiply raw nibble masks without shifting.
        float sumy[2] = {0.0f, 0.0f};
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8; i += 2) {
            sumy[0]   += yb[i + 0] + yb[i + 1];
            yl[i + 0]  = yb[i + 0];
            yl[i + 1]  = yb[i + 1] / 256.0f;
            sumy[1]   += yb[i + 16] + yb[i + 17];
            yl[i + 8]  = yb[i + 16] / 16.0f;
            yl[i + 9]  = yb[i + 17] / 4096.0f;
        }

        float sumy_total = sumy[0] + sumy[1];
        #pragma clang loop unroll(full)
        for (uint r = 0; r < ROWS_PER_SIMD; r++) {
            sumf[r] += half_block_dot(ax[r] + ib, sumy_total, yl, il);
        }

        yb += 32 * NQ;  // advance by NQ blocks (NQ*32 floats)
    }

    // Reduce across simdgroup lanes.
    #pragma clang loop unroll(full)
    for (uint r = 0; r < ROWS_PER_SIMD; r++) {
        float tot = simd_sum(sumf[r]);
        if (tiisg == 0 && r0 + r < m) {
            y[r0 + r] = tot;
        }
    }
}

// Slim variant: ROWS_PER_SIMD=2, NSG=1, 32 threads per TG. Matches classic's
// TG count (m/2 TGs) but uses the fast algorithm. Wins at small m (≤256)
// where TG count matters more than per-TG amortization.
kernel void gemv_q4_0_fast_slim(
    const device uchar* a [[buffer(0)]],
    const device float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant Params& params [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]]
) {
    constexpr uint NR = 2;
    uint m = params.m;
    uint k = params.k;
    uint nb = k / 32;
    uint row_bytes = nb * BLOCK_BYTES;
    uint r0 = tg_id * NR;

    device const block_q4_0* ax[NR];
    #pragma clang loop unroll(full)
    for (uint r = 0; r < NR; r++) {
        ax[r] = (device const block_q4_0*) (a + (r0 + r) * row_bytes);
    }

    uint ix = tiisg / 2;
    uint il = (tiisg & 1u) * 8;

    float sumf[NR] = {0.0f, 0.0f};
    float yl[16];
    device const float* yb = x + ix * 32 + il;

    for (uint ib = ix; ib < nb; ib += NQ) {
        float sumy[2] = {0.0f, 0.0f};
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8; i += 2) {
            sumy[0]   += yb[i + 0] + yb[i + 1];
            yl[i + 0]  = yb[i + 0];
            yl[i + 1]  = yb[i + 1] / 256.0f;
            sumy[1]   += yb[i + 16] + yb[i + 17];
            yl[i + 8]  = yb[i + 16] / 16.0f;
            yl[i + 9]  = yb[i + 17] / 4096.0f;
        }

        float sumy_total = sumy[0] + sumy[1];
        #pragma clang loop unroll(full)
        for (uint r = 0; r < NR; r++) {
            sumf[r] += half_block_dot(ax[r] + ib, sumy_total, yl, il);
        }
        yb += 32 * NQ;
    }

    #pragma clang loop unroll(full)
    for (uint r = 0; r < NR; r++) {
        float tot = simd_sum(sumf[r]);
        if (tiisg == 0 && r0 + r < m) {
            y[r0 + r] = tot;
        }
    }
}

// Fused gate+up using the fast algorithm. Each TG outputs 1 row of y_gate
// AND 1 row of y_up, reading the SAME x (pre-scaled once, used twice).
// 32 threads/TG, m TGs total. Same dispatch shape as the original
// gemv_q4_0_gate_up but with llama.cpp-style inner loop.
kernel void gemv_q4_0_fast_slim_gate_up(
    const device uchar* a_gate [[buffer(0)]],
    const device uchar* a_up [[buffer(1)]],
    const device float* x [[buffer(2)]],
    device float* y_gate [[buffer(3)]],
    device float* y_up [[buffer(4)]],
    constant Params& params [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]]
) {
    uint m = params.m;
    uint k = params.k;
    uint nb = k / 32;
    uint row_bytes = nb * BLOCK_BYTES;
    uint row = tg_id;
    if (row >= m) return;

    device const block_q4_0* gate_ptr =
        (device const block_q4_0*) (a_gate + row * row_bytes);
    device const block_q4_0* up_ptr =
        (device const block_q4_0*) (a_up + row * row_bytes);

    uint ix = tiisg / 2;
    uint il = (tiisg & 1u) * 8;

    float sum_gate = 0.0f;
    float sum_up = 0.0f;
    float yl[16];
    device const float* yb = x + ix * 32 + il;

    for (uint ib = ix; ib < nb; ib += NQ) {
        // Pre-scale y ONCE, shared between gate and up.
        float sumy[2] = {0.0f, 0.0f};
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8; i += 2) {
            sumy[0]   += yb[i + 0] + yb[i + 1];
            yl[i + 0]  = yb[i + 0];
            yl[i + 1]  = yb[i + 1] / 256.0f;
            sumy[1]   += yb[i + 16] + yb[i + 17];
            yl[i + 8]  = yb[i + 16] / 16.0f;
            yl[i + 9]  = yb[i + 17] / 4096.0f;
        }
        float sumy_total = sumy[0] + sumy[1];

        sum_gate += half_block_dot(gate_ptr + ib, sumy_total, yl, il);
        sum_up   += half_block_dot(up_ptr + ib, sumy_total, yl, il);
        yb += 32 * NQ;
    }

    float total_gate = simd_sum(sum_gate);
    float total_up = simd_sum(sum_up);
    if (tiisg == 0) {
        y_gate[row] = total_gate;
        y_up[row] = total_up;
    }
}

// Fused gate+up, 2 rows/TG. Same dispatch as fast_slim_gate_up in thread
// count (32 threads) but halves TG count, amortizing x across 2 rows × 2
// weights = 4 dot products per x load.
kernel void gemv_q4_0_fast_slim2_gate_up(
    const device uchar* a_gate [[buffer(0)]],
    const device uchar* a_up [[buffer(1)]],
    const device float* x [[buffer(2)]],
    device float* y_gate [[buffer(3)]],
    device float* y_up [[buffer(4)]],
    constant Params& params [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]]
) {
    constexpr uint NR = 2;
    uint m = params.m;
    uint k = params.k;
    uint nb = k / 32;
    uint row_bytes = nb * BLOCK_BYTES;
    uint r0 = tg_id * NR;

    device const block_q4_0* ag[NR];
    device const block_q4_0* au[NR];
    #pragma clang loop unroll(full)
    for (uint r = 0; r < NR; r++) {
        ag[r] = (device const block_q4_0*) (a_gate + (r0 + r) * row_bytes);
        au[r] = (device const block_q4_0*) (a_up   + (r0 + r) * row_bytes);
    }

    uint ix = tiisg / 2;
    uint il = (tiisg & 1u) * 8;

    float sg[NR] = {0.0f, 0.0f};
    float su[NR] = {0.0f, 0.0f};
    float yl[16];
    device const float* yb = x + ix * 32 + il;

    for (uint ib = ix; ib < nb; ib += NQ) {
        float sumy[2] = {0.0f, 0.0f};
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8; i += 2) {
            sumy[0]   += yb[i + 0] + yb[i + 1];
            yl[i + 0]  = yb[i + 0];
            yl[i + 1]  = yb[i + 1] / 256.0f;
            sumy[1]   += yb[i + 16] + yb[i + 17];
            yl[i + 8]  = yb[i + 16] / 16.0f;
            yl[i + 9]  = yb[i + 17] / 4096.0f;
        }
        float sumy_total = sumy[0] + sumy[1];
        #pragma clang loop unroll(full)
        for (uint r = 0; r < NR; r++) {
            sg[r] += half_block_dot(ag[r] + ib, sumy_total, yl, il);
            su[r] += half_block_dot(au[r] + ib, sumy_total, yl, il);
        }
        yb += 32 * NQ;
    }

    #pragma clang loop unroll(full)
    for (uint r = 0; r < NR; r++) {
        float tg_v = simd_sum(sg[r]);
        float tu_v = simd_sum(su[r]);
        if (tiisg == 0 && r0 + r < m) {
            y_gate[r0 + r] = tg_v;
            y_up[r0 + r]   = tu_v;
        }
    }
}

// Fused gate+up, multi-row variant. Each TG emits ROWS_PER_TG rows of both
// y_gate AND y_up. Benefits vs the slim variant:
//  - x is pre-scaled once per TG and shared across 2*ROWS_PER_TG dot products.
//  - TG count drops by ROWS_PER_TG× (fewer launch hops).
// Dispatch: ceil(m/ROWS_PER_TG) TGs × 64 threads (2 simdgroups).
kernel void gemv_q4_0_fast_gate_up(
    const device uchar* a_gate [[buffer(0)]],
    const device uchar* a_up [[buffer(1)]],
    const device float* x [[buffer(2)]],
    device float* y_gate [[buffer(3)]],
    device float* y_up [[buffer(4)]],
    constant Params& params [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]]
) {
    uint m = params.m;
    uint k = params.k;
    uint nb = k / 32;
    uint row_bytes = nb * BLOCK_BYTES;
    uint r0 = (tg_id * NSG + sgitg) * ROWS_PER_SIMD;

    device const block_q4_0* ag[ROWS_PER_SIMD];
    device const block_q4_0* au[ROWS_PER_SIMD];
    #pragma clang loop unroll(full)
    for (uint r = 0; r < ROWS_PER_SIMD; r++) {
        ag[r] = (device const block_q4_0*) (a_gate + (r0 + r) * row_bytes);
        au[r] = (device const block_q4_0*) (a_up   + (r0 + r) * row_bytes);
    }

    uint ix = tiisg / 2;
    uint il = (tiisg & 1u) * 8;

    float sg[ROWS_PER_SIMD] = {0.0f, 0.0f, 0.0f, 0.0f};
    float su[ROWS_PER_SIMD] = {0.0f, 0.0f, 0.0f, 0.0f};
    float yl[16];
    device const float* yb = x + ix * 32 + il;

    for (uint ib = ix; ib < nb; ib += NQ) {
        float sumy[2] = {0.0f, 0.0f};
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8; i += 2) {
            sumy[0]   += yb[i + 0] + yb[i + 1];
            yl[i + 0]  = yb[i + 0];
            yl[i + 1]  = yb[i + 1] / 256.0f;
            sumy[1]   += yb[i + 16] + yb[i + 17];
            yl[i + 8]  = yb[i + 16] / 16.0f;
            yl[i + 9]  = yb[i + 17] / 4096.0f;
        }
        float sumy_total = sumy[0] + sumy[1];
        #pragma clang loop unroll(full)
        for (uint r = 0; r < ROWS_PER_SIMD; r++) {
            sg[r] += half_block_dot(ag[r] + ib, sumy_total, yl, il);
            su[r] += half_block_dot(au[r] + ib, sumy_total, yl, il);
        }
        yb += 32 * NQ;
    }

    #pragma clang loop unroll(full)
    for (uint r = 0; r < ROWS_PER_SIMD; r++) {
        float tg_v = simd_sum(sg[r]);
        float tu_v = simd_sum(su[r]);
        if (tiisg == 0 && r0 + r < m) {
            y_gate[r0 + r] = tg_v;
            y_up[r0 + r]   = tu_v;
        }
    }
}

// Fused rmsnorm + gate+up: computes inv_rms of `hidden`, then does the
// gate+up GEMV with normalized & scaled input inline. Saves 1 dispatch
// per FFN layer (the separate rmsnorm pass).
//
// Each TG redundantly computes the sum-of-squares reduction. Hidden is
// ≤4KB so it's L1/L2 resident after the first TG — redundant reads are
// cheap vs saving dispatch overhead.
struct RMSParams {
    uint m;
    uint k;
    uint eps_bits;
    uint _pad;
};

kernel void gemv_q4_0_fast_rmsnorm_gate_up(
    const device uchar* a_gate [[buffer(0)]],
    const device uchar* a_up [[buffer(1)]],
    const device float* hidden [[buffer(2)]],
    const device float* norm_w [[buffer(3)]],
    device float* y_gate [[buffer(4)]],
    device float* y_up [[buffer(5)]],
    constant RMSParams& params [[buffer(6)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]]
) {
    uint m = params.m;
    uint k = params.k;
    float eps = as_type<float>(params.eps_bits);
    uint nb = k / 32;
    uint row_bytes = nb * BLOCK_BYTES;
    uint row = tg_id;
    if (row >= m) return;

    // Phase 1: compute inv_rms. 32 threads stride through hidden[0..k].
    float partial = 0.0f;
    for (uint i = tiisg; i < k; i += 32u) {
        float v = hidden[i];
        partial += v * v;
    }
    float sumsq = simd_sum(partial);
    float inv_rms = 1.0f / sqrt(sumsq / float(k) + eps);

    // Phase 2: gate+up GEMV with normalized input on-the-fly.
    device const block_q4_0* gate_ptr =
        (device const block_q4_0*) (a_gate + row * row_bytes);
    device const block_q4_0* up_ptr =
        (device const block_q4_0*) (a_up + row * row_bytes);

    uint ix = tiisg / 2;
    uint il = (tiisg & 1u) * 8;

    float sum_gate = 0.0f;
    float sum_up = 0.0f;
    float yl[16];

    for (uint ib = ix; ib < nb; ib += NQ) {
        // Compute normalized values on-the-fly as we load x.
        uint base = ib * 32 + il;
        float sumy[2] = {0.0f, 0.0f};
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8; i += 2) {
            float h0 = hidden[base + i +  0] * inv_rms * norm_w[base + i +  0];
            float h1 = hidden[base + i +  1] * inv_rms * norm_w[base + i +  1];
            float h8 = hidden[base + i + 16] * inv_rms * norm_w[base + i + 16];
            float h9 = hidden[base + i + 17] * inv_rms * norm_w[base + i + 17];
            sumy[0]   += h0 + h1;
            yl[i + 0]  = h0;
            yl[i + 1]  = h1 / 256.0f;
            sumy[1]   += h8 + h9;
            yl[i + 8]  = h8 / 16.0f;
            yl[i + 9]  = h9 / 4096.0f;
        }
        float sumy_total = sumy[0] + sumy[1];

        sum_gate += half_block_dot(gate_ptr + ib, sumy_total, yl, il);
        sum_up   += half_block_dot(up_ptr + ib, sumy_total, yl, il);
    }

    float total_gate = simd_sum(sum_gate);
    float total_up = simd_sum(sum_up);
    if (tiisg == 0) {
        y_gate[row] = total_gate;
        y_up[row] = total_up;
    }
}

// Slim variant with accumulate.
kernel void gemv_q4_0_fast_slim_accum(
    const device uchar* a [[buffer(0)]],
    const device float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant Params& params [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]]
) {
    constexpr uint NR = 2;
    uint m = params.m;
    uint k = params.k;
    uint nb = k / 32;
    uint row_bytes = nb * BLOCK_BYTES;
    uint r0 = tg_id * NR;

    device const block_q4_0* ax[NR];
    #pragma clang loop unroll(full)
    for (uint r = 0; r < NR; r++) {
        ax[r] = (device const block_q4_0*) (a + (r0 + r) * row_bytes);
    }

    uint ix = tiisg / 2;
    uint il = (tiisg & 1u) * 8;

    float sumf[NR] = {0.0f, 0.0f};
    float yl[16];
    device const float* yb = x + ix * 32 + il;

    for (uint ib = ix; ib < nb; ib += NQ) {
        float sumy[2] = {0.0f, 0.0f};
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8; i += 2) {
            sumy[0]   += yb[i + 0] + yb[i + 1];
            yl[i + 0]  = yb[i + 0];
            yl[i + 1]  = yb[i + 1] / 256.0f;
            sumy[1]   += yb[i + 16] + yb[i + 17];
            yl[i + 8]  = yb[i + 16] / 16.0f;
            yl[i + 9]  = yb[i + 17] / 4096.0f;
        }

        float sumy_total = sumy[0] + sumy[1];
        #pragma clang loop unroll(full)
        for (uint r = 0; r < NR; r++) {
            sumf[r] += half_block_dot(ax[r] + ib, sumy_total, yl, il);
        }
        yb += 32 * NQ;
    }

    #pragma clang loop unroll(full)
    for (uint r = 0; r < NR; r++) {
        float tot = simd_sum(sumf[r]);
        if (tiisg == 0 && r0 + r < m) {
            y[r0 + r] += tot;
        }
    }
}

// Accumulating variant — y += W × x, for fused residual adds.
kernel void gemv_q4_0_fast_accum(
    const device uchar* a [[buffer(0)]],
    const device float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant Params& params [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]]
) {
    uint m = params.m;
    uint k = params.k;
    uint nb = k / 32;
    uint row_bytes = nb * BLOCK_BYTES;
    uint r0 = (tg_id * NSG + sgitg) * ROWS_PER_SIMD;

    device const block_q4_0* ax[ROWS_PER_SIMD];
    #pragma clang loop unroll(full)
    for (uint r = 0; r < ROWS_PER_SIMD; r++) {
        ax[r] = (device const block_q4_0*) (a + (r0 + r) * row_bytes);
    }

    uint ix = tiisg / 2;
    uint il = (tiisg & 1u) * 8;

    float sumf[ROWS_PER_SIMD] = {0.0f, 0.0f, 0.0f, 0.0f};
    float yl[16];
    device const float* yb = x + ix * 32 + il;

    for (uint ib = ix; ib < nb; ib += NQ) {
        float sumy[2] = {0.0f, 0.0f};
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8; i += 2) {
            sumy[0]   += yb[i + 0] + yb[i + 1];
            yl[i + 0]  = yb[i + 0];
            yl[i + 1]  = yb[i + 1] / 256.0f;
            sumy[1]   += yb[i + 16] + yb[i + 17];
            yl[i + 8]  = yb[i + 16] / 16.0f;
            yl[i + 9]  = yb[i + 17] / 4096.0f;
        }

        float sumy_total = sumy[0] + sumy[1];
        #pragma clang loop unroll(full)
        for (uint r = 0; r < ROWS_PER_SIMD; r++) {
            sumf[r] += half_block_dot(ax[r] + ib, sumy_total, yl, il);
        }
        yb += 32 * NQ;
    }

    #pragma clang loop unroll(full)
    for (uint r = 0; r < ROWS_PER_SIMD; r++) {
        float tot = simd_sum(sumf[r]);
        if (tiisg == 0 && r0 + r < m) {
            y[r0 + r] += tot;
        }
    }
}
