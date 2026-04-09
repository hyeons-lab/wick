#include <metal_stdlib>
using namespace metal;

// Q8_0 GEMV: y[m] = dequant(A_q8_0[m, k]) × x[k]
// Avoids struct for block access — uses raw byte offsets to prevent
// any Metal struct padding issues.
// 32 threads per simdgroup, 2 rows per TG.
// Dispatch: ceil(m/2) threadgroups × 32 threads.

struct Params {
    uint m;
    uint k;
};

constant constexpr uint BLOCK_BYTES = 34; // 2 (f16 scale) + 32 (int8 quants)
constant constexpr uint ROWS_PER_TG = 2;

kernel void gemv_q8_0(
    const device uchar* a [[buffer(0)]],
    const device float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant Params& params [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]]
) {
    uint m = params.m;
    uint k = params.k;
    uint nb = k / 32;
    uint row_bytes = nb * BLOCK_BYTES;

    for (uint r = 0; r < ROWS_PER_TG; r++) {
        uint row = tg_id * ROWS_PER_TG + r;
        if (row >= m) continue;

        const device uchar* row_ptr = a + row * row_bytes;
        float sum = 0.0f;

        for (uint bi = tiisg; bi < nb; bi += 32) {
            const device uchar* blk = row_ptr + bi * BLOCK_BYTES;
            // Scale: first 2 bytes = f16
            float d = float(*(const device half*)blk);
            // Quants: next 32 bytes = int8_t[32]
            const device int8_t* qs = (const device int8_t*)(blk + 2);

            float partial = 0.0f;
            for (uint j = 0; j < 32; j++) {
                partial += float(qs[j]) * x[bi * 32 + j];
            }
            sum += d * partial;
        }

        float total = simd_sum(sum);
        if (tiisg == 0) {
            y[row] = total;
        }
    }
}

kernel void gemv_q8_0_accum(
    const device uchar* a [[buffer(0)]],
    const device float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant Params& params [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]]
) {
    uint m = params.m;
    uint k = params.k;
    uint nb = k / 32;
    uint row_bytes = nb * BLOCK_BYTES;

    for (uint r = 0; r < ROWS_PER_TG; r++) {
        uint row = tg_id * ROWS_PER_TG + r;
        if (row >= m) continue;

        const device uchar* row_ptr = a + row * row_bytes;
        float sum = 0.0f;

        for (uint bi = tiisg; bi < nb; bi += 32) {
            const device uchar* blk = row_ptr + bi * BLOCK_BYTES;
            float d = float(*(const device half*)blk);
            const device int8_t* qs = (const device int8_t*)(blk + 2);

            float partial = 0.0f;
            for (uint j = 0; j < 32; j++) {
                partial += float(qs[j]) * x[bi * 32 + j];
            }
            sum += d * partial;
        }

        float total = simd_sum(sum);
        if (tiisg == 0) {
            y[row] += total;
        }
    }
}
