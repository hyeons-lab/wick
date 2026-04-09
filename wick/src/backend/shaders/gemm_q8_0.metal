#include <metal_stdlib>
using namespace metal;

// GEMM for Q8_0 weights using simdgroup matrix multiply-accumulate.
// Same structure as gemm_q4_0 but with Q8_0 dequantization (34-byte blocks).
//
// Dispatch: (ceil(n/32), ceil(m/64)) TGs × 128 threads (4 simdgroups).
// Threadgroup memory: 8 KB (4 KB weights as half + 4 KB input as float).

#define BLOCK_SIZE_M  64
#define BLOCK_SIZE_N  32
#define BLOCK_SIZE_K  32
#define THREAD_MAT_M   4
#define THREAD_MAT_N   2
#define THREAD_PER_ROW 2
#define THREAD_PER_COL 4
#define SG_MAT_SIZE   64

struct GemmParams {
    uint m;
    uint k;
    uint n;
    uint x_stride;
    uint y_stride;
    uint _pad;
};

struct block_q8_0 {
    half d;
    char qs[32];
};

void dequantize_q8_0(device const block_q8_0 * xb, short il, thread half4x4 & reg) {
    device const char * qs = xb->qs;
    const float d = xb->d;

    float4x4 reg_f;
    for (int i = 0; i < 16; i++) {
        reg_f[i/4][i%4] = qs[i + 16*il] * d;
    }
    reg = (half4x4) reg_f;
}

kernel void gemm_q8_0(
    const device uchar * src0 [[buffer(0)]],
    const device float * src1 [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant GemmParams & params [[buffer(3)]],
    threadgroup char * shmem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    const uint m = params.m;
    const uint k = params.k;
    const uint n = params.n;
    const uint x_stride = params.x_stride;
    const uint y_stride = params.y_stride;

    const uint nb = k / 32;
    const uint row_bytes = nb * 34; // Q8_0: 2 (f16 scale) + 32 (int8 quants) = 34 bytes
    const short nl = 2;

    threadgroup half  * sa = (threadgroup half  *)(shmem);
    threadgroup float * sb = (threadgroup float *)(shmem + 4096);

    const int r0 = tgpig.y;
    const int r1 = tgpig.x;

    const short n_rows = min((int)m - r0 * BLOCK_SIZE_M, BLOCK_SIZE_M);
    const short n_cols = min((int)n - r1 * BLOCK_SIZE_N, BLOCK_SIZE_N);

    const short thread_row = min((short)(tiitg / THREAD_PER_ROW), (short)(n_rows - 1));
    const short thread_col = min((short)(tiitg / THREAD_PER_COL), (short)(n_cols - 1));

    simdgroup_half8x8  ma[4];
    simdgroup_float8x8 mb[2];
    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    short il = (tiitg % THREAD_PER_ROW);

    device const block_q8_0 * x = (device const block_q8_0 *)(src0
        + row_bytes * (r0 * BLOCK_SIZE_M + thread_row)) + il / nl;

    device const float * y = src1
        + x_stride * (r1 * BLOCK_SIZE_N + thread_col)
        + (BLOCK_SIZE_K / THREAD_PER_COL * (tiitg % THREAD_PER_COL));

    for (uint loop_k = 0; loop_k < k; loop_k += BLOCK_SIZE_K) {
        half4x4 temp_a;
        dequantize_q8_0(x, il, temp_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll(16)
        for (short i = 0; i < 16; i++) {
            *(sa + SG_MAT_SIZE * ((tiitg / THREAD_PER_ROW / 8)
            +                     (tiitg % THREAD_PER_ROW) * 16 + (i / 8) * 8)
            +                     (tiitg / THREAD_PER_ROW) % 8  + (i & 7) * 8) = temp_a[i/4][i%4];
        }

        *(threadgroup float2x4 *)(sb + 32 * 8 * (tiitg % THREAD_PER_COL) + 8 * (tiitg / THREAD_PER_COL)) = *((device float2x4 *) y);

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1) / nl : x;
        y += BLOCK_SIZE_K;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half  * lsma = (sa + THREAD_MAT_M * SG_MAT_SIZE * (sgitg % 2));
        threadgroup const float * lsmb = (sb + THREAD_MAT_N * SG_MAT_SIZE * (sgitg / 2));

        #pragma unroll(4)
        for (short ik = 0; ik < BLOCK_SIZE_K / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            #pragma unroll(4)
            for (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + SG_MAT_SIZE * i);
            }

            #pragma unroll(2)
            for (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + SG_MAT_SIZE * i);
            }

            simdgroup_barrier(mem_flags::mem_none);

            #pragma unroll(8)
            for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }

            lsma += (BLOCK_SIZE_M / 8) * SG_MAT_SIZE;
            lsmb += (BLOCK_SIZE_N / 8) * SG_MAT_SIZE;
        }
    }

    if ((r0 + 1) * BLOCK_SIZE_M <= (int)m && (r1 + 1) * BLOCK_SIZE_N <= (int)n) {
        // Fast path: full tile, no accumulate — direct simdgroup store.
        device float * C = dst
            + (BLOCK_SIZE_M * r0 + 32 * (sgitg & 1))
            + (BLOCK_SIZE_N * r1 + 16 * (sgitg >> 1)) * y_stride;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8 * (i % 4) + 8 * y_stride * (i / 4), y_stride);
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float * temp_str = ((threadgroup float *) shmem)
            + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * BLOCK_SIZE_M;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * BLOCK_SIZE_M * (i / 4), BLOCK_SIZE_M);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0) {
            for (int j = tiitg; j < n_cols; j += BLOCK_SIZE_N) {
                device float  * D  = dst + (r0 * BLOCK_SIZE_M) + (r1 * BLOCK_SIZE_N + j) * y_stride;
                threadgroup float  * S  = ((threadgroup float *) shmem) + (j * BLOCK_SIZE_M);
                device float4 * D4 = (device float4 *) D;
                threadgroup float4 * S4 = (threadgroup float4 *) S;
                int i = 0;
                for (; i < n_rows / 4; i++) {
                    *(D4 + i) = *(S4 + i);
                }
                i *= 4;
                for (; i < n_rows; i++) {
                    *(D + i) = *(S + i);
                }
            }
        }
    }
}
