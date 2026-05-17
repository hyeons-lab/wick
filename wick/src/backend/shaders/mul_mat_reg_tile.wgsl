// Register-tiled matmul kernel: dst = src0 * src1
//
// src0 (weights): [m, k] row-major
// src1 (activations): n column-vectors with params.x_stride floats each
// dst (output): n column-vectors with params.y_stride floats each
//
// Tiling strategy:
// Each workgroup covers (WORKGROUP_SIZE_M * TILE_M) rows of dst and
// (WORKGROUP_SIZE_N * TILE_N) cols. Each thread holds a TILE_M x TILE_N
// register accumulator.
//
// OOB note: init_shmem_* uses `select(0, src[..], in_bounds)` which loads
// unconditionally and relies on WebGPU robust buffer access to zero-fill
// past `params.k`. Production LFM2 callers pass `k` aligned to TILE_K so
// the OOB lanes never engage; for ad-hoc callers, trust the spec
// guarantee on the target adapter.

#define BYTE_HELPERS
#include "common_decls.tmpl"
#include "mul_mat_decls.tmpl"

#ifdef VEC
fn pack_acc_tile(acc: ptr<function, array<array<f32, TILE_N>, TILE_M>>, tn: u32, tm: u32) -> vec4<f32> {
    return vec4<f32>((*acc)[tm][tn], (*acc)[tm + 1u][tn], (*acc)[tm + 2u][tn], (*acc)[tm + 3u][tn]);
}
#else
fn pack_acc_tile(acc: ptr<function, array<array<f32, TILE_N>, TILE_M>>, tn: u32, tm: u32) -> f32 {
    return (*acc)[tm][tn];
}
#endif

struct MulMatParams {
    m: u32,
    k: u32,
    n: u32,
    x_stride: u32,
    y_stride: u32,
};

@group(0) @binding(0) var<storage, read> src0: array<SRC0_TYPE>;
@group(0) @binding(1) var<storage, read> src1: array<SRC1_TYPE>;
@group(0) @binding(2) var<storage, read_write> dst: array<DST_TYPE>;
@group(0) @binding(3) var<storage, read> params: MulMatParams;

const TOTAL_WORKGROUP_SIZE = WORKGROUP_SIZE_M * WORKGROUP_SIZE_N;
const TILE_SRC0_ROWS: u32 = WORKGROUP_SIZE_M * TILE_M;
#ifdef INIT_SRC0_SHMEM_Q4_0
const TILE_SRC0_STRIDE: u32 = TILE_K + 1u;
#else
const TILE_SRC0_STRIDE: u32 = TILE_K;
#endif
const TILE_SRC0_SHMEM: u32 = TILE_SRC0_STRIDE * TILE_SRC0_ROWS;
const TILE_SRC1_SHMEM: u32 = TILE_K * WORKGROUP_SIZE_N * TILE_N;

var<workgroup> shmem: array<f32, TILE_SRC0_SHMEM + TILE_SRC1_SHMEM>;

@compute @workgroup_size(TOTAL_WORKGROUP_SIZE)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let thread_id = local_id.x;
    let local_m = thread_id % WORKGROUP_SIZE_M;
    let local_n = thread_id / WORKGROUP_SIZE_M;

    let wg_m_count = (params.m + WORKGROUP_SIZE_M * TILE_M - 1u) / (WORKGROUP_SIZE_M * TILE_M);

    // Linearize wg_id so callers can dispatch as 2D (avoids the 65535
    // num_wg.x limit) without changing the kernel.
    let wg_linear = wg_id.y * num_wg.x + wg_id.x;
    let wg_m = wg_linear % wg_m_count;
    let wg_n = wg_linear / wg_m_count;

    let offset_m = wg_m * WORKGROUP_SIZE_M * TILE_M;
    let offset_n = wg_n * WORKGROUP_SIZE_N * TILE_N;
    let output_row_base = offset_m + local_m * TILE_M;
    let output_col_base = offset_n + local_n * TILE_N;

    var acc: array<array<f32, TILE_N>, TILE_M>;
    for (var tm = 0u; tm < TILE_M; tm++) {
        for (var tn = 0u; tn < TILE_N; tn++) {
            acc[tm][tn] = 0.0;
        }
    }

    for (var k_outer = 0u; k_outer < params.k; k_outer += TILE_K) {
        init_shmem_src0(thread_id, offset_m, k_outer);
        init_shmem_src1(thread_id, offset_n, k_outer);

        workgroupBarrier();

        let k_end = min(TILE_K, params.k - k_outer);

        for (var k_inner = 0u; k_inner < k_end; k_inner++) {
            var src0_tile: array<f32, TILE_M>;
            for (var tm = 0u; tm < TILE_M; tm++) {
                let src0_m = local_m * TILE_M + tm;
                let src0_idx = k_inner + src0_m * TILE_SRC0_STRIDE;
                src0_tile[tm] = shmem[src0_idx];
            }
            for (var tn = 0u; tn < TILE_N; tn++) {
                let src1_n = local_n * TILE_N + tn;
                let src1_idx = src1_n * TILE_K + k_inner;
                let src1_val = shmem[TILE_SRC0_SHMEM + src1_idx];
                for (var tm = 0u; tm < TILE_M; tm++) {
                    acc[tm][tn] += src0_tile[tm] * src1_val;
                }
            }
        }

        workgroupBarrier();
    }

    for (var tn = 0u; tn < TILE_N; tn++) {
        let global_col = output_col_base + tn;
        if (global_col < params.n) {
            for (var tm = 0u; tm < TILE_M; tm += VEC_SIZE) {
                let global_row = output_row_base + tm;
                if (global_row < params.m) {
                    let dst_idx = global_col * params.y_stride + global_row;
                    dst[dst_idx/VEC_SIZE] = pack_acc_tile(&acc, tn, tm);
                }
            }
        }
    }
}
