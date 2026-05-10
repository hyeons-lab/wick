
// Register-tiled matmul kernel: dst = src0 * src1
//
// src0 (weights): [m, k]
// src1 (activations): [k, n]
// dst (output): [m, n]
//
// Tiling strategy:
// Each workgroup handles (WORKGROUP_SIZE_M * TILE_M) rows of src0
// and (WORKGROUP_SIZE_N * TILE_N) columns of src1.
// Each thread handles a (TILE_M x TILE_N) block of the output.

#define BYTE_HELPERS
#include "common_decls.tmpl"
#include "mul_mat_decls.tmpl"

#ifdef VEC
fn store_val(acc: ptr<function, array<array<f32, TILE_N>, TILE_M>>, tn: u32, tm: u32) -> vec4<f32> {
    return vec4<f32>(f32((*acc)[tm][tn]), f32((*acc)[tm + 1u][tn]), f32((*acc)[tm + 2u][tn]), f32((*acc)[tm + 3u][tn]));
}
#else
fn store_val(acc: ptr<function, array<array<f32, TILE_N>, TILE_M>>, tn: u32, tm: u32) -> f32 {
    return f32((*acc)[tm][tn]);
}
#endif

struct MulMatParams {
    m: u32,
    k: u32,
    n: u32,
    x_stride: u32,
    y_stride: u32,
    batch_stride_x: u32,
    batch_stride_y: u32,
    batch_stride_w: u32,
};

@group(0) @binding(0) var<storage, read> src0: array<SRC0_TYPE>;
@group(0) @binding(1) var<storage, read> src1: array<SRC1_TYPE>;
@group(0) @binding(2) var<storage, read_write> dst: array<DST_TYPE>;
@group(0) @binding(3) var<storage, read> params: MulMatParams;

const TOTAL_WORKGROUP_SIZE = WORKGROUP_SIZE_M * WORKGROUP_SIZE_N;
const TILE_SRC0_SHMEM: u32 = TILE_K * WORKGROUP_SIZE_M * TILE_M;
const TILE_SRC1_SHMEM: u32 = TILE_K * WORKGROUP_SIZE_N * TILE_N;

var<workgroup> shmem: array<f32, TILE_SRC0_SHMEM + TILE_SRC1_SHMEM>;

@compute @workgroup_size(TOTAL_WORKGROUP_SIZE)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>
) {
    let thread_id = local_id.x;
    let local_m = thread_id % WORKGROUP_SIZE_M;
    let local_n = thread_id / WORKGROUP_SIZE_M;

    let wg_m_count = (params.m + WORKGROUP_SIZE_M * TILE_M - 1u) / (WORKGROUP_SIZE_M * TILE_M);
    let wg_n_count = (params.n + WORKGROUP_SIZE_N * TILE_N - 1u) / (WORKGROUP_SIZE_N * TILE_N);
    let wg_per_matrix = wg_m_count * wg_n_count;

    // Support 2D dispatch for large grids
    let wg_linear = wg_id.y * num_wg.x + wg_id.x;
    let batch_idx = wg_linear / wg_per_matrix;
    let wg_in_batch = wg_linear % wg_per_matrix;

    let wg_m = wg_in_batch % wg_m_count;
    let wg_n = wg_in_batch / wg_m_count;

    let output_row_base = wg_m * WORKGROUP_SIZE_M * TILE_M + local_m * TILE_M;
    let output_col_base = wg_n * WORKGROUP_SIZE_N * TILE_N + local_n * TILE_N;

    let src0_batch_offset = batch_idx * params.batch_stride_w;
    let src1_batch_offset = batch_idx * params.batch_stride_x;
    let dst_batch_offset = batch_idx * params.batch_stride_y;

    let offset_m = wg_m * WORKGROUP_SIZE_M * TILE_M;
    let offset_n = wg_n * WORKGROUP_SIZE_N * TILE_N;

    var acc: array<array<f32, TILE_N>, TILE_M>;
    // Initialize accumulator
    for (var tm = 0u; tm < TILE_M; tm++) {
        for (var tn = 0u; tn < TILE_N; tn++) {
            acc[tm][tn] = f32(0.0);
        }
    }

    for (var k_outer = 0u; k_outer < params.k; k_outer += TILE_K) {
        init_shmem_src0(thread_id, src0_batch_offset, offset_m, k_outer);
        init_shmem_src1(thread_id, src1_batch_offset, offset_n, k_outer);

        workgroupBarrier();

        let k_end = min(TILE_K, params.k - k_outer);

        for (var k_inner = 0u; k_inner < k_end; k_inner++) {
            var src0_tile: array<f32, TILE_M>;
            for (var tm = 0u; tm < TILE_M; tm++) {
                let src0_m = local_m * TILE_M + tm;
                let src0_idx = k_inner + src0_m * TILE_K;
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

    // Write back
    for (var tn = 0u; tn < TILE_N; tn++) {
        let global_col = output_col_base + tn;
        if (global_col < params.n) {
            for (var tm = 0u; tm < TILE_M; tm += VEC_SIZE) {
                let global_row = output_row_base + tm;
                if (global_row < params.m) {
                    let dst_idx = dst_batch_offset + global_col * params.m + global_row;
                    dst[dst_idx/VEC_SIZE] = store_val(&acc, tn, tm);
                }
            }
        }
    }
}
