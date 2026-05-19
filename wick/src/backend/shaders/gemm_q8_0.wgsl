#define Q8_0_HELPERS
#include "common_decls.tmpl"

// Batched Q8_0 GEMM: output[token, row] = sum_k weight[row, k] * x[token, k].
//
// This mirrors the simple batched Q4_0 kernel shape: one workgroup computes
// 8 output rows for one token. It is intentionally conservative and exists to
// keep Q8_0 prefill on the batched path instead of falling back to per-token
// decode. The dequant math is shared with gemv_q8_0 via common_decls.tmpl
// (Q8_0_HELPERS); each thread stages its 32 activations once and reuses
// them across all 8 rows.
//
// Bind group 0:
//   @binding(0) a: array<u32>     (weights, Q8_0 packed: M rows x nb*34 bytes)
//   @binding(1) x: array<f32>     (activations, N tokens x x_stride floats)
//   @binding(2) y: array<f32>     (output,      N tokens x y_stride floats)
//   @binding(3) params: array<u32, 6>
//        (m, k, n, x_stride, y_stride, _pad)
//
// Dispatch: (ceil(m/8), n, 1) workgroups of 32 threads each. The row tile
// uses wid.x directly (no get_wid flattening, since wid.y carries the
// token axis); the host asserts ceil(m/8) <= 65535.
//
// Subgroup invariant: same as gemv_q8_0 — the 32-thread workgroup is
// finalized with subgroupAdd + a single tid==0 writer, correct only when
// all 32 lanes share one subgroup. GpuContext enforces
// min_subgroup_size >= 32 at init.

@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32, 6>;

const ROWS_PER_WG: u32 = 8u;

@compute @workgroup_size(32, 1, 1)
fn gemm_q8_0(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let m = params[0];
    let k = params[1];
    let x_stride = params[3];
    let y_stride = params[4];

    let tid = lid.x;
    let token = wid.y;
    let row_base = wid.x * ROWS_PER_WG;
    let nb = k / 32u;
    let row_bytes = nb * 34u;
    let token_base = token * x_stride;

    var sums: array<f32, 8>;
    for (var r = 0u; r < ROWS_PER_WG; r += 1u) {
        sums[r] = 0.0;
    }

    var bi = tid;
    while bi < nb {
        let x_base = token_base + bi * 32u;

        var xl: array<f32, 32>;
        for (var i = 0u; i < 32u; i += 1u) {
            xl[i] = x[x_base + i];
        }

        for (var r = 0u; r < ROWS_PER_WG; r += 1u) {
            let row = row_base + r;
            if row < m {
                sums[r] += process_block_q8_0(row, bi, row_bytes, &xl);
            }
        }

        bi += 32u;
    }

    let y_base = token * y_stride;
    for (var r = 0u; r < ROWS_PER_WG; r += 1u) {
        let total = subgroupAdd(sums[r]);
        if tid == 0u && row_base + r < m {
            y[y_base + row_base + r] = total;
        }
    }
}
