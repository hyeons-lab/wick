#define Q8_0_HELPERS
#include "common_decls.tmpl"

// Q8_0 GEMV: y[m] = dequant(A_q8_0[m, k]) * x[k].
//
// Q8_0 block layout and the dequant math live in common_decls.tmpl
// (Q8_0_HELPERS): get_u32_at + process_block_q8_0.
//
// One workgroup handles 8 rows. Threads stride over Q8 blocks, then
// subgroup-reduce one partial sum per output row.
//
// Subgroup invariant: the 32-thread workgroup is finalized with
// subgroupAdd + a single tid==0 writer, which is only correct if all 32
// lanes share one subgroup. GpuContext enforces min_subgroup_size >= 32
// at init.

@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> params: vec2<u32>;

const ROWS_PER_WG: u32 = 8u;

@compute @workgroup_size(32, 1, 1)
fn gemv_q8_0(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let m = params.x;
    let k = params.y;
    let tid = lid.x;
    let row_base = get_wid(wid) * ROWS_PER_WG;

    let nb = k / 32u;
    let row_bytes = nb * 34u;

    var sums: array<f32, 8>;
    for (var r = 0u; r < ROWS_PER_WG; r += 1u) {
        sums[r] = 0.0;
    }

    var bi = tid;
    while bi < nb {
        let col_base = bi * 32u;

        var xl: array<f32, 32>;
        for (var i = 0u; i < 32u; i += 1u) {
            xl[i] = x[col_base + i];
        }

        for (var r = 0u; r < ROWS_PER_WG; r += 1u) {
            let row = row_base + r;
            if row < m {
                sums[r] += process_block_q8_0(row, bi, row_bytes, &xl);
            }
        }

        bi += 32u;
    }

    for (var r = 0u; r < ROWS_PER_WG; r += 1u) {
        let total = subgroupAdd(sums[r]);
        if tid == 0u && row_base + r < m {
            y[row_base + r] = total;
        }
    }
}
