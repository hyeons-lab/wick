// F32 GEMV: y[m] = A[m, k] × x[k]
//
// 8 rows per workgroup, 32 threads (1 subgroup). x is loaded once per WG
// and reused across 8 rows — 8× less x bandwidth than 1-row-per-WG.
//
// Dispatch: (ceil(m/8), 1, 1) workgroups

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> params: vec2<u32>;

const NR: u32 = 8u;

@compute @workgroup_size(32, 1, 1)
fn gemv_f32(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let m = params.x;
    let k = params.y;
    let tid = lid.x;
    let r0 = (wid.x + wid.y * 65535u) * NR;

    var sums: array<f32, 8>;
    for (var r = 0u; r < NR; r += 1u) {
        sums[r] = 0.0;
    }

    // Each thread strides through k in steps of 32.
    var col = tid;
    while col < k {
        let xv = x[col];
        for (var r = 0u; r < NR; r += 1u) {
            sums[r] += a[(r0 + r) * k + col] * xv;
        }
        col += 32u;
    }

    // Subgroup reduction.
    for (var r = 0u; r < NR; r += 1u) {
        let total = subgroupAdd(sums[r]);
        if tid == 0u && r0 + r < m {
            y[r0 + r] = total;
        }
    }
}
