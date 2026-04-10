// NOTE: Uses subgroupAdd without `enable subgroups;` — see gemv_q4_0.wgsl.
//
// Q6_K GEMV — Metal kernel layout with per-byte reads.
// Correct but slow due to per-byte u32 load+shift+mask overhead (regresses
// vs f32 on macOS wgpu). NOT wired into production — kept for future
// optimization on targets where byte extraction is cheaper.
//
// NR=2 rows per WG, 32 threads. Dispatch: ceil(m/2).

@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> params: vec2<u32>;

const QK_K: u32 = 256u;
const Q6K_BYTES: u32 = 210u;
const NR: u32 = 2u;

fn rb(off: u32) -> u32 {
    return (a[off / 4u] >> ((off % 4u) * 8u)) & 0xFFu;
}

fn ri8(off: u32) -> i32 {
    let b = rb(off);
    return i32(b) - select(0, 256, (b & 0x80u) != 0u);
}

fn rf16(off: u32) -> f32 {
    let lo = rb(off);
    let hi = rb(off + 1u);
    return unpack2x16float(lo | (hi << 8u)).x;
}

@compute @workgroup_size(32, 1, 1)
fn gemv_q6_k(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let m = params.x;
    let k = params.y;
    let nb = k / QK_K;
    let row_bytes = nb * Q6K_BYTES;
    let tiisg = lid.x;
    let first_row = wid.x * NR;

    let tid_l = tiisg / 2u;
    let ix  = tiisg & 1u;
    let ip  = tid_l >> 3u;
    let il  = tid_l & 7u;
    let l0  = 4u * il;
    let is_off = 8u * ip + l0 / 16u;

    let y_offset   = 128u * ip + l0;
    let q_offset_l = 64u * ip + l0;
    let q_offset_h = 32u * ip + l0;

    var sumf0: f32 = 0.0;
    var sumf1: f32 = 0.0;

    var b = ix;
    while b < nb {
        let yb = b * QK_K + y_offset;
        var yl: array<f32, 16>;
        for (var l = 0u; l < 4u; l += 1u) {
            yl[4u * l + 0u] = x[yb + l + 0u];
            yl[4u * l + 1u] = x[yb + l + 32u];
            yl[4u * l + 2u] = x[yb + l + 64u];
            yl[4u * l + 3u] = x[yb + l + 96u];
        }

        for (var row = 0u; row < NR; row += 1u) {
            let bb = (first_row + row) * row_bytes + b * Q6K_BYTES;
            let ql1 = bb + q_offset_l;
            let ql2 = ql1 + 32u;
            let qh  = bb + 128u + q_offset_h;
            let sc  = bb + 192u + is_off;
            let d_off = bb + 208u;

            var sums = vec4<f32>(0.0);
            for (var l = 0u; l < 4u; l += 1u) {
                let q1 = rb(ql1 + l);
                let q2 = rb(ql2 + l);
                let qhv = rb(qh + l);

                let q6_1 = i32((q1 & 0x0Fu) | ((qhv & 0x03u) << 4u)) - 32;
                let q6_2 = i32((q2 & 0x0Fu) | ((qhv & 0x0Cu) << 2u)) - 32;
                let q6_3 = i32((q1 >> 4u)   | ( qhv & 0x30u        )) - 32;
                let q6_4 = i32((q2 >> 4u)   | ((qhv & 0xC0u) >> 2u )) - 32;

                sums[0] += yl[4u * l + 0u] * f32(q6_1);
                sums[1] += yl[4u * l + 1u] * f32(q6_2);
                sums[2] += yl[4u * l + 2u] * f32(q6_3);
                sums[3] += yl[4u * l + 3u] * f32(q6_4);
            }

            let dblk = rf16(d_off);
            let s0 = f32(ri8(sc));
            let s2 = f32(ri8(sc + 2u));
            let s4 = f32(ri8(sc + 4u));
            let s6 = f32(ri8(sc + 6u));
            let row_sum = dblk * (sums[0] * s0 + sums[1] * s2 + sums[2] * s4 + sums[3] * s6);

            if row == 0u { sumf0 += row_sum; }
            else { sumf1 += row_sum; }
        }
        b += 2u;
    }

    let total0 = subgroupAdd(sumf0);
    let total1 = subgroupAdd(sumf1);
    if tiisg == 0u {
        if first_row < m { y[first_row] = total0; }
        if first_row + 1u < m { y[first_row + 1u] = total1; }
    }
}
