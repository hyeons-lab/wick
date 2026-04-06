// Fast Q4_0 GEMV ported from Metal/llama.cpp algorithm.
//
// Key optimizations vs gemv_q4_0.wgsl:
//  - 2 threads per block: each processes 16 elements (half block)
//  - Pre-scaled y values eliminate per-element bit shifts
//  - Sumy bias hoisting: delta * (sumy * -8 + acc)
//
// 32 threads, 4 rows per WG. Uses workgroup memory for reduction
// (no subgroup ops required for portability).
// Dispatch: ceil(m/4) workgroups.

@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> params: vec2<u32>;

const NR: u32 = 4u;
const NQ: u32 = 16u;

fn half_block_dot(blk_byte: u32, sumy: f32, yl: ptr<function, array<f32, 16>>, il: u32) -> f32 {
    let word_off = blk_byte / 4u;
    let byte_rem = blk_byte % 4u;

    // Extract f16 scale
    var scale_bits: u32;
    if byte_rem == 0u {
        scale_bits = a[word_off] & 0xFFFFu;
    } else if byte_rem == 1u {
        scale_bits = (a[word_off] >> 8u) & 0xFFFFu;
    } else if byte_rem == 2u {
        scale_bits = (a[word_off] >> 16u) & 0xFFFFu;
    } else {
        scale_bits = ((a[word_off] >> 24u) & 0xFFu) | ((a[word_off + 1u] & 0xFFu) << 8u);
    }
    let d = unpack2x16float(scale_bits).x;

    // Read u16 nibble pairs and apply pre-scaled y multiplies.
    // Metal uses u16 pointer: qs = &d_u16 + 1 + il/2. In bytes: +2 + il.
    let qs_byte = blk_byte + 2u + il;
    var acc0: f32 = 0.0;
    var acc1: f32 = 0.0;
    var acc2: f32 = 0.0;
    var acc3: f32 = 0.0;

    for (var qi = 0u; qi < 8u; qi += 2u) {
        let byte_pos = qs_byte + qi;
        let w_off = byte_pos / 4u;
        let w_rem = byte_pos % 4u;
        var q: u32;
        if w_rem <= 2u {
            q = (a[w_off] >> (w_rem * 8u)) & 0xFFFFu;
        } else {
            q = ((a[w_off] >> 24u) & 0xFFu) | ((a[w_off + 1u] & 0xFFu) << 8u);
        }

        acc0 += (*yl)[qi + 0u] * f32(q & 0x000Fu);
        acc1 += (*yl)[qi + 1u] * f32(q & 0x0F00u);
        acc2 += (*yl)[qi + 8u] * f32(q & 0x00F0u);
        acc3 += (*yl)[qi + 9u] * f32(q & 0xF000u);
    }

    return d * (sumy * -8.0 + acc0 + acc1 + acc2 + acc3);
}

@compute @workgroup_size(32, 1, 1)
fn gemv_q4_0_fast(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let m = params.x;
    let k = params.y;
    let nb = k / 32u;
    let row_bytes = nb * 18u;
    let r0 = wid.x * NR;
    let tid = lid.x;

    let ix = tid / 2u;
    let il = (tid & 1u) * 8u;

    var sumf: array<f32, 4>;
    sumf[0] = 0.0;
    sumf[1] = 0.0;
    sumf[2] = 0.0;
    sumf[3] = 0.0;

    var yl: array<f32, 16>;
    var yb_off: u32 = ix * 32u + il;

    var ib = ix;
    while ib < nb {
        var sumy0: f32 = 0.0;
        var sumy1: f32 = 0.0;
        for (var i = 0u; i < 8u; i += 2u) {
            sumy0 += x[yb_off + i + 0u] + x[yb_off + i + 1u];
            yl[i + 0u] = x[yb_off + i + 0u];
            yl[i + 1u] = x[yb_off + i + 1u] / 256.0;
            sumy1 += x[yb_off + i + 16u] + x[yb_off + i + 17u];
            yl[i + 8u] = x[yb_off + i + 16u] / 16.0;
            yl[i + 9u] = x[yb_off + i + 17u] / 4096.0;
        }
        let sumy_total = sumy0 + sumy1;

        for (var r = 0u; r < NR; r += 1u) {
            let blk_byte = (r0 + r) * row_bytes + ib * 18u;
            sumf[r] += half_block_dot(blk_byte, sumy_total, &yl, il);
        }

        yb_off += 32u * NQ;
        ib += NQ;
    }

    // Subgroup reduction (32 threads = 1 subgroup).
    let t0 = subgroupAdd(sumf[0]);
    let t1 = subgroupAdd(sumf[1]);
    let t2 = subgroupAdd(sumf[2]);
    let t3 = subgroupAdd(sumf[3]);
    if tid == 0u {
        if r0 + 0u < m { y[r0 + 0u] = t0; }
        if r0 + 1u < m { y[r0 + 1u] = t1; }
        if r0 + 2u < m { y[r0 + 2u] = t2; }
        if r0 + 3u < m { y[r0 + 3u] = t3; }
    }
}
