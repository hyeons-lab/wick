// Q4_0 GEMV: y[m] = dequant(A_q4_0[m, k]) × x[k]
//
// Q4_0 block layout (18 bytes per 32 elements):
//   bytes 0-1:  f16 scale (delta)
//   bytes 2-17: 16 bytes of packed 4-bit nibbles (32 elements)
//
// Strategy: Each workgroup processes 4 output rows simultaneously.
// This reuses loaded x values across 4 rows, reducing x reads by 4x.
// 32 threads per workgroup (one subgroup), each thread processes blocks
// in stride-32 pattern. Subgroup reduction for final sum per row.
//
// Dispatch: (ceil(m / 4), 1, 1) workgroups

@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> params: vec2<u32>;

const ROWS_PER_WG: u32 = 4u;

@compute @workgroup_size(32, 1, 1)
fn gemv_q4_0(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let m = params.x;
    let k = params.y;
    let tid = lid.x;
    let row_base = wid.x * ROWS_PER_WG;

    let nb = k / 32u;
    let row_bytes = nb * 18u;

    // Partial sums for 4 rows
    var sum0: f32 = 0.0;
    var sum1: f32 = 0.0;
    var sum2: f32 = 0.0;
    var sum3: f32 = 0.0;

    // Each thread processes blocks in stride-32 pattern
    var bi = tid;
    while bi < nb {
        let col_base = bi * 32u;

        // Load x values for this block (shared across all 4 rows)
        // 32 elements = 16 lo positions + 16 hi positions
        var xl: array<f32, 32>;
        for (var i = 0u; i < 32u; i += 1u) {
            xl[i] = x[col_base + i];
        }

        // Process this block for each of the 4 rows
        sum0 += process_block(row_base + 0u, bi, row_bytes, &xl);
        sum1 += process_block(row_base + 1u, bi, row_bytes, &xl);
        sum2 += process_block(row_base + 2u, bi, row_bytes, &xl);
        sum3 += process_block(row_base + 3u, bi, row_bytes, &xl);

        bi += 32u;
    }

    // Subgroup reduction for each row
    let total0 = subgroupAdd(sum0);
    let total1 = subgroupAdd(sum1);
    let total2 = subgroupAdd(sum2);
    let total3 = subgroupAdd(sum3);

    if tid == 0u {
        if row_base + 0u < m { y[row_base + 0u] = total0; }
        if row_base + 1u < m { y[row_base + 1u] = total1; }
        if row_base + 2u < m { y[row_base + 2u] = total2; }
        if row_base + 3u < m { y[row_base + 3u] = total3; }
    }
}

fn process_block(row: u32, bi: u32, row_bytes: u32, xl: ptr<function, array<f32, 32>>) -> f32 {
    let block_byte = row * row_bytes + bi * 18u;
    let word_off = block_byte / 4u;
    let byte_rem = block_byte % 4u;

    // Load 5 u32 words covering 18 bytes + alignment
    let w0 = a[word_off];
    let w1 = a[word_off + 1u];
    let w2 = a[word_off + 2u];
    let w3 = a[word_off + 3u];
    let w4 = a[word_off + 4u];

    // Extract f16 scale
    var scale_bits: u32;
    if byte_rem == 0u {
        scale_bits = w0 & 0xFFFFu;
    } else if byte_rem == 1u {
        scale_bits = (w0 >> 8u) & 0xFFFFu;
    } else if byte_rem == 2u {
        scale_bits = (w0 >> 16u) & 0xFFFFu;
    } else {
        scale_bits = ((w0 >> 24u) & 0xFFu) | ((w1 & 0xFFu) << 8u);
    }
    let delta = unpack2x16float(scale_bits).x;

    // Reconstruct 16-byte nibble stream
    let nib_start = byte_rem + 2u;
    var n0: u32;
    var n1: u32;
    var n2: u32;
    var n3: u32;
    if nib_start == 2u {
        n0 = (w0 >> 16u) | (w1 << 16u);
        n1 = (w1 >> 16u) | (w2 << 16u);
        n2 = (w2 >> 16u) | (w3 << 16u);
        n3 = (w3 >> 16u) | (w4 << 16u);
    } else if nib_start == 3u {
        n0 = (w0 >> 24u) | (w1 << 8u);
        n1 = (w1 >> 24u) | (w2 << 8u);
        n2 = (w2 >> 24u) | (w3 << 8u);
        n3 = (w3 >> 24u) | (w4 << 8u);
    } else if nib_start == 4u {
        n0 = w1;
        n1 = w2;
        n2 = w3;
        n3 = w4;
    } else {
        n0 = (w1 >> 8u) | (w2 << 24u);
        n1 = (w2 >> 8u) | (w3 << 24u);
        n2 = (w3 >> 8u) | (w4 << 24u);
        n3 = (w4 >> 8u) | (a[word_off + 5u] << 24u);
    }

    // Compute dot product: 4 words × 4 bytes × 2 nibbles = 32 elements
    return dot_word(n0, 0u, xl, delta)
         + dot_word(n1, 4u, xl, delta)
         + dot_word(n2, 8u, xl, delta)
         + dot_word(n3, 12u, xl, delta);
}

fn dot_word(word: u32, offset: u32, xl: ptr<function, array<f32, 32>>, delta: f32) -> f32 {
    let b0 = word & 0xFFu;
    let b1 = (word >> 8u) & 0xFFu;
    let b2 = (word >> 16u) & 0xFFu;
    let b3 = (word >> 24u) & 0xFFu;

    let lo0 = (f32(b0 & 0xFu) - 8.0) * delta;
    let hi0 = (f32((b0 >> 4u) & 0xFu) - 8.0) * delta;
    let lo1 = (f32(b1 & 0xFu) - 8.0) * delta;
    let hi1 = (f32((b1 >> 4u) & 0xFu) - 8.0) * delta;
    let lo2 = (f32(b2 & 0xFu) - 8.0) * delta;
    let hi2 = (f32((b2 >> 4u) & 0xFu) - 8.0) * delta;
    let lo3 = (f32(b3 & 0xFu) - 8.0) * delta;
    let hi3 = (f32((b3 >> 4u) & 0xFu) - 8.0) * delta;

    return lo0 * (*xl)[offset + 0u]
         + lo1 * (*xl)[offset + 1u]
         + lo2 * (*xl)[offset + 2u]
         + lo3 * (*xl)[offset + 3u]
         + hi0 * (*xl)[offset + 16u + 0u]
         + hi1 * (*xl)[offset + 16u + 1u]
         + hi2 * (*xl)[offset + 16u + 2u]
         + hi3 * (*xl)[offset + 16u + 3u];
}
