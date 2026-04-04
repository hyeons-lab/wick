// Q4_0 GEMV: y[m] = dequant(A_q4_0[m, k]) × x[k]
//
// Q4_0 block layout (18 bytes per 32 elements):
//   bytes 0-1:  f16 scale (delta)
//   bytes 2-17: 16 bytes of packed 4-bit nibbles (32 elements)
//     Each byte encodes 2 elements: lo nibble = elem[i], hi nibble = elem[i+16]
//   Dequantized: val = (nibble - 8) * delta
//
// Optimized: 128 threads/workgroup, process 8 nibbles per u32 word load.
//
// Dispatch: (min(m, 65535), ceil(m/65535), 1) — one workgroup per row

@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> params: vec2<u32>;

var<workgroup> shared_sums: array<f32, 128>;

@compute @workgroup_size(128, 1, 1)
fn gemv_q4_0(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let row = wid.x + wid.y * 65535u;
    let m = params.x;
    let k = params.y;
    let tid = lid.x;

    if row >= m { return; }

    let nb = k / 32u;
    // Each row = nb blocks × 18 bytes = nb × 4.5 u32 words
    // We'll read using byte offsets and convert to word indices
    let row_start_byte = row * nb * 18u;

    var partial_sum: f32 = 0.0;

    // Each thread processes blocks in stride-128 pattern
    var bi = tid;
    while bi < nb {
        let block_byte = row_start_byte + bi * 18u;
        let word_off = block_byte / 4u;
        let byte_rem = block_byte % 4u;

        // Read 5 consecutive u32 words to cover all 18 bytes of the block
        // (2 scale bytes + 16 nibble bytes, with up to 3 bytes of prefix alignment)
        let w0 = a[word_off];
        let w1 = a[word_off + 1u];
        let w2 = a[word_off + 2u];
        let w3 = a[word_off + 3u];
        let w4 = a[word_off + 4u];

        // Extract f16 scale from bytes [byte_rem, byte_rem+1]
        // Use unpack2x16float which reads 2 f16 values from a u32.
        // We need the f16 at offset byte_rem within the word sequence.
        var scale_bits: u32;
        if byte_rem == 0u {
            scale_bits = w0 & 0xFFFFu;
        } else if byte_rem == 1u {
            scale_bits = (w0 >> 8u) & 0xFFFFu;
        } else if byte_rem == 2u {
            scale_bits = (w0 >> 16u) & 0xFFFFu;
        } else {
            // byte_rem == 3: straddles w0 and w1
            scale_bits = ((w0 >> 24u) & 0xFFu) | ((w1 & 0xFFu) << 8u);
        }
        let delta = unpack2x16float(scale_bits).x;

        // Build a logical 16-byte sequence of nibbles starting at byte_rem+2
        // The nibble bytes are at positions [byte_rem+2 .. byte_rem+18] across w0-w4.
        // For efficiency, we reconstruct a 16-byte stream as 4 u32 words.
        let nib_start = byte_rem + 2u;
        var n0: u32;
        var n1: u32;
        var n2: u32;
        var n3: u32;
        if nib_start == 2u {
            // nibbles start at byte 2 of w0: [w0>>16|w1<<16, w1>>16|w2<<16, w2>>16|w3<<16, w3>>16|w4<<16]
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
            // nib_start == 5
            n0 = (w1 >> 8u) | (w2 << 24u);
            n1 = (w2 >> 8u) | (w3 << 24u);
            n2 = (w3 >> 8u) | (w4 << 24u);
            // For nib_start==5, we need byte 21 which requires w5
            n3 = (w4 >> 8u) | (a[word_off + 5u] << 24u);
        }

        // Each u32 contains 4 packed bytes = 8 nibbles
        // Process n0, n1, n2, n3 — total 32 elements (16 lo + 16 hi nibbles)
        let col_base = bi * 32u;

        // n0: bytes 0-3 of nibble data → elements 0-3 (lo) + 16-19 (hi)
        partial_sum += process_4bytes(n0, col_base, 0u, delta);
        partial_sum += process_4bytes(n1, col_base, 4u, delta);
        partial_sum += process_4bytes(n2, col_base, 8u, delta);
        partial_sum += process_4bytes(n3, col_base, 12u, delta);

        bi += 128u;
    }

    // Workgroup reduction (8 steps for 128 threads)
    shared_sums[tid] = partial_sum;
    workgroupBarrier();
    if tid < 64u { shared_sums[tid] += shared_sums[tid + 64u]; }
    workgroupBarrier();
    if tid < 32u { shared_sums[tid] += shared_sums[tid + 32u]; }
    workgroupBarrier();
    if tid < 16u { shared_sums[tid] += shared_sums[tid + 16u]; }
    workgroupBarrier();
    if tid < 8u { shared_sums[tid] += shared_sums[tid + 8u]; }
    workgroupBarrier();
    if tid < 4u { shared_sums[tid] += shared_sums[tid + 4u]; }
    workgroupBarrier();
    if tid < 2u { shared_sums[tid] += shared_sums[tid + 2u]; }
    workgroupBarrier();
    if tid < 1u { shared_sums[tid] += shared_sums[tid + 1u]; }
    workgroupBarrier();

    if tid == 0u {
        y[row] = shared_sums[0];
    }
}

// Process 4 packed nibble bytes: extract 8 nibbles, dequantize, multiply by x, accumulate.
// `word` contains 4 bytes, each with 2 nibbles.
// `col_base` is the base column index for this block.
// `offset` is the byte offset within the block (0, 4, 8, or 12).
fn process_4bytes(word: u32, col_base: u32, offset: u32, delta: f32) -> f32 {
    // Extract 4 bytes
    let b0 = word & 0xFFu;
    let b1 = (word >> 8u) & 0xFFu;
    let b2 = (word >> 16u) & 0xFFu;
    let b3 = (word >> 24u) & 0xFFu;

    // Each byte: lo nibble = element[offset+i], hi nibble = element[offset+i+16]
    let lo0 = (f32(b0 & 0xFu) - 8.0) * delta;
    let hi0 = (f32((b0 >> 4u) & 0xFu) - 8.0) * delta;
    let lo1 = (f32(b1 & 0xFu) - 8.0) * delta;
    let hi1 = (f32((b1 >> 4u) & 0xFu) - 8.0) * delta;
    let lo2 = (f32(b2 & 0xFu) - 8.0) * delta;
    let hi2 = (f32((b2 >> 4u) & 0xFu) - 8.0) * delta;
    let lo3 = (f32(b3 & 0xFu) - 8.0) * delta;
    let hi3 = (f32((b3 >> 4u) & 0xFu) - 8.0) * delta;

    return lo0 * x[col_base + offset + 0u]
         + lo1 * x[col_base + offset + 1u]
         + lo2 * x[col_base + offset + 2u]
         + lo3 * x[col_base + offset + 3u]
         + hi0 * x[col_base + offset + 16u + 0u]
         + hi1 * x[col_base + offset + 16u + 1u]
         + hi2 * x[col_base + offset + 16u + 2u]
         + hi3 * x[col_base + offset + 16u + 3u];
}
