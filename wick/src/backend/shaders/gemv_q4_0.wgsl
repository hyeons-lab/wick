// Q4_0 GEMV: y[m] = dequant(A_q4_0[m, k]) × x[k]
//
// Q4_0 block layout (18 bytes per 32 elements):
//   bytes 0-1: f16 scale (delta)
//   bytes 2-17: 16 bytes of packed 4-bit nibbles
//   Each byte encodes 2 elements: lo nibble = elem[i], hi nibble = elem[i+16]
//   Dequantized: val = (nibble - 8) * delta
//
// Weight buffer is raw bytes accessed as array<u32>.
// Each row has nb = k/32 blocks, row_bytes = nb * 18.
//
// Bind group 0:
//   @binding(0) a: array<u32>     (quantized weights as u32 words)
//   @binding(1) x: array<f32>     (input vector, k elements)
//   @binding(2) y: array<f32>     (output vector, m elements, read-write)
//   @binding(3) params: vec2<u32> (m, k)
//
// Dispatch: (min(m, 65535), ceil(m/65535), 1) — one workgroup per row

@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> params: vec2<u32>;

var<workgroup> shared_sums: array<f32, 64>;

// Extract a u8 from a u32 word at byte position (0-3)
fn extract_byte(word: u32, byte_idx: u32) -> u32 {
    return (word >> (byte_idx * 8u)) & 0xFFu;
}

// Decode f16 from two bytes (little-endian)
fn decode_f16(lo: u32, hi: u32) -> f32 {
    let bits = lo | (hi << 8u);
    // Use bitcast via unpack2x16float which interprets u32 as two f16 values
    let pair = unpack2x16float(bits);
    return pair.x;
}

@compute @workgroup_size(64, 1, 1)
fn gemv_q4_0(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let row = wid.x + wid.y * 65535u;
    let m = params.x;
    let k = params.y;
    let tid = lid.x;

    if row >= m { return; }

    let nb = k / 32u;                    // blocks per row
    let block_bytes = 18u;               // Q4_0 block size in bytes
    let row_start_byte = row * nb * block_bytes;

    var partial_sum: f32 = 0.0;

    // Each thread processes blocks in stride-64 pattern
    var bi = tid;
    while bi < nb {
        // Byte offset of this block within the weight buffer
        let block_byte = row_start_byte + bi * block_bytes;
        // Convert to u32 word offset and remainder
        let word_off = block_byte / 4u;
        let byte_rem = block_byte % 4u;

        // Read f16 scale (2 bytes at block start)
        // The scale spans bytes [byte_rem, byte_rem+1] within u32 words
        let w0 = a[word_off];
        var scale_lo: u32;
        var scale_hi: u32;
        if byte_rem <= 2u {
            scale_lo = extract_byte(w0, byte_rem);
            scale_hi = extract_byte(w0, byte_rem + 1u);
        } else {
            // byte_rem == 3: scale straddles two u32 words
            scale_lo = extract_byte(w0, 3u);
            scale_hi = extract_byte(a[word_off + 1u], 0u);
        }
        let delta = decode_f16(scale_lo, scale_hi);

        // Read 16 bytes of nibbles starting at block_byte + 2
        let qs_byte = block_byte + 2u;
        let col_base = bi * 32u;

        for (var qi = 0u; qi < 16u; qi += 1u) {
            let abs_byte = qs_byte + qi;
            let qw = a[abs_byte / 4u];
            let qb = extract_byte(qw, abs_byte % 4u);

            let lo_nibble = qb & 0xFu;
            let hi_nibble = (qb >> 4u) & 0xFu;

            let val_lo = (f32(lo_nibble) - 8.0) * delta;
            let val_hi = (f32(hi_nibble) - 8.0) * delta;

            partial_sum += val_lo * x[col_base + qi];
            partial_sum += val_hi * x[col_base + qi + 16u];
        }

        bi += 64u;
    }

    // Workgroup reduction
    shared_sums[tid] = partial_sum;
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
