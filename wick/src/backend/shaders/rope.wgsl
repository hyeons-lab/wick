// RoPE: Rotary Position Embedding applied to Q and K vectors.
//
// Each thread handles one (cos, sin) pair for one dimension pair in one head.
// Applied to both Q (n_heads) and K (n_kv_heads) concatenated in the same buffer:
//   q_and_k[0..n_heads*head_dim] = Q
//   q_and_k[n_heads*head_dim..] = K (only first n_kv_heads*head_dim used)
//
// Bind group 0:
//   @binding(0) q: array<f32>       (read-write, Q vector)
//   @binding(1) k: array<f32>       (read-write, K vector)
//   @binding(2) params: array<u32>  (pos, n_heads, n_kv_heads, head_dim, freq_base_bits)
//
// Dispatch: (ceil(max(n_heads, n_kv_heads) * head_dim/2 / 256), 1, 1)

@group(0) @binding(0) var<storage, read_write> q: array<f32>;
@group(0) @binding(1) var<storage, read_write> k: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32, 5>;

@compute @workgroup_size(256, 1, 1)
fn rope(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pos = params[0];
    let n_heads = params[1];
    let n_kv_heads = params[2];
    let head_dim = params[3];
    let freq_base = bitcast<f32>(params[4]);

    let half_dim = head_dim / 2u;
    let idx = gid.x;

    // Apply to Q heads
    let q_total = n_heads * half_dim;
    if idx < q_total {
        let head = idx / half_dim;
        let d = idx % half_dim;
        let freq = 1.0 / pow(freq_base, f32(2u * d) / f32(head_dim));
        let angle = f32(pos) * freq;
        let cos_a = cos(angle);
        let sin_a = sin(angle);

        let i0 = head * head_dim + d;
        let i1 = head * head_dim + d + half_dim;
        let x0 = q[i0];
        let x1 = q[i1];
        q[i0] = x0 * cos_a - x1 * sin_a;
        q[i1] = x0 * sin_a + x1 * cos_a;
    }

    // Apply to K heads
    let k_total = n_kv_heads * half_dim;
    if idx < k_total {
        let head = idx / half_dim;
        let d = idx % half_dim;
        let freq = 1.0 / pow(freq_base, f32(2u * d) / f32(head_dim));
        let angle = f32(pos) * freq;
        let cos_a = cos(angle);
        let sin_a = sin(angle);

        let i0 = head * head_dim + d;
        let i1 = head * head_dim + d + half_dim;
        let x0 = k[i0];
        let x1 = k[i1];
        k[i0] = x0 * cos_a - x1 * sin_a;
        k[i1] = x0 * sin_a + x1 * cos_a;
    }
}
