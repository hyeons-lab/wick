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
//   @binding(2) params: array<u32, 5>  (pos, n_heads, n_kv_heads, head_dim, freq_base_bits)
//
// Dispatch: (ceil(max(n_heads, n_kv_heads) * head_dim/2 / 256), 1, 1)

#include "common_decls.tmpl"

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
        let angle = rope_angle(pos, d, head_dim, freq_base);

        let i0 = head * head_dim + d;
        let i1 = head * head_dim + d + half_dim;
        let res = rotate_rope(q[i0], q[i1], angle);
        q[i0] = res.x;
        q[i1] = res.y;
    }

    // Apply to K heads
    let k_total = n_kv_heads * half_dim;
    if idx < k_total {
        let head = idx / half_dim;
        let d = idx % half_dim;
        let angle = rope_angle(pos, d, head_dim, freq_base);

        let i0 = head * head_dim + d;
        let i1 = head * head_dim + d + half_dim;
        let res = rotate_rope(k[i0], k[i1], angle);
        k[i0] = res.x;
        k[i1] = res.y;
    }
}

