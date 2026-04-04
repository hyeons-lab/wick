// Per-head RMS normalization in-place.
//
// Input buffer layout: [n_heads × head_dim] flat.
// For each head h, normalize x[h*head_dim..(h+1)*head_dim] using shared weight.
//
// Dispatch: (n_heads, 1, 1) workgroups — one per head.
//
// Bind group 0:
//   @binding(0) x: array<f32>       (read-write, [n_heads × head_dim])
//   @binding(1) weight: array<f32>  (read-only, [head_dim] — shared across heads)
//   @binding(2) params: vec4<u32>   (head_dim, eps_bits, 0, 0)

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> params: vec4<u32>;

var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn per_head_rmsnorm(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let head = wid.x;
    let tid = lid.x;
    let head_dim = params.x;
    let eps = bitcast<f32>(params.y);
    let offset = head * head_dim;

    // Phase 1: partial sum of squares
    var partial: f32 = 0.0;
    var i = tid;
    while i < head_dim {
        let v = x[offset + i];
        partial += v * v;
        i += 256u;
    }
    shared_sum[tid] = partial;
    workgroupBarrier();

    // Reduction
    if tid < 128u { shared_sum[tid] += shared_sum[tid + 128u]; }
    workgroupBarrier();
    if tid < 64u { shared_sum[tid] += shared_sum[tid + 64u]; }
    workgroupBarrier();
    if tid < 32u { shared_sum[tid] += shared_sum[tid + 32u]; }
    workgroupBarrier();
    if tid < 16u { shared_sum[tid] += shared_sum[tid + 16u]; }
    workgroupBarrier();
    if tid < 8u { shared_sum[tid] += shared_sum[tid + 8u]; }
    workgroupBarrier();
    if tid < 4u { shared_sum[tid] += shared_sum[tid + 4u]; }
    workgroupBarrier();
    if tid < 2u { shared_sum[tid] += shared_sum[tid + 2u]; }
    workgroupBarrier();
    if tid < 1u { shared_sum[tid] += shared_sum[tid + 1u]; }
    workgroupBarrier();

    // Phase 2: normalize
    let rms = sqrt(shared_sum[0] / f32(head_dim) + eps);
    let inv_rms = 1.0 / rms;

    i = tid;
    while i < head_dim {
        x[offset + i] = x[offset + i] * inv_rms * weight[i];
        i += 256u;
    }
}
