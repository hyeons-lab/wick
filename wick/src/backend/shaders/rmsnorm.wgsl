// RMSnorm: x = x / rms(x) * weight
//
// Single-workgroup approach for hidden_size ≤ 8192.
// Phase 1: parallel sum of squares → Phase 2: normalize + scale.
//
// Bind group 0:
//   @binding(0) x: array<f32>       (read-write, normalized in-place)
//   @binding(1) weight: array<f32>  (read-only, per-element scale)
//   @binding(2) params: vec4<u32>   (n, eps_bits, 0, 0)
//
// Dispatch: (1, 1, 1) — single workgroup

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> params: vec4<u32>;

var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn rmsnorm(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let n = params.x;
    let eps = bitcast<f32>(params.y);

    // Phase 1: each thread computes partial sum of squares
    var partial: f32 = 0.0;
    var i = tid;
    while i < n {
        let v = x[i];
        partial += v * v;
        i += 256u;
    }
    shared_sum[tid] = partial;
    workgroupBarrier();

    // Parallel reduction
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

    // Phase 2: normalize x[i] = x[i] * inv_rms * weight[i]
    let rms = sqrt(shared_sum[0] / f32(n) + eps);
    let inv_rms = 1.0 / rms;

    i = tid;
    while i < n {
        x[i] = x[i] * inv_rms * weight[i];
        i += 256u;
    }
}
