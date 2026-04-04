// Softmax in-place: x[i] = exp(x[i] - max) / sum(exp(x - max))
//
// Single-workgroup: find max → exp+sum → normalize.
// Supports up to 256*256 = 65536 elements (covers typical seq_len).
//
// Bind group 0:
//   @binding(0) x: array<f32>     (read-write, softmax in-place)
//   @binding(1) params: vec2<u32> (n, 0)
//
// Dispatch: (1, 1, 1) — single workgroup

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> params: vec2<u32>;

var<workgroup> shared_val: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn softmax(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let n = params.x;

    // Phase 1: find max for numerical stability
    var local_max: f32 = -3.402823e+38; // -FLT_MAX
    var i = tid;
    while i < n {
        local_max = max(local_max, x[i]);
        i += 256u;
    }
    shared_val[tid] = local_max;
    workgroupBarrier();

    // Max reduction
    if tid < 128u { shared_val[tid] = max(shared_val[tid], shared_val[tid + 128u]); }
    workgroupBarrier();
    if tid < 64u { shared_val[tid] = max(shared_val[tid], shared_val[tid + 64u]); }
    workgroupBarrier();
    if tid < 32u { shared_val[tid] = max(shared_val[tid], shared_val[tid + 32u]); }
    workgroupBarrier();
    if tid < 16u { shared_val[tid] = max(shared_val[tid], shared_val[tid + 16u]); }
    workgroupBarrier();
    if tid < 8u { shared_val[tid] = max(shared_val[tid], shared_val[tid + 8u]); }
    workgroupBarrier();
    if tid < 4u { shared_val[tid] = max(shared_val[tid], shared_val[tid + 4u]); }
    workgroupBarrier();
    if tid < 2u { shared_val[tid] = max(shared_val[tid], shared_val[tid + 2u]); }
    workgroupBarrier();
    if tid < 1u { shared_val[tid] = max(shared_val[tid], shared_val[tid + 1u]); }
    workgroupBarrier();
    let max_val = shared_val[0];

    // Phase 2: exp(x - max) and partial sum
    var partial_sum: f32 = 0.0;
    i = tid;
    while i < n {
        let e = exp(x[i] - max_val);
        x[i] = e;
        partial_sum += e;
        i += 256u;
    }
    shared_val[tid] = partial_sum;
    workgroupBarrier();

    // Sum reduction
    if tid < 128u { shared_val[tid] += shared_val[tid + 128u]; }
    workgroupBarrier();
    if tid < 64u { shared_val[tid] += shared_val[tid + 64u]; }
    workgroupBarrier();
    if tid < 32u { shared_val[tid] += shared_val[tid + 32u]; }
    workgroupBarrier();
    if tid < 16u { shared_val[tid] += shared_val[tid + 16u]; }
    workgroupBarrier();
    if tid < 8u { shared_val[tid] += shared_val[tid + 8u]; }
    workgroupBarrier();
    if tid < 4u { shared_val[tid] += shared_val[tid + 4u]; }
    workgroupBarrier();
    if tid < 2u { shared_val[tid] += shared_val[tid + 2u]; }
    workgroupBarrier();
    if tid < 1u { shared_val[tid] += shared_val[tid + 1u]; }
    workgroupBarrier();
    let inv_sum = 1.0 / shared_val[0];

    // Phase 3: normalize
    i = tid;
    while i < n {
        x[i] = x[i] * inv_sum;
        i += 256u;
    }
}
