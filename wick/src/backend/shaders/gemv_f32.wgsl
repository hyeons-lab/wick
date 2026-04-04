// F32 GEMV: y[m] = A[m, k] × x[k]
//
// Each workgroup computes one output row using parallel reduction:
// - 64 threads per workgroup, each accumulates over k/64 elements
// - Shared memory reduction to produce final dot product
//
// Bind group 0:
//   @binding(0) A: array<f32>  (row-major, m × k)
//   @binding(1) x: array<f32>  (k elements)
//   @binding(2) y: array<f32>  (m elements, read-write)
//   @binding(3) params: vec2<u32>  (m, k)
//
// Dispatch: (m, 1, 1) workgroups

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> params: vec2<u32>;

var<workgroup> shared_sums: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn gemv_f32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let row = wid.x;
    let m = params.x;
    let k = params.y;
    let tid = lid.x;

    if row >= m {
        return;
    }

    // Each thread accumulates a partial sum over its stripe of k
    let row_offset = row * k;
    var partial_sum: f32 = 0.0;

    var col = tid;
    while col < k {
        partial_sum += a[row_offset + col] * x[col];
        col += 64u;
    }

    // Store partial sum to shared memory
    shared_sums[tid] = partial_sum;
    workgroupBarrier();

    // Parallel reduction in shared memory (log2(64) = 6 steps)
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

    // Thread 0 writes the final result
    if tid == 0u {
        y[row] = shared_sums[0];
    }
}
