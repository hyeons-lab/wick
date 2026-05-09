// Argmax over an f32 array: writes the index of the maximum element to out[0].
//
// Single-workgroup, 256 threads. Each thread strides through `x` finding its
// local (max_val, max_idx); then a tree reduction over workgroup memory
// collapses to thread 0, which writes the global argmax. Tie-break: the
// lower index wins (matches CPU `cpu_argmax`'s `>` comparator + iter order).
//
// Bind group 0:
//   @binding(0) x: array<f32>     (read-only logits)
//   @binding(1) out: array<u32>   (read-write, len >= 1; writes out[0])
//   @binding(2) params: vec2<u32> (n, 0)
//
// Dispatch: (1, 1, 1) — single workgroup. n up to ~few hundred K is fine
// (typical vocab_size is 32K-200K); per-thread stride loop handles any n.

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
@group(0) @binding(2) var<storage, read> params: vec2<u32>;

var<workgroup> shared_val: array<f32, 256>;
var<workgroup> shared_idx: array<u32, 256>;

@compute @workgroup_size(256, 1, 1)
fn argmax_f32(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let n = params.x;

    // Phase 1: thread-local max with stride 256.
    // Init with -FLT_MAX so any real value wins on first compare.
    var local_max: f32 = -3.402823e+38;
    var local_idx: u32 = 0u;
    var i = tid;
    while i < n {
        let v = x[i];
        // Strict `>`: tie-break favors the lower index (the value already
        // recorded), matching `cpu_argmax`'s behavior.
        if v > local_max {
            local_max = v;
            local_idx = i;
        }
        i += 256u;
    }
    shared_val[tid] = local_max;
    shared_idx[tid] = local_idx;
    workgroupBarrier();

    // Tree reduction. Pair up halves: each thread keeps the (val, idx) of
    // whichever half has the larger value (lower idx on tie, same as above).
    var stride: u32 = 128u;
    loop {
        if tid < stride {
            let other_val = shared_val[tid + stride];
            let other_idx = shared_idx[tid + stride];
            if other_val > shared_val[tid] {
                shared_val[tid] = other_val;
                shared_idx[tid] = other_idx;
            } else if other_val == shared_val[tid] && other_idx < shared_idx[tid] {
                shared_idx[tid] = other_idx;
            }
        }
        workgroupBarrier();
        if stride == 1u { break; }
        stride = stride >> 1u;
    }

    if tid == 0u {
        out[0] = shared_idx[0];
    }
}
