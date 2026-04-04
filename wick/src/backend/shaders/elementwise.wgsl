// Element-wise operations on f32 buffers.
//
// Bind group 0:
//   @binding(0) a: array<f32>  (read-write, modified in-place)
//   @binding(1) b: array<f32>  (read-only)
//   @binding(2) params: vec2<u32>  (n, unused)
//
// Dispatch: (ceil(n / 256), 1, 1) workgroups

@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read> params: vec2<u32>;

@compute @workgroup_size(256, 1, 1)
fn add_inplace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = params.x;
    if i >= n { return; }
    a[i] = a[i] + b[i];
}

@compute @workgroup_size(256, 1, 1)
fn silu_mul_inplace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = params.x;
    if i >= n { return; }
    let g = a[i];
    a[i] = (g / (1.0 + exp(-g))) * b[i];
}
