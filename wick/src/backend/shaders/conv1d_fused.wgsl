// Single-token fused conv block for the LFM2 short-conv path:
//   bx = x * b
//   sum = Σ_k rbuffer[k, ch] * weight[ch, k] + bx * weight[ch, d_conv]
//   shift rbuffer left, append bx
//   output[ch] = c * sum
//
// Collapses three decode-time dispatches (mul1 + conv1d + mul2) and
// three encoder copies (extracting x/c/b from the conv-proj buffer
// into separate per-channel buffers) into one dispatch with zero
// copies. The shader reads x/c/b directly from `proj` at offsets
// 0/hs/2*hs — same layout as `conv1d_fused_batch.wgsl` with
// n_tokens = 1, so PR 2.C-full's batched prefill rewrite can mirror
// this body inside its tile loop.
//
// Mirrors `conv1d_fused.metal` (Metal port shipped in PR #145) and
// the per-token slice of `conv1d_fused_batch.wgsl` (PR #154).
//
// Constraints: kernel_size ≤ 4, d_conv ≤ 3 (LFM2 uses ks=4,
// d_conv=3). Out-of-range params return early — same OOB guard the
// batched twin uses.
//
// Bind group 0:
//   @binding(0) proj: array<f32>     (read; 3*hs floats, [x, c, b] flat)
//   @binding(1) rbuffer: array<f32>  (read-write; d_conv × hs)
//   @binding(2) weight: array<f32>   (read; hs × kernel_size)
//   @binding(3) output: array<f32>   (read-write; hs floats)
//   @binding(4) params: vec4<u32>    (hs, kernel_size, d_conv, _pad)
//
// Dispatch: (ceil(hs / 256), 1, 1) workgroups of 256 threads.

@group(0) @binding(0) var<storage, read> proj: array<f32>;
@group(0) @binding(1) var<storage, read_write> rbuffer: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<storage, read> params: vec4<u32>;

@compute @workgroup_size(256, 1, 1)
fn conv1d_fused(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ch = gid.x;
    let hs = params.x;
    let ks = params.y;
    let d_conv = params.z;

    if ch >= hs { return; }
    // Static guard: rbuffer/weight indexing assumes LFM2's
    // ks=4 / d_conv=3. Bail on host-side dispatches with
    // out-of-range params.
    if ks > 4u || d_conv > 3u { return; }

    let x_val = proj[ch];
    let c_val = proj[hs + ch];
    let b_val = proj[2u * hs + ch];
    let bx = x_val * b_val;

    // Conv accumulation over rolling buffer + current input.
    var sum: f32 = 0.0;
    for (var k_idx = 0u; k_idx < d_conv; k_idx += 1u) {
        sum += rbuffer[k_idx * hs + ch] * weight[ch * ks + k_idx];
    }
    sum += bx * weight[ch * ks + d_conv];

    // Update rolling buffer: shift left, append bx.
    if d_conv > 1u {
        for (var k_idx = 0u; k_idx < d_conv - 1u; k_idx += 1u) {
            rbuffer[k_idx * hs + ch] = rbuffer[(k_idx + 1u) * hs + ch];
        }
    }
    if d_conv > 0u {
        rbuffer[(d_conv - 1u) * hs + ch] = bx;
    }

    output[ch] = c_val * sum;
}
