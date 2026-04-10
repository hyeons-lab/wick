// Depthwise Conv1d with rolling buffer.
//
// For each channel:
//   out[ch] = sum_k(buffer[k, ch] * weight[ch, k]) + input[ch] * weight[ch, d_conv]
//
// Then updates rolling buffer: shift left, append input.
//
// Bind group 0:
//   @binding(0) input: array<f32>   (hidden_size, bx = b ⊙ x from in_proj)
//   @binding(1) buffer: array<f32>  (d_conv × hidden_size, rolling buffer, read-write)
//   @binding(2) weight: array<f32>  (hidden_size × kernel_size, conv weights)
//   @binding(3) output: array<f32>  (hidden_size, conv output, write)
//   @binding(4) params: vec4<u32>   (hidden_size, kernel_size, d_conv, 0)
//
// Dispatch: (ceil(hidden_size / 256), 1, 1)

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> buffer: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<storage, read> params: vec4<u32>;

@compute @workgroup_size(256, 1, 1)
fn conv1d_depthwise(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ch = gid.x;
    let hidden_size = params.x;
    let kernel_size = params.y;
    let d_conv = params.z;

    if ch >= hidden_size { return; }

    // Convolution: sum over rolling buffer slots + current input
    var sum: f32 = 0.0;
    for (var k_idx = 0u; k_idx < d_conv; k_idx += 1u) {
        sum += buffer[k_idx * hidden_size + ch] * weight[ch * kernel_size + k_idx];
    }
    sum += input[ch] * weight[ch * kernel_size + d_conv];
    output[ch] = sum;

    // Update rolling buffer: shift left by one slot, append input
    // (This is sequential per-channel but each channel is independent)
    if d_conv > 1u {
        for (var k_idx = 0u; k_idx < d_conv - 1u; k_idx += 1u) {
            buffer[k_idx * hidden_size + ch] = buffer[(k_idx + 1u) * hidden_size + ch];
        }
    }
    if d_conv > 0u {
        buffer[(d_conv - 1u) * hidden_size + ch] = input[ch];
    }
}
