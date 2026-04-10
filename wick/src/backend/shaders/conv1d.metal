#include <metal_stdlib>
using namespace metal;

// Depthwise Conv1d with rolling buffer.
// Dispatch: ceil(hidden_size / 256) threadgroups × 256 threads.

struct Params { uint hidden_size; uint kernel_size; uint d_conv; uint _pad; };

kernel void conv1d_depthwise(
    const device float* input [[buffer(0)]],
    device float* rbuffer [[buffer(1)]],
    const device float* weight [[buffer(2)]],
    device float* output [[buffer(3)]],
    const device Params& params [[buffer(4)]],
    uint ch [[thread_position_in_grid]]
) {
    uint hs = params.hidden_size;
    uint ks = params.kernel_size;
    uint d_conv = params.d_conv;
    if (ch >= hs) return;

    float sum = 0.0f;
    for (uint k_idx = 0u; k_idx < d_conv; k_idx++) {
        sum += rbuffer[k_idx * hs + ch] * weight[ch * ks + k_idx];
    }
    sum += input[ch] * weight[ch * ks + d_conv];
    output[ch] = sum;

    // Shift rolling buffer left, append input.
    if (d_conv > 1u) {
        for (uint k_idx = 0u; k_idx < d_conv - 1u; k_idx++) {
            rbuffer[k_idx * hs + ch] = rbuffer[(k_idx + 1u) * hs + ch];
        }
    }
    if (d_conv > 0u) {
        rbuffer[(d_conv - 1u) * hs + ch] = input[ch];
    }
}
