#include <metal_stdlib>
using namespace metal;

// Fused: bx = x * b → conv1d(bx, state) → output = c * conv_out
// Combines 3 dispatches into 1 per token per conv layer.
// Dispatch: ceil(hidden_size / 256) threadgroups × 256 threads.

struct Params {
    uint hidden_size;
    uint kernel_size;
    uint d_conv;
    uint _pad;
};

kernel void conv1d_fused(
    const device float* x [[buffer(0)]],       // proj[i*3*hs]     (x component)
    const device float* b [[buffer(1)]],       // proj[i*3*hs+2*hs] (b component)
    const device float* c [[buffer(2)]],       // proj[i*3*hs+hs]  (c component)
    device float* rbuffer [[buffer(3)]],       // rolling conv state
    const device float* weight [[buffer(4)]],  // conv weight
    device float* output [[buffer(5)]],        // result written here
    constant Params& params [[buffer(6)]],
    uint ch [[thread_position_in_grid]]
) {
    uint hs = params.hidden_size;
    uint ks = params.kernel_size;
    uint d_conv = params.d_conv;
    if (ch >= hs) return;

    // Step 1: bx = x * b
    float bx = x[ch] * b[ch];

    // Step 2: conv1d with rolling buffer
    float sum = 0.0f;
    for (uint k_idx = 0u; k_idx < d_conv; k_idx++) {
        sum += rbuffer[k_idx * hs + ch] * weight[ch * ks + k_idx];
    }
    sum += bx * weight[ch * ks + d_conv];

    // Update rolling buffer: shift left, append bx
    if (d_conv > 1u) {
        for (uint k_idx = 0u; k_idx < d_conv - 1u; k_idx++) {
            rbuffer[k_idx * hs + ch] = rbuffer[(k_idx + 1u) * hs + ch];
        }
    }
    if (d_conv > 0u) {
        rbuffer[(d_conv - 1u) * hs + ch] = bx;
    }

    // Step 3: output = c * conv_out
    output[ch] = c[ch] * sum;
}
