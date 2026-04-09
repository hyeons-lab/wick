#include <metal_stdlib>
using namespace metal;

// Batched fused conv1d: processes N tokens in a single dispatch.
// Each thread handles one channel and loops over all N tokens sequentially
// (rolling state requires sequential processing per channel).
//
// This eliminates N-1 dispatch round-trips per conv layer.
// Dispatch: ceil(hidden_size / 256) threadgroups × 256 threads.

struct Params {
    uint hidden_size;
    uint kernel_size;
    uint d_conv;
    uint n_tokens;
    uint proj_stride;   // stride between token projections (floats), = 3*hs
    uint out_stride;    // stride between output vectors (floats), = hs
};

kernel void conv1d_fused_batch(
    const device float* proj [[buffer(0)]],    // [n_tokens × 3*hs]: (x, c, b) per token
    device float* rbuffer [[buffer(1)]],       // rolling conv state [d_conv × hs]
    const device float* weight [[buffer(2)]],  // conv weight [hs × kernel_size]
    device float* output [[buffer(3)]],        // output [n_tokens × hs]
    constant Params& params [[buffer(4)]],
    uint ch [[thread_position_in_grid]]
) {
    uint hs = params.hidden_size;
    uint ks = params.kernel_size;
    uint d_conv = params.d_conv;
    uint n_tokens = params.n_tokens;
    uint proj_stride = params.proj_stride;
    uint out_stride = params.out_stride;
    if (ch >= hs) return;

    // Pre-load the conv weight for this channel (small: kernel_size ≤ 4).
    float w[4]; // max kernel_size = 4
    for (uint k = 0; k <= d_conv && k < 4; k++) {
        w[k] = weight[ch * ks + k];
    }

    // Pre-load rolling buffer state for this channel.
    float rb[3]; // max d_conv = 3
    for (uint k = 0; k < d_conv && k < 3; k++) {
        rb[k] = rbuffer[k * hs + ch];
    }

    for (uint t = 0; t < n_tokens; t++) {
        uint base = t * proj_stride;
        // x is at proj[base + ch], c at proj[base + hs + ch], b at proj[base + 2*hs + ch]
        float x_val = proj[base + ch];
        float b_val = proj[base + 2 * hs + ch];
        float c_val = proj[base + hs + ch];

        float bx = x_val * b_val;

        // Conv1d with registers.
        float sum = 0.0f;
        for (uint k = 0; k < d_conv && k < 3; k++) {
            sum += rb[k] * w[k];
        }
        sum += bx * w[d_conv];

        // Shift rolling buffer in registers.
        if (d_conv > 1u) {
            for (uint k = 0; k < d_conv - 1u && k < 2; k++) {
                rb[k] = rb[k + 1];
            }
        }
        if (d_conv > 0u) {
            rb[d_conv - 1u] = bx;
        }

        // output = c * conv_out
        output[t * out_stride + ch] = c_val * sum;
    }

    // Write rolling buffer back.
    for (uint k = 0; k < d_conv && k < 3; k++) {
        rbuffer[k * hs + ch] = rb[k];
    }
}
