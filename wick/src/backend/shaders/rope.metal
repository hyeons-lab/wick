#include <metal_stdlib>
using namespace metal;

// RoPE: rotary position embedding applied in-place to Q and K.
// Dispatch: ceil(max(n_heads, n_kv_heads) * head_dim/2 / 256) threadgroups × 256 threads.

struct Params {
    uint pos;
    uint n_heads;
    uint n_kv_heads;
    uint head_dim;
    uint freq_base_bits;
};

kernel void rope(
    device float* q [[buffer(0)]],
    device float* k [[buffer(1)]],
    constant Params& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint pos = params.pos;
    uint n_heads = params.n_heads;
    uint n_kv_heads = params.n_kv_heads;
    uint head_dim = params.head_dim;
    float freq_base = as_type<float>(params.freq_base_bits);
    uint half_dim = head_dim / 2u;

    // Q heads.
    uint q_total = n_heads * half_dim;
    if (gid < q_total) {
        uint head = gid / half_dim;
        uint d = gid % half_dim;
        float freq = 1.0f / powr(freq_base, float(2u * d) / float(head_dim));
        float angle = float(pos) * freq;
        float cos_a = cos(angle);
        float sin_a = sin(angle);
        uint i0 = head * head_dim + d;
        uint i1 = i0 + half_dim;
        float x0 = q[i0];
        float x1 = q[i1];
        q[i0] = x0 * cos_a - x1 * sin_a;
        q[i1] = x0 * sin_a + x1 * cos_a;
    }

    // K heads.
    uint k_total = n_kv_heads * half_dim;
    if (gid < k_total) {
        uint head = gid / half_dim;
        uint d = gid % half_dim;
        float freq = 1.0f / powr(freq_base, float(2u * d) / float(head_dim));
        float angle = float(pos) * freq;
        float cos_a = cos(angle);
        float sin_a = sin(angle);
        uint i0 = head * head_dim + d;
        uint i1 = i0 + half_dim;
        float x0 = k[i0];
        float x1 = k[i1];
        k[i0] = x0 * cos_a - x1 * sin_a;
        k[i1] = x0 * sin_a + x1 * cos_a;
    }
}
