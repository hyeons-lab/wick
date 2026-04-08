#include <metal_stdlib>
using namespace metal;

// Element-wise operations on f32 buffers.
// 256 threads per threadgroup. Dispatch ceil(n/256) threadgroups.

struct Params { uint n; uint _pad; };

kernel void memcpy_f32(
    const device float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    const device Params& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.n) return;
    dst[gid] = src[gid];
}

kernel void add_inplace(
    device float* a [[buffer(0)]],
    const device float* b [[buffer(1)]],
    const device Params& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.n) return;
    a[gid] = a[gid] + b[gid];
}

kernel void mul_inplace(
    device float* a [[buffer(0)]],
    const device float* b [[buffer(1)]],
    const device Params& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.n) return;
    a[gid] = a[gid] * b[gid];
}

// Out-of-place mul: dst[i] = a[i] * b[i]. Lets callers pass a/b with byte
// offsets into a bigger buffer, avoiding two prior memcpy dispatches.
kernel void mul_out(
    const device float* a [[buffer(0)]],
    const device float* b [[buffer(1)]],
    device float* dst [[buffer(2)]],
    const device Params& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.n) return;
    dst[gid] = a[gid] * b[gid];
}

// Cast f32 → f16 (for writing to f16 KV cache).
kernel void cast_f32_to_f16(
    const device float* src [[buffer(0)]],
    device half* dst [[buffer(1)]],
    const device Params& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.n) return;
    dst[gid] = half(src[gid]);
}

kernel void silu_mul_inplace(
    device float* a [[buffer(0)]],
    const device float* b [[buffer(1)]],
    const device Params& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.n) return;
    float g = a[gid];
    a[gid] = (g / (1.0f + exp(-g))) * b[gid];
}
