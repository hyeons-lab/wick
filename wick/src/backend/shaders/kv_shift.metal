#include <metal_stdlib>
using namespace metal;

// n_keep context shift on the f16 GPU KV cache. Two kernels here +
// reuse of memcpy_f16:
//
// 1. kv_shift_k_to_scratch: read each retained K cell at its OLD
//    position (n_keep + shift + t_off), apply RoPE delta R(-shift)
//    so the cell's stored angle matches its NEW position
//    (n_keep + t_off), and write to a scratch buffer at compact
//    offset (t_off). Mirror of the per-thread loop in
//    `InferenceState::shift_kv_with_rope` (CPU).
//
// 2. memcpy_f16: generic f16 element copy used to (a) move the
//    rotated K from scratch back into the cache at the new
//    n_keep-aligned offset, (b) move V cells through scratch to
//    the new offset (V isn't RoPE'd, just memmoved). Two-pass
//    via scratch is required because the source range
//    [(n_keep+shift)*kv_dim .. seq_len*kv_dim) and destination
//    [n_keep*kv_dim .. new_seq_len*kv_dim) overlap when
//    `shift < new_seq_len - n_keep`, which is the common case.
//    Metal compute kernels can't synchronize across the entire
//    grid, so an in-place per-thread read+write would race.
//
// RoPE convention matches `qk_norm_rope.metal`'s NeoX layout
// (pairs at [d, d + half_dim]), with `freq[d] = freq_base^(-2d/head_dim)`
// and rotation `(x0, x1) → (x0*c - x1*s, x0*s + x1*c)` where
// `c, s = cos(angle), sin(angle)` and `angle = delta_pos * freq[d]`.
// For the shift case `delta_pos = -shift` so the stored angle is
// reduced by `shift * freq[d]` per dim-pair — exactly what's needed
// for the cell to re-encode its new (smaller) position.

struct KParams {
    uint  n_keep;
    uint  shift;
    uint  new_seq_len;
    uint  n_kv_heads;
    uint  head_dim;
    uint  freq_base_bits;
    int   delta_pos;        // -(shift as i32)
    uint  _pad;
};

kernel void kv_shift_k_to_scratch(
    device const half* k_cache  [[buffer(0)]],
    device half*       scratch  [[buffer(1)]],
    constant KParams&  params   [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint half_dim = params.head_dim / 2u;
    uint retained = params.new_seq_len - params.n_keep;
    uint per_t = params.n_kv_heads * half_dim;
    uint total = retained * per_t;
    if (gid >= total) return;

    uint t_off = gid / per_t;
    uint hd = gid % per_t;
    uint h = hd / half_dim;
    uint d = hd % half_dim;

    uint kv_dim = params.n_kv_heads * params.head_dim;
    uint head_off = h * params.head_dim;

    uint t_old = params.n_keep + t_off + params.shift;
    uint src_i0 = t_old * kv_dim + head_off + d;
    uint src_i1 = src_i0 + half_dim;

    float x0 = float(k_cache[src_i0]);
    float x1 = float(k_cache[src_i1]);

    float freq_base = as_type<float>(params.freq_base_bits);
    // Mathematically equivalent to the forward-time RoPE expression
    // (`rope.metal` uses the same form; `qk_norm_rope*.metal` uses an
    // iterated `powr(theta_scale, d)` shape — different float ops, same
    // value in the limit). Composing this delta with whatever angle
    // the cell already encodes yields the new-position angle to within
    // the f16-storage round-trip error of the surrounding K cache.
    float freq = 1.0f / powr(freq_base, float(2u * d) / float(params.head_dim));
    float angle = float(params.delta_pos) * freq;
    float c = cos(angle);
    float s = sin(angle);

    float y0 = x0 * c - x1 * s;
    float y1 = x0 * s + x1 * c;

    uint dst_i0 = t_off * kv_dim + head_off + d;
    uint dst_i1 = dst_i0 + half_dim;
    scratch[dst_i0] = half(y0);
    scratch[dst_i1] = half(y1);
}

struct CopyParams {
    uint n_elements;
    uint src_offset_elements;
    uint dst_offset_elements;
    uint _pad;
};

// Generic f16 element-wise copy with src/dst element offsets.
// Used by the shift to (a) move rotated K from scratch back into
// the cache, (b) ferry V through scratch (no rotation).
kernel void memcpy_f16_offsets(
    device const half*    src    [[buffer(0)]],
    device half*          dst    [[buffer(1)]],
    constant CopyParams&  params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.n_elements) return;
    dst[params.dst_offset_elements + gid] = src[params.src_offset_elements + gid];
}
