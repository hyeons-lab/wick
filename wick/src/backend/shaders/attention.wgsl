// Fused attention for one head:
//   scores[t] = dot(q_head, k_cache[t]) * scale
//   softmax(scores)
//   out[d] = sum_t(scores[t] * v_cache[t, d])
//
// One workgroup per head. Workgroup size 256.
// Supports seq_len up to 65536 (256 threads × 256 elements each).
//
// Bind group 0:
//   @binding(0) q: array<f32>          (all heads concatenated, read)
//   @binding(1) k_cache: array<f32>    (seq_len × kv_dim, read)
//   @binding(2) v_cache: array<f32>    (seq_len × kv_dim, read)
//   @binding(3) out: array<f32>        (all heads concatenated, read-write)
//   @binding(4) scores_buf: array<f32> (scratch, seq_len per head)
//   @binding(5) params: array<u32>     (n_heads, n_kv_heads, head_dim, kv_dim, seq_len, scale_bits)
//
// Dispatch: (n_heads, 1, 1) — one workgroup per head

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k_cache: array<f32>;
@group(0) @binding(2) var<storage, read> v_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<storage, read_write> scores_buf: array<f32>;
@group(0) @binding(5) var<storage, read> params: array<u32, 8>;

var<workgroup> shared_val: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn attention(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let head = wid.x;
    let tid = lid.x;
    let n_heads = params[0];
    let n_kv_heads = params[1];
    let head_dim = params[2];
    let kv_dim = params[3];
    let seq_len = params[4];
    let scale = bitcast<f32>(params[5]);
    let group_size = n_heads / n_kv_heads;
    let kv_head = head / group_size;
    let kv_h_offset = kv_head * head_dim;
    let q_offset = head * head_dim;

    // Per-head scores scratch area
    let scores_offset = head * seq_len;

    // ── Phase 1: Q×K scores ──────────────────────────────────────────────
    // Each thread computes scores for a subset of timesteps
    var t = tid;
    while t < seq_len {
        var dot: f32 = 0.0;
        let k_base = t * kv_dim + kv_h_offset;
        for (var d = 0u; d < head_dim; d += 1u) {
            dot += q[q_offset + d] * k_cache[k_base + d];
        }
        scores_buf[scores_offset + t] = dot * scale;
        t += 256u;
    }
    workgroupBarrier();

    // ── Phase 2: Softmax over scores ─────────────────────────────────────
    // Find max
    var local_max: f32 = -3.402823e+38;
    t = tid;
    while t < seq_len {
        local_max = max(local_max, scores_buf[scores_offset + t]);
        t += 256u;
    }
    shared_val[tid] = local_max;
    workgroupBarrier();

    if tid < 128u { shared_val[tid] = max(shared_val[tid], shared_val[tid + 128u]); }
    workgroupBarrier();
    if tid < 64u { shared_val[tid] = max(shared_val[tid], shared_val[tid + 64u]); }
    workgroupBarrier();
    if tid < 32u { shared_val[tid] = max(shared_val[tid], shared_val[tid + 32u]); }
    workgroupBarrier();
    if tid < 16u { shared_val[tid] = max(shared_val[tid], shared_val[tid + 16u]); }
    workgroupBarrier();
    if tid < 8u { shared_val[tid] = max(shared_val[tid], shared_val[tid + 8u]); }
    workgroupBarrier();
    if tid < 4u { shared_val[tid] = max(shared_val[tid], shared_val[tid + 4u]); }
    workgroupBarrier();
    if tid < 2u { shared_val[tid] = max(shared_val[tid], shared_val[tid + 2u]); }
    workgroupBarrier();
    if tid < 1u { shared_val[tid] = max(shared_val[tid], shared_val[tid + 1u]); }
    workgroupBarrier();
    let max_val = shared_val[0];

    // Exp + sum
    var partial_sum: f32 = 0.0;
    t = tid;
    while t < seq_len {
        let e = exp(scores_buf[scores_offset + t] - max_val);
        scores_buf[scores_offset + t] = e;
        partial_sum += e;
        t += 256u;
    }
    shared_val[tid] = partial_sum;
    workgroupBarrier();

    if tid < 128u { shared_val[tid] += shared_val[tid + 128u]; }
    workgroupBarrier();
    if tid < 64u { shared_val[tid] += shared_val[tid + 64u]; }
    workgroupBarrier();
    if tid < 32u { shared_val[tid] += shared_val[tid + 32u]; }
    workgroupBarrier();
    if tid < 16u { shared_val[tid] += shared_val[tid + 16u]; }
    workgroupBarrier();
    if tid < 8u { shared_val[tid] += shared_val[tid + 8u]; }
    workgroupBarrier();
    if tid < 4u { shared_val[tid] += shared_val[tid + 4u]; }
    workgroupBarrier();
    if tid < 2u { shared_val[tid] += shared_val[tid + 2u]; }
    workgroupBarrier();
    if tid < 1u { shared_val[tid] += shared_val[tid + 1u]; }
    workgroupBarrier();
    let inv_sum = 1.0 / shared_val[0];

    // Normalize scores
    t = tid;
    while t < seq_len {
        scores_buf[scores_offset + t] *= inv_sum;
        t += 256u;
    }
    workgroupBarrier();

    // ── Phase 3: Weighted V sum ──────────────────────────────────────────
    // Each thread computes a subset of output dimensions
    var d = tid;
    while d < head_dim {
        var val: f32 = 0.0;
        for (var tt = 0u; tt < seq_len; tt += 1u) {
            val += scores_buf[scores_offset + tt] * v_cache[tt * kv_dim + kv_h_offset + d];
        }
        out[q_offset + d] = val;
        d += 256u;
    }
}
