// Batched attention prefill: N queries × n_heads in one dispatch.
//
// One workgroup per (head, query) pair, mirroring the structure of the
// per-query `attention.wgsl` but with:
//   - Per-query position for the causal mask (`pos_q = start_pos + q_idx`):
//     score for token t > pos_q is -∞ before softmax.
//   - Batched Q input (n_queries × q_stride floats, head h at offset
//     `q * q_stride + h * head_dim`).
//   - Batched output (n_queries × out_stride floats, same layout).
//   - Per-query scores scratch via `q_idx * (n_heads × max_seq) + ...`
//     so distinct workgroups don't stomp each other.
//
// Note on perf: this is the simplest correct batched attention — one
// workgroup per (head, query), no K/V tile reuse across queries.
// Metal's `attention_prefill.metal` shares K/V loads across Q_PER_TG=8
// queries via simdgroup matrix MMA; WGSL on naga-24 has no MMA
// intrinsic, so a tiled-shared-memory rewrite that approaches that
// shape is a focused follow-up perf PR. For now, this shader exists so
// PR 2.C-full has a working batched signature to wire forward_prefill
// against.
//
// Constraint (matches `attention.wgsl`): seq_len ≤ 65536 and
// `head_dim` is the per-head dimension implied by `params.head_dim`.
//
// Contract: caller MUST pass `max_seq ≥ start_pos + n_queries`. The K/V
// cache must contain valid entries for positions `[0, start_pos +
// n_queries)`, and `scores_buf` is sized `n_queries × n_heads × max_seq`.
// As a defensive belt against caller mismatches, the shader clamps
// `seq_len = min(pos_q + 1, max_seq)` — under-sized `max_seq` produces
// truncated (incorrect) attention rather than an OOB read, but the
// expected caller behavior is to size everything consistently.
//
// Bind group 0:
//   @binding(0) q_batch:    array<f32>    n_queries × q_stride floats
//   @binding(1) k_cache:    array<f32>    seq_len × kv_dim floats
//   @binding(2) v_cache:    array<f32>    seq_len × kv_dim floats
//   @binding(3) out_batch:  array<f32>    n_queries × out_stride floats
//   @binding(4) scores_buf: array<f32>    n_queries × n_heads × max_seq scratch
//   @binding(5) params:     array<u32, 12>
//        ( n_heads, n_kv_heads, head_dim, kv_dim, max_seq, scale_bits,
//          start_pos, n_queries, q_stride, out_stride, _pad0, _pad1 )
//
// Dispatch: (n_heads, n_queries, 1) workgroups of 256 threads.

@group(0) @binding(0) var<storage, read> q_batch: array<f32>;
@group(0) @binding(1) var<storage, read> k_cache: array<f32>;
@group(0) @binding(2) var<storage, read> v_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_batch: array<f32>;
@group(0) @binding(4) var<storage, read_write> scores_buf: array<f32>;
@group(0) @binding(5) var<storage, read> params: array<u32, 12>;

var<workgroup> shared_val: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn attention_prefill(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let head = wid.x;
    let q_idx = wid.y;
    let tid = lid.x;

    let n_heads = params[0];
    let n_kv_heads = params[1];
    let head_dim = params[2];
    let kv_dim = params[3];
    let max_seq = params[4];
    let scale = bitcast<f32>(params[5]);
    let start_pos = params[6];
    // params[7] (n_queries) is implicit in dispatch.
    let q_stride = params[8];
    let out_stride = params[9];

    // Causal seq_len for this query — attend over [0..pos_q]. Clamp
    // against `max_seq` so a caller passing inconsistent params can
    // only cause silent attention-window truncation, never an OOB read
    // of `k_cache` / `v_cache` / `scores_buf` (which are all sized to
    // `max_seq`-multiples).
    let pos_q = start_pos + q_idx;
    let seq_len = min(pos_q + 1u, max_seq);

    let group_size = n_heads / n_kv_heads;
    let kv_head = head / group_size;
    let kv_h_offset = kv_head * head_dim;
    let q_offset = q_idx * q_stride + head * head_dim;

    // Each (q_idx, head) workgroup gets its own scores slab so concurrent
    // workgroups don't collide. Layout: q_idx-major, then head, then time.
    let scores_offset = (q_idx * n_heads + head) * max_seq;

    // ─── Phase 1: Q × K scores with causal mask ────────────────────────────
    // Each thread computes scores for a stride of timesteps.
    var t = tid;
    while t < seq_len {
        var dot: f32 = 0.0;
        let k_base = t * kv_dim + kv_h_offset;
        for (var d = 0u; d < head_dim; d += 1u) {
            dot += q_batch[q_offset + d] * k_cache[k_base + d];
        }
        scores_buf[scores_offset + t] = dot * scale;
        t += 256u;
    }
    workgroupBarrier();

    // ─── Phase 2: softmax ──────────────────────────────────────────────────
    // Find max.
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

    // exp + partial sum
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

    // Normalize scores.
    t = tid;
    while t < seq_len {
        scores_buf[scores_offset + t] *= inv_sum;
        t += 256u;
    }
    workgroupBarrier();

    // ─── Phase 3: weighted V sum → output ──────────────────────────────────
    let out_offset = q_idx * out_stride + head * head_dim;
    var d = tid;
    while d < head_dim {
        var val: f32 = 0.0;
        for (var tt = 0u; tt < seq_len; tt += 1u) {
            val += scores_buf[scores_offset + tt] * v_cache[tt * kv_dim + kv_h_offset + d];
        }
        out_batch[out_offset + d] = val;
        d += 256u;
    }
}
