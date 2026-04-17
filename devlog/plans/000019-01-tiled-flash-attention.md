# Plan: tiled flash-attention rewrite

## Thinking

Pre-approved by the user in the profile report §7. Attribution data from
§8 (PR #19) locks the target:

- **Prefill @ p=4096:** `attn_kernel` = 1,413,205 µs, 80% of 1,767,590 µs
  total. 2227 tok/s baseline on LFM2.5-VL-450M-Q4_0, M1 Max.
- **Decode @ ctx=4096:** `attn_kernel` = 54% share (from prior report).

Per §8.3: **if `attn_kernel` GPU time at p=4096 doesn't drop ≥40%,
abandon the rewrite.** That's the go/no-go gate.

### Three kernels, three situations

- `attention_prefill.metal` — batched prefill, already tiled flash w/
  online softmax. 80% of p=4096 time. **Primary target.**
- `attention.metal` — classic decode, seq_len ≤ 4096. Highly optimized
  already (all 256 threads score in parallel, simdgroup-distributed V
  with half4 loads). Not the bottleneck at ctx=128/1024, accounts for
  54% at ctx=4096 but only as a function of seq_len growth — per-step
  absolute time is fine. **Secondary target.**
- `flash_attention.metal` — decode-path fallback for seq_len > 4096.
  Currently 14% slower than classic per prior memory; fixed correctness
  in PR #18 but the perf rewrite is still outstanding. Very small blast
  radius (who runs ctx > 4096 decode often?), but it's a prerequisite
  for raising the classic auto-switch threshold or eventually unifying
  with the prefill kernel. **Tertiary.**

### Why the prefill kernel has headroom despite being "already flash"

Inspecting `attention_prefill.metal`:

- `N_THREADS = 128` (4 simdgroups). `attention.metal` uses 256 (8 sg).
- `C = 32` (KV chunk). Small, so the outer chunk loop runs many times
  for long seqs (128 chunks for p=4096).
- V accumulation is scalar: `v_sum += scores[q*C + t] * kv_tile[t*hd + d]`
  — one scalar MAC per thread per timestep. Classic decode's Phase 3
  does simdgroup-distributed half4 V loads from device memory.
- Outer `for (q=0..n_q)` serializes the softmax+V work per query inside
  each chunk — 8 queries × (max reduction + exp + V accum + state
  update) = 8 sets of barriers per chunk. Classic has no such outer
  serialization (it's single-query).

Bandwidth math (for sanity): p=4096, 16 heads, C=32, hd=64. Each chunk
reads (K+V) = 2×32×64×2 = 8 KB from device (half). 128 chunks/TG × 512
TGs = 65,536 chunk-loads × 8 KB = 512 GB. M1 Max bandwidth ≈ 400 GB/s
→ 1.28s memory-bound for 48 attn_prefill dispatches (one per attn layer
per chunk of 512 queries). Measured 1.43s. **Kernel is ~90% of
memory-bound already for its K/V tile staging.** So the wins aren't
free bandwidth; they're in:

1. **Reducing redundant K/V reads** by using larger Q_PER_TG and/or C
   (more scoring work per KV byte loaded).
2. **Improving compute efficiency inside the tile** via vectorized V
   (this doesn't save bandwidth but keeps compute hidden under the
   memory wait — useful if the current kernel is compute-bound at the
   per-lane level).
3. **Removing outer-`q` serialization** by parallelizing softmax state
   updates across queries rather than through 8 sequential iterations.

### Iterations (narrow-then-widen, gated on measurements)

#### Iteration 1 — Larger C, simdgroup-distributed V, 256 threads

Scope: `attention_prefill.metal` only. No Rust-side changes (same
dispatch, same buffer layout, same params struct). Just rewrite the
kernel body.

Changes:
- `N_THREADS = 256`, `NSG = 8` (matches classic decode).
- `C = 128` (4× larger KV chunk → 4× fewer outer-chunk iterations,
  less barrier overhead).
- Q_PER_TG stays at 8 (Q_PER_TG=16 blows the TG memory budget at C=128).
- Simdgroup-distributed V accumulation matching classic Phase 3:
  - Per-query: each simdgroup owns a contiguous slice of the C
    timesteps (C/8 = 16 each).
  - Each lane owns `dims_per_lane = head_dim/32` output dims (2 for
    hd=64).
  - Half4 V loads from device memory directly (not staged through
    kv_tile).
  - Per-lane partials reduced across simdgroups via threadgroup scratch,
    then combined with the running `out_tg` via the softmax rescale.
- Keep K tile staging: K is read once per chunk and reused across 8
  queries. Worth the threadgroup memory.
- Keep the outer `for q in 0..n_q` loop for now — the softmax state is
  per-query and parallelizing its update across queries is a bigger
  refactor.

TG memory check for C=128, hd=64, Q_PER_TG=8:
- q_tg: 8×64 = 512
- kv_tile (K only now — V from device): 128×64 = 8192
- scores: 8×128 = 1024
- out_tg: 8×64 = 512
- state: 16
- sg_val: 8
- partials scratch: 8 × 64 = 512
- **Total: 10,776 floats = 43 KB** — over M1 Max's 32 KB per-SM limit.

Fix: either drop K tile staging (read K from device too) or shrink C.
**C=64 (still 2× current)**:
- kv_tile: 64×64 = 4096
- scores: 8×64 = 512
- Total: 5648 floats = 22.6 KB ✓

So Iteration 1 uses C=64, not 128. Still doubles the chunk size.

Expected wins:
- 2× fewer outer chunk iterations → 2× fewer barrier sweeps.
- All 256 threads doing score work (vs 128) → 2× better thread util
  in the score phase.
- Simdgroup-distributed V → vectorized half4 device loads on V.

#### Iteration 2 — If Iter 1 clears the 40% bar, ship. Otherwise:

Attempt parallelizing the per-query softmax update. Rewrite inner
outer-`q` loop so all queries are processed in parallel across simd-
groups/lanes. This is a deeper rewrite; defer until Iter 1 data is in.

#### Iteration 3 — Decode kernel unification

Rewrite `flash_attention.metal` as a tiled extension of `attention.metal`:
- Single-tile path (seq_len ≤ TILE_K): identical to classic (correction
  factor = 1, zero overhead).
- Multi-tile path (seq_len > TILE_K): online softmax across tiles.
- This lets us remove the MAX_SEQ_LEN=4096 cap on the classic kernel
  without perf regression.

Defer — this only helps decode > 4096, which isn't the money phase.
Track as separate plan if Iter 1 succeeds.

### Test strategy

- Correctness: new `wick/tests/attention_prefill_parity.rs` ignored test
  that generates random Q/K/V at p=2048, runs both the current
  attention_prefill and (somehow) a reference — actually the cheaper
  option is end-to-end greedy-decode parity against a saved reference
  output. Follow the same pattern as
  `wick/tests/attention_metal_parity.rs`.

- CI tests: extend `wick/tests/metal_shaders_parity.rs` with a
  synthetic-input test against a scalar CPU reference. The existing
  `test_classic_vs_flash_attention_synthetic` pattern applies.

- Perf: measure on `LFM2.5-VL-450M-Q4_0` at p=128, 1024, 4096 (bench)
  AND GPU-timestamp attribution at p=4096 (via
  `WICK_PROFILE=gpu test_profile_longctx_2_5_450m_n4096`). Compare:
  - Baseline: 2227 tok/s, attn_kernel=1,413,205 µs.
  - Iter 1 target (pass): ≥ 40% drop in attn_kernel → ≤ 848k µs,
    prefill ≥ 3200 tok/s expected.
  - Absolute target: ≥ 4500 tok/s at p=4096 (≥ 2× baseline, §7 goal).

### What this does *not* do

- No Rust dispatch changes. Buffer layout, param struct, threadgroup
  count computation all identical.
- No new WICK_ATTN modes or env vars.
- No changes to classic `attention.metal` or `flash_attention.metal`
  beyond what Iter 3 (if reached) requires.
- No KV-cache layout changes (stays f16 in device memory).
- No cross-layer optimization (not fusing attention with FFN).

## Plan

### Step 0 — Worktree set up

Done — `worktrees/tiled-flash-attn`, branch `perf/tiled-flash-attn`
off `origin/main` (d0a7727).

### Step 1 — Baseline captured

Done — 2227 tok/s prefill, attn_kernel 1,413,205 µs @ p=4096 on
LFM2.5-VL-450M-Q4_0, M1 Max.

### Step 2 — Iteration 1: rewrite attention_prefill.metal

- Bump N_THREADS 128→256, NSG 4→8.
- Bump C 32→64.
- Remove V tile staging; add simdgroup-distributed half4 V accumulation
  from device memory.
- Keep everything else the same (dispatch, params, buffer layout).

Verify builds. If the dispatch caller in `metal_lfm2.rs` computed the
old `smem_bytes` based on 32 C, update the smem_bytes formula for C=64.

### Step 3 — Correctness tests

- Run `wick/tests/attention_metal_parity.rs` (existing end-to-end
  greedy-decode parity vs classic) to confirm no regression.
- Add a synthetic CI test in `wick/tests/metal_shaders_parity.rs` if
  one doesn't exist for `attention_prefill`.
- If tests fail, debug before benching.

### Step 4 — Perf measurement

- `wick bench` at p=128, 1024, 4096 (5 runs, warmup 2, --device metal,
  --no-cache).
- `WICK_PROFILE=gpu test_profile_longctx_2_5_450m_n4096` for per-category.
- Decide: ≥ 40% attn_kernel drop → ship Iter 1. Else → either (a)
  attempt Iter 2 (per-query parallelism), or (b) abandon and record
  why.

### Step 5 — Commit, PR

Standard workflow — devlog, plan, commit, push, PR with the perf table.

## Verification

- `cargo fmt --check`, `cargo clippy -- -D warnings` clean.
- Non-longctx tests pass (`cargo test --workspace`).
- `test_classic_vs_flash_attention_parity` passes (end-to-end).
- Prefill p=128 tok/s within 5% of baseline (no regression on short
  prompts).
- Attn_kernel GPU time at p=4096 ≤ 0.6× baseline.

## Out of scope

- Decode kernel rewrite (Iter 3, separate plan).
- Cross-layer attention fusion.
- New dtype for KV cache.
- Any Rust-side API surface changes.
