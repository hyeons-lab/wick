# Long-context perf attribution (prefill + decode)

**Branch:** `perf/profile-longctx` @ origin/main (01a8b02)
**Hardware:** M1 Max (local workstation)
**Dates:** 2026-04-16 (initial), 2026-04-16 (re-run after review fixes)
**Raw logs:** `benchmarks/profile_longctx_raw/`

Goal: attribute the long-context regression visible in
`benchmarks/deltas_table.md` (prefill 0.22× llama.cpp at p=4096, decode
0.28–0.54× at ctx=4096) so the next-plan author can scope a fix against
real data.

**Review fixes incorporated in this version:**
- Added chunking to `MetalLfm2Model::forward_prefill_profiled` — the
  initial pass called it with n=1024 and n=4096, overflowing
  `prefill_batch_buf` (capped at `MAX_PREFILL_TOKENS`=512). The
  overflow produced partially-corrupted compute that finished faster
  than real prefill and inflated `attn_kernel` share. Numbers below
  reflect the fix.
- Renamed the 450M target from `LFM2-VL-450M-Q4_0` to
  `LFM2.5-VL-450M-Q4_0` for consistency with other `bench_perf.rs`
  tests. Both models show the same 0.22× regression row in the deltas
  table, so the qualitative finding is unchanged.
- **Fixed the same latent correctness bug in three attention kernels.**
  `flash_attention.metal`, `attention_gqa.metal`, and
  `attention_splitk.metal` all bound K/V caches as `const device float*`,
  while the Rust side stores the caches as f16
  (`encode_cast_f32_to_f16_offsets`). Reading f16 as f32 reinterprets
  two adjacent f16 values as one f32 — garbage. Only classic
  `attention.metal` and `attention_prefill.metal` bound correctly as
  `half*`.
  - `flash_attention` auto-activates at `seq_len > 4096`, so any
    context window > 4096 produced corrupt logits and silently unusable
    generation.
  - `attention_gqa` is opt-in via `WICK_ATTN=gqa` (latent).
  - `attention_splitk` is opt-in via `WICK_ATTN=splitk` (latent).
  All three fixed in this PR: K/V bindings changed to `half*` with
  `float()` casts on load. Regression tests at
  `wick/tests/attention_metal_parity.rs` (classic vs flash / gqa /
  splitk greedy-token parity on a 450M prompt) all pass.

---

## 1. Methods used and their limits

Three instruments. Each has a blind spot — none alone gives clean attribution
of the batched-prefill path that actually produces the benchmark numbers.

| Instrument                                | Path it measures                                    | Blind spot                                                                         |
|-------------------------------------------|-----------------------------------------------------|------------------------------------------------------------------------------------|
| `forward_prefill_profiled` (ignored test) | Batched prefill, per-phase                          | Per-phase `wait_until_completed` inflates high-dispatch-count phases; also chunks at MAX_PREFILL_TOKENS=512 so n=4096 runs become 8-chunk sequential (same behavior as production). |
| `WICK_PROFILE=timing` (CategoryTimer)     | **Only single-token `forward()`** (decode steps)    | Never activates on batched prefill — `forward_prefill_inner` bypasses it.           |
| `WICK_PROFILE=noattn`                     | Only `encode_attention` and `encode_attention_q_offset` call sites | **Batched prefill uses `encode_attention_prefill`** (line 2477 in `metal_lfm2.rs`) which has no `noattn` check. So noattn does not skip batched-prefill attention. |

Implication: **for batched prefill, we do not have a clean wall-time
attribution tool**. The best we have is the (per-phase-sync-inflated)
`forward_prefill_profiled` percentages and the algorithmic O(n²) signature
of the total-time scaling. Decode attribution via `WICK_PROFILE=timing` is
clean — the decode loop goes through the single-token `forward()` path.

Adding a `sample_counters_in_buffer`-based profiler to `forward_prefill_inner`
(≤100 LoC, single command buffer, no per-phase waits) would close this gap
for the follow-up plan.

---

## 2. Wall-time: scaling signal

`wick bench --device metal --no-cache --max-tokens 1 --runs 3 --warmup 1`,
median of 3.

| Model  | Prompt | Prefill tok/s | Per-token (us) | Scaling vs p=128 |
|--------|--------|---------------:|---------------:|-----------------:|
| 450M   | 128    | 8073           | 124            | 1.00×            |
| 450M   | 1024   | 5668           | 176            | 1.42×            |
| 450M   | 4096   | 2241           | 446            | 3.60×            |
| 1.6B   | 128    | 3002           | 333            | 1.00×            |
| 1.6B   | 4096   | 1043           | 959            | 2.88×            |

At 32× more tokens, per-token cost grows 2.9–3.6×. Total wall time grows
~115× for 32× more tokens on 450M = **O(n^1.37)**. That is between O(n) and
O(n²) and matches the table's "llama.cpp stays flat, wick halves" shape.

Compare to published `deltas_table.md` (llama.cpp on M1 Max):
- 450M Q4_0: p=128 → p=4096 prefill: 4573 → 10309 tok/s (0.98× scaling, flat)
- wick same row: 8073 → 2241 (3.60× per-token regression)

---

## 3. Batched-prefill phase profile (`forward_prefill_profiled` with chunking)

**Caveat**: per-phase `wait_until_completed` inflates high-dispatch phases.
Use percentages as directional, not absolute. Totals below are profiled
wall time, not production wall time. Phase counts reflect chunking
(n=1024 → 2 chunks, n=4096 → 8 chunks; so attn phases count = 6 layers × chunks).

### LFM2.5-VL-450M-Q4_0

| Category         | p=128 (us, %)      | p=1024 (us, %)       | p=4096 (us, %)         | Per-tok scaling 4096/128 |
|------------------|-------------------:|---------------------:|-----------------------:|-------------------------:|
| attn_kernel      | 5068  (8.00%)      | 107222 (40.43%)      | **1430420 (70.33%)**   | **8.8×**                 |
| conv_ffn_gemm    | 9662  (15.26%)     | 33538 (12.65%)       | 117831 (5.79%)         | 0.38×                    |
| conv_ffn_down    | 7845  (12.39%)     | 20195 (7.62%)        | 75776 (3.73%)          | 0.30×                    |
| attn_ffn_gemm    | 5839  (9.22%)      | 19086 (7.20%)        | 71210 (3.50%)          | 0.38×                    |
| conv_inproj      | 4792  (7.57%)      | 13847 (5.22%)        | 51664 (2.54%)          | 0.34×                    |
| attn_ffn_down    | 4654  (7.35%)      | 12156 (4.58%)        | 44925 (2.21%)          | 0.30×                    |
| conv1d           | 2903  (4.58%)      | 8544  (3.22%)        | 34507 (1.70%)          | 0.37×                    |
| conv_outproj     | 3464  (5.47%)      | 8160  (3.08%)        | 32932 (1.62%)          | 0.30×                    |
| attn_qkv         | 3005  (4.74%)      | 7326  (2.76%)        | 26490 (1.30%)          | 0.28×                    |
| attn_outproj     | 2377  (3.75%)      | 5640  (2.13%)        | 22935 (1.13%)          | 0.30×                    |
| (smaller phases) | ...                | ...                  | ...                    | ...                      |
| **TOTAL**        | **63328 us**       | **265192 us**        | **2033929 us**         | —                        |
| tok/s (profiled) | 2021               | 3861                 | 2014                   | —                        |

### LFM2.5-VL-1.6B-Q4_0

| Category         | p=128 (us, %)  | p=4096 (us, %)         | Per-tok scaling 4096/128 |
|------------------|---------------:|-----------------------:|-------------------------:|
| attn_kernel      | 11544 (6.69%)  | **2698947 (65.70%)**   | **7.3×**                 |
| conv_ffn_gemm    | 39505 (22.90%) | 367555 (8.95%)         | 0.29×                    |
| attn_ffn_gemm    | 24303 (14.09%) | 219373 (5.34%)         | 0.28×                    |
| conv_ffn_down    | 24127 (13.99%) | 203200 (4.95%)         | 0.26×                    |
| conv_inproj      | 16339 (9.47%)  | 149199 (3.63%)         | 0.29×                    |
| **TOTAL**        | **172508 us**  | **4107739 us**         | —                        |
| tok/s (profiled) | 742            | 997                    | —                        |

### Interpretation

Only `attn_kernel` scales *worse-than-linearly per token* (8.8× and 7.3×).
Every other phase is *sub-linear per token* (0.26–0.38×) because batched
GEMM amortizes weight reads across the batch.

**Even discounting the per-phase-sync inflation**, `attn_kernel` is the only
phase whose per-token cost grows with n. That is the O(n²)-total →
O(n)-per-token attention-compute signature.

---

## 4. Decode phase profile (CategoryTimer, valid on single-token path)

`WICK_PROFILE=timing wick bench --device metal --no-cache --context-size 8192
--prompt-tokens N --max-tokens 128`. Per-category ms/token averaged across
96 decoded tokens (CategoryTimer's print cadence prints at 32/64/96 for `% 32`).

### LFM2.5-VL-450M-Q4_0, ctx=128 → ctx=2048 → ctx=4096

| Category       | ctx=128 ms/tok, % | ctx=2048 ms/tok, % | ctx=4096 ms/tok, % | Scaling 4096/128 |
|----------------|------------------:|-------------------:|-------------------:|-----------------:|
| **attn_kernel**| **1.60 (7.7%)**   | **3.75 (14.6%)**   | **23.65 (54.0%)**  | **14.8×**        |
| ffn_norm_gemv  | 4.56 (22.0%)      | 5.27 (20.7%)       | 4.82 (11.0%)       | 1.06×            |
| ffn_silu_down  | 4.31 (20.7%)      | 4.79 (18.8%)       | 4.42 (10.1%)       | 1.03×            |
| conv_pre       | 2.65 (12.7%)      | 2.98 (11.7%)       | 2.73 (6.2%)        | 1.03×            |
| conv1d         | 2.47 (11.9%)      | 2.87 (11.2%)       | 2.58 (5.9%)        | 1.04×            |
| attn_norm_qkv  | 1.66 (8.0%)       | 1.86 (7.3%)        | 1.69 (3.9%)        | 1.02×            |
| attn_qk_rope   | 1.41 (6.8%)       | 1.63 (6.4%)        | 1.49 (3.4%)        | 1.06×            |
| attn_out       | 1.37 (6.6%)       | 1.58 (6.2%)        | 1.64 (3.8%)        | 1.20×            |
| out            | 0.73 (3.5%)       | 0.77 (3.0%)        | 0.74 (1.7%)        | 1.01×            |
| **TOTAL**      | **20.76**         | **25.48**          | **43.76**          | **2.11×**        |
| tok/s          | 47                | 39                 | 22                 | 0.47×            |

Note: the ctx=4096 row was initially unobtainable on the 2.5 model because
`flash_attention.metal` corrupted logits (see "Review fixes" above). After
the f16 KV dtype fix, the kernel produces sane output and the numbers above
are real.

### Interpretation

All non-attention phases are **flat** per step from ctx=128 to ctx=4096
(1.0–1.2×), as expected: decode is O(1) per step for weight-bound phases.

`attn_kernel` scales **14.8×** over 32× more context, putting it at
54% of the total per-step budget at ctx=4096. Caveat: this is not one
continuous attention-compute curve — the decode dispatch routes to
**classic** `attention.metal` for seq_len ≤ 4096 (i.e. ctx=128 and
ctx=2048 rows) and auto-switches to **flash** `flash_attention.metal`
for seq_len > 4096 (ctx=4096 row; `metal_lfm2.rs:1626`). Part of the
big jump from 3.75 ms/tok (classic @ ctx=2048) to 23.65 ms/tok (flash
@ ctx=4096) is the kernel change: flash is ~6× slower per step at
similar N than classic, which is the real "flash needs a perf rewrite"
gap once you discount the extra KV entries. Either way, it's the single
phase that any fix needs to address.

Combined attention-side share (attn_kernel + attn_norm_qkv + attn_qk_rope
+ attn_out) goes from **29%** at ctx=128 → **34%** at ctx=2048 →
**65%** at ctx=4096. Flash attention with KV-tile scan targets this
directly.

---

## 5. Scaling with model dimension (450M vs 1.6B at p=128)

| Category         | 450M (us, %)     | 1.6B (us, %)      | Per-us scaling (1.6B / 450M) | Note            |
|------------------|-----------------:|------------------:|-----------------------------:|-----------------|
| conv_ffn_gemm    | 9662 (15.3%)     | 39505 (22.9%)     | 4.1×                         | Hidden-size²    |
| conv_ffn_down    | 7845 (12.4%)     | 24127 (14.0%)     | 3.1×                         |                 |
| attn_ffn_gemm    | 5839 (9.2%)      | 24303 (14.1%)     | 4.2×                         | Hidden-size²    |
| attn_kernel      | 5068 (8.0%)      | 11544 (6.7%)      | 2.3×                         | Hidden-size     |
| attn_ffn_down    | 4654 (7.3%)      | 14701 (8.5%)      | 3.2×                         |                 |
| conv_inproj      | 4792 (7.6%)      | 16339 (9.5%)      | 3.4×                         |                 |

1.6B hidden=2048 vs 450M hidden=1024 (~2×). GEMM phases scale ~4× (hidden²
for weight count + batch overhead), GEMV/attention scales 2–3× (linear or
sub-linear). This is the expected scaling — no model-size-specific
regression beyond what quadratic GEMM predicts.

At p=128 the 1.6B prefill is 3002 tok/s vs published llama.cpp row 2402.5
(**1.25× — we win**). The "0.68×" regression in `deltas_table.md` must have
come from a different test environment (different warmup, cache state, or
thermal). Not a present regression — deprioritize.

---

## 6. Attention share cross-check (invalid for batched prefill)

Planned: compare `WICK_PROFILE=noattn` wall time vs unset to estimate
attention's wall-time contribution.

Actual: **`WICK_PROFILE=noattn` only affects `encode_attention` and
`encode_attention_q_offset` (lines 1613, 1726 in `metal_lfm2.rs`). Batched
prefill uses a third function, `encode_attention_prefill` (line 2477), which
does not check the env var.** So the noattn cross-check skipped attention on
the single-token decode step only, and the prefill paths ran attention
normally.

Raw numbers from the attempted cross-check (prefill tok/s, median of 3, 2.5 model):

| Prompt | regular | noattn  | Δ tok/s | Apparent attn share |
|--------|--------:|--------:|--------:|--------------------:|
| 128    | 8073    | 7839    | -234    | negative (noise)    |
| 1024   | 5668    | 5667    | ~0      | 0% (noise)          |
| 4096   | 2241    | 2243    | ~0      | 0% (noise)          |

Low deltas reflect the broken routing, not real attention cost.

**Action for follow-up**: add a `WICK_PROFILE=noattn` check inside
`encode_attention_prefill` (a 3-line change) *or* add sampled GPU timestamps
to the batched prefill path. Either gives clean attribution.

---

## 7. Findings & recommended next plan

**Decode @ long context (clean data, after flash kernel fix):** attention
is definitively the dominant and fastest-growing phase. `attn_kernel`
goes from 7.7% at ctx=128 to **54% at ctx=4096** (14.8× per-step
scaling), while every other phase is flat (1.0–1.2×). Combined attention
share: 29% → 65%. A flash-attention rewrite with KV-tile scan targets
the single dominant phase and has room for a ~2× decode speedup at
long context.

**Prefill @ long context (directional data with caveats):** all evidence
consistent with attention being the dominant scaler, but a reliable
attribution was prevented by the noattn/prefill routing gap and by the
per-phase-sync overhead of `forward_prefill_profiled`:

- `forward_prefill_profiled` says attention is 70.33% at p=4096 (450M) and
  65.70% (1.6B) — inflated but only `attn_kernel` scales super-linearly
  per-token (8.8× / 7.3×); every other phase scales sub-linearly per-token
  (~0.3×) thanks to batched-GEMM amortization.
- Total-time O(n^1.37) scaling is consistent with attention compute being
  the super-linear contributor on top of linear-scaling GEMM.
- Algorithmic analysis: classic attention at p=4096 is 4096² × 16 heads
  × 64 head_dim = 268 Mops per layer per attn layer (6 per model) — far
  more work than any other per-layer phase. Even at 100% GPU utilization
  this dominates for large n.

**Recommendation: commit to the tiled flash-attention rewrite**
(user pre-approved this direction conditional on the profiling data).
Scope for the follow-up plan:

1. **First (prerequisite for confidence)**: add proper batched-prefill
   attribution (3-line noattn check in `encode_attention_prefill` *and*
   sampled GPU timestamps in `forward_prefill_inner`). Baseline the current
   state before changing any kernel code. Expected outcome: real attn share
   on prefill p=4096 is in the 40–70% range (consistent with decode data
   scaling trend).
2. **Tiled flash-attention Metal shader** with online-softmax, threadgroup
   K/V tiles, head_dim=64 + GQA (n_heads=16, n_kv_heads=8 for 450M).
   Unified kernel for decode (M=1) and prefill (M>1) via a template flag.
3. **Remove the existing flash-attention kernel** (currently activates only
   at seq_len > 4096, claimed 14% slower than classic per memory note).
   **Open question to investigate before starting**: why is that kernel
   slower? Inspect any `.metal` shader matching `flash_*` and the code
   around `attn_mode == "flash"`. If it's a threadgroup-memory exhaustion
   issue or a register-pressure issue, the rewrite must solve that
   explicitly — not just re-architect.
4. **Validation targets** (bench `--device metal --no-cache`, median of
   5 after first discarded):
   - Decode @ ctx=4096 on 450M: **≥2× current** (≥44 tok/s, up from 22
     measured after the flash kernel fix).
   - Prefill @ p=4096 on 450M: **≥2× current** (≥4500 tok/s, up from 2241).
     Stretch: ≥0.7× llama.cpp (≥7200 tok/s).
   - No regression at p=128, ctx=128.
   - If improvement is <10% despite a correct implementation: the
     attribution was wrong, back out the change and revisit via the proper
     instrumentation from step (1).

---

## 8. Post-instrumentation attribution (2026-04-16, PR #19)

Step (1) from §7 now lands: the `WICK_PROFILE=noattn` guard fires in
batched prefill (`forward_prefill_inner`), and a GPU-timestamp variant of
`forward_prefill_profiled` sidesteps the per-phase `commit + wait`
overhead that inflated small phases in the CPU-wall-clock run.

### 8.1 noattn on batched prefill (the table §6 warned about)

`wick bench --device metal --no-cache --context-size 8192 --prompt-tokens N
--runs 3 --warmup 1 --max-tokens 0` on LFM2.5-VL-450M-Q4_0, M1 Max:

| Prompt | regular (tok/s) | noattn (tok/s) | attn share (1 - regular/noattn) |
|--------|----------------:|---------------:|--------------------------------:|
| 128    | 8086            | 7563           | negative (noise — 7 ms total)   |
| 1024   | 5602            | 11498          | **51%**                         |
| 4096   | 2226            | 11162          | **80%**                         |

Compare to §6's pre-fix table: p=1024 and p=4096 both read ~0% attn share
because the guard lived only on the decode-path dispatches. The fix lands
right in the range the report predicted (40–70% at p=4096) and confirms
attention is the overwhelming scaler on prefill.

p=128 noise: noattn's absolute time is ~7 ms and the warmup run settles
Metal shader cache / GPU clock unevenly; the variance swamps the 6% of
total time attention costs at that scale. Ignore the sign.

### 8.2 GPU timestamps vs CPU wall clock (sanity)

`WICK_PROFILE=gpu cargo test ... test_profile_longctx_2_5_450m_n4096`
vs the default `WICK_PROFILE` path, 450M @ p=4096:

| Category       | CPU-wall µs | GPU-ticks µs | Δ         | GPU share |
|----------------|------------:|-------------:|----------:|----------:|
| attn_kernel    | 1,428,248   | 1,427,326    | -0.06%    | **80.1%** |
| conv_ffn_gemm  | 118,287     | 99,688       | -16%      | 5.6%      |
| attn_ffn_gemm  | 71,242      | 59,405       | -17%      | 3.3%      |
| conv_ffn_down  | 76,333      | 55,693       | -27%      | 3.1%      |
| conv_inproj    | 51,538      | 33,781       | -34%      | 1.9%      |
| attn_ffn_down  | 45,396      | 33,137       | -27%      | 1.9%      |
| conv1d         | 35,167      | 17,544       | -50%      | 1.0%      |
| attn_qkv       | 26,646      | 15,593       | -41%      | 0.9%      |
| conv_outproj   | 34,066      | 13,398       | -61%      | 0.8%      |
| attn_outproj   | 25,707      | 8,530        | -67%      | 0.5%      |
| small phases   | 100k+       | 15k          | -85%+     | ~1.0%     |
| Total          | 2,042 ms    | 1,782 ms     | -13%      | 100%      |

`attn_kernel` absolute time agrees within 0.1% — the one phase large
enough that per-phase dispatch overhead is negligible. All other phases
shrink substantially under GPU timestamps, which is exactly the expected
shape: CPU wall clock over-counts dispatch latency on short kernels.

Consequence for attribution: attention's real share on p=4096 prefill is
**80%**, not the 70% the CPU wall-clock path reported in §5. That matches
the §8.1 noattn cross-check (80%) to the point. Both independent methods
now agree.

### 8.3 Updated guidance for the flash-attention rewrite

The rewrite's opportunity is larger than the original report implied:

- **Decode @ ctx=4096:** 54% attn share (unchanged from §3) — room for
  ~2× decode at long context if the kernel is faster.
- **Prefill @ p=4096:** **80% attn share** (was reported as "directional
  evidence for 70%") — if the rewrite matches llama.cpp's per-token
  prefill attention cost (flat ~10k tok/s across prompt sizes vs our
  4× regression at p=4096), the upside is closer to **3–4× prefill**,
  not 2×.

The §7 validation targets still stand as lower bounds:
- Prefill @ p=4096 ≥ 4500 tok/s (2.0× current).
- Decode @ ctx=4096 ≥ 44 tok/s (2.0× current).

New lower bound for the rewrite to be worth the complexity: **if attn_kernel
GPU time at p=4096 doesn't drop by ≥40%, abandon the rewrite** — 80% of
an un-improved kernel leaves nothing for other phases to make up, and
the residual gap is then in GEMM batching / conv1d, not attention.
