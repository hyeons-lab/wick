# Plan: batched-prefill profile instrumentation

## Thinking

The follow-up called out in `benchmarks/profile_longctx.md` §7 ("recommended
next plan") asks for two small instrumentation changes so the attribution
story for prefill is as clean as decode's was in the original report. The
motivation (paraphrased from the report):

1. `WICK_PROFILE=noattn` currently only skips the decode-path dispatches
   (`encode_attention`, `encode_attention_q_offset`). Batched prefill lives
   in a third function with an inline `attention_prefill` dispatch inside
   `forward_prefill_inner` — `noattn` has no effect there. The noattn
   cross-check in the original report measured ~0% attention share at
   p=1024 and p=4096 purely because the skip did not fire.
2. `WICK_PROFILE=timing` (i.e. `forward_prefill_profiled_inner`) commits
   and waits once per phase, which serializes GPU work and inflates absolute
   shares. A GPU-timestamp variant that uses one command buffer with
   `sample_counters_in_buffer` attachments per phase would give us
   dispatch-overhead-free numbers — same pattern as
   `encode_layers_gpu_timed` for decode.

Scope is intentionally tight: no kernel changes, no new benchmark
dimensions, no refactor. We want a clean baseline to measure against
before the tiled-flash-attention rewrite the report pre-approved.

### 1. noattn for prefill

The attention dispatch lives in `forward_prefill_inner` (around lines
2463-2494 in `metal_lfm2.rs` — verified before editing) and does:

- set `attention_prefill` pipeline
- bind Q / K cache / V cache / out
- set bytes
- set threadgroup memory
- dispatch

Skipping all of that when `WICK_PROFILE=noattn` leaves `prefill_proj_buf`
holding the projected/RoPE'd Q input, while `prefill_normed_buf`
(the attention kernel's output slot) keeps the stale RMSNorm'd hidden
state from Phase 1. Downstream consumers (attn output projection, FFN)
read that stale data and produce garbage, but that's already the
contract of `noattn` on the decode path: the output logits are
meaningless, the timing is what we want.

One cache for the env var at the top of the function keeps the check
cheap in a per-layer hot loop (16 layers × 1 lookup = 16 syscalls
currently, vs 1 per forward with caching).

### 2. GPU timestamps for prefill

Cleanest integration: mirror `forward_prefill_profiled_inner` into a
sibling `forward_prefill_profiled_gpu_inner` that uses one command
buffer with `gpu_sampled_pass`-style encoder splits and returns the same
`Vec<(String, f64)>` shape so `aggregate_prefill_phases` in bench_perf
works unchanged.

`forward_prefill_profiled` (the public chunking wrapper) picks between
the two inner functions based on `WICK_PROFILE`:
- `gpu` → new GPU-timestamps path
- anything else (including `timing`) → existing CPU-wall-clock path

The chunking wrapper's buffer-bounds assertion already gates both
paths, so no duplication there.

Sample buffer capacity: 16 layers × 9 phases + 1 output = 145 phases,
290 timestamp indices. Allocate 512 to keep headroom, same as the
existing global gpu_timer.

### 3. Tests + data capture

- `cargo test -p wick --test bench_perf test_profile_longctx_2_5_450m_n4096 -- --ignored --nocapture`
  with and without `WICK_PROFILE=noattn` → confirms the noattn share
  jumps from ~0% to a realistic fraction of prefill time.
- `WICK_PROFILE=gpu cargo test ... test_profile_longctx_2_5_450m_n4096 -- --ignored --nocapture`
  → confirms per-category GPU-tick totals are within 15% of the CPU-
  wall-clock `timing` numbers (the report's sanity bar). Discrepancies >
  15% indicate per-phase-sync overhead in the timing path, which is
  the whole reason we're adding the GPU variant.

No new ignored tests are needed — the existing `test_profile_longctx_*`
suite already routes through `forward_prefill_profiled`, and the new
GPU variant is selected by env var, not by a new test entry point.

### 4. What this does *not* do

- No flash/attention kernel rewrite (separate plan, gated on this data).
- No changes to decode-path profiling — already correct per §4 of the
  report.
- No new WICK_PROFILE values. `gpu` and `timing` and `noattn` are the
  existing ones; we just make them meaningful for prefill.
- No changes to `aggregate_prefill_phases` in bench_perf.rs — the new
  GPU path emits the same `L{layer}_{phase}` label shape.

## Plan

### Step 0 — Worktree setup

Done (this file is inside `worktrees/profile-instrumentation`).

### Step 1 — noattn for batched prefill

Edit `wick/src/model/metal_lfm2.rs`:
- In `forward_prefill_inner`, cache `WICK_PROFILE` env var once near the
  top of the function.
- Wrap the `attention_prefill` dispatch (Phase C block, ~lines 2463-2494)
  in `if !noattn { ... }`.

### Step 2 — GPU-timestamps variant of prefill-profiled

Edit `wick/src/model/metal_lfm2.rs`:
- Add `forward_prefill_profiled_gpu_inner` that mirrors
  `forward_prefill_profiled_inner` but:
  - Single `CommandBufferRef` for all phases.
  - Per-phase `gpu_sampled_pass`-style encoder with start/end sample
    attachments (implement a local helper; don't reuse the global
    `GpuTimer` since its label storage is cross-forward).
  - Returns `(String, f64_us)` by resolving sample deltas and converting
    via the calibrated `ns_per_tick`.
- Update `forward_prefill_profiled` to dispatch on
  `WICK_PROFILE=gpu` to the new variant, keeping the existing wrapper's
  chunking + bounds assertion.

### Step 3 — Run + report

- `just fmt && just clippy && cargo test -p wick -- --skip _longctx` (fast
  CI-equivalent).
- Collect before/after for p=1024 and p=4096 on the 450M model:
  - `WICK_PROFILE=noattn wick bench ...` vs regular bench.
  - `WICK_PROFILE=gpu cargo test ... test_profile_longctx_2_5_450m_n4096 -- --ignored`
    vs `WICK_PROFILE=timing ...`.
- Append a "Post-instrumentation attribution" section to
  `benchmarks/profile_longctx.md` with the real noattn share and the
  GPU-vs-timing sanity check.

### Step 4 — Commit, push, PR

- Commits (expected, may be merged before push):
  - `perf(profile): enable noattn for batched prefill path`
  - `perf(profile): add GPU-timestamp variant of forward_prefill_profiled`
  - `docs(bench): record post-instrumentation attribution for long context`
- Single push, draft PR if the numbers surprise, non-draft otherwise.

## Verification

- `cargo fmt --check` clean.
- `just clippy` clean.
- Non-longctx tests pass.
- `WICK_PROFILE=noattn` at p=4096 now produces a measurably lower
  tok/s (or attn_share > 20%, which the decode data predicts for
  prefill too). If still ~0%, the guard didn't fire — investigate.
- `WICK_PROFILE=gpu` at p=4096 reports per-category µs/tok within 15%
  of `WICK_PROFILE=timing` on the same phases. If not, note it in the
  appended section but don't block the PR.

## Out of scope

- Flash-attention rewrite (follow-up plan, gated on this data).
- Any kernel or shader changes.
- New benchmark dimensions or new bench_perf tests.
- CPU-backend profiling parity.
