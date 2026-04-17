# 000018 — perf/profile-instrumentation

**Agent:** Claude (claude-opus-4-7) @ wick branch perf/profile-instrumentation

## Intent

Close the two instrumentation gaps identified in `benchmarks/profile_longctx.md`
§7 so we can measure the true attention share of batched prefill before
committing to the tiled-flash-attention rewrite that's been pre-approved:

1. Make `WICK_PROFILE=noattn` skip the attention dispatch in the
   batched-prefill path (`forward_prefill_inner`), matching the decode-path
   behavior. Currently the guard lives only in `encode_attention` /
   `encode_attention_q_offset`, so the prefill cross-check in the original
   report measured ~0% attention share purely because the skip didn't fire.
2. Add a GPU-timestamp (`sample_counters_in_buffer`) variant of
   `forward_prefill_profiled` so per-category attribution is not distorted
   by the per-phase `commit + wait_until_completed` serialization overhead
   of the existing CPU-wall-clock path.

## Plan

See `devlog/plans/000018-01-prefill-profile-instrumentation.md`.

## What Changed

2026-04-16T21:48-0700 devlog/plans/000018-01-prefill-profile-instrumentation.md —
created plan.

2026-04-16T21:52-0700 wick/src/model/metal_lfm2.rs — added `skip_attn`
guard in `forward_prefill_inner` around the batched `attention_prefill`
dispatch (matches the existing `encode_attention` / `encode_attention_q_offset`
noattn guards), cached the env var once per forward-call to keep the
per-layer loop free of syscalls.

2026-04-16T21:55-0700 wick/src/model/metal_lfm2.rs — added
`forward_prefill_profiled_gpu_inner` (single command buffer, per-phase
encoder with `sample_counters_in_buffer` attachments, resolve ticks →
µs after wait). `forward_prefill_profiled` dispatches to it when
`WICK_PROFILE=gpu`. Falls back to CPU-timing path if the device doesn't
expose timestamp counters (build_gpu_timer returns None).

2026-04-16T21:58-0700 benchmarks/profile_longctx.md — appended §8
"Post-instrumentation attribution" with the two clean-data tables:
  - noattn at p=128/1024/4096: attn share 0/51/80% (was 0/0/0 in the §6
    pre-fix table).
  - GPU-timestamp vs CPU-wall breakdown at p=4096: `attn_kernel` share
    rises from 70% to 80% once dispatch overhead is excluded; the two
    methods agree on the absolute `attn_kernel` time (-0.06%).
  - Revised flash-rewrite expectations: 3–4× prefill target is now
    reachable, with a new "≥40% attn_kernel GPU-time drop or abandon"
    lower bound.

## Decisions

2026-04-16T21:48-0700 Scope = instrumentation only, no kernel changes —
matches report §7's "first (prerequisite for confidence)" step. The flash
rewrite that depends on this data is a separate plan.

2026-04-16T21:48-0700 Route GPU-timestamps via a sibling
`forward_prefill_profiled_gpu_inner` rather than parameterizing the
existing function, because the two timing models differ enough (per-phase
commit+wait vs single commit with sample attachments) that a shared
`run_phase` closure would be harder to read than two parallel functions.

2026-04-16T21:55-0700 Built a fresh `GpuTimer` per call via
`build_gpu_timer` instead of reusing `self.gpu_timer`, because
`self.gpu_timer` is keyed to `WICK_PROFILE=gpu` at model load time and
accumulates labels as `&'static str` across calls — per-layer prefill
labels are `String` and per-call, so the existing timer's storage
shapes don't match. The calibration overhead is ~5 ms per profile call
and the profile path is diagnostic-only, so the cost is acceptable.

2026-04-16T21:58-0700 Kept `aggregate_prefill_phases` in bench_perf.rs
untouched — the GPU-ticks variant emits the same `L{layer}_{phase}`
label shape as the CPU variant, so both modes go through the same
post-processing and the test output stays identical.

## Issues

2026-04-16T21:56-0700 `p=128 noattn` reported ~6% attn share with high
variance — first noattn run at 4151 tok/s vs 8728 tok/s on run 3.
Investigated: total prefill wall time at p=128 is ~16 ms; the first
measured run still includes some Metal shader-cache / GPU-clock warmup
after the single warmup iteration, and noattn's absolute savings at
p=128 are in the noise. Not a bug in the guard — confirmed by p=1024
(51%) and p=4096 (80%), both in expected range. Noted explicitly in the
report so future readers don't chase the p=128 delta.

## Commits

f020eae — perf(profile): batched-prefill noattn + GPU-timestamp attribution
511729b — review: share GpuTimer across chunks, warn on noattn+profiled, fix buffer-role comment
HEAD — review: cache skip_attn on model + dedupe profiled prefill layer loop

## PR review round 2 addressed (2026-04-16T22:30-0700)

- **Copilot/Junie 2206 (repeat)**: finally cached `skip_attn` as a field
  on `MetalLfm2Model`, populated from `WICK_PROFILE=noattn` at load time
  alongside the existing `force_flash` / `attn_mode`. Three call sites
  (`encode_attention`, `encode_attention_q_offset`, `forward_prefill_inner`)
  now read `self.skip_attn` instead of hitting `std::env::var` per
  layer / per forward call. Verified: noattn bench at p=4096 still
  produces 11483 tok/s vs 2246 baseline (~80% attn share).
- **Copilot/Junie 3153**: deduped `forward_prefill_profiled_inner` and
  `forward_prefill_profiled_gpu_inner` by extracting
  `encode_prefill_phases<F>` — a single shared method that owns the
  embedding stage + layer loop + `output` epilogue and calls
  `run_phase(name, encode_fn)` once per logical phase. The two wrappers
  become thin adapters: CPU variant commits per phase and times with
  `Instant`, GPU variant accumulates encoders with sample attachments
  into one command buffer and resolves timestamps after wait. Net
  -317 lines, identical output shape. Verified: both variants still
  produce the same `L{layer}_{phase}` label shape and agree on
  `attn_kernel` absolute time (CPU=1,430,595 µs, GPU=1,416,216 µs,
  within 1%).
- **Copilot/Junie 3329 (granularity)**: resolved automatically — both
  variants now go through the same phase list by construction. No code
  change needed beyond the dedupe above.

## PR review addressed (2026-04-16T22:12-0700)

- **Copilot 3129**: `build_gpu_timer` fired per-chunk (5 ms calibration ×
  N chunks on prompts > 512 tokens). Fixed: `forward_prefill_profiled`
  builds the `GpuTimer` once and passes it as a parameter into
  `forward_prefill_profiled_gpu_inner`, which now resets `next_idx` /
  `labels` per chunk but reuses the calibrated sample buffer.
  Verified: p=4096 run now logs 3 calibrations (1 at load + 1 warmup + 1
  measured), vs ~16 before (8 chunks × 2 calls).
- **Copilot 2734**: `WICK_PROFILE=noattn` silently ignored on the profiled
  paths. Fixed: `forward_prefill_profiled` now logs an eprintln warning
  when it detects `noattn` so the user knows attention still ran. Also
  documented in the function's doc comment.
- **Copilot 2470**: comment in `forward_prefill_inner` mis-described
  `prefill_normed_buf` as holding Q. Rewrote to explain correctly that Q
  lives in `prefill_proj_buf` and `prefill_normed_buf` is the attention
  kernel's output slot (pre-attention it holds the RMSNorm'd hidden
  state from Phase 1, which is what downstream reads when we skip).
- **Plan §39 (same wording fix)**: updated for consistency.
- **Junie / Copilot 2206**: "cache `skip_attn` in the model struct" — not
  applied. The env var is already cached once per forward call; moving
  it to the struct would save a single syscall per forward (negligible)
  at the cost of making the flag non-dynamic (can't flip mid-process).
