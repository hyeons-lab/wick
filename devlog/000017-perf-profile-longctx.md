# 000017 — perf/profile-longctx

**Agent:** Claude (claude-opus-4-7) @ wick branch perf/profile-longctx

## Intent

Attribute the long-context perf gap. `benchmarks/deltas_table.md` shows wick's
prefill dropping from 1.29–1.83× llama.cpp at p=128 to 0.22–0.30× at p=4096,
and decode at ctx=4096 dropping to 0.28–0.54×. Produce a per-phase breakdown
report so the next fix (likely a tiled flash-attention rewrite) can be
scoped against real data.

Plan: `~/.claude/plans/given-the-benchmarks-and-sharded-pebble.md`.

## Discoveries from planning phase

- `wick bench --prompt-tokens N --max-tokens M --runs 1 --warmup W` is the
  only CLI entry that controls prefill size precisely. `WICK_PROFILE` env
  var triggers CategoryTimer/GPU-timer/noattn paths inside forward().
- **Critical**: `WICK_PROFILE=timing` routes prefill through *single-token*
  forward (`metal_lfm2.rs:1875-1877` in `forward_greedy` — "Profile paths
  still go through forward() + CPU argmax"). That is NOT the batched prefill
  path that produces the 2250 tok/s number in the bench table. To attribute
  the batched path we must use `forward_prefill_profiled`, an instrumented
  Rust-level function that commits+waits between every encode phase.
- Existing ignored test `wick/tests/bench_perf.rs:467` (`test_prefill_phase_profile`)
  already uses `forward_prefill_profiled` for n=128 on the 1.6B model and
  aggregates by category. Extend/duplicate this for the full matrix.
- `MAX_PREFILL_TOKENS = 512` (`metal_lfm2.rs:24`). Prefill is chunked in
  512-token blocks for n>512 (`metal_lfm2.rs:2178-2191`). So at p=4096 we do
  8 sequential batched chunks. The per-chunk attention is batched GEMM but
  the inter-chunk sequencing is a key candidate for the O(n²) signature.
- Flash attention currently engages at `seq_len > 4096`, so p=4096 uses
  classic attention (`metal_lfm2.rs:1626`). The memory note about "flash is
  14% slower than classic" applies to seq_len=4097+, not the p=4096
  benchmark row.

## Data-collection strategy

Two-track attribution:

1. **Batched-prefill track** (primary, matches bench numbers):
   `forward_prefill_profiled` via new ignored tests in `bench_perf.rs`.
   Per-layer per-phase us with sync between phases.
2. **Single-token track** (decode + profiled-single-token prefill):
   `wick bench --max-tokens N` with `WICK_PROFILE=timing`. CategoryTimer
   output via stderr.

Cross-check: `WICK_PROFILE=noattn wick bench` vs unset — the wall-time delta
estimates attention cost independently of the timing-mode code path.

## What Changed

- 2026-04-16T13:25-0700 `wick/tests/bench_perf.rs` — added
  `profile_longctx_run` helper + 5 ignored tests
  (`test_profile_longctx_{450m,1_6b}_n{128,1024,4096}`). Each runs
  `forward_prefill_profiled`, aggregates by category, and emits a
  grep-able `PROFILE_LONGCTX BEGIN/END` block to stderr.
- 2026-04-16T13:25-0700 `scripts/profile_longctx.sh` — one-shot driver
  that runs the five prefill-profile tests + `wick bench` matrix
  (prefill wall-time, decode CategoryTimer at ctx=128/4096, noattn cross-
  check). Dumps raw logs to `benchmarks/profile_longctx_raw/`.
- 2026-04-16T13:25-0700 `benchmarks/profile_longctx.md` — final
  attribution report with methodology caveats, data tables, and the Step
  3 findings+recommendation section.
- 2026-04-16T13:25-0700 `benchmarks/profile_longctx_raw/*` — captured
  stderr from all matrix runs (22 logs, ~48 KB).
- 2026-04-16T20:50-0700 `wick/src/model/metal_lfm2.rs` — split
  `forward_prefill_profiled` into a chunking wrapper (public) that
  loops over MAX_PREFILL_TOKENS-sized chunks and a new
  `forward_prefill_profiled_inner` (private) that does the per-phase
  encoding. The old signature overflowed `prefill_batch_buf` for n > 512,
  which flagged in PR review and explained why profiled totals were
  *lower* than real bench wall time at p=4096 (UB ran faster).
- 2026-04-16T20:50-0700 `wick/tests/bench_perf.rs` — renamed 450M target
  from `LFM2-VL-450M-Q4_0` to `LFM2.5-VL-450M-Q4_0` to match the naming
  used elsewhere in the file. Both models show the same 0.22× regression
  row in `deltas_table.md`, so qualitative findings unchanged.
- 2026-04-16T20:50-0700 `scripts/profile_longctx.sh` — added `--no-cache`
  and `--context-size 8192` flags (they had been in my inline runs but
  missing from the committed script). Also replaced model path for 2.5.
- 2026-04-16T20:50-0700 Re-ran the full matrix with the chunking fix.
  `attn_kernel` at p=4096 drops from 78.55% → 70.33% (450M) and
  69.95% → 65.70% (1.6B) — still unambiguously dominant.
- 2026-04-16T20:50-0700 Added ctx=2048 decode run to replace
  the ctx=4096 decode row (2.5 model EOS-es on the synthetic 4096-token
  prompt at decode step 1, so CategoryTimer never prints). Scaling trend
  from ctx=128 → ctx=2048 confirms attention grows while everything else
  is flat.
- 2026-04-16T20:50-0700 `benchmarks/profile_longctx.md` — rewrote with
  corrected numbers + added caveats about the chunking fix, 2.5 model
  EOS behavior, and added follow-up action item for `--ignore-eos`
  diagnostic flag.
- 2026-04-16T22:50-0700 **Discovered and fixed a latent correctness
  bug in `flash_attention.metal`.** User asked whether the "2.5 model
  EOS-es at ctx=4096" behavior was a bug — investigation found that
  `flash_attention.metal` bound K/V caches as `const device float*`
  while the caches are stored as f16 (classic `attention.metal` binds
  them as `half*`). Reading f16 data as f32 reinterprets two adjacent
  halves as one float → garbage attention output → garbage logits →
  arbitrary greedy tokens (including EOS). Fix: change bindings to
  `half*` and add `float()` casts on load.
  - `wick/src/backend/shaders/flash_attention.metal` — 3 targeted
    edits (kernel signature + 2 load sites) + explanatory comment.
  - `wick/tests/attention_metal_parity.rs` — new test file with
    `test_classic_vs_flash_attention_parity`. Fresh-model-per-variant
    to pick up `WICK_FLASH` at construction. Asserts classic and
    forced-flash greedy-decode tokens are identical. Fails cleanly
    with the bug (classic emits 6 coherent tokens ending at EOS;
    flash emits 12 random tokens); passes after the fix.
  - Re-ran `bench --prompt-tokens 4096 --max-tokens 128
    --context-size 8192`: previously EOS-ed after 1 token, now decodes
    96 tokens cleanly. Long-context decode data at ctx=4096 is real
    now and shows `attn_kernel` at 54% of 43.8 ms/tok (23.65 ms/tok
    per-step, 14.8× scaling vs ctx=128). Replaced the "unavailable
    due to EOS" caveat in the report with actual data and removed
    the stale `--ignore-eos` follow-up recommendation.
- 2026-04-16T22:50-0700 `benchmarks/profile_longctx_raw/450m_decode_ctx4096.stderr`
  regenerated with the fix in place.
- 2026-04-16T23:20-0700 **Critical review of the prior fix surfaced the
  same bug in two more kernels.** Greped all attention shaders for the
  `float*` K/V pattern:
  - `attention.metal`, `attention_prefill.metal`: `half*` — correct.
  - `flash_attention.metal`: `float*` — fixed in 01b8e53.
  - `attention_gqa.metal`: `float*` — same bug. Opt-in via
    `WICK_ATTN=gqa` (latent unless user opts in, but silently wrong
    whenever they do).
  - `attention_splitk.metal`: `float*` — same bug on the compute
    kernel. Opt-in via `WICK_ATTN=splitk`.
  Fix identical to flash: change K/V bindings to `half*`, cast to
  float on load. Extended `attention_metal_parity.rs` with
  `test_classic_vs_gqa_attention_parity` and
  `test_classic_vs_splitk_attention_parity`; also refactored to share
  a `with_env()` helper for cleaner env-var flipping. All three parity
  tests pass.
- 2026-04-16T23:20-0700 `.gitignore` + `git rm` for .claude files.
  User asked to gitignore `.claude/` and remove any tracked files
  under it. Added `.claude/` to `.gitignore`; `git rm`-d the two
  files that had been accidentally committed in an earlier branch
  (`.claude/memory/feedback_critical_review_multiple.md` and
  `feedback_no_sed_on_files.md`). These are Claude Code local state
  and shouldn't be in the repo.

## Decisions

- 2026-04-16T13:25-0700 — Use `forward_prefill_profiled` via ignored tests
  for batched-prefill attribution, because the existing CategoryTimer path
  silently rewrites prefill into single-token mode. **Why:** the benchmark
  table numbers (2250 tok/s @ p=4096) are for the batched path; profiling
  the single-token path would attribute a different code path's costs.
  **How to apply:** when attributing any future prefill regression, verify
  which path the benchmark measurement used and match the profiling path.
- 2026-04-16T13:25-0700 — Branch off `origin/main` (01a8b02) rather than
  the locally-modified main checkout. **Why:** the local modifications are
  non-perf (CLI flags for footprint sim, a print-cadence tweak) and the
  engine behavior we're profiling is the merged baseline that produced the
  published benchmark numbers.

## Issues

- 2026-04-16T13:25-0700 **`wick bench` fell back to CPU on first attempt.**
  The `--device auto` default picks CPU despite Metal support. Fixed by
  passing `--device metal` explicitly. Re-ran all CLI steps. Pre-fix logs
  still on disk but not used in the report.
- 2026-04-16T13:25-0700 **First wall-time run at p=128 reported prefill
  37874 tok/s — 5× higher than published benchmark (7270).** Root cause:
  KV prefix cache hit between warmup and measured runs. Fixed by adding
  `--no-cache`; re-measured numbers match the published benchmark within
  10%.
- 2026-04-16T13:25-0700 **Decode profile at ctx=4096 hit
  `pos >= max_seq_len` and produced 0 decode steps.** `wick bench` uses
  default `--context-size 4096`, so prefilling 4096 tokens leaves no
  room. Fixed by passing `--context-size 8192`.
- 2026-04-16T13:25-0700 **`WICK_PROFILE=noattn` cross-check invalid for
  batched prefill.** The env var only gates `encode_attention` and
  `encode_attention_q_offset` (lines 1613, 1726 in `metal_lfm2.rs`), but
  batched prefill uses `encode_attention_prefill` (line 2477) which has
  no such check. So noattn prefill numbers reflect only the single-token
  decode step (max_tokens=1) being skipped, not prefill attention.
  Logged as "Action for follow-up" in the report: add a 3-line noattn
  check to `encode_attention_prefill` for proper cross-checks.
- 2026-04-16T13:25-0700 **`forward_prefill_profiled` over-attributes
  `attn_kernel`.** Per-phase `wait_until_completed` inflates phases with
  many dispatches. Forced the report to treat its percentages as
  directional, not absolute, and triangulate with algorithmic scaling
  signals.

## Commits

09d42aa — perf(profile): long-context attribution report + tooling
a7ee530 — fix(profile): chunked forward_prefill_profiled + 2.5 model + report rerun
01b8e53 — fix(attn): flash_attention.metal K/V dtype (f32→f16) + parity test
HEAD — fix(attn): apply same f16 fix to gqa + splitk kernels; gitignore .claude

## Progress

- [x] Worktree + devlog + plan created
- [x] Write parameterized prefill-profile tests in bench_perf.rs
- [x] Run batched-prefill matrix (450M × {p=128,1024,4096}; 1.6B × {p=128,4096})
- [x] Run single-token + decode matrix via `wick bench` + WICK_PROFILE
- [x] Run noattn wall-time cross-check (discovered it's invalid for prefill)
- [x] Aggregate into `benchmarks/profile_longctx.md`
- [x] Add Step 3 findings section
- [ ] Commit
- [ ] Push + open PR (pending user OK)
