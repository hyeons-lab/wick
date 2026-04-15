# 000016 — perf(metal): PR #17 review fixes

**Agent:** Claude (claude-opus-4-6) @ wick branch feat/perf-opt-split-k-2

## Intent

PR #17 (Split-K GEMV for Metal) CI is failing on `cargo fmt` and has 14 line-level review comments. Address the correctness-critical ones so the PR can land, then the nitpicks.

## Review comment triage

**CRITICAL (correctness, will produce wrong results or data races):**

1. `metal_lfm2.rs:1051` + `gemv_q4_0_fast.metal:612` — Phase A grid is dispatched as 2D `(rows_per_split, n_splits, 1)` but the shader reads `uint tg_id [[threadgroup_position_in_grid]]` as a 1D scalar (only gets the X component). Result: every threadgroup computes `split_id = 0` and the split dimension is silently collapsed. Need to either dispatch 1D `(rows_per_split * n_splits, 1, 1)` or change the shader to take `uint2` and use `tg_pos.y`.

2. `gemv_q4_0_fast.metal:686, 705` + `metal_lfm2.rs:1076` — Phase B merge kernels (`gemv_q4_0_splitk_merge` and `_merge_accum`) are dispatched with `sz1d(32)` (32 threads per TG) but the kernel only uses `row [[threadgroup_position_in_grid]]` without gating on `tid`. All 32 threads in the TG run the full reduction loop and write `y[row]` concurrently → data race / UB. Fix: gate the write with `if (tid == 0)` or dispatch with 1 thread per TG.

**BLOCKING CI (compile/fmt):**

3. `gemv_q4_0_bench.rs:186, 293, 451` — loop variable was renamed `_i` but body still references `i`. Compile error. Revert to `i`.
4. `cargo fmt --check` diff in the same file — formatting drift.

**MEDIUM (correctness/UX, not blocking):**

5. `wick-cli/src/main.rs:605` — `--prompt-tokens` generates IDs via `(i % 1000) + 100` which can hit special tokens and isn't unique beyond 1000 tokens. Should sample real vocab IDs.

**LOW (nitpicks, skip for this pass):**

6. `scripts/benchmark_matrix.py:12` — hardcoded `/Users/...` `LLAMA_BENCH` path.
7. `metal_audio_decoder.rs:120` — `__vocoder_path` double-underscore convention (should be `_vocoder_path`).
8. `benchmark_results.csv`, `wick_time.txt`, `llama_time.txt` — committed machine-specific artifacts.

## What Changed

2026-04-14T06:04-0700 `wick/src/model/metal_lfm2.rs` — Phase A dispatch changed from 2D `(rows_per_split, n_splits, 1)` to 1D `(rows_per_split * n_splits, 1, 1)`. Shader reads `tg_id` as `uint` (not `uint2`), so the 2D dispatch silently collapsed the split dimension — every TG computed `split_id = 0` and Phase B then summed one real partial plus (n_splits-1) uninitialized reads. Phase B dispatch changed from `sz1d(32)` to `sz1d(1)` so the scalar reduction + `y[row] = acc` happens exactly once per row instead of racing 32 threads on the same address.

2026-04-14T06:04-0700 `wick/tests/gemv_q4_0_bench.rs` — same Phase B fix (`threads_merge = MTLSize::new(1, 1, 1)`). Also reverted the `_i` → `i` rename in three timing loops where the body still referenced `i` (`if i == iters - 1 { ... }` for last-iter wait), which was a hard compile error. Applied `cargo fmt`.

2026-04-14T06:04-0700 `wick/src/model/metal_audio_decoder.rs` — trailing `cargo fmt` whitespace fix.

## Verification

| Check | Result |
|---|---|
| `cargo fmt --check` | clean |
| `cargo clippy --workspace -- -D warnings` | clean |
| `cargo test --workspace` | all pass |
| Coherence (Metal, LFM2.5-VL-1.6B-Q4_0) | "Paris." ✅ |
| Audio TTS (Metal) | coherent text + valid WAV ✅ |
| Decode tok/s (Metal, 1.6B Q4_0) | 381 tok/s (up from ~267 without Split-K, so the fix is strictly better than not Split-K-ing, AND it's correct) |

The previous broken-Split-K numbers in the PR body (283 tok/s) were measured with the 2D-grid dispatch bug silently collapsing 4 splits to 1, and Phase B racing 32 threads on every write. Because all 32 threads in the race computed the same `acc` value, the race was benign for correctness at the merge step — but the dispatch collapse meant Phase A was only doing 1/4 of the K-dim work and reading uninitialized memory for the other 3 partials. The fact that output looked coherent was luck from `y_partial` being zero-initialized (so uninitialized partials summed to zero and merge returned the single good partial).

## Skipped review comments

**Intentional (not blocking):**
- `scripts/benchmark_matrix.py:12` — hardcoded `LLAMA_BENCH` path. Script is a developer tool, not shipped.
- `metal_audio_decoder.rs:120` — `__vocoder_path` double-underscore. Trivial convention.
- `benchmark_results.csv`, `wick_time.txt`, `llama_time.txt` — committed machine-specific artifacts. Would need to decide whether to delete them or relocate; not a correctness concern and the PR author's call.
- `wick-cli/src/main.rs:605` — `--prompt-tokens` special-token handling. Legit UX concern but the current form is good enough for prefill benchmarks.

## Commits

- 51cd656 — fix(metal): Split-K GEMV dispatch bug and Phase B data race
- HEAD — devlog: record commit hash
