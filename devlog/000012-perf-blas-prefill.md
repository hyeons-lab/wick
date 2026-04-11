# 000012 — perf: BLAS CPU prefill

## Agent
Claude (claude-opus-4-6[1m]) @ wick branch perf/blas-prefill

## Intent
Close the 4.2× CPU prefill gap vs llama.cpp (144 vs 610 tok/s measured on LFM2.5-VL-1.6B-Q4_0, 2k prompt) by routing prefill GEMM through Apple's Accelerate framework on macOS (unlocking AMX) and OpenBLAS on Linux. Also unlock CPU prefill on x86_64 which currently falls through to an unusable per-token GEMV loop.

## What Changed
- 2026-04-10T20:45-0700 `Cargo.toml` (workspace) + `wick/Cargo.toml` — added `cblas-sys`, `accelerate-src` (macOS), `openblas-src` (Linux) as workspace deps; `blas` feature on `wick` defaults on and gates the new wrapper.
- 2026-04-10T20:45-0700 `wick/src/backend/blas.rs` (new) + `wick/src/backend/mod.rs` — `sgemm_rowmajor_nn(m, n, k, a, b, c)` wrapper around `cblas_sgemm` with `CblasRowMajor`, `CblasNoTrans`/`NoTrans`, α=1/β=0. Pulls in the provider crate via `use … as _` so its `#[link]` attribute fires. Two unit tests (identity and 2×2 multiply).
- 2026-04-10T21:15-0700 `wick/src/quant.rs` — added `dequantize_q4_0_matrix` and `dequantize_q8_0_matrix` that loop the existing row helpers. Parallelized with rayon via `par_chunks_mut(k).zip(par_chunks(row_bytes))` when `m >= MATRIX_DEQUANT_PAR_THRESHOLD` (64). Two parity tests verify the matrix helper output matches a sequential row-by-row loop byte-for-byte.

## Decisions
- 2026-04-10T20:27-0700 Use `cblas-sys` + platform-gated providers (`accelerate-src` on macOS, `openblas-src` elsewhere). Avoids hand-rolling `extern "C"` bindings and keeps the BLAS dispatch behind a single `blas` feature flag.
- 2026-04-10T20:27-0700 Staged rollout: start with a single call site (`ffn_up`, largest matrix) as a smoke test. If measured prefill doesn't improve, stop and profile before refactoring all 8 call sites.
- 2026-04-10T20:27-0700 Dequantize weights into a scratch buffer on `ScratchBuffers`, reused across layers. Sized at `max(3*hs*hs, is*hs)` floats at `from_config_with_compression` time.
- 2026-04-10T20:27-0700 Default the `blas` feature on for macOS only. Linux users opt in via `cargo build --features blas` and bring their own OpenBLAS.

## Issues
(to be filled)

## Commits
- 7cb64f9 — docs: devlog + plan for BLAS CPU prefill (000012)
- e77cf0e — feat(blas): add cblas_sgemm wrapper behind `blas` feature
- HEAD — feat(quant): dequantize_q4_0_matrix / dequantize_q8_0_matrix with rayon

## Next Steps
1. ~~Add Cargo deps~~ ✓
2. ~~Create wick/src/backend/blas.rs~~ ✓
3. ~~Add `dequantize_q4_0_matrix` / `dequantize_q8_0_matrix`~~ ✓
4. Add `dequant_weight` scratch to InferenceState
5. Smoke test: wire ffn_up through BLAS only, benchmark
6. Full rollout: wire all 8 GEMM call sites, benchmark
7. Update benchmark doc with new numbers
