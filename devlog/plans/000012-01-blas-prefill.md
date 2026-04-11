# 000012-01 — BLAS CPU prefill implementation plan

## Thinking

Benchmark: wick CPU prefill 144 tok/s vs llama.cpp CPU prefill 610 tok/s on LFM2.5-VL-1.6B-Q4_0 at 2k-token prompt. Gap ≈ 4.2×.

Investigation shows wick's NEON GEMM is ALREADY multi-threaded via rayon's `par_rows_n` when `m ≥ 256`, which our prefill shapes (`m ∈ {2048, 6144, 6912}`) always satisfy. So parallelism isn't the issue.

The root cause is **NEON vs AMX**: Apple Silicon has two separate matmul paths. NEON `vdotq_s32` tops out around 500-600 GFLOPs/s aggregate, while AMX (Apple Matrix Extension) delivers ~1.5-2 TFLOPs f32. AMX is only accessible via the Accelerate framework (SGEMM through vDSP/BLAS). Routing our prefill GEMM through Accelerate unlocks AMX.

Dequantization overhead: Q4_0 → f32 costs ~200 µs per matrix, negligible vs the GEMM itself.

x86_64 Linux path is a bonus: currently it falls through to a per-token `gemv_dispatch` loop which is catastrophically slow. Adding BLAS eliminates that fallback entirely.

## Plan

### Phase 1: Dependencies + BLAS binding (no behavior change)
1. Add `cblas-sys`, `accelerate-src` (macOS), `openblas-src` (Linux) to `wick/Cargo.toml` behind a `blas` feature
2. Default the `blas` feature on for macOS, off elsewhere
3. Create `wick/src/backend/blas.rs` with a thin `sgemm_rowmajor_nn` wrapper (no transpose, row-major) that takes slices and dispatches to `cblas_sgemm`
4. Make sure `cargo build --features blas` works on macOS and `cargo build --no-default-features` still works

### Phase 2: Matrix-level dequantization helpers
1. Add `dequantize_q4_0_matrix(data: &[u8], m: usize, k: usize, out: &mut [f32])` to `wick/src/quant.rs` — loops over rows calling existing `dequantize_q4_0_row`
2. Same for Q8_0: `dequantize_q8_0_matrix`
3. Parallelize with rayon when `m ≥ 64`

### Phase 3: Add dequant scratch to InferenceState
1. Add `pub dequant_weight: Vec<f32>` to `ScratchBuffers` in `wick/src/kv_cache.rs`
2. Compute `max_weight_floats = max(3*hs*hs, is*hs)` at `from_config_with_compression` time
3. Pre-allocate to `max_weight_floats`

### Phase 4: Smoke test — ffn_up through BLAS only
1. Add `self.prefill_gemm_blas` helper on `Lfm2Model`
2. Replace ONLY the `ffn_up` GEMM call site with the new helper
3. Run tests + benchmark
4. **Go/no-go**: look for +15–30% prefill tok/s on the 2k bench

### Phase 5: Full rollout
1. Wire the remaining 7 GEMM call sites through BLAS
2. Remove `#[cfg(target_arch = "aarch64")]` gate on the batched prefill path (x86_64 gets the win too)
3. Re-run full benchmark matrix

### Phase 6: Documentation + benchmark update
1. Update the benchmark doc with new wick CPU numbers
2. Update README perf section
3. Devlog entry with final numbers

## Verification

- `cargo fmt --check`
- `cargo clippy --workspace -- -D warnings`
- `cargo test --workspace` — all tests pass, especially `test_batched_prefill_logits_match_sequential`
- `WICK_QUALITY_GATE=1 cargo test -p wick --release --test quality_gate -- --ignored --nocapture` — cosine stays within ±0.005
- Benchmark: wick CPU prefill > 300 tok/s at 2k prompt (current 144)
- `cargo build --release --no-default-features` still works
- `cargo build --release --features blas` works on macOS

## Out of scope
- Metal prefill (item A from roadmap)
- TurboQuant on Metal (item C)
- SWA (item D)
- Chunked prefill scaling bug (item E)
- KV ring buffer layout (item F)
