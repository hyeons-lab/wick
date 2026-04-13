# 000014 — perf: CPU decode investigation + E-core fix

## Agent
Claude (claude-opus-4-6[1m]) @ wick branch perf/cpu-decode

## Intent
Investigate and close the 2.4× CPU decode gap vs llama.cpp (59.5 vs 143 tok/s). Decode is GEMV (n=1), memory-bandwidth bound.

## What Changed
- 2026-04-12T20:00-0700 `wick/src/backend/cpu.rs` — added `configure_thread_pool()` that auto-detects P-core count via `sysctlbyname("hw.perflevel0.logicalcpu")` on macOS and configures rayon to exclude E-cores. E-cores cause 12% decode regression (straggler threads in par_chunks_mut).
- 2026-04-12T20:00-0700 `wick-cli/src/main.rs` — call `configure_thread_pool()` early in main().
- 2026-04-12T20:00-0700 `wick/src/backend/cpu.rs` — added `microbench_gemv_q4_0` test (ignored) for measuring isolated GEMV bandwidth.

## Decisions
- 2026-04-12T14:00-0700 Investigated thread pool replacement (spin-wait à la ggml). Thread scaling analysis showed the gap ratio is constant at ~51% across 1/4/8 threads — ruling out thread pool as the cause.
- 2026-04-12T18:00-0700 Investigated NEON kernel differences (vmmlaq/I8MM). M1 Max does NOT support FEAT_I8MM (`sysctl: 0`). llama.cpp binary contains 0 vmmlaq instructions. Both use identical vdotq_s32 algorithm.
- 2026-04-12T19:00-0700 Validated with standalone C microbench: Apple Clang achieves 19.7 GB/s per core, Rust achieves 19.5 GB/s. **Kernel codegen is not the bottleneck.**
- 2026-04-12T20:00-0700 Identified the real gap: wick's full decode model overhead (conv/attn compute, buffer copies, dispatch) costs ~11 ms per token vs C chain benchmark's ~1 ms. The 2× gap vs llama.cpp comes from model-level dispatch efficiency, not the GEMV kernel.

## Issues
- The 2× per-core gap (25 vs 49 tok/s single-threaded) is NOT from the NEON kernel or the thread pool. Both wick and llama.cpp achieve ~20 GB/s per core on the isolated GEMV. The gap is from wick's model-level overhead: more buffer copies, closure/iterator dispatch, and ggml's more efficient operator scheduling. Closing this requires either ggml-style operator fusion or restructuring `run_layers`.

## Commits
- HEAD — perf(cpu): auto-detect P-cores on Apple Silicon, exclude E-cores from rayon

## Next Steps
- Profile `run_layers` with fine-grained per-operation timing to identify the biggest overhead contributors
- Consider operator-level optimizations: fuse gate+up GEMVs, reduce buffer copies, restructure dispatch
- On newer hardware (M3+/M4 with FEAT_I8MM), vmmlaq_s32 2-row kernel would give ~2× per-core improvement
