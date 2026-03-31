# 000003 — docs/implementation-plan

**Agent:** Claude Code (claude-opus-4-6) @ repository branch docs/implementation-plan

**Intent:** Address PR #2 review comments, incorporate per-shape kernel tuning research, and add CLAUDE.md for future Claude Code sessions.

## What Changed

- 2026-03-30T09:07-0700 wick/src/backend/cpu.rs — added explicit `use std::mem::size_of` import
- 2026-03-30T09:07-0700 docs/IMPLEMENTATION_PLAN.md — added note clarifying wgpu/gpu feature is planned V2, not current
- 2026-03-30T09:07-0700 wick/src/gguf.rs — replaced unchecked arithmetic with `checked_add`/`usize::try_from` in parsing loop, `get_tensor()`, and `tensor_data()`; added unsupported-type check to `tensor_data()` for consistency with `get_tensor()`

- 2026-03-30T09:44-0700 docs/IMPLEMENTATION_PLAN.md — added V2.7 per-shape kernel tuning section inspired by kernel-anvil; added design note to Phase 5 wgpu shaders; added AMD GPU performance prioritization path; renumbered V2.8-V2.14
- 2026-03-30T09:54-0700 CLAUDE.md — created with build/test commands, architecture overview, pre-commit fmt requirement, and project conventions
- 2026-03-30T10:09-0700 .github/workflows/code-review.yml — added Junie automated code review workflow for PRs
- 2026-03-31T07:40-0700 wick/src/gguf.rs — `tensor_data_size()` now returns `Result` with checked arithmetic and block alignment validation; `data_offset` computation uses `checked_mul`; extracted `tensor_range()` helper deduplicating validation in `get_tensor()`/`tensor_data()`; added 3 new tests (overflow, bad alignment, unsupported type rejection)
- 2026-03-31T07:40-0700 scripts/compare_tokenizer.py — fail fast on non-zero cargo run exit code

## Decisions

- 2026-03-30T09:07-0700 Merged comments 3 and 5 into a single clarifying note — both flagged the same wgpu/gpu mismatch
- 2026-03-30T09:07-0700 Applied checked arithmetic to all three offset computation sites (parsing loop, get_tensor, tensor_data) for consistency, not just the one the reviewer flagged
- 2026-03-30T09:44-0700 Added kernel-anvil's per-shape GEMV tuning as V2.7 rather than integrating into Phase 5 — V1 shaders should use fixed workgroup sizes for simplicity, but be designed with configurable params so V2.7 is a config change. Added note to Phase 5 about this.

## Research & Discoveries

- 2026-03-30T09:44-0700 kernel-anvil (https://github.com/apollosenvy/kernel-anvil) demonstrates that per-shape kernel config tuning yields massive decode speedups on AMD RDNA3. Key findings:
  - llama.cpp MMVQ kernels use identical nwarps/rows_per_block for all layer shapes — suboptimal
  - Profiling unique (quant_type, N, K) shapes and applying per-shape configs: 2.25x on Qwen3.5-27B Q4_K_M (7900 XTX)
  - Bottleneck classification (bandwidth-bound, occupancy-limited, compute-bound) determines sweep strategy
  - Only targets decode path (batch=1 GEMV), not prefill GEMM
  - AMD RDNA2/3/3.5/4 supported; CUDA/Metal planned
  - ~50 line patch to llama.cpp; configs stored as JSON loaded at runtime

## Commits

- 9719c04 — fix: address PR #2 review comments — checked arithmetic, consistent tensor_data, docs note, size_of import
- afb1f8f — docs: add per-shape kernel tuning (V2.7) to implementation plan
- 187c622 — style: cargo fmt checked arithmetic in gguf.rs
- 4bf00bb — docs: add CLAUDE.md with build commands, architecture, and conventions
- 56ab988 — ci: add Junie automated code review workflow
- HEAD — fix: address Junie review — checked tensor_data_size, data_offset overflow, tensor_range helper, tokenizer script error handling, tests
