# 000002 — feature/tensor-cpu-compute

**Agent:** Claude Code (claude-opus-4-6) @ repository branch feature/tensor-cpu-compute

**Intent:** Implement Phases 1-2 of Wick — tensor types, quantization, CPU compute ops, SIMD kernels, GGUF parser, BPE tokenizer, and chat templates.

## What Changed

- 2026-03-29T16:26-0700 Cargo.toml (root) — added bytemuck workspace dependency
- 2026-03-29T16:26-0700 wick/Cargo.toml — added bytemuck dep
- 2026-03-29T16:26-0700 wick/src/tensor.rs — added from_f32, zeros_f32, as_f32_slice, as_f32_slice_mut, to_f32_vec (with auto-dequant for Q8_0/Q4_K_M/F16/BF16)
- 2026-03-29T16:30-0700 wick/src/quant.rs — full Q8_0 and Q4_K_M dequantization, scalar vec_dot, scale decoding. Ported from llama.cpp's ggml-quants.c layout.
- 2026-03-29T16:32-0700 wick/src/backend/cpu.rs — all naive CPU ops: matmul_f32, matmul_q8_0_f32, matmul_q4km_f32, rmsnorm, silu, softmax, rope, conv1d_depthwise, add_inplace, mul_inplace
- 2026-03-29T16:34-0700 wick/src/backend/simd.rs — NEON (aarch64) and AVX2 (x86_64) optimized vec_dot for Q8_0 and Q4_K_M with runtime dispatch
- 2026-03-29T16:34-0700 wick/src/backend/mod.rs — added simd module
- 2026-03-29T17:20-0700 wick/src/gguf.rs — full GGUF v3 parser: header, KV metadata (all types including arrays), tensor info, memory-mapped data via memmap2, get_tensor/tensor_data, print_inspect
- 2026-03-29T17:22-0700 wick-cli/src/main.rs — wired `wick inspect` command to GgufFile::open + print_inspect
- 2026-03-29T17:24-0700 wick/src/tokenizer.rs — byte-level BPE tokenizer loaded from GGUF metadata, encode/decode, special token handling, chat template rendering via minijinja

## Decisions

- 2026-03-29T16:26-0700 Used bytemuck for f32 ↔ byte slice casting — zero-copy, well-maintained, no unsafe in our code
- 2026-03-29T16:30-0700 Q4_K_M scale decoding follows llama.cpp's get_scale_min_k4 exactly — 6-bit scales packed into 12 bytes with split high bits
- 2026-03-29T16:32-0700 Quantized matmul extracts column slices into temp Vec for vec_dot — correct but not fast. OK for Phase 1 correctness; will optimize in Phase 5 when weights are transposed.
- 2026-03-29T16:34-0700 SIMD dispatch: compile-time cfg for aarch64 (NEON always available), runtime feature detection for x86_64 AVX2+FMA
- 2026-03-29T17:20-0700 GGUF parser reads header via BufReader then mmaps the whole file — mmap gives zero-copy tensor access, BufReader handles sequential metadata parsing
- 2026-03-29T17:20-0700 Unsupported quant types (Q2_K, Q3_K, Q5_K, Q6_K, etc.) still parse and appear in inspect, stored with DType::F32 placeholder — we don't load their data but can show their shapes
- 2026-03-29T17:24-0700 Token unescaping handles <0xHH> byte tokens and ▁ sentencepiece space markers — covers the conventions used by LLaMA, LFM2, and GPT-NeoX vocabularies
- 2026-03-29T17:24-0700 BPE merge priority uses rank (index in merge list) — lower rank = higher priority, matches HuggingFace tokenizers behavior

## Issues

- Rust 2024 `unsafe_op_in_unsafe_fn` lint: first SIMD build had ~12 warnings. Fixed by wrapping function bodies.
- GGUF magic was initially set to 0x46475547 ("FGUG") instead of 0x46554747 ("GGUF"). Fixed by verifying with Python struct.unpack.
- Clippy `manual_div_ceil`: alignment rounding used manual formula, replaced with `.div_ceil()`.
- Q4_0 quant type in the LFM2-VL-450M-Q4_0 GGUF shows 0.00 MB sizes in inspect — expected, since Q4_0 is not yet a supported quant type (only Q4_K_M and Q8_0 in v1). Parser still reads all metadata and tensor names correctly.
- 2026-03-29T19:36-0700 CI failed on x86_64: unnecessary `unsafe` block in hsum_avx — the function is already `unsafe fn` so the inner block is redundant. Only triggers on x86 targets (not aarch64 where NEON path compiles). Fixed by removing the inner unsafe block.

## Research & Discoveries

- 2026-03-29T17:57-0700 Real LFM2 GGUF tensor naming verified from ~/.leap/models/LFM2-VL-450M-Q8_0:
  - Conv blocks: `blk.N.shortconv.{in_proj,conv,out_proj}.weight`, `blk.N.attn_norm.weight`, `blk.N.ffn_{gate,up,down}.weight`, `blk.N.ffn_norm.weight`
  - Attn blocks: `blk.N.attn_{q,k,v}.weight`, `blk.N.attn_{q,k}_norm.weight`, `blk.N.attn_output.weight`, plus same ffn and norms
  - Global: `token_embd.weight`, `token_embd_norm.weight`
  - LFM2-450M: 16 blocks, 8 conv + 8 attn (alternating pattern: 0,1 conv, 2 attn, 3,4 conv, 5 attn, ...)
  - `lfm2.attention.head_count_kv` is an i32 array (per-layer KV heads), not a scalar

## Commits

- 71e50ac — feat: implement tensor ops, quantization, CPU compute, and SIMD kernels (Phase 1)
- 4221114 — feat: implement GGUF parser, BPE tokenizer, and chat templates (Phase 2)
- 274b6e6 — ci: add GitHub Actions workflow and just ci recipe
- fead476 — feat: add Q4_0 quantization support
- HEAD — fix: remove unnecessary unsafe block in hsum_avx for x86_64 CI
