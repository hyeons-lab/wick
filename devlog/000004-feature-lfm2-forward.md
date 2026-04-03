# Devlog: feature/lfm2-forward

**Agent:** Claude Code (claude-opus-4-6) @ wick branch feature/lfm2-forward

## Intent

Implement Phase 3 of the wick inference engine: LFM2 forward pass, sampling, generation loop, and `wick run` CLI command. This enables end-to-end text generation using LFM2/LFM2.5 GGUF models.

## What Changed

- 2026-04-02T19:34-0700 devlog/000004-feature-lfm2-forward.md — initial devlog
- 2026-04-02T19:34-0700 devlog/plans/000004-01-lfm2-forward-pass.md — implementation plan
- 2026-04-02T20:50-0700 wick/src/quant.rs — Added BlockQ6K struct and dequantize/vec_dot functions (ported from llama.cpp dequantize_row_q6_K)
- 2026-04-02T20:50-0700 wick/src/tensor.rs — Added DType::Q6K variant with block_size=256, block_bytes=210
- 2026-04-02T20:50-0700 wick/src/gguf.rs — Mapped Q6_K type, added get_i32_array(), get_bool(), mmap_data(), data_offset() accessors
- 2026-04-02T20:50-0700 wick/src/model/mod.rs — Added load_model() dispatch, kv_heads_per_layer to ModelConfig
- 2026-04-02T20:50-0700 wick/src/model/lfm2.rs — Full LFM2 model: from_gguf loader, WeightRef pre-computation, forward pass with shortconv + GQA attention + SwiGLU FFN
- 2026-04-02T20:50-0700 wick/src/kv_cache.rs — Added InferenceState::from_config(), append_kv(), kv_cache() methods
- 2026-04-02T20:50-0700 wick/src/backend/cpu.rs — Added GEMV functions (q4_0, q8_0, q6k, q4km) and gemv_dispatch()
- 2026-04-02T20:50-0700 wick/src/sampler.rs — Added Sampler struct with temperature, top-k, top-p, greedy modes
- 2026-04-02T20:50-0700 wick/src/engine.rs — Added generate() function with prefill + decode loop + timing
- 2026-04-02T20:50-0700 wick-cli/src/main.rs — Wired `wick run` command

## Decisions

- 2026-04-02T19:34-0700 `token_embd_norm.weight` is the OUTPUT norm (applied after all layers), not an embedding norm — verified against llama.cpp `LLM_TENSOR_OUTPUT_NORM_LFM2` mapping
- 2026-04-02T19:34-0700 Do NOT reuse `conv1d_depthwise` for shortconv — it uses same-padding; LFM2 needs valid convolution on pre-padded input (rolling buffer)
- 2026-04-02T19:34-0700 Pre-compute tensor references (WeightRef) at model load time — avoids ~148 HashMap lookups per forward pass
- 2026-04-02T19:34-0700 Sequential single-token prefill for Phase 3 — batched prefill deferred
- 2026-04-02T20:50-0700 TensorInfo.offset is already absolute — do NOT add data_offset again in resolve_weight

## Issues

- 2026-04-02T20:45-0700 Initial resolve_weight added data_offset to already-absolute TensorInfo.offset, causing mmap out-of-bounds. Fixed by using offset directly.

## Research & Discoveries

- Q6_K dequantization processes 256 values in 2 passes of 128. Inner loop of 32 produces 4 values each from interleaved ql (low 4 bits) and qh (high 2 bits). Values are 6-bit unsigned offset by -32.
- LFM2 shortconv: `(B,C,x) = in_proj(h); bx = B⊙x; conv1d(bx); C⊙conv_out; out_proj`
- The conv rolling buffer stores pre-conv bx values, NOT raw hidden states
- ggml_ssm_conv is valid (no-padding) depthwise conv — for single token it's a per-channel dot product

## Performance

Release build (Apple Silicon Mac):
| Model | Prefill | Decode |
|-------|---------|--------|
| LFM2-VL-450M Q8_0 | 11.8 tok/s | 25.1 tok/s |
| LFM2-VL-450M Q4_0 | 4.4 tok/s | 4.9 tok/s |
| LFM2.5-VL-1.6B Q4_0 | 1.4 tok/s | 1.6 tok/s |

## Commits

HEAD — feat: implement LFM2 forward pass, sampling, generation loop, and `wick run` CLI
