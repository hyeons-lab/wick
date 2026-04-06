# 000010 — feature/gpu-backend

**Agent:** Claude Code (claude-opus-4-6) @ wick branch feature/gpu-backend

## Intent

Add wgpu GPU compute backend for cross-platform GPU inference (Metal, Vulkan, DX12, WebGPU). Phase 1: foundation + f32 GEMV shader. Later phases add quantized shaders and full forward pass.

## What Changed

- 2026-04-04T00:15-0700 `Cargo.toml` — Added `pollster = "0.4"` to workspace deps
- 2026-04-04T00:15-0700 `wick/Cargo.toml` — Wired `wgpu` and `pollster` as optional deps, `gpu = ["dep:wgpu", "dep:pollster"]`
- 2026-04-04T00:15-0700 `wick/src/backend/wgpu.rs` — GpuContext: device init, buffer upload/download, pipeline creation, f32 readback
- 2026-04-04T00:15-0700 `wick/src/backend/shaders/gemv_f32.wgsl` — F32 GEMV with workgroup parallel reduction (64 threads/row, shared memory reduction)
- 2026-04-04T00:15-0700 `wick/src/backend/shaders/elementwise.wgsl` — add_inplace, silu_mul_inplace

## Decisions

- 2026-04-04T00:15-0700 f32-first approach: dequantize weights on CPU, upload f32 to GPU. Gets e2e working before fighting WGSL byte alignment. Quantized shaders added in later phase.
- 2026-04-04T00:15-0700 pollster for blocking async: wgpu's adapter/device requests are async. pollster::block_on is the simplest bridge for V1.
- 2026-04-04T00:15-0700 64 threads/row GEMV: parallel reduction via shared memory. Each thread accumulates k/64 elements, then 6-step log2 reduction.

## Commits

- HEAD — feat: wgpu GPU backend foundation — device init, f32 GEMV shader, elementwise ops

## Next Steps

- Phase 2: rmsnorm, softmax, rope, attention, conv1d shaders
- Phase 2: GpuLfm2Model + GpuInferenceState + CLI --device dispatch
- Phase 3: Q4_0/Q8_0 quantized GEMV shaders
- Phase 4: prefill GEMM
