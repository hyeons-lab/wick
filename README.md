# Wick

Rust-native LLM inference engine. Load a GGUF, generate text, make it fast.

> **Note:** This project is a learning experiment — built to explore LLM inference internals, GGUF parsing, quantization, and SIMD/GPU compute in Rust. Not intended for production use.

## Benchmarks

Measured on Apple M-series (aarch64), single-socket. All models loaded from GGUF with memory-mapped weights.

### Decode (single-token generation)

| Model | Quant | tok/s |
|-------|-------|------:|
| LFM2-450M | Q4_0 | 119 |
| LFM2-450M | Q8_0 | 107 |
| LFM2.5-1.6B | Q4_0 | 57 |
| LFM2.5-1.6B | Q8_0 | 51 |

### Prefill (prompt processing)

| Model | Quant | 32 tok | 117 tok |
|-------|-------|-------:|--------:|
| LFM2-450M | Q4_0 | 475 | 539 |
| LFM2-450M | Q8_0 | 407 | 451 |
| LFM2.5-1.6B | Q4_0 | 160 | 191 |
| LFM2.5-1.6B | Q8_0 | 131 | 158 |

Q4_0 is faster than Q8_0 for both decode and prefill (less weight data to read per row), matching llama.cpp behavior. Prefill scales well with prompt length due to batched GEMM amortizing weight reads across all tokens.

### GPU backends

Two GPU backends with runtime selection via `--device`:

**Native Metal** (`--device metal`, macOS/iOS) — hand-written MSL shaders, single-encoder dispatch, GPU argmax. Beats llama.cpp across all tested models and context lengths.

**wgpu** (`--device gpu`, cross-platform) — WGSL shaders targeting Metal/Vulkan/DX12/WebGPU. Portable but slower due to API translation overhead.

#### Decode throughput vs llama.cpp (greedy, M1 Max)

| Model | Context | llama.cpp | wick Metal | wick wgpu | wick CPU |
|-------|---------|----------:|-----------:|----------:|---------:|
| LFM2-450M | tg128 | 301 | **379** (+26%) | 75 | 120 |
| LFM2-450M | tg512 | 325 | **345** (+6%) | 70 | — |
| LFM2.5-VL-1.6B | tg128 | 262 | **278** (+6%) | — | — |
| LFM2.5-VL-1.6B | tg512 | 200 | **242** (+21%) | — | — |
| LFM2.5-Audio-1.5B | tg128 | 267 | **278** (+4%) | — | — |
| LFM2.5-Audio-1.5B | tg512 | 201 | **253** (+26%) | — | — |

Measured with `wick bench --runs 20 --warmup 3` (sustained in-process, p50 reported). llama.cpp numbers from `llama-bench -r 10` (9438fcb27).

#### Key Metal optimizations

- **GPU argmax** — greedy sampling on GPU, avoids 256KB logits readback (+57%)
- **Q6_K native embedding GEMV** — reads 52 MB Q6_K bytes directly, no f32 dequant
- **llama.cpp-derived fast Q4_0 GEMV** — pre-scaled y, uint16 nibble loads, sumy bias hoisting
- **Fused gate+up GEMV** — single dispatch for both FFN projections
- **Fused QK norm + RoPE** — 3 dispatches → 1 per attention layer
- **Vectorized attention V loads** — float2 loads in weighted-sum phase
- **Residual accumulate in GEMV** — `y += W×x` instead of separate add

#### Key wgpu optimizations

- **Compute pass batching** — 300 passes → ~80 per token (+30%)
- **Fast Q4_0 GEMV** — ported Metal algorithm to WGSL with subgroupAdd
- **Multi-row f32 GEMV** — 8 rows per workgroup, 8× less input bandwidth

### Key optimizations

- **Batched GEMM prefill** — reads each weight matrix once for all N tokens (vs N times with per-token GEMV)
- **8-column grouped Q4_0 GEMM** — decode weight blocks once, dot against 8 input columns
- **Integer Q4_0/Q8_0/Q6_K GEMV** — quantize activations to Q8_0, integer dot product via `vdotq_s32`
- **Pre-quantize shared inputs** — one Q8_0 quantization reused across Q/K/V and gate/up projections
- **NEON attention** — vectorized Q*K scores and softmax*V weighted sums with `vfmaq_f32`
- **3-phase batched prefill** — batch input projections (GEMM) -> sequential core (conv/attention) -> batch output projections (GEMM)
- **Software prefetch** in GEMV/GEMM inner loops

## Features

- **GGUF model loading** with memory-mapped tensors
- **Three compute backends** — CPU (NEON SIMD), native Metal (MSL), wgpu (WGSL/Vulkan/Metal/DX12)
- **Hybrid architectures** — LFM2/LFM2.5 (gated conv + grouped query attention)
- **Quantization** — Q4_0, Q8_0, Q6_K, Q4_K_M
- **Built-in BPE tokenizer** — no Python, no runtime dependencies
- **Bench harness** — `wick bench` with p10/p50/p90/stddev for reproducible A/B comparisons
- **Chat mode** with Jinja2 chat template rendering
- **Single static binary**

## Build

```bash
# Debug build
cargo build --workspace

# Optimized release build (LTO, stripped)
just release

# Run
cargo run --release -p wick-cli -- run --model model.gguf --prompt "Hello, world!"
```

## Usage

```bash
# Generate text
wick run --model model.gguf --prompt "What is Rust?"

# Inspect a model file
wick inspect --model model.gguf

# Interactive chat
wick chat --model model.gguf

# Tokenize (for debugging)
wick tokenize --model model.gguf "Hello world"
```

## Architecture

Two-crate workspace:

- **`wick`** — core library (GGUF parsing, quantization, compute backends, models, tokenizer)
- **`wick-cli`** — CLI binary (clap, dispatches to `wick`)

Compute backends: scalar CPU → NEON SIMD (aarch64) → native Metal (macOS/iOS) → wgpu (cross-platform GPU).

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT License](LICENSE-MIT) at your option.
