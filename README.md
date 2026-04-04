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

### GPU backend (experimental, wgpu/Metal)

Cross-platform GPU inference via wgpu (Metal on macOS, Vulkan on Linux, DX12 on Windows, WebGPU in browsers). Feature-gated behind `gpu`.

| Backend | Prefill | Decode |
|---------|--------:|-------:|
| llama.cpp (native Metal, reference) | 655 tok/s | 158 tok/s |
| wick CPU (NEON Q4_0) | 197 tok/s | 98 tok/s |
| **wick GPU (wgpu Metal)** | **38 tok/s** | **51 tok/s** |

The wgpu GPU backend is **3x slower than llama.cpp's native Metal** and **2x slower than our own NEON CPU path** on Apple Silicon. The gap comes from:
- wgpu's per-dispatch validation/state-tracking overhead (Rust-side)
- WGSL → MSL translation produces less optimal code than hand-written MSL
- No simdgroup matrix ops in WGSL (llama.cpp uses them for GEMV)
- Shared memory bus on Apple Silicon means GPU gets no bandwidth advantage

The wgpu backend's value is **portability** — it runs on Linux/Windows discrete GPUs and WebGPU browsers where NEON isn't available. On M-series Macs, use the CPU backend for best performance.

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
- **CPU inference** with SIMD-optimized kernels (NEON dotprod on aarch64)
- **Hybrid architectures** — LFM2/LFM2.5 (gated conv + grouped query attention)
- **Quantization** — Q4_0, Q8_0, Q6_K, Q4_K_M
- **Built-in BPE tokenizer** — no Python, no runtime dependencies
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

Compute backends: scalar CPU -> NEON SIMD (aarch64) -> wgpu GPU (planned).

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT License](LICENSE-MIT) at your option.
