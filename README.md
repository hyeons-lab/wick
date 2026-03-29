# Wick

Rust-native LLM inference engine. Load a GGUF, generate text, make it fast.

## Features

- **GGUF model loading** with memory-mapped tensors
- **CPU inference** with SIMD-optimized kernels (AVX2, NEON)
- **GPU inference** via wgpu (Vulkan, Metal, D3D12, WebGPU)
- **Hybrid architectures** — LFM2 (conv+attention) and LLaMA-family models
- **Quantization** — Q4_K_M and Q8_0
- **Built-in BPE tokenizer** — no Python, no runtime dependencies
- **Single static binary**

## Build

```bash
# Build
cargo build --workspace

# Run
cargo run --bin wick -- run -m model.gguf -p "Hello, world!"

# With GPU support
cargo build --workspace --features gpu
```

## Usage

```bash
# Generate text
wick run -m model.gguf -p "What is Rust?"

# Inspect a model file
wick inspect -m model.gguf

# Interactive chat
wick chat -m model.gguf

# Benchmark
wick bench -m model.gguf
```

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT License](LICENSE-MIT) at your option.
