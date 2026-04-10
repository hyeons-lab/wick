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

Measured with `wick bench --runs 20 --warmup 3` (sustained in-process, p50 reported). llama.cpp numbers from `llama-bench -r 10` on the Liquid4All fork (9438fcb27).

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

## TurboQuant KV Cache Compression

Wick includes the **first implementation of the TurboQuant algorithm** ([arXiv:2504.19874](https://arxiv.org/abs/2504.19874), Google Research 2025) **for LFM2 architectures**, compressing **both keys and values**. The Liquid4All llama.cpp fork already supports simpler quantized KV caches (q4_0, q4_1, q8_0) for LFM2; TurboQuant is a newer, more sophisticated algorithm that offers better accuracy-per-bit via a data-oblivious rotation + residual correction pipeline.

### What it is

TurboQuant compresses KV cache **keys to ~3 bits/element** and **values to ~2 bits/element** (together ~12× reduction vs f32) with near-lossless accuracy and **no calibration or fine-tuning** required.

**Keys (3-bit)** — two-stage compression:
1. **PolarQuant (2-bit)** — Randomized Hadamard Transform rotates each key vector, then quantizes coordinates to 4 Lloyd-Max centroids for the resulting Beta distribution. The rotation makes the distribution predictable, so no per-block scale factors are needed.
2. **QJL (1-bit)** — Quantized Johnson-Lindenstrauss sign bits on the residual provide an *unbiased* inner-product estimator that corrects PolarQuant bias during attention score computation.

**Values (2-bit)** — PolarQuant only. The attention operation on values is a weighted sum (not an inner product), so the QJL residual estimator doesn't apply — values need actual vector reconstruction, which PolarQuant gives directly. The weighted sum is computed in *rotated* 2-bit centroid space; a single `rht_inverse` per attention head at the end recovers the original basis (exploiting linearity of the transform). This makes values even cheaper than keys while maintaining good quality.

### How it compares

| Approach | Bits per key | Bits per value | Calibration | Unbiased estimator |
|----------|---:|---:|:-:|:-:|
| f32 (wick default) | 32 | 32 | — | — |
| f16 (llama.cpp default) | 16 | 16 | — | — |
| llama.cpp q8_0 KV | 8 | 8 | — | — |
| llama.cpp q4_0 KV | 4 | 4 | — | — |
| **wick TurboQuant tq3** | **3** | **2** | **no** | **yes** (keys) |

TurboQuant's differentiators:

- **Fewer bits per element** — 3+2 bits/KV vs q4_0's 4+4 bits/KV, via the rotation that eliminates the need for per-block scales.
- **Unbiased attention score estimator** — the QJL correction term is provably unbiased for keys, while q4_0/q8_0 introduce bias that can compound across long contexts.
- **Data-oblivious** — no calibration pass or fine-tuning; works on any model at any time.

### Memory footprint

| Format | Bytes/key + value (head_dim=128) | Compression |
|--------|---:|---:|
| f32 (wick default) | 512 + 512 = 1024 | 1× |
| f16 | 256 + 256 = 512 | 2× |
| **TurboQuant tq3** | **52 + 34 = 86** | **~12×** |

For a 1.6B LFM2 model with 6 attention layers, n_kv_heads=8, at 4096 tokens:
- **Uncompressed:** ~192 MB
- **TurboQuant tq3:** ~16 MB

Savings scale linearly with context length — at 8K+ tokens the KV cache dominates and savings approach the theoretical ~12×.

### Throughput

After full optimization (NEON SIMD, GQA batching, zero heap in hot path, fused encode pipeline), **TurboQuant decode is within ±5% of wick f32** and sometimes *faster* (the GQA-batched attention with pre-computed scratch amortizes better than the original f32 path at short contexts). Values reuse the same NEON kernel pattern as keys for the hot inner loop.

### Backend support

**TurboQuant is currently implemented only on the CPU backend (`Lfm2Model`).** The Metal and wgpu GPU backends do not yet honor the compressed caches — passing `--kv-cache-keys tq3` to a GPU run will log a warning and fall back to f32 KV. See the CPU-only limitation note in the memory-footprint comparison above if you need to pick a backend.

### CLI: enabling TurboQuant

Both `wick run` and `wick bench` accept `--kv-cache-keys` (the flag name is kept for backwards compatibility; it now covers values too):

```bash
# Uncompressed (default) — keys and values stored as f32
wick run -m lfm2.gguf -p "Hello" --kv-cache-keys f32 --device cpu

# Full TurboQuant — 3-bit keys + 2-bit values (production default, CPU backend only)
wick run -m lfm2.gguf -p "Hello" --kv-cache-keys tq3 --device cpu

# Keys only — 3-bit keys, values stay f32 (debugging: isolate key error)
wick run -m lfm2.gguf -p "Hello" --kv-cache-keys tq3-keys --device cpu

# Values only — values stay 2-bit, keys stay f32 (debugging: isolate value error)
wick run -m lfm2.gguf -p "Hello" --kv-cache-keys tq3-values --device cpu
```

Accepted values for `--kv-cache-keys`:

| Value | Keys | Values | Use case |
|-------|:-:|:-:|---|
| `f32` / `none` | f32 | f32 | Baseline, no compression (default) |
| `tq3` / `turboquant` | 3-bit | 2-bit | Production — both sides compressed |
| `tq3-keys` | 3-bit | f32 | Debug: measure key-only drift |
| `tq3-values` | f32 | 2-bit | Debug: measure value-only drift |

The same flag applies to `wick bench` for A/B benchmarking:

```bash
wick bench -m lfm2.gguf --kv-cache-keys tq3 --max-tokens 256 --device cpu
```

### Programmatic API

TurboQuant is configured in a **single call** — construct a `KvCompression` and pass it to `InferenceState::from_config_with_compression` or via `GenerateConfig`. No separate `enable_turboquant` call is needed; the rotation state, compressed caches, and scratch buffers are all set up on the `InferenceState` from the same configuration.

```rust
use wick::kv_cache::{InferenceState, KvCompression};

// Production: both sides compressed. The single `seed` drives the
// per-layer randomized Hadamard rotations deterministically.
let state = InferenceState::from_config_with_compression(
    model.config(),
    &KvCompression::turboquant(42),
);

// Or via the bench/engine config:
let gen_cfg = wick::engine::GenerateConfig {
    kv_compression: KvCompression::turboquant(42),
    ..Default::default()
};
```

All four modes:

```rust
// Disabled — f32 KV for both keys and values (default)
let cfg = KvCompression::None;

// Both compressed — the shortcut wraps the common case
let cfg = KvCompression::turboquant(42);

// Or with explicit key/value flags (matches the CLI modes)
let cfg = KvCompression::TurboQuant { seed: 42, keys: true,  values: true  }; // == tq3
let cfg = KvCompression::TurboQuant { seed: 42, keys: true,  values: false }; // == tq3-keys
let cfg = KvCompression::TurboQuant { seed: 42, keys: false, values: true  }; // == tq3-values
```

To check whether a loaded model's backend supports TurboQuant, call `model.turboquant_supported()` before asking for compression — it returns `false` on GPU backends and on any model whose `head_dim` isn't a power of 2 (the Walsh-Hadamard transform requires it).

See `wick/src/turboquant.rs` for the implementation and `wick/src/kv_cache.rs` for the `KvCompression` enum.

## Features

- **GGUF model loading** with memory-mapped tensors
- **Three compute backends** — CPU (NEON SIMD), native Metal (MSL), wgpu (WGSL/Vulkan/Metal/DX12)
- **Hybrid architectures** — LFM2/LFM2.5 (gated conv + grouped query attention)
- **Quantization** — Q4_0, Q8_0, Q6_K, Q4_K_M for weights
- **TurboQuant KV cache compression** — ~12× KV reduction (3-bit keys + 2-bit values), unbiased attention estimator, first TurboQuant implementation for LFM2
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

### Module layout

```
wick/src/
├── gguf.rs              # mmap-based GGUF parser (zero-copy tensor access)
├── tensor.rs            # DType enum (F32, F16, BF16, Q4_0, Q8_0, Q4_K_M, Q6_K)
├── quant.rs             # Block structs + scalar dequant/vec_dot kernels
├── turboquant.rs        # TurboQuant KV cache compression (PolarQuant + QJL)
├── kv_cache.rs          # InferenceState, LayerState (attention/conv), scratch buffers
├── tokenizer.rs         # BPE tokenizer + minijinja chat template rendering
├── sampler.rs           # Temperature/top-k/top-p sampling, greedy fast path
├── engine.rs            # generate() loop: prefill → decode
├── backend/
│   ├── cpu.rs           # Scalar reference + NEON attention kernels
│   ├── simd.rs          # NEON (aarch64) + AVX2 (x86_64) GEMV/GEMM kernels
│   ├── wgpu.rs          # wgpu cross-platform GPU backend
│   └── metal.rs         # Native Metal backend (MSL shaders)
└── model/
    ├── mod.rs           # Model trait, ModelConfig, BlockType enum
    ├── lfm2.rs          # LFM2 hybrid (conv + attention), CPU path
    ├── gpu_lfm2.rs      # LFM2 on wgpu
    ├── metal_lfm2.rs    # LFM2 on native Metal
    └── llama.rs         # LLaMA-family stub (stub; future work)
```

### Model loading

GGUF files are memory-mapped via `memmap2`. Two tensor access patterns:

- `get_tensor(name)` — copies data into an owned `Tensor` (f32 dequantize for small weights like norms)
- `tensor_data(name)` — returns a zero-copy `&[u8]` slice into the mmap for quantized weights

All offsets validated with checked arithmetic (`checked_add`, `usize::try_from`).

### Quantization formats

| Format | Block size | Bytes/block | Use case |
|--------|---:|---:|----------|
| Q4_0 | 32 values | 18 B | Most weight matrices |
| Q8_0 | 32 values | 34 B | Activations (dynamic quantization for integer GEMV) |
| Q4_K_M | 256 values | 144 B | Higher-quality 4-bit with sub-block scales |
| Q6_K | 256 values | 210 B | Embedding matrices (Metal backend) |

Each format has `dequantize_*` (block → f32) and `vec_dot_*` (dot product without full dequant) paths. SIMD variants in `backend/simd.rs` use `vdotq_s32` on aarch64 and AVX2 on x86_64.

### Compute backend tiers

1. **Scalar CPU** (`backend/cpu.rs`) — reference implementations operating on raw `&[f32]` slices, no `Tensor` in the hot path
2. **NEON SIMD** (aarch64) — vectorized Q4_0/Q8_0/Q6_K/Q4_K_M GEMV, attention scores, attention values
3. **AVX2 SIMD** (x86_64) — dot product kernels for dense dispatch
4. **Native Metal** (macOS/iOS only, `backend/metal.rs`) — hand-written MSL shaders, single-encoder dispatch, GPU argmax
5. **wgpu** (cross-platform) — WGSL shaders targeting Metal/Vulkan/DX12/WebGPU via the wgpu crate

Runtime dispatch via `--device` flag: `cpu | metal | gpu | auto`.

### Hybrid model support (LFM2/LFM2.5)

Unlike pure transformers, LFM2 interleaves two block types per layer:

- **Gated convolution blocks** — depthwise 1D conv with gating, rolling buffer for O(1) decode
- **Grouped query attention blocks** — standard GQA with per-head QK RMSnorm and RoPE

`ModelConfig.block_types: Vec<BlockType>` specifies the per-layer pattern. `LayerState` tracks either a KV cache (attention) or a conv rolling buffer. TurboQuant only applies to attention layers.

### Inference loop

```
generate() in engine.rs:
  1. Prefill: forward_prefill(prompt_tokens, ...)
     - Batched GEMM for Q/K/V/FFN projections (read weights once)
     - Per-token attention + conv (sequential)
     - Batched GEMM for output projection
  2. Decode loop:
     - forward(next_token, ...)
     - Per-token GEMV (reads weights every token)
     - Sampler picks next token (greedy fast path avoids logits readback)
```

See the benchmark tables above for per-backend performance.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT License](LICENSE-MIT) at your option.
