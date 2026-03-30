# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
just build              # debug build
just release            # optimized release build (LTO thin, stripped)
just test               # run all tests
just fmt                # format code
just clippy             # lint
just ci                 # full CI check: fmt + clippy + test

# Single test or module
cargo test -p wick -- <test_name>
cargo test -p wick <module>::tests       # e.g. quant::tests, gguf::tests

# CLI commands (working features)
cargo run -p wick-cli -- inspect <path.gguf>
cargo run -p wick-cli -- tokenize <path.gguf> "text"
```

**Always run `cargo fmt` before committing.** CI enforces `cargo fmt --check` and will fail on unformatted code.

## Architecture

Two-crate Cargo workspace:

- **`wick`** — core library (all inference logic)
- **`wick-cli`** — binary (clap CLI that dispatches to `wick`)

### GGUF Parsing (`gguf.rs`)

GGUF files are memory-mapped via `memmap2`. Two access patterns:
- `get_tensor(name)` — copies data from mmap into an owned `Tensor`
- `tensor_data(name)` — returns a zero-copy `&[u8]` slice into the mmap

Both validate offsets with checked arithmetic (`checked_add`, `usize::try_from`).

### Tensor & Quantization (`tensor.rs`, `quant.rs`)

`DType` enum covers dense types (F32, F16, BF16) and quantized types (Q4_0, Q4KM, Q8_0). Each quantized format has:
- A block struct (e.g. `BlockQ4_0`, `BlockQ4KM`, `BlockQ8_0`)
- `dequantize_*()` — block/row to f32
- `vec_dot_*()` — dot product without full dequantization

### Compute Backends (`backend/`)

Three tiers with runtime dispatch:
1. **`cpu.rs`** — scalar reference implementations operating on raw `&[f32]` slices (no Tensor in the hot path)
2. **`simd.rs`** — NEON (aarch64) and AVX2 (x86_64) optimized `vec_dot` kernels with compile-time + runtime dispatch
3. **`wgpu.rs`** — GPU backend placeholder (Phase 5)

### Models (`model/`)

`Model` trait with `forward()` and `config()`. `ModelConfig` supports per-layer `BlockType` (Attention or GatedConv) for hybrid architectures like LFM2. LLaMA and LFM2 model implementations are Phase 3-4 work.

### Tokenizer (`tokenizer.rs`)

Self-contained BPE tokenizer that loads vocab, merges, and special tokens directly from GGUF metadata. Chat template rendering via `minijinja`.

## Conventions

- Edition 2024, MSRV 1.85
- `.cargo/config.toml` sets native CPU feature flags per target architecture
- `gpu` feature flag exists but `wgpu` is not wired into `wick/Cargo.toml` yet (planned for V2)
- Error handling: `anyhow` with `ensure!` / `with_context()` / `bail!`
- Release profile: LTO thin, single codegen unit, stripped symbols
