# Wick — Implementation Plan

A Rust-native LLM inference engine. Load a GGUF, generate text, make it fast.

---

## Guiding Principles

1. **Build for the hard case first.** LFM2's hybrid conv+attention architecture is more complex than LLaMA. If the abstractions handle LFM2, every pure-transformer model falls out for free.
2. **Two crates, not nine.** `wick` (library) and `wick-cli` (binary). Split later when API boundaries are stable. Every additional crate is compile-time overhead and API surface to maintain.
3. **No CUDA in v1.** wgpu gives us Vulkan, Metal, D3D12, and WebGPU from one set of WGSL shaders. Accept the 10-20% gap vs cuBLAS on datacenter GPUs. Add CUDA as a v2 backend if demand warrants.
4. **Two quant types, not twenty.** Q4_K_M and Q8_0 cover >90% of models people actually download. Each quant type requires a dequant kernel × every backend. Expand later.
5. **Own the tokenizer.** Write a minimal BPE implementation (~300 lines) instead of pulling in the HF `tokenizers` crate (15+ deps, doesn't compile to WASM). LFM2's byte-level BPE with 65K vocab is simple.
6. **Correctness first, then speed.** Naive implementations → verify against llama.cpp → then optimize with SIMD/GPU. Never optimize unverified code.

---

# V1 — "Load a GGUF, generate text, make it fast"

**Target: 6-8 weeks.** One developer + Claude Code.

**End state:** `wick run -m LFM2.5-1.2B-Q4_K_M.gguf -p "Hello"` generates coherent text at 15-30+ tok/s on CPU with SIMD, 40+ tok/s on GPU via wgpu. Supports LFM2 and LLaMA-family models. Single static binary, no Python, no runtime dependencies.

---

## Phase 0: Scaffold ✅
**Time: 1 day**

```
0.1  Create the workspace:

     wick/
     ├── Cargo.toml              # workspace root
     ├── wick/                  # library crate (everything lives here)
     │   ├── Cargo.toml
     │   └── src/
     │       ├── lib.rs
     │       ├── tensor.rs       # Tensor types, dtypes, storage
     │       ├── quant.rs        # Q4_0, Q4_K_M, Q8_0 block dequantization
     │       ├── gguf.rs         # GGUF file parser
     │       ├── tokenizer.rs    # Minimal BPE tokenizer
     │       ├── sampler.rs      # Sampling strategies
     │       ├── backend/
     │       │   ├── mod.rs      # Backend trait
     │       │   ├── cpu.rs      # CPU compute (SIMD)
     │       │   ├── simd.rs     # SIMD-optimized kernels (NEON, AVX2)
     │       │   └── wgpu.rs     # wgpu compute (GPU)
     │       ├── model/
     │       │   ├── mod.rs      # Model trait + loader dispatch
     │       │   ├── lfm2.rs     # LFM2 / LFM2.5
     │       │   └── llama.rs    # LLaMA / Mistral / Qwen / Gemma / Phi
     │       ├── kv_cache.rs     # KV cache (simple contiguous, then paged)
     │       └── engine.rs       # Top-level generate() orchestration
     └── wick-cli/              # CLI binary
         ├── Cargo.toml
         └── src/main.rs

0.2  Workspace Cargo.toml:
     - edition = "2024", rust-version = "1.85"
     - Feature flags: "wgpu" (optional GPU backend)
     - Workspace dependencies: anyhow, thiserror, tracing, byteorder,
       serde, serde_json, half, memmap2, clap, minijinja, bytemuck

0.3  .cargo/config.toml:
     - Release profile: LTO = "thin", codegen-units = 1
     - Target-specific RUSTFLAGS for native CPU features

0.4  justfile:
     - just build, just test, just run -- <args>, just bench, just ci

0.5  README.md, LICENSE (Apache-2.0 + MIT), .gitignore
```

---

## Phase 1: Tensor + CPU Compute ✅
**Time: 5-7 days**

```
1.1  tensor.rs — Core types:

     pub enum DType { F32, F16, BF16, I32, U8, Q4_0, Q4KM, Q8_0 }

     pub struct Tensor {
         data: Vec<u8>,       // raw bytes
         shape: Vec<usize>,
         dtype: DType,
     }

     Methods: shape(), dtype(), numel(), size_bytes(),
     to_f32_vec(), from_f32_vec(), as_f32_slice(), zeros_f32()

1.2  quant.rs — Q4_0, Q4_K_M and Q8_0:

     Q4_0 block (18 bytes):
       d: f16                  // scale
       qs: [u8; 16]            // 32 4-bit unsigned values, offset by -8

     Q8_0 block (34 bytes):
       delta: f16              // scale
       quants: [i8; 32]        // 32 signed 8-bit values

     Q4_K_M block (144 bytes):
       d: f16                  // super-block scale
       dmin: f16               // super-block min
       scales: [u8; 12]        // sub-block scales and mins (packed)
       qs: [u8; 128]           // 256 4-bit quants (128 bytes)

     Implement dequantize and vec_dot for each.

1.3  backend/cpu.rs — Naive reference implementations:
     fn matmul_f32, matmul_q4_0_f32, matmul_q8_0_f32, matmul_q4km_f32
     fn rmsnorm, silu_inplace, softmax_inplace
     fn rope, conv1d_depthwise
     fn add_inplace, mul_inplace

1.4  backend/simd.rs — SIMD-optimized vec_dot:
     NEON (aarch64) and AVX2 (x86_64) implementations
     with compile-time / runtime dispatch.
```

---

## Phase 2: GGUF Parser + Tokenizer ✅
**Time: 3-4 days**

```
2.1  gguf.rs — Parser:
     - Parse header: magic (0x46554747), version, tensor_count, kv_count
     - Parse KV metadata: all 13 GGUF value types
     - Parse tensor info: name, dims, dtype (with raw ggml_type_id), offset
     - Memory-map tensor data with memmap2 (zero-copy)
     - get_tensor(), tensor_data(), print_inspect()

2.2  wick inspect CLI command — dumps metadata + tensor info

2.3  tokenizer.rs — Minimal BPE:
     - Load vocab + merges from GGUF metadata
     - Byte-level BPE encode/decode
     - Special token detection from token_type array
     - Chat template rendering via minijinja

2.4  wick tokenize CLI command + Python comparison script
```

---

## Phase 3: LFM2 Forward Pass
**Time: 7-10 days**

Build LFM2 FIRST. This is the hard case. LLaMA comes after, trivially.

```
3.1  Determine LFM2 GGUF tensor naming:
     BEFORE writing any model code, run `wick inspect` on the LFM2 GGUF
     and document every tensor name and shape.

     Known from real LFM2-VL-450M inspection:
     - Conv blocks: blk.N.shortconv.{in_proj,conv,out_proj}.weight
     - Attn blocks: blk.N.attn_{q,k,v}.weight, blk.N.attn_{q,k}_norm.weight,
       blk.N.attn_output.weight
     - All blocks: blk.N.attn_norm.weight, blk.N.ffn_{gate,up,down}.weight,
       blk.N.ffn_norm.weight
     - Global: token_embd.weight, token_embd_norm.weight
     - Note: lfm2.attention.head_count_kv is an i32 array (per-layer), not scalar

3.2  model/mod.rs — Model loading dispatch:

     pub struct ModelConfig { ... }
     pub enum BlockType { Attention, GatedConv }
     pub trait Model: Send { fn forward(...), fn config() }
     pub fn load_model(gguf: &GgufFile) -> Result<Box<dyn Model>>

3.3  kv_cache.rs — Simple contiguous KV cache (NOT paged yet).

3.4  model/lfm2.rs — LFM2 model struct + forward pass

3.5  sampler.rs — greedy, temperature, top_k, top_p, sample

3.6  engine.rs — Generation loop with prefill + decode

3.7  wick run CLI command

3.8  Correctness validation against llama.cpp
```

---

## Phase 4: LLaMA + Additional Architectures
**Time: 3-5 days**

```
4.1  model/llama.rs — LLaMA is all-attention blocks.
4.2  Architecture variants: mistral, qwen2, gemma, phi3
4.3  Test each on a real GGUF. Greedy decoding matches llama.cpp.
```

---

## Phase 5: wgpu GPU Backend
**Time: 10-14 days**

```
5.1  backend/wgpu.rs — Device init, buffer pool, weight upload.
5.2  WGSL shaders: matmul, quantized matmul, rmsnorm, silu, rope, softmax,
     attention, conv1d, element-wise ops
5.3  Subgroup-enhanced variants (feature-detect at init)
5.4  Full GPU forward pass: single CommandEncoder, read back logits only.
5.5  CLI: --device gpu/cpu/auto. Benchmark CPU vs GPU.

     Note: V1 shaders use fixed workgroup sizes. Per-shape kernel tuning
     (V2.7) adds profile-guided dispatch for decode GEMV — significant
     wins on AMD RDNA3 (see kernel-anvil results: 2.25x on 7900 XTX).
     Design shader dispatch to accept configurable workgroup params from
     the start so V2.7 is a config change, not a rewrite.
```

---

## Phase 6: Polish v1 for Release
**Time: 3-5 days**

```
6.1  HuggingFace model download: wick run -m LiquidAI/LFM2.5-1.2B-Instruct
6.2  Interactive chat mode: wick chat -m model.gguf
6.3  Benchmark command: wick bench -m model.gguf
6.4  Correctness: perplexity on WikiText-2 for Q4_K_M and Q8_0
6.5  CI + static binary releases (Linux, macOS, Windows)
6.6  README with benchmarks, install instructions, supported models
```

---

# V1 Complete. Everything below is V2.

---

# V2 — Roadmap

Ordered by estimated impact. Many can be worked in parallel.

### V2.1: Server + Continuous Batching — 3-4 weeks
OpenAI-compatible HTTP server (axum + SSE), continuous batching scheduler, paged attention (replaces contiguous KV cache), request queue, Prometheus metrics, preemption.

### V2.2: Browser / WASM — 3-4 weeks (parallel with V2.1)
WASM build (dual: threaded + single-threaded), wasm-bindgen-rayon for multi-threaded CPU, Web Worker architecture, OPFS model caching, JS API + npm package, Chrome enhanced (subgroups, dot4U8Packed, f16), Safari baseline (f16, standard WGSL), feature detection.

### V2.3: Structured Output — 1-2 weeks
GBNF grammar parser, JSON schema → grammar compiler, regex constraints, async FSM mask computation overlapped with forward pass.

### V2.4: KV Cache Serialization — 1-2 weeks
Serialize KV cache + conv buffers to .lmkv files, system prompt caching, conversation checkpointing, KV quantization for storage.

### V2.5: Prefix Caching (Radix Attention) — 1-2 weeks
Radix tree for in-memory prefix matching, LRU eviction, scheduler integration. 5-6x speedup on prefix-heavy workloads.

### V2.5b: TurboQuant KV Cache Compression — 1-2 weeks
Google Research's data-oblivious KV cache compression (ICLR 2026). Compresses KV cache to 3-3.5 bits with zero accuracy loss.

### V2.6: More Quantization Formats — 1 week per format
Q2_K through Q6_K, IQ quants, GPTQ, AWQ, FP8, in-situ quantization.

### V2.7: Per-Shape Kernel Tuning (GEMV/MMVQ) — 1-2 weeks
Profile-guided kernel optimization for quantized decode (batch=1 GEMV). Instead of using one-size-fits-all thread/block configs for all layers, profile each unique (quant_type, N, K) shape on the target GPU and apply optimal nwarps/rows_per_block at runtime. Inspired by [kernel-anvil](https://github.com/apollosenvy/kernel-anvil) which demonstrated 2.25x decode speedup on Qwen3.5-27B Q4_K_M (12→27 tok/s on RX 7900 XTX) by auto-tuning llama.cpp's MMVQ kernels per model shape. Key insight: a 1024-row GQA projection and a 17408-row FFN layer have very different optimal configs. The bottleneck classification (bandwidth-bound vs occupancy-limited vs compute-bound) determines the sweep strategy. For wick: implement shape-aware dispatch in wgpu compute shaders (WGSL workgroup size, rows per invocation) and optionally in CPU SIMD (loop tiling). Store per-model configs as JSON; profile on first run or via `wick tune` command.

### V2.8: Speculative Decoding — 1-2 weeks
Draft model + verification, self-speculative. 1.3-2x decode speedup.

### V2.9: LoRA Adapters — 1-2 weeks
Runtime LoRA loading, merge/unmerge, per-request LoRA selection.

### V2.10: MoE Support — 2-3 weeks
Top-K expert routing for Mixtral, LFM2-8B-A1B, LFM2-24B-A2B.

### V2.11: Multi-GPU — 3-4 weeks
Pipeline parallelism, tensor parallelism, CPU offloading.

### V2.12: CUDA Backend — 3-4 weeks
Optional cuBLAS + FlashAttention + CUDA graphs. Requires nvcc.

### V2.13: Python Bindings — 1-2 weeks
PyO3 bindings, `pip install wick-engine`.

### V2.14: Kotlin Multiplatform Bindings — 2-3 weeks
C ABI via cbindgen + platform-native FFI per KMP target (cinterop, Panama FFM, PanamaPort, JS interop).

---

## V2 Prioritization

**Local inference on laptop:** V1 is sufficient. Add V2.6 for more quants, V2.7 for per-shape tuning.

**Production API server:** V2.1 → V2.5 → V2.5b (TurboQuant) → V2.3

**Browser inference (differentiator):** V2.2 → V2.5b (TurboQuant) → V2.4 → V2.3

**Mobile / on-device apps:** V2.14 → V2.5b (TurboQuant) → V2.4 (KV serialization) → V2.3

**AMD GPU performance:** V2.7 (per-shape tuning) → V2.6 (more quants) → V2.8 (speculative)

**Long-context use cases (32K+):** V2.5b (TurboQuant) → V2.1 (paged attention) → V2.5 (prefix caching)

**Largest models:** V2.10 → V2.11 → V2.12

---

## Dependencies (V1)

```toml
[dependencies]
anyhow = "1"
thiserror = "2"
tracing = "0.1"
tracing-subscriber = "0.3"
byteorder = "1"
bytemuck = "1"
half = "2"
memmap2 = "0.9"
clap = { version = "4", features = ["derive"] }
minijinja = "2"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
rand = "0.8"

# Optional GPU backend
wgpu = { version = "24", optional = true }

[features]
default = []
gpu = ["dep:wgpu"]
```

> **Note:** The `wgpu` dependency and `gpu = ["dep:wgpu"]` feature shown above are
> illustrative of the planned V2 layout. The current `wick/Cargo.toml` has `gpu = []`
> as a placeholder with no `wgpu` dependency wired in yet.

No `tokenizers`, no `rayon`, no `axum`, no `tokio`, no `wasm-bindgen`.
Add these in v2 modules that need them.

---

## Claude Code Session Plan (V1)

| Session | Phase | Goal |
|---------|-------|------|
| 1 | 0 | Scaffold workspace, all files created, compiles ✅ |
| 2 | 1a | Tensor types, Q4_0/Q4_K_M/Q8_0 dequantization, tests ✅ |
| 3 | 1b | Naive CPU matmul + all element-wise ops, tests ✅ |
| 4 | 1c | SIMD matmul (AVX2 + NEON), benchmarks ✅ |
| 5 | 2a | GGUF parser, inspect command, test with real file ✅ |
| 6 | 2b | BPE tokenizer, chat templates, test against HF ✅ |
| 7 | 3a | LFM2 model struct, from_gguf loading, tensor name mapping |
| 8 | 3b | LFM2 conv block forward, attention forward, KV cache |
| 9 | 3c | Full forward pass + sampling + generate loop. First text! |
| 10 | 3d | Debug until output matches llama.cpp reference |
| 11 | 4 | LLaMA model + 2-3 variants (Mistral, Qwen, Gemma) |
| 12 | 5a | wgpu init, naive matmul shader, test against CPU |
| 13 | 5b | Tiled matmul, quantized matmul, element-wise shaders |
| 14 | 5c | Attention + conv1d shaders, subgroup variants |
| 15 | 5d | Full GPU forward pass integration, benchmark |
| 16 | 6 | HF download, chat mode, bench command, CI, README |
