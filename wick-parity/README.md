# wick-parity

Parity harness for [wick](../wick) — runs the same prompt through every
binding leg and byte-compares the resulting greedy-decoded token
streams. Divergence between any two legs points to a marshalling bug
at the FFI layer.

The harness is built around three constraints:

1. **Greedy decoding only.** Temperature 0, top_k 1 — fully
   deterministic regardless of sampler RNG state. Sampling parity
   would just add seed-handling complexity without strengthening the
   bug-detection bar.
2. **CPU backend.** Token equality across CPU and Metal/wgpu requires
   bit-exact GPU kernels (which is not a goal); CPU is the
   lowest-common-denominator that every leg can run.
3. **Real model, real download.** Synthetic GGUFs would short-circuit
   the manifest resolution path that's the whole point of the FFI
   surface. We pull `LFM2-350M-Extract-GGUF/Q4_0` (~200 MB) from the
   `LiquidAI/LeapBundles` HF repo via [`BundleRepo::from_bundle_id`]
   on first run and cache it locally.

## Status

| Leg | What it exercises | Status |
|---|---|---|
| `rust` | `wick::WickEngine` directly (reference) | Shipped |
| `ffi` | `wick_ffi::WickEngine` through its Rust public surface | Shipped |
| `kotlin-jna` | Kotlin binding via JNA, in-process | Pending (roadmap PR 16) |
| `swift-uniffi` | Swift binding, native macOS / iOS Simulator | Pending (roadmap PR 17) |
| `kotlin-aidl` | Cross-process via `wick-serviceapp` AIDL | Pending (Phase 4.5) |

## CLI

```bash
# Run one leg, emit JSON to stdout
cargo run -p wick-parity -- dump --via rust \
    --bundle LFM2-350M-Extract-GGUF --quant Q4_0 \
    --prompt "The capital of France is" \
    --max-tokens 16

# Run every Rust-side leg and diff token-by-token
cargo run -p wick-parity -- check \
    --bundle LFM2-350M-Extract-GGUF --quant Q4_0 \
    --prompt "The capital of France is" \
    --max-tokens 16
# OK: rust ↔ ffi parity (bundle=LFM2-350M-Extract-GGUF quant=Q4_0 tokens=16)
```

`check` exits 0 on match, 1 with a windowed diff summary on
mismatch, 2 on any other error (load failure, network error, …).
The cache directory survives across `dump` ↔ `check` invocations so
the model downloads once.

### Cache root

The harness uses `target/tmp/wick-parity-cache/` under the workspace
root by default. Override with `--cache-dir <path>` (CLI) or
`$WICK_PARITY_CACHE_DIR` (env). Distinct from `wick-test-models/`
(used by `wick`'s `tests/common/download.rs`) so the two caches can
be wiped independently.

## Integration test

```bash
WICK_PARITY_RUN=1 cargo test -p wick-parity -- --ignored
```

Without the env var the test prints a skip line and returns; without
`--ignored` the test is filtered out by `#[ignore]`. Belt-and-suspenders
gating mirrors `wick/tests/shift_real_model.rs`.

CI runs this on push to `main` + manual `workflow_dispatch` only —
not on PRs. See `.github/workflows/ci.yml` job `Parity Harness
(gated)`. The cache is keyed
`wick-parity-cache-lfm2-350m-q4_0-v1`; bump the `-vN` suffix when
the bundle id or quant in `tests/parity.rs` changes.

## Adding a new leg

A new binding leg only has to satisfy two contracts:

1. **Take a [`RunArgs`]** (or its JSON-serialized
   [`RunArgsOwned`] mirror, for subprocess legs) and
2. **Return a `Vec<u32>`** of greedy-decoded token IDs in emission
   order, excluding prompt tokens and any tokenizer-injected `<bos>`.

For an in-process Rust leg, drop a `pub fn run_<name>(args: &RunArgs<'_>) -> Result<Vec<u32>>`
into [`src/lib.rs`] alongside `run_rust` / `run_ffi`, add a new
[`Leg`] variant in [`src/main.rs`], and extend `Cmd::Check` to diff
against it.

For a subprocess leg (Kotlin via JNA, Swift via UniFFI):

1. Build a small wrapper binary in `legs/<name>/` that:
   - Reads `RunArgsOwned` JSON from stdin.
   - Drives the binding to produce a `Vec<u32>`.
   - Emits `RunOutput` JSON on stdout.
2. Add a `--via <name>` value to the harness `Leg` enum and a
   `Leg::<Name> => spawn_subprocess(...)` arm in `run_leg`.
3. Add a CI step that builds the leg's wrapper before the
   `Run parity harness` step.

The token-stream contract is the only thing the harness assumes —
how a leg gets there (in-process, subprocess, IPC) is up to the leg.

## Out of scope

Things this crate intentionally does **not** do (yet):

- **Performance parity alarm.** A wall-clock-diff threshold (e.g.
  >15% across legs trips CI) is documented in
  `devlog/plans/000037-01-ffi-multitarget.md` Phase 2.5 but isn't
  worth wiring before we have more than one non-Rust leg to compare.
- **Layer-0 hidden state diff.** Catches divergence earlier than
  token equality but needs per-layer hooks across both APIs; not
  worth the surgery until token-equality misses a bug.
- **Sampling parity.** Top-k / top-p / temperature > 0. Greedy is
  sufficient for the construction-path correctness check.
- **Fixture rotation policy.** A new LFM2 bumping the cache key is
  fine ad-hoc; defer the policy doc until the first rotation.
