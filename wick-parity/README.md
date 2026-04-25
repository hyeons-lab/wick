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
   `LiquidAI/LeapBundles` HF repo via [`WickEngine::from_bundle_id`]
   on first run and cache it locally.

## Status

| Leg | What it exercises | Status |
|---|---|---|
| `rust` | `wick::WickEngine` directly (reference) | Shipped |
| `ffi` | `wick_ffi::WickEngine` through its Rust public surface | Shipped |
| `kotlin-jna` | Kotlin runner under `legs/kotlin/` loading `libwick_ffi.{so,dylib}` via JNA — first leg to actually cross the FFI boundary | Shipped (PR 16) |
| `swift-uniffi` | Swift runner under `legs/swift/` (SPM), loading the generated UniFFI Swift bindings + `libwick_ffi.dylib` linked at build time. macOS-only. | Shipped (PR 17) |
| `kotlin-aidl` | Cross-process via `wick-serviceapp` AIDL | Pending (Phase 4.5) |

## CLI

```bash
# Run one leg, emit JSON to stdout (includes wall_clock_ms field)
cargo run -p wick-parity -- dump --via rust \
    --bundle LFM2-350M-Extract-GGUF --quant Q4_0 \
    --prompt "The capital of France is" \
    --max-tokens 16

# Run rust + ffi (always) plus optional kotlin/swift legs, then
# diff tokens + wall-clock latency
cargo run -p wick-parity -- check \
    --bundle LFM2-350M-Extract-GGUF --quant Q4_0 \
    --prompt "The capital of France is" \
    --max-tokens 16 \
    --kotlin-runner $(pwd)/wick-parity/legs/kotlin/build/libs/wick-parity-kotlin-all.jar \
    --swift-runner  $(pwd)/wick-parity/legs/swift/.build/release/WickParitySwift
#
# perf (max-slowdown threshold = 2.00×):
#   rust              582 ms  (reference)
#   ffi               549 ms  (0.94×)
#   kotlin-jna       7173 ms  (12.32×) WARN
#   swift-uniffi     6359 ms  (10.93×) WARN
#
# OK: 4 legs, all token streams match (bundle=LFM2-350M-Extract-GGUF quant=Q4_0 tokens=16)
```

`check` exits 0 on token match (regardless of perf threshold breach
unless `--fail-on-slowdown` is set), 1 with a windowed diff summary
on token mismatch, 2 on any other error (load failure, network error,
…). The cache directory survives across `dump` ↔ `check` invocations
so the model downloads once.

### Performance parity

`check` measures wall-clock latency per leg and reports the
`non_rust_ms / rust_ms` ratio. Subprocess legs (`kotlin-jna`,
`swift-uniffi`) measure inside their `runOnce` body and emit the
result via `RunOutput.wall_clock_ms`, so JVM startup (~500ms) and
Swift cold-start (~50ms) don't pollute the FFI overhead measurement.

**Threshold tuning.** Default `--max-slowdown 2.0` is appropriate
for in-process legs (rust ↔ ffi). Subprocess legs see fundamental
overhead from spawn + fresh-process mmap (no shared OS page cache
with rust), typically running 5-15× the rust reference on a fresh
fixture. To gate CI on subprocess perf, set a higher ceiling tuned
to your runner's actual variance:

```bash
cargo run -p wick-parity -- check ... \
    --kotlin-runner ... --swift-runner ... \
    --max-slowdown 15.0 --fail-on-slowdown
```

`--fail-on-slowdown` is opt-in (default warn-only) so a noisy CI
runner doesn't block merges on a single bad measurement. Once
you've watched the variance for a few runs and picked a stable
threshold, flip it on.

### Cache root

The harness uses `target/tmp/wick-parity-cache/` under the workspace
root by default. Override with `--cache-dir <path>` (CLI) or
`$WICK_PARITY_CACHE_DIR` (env). Distinct from `wick-test-models/`
(used by `wick`'s `tests/common/download.rs`) so the two caches can
be wiped independently.

## Integration tests

Two test targets, both env-gated and `#[ignore]`'d so default
`cargo test` skips them.

### `parity` — rust ↔ ffi

```bash
WICK_PARITY_RUN=1 cargo test -p wick-parity --test parity -- --ignored
```

### `parity_kotlin` — rust ↔ kotlin-jna

Build the Kotlin runner once, then run the test:

```bash
cd wick-parity/legs/kotlin && ./gradlew shadowJar && cd -
cargo build -p wick-ffi   # produces target/debug/libwick_ffi.{so,dylib}
WICK_PARITY_RUN=1 \
  WICK_PARITY_KOTLIN_RUNNER=$(pwd)/wick-parity/legs/kotlin/build/libs/wick-parity-kotlin-all.jar \
  cargo test -p wick-parity --test parity_kotlin -- --ignored
```

Set `WICK_PARITY_LIB_DIR=<path>` if `libwick_ffi` is somewhere
other than `<workspace>/target/debug` (e.g. release build).

### `parity_swift` — rust ↔ swift-uniffi (macOS only)

Build the Swift runner once, then run the test. Run from the
workspace root; `WS=$(pwd)` keeps the absolute path stable across
the subshell that builds the Swift package:

```bash
WS=$(pwd)
cargo build -p wick-ffi   # produces $WS/target/debug/libwick_ffi.dylib
swift build -c release \
  --package-path "$WS/wick-parity/legs/swift" \
  -Xlinker -L"$WS/target/debug"
WICK_PARITY_RUN=1 \
  WICK_PARITY_SWIFT_RUNNER="$WS/wick-parity/legs/swift/.build/release/WickParitySwift" \
  cargo test -p wick-parity --test parity_swift -- --ignored
```

`-Xlinker -L<dir>` is the SPM-portable way to add a library search
path; `LDFLAGS` is not honored consistently by `swift build`. The
binary's install_name points at the absolute build-time location of
`libwick_ffi.dylib` (under `target/debug/`), which works as long as
the dylib stays put after build — fine for CI / local dev. Set
`WICK_PARITY_LIB_DIR=<path>` if you need to override the
`DYLD_LIBRARY_PATH` the runner sees.

CI runs the gated tests on push to `main` + manual
`workflow_dispatch` only — not on PRs:
- `Parity Harness (gated)` (`.github/workflows/ci.yml`) runs on
  Ubuntu and exercises `parity` + `parity_kotlin`.
- `Parity Harness Swift (gated)` runs on `macos-15` and exercises
  `parity_swift`.

Both jobs use the cache key `wick-parity-cache-lfm2-350m-q4_0-v1`;
GitHub's cache pools are OS-segregated so each OS downloads the
fixture once. Bump the `-vN` suffix when the bundle id or quant in
either test changes.

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

Two worked examples ship today:

- `legs/kotlin/` — Gradle subproject + Shadow plugin → fat jar;
  vendored binding via `sourceSets.main.kotlin.srcDirs` so the file
  isn't duplicated; `Main.kt` reads stdin / writes stdout in the
  contract shape; `wick_parity::run_kotlin_jna` spawns it.
- `legs/swift/` — SPM Package with two targets: `wick_ffiFFI`
  (`.systemLibrary` exposing the generated C FFI header via
  `module.modulemap` — name MUST match what the generated
  `wick_ffi.swift` `canImport`s) and `WickParitySwift`
  (`.executableTarget` consuming the bindings + linking against
  the wick-ffi cdylib via `-Xlinker -L`). Bindings vendored via
  symlinks so `just bindings` regenerations are picked up
  automatically; `wick_parity::run_swift_uniffi` spawns the
  built binary with `DYLD_LIBRARY_PATH` set.

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
