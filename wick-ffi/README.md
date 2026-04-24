# wick-ffi

UniFFI bindings for [`wick`](../wick/) — exposes the core inference
engine to Kotlin, Swift, Python, and every other language
[`uniffi-rs`](https://mozilla.github.io/uniffi-rs/) supports.

## Status

**Bootstrap.** The crate exists, the UniFFI build pipeline works, and
one smoke-test function (`wick_ffi_version()`) is exported. The real
surface grows in follow-up PRs per the Phase 2 roadmap:

| PR | Scope |
|---|---|
| 1 *(this one)* | Crate shell + UniFFI scaffolding + smoke-test export |
| 2 | `WickEngine::from_path`, `EngineConfig`, `ModelMetadata`, `ModalityCapabilities` |
| 3 | `Session`, `SessionConfig`, `GenerateOpts`, `GenerateSummary`, sync `generate` |
| 4 | `ModalitySink` as UniFFI foreign-trait callback |
| 5 | `async` `generate` via `#[uniffi::export(async_runtime = "tokio")]` |
| 6 | Kotlin + Swift binding generation + vendored outputs + CI |
| 7+ | Error-type marshalling, parity harness, Android ABIs, iOS XCFramework |

Don't add FFI exposure to `wick` directly — the `wick` crate keeps its
idiomatic Rust surface, and everything UniFFI-specific lives here.

## Crate types

```toml
[lib]
crate-type = ["lib", "cdylib", "staticlib"]
```

- **`cdylib`** — Android / Linux / macOS dynamic loading (`System.loadLibrary` in Kotlin).
- **`staticlib`** — iOS XCFramework archive (Swift Package Manager).
- **`lib`** — other Rust crates in this workspace (future parity harness, examples).

## Build

```bash
cargo build -p wick-ffi
# Produces (on macOS):
# - target/debug/libwick_ffi.dylib   (cdylib)
# - target/debug/libwick_ffi.a       (staticlib)
# - target/debug/libwick_ffi.rlib    (lib)
```

Binding generation (Kotlin, Swift, Python) lands in PR 6; until then
the crate is Rust-only.

## Design notes

- **Proc-macro path** chosen over UDL for smaller surface ergonomics.
  Annotations live next to the Rust types they describe; no separate
  grammar to maintain. Can migrate to UDL if the annotation density
  ever becomes unmanageable.
- **Async runtime** will be `tokio` when `async fn generate` lands
  (PR 5). `tokio` is a `wick-ffi` dep only, never `wick` itself —
  keeps the core crate runtime-agnostic.
- **Send + Sync** is already guaranteed on every `wick` type we plan
  to expose (landed in PR #42). UniFFI requires it for every
  `#[uniffi::Object]`.
