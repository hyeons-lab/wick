# wick-ffi

UniFFI bindings for [`wick`](../wick/) — exposes the core inference
engine to Kotlin, Swift, Python, and every other language
[`uniffi-rs`](https://mozilla.github.io/uniffi-rs/) supports.

## Status

**Async.** `WickEngine` + `Session` + sync `generate` + streaming
`generate_streaming` shipped in PRs 2–4; PR 5 adds `async` twins
(`generate_async` + `generate_streaming_async`) via
`#[uniffi::export(async_runtime = "tokio")]` so Kotlin coroutines /
Swift `async` / Python `asyncio` callers can `.await` decode without
stalling their async thread. Surface grows in follow-up PRs:

| PR | Scope |
|---|---|
| 1 | Crate shell + UniFFI scaffolding + smoke-test export |
| 2 | `WickEngine::from_path`, `EngineConfig`, `ModelMetadata`, `ModalityCapabilities` |
| 3 | `Session`, `SessionConfig`, `GenerateOpts`, `GenerateSummary`, sync `generate` |
| 4 | `ModalitySink` as UniFFI foreign-trait callback + streaming `generate` |
| 5 *(this one)* | `async` `generate` + `generate_streaming` via `#[uniffi::export(async_runtime = "tokio")]` |
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
- **Async runtime** is `tokio` (via UniFFI's `tokio` feature flag +
  `#[uniffi::export(async_runtime = "tokio")]`). `tokio` is a `wick-ffi`
  dep only, never `wick` itself — keeps the core crate runtime-agnostic.
  Sync decode work runs on `tokio::task::spawn_blocking` so the async
  worker pool stays free to poll other futures while a generate is in
  flight.
- **Send + Sync** is already guaranteed on every `wick` type we plan
  to expose (landed in PR #42). UniFFI requires it for every
  `#[uniffi::Object]`.
