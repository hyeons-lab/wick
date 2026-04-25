# wick-parity Swift-via-UniFFI leg

SPM package that builds an executable
(`.build/release/WickParitySwift`) loading the generated UniFFI Swift
bindings + the wick-ffi cdylib. Read by the Rust harness
(`wick_parity::run_swift_uniffi`) over stdin/stdout JSON; sister to
`legs/kotlin/`.

## Layout

```
legs/swift/
├── Package.swift                 — two-target SPM manifest
└── Sources/
    ├── wick_ffiFFI/              — .systemLibrary target
    │   ├── module.modulemap      — exposes wick_ffiFFI.h as `wick_ffiFFI`
    │   └── wick_ffiFFI.h         — symlink → ../../../../../wick-ffi/bindings/swift/wick_ffiFFI.h
    └── WickParitySwift/          — .executableTarget
        ├── main.swift            — runner (stdin JSON → bindings → stdout JSON)
        └── wick_ffi.swift        — symlink → ../../../../../wick-ffi/bindings/swift/wick_ffi.swift
```

The two binding files are git symlinks (mode 120000), so
`just bindings` regenerations propagate without a manual copy step.
The system-library target name is `wick_ffiFFI` exactly because the
generated `wick_ffi.swift` does `#if canImport(wick_ffiFFI) ; import
wick_ffiFFI` — any other name and the C FFI declarations fall out of
scope at compile time.

## Build

From the workspace root:

```bash
WS=$(pwd)
cargo build -p wick-ffi    # produces $WS/target/debug/libwick_ffi.dylib
swift build -c release \
  --package-path "$WS/wick-parity/legs/swift" \
  -Xlinker -L"$WS/target/debug"
```

`-Xlinker -L<dir>` is the SPM-portable way to add a library search
path. `LDFLAGS` is not honored consistently by `swift build`.
`--package-path` keeps the build out-of-process from the workspace
shell so no `cd` / subshell is required.

## Run (manual smoke test)

```bash
WS=$(pwd)
echo '{
  "bundle": "LFM2-350M-Extract-GGUF",
  "quant": "Q4_0",
  "prompt": "The capital of France is",
  "max_tokens": 16,
  "seed": 0,
  "cache_dir": "'"$WS"'/target/tmp/wick-parity-cache"
}' | DYLD_LIBRARY_PATH="$WS/target/debug" \
  "$WS/wick-parity/legs/swift/.build/release/WickParitySwift"
```

Emits a `RunOutput` JSON document on stdout. First run downloads the
~210 MB fixture into `cache_dir`; subsequent runs are cache hits.
Cache root mirrors the harness default so the manual smoke shares
state with `cargo test`.
