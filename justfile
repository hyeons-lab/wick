# Default recipe
default: build

# Build all crates
build:
    cargo build --workspace

# Build in release mode
release:
    cargo build --workspace --release

# Run all tests
test:
    cargo test --workspace

# Run clippy lints
clippy:
    cargo clippy --workspace -- -D warnings

# Check formatting
fmt:
    cargo fmt --check

# Format code
fmt-fix:
    cargo fmt

# Run the CLI with arguments
run *ARGS:
    cargo run --bin wick -- {{ARGS}}

# Run benchmarks
bench *ARGS:
    cargo run --release --bin wick -- bench {{ARGS}}

# Run all CI checks locally (mirrors GitHub Actions)
ci: fmt clippy test

# Platform-specific shared-library path for the uniffi-bindgen
# `--library` argument. `os()` is a just built-in.
# - macOS: `libwick_ffi.dylib`
# - Linux / other unix: `libwick_ffi.so`
# - Windows: `wick_ffi.dll` (no `lib` prefix — Rust follows the
#   Windows convention on that target).
WICK_FFI_DYLIB := if os() == "macos" {
    "target/debug/libwick_ffi.dylib"
} else if os() == "windows" {
    "target/debug/wick_ffi.dll"
} else {
    "target/debug/libwick_ffi.so"
}

# Regenerate the vendored Kotlin + Swift bindings in wick-ffi/bindings/.
# Runs the `uniffi-bindgen` binary in this repo against the freshly-built
# debug cdylib. Kotlin output is ktlint-formatted automatically (uniffi
# invokes ktlint on PATH); Swift is formatter-free (no standard Swift
# formatter in the pipeline). Commit the resulting diff when Rust-side
# exports change.
#
# `--features bindgen` on the `cargo run` invocations turns on the
# opt-in `wick-ffi/bindgen` crate feature, which pulls in
# `uniffi/cli` (clap + friends) only for the binary build. Mobile
# consumers of the library / cdylib / staticlib never build with
# this feature, so their binaries stay lean.
#
# Requires `ktlint` on PATH — macOS: `brew install ktlint`; Linux:
# download the standalone binary from ktlint releases or use your
# package manager. CI installs it as part of the ffi-bindings-drift
# job.
bindings:
    cargo build -p wick-ffi
    cargo run -q -p wick-ffi --bin uniffi-bindgen --features bindgen -- generate \
        --library {{WICK_FFI_DYLIB}} \
        --language kotlin \
        --out-dir wick-ffi/bindings/kotlin
    cargo run -q -p wick-ffi --bin uniffi-bindgen --features bindgen -- generate \
        --library {{WICK_FFI_DYLIB}} \
        --language swift \
        --out-dir wick-ffi/bindings/swift

# Verify the committed Kotlin + Swift bindings are up to date with the
# current Rust FFI surface. Regenerates in-place and fails if `git diff`
# shows changes — signals that someone touched a `#[uniffi::*]` export
# without running `just bindings`. CI runs this too; see ci.yml.
bindings-check: bindings
    @if [ -n "$(git status --porcelain wick-ffi/bindings)" ]; then \
        echo "ERROR: vendored bindings are stale. Run \`just bindings\` and commit the diff."; \
        git --no-pager diff wick-ffi/bindings; \
        exit 1; \
    fi

# Cross-compile `wick-ffi` as a `.so` for every Android ABI supported
# by the Android NDK: arm64-v8a (modern devices), armeabi-v7a (older),
# x86_64 (emulator on Intel hosts), x86 (emulator on legacy Intel hosts).
# Produces `target/<triple>/release/libwick_ffi.so` per ABI.
#
# Requires `cargo-ndk` v4.x (`cargo install cargo-ndk --version '^4'
# --locked` — pin the major because 4.0 changed the flag shape to
# `--target <abi>`; earlier releases used `--arch` / `--platform`
# and would fail against the recipes below) and the Rust targets:
# `rustup target add aarch64-linux-android armv7-linux-androideabi
# x86_64-linux-android i686-linux-android`. The NDK itself comes from
# Android Studio (ndk/<version>/) or `sdkmanager --install ndk`.
# `ANDROID_NDK_HOME` must point at the NDK root; CI sets it via the
# `nttld/setup-ndk` action.
#
# Release profile for the size drop — debug builds are ~75 MB per .so
# due to embedded debuginfo, release is ~2.5 MB with LTO + strip.
android-all:
    cargo ndk \
        --target arm64-v8a \
        --target armeabi-v7a \
        --target x86_64 \
        --target x86 \
        build -p wick-ffi --release

# Single-ABI variant — useful when iterating on one device architecture
# and you don't need to rebuild all four every cycle. Picks arm64-v8a
# as the default since it's what real Android phones ship with today.
android-arm64:
    cargo ndk --target arm64-v8a build -p wick-ffi --release

# Cross-compile `wick-ffi` to all three arm64-only Apple-platform
# targets and assemble a `WickFFI.xcframework` ready for Swift
# Package Manager / Xcode consumption. Three single-arch slices:
# real iPhones (`ios-arm64`), Apple Silicon Mac iOS Simulator
# (`ios-arm64-simulator`), and native Apple Silicon Macs
# (`macos-arm64`). x86_64 is deliberately omitted — Apple stopped
# selling Intel Macs in 2023 and modern consumer apps drop support.
#
# Requires Xcode (for `xcodebuild`) + the rustup targets:
# `rustup target add aarch64-apple-ios aarch64-apple-ios-sim
# aarch64-apple-darwin`. `RUSTFLAGS=""` overrides the workspace's
# `target-cpu=native` for the apple-darwin slice so the shipped
# staticlib is portable across Apple Silicon Macs (otherwise the
# build host's specific microarch leaks into the binary).
#
# The vendored Swift bindings under `wick-ffi/bindings/swift/`
# provide the headers + module map; CI regenerates them via the
# `ffi-bindings-drift` job so they stay locked to the current Rust
# surface.
#
# Output: `target/xcframework-build/WickFFI.xcframework` (~125 MB,
# 42 MB per slice). CI uploads the same path as a per-run artifact.
apple-xcframework:
    #!/usr/bin/env bash
    set -euo pipefail
    RUSTFLAGS="" cargo build -p wick-ffi --target aarch64-apple-ios --release
    RUSTFLAGS="" cargo build -p wick-ffi --target aarch64-apple-ios-sim --release
    RUSTFLAGS="" cargo build -p wick-ffi --target aarch64-apple-darwin --release
    OUT=target/xcframework-build
    rm -rf "$OUT"
    mkdir -p "$OUT/headers"
    # Stage the headers + module map next to where xcodebuild will
    # look. UniFFI-generated `wick_ffiFFI.modulemap` is renamed to
    # `module.modulemap` on the way in — Xcode's framework conventions
    # require that exact filename inside a `Headers/` directory.
    cp wick-ffi/bindings/swift/wick_ffiFFI.h "$OUT/headers/"
    cp wick-ffi/bindings/swift/wick_ffiFFI.modulemap "$OUT/headers/module.modulemap"
    xcodebuild -create-xcframework \
        -library target/aarch64-apple-ios/release/libwick_ffi.a -headers "$OUT/headers" \
        -library target/aarch64-apple-ios-sim/release/libwick_ffi.a -headers "$OUT/headers" \
        -library target/aarch64-apple-darwin/release/libwick_ffi.a -headers "$OUT/headers" \
        -output "$OUT/WickFFI.xcframework"
    echo "Built $OUT/WickFFI.xcframework"

# Single-target iOS smoke test — verifies the device cross-compile
# works without paying for the full apple-xcframework pipeline (3
# cross-compiles + xcodebuild → ~90s+; this single build → ~30s).
# Output `.a` isn't directly usable in an iOS app (consumers need
# the XCFramework or a custom SPM `linkedLibrary` wiring); this
# recipe is mostly a "did the cross-compile break?" fast probe.
# Assumes `aarch64-apple-ios` is rustup-installed.
#
# `RUSTFLAGS=""` mirrors the `apple-xcframework` + `swift-smoke`
# recipes for consistency. Strictly a no-op for iOS targets
# (`.cargo/config.toml` only sets `target-cpu=native` on
# apple-darwin), but the override forestalls an externally-set
# RUSTFLAGS environment variable from contaminating this smoke build.
ios-arm64:
    RUSTFLAGS="" cargo build -p wick-ffi --target aarch64-apple-ios --release

# End-to-end Swift integration test against the macOS slice. Compiles
# `wick-ffi/tests/swift/main.swift` together with the vendored Swift
# binding, links against the freshly-built `aarch64-apple-darwin`
# staticlib, runs the resulting binary. Exercises function calls,
# enum + record marshaling, and FfiError round-trip end-to-end.
#
# Why macOS-only smoke: the Rust FFI is identical across iOS device,
# iOS Simulator, and native macOS — same Swift binding, same C ABI,
# same staticlib. Validating macOS proves the integration; iOS
# device + Simulator share the same code path so the test covers
# them by proxy.
#
# Requires Xcode (`swiftc`) + `aarch64-apple-darwin` rustup target.
# Builds the staticlib first if it isn't already cached.
swift-smoke:
    #!/usr/bin/env bash
    set -euo pipefail
    RUSTFLAGS="" cargo build -p wick-ffi --target aarch64-apple-darwin --release
    swiftc \
        wick-ffi/tests/swift/main.swift \
        wick-ffi/bindings/swift/wick_ffi.swift \
        -import-objc-header wick-ffi/bindings/swift/wick_ffiFFI.h \
        -L target/aarch64-apple-darwin/release \
        -lwick_ffi \
        -o target/wick-swift-smoke
    target/wick-swift-smoke

# Build the `wick-wasm` npm-shaped package via `wasm-pack`
# (bundler target — see `wasm-web` / `wasm-node` for siblings).
#
# Wraps `cargo build --target wasm32-unknown-unknown` + `wasm-bindgen-cli`
# + `wasm-opt -O3` and writes the output to `wick-wasm/pkg-bundler/`
# (gitignored — the matrix layout uses `pkg-<target>` to keep the
# three target outputs from colliding). The result includes
# `package.json`, `wick_wasm.js`, `wick_wasm.d.ts`,
# `wick_wasm_bg.wasm`, and the README — drop-in for
# `npm install ./wick-wasm/pkg-bundler`.
#
# Target is `bundler` (webpack / Vite / Rollup-friendly ESM). Use
# `just wasm-web` for direct browser ESM (`<script type="module">`)
# or `just wasm-node` for CommonJS Node consumers.
#
# `--scope hyeonslab` makes the generated `package.json.name`
# `@hyeonslab/wick-wasm` so a published artifact lands under the
# right npm scope. The publish workflow itself is a follow-up PR;
# this just locks the name.
#
# Requires:
#   - `wasm-pack`            (`cargo install wasm-pack`)
#   - `wasm-opt` on PATH     (macOS: `brew install binaryen`,
#                             linux: `apt-get install -y binaryen`)
#   - `wasm32-unknown-unknown` rustup target
#     (`rustup target add wasm32-unknown-unknown`)
#
# wasm-opt flags are pinned in `wick-wasm/Cargo.toml` under
# `[package.metadata.wasm-pack.profile.release]` so this recipe and the
# CI `wick-wasm-pack` job produce byte-identical output.
wasm:
    wasm-pack build wick-wasm --target bundler --release --scope hyeonslab --out-dir pkg-bundler
    @echo "--- wick-wasm/pkg-bundler/ ---"
    @ls -lh wick-wasm/pkg-bundler/

# Build the `--target web` variant — direct browser ESM, no bundler
# required. Consumers `import init, { ... } from './wick_wasm.js'`
# and `await init()` once before calling exports. Right shape for
# `<script type="module">` and bundler-less workflows.
wasm-web:
    wasm-pack build wick-wasm --target web --release --scope hyeonslab --out-dir pkg-web
    @echo "--- wick-wasm/pkg-web/ ---"
    @ls -lh wick-wasm/pkg-web/

# Build the `--target nodejs` variant — CommonJS module that Node
# consumers `require('@hyeonslab/wick-wasm')` directly without the
# experimental-wasm-modules dance. Right shape for Node CLI tools
# / scripts that prefer CommonJS or are stuck on older Node.
wasm-node:
    wasm-pack build wick-wasm --target nodejs --release --scope hyeonslab --out-dir pkg-nodejs
    @echo "--- wick-wasm/pkg-nodejs/ ---"
    @ls -lh wick-wasm/pkg-nodejs/

# ── Multi-threaded wasm builds ──────────────────────────────────────────
#
# Threaded variants light up `wick`'s rayon paths (batched prefill
# GEMM, parallel GEMV row sweeps, dequant_rows_to_f32) on the wasm
# target via `wasm-bindgen-rayon`. The generated package surfaces a
# `initThreadPool(numThreads)` JS export that callers `await` once
# before driving inference.
#
# Three things turn this on together — none of them are useful
# without the others:
#   1. `--features parallel` on `wick-wasm` enables `wick/parallel`
#      (rayon) and links `wasm-bindgen-rayon` (the JS thread-pool
#      shim).
#   2. `RUSTFLAGS="-C target-feature=+atomics,+bulk-memory,+mutable-globals"`
#      makes rustc emit atomic ops + thread-local storage
#      instructions. bulk-memory and mutable-globals are already
#      enabled by wasm-opt; the rustflags entry forces them on at
#      compile time too because atomics requires both.
#   3. `-Z build-std=panic_abort,std` rebuilds std with atomics on.
#      The precompiled std rustup ships isn't built with atomics,
#      so anything that touches a sync primitive (rayon definitely
#      does) fails to link without this. Requires the `rust-src`
#      rustup component (`rustup component add rust-src --toolchain
#      $(cat rust-toolchain.toml | grep channel | cut -d'"' -f2)`)
#      and a nightly toolchain — both already in
#      `rust-toolchain.toml`.
#
# Browsers also need cross-origin isolation (COOP `same-origin` +
# COEP `require-corp` headers on the host page) for
# `SharedArrayBuffer`. Node has no equivalent gate.
#
# `--target bundler` is intentionally not provided — `wasm-bindgen-rayon`
# doesn't have canonical bundler-side worker glue, so we ship `web` +
# `nodejs` only.
#
# Link-arg breakdown (all required, none optional):
#   --shared-memory          memory definition gets the SHARED flag.
#                            Without it the linker emits non-shared memory
#                            even with `+atomics`, and Web Workers can't
#                            see the same heap.
#   --import-memory          memory comes from JS (`env.memory`) instead
#                            of being defined inside the wasm. Required
#                            because each Web Worker creates its own
#                            wasm instance and they all need to share
#                            the same `WebAssembly.Memory` — the only
#                            way to do that is to import it.
#   --max-memory=<bytes>     shared memory must declare a max. 4 GB
#                            (`4294967296`) is the wasm32 ceiling and
#                            matches what `wasm-bindgen-rayon`'s docs
#                            recommend.
#   --export=__wasm_init_tls + __tls_size + __tls_align + __tls_base
#                            wasm-bindgen-cli's threading transform
#                            looks these up by name in the export
#                            table. LLD generates them when shared
#                            memory is on but doesn't auto-export them
#                            — without these four flags wasm-bindgen
#                            fails with `failed to find __wasm_init_tls`.
WASM_MT_RUSTFLAGS := "-C target-feature=+atomics,+bulk-memory,+mutable-globals" + \
    " -C link-arg=--shared-memory" + \
    " -C link-arg=--import-memory" + \
    " -C link-arg=--max-memory=4294967296" + \
    " -C link-arg=--export=__wasm_init_tls" + \
    " -C link-arg=--export=__tls_size" + \
    " -C link-arg=--export=__tls_align" + \
    " -C link-arg=--export=__tls_base"

# Build the `--target web` threaded variant — `pkg-web-mt/`.
# Browser consumers `await initThreadPool(navigator.hardwareConcurrency)`
# once after `await init()` resolves; subsequent `Session.generate`
# calls run rayon work on the worker pool.
wasm-web-mt:
    RUSTFLAGS="{{WASM_MT_RUSTFLAGS}}" \
    wasm-pack build wick-wasm \
        --target web --release \
        --scope hyeonslab --out-dir pkg-web-mt \
        -- --features parallel \
        -Z build-std=panic_abort,std
    @echo "--- wick-wasm/pkg-web-mt/ ---"
    @ls -lh wick-wasm/pkg-web-mt/

# Build the `--target nodejs` threaded variant — `pkg-nodejs-mt/`.
# Node consumers `await initThreadPool(os.cpus().length)` once before
# driving inference; the pool is backed by `worker_threads`.
wasm-node-mt:
    RUSTFLAGS="{{WASM_MT_RUSTFLAGS}}" \
    wasm-pack build wick-wasm \
        --target nodejs --release \
        --scope hyeonslab --out-dir pkg-nodejs-mt \
        -- --features parallel \
        -Z build-std=panic_abort,std
    @echo "--- wick-wasm/pkg-nodejs-mt/ ---"
    @ls -lh wick-wasm/pkg-nodejs-mt/

# Clean build artifacts
clean:
    cargo clean
