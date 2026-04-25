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

# Clean build artifacts
clean:
    cargo clean
