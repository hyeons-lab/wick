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

# Cross-compile `wick-ffi` to all three iOS targets and assemble a
# `WickFFI.xcframework` ready for Swift Package Manager / Xcode
# consumption. Pairs the device-arm64 staticlib with a fat
# (arm64 + x86_64) simulator staticlib in two slices of one
# `.xcframework` so consumer apps work on both real iPhones and the
# simulator on Apple Silicon and Intel Macs alike.
#
# Requires Xcode (for `xcodebuild` + `lipo`) plus the rustup targets:
# `rustup target add aarch64-apple-ios aarch64-apple-ios-sim
# x86_64-apple-ios`. The vendored Swift bindings under
# `wick-ffi/bindings/swift/` provide the headers + module map; CI
# regenerates them via the `ffi-bindings-drift` job so they stay
# locked to the current Rust surface.
#
# Output: `target/xcframework-build/WickFFI.xcframework`. CI uploads
# the same path as a per-run artifact.
ios-xcframework:
    #!/usr/bin/env bash
    set -euo pipefail
    cargo build -p wick-ffi --target aarch64-apple-ios --release
    cargo build -p wick-ffi --target aarch64-apple-ios-sim --release
    cargo build -p wick-ffi --target x86_64-apple-ios --release
    OUT=target/xcframework-build
    rm -rf "$OUT"
    mkdir -p "$OUT/headers"
    # Stage the headers + module map next to where xcodebuild will
    # look. UniFFI-generated `wick_ffiFFI.modulemap` is renamed to
    # `module.modulemap` on the way in — Xcode's framework conventions
    # require that exact filename inside a `Headers/` directory.
    cp wick-ffi/bindings/swift/wick_ffiFFI.h "$OUT/headers/"
    cp wick-ffi/bindings/swift/wick_ffiFFI.modulemap "$OUT/headers/module.modulemap"
    # Fat simulator slice: arm64 + x86_64 in a single .a so one
    # XCFramework slice covers both Apple Silicon and Intel Mac
    # simulator hosts.
    lipo -create \
        target/aarch64-apple-ios-sim/release/libwick_ffi.a \
        target/x86_64-apple-ios/release/libwick_ffi.a \
        -output "$OUT/libwick_ffi-sim.a"
    xcodebuild -create-xcframework \
        -library target/aarch64-apple-ios/release/libwick_ffi.a -headers "$OUT/headers" \
        -library "$OUT/libwick_ffi-sim.a" -headers "$OUT/headers" \
        -output "$OUT/WickFFI.xcframework"
    echo "Built $OUT/WickFFI.xcframework"

# Single-target iOS smoke test — verifies the device cross-compile
# works without paying for the full sim + lipo + xcodebuild pipeline
# (~30s vs ~90s+). Output `.a` isn't directly usable in an iOS app
# (consumers need the XCFramework or a custom SPM `linkedLibrary`
# wiring); this recipe is mostly a "did the cross-compile break?"
# fast probe. Assumes `aarch64-apple-ios` is rustup-installed.
ios-arm64:
    cargo build -p wick-ffi --target aarch64-apple-ios --release

# Clean build artifacts
clean:
    cargo clean
