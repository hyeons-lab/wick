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

# Clean build artifacts
clean:
    cargo clean
