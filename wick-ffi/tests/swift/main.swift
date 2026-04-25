// Swift smoke test that links against the macOS slice of the
// `WickFFI.xcframework` (or the equivalent `aarch64-apple-darwin`
// staticlib in dev) and exercises a few cross-language calls.
//
// Why macOS only: the Rust FFI surface is identical across iOS
// device, iOS simulator, and native macOS — same generated Swift
// binding, same C ABI, same staticlib code path. Validating macOS
// gives us "the entire pipeline links and a Swift call can reach
// Rust and back" without paying for an iOS simulator boot in CI.
// If a regression broke the FFI on iOS specifically (e.g., a
// platform-conditional staticlib symbol), this test wouldn't catch
// it — but those have not been a real failure mode for UniFFI 0.31.
//
// Coverage:
//   1. `wickFfiVersion()` — minimal function call. Proves symbol
//      resolution + UniFFI scaffolding initialization.
//   2. `BackendPreference.cpu` / `.metal` — enum marshaling.
//   3. `EngineConfig(...)` — record (struct) marshaling.
//   4. `FfiError` — error-type round-trip via a contrived call
//      that we expect to fail loudly. The error message format
//      must match `wick::WickError`'s Display per the parity test
//      shipped in PR 7.
//
// Out of scope: actually loading a model. That'd need a GGUF on
// disk + the `remote` feature or test fixtures — separate scope.

import Foundation

// Compile-time link to the UniFFI binding via `-import-objc-header
// wick_ffiFFI.h`. The Swift binding's `#if canImport(wick_ffiFFI)`
// branch is false in this single-binary swiftc invocation; the C
// symbols are resolved through the bridging header instead.

func fail(_ msg: String) -> Never {
    FileHandle.standardError.write(Data("FAIL: \(msg)\n".utf8))
    exit(1)
}

// 1. Function call → version string.
let version = wickFfiVersion()
guard !version.isEmpty else { fail("wickFfiVersion() returned empty") }
print("OK: wick_ffi v\(version)")

// 2. Enum marshaling — construct each variant + Swift's switch
// exhaustiveness check guards against future variant adds breaking
// silently.
let backends: [BackendPreference] = [.auto, .cpu, .gpu, .metal]
for b in backends {
    switch b {
    case .auto: print("OK: BackendPreference.auto")
    case .cpu: print("OK: BackendPreference.cpu")
    case .gpu: print("OK: BackendPreference.gpu")
    case .metal: print("OK: BackendPreference.metal")
    }
}

// 3. Record construction. EngineConfig is a Swift struct generated
// from the UniFFI Record. Setting context_size = 0 is the FFI
// sentinel for "use the model's default max_seq_len" (per the
// docstring on `EngineConfig::context_size`).
let config = EngineConfig(contextSize: 0, backend: .cpu)
guard config.contextSize == 0 else { fail("EngineConfig.contextSize round-trip") }
guard config.backend == .cpu else { fail("EngineConfig.backend round-trip") }
print("OK: EngineConfig(contextSize: 0, backend: .cpu)")

// 4. Error type — try a constructor that we know will fail and
// confirm we get a typed `FfiError` back, not a panic / abort.
// `WickEngine.fromPath` on a nonexistent path should surface as
// `FfiError.io` (the `io::Error` underneath comes from open()).
do {
    _ = try WickEngine.fromPath(
        path: "/no/such/path/__swift_smoke_test__",
        config: config
    )
    fail("WickEngine.fromPath on bogus path unexpectedly succeeded")
} catch let err as FfiError {
    // Expected. The wrapper currently reports `Backend(detail: ...)`
    // for this input because `WickEngine::from_path` validates the
    // path shape (.gguf / .json / dir) before any io::Error can
    // surface. The test is deliberately permissive — any `FfiError`
    // variant is acceptable; the goal is "the typed error
    // round-tripped through Swift at all", not a specific variant
    // or message format. Variant-level assertions belong in Rust
    // unit tests (see `wick-ffi/src/lib.rs`), not here.
    let msg = String(describing: err)
    print("OK: FfiError caught (\(msg))")
} catch {
    fail("unexpected non-FfiError thrown: \(error)")
}

print("OK: all swift smoke tests passed")
