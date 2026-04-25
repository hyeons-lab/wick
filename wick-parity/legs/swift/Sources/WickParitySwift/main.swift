// wick-parity Swift-via-UniFFI leg.
//
// Reads a `RunArgsOwned`-shaped JSON request from stdin, drives the
// generated UniFFI Swift bindings end-to-end (against `libwick_ffi.dylib`
// found via `DYLD_LIBRARY_PATH`), emits a `RunOutput` JSON document on
// stdout, and exits 0 on success or 1 on any error (with a one-line
// `error: <msg>` to stderr).
//
// Constants here MUST mirror `wick_parity::settings` in
// `wick-parity/src/lib.rs` AND `Settings` in the Kotlin runner at
// `wick-parity/legs/kotlin/.../Main.kt` — drift between the three would
// surface as false parity failures that aren't actually FFI bugs. Keep
// them in sync byte-for-byte.

import Foundation
import wick_ffiFFI

struct RunArgsOwned: Codable {
    let bundle: String
    let quant: String
    let prompt: String
    let maxTokens: UInt32
    let seed: UInt64
    let cacheDir: String

    enum CodingKeys: String, CodingKey {
        case bundle
        case quant
        case prompt
        case maxTokens = "max_tokens"
        case seed
        case cacheDir = "cache_dir"
    }
}

struct RunOutput: Codable {
    let via: String
    let bundle: String
    let quant: String
    let prompt: String
    let maxTokens: UInt32
    let seed: UInt64
    let tokens: [UInt32]
    // Wall-clock latency of `runOnce` only — excludes Swift binary
    // cold-start (~50ms) so the rust harness can compute
    // apples-to-apples ratios against the in-process rust leg's
    // `Instant::now()` measurement. Optional to mirror
    // `RunOutput.wall_clock_ms: Option<u64>` on the Rust side, for
    // forward compat with older harness versions that ignore it.
    let wallClockMs: UInt64?

    enum CodingKeys: String, CodingKey {
        case via
        case bundle
        case quant
        case prompt
        case maxTokens = "max_tokens"
        case seed
        case tokens
        case wallClockMs = "wall_clock_ms"
    }
}

// Mirror of `wick_parity::settings` (Rust) and `Settings` (Kotlin).
// Manual copy across three languages — no shared schema. Drift here
// produces a false parity failure, which the integration test catches.
private enum Settings {
    static let contextSize: UInt64 = 256
    static let temperature: Float = 0.0
    static let topP: Float = 1.0
    static let topK: UInt32 = 1
    static let repetitionPenalty: Float = 1.0
    static let flushEveryTokens: UInt32 = 1
    static let flushEveryMs: UInt32 = 0
}

func runOnce(_ args: RunArgsOwned) throws -> [UInt32] {
    let repo = BundleRepo(storeDir: args.cacheDir)
    let cfg = EngineConfig(
        contextSize: Settings.contextSize,
        backend: .cpu,
        bundleRepo: repo
    )
    let engine = try WickEngine.fromBundleId(
        bundleId: args.bundle,
        quant: args.quant,
        config: cfg
    )

    // `ubatchSize: 512` mirrors `wick::SessionConfig::default()` —
    // the Rust legs in `wick-parity/src/lib.rs` build their session
    // config with `..Default::default()`, which fills in 512. Drifting
    // here (e.g. monolithic `0`) takes the prefill through a different
    // kernel path; greedy decode happens to be insensitive to that for
    // the parity prompt today, but the harness's whole job is to catch
    // silent divergences — keep the configs aligned.
    let sessionCfg = SessionConfig(
        maxSeqLen: nil,
        kvCompression: .none,
        nKeep: 0,
        seed: args.seed,
        ubatchSize: 512
    )
    let session = engine.newSession(config: sessionCfg)

    if !args.prompt.isEmpty {
        try session.appendText(text: args.prompt)
    }

    let opts = GenerateOpts(
        maxTokens: args.maxTokens,
        temperature: Settings.temperature,
        topP: Settings.topP,
        topK: Settings.topK,
        repetitionPenalty: Settings.repetitionPenalty,
        stopTokens: [],
        flushEveryTokens: Settings.flushEveryTokens,
        flushEveryMs: Settings.flushEveryMs
    )
    let output = try session.generate(opts: opts)
    return output.tokens
}

func main() {
    do {
        let stdinData = FileHandle.standardInput.readDataToEndOfFile()
        let request = try JSONDecoder().decode(RunArgsOwned.self, from: stdinData)
        // Bracket only `runOnce` so the timing excludes JSON
        // decode + framework load; rust harness pairs it against
        // `Instant::now()` deltas around `run_rust`'s engine +
        // session work.
        let started = DispatchTime.now()
        let tokens = try runOnce(request)
        let elapsedMs = (DispatchTime.now().uptimeNanoseconds - started.uptimeNanoseconds) / 1_000_000
        let response = RunOutput(
            via: "swift-uniffi",
            bundle: request.bundle,
            quant: request.quant,
            prompt: request.prompt,
            maxTokens: request.maxTokens,
            seed: request.seed,
            tokens: tokens,
            wallClockMs: elapsedMs
        )
        let encoder = JSONEncoder()
        // Pretty-print mirrors the Rust `dump` subcommand's output.
        // The Rust harness parses with `serde_json` either way.
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let stdoutData = try encoder.encode(response)
        FileHandle.standardOutput.write(stdoutData)
        FileHandle.standardOutput.write("\n".data(using: .utf8)!)
    } catch {
        // One-line error so the Rust harness has a stable shape to
        // bubble up. Stack-style detail would clutter the
        // integration-test output without adding signal — local
        // debugging can re-run with `swift run` for a fuller trace.
        let line = "error: \(error)\n"
        FileHandle.standardError.write(line.data(using: .utf8)!)
        exit(1)
    }
}

main()
