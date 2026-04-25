// wick-parity Kotlin-via-JNA leg.
//
// Reads a `RunArgsOwned`-shaped JSON request from stdin, drives the
// generated UniFFI Kotlin bindings end-to-end (via JNA → libwick_ffi),
// emits a `RunOutput` JSON document on stdout, and exits 0 on success
// or 1 on any error (with a one-line `error: <msg>` to stderr).
//
// Constants here MUST mirror `wick_parity::settings` in
// `wick-parity/src/lib.rs` — drift between the two would surface as
// false parity failures that aren't actually FFI bugs. Keep them in
// sync byte-for-byte.

package com.hyeonslab.wickparity

import kotlinx.serialization.Serializable
import kotlinx.serialization.SerialName
import kotlinx.serialization.json.Json
import uniffi.wick_ffi.BackendPreference
import uniffi.wick_ffi.BundleRepo
import uniffi.wick_ffi.EngineConfig
import uniffi.wick_ffi.GenerateOpts
import uniffi.wick_ffi.KvCompression
import uniffi.wick_ffi.SessionConfig
import uniffi.wick_ffi.WickEngine
import kotlin.system.exitProcess

@Serializable
data class RunArgsOwned(
    val bundle: String,
    val quant: String,
    val prompt: String,
    @SerialName("max_tokens") val maxTokens: UInt,
    val seed: ULong,
    @SerialName("cache_dir") val cacheDir: String,
)

@Serializable
data class RunOutput(
    val via: String,
    val bundle: String,
    val quant: String,
    val prompt: String,
    @SerialName("max_tokens") val maxTokens: UInt,
    val seed: ULong,
    val tokens: List<UInt>,
)

// Mirror of `wick_parity::settings`. Manual copy (no shared schema)
// because the harness lives across two languages; the integration test
// catches drift end-to-end. `val` not `const val` — Kotlin's
// `const val` doesn't support unsigned types.
private object Settings {
    val CONTEXT_SIZE: ULong = 256uL
    val TEMPERATURE: Float = 0.0f
    val TOP_P: Float = 1.0f
    val TOP_K: UInt = 1u
    val REPETITION_PENALTY: Float = 1.0f
    val FLUSH_EVERY_TOKENS: UInt = 1u
    val FLUSH_EVERY_MS: UInt = 0u
}

private val json = Json {
    ignoreUnknownKeys = true
    encodeDefaults = true
}

fun main() {
    try {
        val request = json.decodeFromString<RunArgsOwned>(System.`in`.readBytes().toString(Charsets.UTF_8))
        val tokens = runOnce(request)
        val response = RunOutput(
            via = "kotlin-jna",
            bundle = request.bundle,
            quant = request.quant,
            prompt = request.prompt,
            maxTokens = request.maxTokens,
            seed = request.seed,
            tokens = tokens,
        )
        // Pretty-print for parity with the Rust `dump` subcommand.
        // The Rust harness parses with `serde_json` either way.
        val pretty = Json { prettyPrint = true; encodeDefaults = true }
        println(pretty.encodeToString(RunOutput.serializer(), response))
    } catch (t: Throwable) {
        // One-line error so the Rust harness has a stable shape to
        // bubble up. Stack traces would clutter the integration-test
        // output without adding signal — reraise with `--stacktrace`
        // when running locally if you need them.
        System.err.println("error: ${t.javaClass.simpleName}: ${t.message ?: "(no message)"}")
        exitProcess(1)
    }
}

private fun runOnce(args: RunArgsOwned): List<UInt> {
    val repo = BundleRepo(args.cacheDir)
    val cfg = EngineConfig(
        contextSize = Settings.CONTEXT_SIZE,
        backend = BackendPreference.CPU,
        bundleRepo = repo,
    )
    val engine = WickEngine.fromBundleId(args.bundle, args.quant, cfg)

    // `ubatchSize = 512u` mirrors `wick::SessionConfig::default()` —
    // the Rust legs in `wick-parity/src/lib.rs` build their session
    // config with `..Default::default()`, which fills in 512. Drifting
    // here (e.g. monolithic `0u`) takes the prefill through a
    // different kernel path; greedy decode happens to be insensitive
    // to that for the parity prompt today, but the harness's whole
    // job is to catch silent divergences — keep the configs aligned.
    val sessionCfg = SessionConfig(
        maxSeqLen = null,
        kvCompression = KvCompression.None,
        nKeep = 0u,
        seed = args.seed,
        ubatchSize = 512u,
    )
    val session = engine.newSession(sessionCfg)

    if (args.prompt.isNotEmpty()) {
        session.appendText(args.prompt)
    }

    val opts = GenerateOpts(
        maxTokens = args.maxTokens,
        temperature = Settings.TEMPERATURE,
        topP = Settings.TOP_P,
        topK = Settings.TOP_K,
        repetitionPenalty = Settings.REPETITION_PENALTY,
        stopTokens = emptyList(),
        flushEveryTokens = Settings.FLUSH_EVERY_TOKENS,
        flushEveryMs = Settings.FLUSH_EVERY_MS,
    )
    val output = session.generate(opts)
    return output.tokens
}
