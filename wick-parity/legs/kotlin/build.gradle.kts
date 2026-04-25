// wick-parity Kotlin-via-JNA leg.
//
// Builds a fat jar (`build/libs/wick-parity-kotlin-all.jar`) that
// loads the generated `wick_ffi.kt` bindings + `libwick_ffi.so` (or
// `libwick_ffi.dylib`) through JNA, drives one harness run per stdin
// JSON request, and writes the resulting `RunOutput` to stdout. The
// Rust harness in `wick-parity/src/lib.rs::run_kotlin_jna` spawns
// this jar as a subprocess and diffs the tokens against the Rust
// reference leg.
//
// Plain Kotlin/JVM — no Android plugins, no Maven publishing here.
// The mobile-consumer story for these bindings lives elsewhere.

plugins {
    kotlin("jvm") version "2.1.10"
    kotlin("plugin.serialization") version "2.1.10"
    application
    // Shadow packs all runtime classpath deps (binding + JNA + kotlinx
    // libs) into a single self-contained jar. Subprocess invocation is
    // simpler when there's exactly one artifact to point `java -jar` at.
    id("com.gradleup.shadow") version "8.3.5"
}

repositories {
    mavenCentral()
}

dependencies {
    // UniFFI's Kotlin output speaks `com.sun.jna.*` — required at
    // both compile + runtime to call the loaded `libwick_ffi.so`.
    // Pin a version recent enough to support all the JNA features
    // UniFFI 0.31's Kotlin generator emits.
    implementation("net.java.dev.jna:jna:5.15.0")

    // The generated binding compiles against kotlinx.coroutines for
    // the `async fn` exports (`Session.generateAsync` and friends).
    // Even legs that don't await still need it on the classpath for
    // the binding to type-check.
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.10.2")

    // JSON I/O: kotlinx.serialization keeps the runner self-contained
    // (no Jackson / Moshi) and matches the `serde_json` shape the
    // Rust harness emits.
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.8.1")
}

// Pull the vendored UniFFI Kotlin binding directory into our source
// set. The generated `wick_ffi.kt` lives next to the wick-ffi crate
// in `wick-ffi/bindings/kotlin/` and is regenerated via
// `just bindings` whenever the FFI surface changes — wiring it
// in via srcDirs avoids a duplicate copy that would silently drift.
sourceSets {
    main {
        kotlin.srcDirs(
            "src/main/kotlin",
            file("../../../wick-ffi/bindings/kotlin"),
        )
    }
}

application {
    mainClass = "com.hyeonslab.wickparity.MainKt"
}

kotlin {
    // Match the JDK 21 baseline used everywhere else in the wick
    // ecosystem. SDKMAN's 21.0.9-zulu is the reference per the user's
    // global CLAUDE.md guidance.
    jvmToolchain(21)
}

// Quiet the shadow plugin's "implementation has only test deps" warning;
// keep the assembled jar deterministic (no timestamps in the manifest).
tasks.shadowJar {
    archiveClassifier.set("all")
    isPreserveFileTimestamps = false
    isReproducibleFileOrder = true
}
