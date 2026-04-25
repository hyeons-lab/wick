//! Parity harness CLI.
//!
//! ```text
//! wick-parity dump --via {rust|ffi|kotlin-jna|swift-uniffi} \
//!     --bundle <id> --quant <q> --prompt <text> \
//!     [--max-tokens 16] [--seed 0] \
//!     [--runner <jar-or-bin>] [--lib-dir <dir>]
//!
//! wick-parity check --bundle <id> --quant <q> --prompt <text> \
//!     [--max-tokens 16] [--seed 0] \
//!     [--kotlin-runner <jar>] [--swift-runner <bin>] [--lib-dir <dir>] \
//!     [--max-slowdown 2.0] [--fail-on-slowdown]
//! ```
//!
//! `dump` runs one leg and emits a [`RunOutput`] (token vec + wall-clock
//! latency) as JSON to stdout.
//!
//! `check` always runs rust + ffi; if `--kotlin-runner` / `--swift-runner`
//! are supplied, those legs are added. Token streams are diffed against
//! rust (the reference). Wall-clock latency per leg is printed alongside
//! a `non_rust_ms / rust_ms` ratio. Default `--max-slowdown 2.0` prints
//! `WARN` on breach but exits 0; flip `--fail-on-slowdown` to make the
//! threshold load-bearing.
//!
//! Future binding legs plug in by shelling out to their own
//! `dump`-equivalent binary that emits a [`RunOutput`] (with the
//! `wall_clock_ms` field populated by the runner so subprocess startup
//! cost doesn't pollute the FFI overhead measurement).

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use wick_parity::{
    RunArgs, RunOutput, default_cache_dir, first_divergence, run_ffi, run_kotlin_jna, run_rust,
    run_swift_uniffi,
};

#[derive(Parser, Debug)]
#[command(
    name = "wick-parity",
    about = "Parity harness — run a fixed prompt through every binding leg and compare outputs."
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Run one leg and print the resulting tokens as JSON.
    Dump {
        #[arg(long, value_enum)]
        via: Leg,
        /// Path to a subprocess runner — either the Kotlin fat jar
        /// (`wick-parity/legs/kotlin/build/libs/wick-parity-kotlin-all.jar`)
        /// or the Swift SPM binary
        /// (`wick-parity/legs/swift/.build/release/WickParitySwift`).
        /// Required when `--via kotlin-jna` or `--via swift-uniffi`;
        /// passing this flag with any other `--via` value is rejected
        /// as a misuse.
        #[arg(long)]
        runner: Option<PathBuf>,
        /// Directory containing `libwick_ffi.{so,dylib}`. Defaults to
        /// `<workspace>/target/debug`. Used by the `kotlin-jna` leg
        /// (passed to JNA via `-Djna.library.path=...`) and the
        /// `swift-uniffi` leg (exported as `DYLD_LIBRARY_PATH`); a
        /// no-op for any other `--via` value.
        #[arg(long)]
        lib_dir: Option<PathBuf>,
        #[command(flatten)]
        common: CommonArgs,
    },
    /// Run rust + ffi (and optionally kotlin-jna / swift-uniffi if
    /// runner paths are provided) and diff token streams +
    /// wall-clock latency. Always runs rust + ffi; subprocess legs
    /// are opt-in via `--kotlin-runner` and `--swift-runner`.
    Check {
        /// Path to the kotlin-jna runner fat jar
        /// (`wick-parity/legs/kotlin/build/libs/wick-parity-kotlin-all.jar`).
        /// If set, the kotlin-jna leg runs alongside rust + ffi and
        /// feeds into the perf threshold check below.
        #[arg(long)]
        kotlin_runner: Option<PathBuf>,
        /// Path to the swift-uniffi runner binary
        /// (`wick-parity/legs/swift/.build/release/WickParitySwift`).
        /// If set, the swift-uniffi leg runs alongside rust + ffi.
        #[arg(long)]
        swift_runner: Option<PathBuf>,
        /// Directory containing `libwick_ffi.{so,dylib}`. Defaults
        /// to `<workspace>/target/debug`. Used by both subprocess
        /// legs; ignored if neither `--kotlin-runner` nor
        /// `--swift-runner` is set.
        #[arg(long)]
        lib_dir: Option<PathBuf>,
        /// Performance threshold — `non_rust_ms / rust_ms` ratio.
        /// Default `2.0`: a non-rust leg taking >2× the rust
        /// reference's wall clock prints a `WARN`. Set to a smaller
        /// value to tighten the alarm; set to a very large number
        /// (e.g. `1e9`) to disable the perf check entirely. Must be
        /// strictly positive — `0` and negative values are rejected
        /// at parse time since the threshold logic only makes sense
        /// for a positive multiplier.
        #[arg(long, default_value_t = 2.0, value_parser = parse_positive_f64)]
        max_slowdown: f64,
        /// Make perf threshold breaches exit non-zero (default is
        /// warn-only). Off by default to avoid CI flakiness from
        /// runner-load variance; flip on once the threshold has
        /// been validated against real CI variance.
        #[arg(long, default_value_t = false)]
        fail_on_slowdown: bool,
        #[command(flatten)]
        common: CommonArgs,
    },
}

#[derive(Debug, Clone, ValueEnum)]
enum Leg {
    /// `wick::WickEngine` directly.
    #[value(name = "rust")]
    Rust,
    /// `wick_ffi::WickEngine` through its Rust public surface.
    #[value(name = "ffi")]
    Ffi,
    /// Kotlin runner under `wick-parity/legs/kotlin/`, loading the
    /// generated UniFFI bindings + `libwick_ffi.{so,dylib}` via JNA.
    #[value(name = "kotlin-jna")]
    KotlinJna,
    /// Swift runner under `wick-parity/legs/swift/`, built via SPM,
    /// loading the generated UniFFI Swift bindings + linked against
    /// the wick-ffi cdylib at build time.
    #[value(name = "swift-uniffi")]
    SwiftUniffi,
}

impl Leg {
    fn label(&self) -> &'static str {
        match self {
            Leg::Rust => "rust",
            Leg::Ffi => "ffi",
            Leg::KotlinJna => "kotlin-jna",
            Leg::SwiftUniffi => "swift-uniffi",
        }
    }

    fn needs_runner(&self) -> bool {
        matches!(self, Leg::KotlinJna | Leg::SwiftUniffi)
    }
}

/// Clap value parser for `--max-slowdown`. Rejects non-finite and
/// non-positive values at parse time so the perf threshold logic
/// downstream doesn't have to defend against `0` (every leg would
/// breach), negatives (every leg appears under threshold), or NaN
/// / infinity (comparison semantics get weird).
fn parse_positive_f64(s: &str) -> Result<f64, String> {
    let v: f64 = s.parse().map_err(|e| format!("not a number: {e}"))?;
    if !v.is_finite() {
        return Err(format!("must be finite, got {v}"));
    }
    if v <= 0.0 {
        return Err(format!("must be > 0, got {v}"));
    }
    Ok(v)
}

fn default_lib_dir() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .expect("CARGO_MANIFEST_DIR has no parent")
        .join("target")
        .join("debug")
}

#[derive(clap::Args, Debug)]
struct CommonArgs {
    #[arg(long)]
    bundle: String,
    #[arg(long)]
    quant: String,
    #[arg(long)]
    prompt: String,
    #[arg(long, default_value_t = 16)]
    max_tokens: u32,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Override the default cache root. Falls back to
    /// `$WICK_PARITY_CACHE_DIR` and then to
    /// `<workspace>/target/tmp/wick-parity-cache/`.
    #[arg(long)]
    cache_dir: Option<PathBuf>,
}

fn resolve_cache_dir(c: &CommonArgs) -> Result<PathBuf> {
    if let Some(p) = &c.cache_dir {
        std::fs::create_dir_all(p).with_context(|| format!("create {}", p.display()))?;
        return Ok(p.clone());
    }
    default_cache_dir()
}

fn run_leg(
    leg: &Leg,
    args: &RunArgs<'_>,
    runner: Option<&Path>,
    lib_dir: Option<&Path>,
) -> Result<(Vec<u32>, Option<u64>)> {
    match leg {
        Leg::Rust => run_rust(args),
        Leg::Ffi => run_ffi(args),
        Leg::KotlinJna | Leg::SwiftUniffi => {
            let leg_label = leg.label();
            let runner = runner.ok_or_else(|| {
                anyhow::anyhow!("--runner <path> is required for --via {leg_label}")
            })?;
            let default_lib;
            let lib_dir = match lib_dir {
                Some(p) => p,
                None => {
                    default_lib = default_lib_dir();
                    &default_lib
                }
            };
            match leg {
                Leg::KotlinJna => run_kotlin_jna(args, runner, lib_dir),
                Leg::SwiftUniffi => run_swift_uniffi(args, runner, lib_dir),
                _ => unreachable!("outer match guarded the variants"),
            }
        }
    }
}

fn build_output(
    via: &str,
    args: &RunArgs<'_>,
    tokens: Vec<u32>,
    wall_clock_ms: Option<u64>,
) -> RunOutput {
    RunOutput {
        via: via.to_string(),
        bundle: args.bundle.to_string(),
        quant: args.quant.to_string(),
        prompt: args.prompt.to_string(),
        max_tokens: args.max_tokens,
        seed: args.seed,
        tokens,
        wall_clock_ms,
    }
}

fn main() -> ExitCode {
    match real_main() {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::from(2)
        }
    }
}

fn real_main() -> Result<ExitCode> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Dump {
            via,
            runner,
            lib_dir,
            common,
        } => {
            // Surface the misuse early — `--runner` only matters for
            // the subprocess legs (kotlin-jna / swift-uniffi); passing
            // it with any other `--via` value probably means the user
            // got the flag wrong.
            if runner.is_some() && !via.needs_runner() {
                bail!("--runner is only valid with --via kotlin-jna or --via swift-uniffi");
            }
            let cache = resolve_cache_dir(&common)?;
            let args = RunArgs {
                bundle: &common.bundle,
                quant: &common.quant,
                prompt: &common.prompt,
                max_tokens: common.max_tokens,
                seed: common.seed,
                cache_dir: &cache,
            };
            let (tokens, wall_clock_ms) =
                run_leg(&via, &args, runner.as_deref(), lib_dir.as_deref())?;
            let out = build_output(via.label(), &args, tokens, wall_clock_ms);
            // Pretty JSON makes the diff readable when a human inspects
            // the dump; one-line JSON would be friendlier for piping
            // into `jq` but the dump is human-driven, not pipeline-fed.
            println!("{}", serde_json::to_string_pretty(&out)?);
            Ok(ExitCode::SUCCESS)
        }
        Cmd::Check {
            kotlin_runner,
            swift_runner,
            lib_dir,
            max_slowdown,
            fail_on_slowdown,
            common,
        } => {
            let cache = resolve_cache_dir(&common)?;
            let args = RunArgs {
                bundle: &common.bundle,
                quant: &common.quant,
                prompt: &common.prompt,
                max_tokens: common.max_tokens,
                seed: common.seed,
                cache_dir: &cache,
            };
            run_check(
                &args,
                kotlin_runner.as_deref(),
                swift_runner.as_deref(),
                lib_dir.as_deref(),
                max_slowdown,
                fail_on_slowdown,
            )
        }
    }
}

/// Resolved per-leg result for `Cmd::Check`. `wall_clock_ms` is
/// `Option<u64>` to honestly represent legs whose runner predates
/// the timing field — printing `n/a` and skipping the ratio check
/// beats reporting `0 ms` and a bogus `0.00×` that would mask a
/// real regression.
struct LegResult {
    label: &'static str,
    tokens: Vec<u32>,
    wall_clock_ms: Option<u64>,
}

/// Format a wall-clock measurement for the right-aligned `ms`
/// column of the perf table. `None` → `n/a`.
fn fmt_ms(ms: Option<u64>) -> String {
    match ms {
        Some(v) => format!("{v:>6} ms"),
        None => format!("{:>9}", "n/a"),
    }
}

fn run_check(
    args: &RunArgs<'_>,
    kotlin_runner: Option<&Path>,
    swift_runner: Option<&Path>,
    lib_dir: Option<&Path>,
    max_slowdown: f64,
    fail_on_slowdown: bool,
) -> Result<ExitCode> {
    // Resolve lib_dir once if either subprocess leg needs it. The
    // closure-style dispatch in `run_leg` would re-resolve per call;
    // doing it here keeps the side-channel for the `--lib-dir` arg
    // explicit + lets us reuse the resolved path.
    let resolved_lib_dir;
    let lib_dir_resolved = match lib_dir {
        Some(p) => p,
        None => {
            resolved_lib_dir = default_lib_dir();
            &resolved_lib_dir
        }
    };

    let mut results: Vec<LegResult> = Vec::with_capacity(4);

    let (rust_tokens, rust_ms) = run_rust(args).context("rust leg")?;
    results.push(LegResult {
        label: "rust",
        tokens: rust_tokens,
        wall_clock_ms: rust_ms,
    });

    let (ffi_tokens, ffi_ms) = run_ffi(args).context("ffi leg")?;
    results.push(LegResult {
        label: "ffi",
        tokens: ffi_tokens,
        wall_clock_ms: ffi_ms,
    });

    if let Some(jar) = kotlin_runner {
        let (tokens, ms) = run_kotlin_jna(args, jar, lib_dir_resolved).context("kotlin-jna leg")?;
        results.push(LegResult {
            label: "kotlin-jna",
            tokens,
            wall_clock_ms: ms,
        });
    }
    if let Some(bin) = swift_runner {
        let (tokens, ms) =
            run_swift_uniffi(args, bin, lib_dir_resolved).context("swift-uniffi leg")?;
        results.push(LegResult {
            label: "swift-uniffi",
            tokens,
            wall_clock_ms: ms,
        });
    }

    // Token diff: every non-rust leg is compared against rust.
    // `split_first` is safe — we always push rust first.
    let (rust_result, rest) = results.split_first().expect("rust leg always pushed first");
    let rust_label = rust_result.label;
    let rust_tokens = &rust_result.tokens;
    let rust_ms = rust_result.wall_clock_ms;

    let mut diff_failure = false;
    for r in rest {
        if let Some(idx) = first_divergence(rust_tokens, &r.tokens) {
            eprintln!(
                "FAIL: {rust_label} ↔ {} diverged at index {idx} (bundle={} quant={})",
                r.label, args.bundle, args.quant
            );
            let start = idx.saturating_sub(2);
            let end = idx.saturating_add(3);
            let rl_window = start..end.min(rust_tokens.len());
            let rr_window = start..end.min(r.tokens.len());
            eprintln!(
                "  {rust_label}[{rl_window:?}] = {:?}",
                &rust_tokens[rl_window.clone()]
            );
            eprintln!(
                "  {:>w$}[{rr_window:?}] = {:?}",
                r.label,
                &r.tokens[rr_window.clone()],
                w = rust_label.len()
            );
            eprintln!(
                "  {rust_label}.len() = {}, {}.len() = {}",
                rust_tokens.len(),
                r.label,
                r.tokens.len()
            );
            diff_failure = true;
        }
    }

    // Always print per-leg timing + ratio vs rust. Even when there's
    // no token divergence, this is the visible signal users want when
    // debugging "why is parity_kotlin slow today".
    eprintln!();
    eprintln!("perf (max-slowdown threshold = {max_slowdown:.2}×):");
    eprintln!("  {:<14} {}  (reference)", rust_label, fmt_ms(rust_ms));
    let mut perf_failure = false;
    for r in rest {
        // Ratio is meaningful only when both sides measured AND the
        // reference is non-zero (sub-millisecond rust on a tiny
        // fixture + hot cache would otherwise give `inf×`). Anything
        // else collapses to `(n/a)`, no breach. Honest about what we
        // know — better than printing `(0.00×)` or `(nan×)` and
        // pretending the threshold logic ran.
        let (ratio_display, breached) = match (r.wall_clock_ms, rust_ms) {
            (Some(leg_ms), Some(ref_ms)) if ref_ms > 0 => {
                let ratio = leg_ms as f64 / ref_ms as f64;
                (format!("({ratio:.2}×)"), ratio > max_slowdown)
            }
            _ => ("(n/a)".to_string(), false),
        };
        let marker = if breached { " WARN" } else { "" };
        eprintln!(
            "  {:<14} {}  {ratio_display}{marker}",
            r.label,
            fmt_ms(r.wall_clock_ms)
        );
        if breached {
            perf_failure = true;
        }
    }

    if diff_failure {
        return Ok(ExitCode::from(1));
    }
    if perf_failure && fail_on_slowdown {
        return Ok(ExitCode::from(1));
    }
    eprintln!(
        "\nOK: {} legs, all token streams match (bundle={} quant={} tokens={})",
        results.len(),
        args.bundle,
        args.quant,
        rust_tokens.len()
    );
    Ok(ExitCode::SUCCESS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_parses_dump_with_defaults() {
        let cli = Cli::try_parse_from([
            "wick-parity",
            "dump",
            "--via",
            "rust",
            "--bundle",
            "B",
            "--quant",
            "Q",
            "--prompt",
            "p",
        ])
        .expect("parse dump");
        match cli.cmd {
            Cmd::Dump {
                via: Leg::Rust,
                runner: None,
                lib_dir: None,
                common,
            } => {
                assert_eq!(common.bundle, "B");
                assert_eq!(common.quant, "Q");
                assert_eq!(common.prompt, "p");
                assert_eq!(common.max_tokens, 16);
                assert_eq!(common.seed, 0);
                assert!(common.cache_dir.is_none());
            }
            other => panic!("unexpected cmd: {other:?}"),
        }
    }

    #[test]
    fn cli_parses_check_with_overrides() {
        let cli = Cli::try_parse_from([
            "wick-parity",
            "check",
            "--bundle",
            "B",
            "--quant",
            "Q",
            "--prompt",
            "p",
            "--max-tokens",
            "32",
            "--seed",
            "7",
        ])
        .expect("parse check");
        match cli.cmd {
            Cmd::Check {
                kotlin_runner: None,
                swift_runner: None,
                lib_dir: None,
                max_slowdown,
                fail_on_slowdown: false,
                common,
            } => {
                assert_eq!(common.max_tokens, 32);
                assert_eq!(common.seed, 7);
                assert!((max_slowdown - 2.0).abs() < f64::EPSILON);
            }
            other => panic!("unexpected cmd: {other:?}"),
        }
    }

    #[test]
    fn cli_parses_check_with_perf_flags() {
        let cli = Cli::try_parse_from([
            "wick-parity",
            "check",
            "--bundle",
            "B",
            "--quant",
            "Q",
            "--prompt",
            "p",
            "--kotlin-runner",
            "/tmp/r.jar",
            "--swift-runner",
            "/tmp/r",
            "--max-slowdown",
            "1.5",
            "--fail-on-slowdown",
        ])
        .expect("parse check perf flags");
        match cli.cmd {
            Cmd::Check {
                kotlin_runner: Some(j),
                swift_runner: Some(s),
                max_slowdown,
                fail_on_slowdown: true,
                ..
            } => {
                assert_eq!(j, std::path::PathBuf::from("/tmp/r.jar"));
                assert_eq!(s, std::path::PathBuf::from("/tmp/r"));
                assert!((max_slowdown - 1.5).abs() < f64::EPSILON);
            }
            other => panic!("unexpected cmd: {other:?}"),
        }
    }

    #[test]
    fn build_output_round_trips_args() {
        let args = RunArgs {
            bundle: "B",
            quant: "Q",
            prompt: "p",
            max_tokens: 16,
            seed: 0,
            cache_dir: std::path::Path::new("/tmp"),
        };
        let out = build_output("rust", &args, vec![1, 2, 3], Some(42));
        assert_eq!(out.via, "rust");
        assert_eq!(out.bundle, "B");
        assert_eq!(out.tokens, vec![1, 2, 3]);
        assert_eq!(out.wall_clock_ms, Some(42));
    }

    #[test]
    fn parse_positive_f64_accepts_positive() {
        assert!((parse_positive_f64("2.0").unwrap() - 2.0).abs() < f64::EPSILON);
        assert!((parse_positive_f64("0.5").unwrap() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn parse_positive_f64_rejects_zero_negative_nan_inf() {
        assert!(parse_positive_f64("0").is_err());
        assert!(parse_positive_f64("-1.5").is_err());
        assert!(parse_positive_f64("nan").is_err());
        assert!(parse_positive_f64("inf").is_err());
        assert!(parse_positive_f64("not-a-number").is_err());
    }
}
