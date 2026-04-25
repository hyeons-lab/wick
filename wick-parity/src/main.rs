//! Parity harness CLI.
//!
//! ```text
//! wick-parity dump --via {rust|ffi} --bundle <id> --quant <q> \
//!     --prompt <text> [--max-tokens 16] [--seed 0]
//!
//! wick-parity check --bundle <id> --quant <q> --prompt <text> \
//!     [--max-tokens 16] [--seed 0]
//! ```
//!
//! `dump` runs one leg and emits a [`RunOutput`] as JSON to stdout.
//! `check` runs every Rust-side leg, diffs token-by-token, exits 0
//! on match or 1 with a diff summary on mismatch.
//!
//! Future binding legs (Kotlin via JNA, Swift via UniFFI) plug in by
//! shelling out to their own `dump`-equivalent binary and feeding the
//! resulting JSON into `check`'s diff. The contract is intentionally
//! minimal: produce a `Vec<u32>` from the same `RunArgs` and emit it
//! as `RunOutput.tokens`.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use wick_parity::{
    RunArgs, RunOutput, default_cache_dir, first_divergence, run_ffi, run_kotlin_jna, run_rust,
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
        /// Path to the Kotlin runner fat jar
        /// (`wick-parity/legs/kotlin/build/libs/wick-parity-kotlin-all.jar`).
        /// Required when `--via kotlin-jna`; ignored otherwise.
        #[arg(long)]
        runner: Option<PathBuf>,
        /// Directory containing `libwick_ffi.{so,dylib}`. Defaults to
        /// `<workspace>/target/debug`. Used only by the `kotlin-jna`
        /// leg (passed to JNA via `-Djna.library.path=...`).
        #[arg(long)]
        lib_dir: Option<PathBuf>,
        #[command(flatten)]
        common: CommonArgs,
    },
    /// Run every Rust-side leg and diff the outputs.
    Check {
        #[command(flatten)]
        common: CommonArgs,
    },
}

#[derive(Debug, Clone, ValueEnum)]
enum Leg {
    /// `wick::WickEngine` directly.
    Rust,
    /// `wick_ffi::WickEngine` through its Rust public surface.
    Ffi,
    /// Kotlin runner under `wick-parity/legs/kotlin/`, loading the
    /// generated UniFFI bindings + `libwick_ffi.{so,dylib}` via JNA.
    KotlinJna,
}

impl Leg {
    fn label(&self) -> &'static str {
        match self {
            Leg::Rust => "rust",
            Leg::Ffi => "ffi",
            Leg::KotlinJna => "kotlin-jna",
        }
    }
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
) -> Result<Vec<u32>> {
    match leg {
        Leg::Rust => run_rust(args),
        Leg::Ffi => run_ffi(args),
        Leg::KotlinJna => {
            let runner = runner.ok_or_else(|| {
                anyhow::anyhow!("--runner <jar> is required for --via kotlin-jna")
            })?;
            let default_lib;
            let lib_dir = match lib_dir {
                Some(p) => p,
                None => {
                    default_lib = default_lib_dir();
                    &default_lib
                }
            };
            run_kotlin_jna(args, runner, lib_dir)
        }
    }
}

fn build_output(via: &str, args: &RunArgs<'_>, tokens: Vec<u32>) -> RunOutput {
    RunOutput {
        via: via.to_string(),
        bundle: args.bundle.to_string(),
        quant: args.quant.to_string(),
        prompt: args.prompt.to_string(),
        max_tokens: args.max_tokens,
        seed: args.seed,
        tokens,
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
            // the kotlin leg; passing it without `--via kotlin-jna`
            // probably means the user got the flag wrong.
            if runner.is_some() && !matches!(via, Leg::KotlinJna) {
                bail!("--runner is only valid with --via kotlin-jna");
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
            let tokens = run_leg(&via, &args, runner.as_deref(), lib_dir.as_deref())?;
            let out = build_output(via.label(), &args, tokens);
            // Pretty JSON makes the diff readable when a human inspects
            // the dump; one-line JSON would be friendlier for piping
            // into `jq` but the dump is human-driven, not pipeline-fed.
            println!("{}", serde_json::to_string_pretty(&out)?);
            Ok(ExitCode::SUCCESS)
        }
        Cmd::Check { common } => {
            let cache = resolve_cache_dir(&common)?;
            let args = RunArgs {
                bundle: &common.bundle,
                quant: &common.quant,
                prompt: &common.prompt,
                max_tokens: common.max_tokens,
                seed: common.seed,
                cache_dir: &cache,
            };
            let rust = run_rust(&args)?;
            let ffi = run_ffi(&args)?;
            match first_divergence(&rust, &ffi) {
                None => {
                    eprintln!(
                        "OK: rust ↔ ffi parity (bundle={} quant={} tokens={})",
                        args.bundle,
                        args.quant,
                        rust.len()
                    );
                    Ok(ExitCode::SUCCESS)
                }
                Some(idx) => {
                    eprintln!(
                        "FAIL: rust ↔ ffi diverged at index {idx} (bundle={} quant={})",
                        args.bundle, args.quant
                    );
                    // Clamp each window to the slice's own length —
                    // when divergence is at the tail (e.g. one leg
                    // returned fewer tokens), an unclamped end goes
                    // out of bounds and `slice[range]` panics; even
                    // `get()` returns `None`, hiding the surrounding
                    // tokens that are the whole point of the dump.
                    let start = idx.saturating_sub(2);
                    let end = idx.saturating_add(3);
                    let rust_window = start..end.min(rust.len());
                    let ffi_window = start..end.min(ffi.len());
                    eprintln!("  rust[{rust_window:?}] = {:?}", &rust[rust_window.clone()]);
                    eprintln!("  ffi [{ffi_window:?}] = {:?}", &ffi[ffi_window.clone()]);
                    eprintln!("  rust.len() = {}, ffi.len() = {}", rust.len(), ffi.len());
                    Ok(ExitCode::from(1))
                }
            }
        }
    }
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
            Cmd::Check { common } => {
                assert_eq!(common.max_tokens, 32);
                assert_eq!(common.seed, 7);
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
        let out = build_output("rust", &args, vec![1, 2, 3]);
        assert_eq!(out.via, "rust");
        assert_eq!(out.bundle, "B");
        assert_eq!(out.tokens, vec![1, 2, 3]);
    }
}
