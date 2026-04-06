use std::path::Path;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "wick", version, about = "Rust-native LLM inference engine")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run inference on a prompt.
    Run {
        /// Path to the GGUF model file.
        #[arg(short, long)]
        model: String,

        /// The prompt to generate from.
        #[arg(short, long)]
        prompt: String,

        /// Maximum number of tokens to generate.
        #[arg(long, default_value_t = 256)]
        max_tokens: usize,

        /// Sampling temperature.
        #[arg(long, default_value_t = 0.7)]
        temperature: f32,

        /// Device to use: cpu, gpu, or auto.
        #[arg(long, default_value = "auto")]
        device: String,

        /// Raw token IDs (comma-separated). Overrides --prompt when set.
        #[arg(long)]
        token_ids: Option<String>,
    },

    /// Inspect a GGUF model file.
    Inspect {
        /// Path to the GGUF model file.
        #[arg(short, long)]
        model: String,
    },

    /// Interactive chat session.
    Chat {
        /// Path to the GGUF model file.
        #[arg(short, long)]
        model: String,

        /// System prompt.
        #[arg(long)]
        system: Option<String>,
    },

    /// Tokenize text and print token IDs (for comparison with HuggingFace).
    Tokenize {
        /// Path to the GGUF model file.
        #[arg(short, long)]
        model: String,

        /// Text to tokenize.
        #[arg(short, long)]
        text: String,
    },

    /// Benchmark decode throughput with stable in-process measurements.
    ///
    /// Loads the model once, runs a short warmup, then measures decode tok/s
    /// over N runs and reports p10/p50/p90/mean/stddev. Production mode only
    /// (WICK_PROFILE must be unset) — profile-mode timings are diagnostic and
    /// don't predict real decode throughput.
    Bench {
        /// Path to the GGUF model file.
        #[arg(short, long)]
        model: String,

        /// The prompt to benchmark on.
        #[arg(short, long, default_value = "The capital of France is")]
        prompt: String,

        /// Number of measured runs.
        #[arg(long, default_value_t = 20)]
        runs: usize,

        /// Warmup runs (discarded). Primes Metal shader cache and GPU clock.
        #[arg(long, default_value_t = 2)]
        warmup: usize,

        /// Max tokens to decode per run.
        #[arg(long, default_value_t = 128)]
        max_tokens: usize,

        /// Device to use: cpu, gpu, metal, or auto.
        #[arg(long, default_value = "auto")]
        device: String,
    },
}

fn load_model_for_device(path: &Path, device: &str) -> Result<Box<dyn wick::model::Model>> {
    let open = || wick::gguf::GgufFile::open(path);

    match device {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        "metal" => {
            eprintln!("Using native Metal backend");
            wick::model::load_model_metal(open()?, path)
        }
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        "metal" => {
            anyhow::bail!("Metal backend not available (compile with --features metal on macOS)")
        }
        #[cfg(feature = "gpu")]
        "gpu" | "wgpu" => {
            eprintln!("Using wgpu GPU backend");
            wick::model::load_model_gpu(open()?)
        }
        #[cfg(not(feature = "gpu"))]
        "gpu" | "wgpu" => anyhow::bail!("GPU backend not available (compile with --features gpu)"),
        "cpu" => {
            eprintln!("Using CPU backend");
            wick::model::load_model(open()?)
        }
        _ => load_model_auto(path),
    }
}

/// Auto device selection: metal > wgpu > cpu, with runtime fallback.
/// Each attempt opens the GGUF fresh so a failed GPU init doesn't
/// consume the file handle needed for the CPU fallback.
fn load_model_auto(path: &Path) -> Result<Box<dyn wick::model::Model>> {
    let open = || wick::gguf::GgufFile::open(path);

    // Try Metal first (macOS/iOS only).
    #[cfg(all(feature = "metal", target_os = "macos"))]
    match wick::model::load_model_metal(open()?, path) {
        Ok(m) => {
            eprintln!("Using native Metal backend (auto)");
            return Ok(m);
        }
        Err(e) => {
            eprintln!("Metal unavailable ({e}), trying next backend");
        }
    }

    // Try wgpu (any platform with Vulkan/Metal/DX12).
    #[cfg(feature = "gpu")]
    match open().and_then(wick::model::load_model_gpu) {
        Ok(m) => {
            eprintln!("Using wgpu GPU backend (auto)");
            return Ok(m);
        }
        Err(e) => {
            eprintln!("wgpu GPU unavailable ({e}), falling back to CPU");
        }
    }

    // CPU fallback — always available.
    eprintln!("Using CPU backend (auto)");
    wick::model::load_model(open()?)
}

/// (p10, p50, p90, mean, stddev)
fn summarize(mut xs: Vec<f64>) -> (f64, f64, f64, f64, f64) {
    assert!(!xs.is_empty());
    xs.sort_by(|a, b| a.total_cmp(b));
    let n = xs.len();
    let p = |q: f64| {
        let idx = ((n as f64 - 1.0) * q).round() as usize;
        xs[idx]
    };
    let mean = xs.iter().sum::<f64>() / n as f64;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    (p(0.1), p(0.5), p(0.9), mean, var.sqrt())
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    match cli.command {
        Command::Run {
            model,
            prompt,
            max_tokens,
            temperature,
            device,
            token_ids,
        } => {
            let gguf = wick::gguf::GgufFile::open(Path::new(&model))?;
            let tokenizer = wick::tokenizer::BpeTokenizer::from_gguf(&gguf)?;
            let add_bos = gguf
                .get_bool("tokenizer.ggml.add_bos_token")
                .unwrap_or(false);

            let loaded_model = load_model_for_device(Path::new(&model), &device)?;

            let tokens = if let Some(ids) = &token_ids {
                // Parse comma-separated token IDs
                ids.split(',')
                    .map(|s| s.trim().parse::<u32>())
                    .collect::<Result<Vec<_>, _>>()?
            } else {
                let mut toks = Vec::new();
                if add_bos {
                    if let Some(bos) = tokenizer.bos_token() {
                        toks.push(bos);
                    }
                }
                toks.extend_from_slice(&tokenizer.encode(&prompt));
                toks
            };

            eprintln!(
                "Model: {} | {} layers | hidden={}",
                loaded_model.config().architecture,
                loaded_model.config().n_layers,
                loaded_model.config().hidden_size
            );
            eprintln!("Prompt tokens: {}", tokens.len());

            let config = wick::engine::GenerateConfig {
                max_tokens,
                sampler: wick::sampler::SamplerConfig {
                    temperature,
                    ..Default::default()
                },
                silent: false,
            };

            let result =
                wick::engine::generate(loaded_model.as_ref(), &tokenizer, &tokens, &config)?;

            eprintln!();
            eprintln!("---");
            eprintln!("Prompt tokens: {}", result.prompt_tokens);
            eprintln!("Generated tokens: {}", result.generated_tokens);
            eprintln!("Prefill: {:.1} tok/s", result.prefill_tok_per_sec);
            eprintln!("Decode: {:.1} tok/s", result.decode_tok_per_sec);
        }
        Command::Inspect { model } => {
            let gguf = wick::gguf::GgufFile::open(Path::new(&model))?;
            gguf.print_inspect();
        }
        Command::Tokenize { model, text } => {
            let gguf = wick::gguf::GgufFile::open(Path::new(&model))?;
            let tok = wick::tokenizer::BpeTokenizer::from_gguf(&gguf)?;
            let ids = tok.encode(&text);
            println!("{ids:?}");
        }
        Command::Chat { model, .. } => {
            println!("wick chat: model={model}");
            println!("Not yet implemented — coming in Phase 6.");
        }
        Command::Bench {
            model,
            prompt,
            runs,
            warmup,
            max_tokens,
            device,
        } => {
            anyhow::ensure!(runs >= 1, "--runs must be >= 1");
            if std::env::var("WICK_PROFILE").is_ok() {
                eprintln!(
                    "warning: WICK_PROFILE is set — bench numbers will be inflated by profile overhead"
                );
            }

            let gguf = wick::gguf::GgufFile::open(Path::new(&model))?;
            let tokenizer = wick::tokenizer::BpeTokenizer::from_gguf(&gguf)?;
            let add_bos = gguf
                .get_bool("tokenizer.ggml.add_bos_token")
                .unwrap_or(false);
            let loaded_model = load_model_for_device(Path::new(&model), &device)?;

            let mut tokens = Vec::new();
            if add_bos {
                if let Some(bos) = tokenizer.bos_token() {
                    tokens.push(bos);
                }
            }
            tokens.extend_from_slice(&tokenizer.encode(&prompt));

            eprintln!(
                "Model: {} | {} layers | hidden={}",
                loaded_model.config().architecture,
                loaded_model.config().n_layers,
                loaded_model.config().hidden_size
            );
            eprintln!(
                "Prompt tokens: {} | max_tokens: {} | warmup: {} | runs: {}",
                tokens.len(),
                max_tokens,
                warmup,
                runs
            );

            // Greedy (temp=0): deterministic, bench-friendly.
            let config = wick::engine::GenerateConfig {
                max_tokens,
                sampler: wick::sampler::SamplerConfig {
                    temperature: 0.0,
                    ..Default::default()
                },
                silent: true,
            };

            let run_once = || -> Result<(f64, f64)> {
                let r =
                    wick::engine::generate(loaded_model.as_ref(), &tokenizer, &tokens, &config)?;
                Ok((r.prefill_tok_per_sec, r.decode_tok_per_sec))
            };

            for i in 0..warmup {
                eprintln!("warmup {}/{}", i + 1, warmup);
                let _ = run_once()?;
            }

            let mut decode_tps = Vec::with_capacity(runs);
            let mut prefill_tps = Vec::with_capacity(runs);
            for i in 0..runs {
                let (pf, dc) = run_once()?;
                eprintln!(
                    "run {}/{}: prefill={pf:.0} decode={dc:.1} tok/s",
                    i + 1,
                    runs
                );
                decode_tps.push(dc);
                prefill_tps.push(pf);
            }
            eprintln!();

            let (p10, p50, p90, mean, stddev) = summarize(decode_tps);
            eprintln!(
                "decode tok/s: p50={p50:.1} p10={p10:.1} p90={p90:.1} mean={mean:.1} stddev={stddev:.1} (n={runs})"
            );
            let (p10, p50, p90, mean, stddev) = summarize(prefill_tps);
            eprintln!(
                "prefill tok/s: p50={p50:.0} p10={p10:.0} p90={p90:.0} mean={mean:.0} stddev={stddev:.0} (n={runs})"
            );
        }
    }

    Ok(())
}
