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

    /// Run benchmarks on a model.
    Bench {
        /// Path to the GGUF model file.
        #[arg(short, long)]
        model: String,

        /// Number of prompt tokens to benchmark.
        #[arg(long, default_value_t = 128)]
        prompt_tokens: usize,

        /// Number of tokens to generate.
        #[arg(long, default_value_t = 128)]
        gen_tokens: usize,
    },
}

fn load_model_for_device(
    gguf: wick::gguf::GgufFile,
    device: &str,
) -> Result<Box<dyn wick::model::Model>> {
    match device {
        #[cfg(feature = "gpu")]
        "gpu" => {
            eprintln!("Using GPU backend");
            wick::model::load_model_gpu(gguf)
        }
        #[cfg(not(feature = "gpu"))]
        "gpu" => anyhow::bail!("GPU backend not available (compile with --features gpu)"),
        "cpu" => wick::model::load_model(gguf),
        _ => {
            // "auto" or unknown: try GPU if available, fall back to CPU
            #[cfg(feature = "gpu")]
            {
                eprintln!("Using GPU backend (auto)");
                return wick::model::load_model_gpu(gguf);
            }
            #[cfg(not(feature = "gpu"))]
            wick::model::load_model(gguf)
        }
    }
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

            let loaded_model = load_model_for_device(gguf, &device)?;

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
        Command::Bench { model, .. } => {
            println!("wick bench: model={model}");
            println!("Not yet implemented — coming in Phase 6.");
        }
    }

    Ok(())
}
