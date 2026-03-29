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

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    match cli.command {
        Command::Run { model, prompt, .. } => {
            println!("wick run: model={model}, prompt={prompt:?}");
            println!("Not yet implemented — coming in Phase 3.");
        }
        Command::Inspect { model } => {
            println!("wick inspect: model={model}");
            println!("Not yet implemented — coming in Phase 2.");
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
