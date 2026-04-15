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

        /// Max context window size (KV cache). Default 4096. Larger values use more
        /// memory. Context >4096 auto-switches to flash attention (~14% slower).
        #[arg(long, default_value_t = 4096)]
        context_size: usize,

        /// Path to vocoder GGUF for audio generation. Enables audio output.
        #[arg(long)]
        vocoder: Option<String>,

        /// Output WAV file for generated audio.
        #[arg(long)]
        audio_out: Option<String>,

        /// System prompt (used with --vocoder for audio mode selection).
        /// E.g. "Perform TTS." or "Respond with interleaved text and audio."
        #[arg(long)]
        system: Option<String>,

        /// Audio sampling temperature (0.0 = greedy, >0 = stochastic).
        #[arg(long, default_value_t = 0.8)]
        audio_temperature: f32,

        /// Audio top-k for stochastic sampling.
        #[arg(long, default_value_t = 4)]
        audio_top_k: usize,

        /// Directory for KV prefix cache files. Enables disk caching for prompt reuse.
        #[arg(long)]
        cache_dir: Option<String>,

        /// Max warm (memory) cache size in MB. Default 256.
        #[arg(long, default_value_t = 256)]
        cache_warm_mb: u64,

        /// Max disk cache size in GB. Default 10.
        #[arg(long, default_value_t = 10)]
        cache_disk_gb: u64,

        /// Disable KV prefix caching entirely.
        #[arg(long)]
        no_cache: bool,

        /// KV cache key compression: f32 (default) or tq3 (TurboQuant 3-bit).
        #[arg(long, default_value = "f32")]
        kv_cache_keys: String,
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

        /// Number of tokens to use for the prompt (ignores --prompt if set).
        #[arg(long)]
        prompt_tokens: Option<usize>,

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

        /// Max context window size (KV cache). Default 4096.
        #[arg(long, default_value_t = 4096)]
        context_size: usize,

        /// Disable KV prefix caching entirely.
        #[arg(long)]
        no_cache: bool,

        /// KV cache key compression: f32 (default) or tq3 (TurboQuant 3-bit).
        #[arg(long, default_value = "f32")]
        kv_cache_keys: String,
    },
}

fn load_model_for_device(
    path: &Path,
    device: &str,
    context_size: usize,
) -> Result<Box<dyn wick::model::Model>> {
    let open = || wick::gguf::GgufFile::open(path);

    match device {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        "metal" => {
            eprintln!("Using native Metal backend");
            wick::model::load_model_metal(open()?, path, context_size)
        }
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        "metal" => {
            anyhow::bail!("Metal backend not available (compile with --features metal on macOS)")
        }
        #[cfg(feature = "gpu")]
        "gpu" | "wgpu" => {
            eprintln!("Using wgpu GPU backend");
            wick::model::load_model_gpu(open()?, context_size)
        }
        #[cfg(not(feature = "gpu"))]
        "gpu" | "wgpu" => anyhow::bail!("GPU backend not available (compile with --features gpu)"),
        "cpu" => {
            eprintln!("Using CPU backend");
            wick::model::load_model(open()?, context_size)
        }
        _ => load_model_auto(path, context_size),
    }
}

/// Auto device selection: metal > wgpu > cpu, with runtime fallback.
fn load_model_auto(path: &Path, context_size: usize) -> Result<Box<dyn wick::model::Model>> {
    let open = || wick::gguf::GgufFile::open(path);

    // Try Metal first (macOS/iOS only).
    #[cfg(all(feature = "metal", target_os = "macos"))]
    match wick::model::load_model_metal(open()?, path, context_size) {
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
    match open().and_then(|g| wick::model::load_model_gpu(g, context_size)) {
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
    wick::model::load_model(open()?, context_size)
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

/// Write PCM float32 samples as a WAV file (16-bit PCM, mono).
fn write_wav(path: &str, samples: &[f32], sample_rate: u32) -> Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    let n = samples.len() as u32;
    let data_size = n * 2;
    let file_size = 36 + data_size;
    f.write_all(b"RIFF")?;
    f.write_all(&file_size.to_le_bytes())?;
    f.write_all(b"WAVE")?;
    f.write_all(b"fmt ")?;
    f.write_all(&16u32.to_le_bytes())?;
    f.write_all(&1u16.to_le_bytes())?;
    f.write_all(&1u16.to_le_bytes())?;
    f.write_all(&sample_rate.to_le_bytes())?;
    f.write_all(&(sample_rate * 2).to_le_bytes())?;
    f.write_all(&2u16.to_le_bytes())?;
    f.write_all(&16u16.to_le_bytes())?;
    f.write_all(b"data")?;
    f.write_all(&data_size.to_le_bytes())?;
    for &s in samples {
        let i16_val = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        f.write_all(&i16_val.to_le_bytes())?;
    }
    Ok(())
}

/// Parse a CLI KV-cache-compression flag value into a `KvCompression`.
///
/// Modes:
/// - `f32` / `none`: uncompressed (default)
/// - `tq3` / `turboquant`: TurboQuant on both keys (3-bit) and values (2-bit)
/// - `tq3-keys`: TurboQuant keys only (values stay f32) — debugging
/// - `tq3-values`: TurboQuant values only (keys stay f32) — debugging
fn setup_kv_compression(
    model: &dyn wick::model::Model,
    kv_cache_mode: &str,
) -> Result<wick::kv_cache::KvCompression> {
    use wick::kv_cache::KvCompression;
    let seed: u64 = 42; // deterministic default seed

    let (keys, values) = match kv_cache_mode {
        "f32" | "none" => return Ok(KvCompression::None),
        "tq3" | "turboquant" => (true, true),
        "tq3-keys" => (true, false),
        "tq3-values" => (false, true),
        other => anyhow::bail!(
            "unknown --kv-cache-keys mode: {other} (use f32, tq3, tq3-keys, or tq3-values)"
        ),
    };

    if model.turboquant_supported() {
        eprintln!(
            "TurboQuant KV compression enabled (keys: {}, values: {})",
            if keys { "3-bit" } else { "f32" },
            if values { "2-bit" } else { "f32" }
        );
        Ok(KvCompression::TurboQuant { seed, keys, values })
    } else {
        eprintln!(
            "warning: TurboQuant not supported by this model/backend; falling back to f32 KV"
        );
        Ok(KvCompression::None)
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    // Configure rayon to use P-cores only on Apple Silicon.
    wick::backend::cpu::configure_thread_pool();

    let cli = Cli::parse();

    match cli.command {
        Command::Run {
            model,
            prompt,
            max_tokens,
            temperature,
            device,
            token_ids,
            context_size,
            vocoder,
            audio_out,
            system,
            audio_temperature,
            audio_top_k,
            cache_dir,
            cache_warm_mb,
            cache_disk_gb,
            no_cache,
            kv_cache_keys,
        } => {
            let gguf = wick::gguf::GgufFile::open(Path::new(&model))?;
            let tokenizer = wick::tokenizer::BpeTokenizer::from_gguf(&gguf)?;
            let add_bos = gguf
                .get_bool("tokenizer.ggml.add_bos_token")
                .unwrap_or(false);

            let loaded_model = load_model_for_device(Path::new(&model), &device, context_size)?;
            let kv_compression = setup_kv_compression(loaded_model.as_ref(), &kv_cache_keys)?;

            // Configure KV prefix cache.
            if no_cache {
                loaded_model.configure_cache(wick::kv_cache::KvCacheConfig {
                    cache_dir: None,
                    max_warm_entries: 0,
                    max_warm_bytes: 0,
                    max_cold_bytes: 0,
                });
            } else {
                let dir = cache_dir.map(std::path::PathBuf::from).or_else(|| {
                    std::env::var("HOME")
                        .ok()
                        .map(|h| std::path::PathBuf::from(h).join(".cache/wick/kv"))
                });
                loaded_model.configure_cache(wick::kv_cache::KvCacheConfig {
                    cache_dir: dir,
                    max_warm_entries: 32,
                    max_warm_bytes: cache_warm_mb * 1024 * 1024,
                    max_cold_bytes: cache_disk_gb * 1024 * 1024 * 1024,
                });
            }

            // Build token sequence.
            let mut tokens = Vec::new();
            if system.is_some() || vocoder.is_some() {
                // Use chat template when --system or --vocoder is set.
                anyhow::ensure!(
                    system.is_some(),
                    "--system is required with --vocoder. Supported:\n  \
                     \"Respond with interleaved text and audio.\"\n  \
                     \"Perform TTS. <voice description>\"\n  \
                     \"Perform ASR.\" (not yet supported — requires audio encoder)"
                );
                let sys = system.as_deref().unwrap();
                let messages = vec![
                    wick::tokenizer::ChatMessage {
                        role: "system".into(),
                        content: sys.into(),
                    },
                    wick::tokenizer::ChatMessage {
                        role: "user".into(),
                        content: prompt.clone(),
                    },
                ];
                let formatted = wick::tokenizer::apply_chat_template(&tokenizer, &messages, true)?;
                eprintln!("Chat template applied ({} chars)", formatted.len());
                tokens = tokenizer.encode(&formatted);
            } else if let Some(ids) = &token_ids {
                tokens = ids
                    .split(',')
                    .map(|s| s.trim().parse::<u32>())
                    .collect::<Result<Vec<_>, _>>()?;
            } else {
                if add_bos {
                    if let Some(bos) = tokenizer.bos_token() {
                        tokens.push(bos);
                    }
                }
                tokens.extend_from_slice(&tokenizer.encode(&prompt));
            }

            eprintln!(
                "Model: {} | {} layers | hidden={}",
                loaded_model.config().architecture,
                loaded_model.config().n_layers,
                loaded_model.config().hidden_size
            );
            eprintln!("Prompt tokens: {}", tokens.len());

            if let Some(vocoder_path) = &vocoder {
                // Audio generation mode.
                let voc_gguf = wick::gguf::GgufFile::open(Path::new(vocoder_path))?;
                let decoder_weights =
                    wick::model::audio_decoder::AudioDecoderWeights::from_gguf(&voc_gguf)?;
                let detok_weights =
                    wick::model::audio_decoder::DetokenizerWeights::from_gguf(&voc_gguf)?;

                #[cfg(all(feature = "metal", target_os = "macos"))]
                let gpu_detok = {
                    match wick::model::metal_audio_decoder::MetalAudioDecoder::from_gguf(
                        &voc_gguf,
                        Path::new(vocoder_path),
                    ) {
                        Ok(d) => {
                            eprintln!("Metal detokenizer loaded");
                            Some(d)
                        }
                        Err(e) => {
                            eprintln!("Metal detokenizer failed: {e}, using CPU");
                            None
                        }
                    }
                };
                #[cfg(not(all(feature = "metal", target_os = "macos")))]
                let _gpu_detok: Option<()> = None;

                let mut all_pcm = Vec::new();
                let sys = system.as_deref().unwrap();
                let mode = if sys == "Respond with interleaved text and audio." {
                    wick::audio_engine::AudioMode::Interleaved
                } else {
                    wick::audio_engine::AudioMode::Sequential
                };
                let audio_config = wick::audio_engine::AudioGenerateConfig {
                    max_tokens,
                    sampler: wick::sampler::SamplerConfig {
                        temperature,
                        ..Default::default()
                    },
                    audio_temperature,
                    audio_top_k,
                    mode,
                    gpu_depthformer: std::env::var("WICK_GPU_DF").as_deref() == Ok("1"),
                };

                #[cfg(all(feature = "metal", target_os = "macos"))]
                let gpu_ref: Option<&dyn wick::model::audio_decoder::AudioGpu> = gpu_detok
                    .as_ref()
                    .map(|d| d as &dyn wick::model::audio_decoder::AudioGpu);
                #[cfg(not(all(feature = "metal", target_os = "macos")))]
                let gpu_ref: Option<&dyn wick::model::audio_decoder::AudioGpu> = None;

                let result = wick::audio_engine::generate_audio(
                    loaded_model.as_ref(),
                    &decoder_weights,
                    &detok_weights,
                    &tokenizer,
                    &tokens,
                    &audio_config,
                    gpu_ref,
                    |text| {
                        print!("{text}");
                    },
                    |pcm, _sr| {
                        all_pcm.extend_from_slice(pcm);
                    },
                )?;

                eprintln!();
                eprintln!("---");
                eprintln!("Text tokens: {}", result.text_tokens);
                eprintln!("Audio frames: {}", result.audio_frames);
                eprintln!(
                    "Audio: {} samples ({:.1}s at 24kHz)",
                    all_pcm.len(),
                    all_pcm.len() as f64 / 24000.0
                );
                eprintln!("Elapsed: {:.1}s", result.elapsed_secs);
                eprintln!(
                    "Breakdown: depthformer {}ms ({:.1}ms/frame), detokenizer {}ms ({:.1}ms/frame), other {}ms",
                    (result.depthformer_secs * 1000.0) as u64,
                    if result.audio_frames > 0 {
                        result.depthformer_secs * 1000.0 / result.audio_frames as f64 * 1000.0
                    } else {
                        0.0
                    },
                    (result.detokenizer_secs * 1000.0) as u64,
                    if result.audio_frames > 0 {
                        result.detokenizer_secs * 1000.0 / result.audio_frames as f64 * 1000.0
                    } else {
                        0.0
                    },
                    (result.elapsed_secs * 1000.0
                        - result.depthformer_secs * 1000.0
                        - result.detokenizer_secs * 1000.0) as u64,
                );
                eprintln!(
                    "Throughput: {:.1} tok/s (text+audio)",
                    (result.text_tokens + result.audio_frames * 12) as f64 / result.elapsed_secs
                );

                if let Some(wav_path) = &audio_out {
                    write_wav(wav_path, &all_pcm, 24000)?;
                    eprintln!("Wrote {wav_path}");
                } else if !all_pcm.is_empty() {
                    let default_path = "/tmp/wick_audio.wav";
                    write_wav(default_path, &all_pcm, 24000)?;
                    eprintln!("Wrote {default_path}");
                }
            } else {
                // Text-only generation.
                let config = wick::engine::GenerateConfig {
                    max_tokens,
                    sampler: wick::sampler::SamplerConfig {
                        temperature,
                        ..Default::default()
                    },
                    silent: false,
                    kv_compression,
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
            prompt_tokens,
            runs,
            warmup,
            max_tokens,
            device,
            context_size,
            no_cache,
            kv_cache_keys,
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
            let loaded_model = load_model_for_device(Path::new(&model), &device, context_size)?;
            let kv_compression = setup_kv_compression(loaded_model.as_ref(), &kv_cache_keys)?;

            if no_cache {
                loaded_model.configure_cache(wick::kv_cache::KvCacheConfig {
                    cache_dir: None,
                    max_warm_entries: 0,
                    max_warm_bytes: 0,
                    max_cold_bytes: 0,
                });
            }

            let mut tokens = Vec::new();
            if let Some(n) = prompt_tokens {
                // Generate N tokens by sampling from the vocabulary, skipping special tokens.
                let vocab_size = tokenizer.vocab_size() as u32;
                let mut tid = 100; // Start after typical special token range
                while tokens.len() < n {
                    if !tokenizer.is_special_token(tid % vocab_size) {
                        tokens.push(tid % vocab_size);
                    }
                    tid += 1;
                    if tid > vocab_size * 2 + n as u32 {
                        break; // Safety break
                    }
                }
                if tokens.len() != n {
                    return Err(anyhow::anyhow!(
                        "tokenizer only provides {} usable prompt token(s), but {} were requested",
                        tokens.len(),
                        n
                    ));
                }
            } else {
                if add_bos {
                    if let Some(bos) = tokenizer.bos_token() {
                        tokens.push(bos);
                    }
                }
                tokens.extend_from_slice(&tokenizer.encode(&prompt));
            }

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
                kv_compression,
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
