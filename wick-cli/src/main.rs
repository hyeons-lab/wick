use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;
use clap::{Parser, Subcommand};
use wick::tokenizer::BpeTokenizer;
use wick::{BackendPreference, EngineConfig, FinishReason, ModalitySink, WickEngine};

/// Decode tokens to stdout as they stream. Used by the `run` command.
///
/// Holds a clone of the session's cancel flag so we can trigger a clean
/// shutdown when stdout closes (e.g. `wick run ... | head` detaches). Using
/// `print!` there would panic on `BrokenPipe`; we write manually and flip
/// cancel on any write error, letting the decode loop exit gracefully.
struct StdoutSink<'a> {
    tokenizer: &'a BpeTokenizer,
    cancel: Arc<AtomicBool>,
}

impl<'a> StdoutSink<'a> {
    fn new(tokenizer: &'a BpeTokenizer, cancel: Arc<AtomicBool>) -> Self {
        Self { tokenizer, cancel }
    }
}

impl ModalitySink for StdoutSink<'_> {
    fn on_text_tokens(&mut self, tokens: &[u32]) {
        if self.cancel.load(Ordering::Relaxed) {
            return;
        }
        let piece = self.tokenizer.decode(tokens);
        let mut out = std::io::stdout().lock();
        if out.write_all(piece.as_bytes()).is_err() || out.flush().is_err() {
            // Downstream closed (BrokenPipe). Signal the session to stop
            // decoding on the next iteration.
            self.cancel.store(true, Ordering::Relaxed);
        }
    }
    fn on_done(&mut self, _reason: FinishReason) {}
}

/// Swallows every event. Used by `bench` to avoid stdout inside the timed loop.
struct NoopSink;

impl ModalitySink for NoopSink {
    fn on_done(&mut self, _reason: FinishReason) {}
}

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
        /// Path to the model: a `.gguf` file, a `.json` LeapBundles manifest,
        /// or a directory containing exactly one `.json` manifest.
        #[arg(short, long)]
        model: String,

        /// The prompt to generate from. Required for text mode; optional
        /// when `--audio-in` is set (in which case it becomes a leading
        /// text instruction before the audio).
        #[arg(short, long)]
        prompt: Option<String>,

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

        /// Path to a mono PCM16 WAV file at 16 kHz to feed as audio
        /// input. Encoded via the bundle's mmproj
        /// (`AudioEncoderWeights`) and prefilled into the LLM as soft
        /// tokens via `Session::append_audio`.
        ///
        /// Combinations:
        /// - With `--prompt` alone: a leading text prefix (after BOS)
        ///   then audio. Plain mode, no chat template.
        /// - With `--system` (and optionally `--prompt`): renders the
        ///   model's chat template and inserts audio at the end of
        ///   the user turn (e.g. `--system "Perform ASR."` for
        ///   transcription). Requires the tokenizer to expose a
        ///   reserved special token (`<|reserved_4|>` or similar) to
        ///   mark the audio insertion point in the rendered prompt.
        ///
        /// Mutually exclusive with `--vocoder` (audio output),
        /// `--audio-out` (output WAV writer — not produced in
        /// audio-in mode), and `--token-ids` (audio-in builds its
        /// own KV from the encoder, not raw tokens).
        #[arg(long, conflicts_with_all = ["vocoder", "audio_out", "token_ids"])]
        audio_in: Option<String>,

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

        /// Prefill chunk size (ubatch). Long prompts split into chunks of
        /// this many tokens so `cancel()` can interrupt within one chunk.
        /// Default 512 matches the Phase 1.4 sweep on LFM2; values under
        /// 256 lose ≥5% prefill throughput. 0 disables chunking (monolithic).
        #[arg(long, default_value_t = 512)]
        ubatch_size: u32,

        /// Tokens to pin at the front when the KV window fills up. 0
        /// disables shifting — overflow returns ContextOverflow. A
        /// positive value lets the session drop a middle range to make
        /// room, but ONLY if (a) the pinned prefix plus incoming tokens
        /// leave real space in the context window (so setting
        /// `--n-keep` >= `--context-size` is a no-op) and (b) the prompt
        /// itself fits (overflow still occurs if the raw prompt is
        /// larger than the window). Not supported with any TurboQuant
        /// KV-cache mode (`tq3`, `tq3-keys`, `tq3-values`) — shifting
        /// compressed caches lands in a follow-up.
        #[arg(long, default_value_t = 0)]
        n_keep: u32,
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
        /// Path to the model (`.gguf`, `.json` manifest, or manifest dir).
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

        /// Prefill chunk size (ubatch). Lower = more cancel-responsive
        /// but slower prefill. 0 disables chunking (monolithic).
        #[arg(long, default_value_t = 512)]
        ubatch_size: u32,
    },
}

/// Load a `WickEngine` from a path that may be a bare `.gguf`, a `.json`
/// manifest, or a directory containing one `.json` manifest. The engine
/// owns the model + tokenizer for the CLI's lifetime; callers get
/// `engine.new_session(...)` for text and `engine.model()` / `engine.tokenizer()`
/// handles for the audio pipeline.
fn load_engine(path: &Path, device: &str, context_size: usize) -> Result<WickEngine> {
    let backend = BackendPreference::parse_str(device).map_err(|e| anyhow::anyhow!("{e}"))?;
    // `..Default::default()` picks up optional fields (e.g. `bundle_repo`
    // under the `remote` feature, which shows up whenever a workspace
    // member pulls it transitively). Without the spread, enabling remote
    // anywhere in the workspace breaks this construction.
    let engine = WickEngine::from_path(
        path,
        EngineConfig {
            context_size,
            backend,
            ..Default::default()
        },
    )?;
    eprintln!(
        "Using {} backend ({})",
        match backend {
            BackendPreference::Auto => "auto-selected",
            BackendPreference::Cpu => "CPU",
            BackendPreference::Gpu => "wgpu",
            BackendPreference::Metal => "native Metal",
        },
        engine.metadata().architecture,
    );
    Ok(engine)
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

/// Read a WAV file and return (samples_f32_in_minus1_to_1,
/// sample_rate). Strict: accepts only mono PCM16 (audio_format = 1,
/// channels = 1, bits_per_sample = 16). Errors on anything else
/// rather than silently down-mixing or converting — the caller
/// (today: `--audio-in`) must produce a clean WAV.
///
/// Skips unknown subchunks (LIST, JUNK, etc.) between fmt and data
/// per the RIFF spec.
fn read_wav_pcm16_mono(path: &str) -> Result<(Vec<f32>, u32)> {
    use anyhow::{Context, anyhow, bail};
    use std::io::Read;
    let mut f = std::fs::File::open(path).with_context(|| format!("opening WAV `{path}`"))?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)
        .with_context(|| format!("reading WAV `{path}`"))?;

    let read_u16 = |o: usize| -> Result<u16> {
        let end = o.checked_add(2).ok_or_else(|| anyhow!("offset overflow"))?;
        if end > buf.len() {
            bail!(
                "WAV `{path}` truncated: needed u16 at offset {o}, file is {} bytes",
                buf.len()
            );
        }
        Ok(u16::from_le_bytes(buf[o..end].try_into().unwrap()))
    };
    let read_u32 = |o: usize| -> Result<u32> {
        let end = o.checked_add(4).ok_or_else(|| anyhow!("offset overflow"))?;
        if end > buf.len() {
            bail!(
                "WAV `{path}` truncated: needed u32 at offset {o}, file is {} bytes",
                buf.len()
            );
        }
        Ok(u32::from_le_bytes(buf[o..end].try_into().unwrap()))
    };

    if buf.len() < 12 || &buf[0..4] != b"RIFF" || &buf[8..12] != b"WAVE" {
        bail!("WAV `{path}`: missing RIFF/WAVE header");
    }

    // Walk subchunks starting at offset 12. Find "fmt " then "data".
    let mut o = 12usize;
    let mut fmt_off: Option<usize> = None;
    let mut data: Option<(usize, usize)> = None;
    while o + 8 <= buf.len() {
        let id = &buf[o..o + 4];
        let sz = read_u32(o + 4)? as usize;
        let body = o + 8;
        if id == b"fmt " {
            fmt_off = Some(body);
        } else if id == b"data" {
            if body + sz > buf.len() {
                bail!(
                    "WAV `{path}`: data chunk size {sz} exceeds file (body+sz={} > len={})",
                    body + sz,
                    buf.len()
                );
            }
            data = Some((body, sz));
        }
        // Subchunks are word-aligned: pad odd sizes by 1.
        o = body + sz + (sz & 1);
    }
    let fmt = fmt_off.ok_or_else(|| anyhow!("WAV `{path}`: no `fmt ` chunk"))?;
    let (data_off, data_sz) = data.ok_or_else(|| anyhow!("WAV `{path}`: no `data` chunk"))?;

    let audio_format = read_u16(fmt)?;
    let channels = read_u16(fmt + 2)?;
    let sample_rate = read_u32(fmt + 4)?;
    let bits = read_u16(fmt + 14)?;
    if audio_format != 1 {
        bail!(
            "WAV `{path}`: audio_format {audio_format} (expected 1=PCM). Re-encode as 16-bit PCM."
        );
    }
    if channels != 1 {
        bail!(
            "WAV `{path}`: {channels} channels (expected 1=mono). Down-mix externally before passing in."
        );
    }
    if bits != 16 {
        bail!("WAV `{path}`: {bits} bits/sample (expected 16). Re-encode as 16-bit PCM.");
    }

    if data_sz % 2 != 0 {
        bail!("WAV `{path}`: data chunk size {data_sz} is not a multiple of 2 (PCM16 frame size)");
    }
    let n_samples = data_sz / 2;
    let mut samples = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let lo = buf[data_off + i * 2];
        let hi = buf[data_off + i * 2 + 1];
        let s = i16::from_le_bytes([lo, hi]);
        // Symmetric scale: i16::MIN -> -1.0, i16::MAX -> ~1.0. Using 32768
        // (vs 32767) keeps zero exactly at zero and avoids the asymmetric
        // off-by-one when round-tripping through `write_wav` (which clamps
        // before scaling by 32767).
        samples.push(s as f32 / 32768.0);
    }
    Ok((samples, sample_rate))
}

/// Pick a vocab-resident special token to use as the audio
/// insertion marker in chat-template renders. The token must
/// (1) tokenize as a single ID regardless of context (i.e. be a
/// real special token, not a unicode placeholder), and (2) never
/// appear in real user content so we don't false-match it.
///
/// Tries the LFM2-family reserved slots in order. Returns
/// `(token_id, token_name_for_template_substitution)`. Errors
/// when none are present — without a marker we can't split the
/// rendered token stream deterministically; the caller should
/// drop `--system` to fall back to plain audio-in mode.
fn pick_audio_marker_token(tok: &BpeTokenizer) -> Result<(u32, &'static str)> {
    for name in [
        "<|reserved_4|>",
        "<|reserved_5|>",
        "<|reserved_6|>",
        "<|reserved_7|>",
    ] {
        if let Some(id) = tok.special_token_id(name) {
            return Ok((id, name));
        }
    }
    anyhow::bail!(
        "no suitable special token for audio chat-template insertion. \
         Tried <|reserved_4|>..<|reserved_7|>. Drop --system to use \
         plain audio-in mode (text prefix → audio, no template)."
    );
}

/// Find the unique index of `marker_id` in `tokens`. Returns the
/// zero-based position; the caller slices `tokens[..idx]` and
/// `tokens[idx + 1..]` for the prefix and suffix around the marker.
/// Single-pass scan — no allocation.
///
/// Errors include `marker_name` (the human-readable special-token
/// string, not just the numeric id) so users hitting these cases
/// know which literal to inspect / remove from `--prompt` or
/// `--system`. The two failures:
///
/// - **Marker not found**: chat template stripped the placeholder
///   (e.g. an aggressive escape filter), OR rendering mismatched
///   the user content. Caller should drop `--system` to fall back
///   to plain audio-in.
/// - **Marker appears more than once**: user-supplied
///   `--prompt`/`--system` text contains a literal occurrence of
///   the marker token name, making the insertion point ambiguous.
///   Caller should remove that literal.
fn split_at_marker(tokens: &[u32], marker_id: u32, marker_name: &str) -> Result<usize> {
    let mut found: Option<usize> = None;
    let mut count: usize = 0;
    for (i, &t) in tokens.iter().enumerate() {
        if t == marker_id {
            count += 1;
            if found.is_none() {
                found = Some(i);
            }
        }
    }
    match (count, found) {
        (1, Some(idx)) => Ok(idx),
        (0, _) => anyhow::bail!(
            "audio marker token `{marker_name}` (id {marker_id}) not found in rendered \
             chat-template tokens — the template may have stripped or escaped the \
             placeholder. Drop `--system` to fall back to plain audio-in."
        ),
        (n, _) => anyhow::bail!(
            "audio marker token `{marker_name}` (id {marker_id}) appears {n} times in \
             rendered tokens; expected exactly one insertion point. Check that \
             `--prompt` and `--system` text don't contain literal `{marker_name}`."
        ),
    }
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
            audio_in,
            audio_out,
            system,
            audio_temperature,
            audio_top_k,
            cache_dir,
            cache_warm_mb,
            cache_disk_gb,
            no_cache,
            kv_cache_keys,
            ubatch_size,
            n_keep,
        } => {
            let engine = load_engine(Path::new(&model), &device, context_size)?;
            let tokenizer = engine.tokenizer();
            let add_bos = engine.metadata().add_bos_token;

            let kv_compression = setup_kv_compression(engine.model(), &kv_cache_keys)?;

            // Configure KV prefix cache.
            if no_cache {
                engine.configure_cache(wick::kv_cache::KvCacheConfig {
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
                engine.configure_cache(wick::kv_cache::KvCacheConfig {
                    cache_dir: dir,
                    max_warm_entries: 32,
                    max_warm_bytes: cache_warm_mb * 1024 * 1024,
                    max_cold_bytes: cache_disk_gb * 1024 * 1024 * 1024,
                });
            }

            // Audio-input path (mutually exclusive with --vocoder via clap
            // `conflicts_with`). Skips the chat-template / token-building
            // dance entirely — the audio is fed in as soft tokens via
            // `Session::append_audio`, which uses the engine's
            // auto-attached `AudioEncoderWeights` (PR #106).
            if let Some(wav_path) = &audio_in {
                let (pcm, sr) = read_wav_pcm16_mono(wav_path)?;
                eprintln!(
                    "Loaded {} samples ({:.2}s @ {sr} Hz) from {wav_path}",
                    pcm.len(),
                    pcm.len() as f32 / sr as f32
                );
                eprintln!(
                    "Model: {} | {} layers | hidden={}",
                    engine.model().config().architecture,
                    engine.model().config().n_layers,
                    engine.model().config().hidden_size
                );

                let mut session = engine.new_session(wick::SessionConfig {
                    kv_compression,
                    seed: None,
                    ubatch_size,
                    n_keep,
                    ..Default::default()
                });

                let prefill_start = std::time::Instant::now();
                if let Some(sys) = &system {
                    // Chat-template flow: render the model's template
                    // with a placeholder marker where audio should land
                    // (end of user content). Split the rendered tokens
                    // at the marker and feed prefix → audio → suffix so
                    // audio sits inside the user turn, before <im_end>.
                    // The template's `{{ bos_token }}` already adds
                    // BOS — don't add it again.
                    let (marker_id, marker_name) = pick_audio_marker_token(tokenizer)?;
                    let user_text = prompt.as_deref().unwrap_or("");
                    let user_content = format!("{user_text}{marker_name}");
                    let messages = vec![
                        wick::tokenizer::ChatMessage {
                            role: "system".into(),
                            content: sys.clone(),
                        },
                        wick::tokenizer::ChatMessage {
                            role: "user".into(),
                            content: user_content,
                        },
                    ];
                    let formatted =
                        wick::tokenizer::apply_chat_template(tokenizer, &messages, true)?;
                    let toks = tokenizer.encode(&formatted);
                    let split = split_at_marker(&toks, marker_id, marker_name)?;
                    let (prefix, suffix) = (&toks[..split], &toks[split + 1..]);
                    eprintln!(
                        "Chat template: {} prefix tokens, audio, {} suffix tokens",
                        prefix.len(),
                        suffix.len()
                    );
                    if !prefix.is_empty() {
                        session.append_tokens(prefix)?;
                    }
                    session.append_audio(&pcm, sr)?;
                    if !suffix.is_empty() {
                        session.append_tokens(suffix)?;
                    }
                } else {
                    // Plain audio-in: BOS (when the model wants it),
                    // optional --prompt as a leading text prefix, then
                    // audio. No template, no system role.
                    if add_bos {
                        if let Some(bos) = tokenizer.bos_token() {
                            session.append_tokens(&[bos])?;
                        }
                    }
                    if let Some(p) = &prompt {
                        if !p.is_empty() {
                            session.append_text(p)?;
                        }
                    }
                    session.append_audio(&pcm, sr)?;
                }
                let prefill_elapsed = prefill_start.elapsed();
                // Snapshot KV size before generate(): that call advances
                // `position` by every emitted token, so reading it after
                // would overreport the prefill frame count.
                let kv_after_prefill = session.position();

                let opts = wick::GenerateOpts {
                    max_tokens: max_tokens as u32,
                    temperature,
                    ..Default::default()
                };

                let mut sink = StdoutSink::new(tokenizer, session.cancel_handle());
                let summary = session.generate(&opts, &mut sink)?;

                let decode_tps = if summary.decode_ms > 0 {
                    summary.tokens_generated as f64 / (summary.decode_ms as f64 / 1000.0)
                } else {
                    0.0
                };

                eprintln!();
                eprintln!("---");
                eprintln!("Frames in KV after prefill: {kv_after_prefill}");
                eprintln!("Generated tokens: {}", summary.tokens_generated);
                eprintln!(
                    "Prefill (encode + LLM): {:.2}s",
                    prefill_elapsed.as_secs_f64()
                );
                eprintln!("Decode: {:.1} tok/s", decode_tps);
                return Ok(());
            }

            // Text mode (audio-in path returned above). `--prompt` is
            // required here — the audio-in branch is the only context
            // where it's optional. Treat absence as an explicit usage
            // error rather than silently prefilling an empty/BOS-only
            // prefix that would surprise the caller.
            let prompt = match &prompt {
                Some(p) => p,
                None => anyhow::bail!(
                    "--prompt is required for text mode. Use `wick run --prompt <text> ...` \
                     or pass `--audio-in <wav>` for the audio-input path."
                ),
            };

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
                let formatted = wick::tokenizer::apply_chat_template(tokenizer, &messages, true)?;
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
                tokens.extend_from_slice(&tokenizer.encode(prompt));
            }

            eprintln!(
                "Model: {} | {} layers | hidden={}",
                engine.model().config().architecture,
                engine.model().config().n_layers,
                engine.model().config().hidden_size
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
                    engine.model(),
                    &decoder_weights,
                    &detok_weights,
                    tokenizer,
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
                // Text-only generation via Session + StdoutSink.
                let mut session = engine.new_session(wick::SessionConfig {
                    kv_compression,
                    seed: None,
                    ubatch_size,
                    n_keep,
                    ..Default::default()
                });

                let prefill_start = std::time::Instant::now();
                session.append_tokens(&tokens)?;
                let prefill_elapsed = prefill_start.elapsed();

                let opts = wick::GenerateOpts {
                    max_tokens: max_tokens as u32,
                    temperature,
                    ..Default::default()
                };

                let mut sink = StdoutSink::new(tokenizer, session.cancel_handle());
                let summary = session.generate(&opts, &mut sink)?;

                let prefill_tps = if prefill_elapsed.as_secs_f64() > 0.0 {
                    tokens.len() as f64 / prefill_elapsed.as_secs_f64()
                } else {
                    0.0
                };
                let decode_tps = if summary.decode_ms > 0 {
                    summary.tokens_generated as f64 / (summary.decode_ms as f64 / 1000.0)
                } else {
                    0.0
                };

                eprintln!();
                eprintln!("---");
                eprintln!("Prompt tokens: {}", tokens.len());
                eprintln!("Generated tokens: {}", summary.tokens_generated);
                eprintln!("Prefill: {:.1} tok/s", prefill_tps);
                eprintln!("Decode: {:.1} tok/s", decode_tps);
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
            ubatch_size,
        } => {
            anyhow::ensure!(runs >= 1, "--runs must be >= 1");
            if std::env::var("WICK_PROFILE").is_ok() {
                eprintln!(
                    "warning: WICK_PROFILE is set — bench numbers will be inflated by profile overhead"
                );
            }

            let engine = load_engine(Path::new(&model), &device, context_size)?;
            let tokenizer = engine.tokenizer();
            let add_bos = engine.metadata().add_bos_token;
            let kv_compression = setup_kv_compression(engine.model(), &kv_cache_keys)?;

            if no_cache {
                engine.configure_cache(wick::kv_cache::KvCacheConfig {
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
                engine.model().config().architecture,
                engine.model().config().n_layers,
                engine.model().config().hidden_size
            );
            eprintln!(
                "Prompt tokens: {} | max_tokens: {} | warmup: {} | runs: {}",
                tokens.len(),
                max_tokens,
                warmup,
                runs
            );

            // Greedy (temp=0): deterministic, bench-friendly. NoopSink swallows tokens.
            let run_once = || -> Result<(f64, f64)> {
                let mut session = engine.new_session(wick::SessionConfig {
                    kv_compression: kv_compression.clone(),
                    seed: None,
                    ubatch_size,
                    ..Default::default()
                });
                let prefill_start = std::time::Instant::now();
                session.append_tokens(&tokens)?;
                let prefill_elapsed = prefill_start.elapsed();

                let opts = wick::GenerateOpts {
                    max_tokens: max_tokens as u32,
                    temperature: 0.0,
                    ..Default::default()
                };
                let mut sink = NoopSink;
                let summary = session.generate(&opts, &mut sink)?;

                let prefill_tps = if prefill_elapsed.as_secs_f64() > 0.0 {
                    tokens.len() as f64 / prefill_elapsed.as_secs_f64()
                } else {
                    0.0
                };
                let decode_tps = if summary.decode_ms > 0 {
                    summary.tokens_generated as f64 / (summary.decode_ms as f64 / 1000.0)
                } else {
                    0.0
                };
                Ok((prefill_tps, decode_tps))
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

#[cfg(test)]
mod tests {
    use super::{Cli, read_wav_pcm16_mono, split_at_marker, write_wav};
    use clap::Parser;

    /// Round-trip: deterministic samples → write_wav → read_wav_pcm16_mono.
    /// The reader must recover the same sample count + sample rate, with
    /// values within one quantization step of the originals.
    #[test]
    fn write_then_read_wav_roundtrips() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rt.wav");
        let path_str = path.to_str().unwrap();

        let sr = 16_000u32;
        // 100 samples of a small triangle wave well inside [-1, 1].
        let samples: Vec<f32> = (0..100)
            .map(|i| ((i as f32) / 50.0 - 1.0).clamp(-0.9, 0.9))
            .collect();

        write_wav(path_str, &samples, sr).unwrap();
        let (back, back_sr) = read_wav_pcm16_mono(path_str).unwrap();

        assert_eq!(back_sr, sr, "sample rate must round-trip");
        assert_eq!(back.len(), samples.len(), "sample count must round-trip");
        // 16-bit PCM has ≈ 1/32768 quantization step; allow 2× that
        // for safety (write_wav uses 32767 scale, read uses 32768).
        let eps = 2.0 / 32768.0;
        for (i, (a, b)) in samples.iter().zip(back.iter()).enumerate() {
            assert!(
                (a - b).abs() < eps,
                "sample {i}: orig={a} read={b} diff={}",
                (a - b).abs()
            );
        }
    }

    /// `--audio-in` must parse cleanly without `--prompt`. This is the
    /// advertised audio-only entry point — clap-rejecting it would
    /// make the documented `wick run --audio-in <wav>` form unusable.
    #[test]
    fn audio_in_parses_without_prompt() {
        let r = Cli::try_parse_from([
            "wick",
            "run",
            "--model",
            "/tmp/x",
            "--audio-in",
            "/tmp/y.wav",
        ]);
        assert!(
            r.is_ok(),
            "expected --audio-in alone to parse, got: {:?}",
            r.err()
        );
    }

    /// `--audio-in` is mutually exclusive with several other flags so
    /// callers don't silently get wrong behavior. Verify each
    /// individual conflict surfaces from clap rather than reaching the
    /// dispatch. `--system` is NOT in this list — it's intentionally
    /// allowed alongside `--audio-in` so the chat-template flow can
    /// wrap audio in a system+user turn.
    #[test]
    fn audio_in_conflicts_are_enforced_by_clap() {
        for (label, extra) in [
            ("--vocoder", vec!["--vocoder", "/tmp/v.gguf"]),
            ("--audio-out", vec!["--audio-out", "/tmp/o.wav"]),
            ("--token-ids", vec!["--token-ids", "1,2,3"]),
        ] {
            let mut argv = vec![
                "wick",
                "run",
                "--model",
                "/tmp/x",
                "--audio-in",
                "/tmp/y.wav",
            ];
            argv.extend_from_slice(&extra);
            let r = Cli::try_parse_from(&argv);
            assert!(
                r.is_err(),
                "expected --audio-in + {label} to be rejected by clap, but parsing succeeded"
            );
        }
    }

    /// `--audio-in` + `--system` (and optionally `--prompt`) must
    /// parse cleanly — the chat-template flow runs at dispatch time.
    /// Negative case (missing audio marker token) lives in the
    /// `split_at_marker` test below; this one just guards the clap
    /// surface.
    #[test]
    fn audio_in_plus_system_parses() {
        let r = Cli::try_parse_from([
            "wick",
            "run",
            "--model",
            "/tmp/x",
            "--audio-in",
            "/tmp/y.wav",
            "--system",
            "Perform ASR.",
            "--prompt",
            "What did the speaker say?",
        ]);
        assert!(
            r.is_ok(),
            "expected --audio-in + --system + --prompt to parse, got: {:?}",
            r.err()
        );
    }

    /// `split_at_marker` happy path: a marker in the middle of a
    /// token list returns the unique index, and the caller can
    /// slice prefix/suffix around it.
    #[test]
    fn split_at_marker_normal_case() {
        let toks = [10u32, 20, 99, 30, 40];
        let idx = split_at_marker(&toks, 99, "<|reserved_4|>").unwrap();
        assert_eq!(idx, 2);
        assert_eq!(&toks[..idx], &[10, 20]);
        assert_eq!(&toks[idx + 1..], &[30, 40]);
    }

    /// Marker at position 0 → empty prefix slice.
    #[test]
    fn split_at_marker_at_start() {
        let toks = [99u32, 1, 2, 3];
        let idx = split_at_marker(&toks, 99, "<|reserved_4|>").unwrap();
        assert_eq!(idx, 0);
        assert!(toks[..idx].is_empty());
        assert_eq!(&toks[idx + 1..], &[1, 2, 3]);
    }

    /// Marker at the last position → empty suffix slice.
    #[test]
    fn split_at_marker_at_end() {
        let toks = [1u32, 2, 3, 99];
        let idx = split_at_marker(&toks, 99, "<|reserved_4|>").unwrap();
        assert_eq!(idx, 3);
        assert_eq!(&toks[..idx], &[1, 2, 3]);
        assert!(toks[idx + 1..].is_empty());
    }

    /// Missing marker → typed error naming the marker by string
    /// AND id, so users know what literal to inspect (not just
    /// "id 99 missing"). Plus a hint to drop `--system`.
    #[test]
    fn split_at_marker_missing_errors() {
        let toks = [10u32, 20, 30];
        let err = split_at_marker(&toks, 99, "<|reserved_4|>").unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("<|reserved_4|>") && msg.contains("99") && msg.contains("not found"),
            "error should name `<|reserved_4|>`, id 99, and 'not found': {msg}"
        );
        assert!(
            msg.contains("--system"),
            "error should hint at dropping --system: {msg}"
        );
    }

    /// Marker appearing more than once → typed error pointing at
    /// the user-supplied text as the likely culprit. Both the
    /// occurrence count and the marker name appear in the message.
    #[test]
    fn split_at_marker_duplicate_errors() {
        let toks = [10u32, 99, 20, 99, 30];
        let err = split_at_marker(&toks, 99, "<|reserved_4|>").unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("<|reserved_4|>") && msg.contains("2 times"),
            "error should name `<|reserved_4|>` and report '2 times': {msg}"
        );
        assert!(
            msg.contains("--prompt") && msg.contains("--system"),
            "error should point at user-supplied text as the source: {msg}"
        );
    }

    /// Text mode without `--audio-in` and without `--prompt` is the
    /// negative case for the new optional `prompt`. Clap accepts the
    /// args (since prompt is now `Option<String>`); the runtime check
    /// in the dispatcher should bail with a clear "--prompt is
    /// required" message — but that path isn't exercised here (it
    /// runs after engine load). What we CAN check in unit tests is
    /// that clap parses both forms, leaving runtime validation as
    /// the single source of truth.
    #[test]
    fn run_without_prompt_or_audio_in_parses() {
        // Clap parses; the dispatcher will reject at runtime with the
        // usage error message. We don't try to construct an Engine here.
        let r = Cli::try_parse_from(["wick", "run", "--model", "/tmp/x"]);
        assert!(
            r.is_ok(),
            "expected bare `run --model` to parse, got: {:?}",
            r.err()
        );
    }

    /// Multi-channel WAV must be rejected with a typed error mentioning
    /// the channel count.
    #[test]
    fn read_wav_rejects_stereo() {
        // Hand-craft a minimal stereo WAV header (44 bytes + 0 data).
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&36u32.to_le_bytes()); // file size - 8
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes()); // fmt size
        buf.extend_from_slice(&1u16.to_le_bytes()); // audio_format = PCM
        buf.extend_from_slice(&2u16.to_le_bytes()); // 2 channels
        buf.extend_from_slice(&16_000u32.to_le_bytes());
        buf.extend_from_slice(&64_000u32.to_le_bytes()); // byte rate
        buf.extend_from_slice(&4u16.to_le_bytes()); // block align
        buf.extend_from_slice(&16u16.to_le_bytes()); // bits/sample
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&0u32.to_le_bytes()); // empty data

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("stereo.wav");
        std::fs::write(&path, &buf).unwrap();

        let err = read_wav_pcm16_mono(path.to_str().unwrap()).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("2 channels") && msg.contains("mono"),
            "error should mention channels=2 and mono requirement; got: {msg}"
        );
    }
}
