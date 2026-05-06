use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use clap::{Parser, Subcommand};
use wick::tokenizer::BpeTokenizer;
use wick::{BackendPreference, EngineConfig, FinishReason, ModalitySink, WickEngine};

mod chat_tui;

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

/// Streams decoded tokens to stdout *and* accumulates them into a
/// `String` buffer so the chat REPL can capture the full assistant
/// reply for the next turn's history. Otherwise mirrors `StdoutSink`'s
/// BrokenPipe-aware write behavior.
struct ChatSink<'a> {
    tokenizer: &'a BpeTokenizer,
    cancel: Arc<AtomicBool>,
    buffer: String,
}

impl<'a> ChatSink<'a> {
    fn new(tokenizer: &'a BpeTokenizer, cancel: Arc<AtomicBool>) -> Self {
        Self {
            tokenizer,
            cancel,
            buffer: String::new(),
        }
    }
    fn into_text(self) -> String {
        self.buffer
    }
}

impl ModalitySink for ChatSink<'_> {
    fn on_text_tokens(&mut self, tokens: &[u32]) {
        if self.cancel.load(Ordering::Relaxed) {
            return;
        }
        let piece = self.tokenizer.decode(tokens);
        let mut out = std::io::stdout().lock();
        if out.write_all(piece.as_bytes()).is_err() || out.flush().is_err() {
            self.cancel.store(true, Ordering::Relaxed);
            return;
        }
        self.buffer.push_str(&piece);
    }
    fn on_done(&mut self, _reason: FinishReason) {}
}

/// `BundleRepo` progress callback that renders a single-line live status to
/// stderr while bundle files stream in. Prevents a multi-MB cache-miss
/// download from looking like a hung process when `wick chat --bundle-id`
/// runs the first time.
///
/// Trait throttling is handled by the library (~1 call per 256 KB written +
/// once at end-of-stream) so this just formats and rewrites the line.
/// Multiple files in one resolve (manifest.json then the GGUF) are
/// distinguished by `url`; on a URL change we emit a newline first so the
/// previous file's final progress line stays visible.
#[derive(Debug, Default)]
struct CliDownloadProgress {
    last_url: Mutex<Option<String>>,
    /// `true` once `on_progress` has been called at least once. Used by
    /// the caller to decide whether to emit a trailing newline after the
    /// resolve — cache hits never call back, so an unconditional newline
    /// would leak a blank line on every silent resolve.
    printed_any: AtomicBool,
}

impl CliDownloadProgress {
    fn printed_any(&self) -> bool {
        self.printed_any.load(Ordering::Relaxed)
    }
}

impl wick::bundle::DownloadProgress for CliDownloadProgress {
    fn on_progress(&self, url: &str, bytes: u64, total: Option<u64>) {
        self.printed_any.store(true, Ordering::Relaxed);
        // URL transition → seal the prior line with a newline so it stays
        // legible after the next file's progress overwrites the position.
        {
            let mut guard = self
                .last_url
                .lock()
                .expect("CliDownloadProgress lock poisoned");
            let new_file = guard.as_deref() != Some(url);
            if new_file {
                if guard.is_some() {
                    eprintln!();
                }
                *guard = Some(url.to_string());
            }
        }

        let filename = url.rsplit('/').next().unwrap_or(url);
        let mb = bytes as f64 / (1024.0 * 1024.0);
        let line = match total {
            Some(t) if t > 0 => {
                let total_mb = t as f64 / (1024.0 * 1024.0);
                let pct = ((bytes * 100) / t).min(100);
                format!("\rDownloading {filename}: {pct:>3}% ({mb:>6.1} / {total_mb:.1} MiB)")
            }
            // No Content-Length — chunked-transfer or HEAD-less stream.
            // Show bytes downloaded only.
            _ => format!("\rDownloading {filename}: {mb:>6.1} MiB"),
        };
        let mut err = std::io::stderr().lock();
        let _ = err.write_all(line.as_bytes());
        let _ = err.flush();
    }
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
        /// Path to the model: a `.gguf` file, a `.json` LeapBundles
        /// manifest, or a directory containing exactly one `.json`
        /// manifest. Mutually exclusive with `--bundle-id` /
        /// `--quant`; one of the two source forms must be set.
        #[arg(
            short,
            long,
            conflicts_with_all = ["bundle_id", "quant"],
        )]
        model: Option<String>,

        /// LeapBundles bundle id (e.g. `LFM2.5-1.2B-Instruct` or
        /// `LFM2.5-1.2B-Instruct-GGUF` — `-GGUF` is appended
        /// automatically if missing). Pairs with `--quant` for
        /// auto-download from
        /// `huggingface.co/LiquidAI/LeapBundles`. Cached under
        /// `--cache-dir` (default `$HOME/.cache/wick`). Use
        /// `wick list-bundles` to discover available IDs.
        #[arg(long, requires = "quant")]
        bundle_id: Option<String>,

        /// Quantization label for `--bundle-id` (e.g. `Q4_0`,
        /// `Q8_0`). Pairs with `--bundle-id`: clap rejects
        /// either flag without the other.
        #[arg(long, requires = "bundle_id")]
        quant: Option<String>,

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

        /// Path to a PCM16 WAV file to feed as audio input. Any
        /// sample rate accepted — non-16 kHz inputs are resampled
        /// with linear interpolation before encoding. Multi-channel
        /// inputs are down-mixed to mono by averaging across
        /// channels. For studio-quality conversion, pre-resample
        /// and down-mix externally (sox/ffmpeg) and pass 16 kHz
        /// mono PCM16 to bypass both steps. Encoded via the
        /// bundle's mmproj (`AudioEncoderWeights`) and prefilled
        /// into the LLM as soft tokens via `Session::append_audio`.
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

        /// Cache root, shared between KV prefix cache files
        /// (under `<dir>/kv/`, enables disk caching for prompt
        /// reuse) AND the `BundleRepo` download store when
        /// `--bundle-id` is set (downloads under
        /// `<dir>/huggingface.co/...`). Default: `$HOME/.cache/wick`.
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

    /// Interactive multi-turn chat REPL.
    ///
    /// Reads user messages from stdin one line at a time, renders
    /// the model's chat template per turn, and streams the assistant
    /// reply to stdout. The Session is kept alive across turns so
    /// the engine's prefix cache accelerates each successive prefill.
    ///
    /// Slash commands (a line starting with `/` is interpreted as a
    /// command, not sent to the model):
    /// - `/help` — list available commands.
    /// - `/clear` — reset history; the system prompt (if any) is
    ///   preserved.
    /// - `/exit`, `/quit`, EOF (Ctrl+D) — leave the REPL.
    Chat {
        /// Path to the model: a `.gguf` file, a `.json` LeapBundles
        /// manifest, or a directory containing exactly one `.json`
        /// manifest. Mutually exclusive with `--bundle-id` /
        /// `--quant`; one of the two source forms must be set.
        #[arg(
            short,
            long,
            conflicts_with_all = ["bundle_id", "quant"],
        )]
        model: Option<String>,

        /// LeapBundles bundle id (e.g. `LFM2.5-1.2B-Instruct` or
        /// `LFM2.5-1.2B-Instruct-GGUF` — `-GGUF` is appended
        /// automatically if missing). Pairs with `--quant` for
        /// auto-download from
        /// `huggingface.co/LiquidAI/LeapBundles`. Cached under
        /// `--cache-dir` (default `$HOME/.cache/wick`). Use
        /// `wick list-bundles` to discover available IDs.
        #[arg(long, requires = "quant")]
        bundle_id: Option<String>,

        /// Quantization label for `--bundle-id` (e.g. `Q4_0`,
        /// `Q8_0`). Required when `--bundle-id` is set.
        #[arg(long, requires = "bundle_id")]
        quant: Option<String>,

        /// Cache root: shared between `--bundle-id` downloads
        /// (under `<dir>/huggingface.co/...`) and the KV prefix
        /// cache (under `<dir>/kv/`). Default: `$HOME/.cache/wick`.
        /// The disk-tier prefix cache survives process restarts —
        /// useful for mobile / FFI consumers that get killed and
        /// resumed; on next launch the historical conversation
        /// prefix rehydrates instead of re-prefilling cold.
        #[arg(long)]
        cache_dir: Option<String>,

        /// Max warm (in-memory) prefix-cache size in MB. Default 256.
        #[arg(long, default_value_t = 256)]
        cache_warm_mb: u64,

        /// Max cold (disk) prefix-cache size in GB. Default 10.
        /// Only consumed when `--cache-dir` (or default) is writable.
        #[arg(long, default_value_t = 10)]
        cache_disk_gb: u64,

        /// Disable the KV prefix cache entirely. Bundle downloads
        /// still use `--cache-dir` (this flag only gates the KV
        /// prefix cache, not the bundle store).
        #[arg(long)]
        no_cache: bool,

        /// Optional system prompt pinned at the head of the
        /// conversation. Carried through every turn unchanged.
        #[arg(long)]
        system: Option<String>,

        /// Device to use: cpu, gpu, metal, or auto.
        #[arg(long, default_value = "auto")]
        device: String,

        /// Max KV context window size. Default 4096.
        #[arg(long, default_value_t = 4096)]
        context_size: usize,

        /// Max tokens generated per assistant turn.
        #[arg(long, default_value_t = 512)]
        max_tokens: usize,

        /// Sampling temperature. `<= 0` selects greedy decoding
        /// (reproducible). Default 0 so chat output is deterministic
        /// without a seed.
        #[arg(long, default_value_t = 0.0)]
        temperature: f32,

        /// RNG seed for sampling. Only meaningful when
        /// `--temperature > 0`; ignored under greedy decoding.
        #[arg(long)]
        seed: Option<u64>,

        /// Disable the inline TUI even when stdin/stdout are TTYs.
        /// Falls back to the line-based REPL — useful for shell
        /// scripting, log capture, or terminals that don't speak
        /// the ratatui control sequences cleanly. Auto-detection
        /// also falls back to the line REPL when either stdin or
        /// stdout is redirected.
        #[arg(long)]
        no_tui: bool,
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
        /// Path to the model: a `.gguf` file, a `.json` LeapBundles
        /// manifest, or a directory containing exactly one `.json`
        /// manifest. Mutually exclusive with `--bundle-id` /
        /// `--quant`; one of the two source forms must be set.
        #[arg(
            short,
            long,
            conflicts_with_all = ["bundle_id", "quant"],
        )]
        model: Option<String>,

        /// LeapBundles bundle id (e.g. `LFM2.5-1.2B-Instruct` or
        /// `LFM2.5-1.2B-Instruct-GGUF` — `-GGUF` is appended
        /// automatically if missing). Pairs with `--quant` for
        /// auto-download from
        /// `huggingface.co/LiquidAI/LeapBundles`. Cached under
        /// `--cache-dir` (default `$HOME/.cache/wick`). Use
        /// `wick list-bundles` to discover available IDs.
        #[arg(long, requires = "quant")]
        bundle_id: Option<String>,

        /// Quantization label for `--bundle-id` (e.g. `Q4_0`,
        /// `Q8_0`). Pairs with `--bundle-id`: clap rejects
        /// either flag without the other.
        #[arg(long, requires = "bundle_id")]
        quant: Option<String>,

        /// Cache root for `--bundle-id` downloads. Default:
        /// `$HOME/.cache/wick`. Used only for bundle download
        /// caching; bench's KV prefix cache is the engine default
        /// (warm-only, in-memory) regardless of this flag —
        /// `wick run --cache-dir <d>` is the right entrypoint
        /// for KV-cache-aware benchmarks. No-op when `--model`
        /// is used.
        #[arg(long)]
        cache_dir: Option<String>,

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

    /// List bundles published on `huggingface.co/LiquidAI/LeapBundles`.
    ///
    /// Discovery surface for the `--bundle-id` / `--quant` flags on
    /// `wick run` / `wick chat` / `wick bench`. Hits the HF
    /// model-info API once and prints sorted bundle names; pass
    /// `--quants` to also surface available quantization labels per
    /// bundle.
    ListBundles {
        /// Also print available quantizations under each bundle,
        /// space-separated and indented.
        #[arg(long)]
        quants: bool,
    },
}

/// Load a `WickEngine` from a path that may be a bare `.gguf`, a `.json`
/// manifest, or a directory containing one `.json` manifest. The engine
/// owns the model + tokenizer for the CLI's lifetime; callers get
/// `engine.new_session(...)` for text and `engine.model()` / `engine.tokenizer()`
/// handles for the audio pipeline.
///
/// Either a local file/manifest path or a LeapBundles `bundle_id`+`quant` pair.
/// The bundle path constructs a `BundleRepo` rooted at `cache_dir` so the
/// downloaded manifest + GGUFs land under a stable location and turn-2+ runs
/// hit the cache.
enum ModelSpec<'a> {
    Path(&'a Path),
    Bundle {
        id: &'a str,
        quant: &'a str,
        cache_dir: PathBuf,
    },
}

/// Suffix appended to bare bundle IDs so users can type
/// `--bundle-id LFM2.5-1.2B-Instruct` instead of
/// `--bundle-id LFM2.5-1.2B-Instruct-GGUF`. Every entry in
/// `LiquidAI/LeapBundles` today ends in `-GGUF` (the directory
/// names are derived from the upstream model + format pair). When
/// other formats land we'll revisit, but for the present catalog
/// this is a pure UX win.
const LEAP_BUNDLE_GGUF_SUFFIX: &str = "-GGUF";

/// Append `-GGUF` to `input` if it isn't already present. The
/// canonical bundle ID on `LiquidAI/LeapBundles` is the HF
/// directory name (always `<base>-GGUF` today), and the library
/// `wick::bundle::leap_bundles_manifest_url` interpolates that
/// verbatim into the URL — so normalization belongs at the CLI
/// boundary where the typo-friendly bare form is accepted.
///
/// Idempotent: passing `LFM2-1.2B-GGUF` returns the same string.
fn normalize_bundle_id(input: &str) -> String {
    if input.ends_with(LEAP_BUNDLE_GGUF_SUFFIX) {
        input.to_string()
    } else {
        format!("{input}{LEAP_BUNDLE_GGUF_SUFFIX}")
    }
}

/// Inverse of [`normalize_bundle_id`] for display: strip a trailing
/// `-GGUF` so `wick list-bundles` shows the same form users type.
/// `strip_suffix` returns `None` when the suffix isn't present,
/// preserving the original name (a future non-GGUF bundle would
/// just display verbatim).
fn display_bundle_id(name: &str) -> &str {
    name.strip_suffix(LEAP_BUNDLE_GGUF_SUFFIX).unwrap_or(name)
}

/// Default cache root, used by the bundle-id flow when `--cache-dir` is
/// unset: `$HOME/.cache/wick` when `$HOME` is set, otherwise
/// `<TMPDIR>/.cache/wick`. Shared between the `BundleRepo` (downloads land
/// under `<root>/huggingface.co/...`) and the KV prefix cache (under
/// `<root>/kv`), so a single `--cache-dir` flag covers both — unify the
/// user's cache footprint instead of fragmenting it.
fn default_cache_dir() -> PathBuf {
    let base = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    base.join(".cache/wick")
}

/// Take the four CLI flags `(--model, --bundle-id, --quant, --cache-dir)` and
/// validate "exactly one source set". Then build a `ModelSpec` and load
/// through [`load_engine_from_spec`]. Errors before any I/O so a flag-misuse
/// surfaces fast with a clear message rather than a downstream HTTP/file
/// error.
fn resolve_engine(
    model: Option<&str>,
    bundle_id: Option<&str>,
    quant: Option<&str>,
    cache_dir: Option<&str>,
    device: &str,
    context_size: usize,
) -> Result<WickEngine> {
    match (model, bundle_id, quant) {
        (Some(p), None, None) => {
            load_engine_from_spec(ModelSpec::Path(Path::new(p)), device, context_size)
        }
        (None, Some(id), Some(q)) => {
            let cache = cache_dir
                .map(PathBuf::from)
                .unwrap_or_else(default_cache_dir);
            // Append `-GGUF` if the user didn't include it. Keeps
            // `--bundle-id LFM2.5-1.2B-Instruct` and the explicit
            // `--bundle-id LFM2.5-1.2B-Instruct-GGUF` both valid;
            // the URL Liquid actually publishes always has the
            // suffix.
            let normalized = normalize_bundle_id(id);
            load_engine_from_spec(
                ModelSpec::Bundle {
                    id: &normalized,
                    quant: q,
                    cache_dir: cache,
                },
                device,
                context_size,
            )
        }
        (None, Some(_), None) | (None, None, Some(_)) => {
            anyhow::bail!(
                "`--bundle-id` and `--quant` must be passed together \
                 (e.g. `--bundle-id LFM2-1.2B-GGUF --quant Q4_0`)"
            )
        }
        (Some(_), Some(_), _) | (Some(_), None, Some(_)) => anyhow::bail!(
            "`--model` and `--bundle-id`/`--quant` are mutually exclusive — \
             pick one source"
        ),
        (None, None, None) => anyhow::bail!(
            "no model source: pass either `--model <path>` or \
             `--bundle-id <id> --quant <quant>`"
        ),
    }
}

fn load_engine_from_spec(
    spec: ModelSpec<'_>,
    device: &str,
    context_size: usize,
) -> Result<WickEngine> {
    let backend = BackendPreference::parse_str(device).map_err(|e| anyhow::anyhow!("{e}"))?;
    let engine = match spec {
        ModelSpec::Path(path) => {
            // `..Default::default()` picks up optional fields (e.g.
            // `bundle_repo`, which is gated behind the `remote` feature
            // — wick-cli now enables it unconditionally to power
            // `--bundle-id`).
            WickEngine::from_path(
                path,
                EngineConfig {
                    context_size,
                    backend,
                    ..Default::default()
                },
            )?
        }
        ModelSpec::Bundle {
            id,
            quant,
            cache_dir,
        } => {
            eprintln!(
                "Resolving bundle `{id}` (quant `{quant}`) into cache `{}`…",
                cache_dir.display()
            );
            // Concrete `Arc<CliDownloadProgress>` first so we can check
            // `printed_any` after the resolve; coerce to `Arc<dyn ...>`
            // only at the BundleRepo call site.
            let progress = Arc::new(CliDownloadProgress::default());
            let repo = wick::bundle::BundleRepo::with_progress(
                cache_dir,
                progress.clone() as Arc<dyn wick::bundle::DownloadProgress>,
            );
            let engine = WickEngine::from_bundle_id(
                id,
                quant,
                EngineConfig {
                    context_size,
                    backend,
                    bundle_repo: Some(repo),
                },
            )?;
            // Seal the final progress line — but only if any progress
            // actually printed. Cache hits resolve without firing the
            // callback, and an unconditional newline would leak a blank
            // line on every silent resolve.
            if progress.printed_any() {
                eprintln!();
            }
            engine
        }
    };
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

/// Configure the engine's KV prefix cache from the four CLI flags shared by
/// `Run` and `Chat`. Encapsulates the `<root>/kv` derivation rule + the
/// `--no-cache` short-circuit so both subcommands stay in sync.
///
/// Behavior:
/// - `no_cache == true` → all-zeros config (warm + cold both disabled).
/// - explicit `--cache-dir foo` → KV files under `foo/kv` (peer of
///   `foo/huggingface.co/...` for bundle downloads).
/// - default `$HOME/.cache/wick/kv` when `$HOME` is set.
/// - no `$HOME` and no `--cache-dir` → KV stays disabled (TMPDIR fallback
///   in `default_cache_dir()` is bundle-only).
fn configure_prefix_cache(
    engine: &WickEngine,
    cache_dir: Option<&str>,
    no_cache: bool,
    cache_warm_mb: u64,
    cache_disk_gb: u64,
) {
    if no_cache {
        engine.configure_cache(wick::kv_cache::KvCacheConfig {
            cache_dir: None,
            max_warm_entries: 0,
            max_warm_bytes: 0,
            max_cold_bytes: 0,
        });
        return;
    }
    let kv_dir: Option<PathBuf> = if let Some(d) = cache_dir {
        Some(PathBuf::from(d).join("kv"))
    } else if std::env::var_os("HOME").is_some() {
        Some(default_cache_dir().join("kv"))
    } else {
        None
    };
    engine.configure_cache(wick::kv_cache::KvCacheConfig {
        cache_dir: kv_dir,
        max_warm_entries: 32,
        max_warm_bytes: cache_warm_mb * 1024 * 1024,
        max_cold_bytes: cache_disk_gb * 1024 * 1024 * 1024,
    });
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

/// Read a PCM16 WAV file and return (samples_f32_in_minus1_to_1,
/// sample_rate). Output is always mono: multi-channel inputs are
/// down-mixed by averaging across channels per frame. Sample rate
/// is returned untouched — resampling happens at the call site.
///
/// Constraints: `audio_format == 1` (PCM), `bits_per_sample == 16`,
/// `channels >= 1`. Anything else errors with a typed message
/// pointing at the offending field.
///
/// Skips unknown subchunks (LIST, JUNK, etc.) between fmt and data
/// per the RIFF spec. Emits a `note:` line on stderr when down-mix
/// happens so the user can see why the channel count dropped.
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
    if channels == 0 {
        bail!("WAV `{path}`: channels=0 in fmt header (must be >= 1)");
    }
    if bits != 16 {
        bail!("WAV `{path}`: {bits} bits/sample (expected 16). Re-encode as 16-bit PCM.");
    }

    // PCM16 frame = `channels` samples × 2 bytes.
    let frame_bytes = 2usize * channels as usize;
    if data_sz % frame_bytes != 0 {
        bail!(
            "WAV `{path}`: data chunk size {data_sz} is not a multiple of frame size \
             {frame_bytes} ({channels} channels × 2 bytes)"
        );
    }
    let n_frames = data_sz / frame_bytes;

    // Symmetric scale: i16::MIN -> -1.0, i16::MAX -> ~1.0. Using 32768
    // (vs 32767) keeps zero exactly at zero and avoids the asymmetric
    // off-by-one when round-tripping through `write_wav` (which clamps
    // before scaling by 32767).
    let read_sample = |frame_idx: usize, ch: usize| -> f32 {
        let o = data_off + (frame_idx * channels as usize + ch) * 2;
        let s = i16::from_le_bytes([buf[o], buf[o + 1]]);
        s as f32 / 32768.0
    };

    let mut samples = Vec::with_capacity(n_frames);
    if channels == 1 {
        for f in 0..n_frames {
            samples.push(read_sample(f, 0));
        }
    } else {
        // Down-mix by averaging across channels. Average (vs sum) keeps
        // amplitudes inside [-1, 1] when each channel is in range; a
        // sum could clip a stereo input where both channels are at full
        // scale.
        let inv = 1.0_f32 / channels as f32;
        for f in 0..n_frames {
            let mut acc = 0.0_f32;
            for c in 0..channels as usize {
                acc += read_sample(f, c);
            }
            samples.push(acc * inv);
        }
        eprintln!(
            "note: down-mixing {channels}-channel WAV `{path}` to mono \
             by averaging across channels. To skip this step, pass mono \
             PCM16 directly — e.g. `sox in.wav -c 1 out.wav` or \
             `ffmpeg -i in.wav -ac 1 out.wav`."
        );
    }
    Ok((samples, sample_rate))
}

/// Linearly resample `samples` from `sr_in` to `sr_out` Hz.
/// Returns `samples` unchanged when `sr_in == sr_out`.
///
/// Linear interpolation is the simplest viable resampler:
/// - Upsample (e.g. 8 kHz → 16 kHz): introduces a smoothed
///   high-frequency rolloff but no aliasing — adequate for ASR.
/// - Downsample (e.g. 44.1 kHz → 16 kHz): does NOT apply an
///   anti-aliasing low-pass filter, so frequencies above the
///   output Nyquist (8 kHz here) fold back as aliasing artifacts.
///   Speech energy is mostly under 8 kHz so this is tolerable for
///   ASR but not studio-quality. Users who care can pre-resample
///   externally with `sox` / `ffmpeg` and pass a 16 kHz WAV
///   directly to bypass this path.
///
/// Time complexity: O(n_out). Space: one allocation of size
/// `n_out * 4 bytes`. No SIMD; the audio path is dwarfed by
/// model inference so the linear scan isn't a bottleneck.
///
/// Empty input or zero rates return an empty `Vec` — these are
/// degenerate but cheap to handle here so the caller doesn't
/// have to special-case them.
fn resample_linear(samples: &[f32], sr_in: u32, sr_out: u32) -> Vec<f32> {
    if samples.is_empty() || sr_in == 0 || sr_out == 0 {
        return Vec::new();
    }
    if sr_in == sr_out {
        return samples.to_vec();
    }
    let n_in = samples.len();
    // Output length scales by the rate ratio. Use f64 to avoid
    // precision loss on long inputs (a 60s @ 44.1kHz clip is
    // 2.6M samples — f32 mantissa starts losing integer fidelity
    // around 16M, so f32 would be fine here, but f64 is free).
    let ratio = sr_out as f64 / sr_in as f64;
    // Clamp to ≥ 1 for non-empty input. Without this a tiny input
    // (e.g. `n_in=1` with `sr_in=48_000, sr_out=16_000`) would
    // round `n_in * ratio = 0.333 → 0` and the resampler would
    // hand back an empty buffer, which `Session::append_audio`
    // surfaces as `EmptyInput`. The empty-input early return
    // above handles `n_in == 0`; this handles the round-to-zero
    // edge case for non-empty input.
    let n_out = ((n_in as f64) * ratio).round().max(1.0) as usize;
    let mut out = Vec::with_capacity(n_out);
    let step = sr_in as f64 / sr_out as f64;
    for i in 0..n_out {
        let pos = i as f64 * step;
        let idx = pos.floor() as usize;
        let frac = (pos - idx as f64) as f32;
        // Clamp to [0, n_in - 1]. The last interval (idx == n_in - 1,
        // frac > 0) interpolates against itself — equivalent to
        // hold-the-last-sample, which is the standard end-of-buffer
        // handling for linear resampling.
        let a = samples[idx.min(n_in - 1)];
        let b = samples[(idx + 1).min(n_in - 1)];
        out.push(a + (b - a) * frac);
    }
    out
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
            bundle_id,
            quant,
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
            // `cache_dir` is shared between bundle downloads (when
            // `--bundle-id` is set) and the KV prefix cache below.
            let engine = resolve_engine(
                model.as_deref(),
                bundle_id.as_deref(),
                quant.as_deref(),
                cache_dir.as_deref(),
                &device,
                context_size,
            )?;
            let tokenizer = engine.tokenizer();
            let add_bos = engine.metadata().add_bos_token;

            let kv_compression = setup_kv_compression(engine.model(), &kv_cache_keys)?;

            // Configure KV prefix cache (shared logic with `Chat`).
            configure_prefix_cache(
                &engine,
                cache_dir.as_deref(),
                no_cache,
                cache_warm_mb,
                cache_disk_gb,
            );

            // Audio-input path (mutually exclusive with --vocoder via clap
            // `conflicts_with`). Skips the chat-template / token-building
            // dance entirely — the audio is fed in as soft tokens via
            // `Session::append_audio`, which uses the engine's
            // auto-attached `AudioEncoderWeights` (PR #106).
            if let Some(wav_path) = &audio_in {
                let (pcm_in, sr_in) = read_wav_pcm16_mono(wav_path)?;
                eprintln!(
                    "Loaded {} samples ({:.2}s @ {sr_in} Hz) from {wav_path}",
                    pcm_in.len(),
                    pcm_in.len() as f32 / sr_in as f32
                );
                // LFM2A's audio encoder expects 16 kHz mono PCM. If the
                // WAV is at any other rate, resample with linear
                // interpolation. Quality is adequate for ASR speech
                // (no anti-aliasing on downsample, but speech energy
                // is mostly under 8 kHz so the worst aliasing folds
                // outside the recognized band). For studio quality on
                // long clips, pre-resample externally with sox/ffmpeg.
                const TARGET_SR: u32 = 16_000;
                let (pcm, sr) = if sr_in == TARGET_SR {
                    (pcm_in, sr_in)
                } else {
                    let resampled = resample_linear(&pcm_in, sr_in, TARGET_SR);
                    eprintln!(
                        "note: resampling {sr_in} Hz → {TARGET_SR} Hz \
                         ({} → {} samples, linear interpolation). \
                         For best quality and performance pass \
                         16 kHz mono PCM16 directly — e.g. \
                         `sox in.wav -r 16000 -c 1 -b 16 out.wav` \
                         or `ffmpeg -i in.wav -ar 16000 -ac 1 -sample_fmt s16 out.wav` \
                         — to skip this step.",
                        pcm_in.len(),
                        resampled.len()
                    );
                    (resampled, TARGET_SR)
                };
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
        Command::ListBundles { quants } => {
            // One HTTP round-trip; sorted output. Quants are
            // space-joined on a single indented line per bundle —
            // compact enough to fit in a terminal width even for
            // bundles with several quant variants, and grep-able.
            //
            // Display strip: every entry in the live catalog ends
            // in `-GGUF`. Trim the suffix on output so what's
            // shown matches what users type at `--bundle-id`
            // (`normalize_bundle_id` re-appends the suffix on the
            // way back in). `display_bundle_id` is a no-op for
            // any future non-GGUF entry that ships.
            let entries = wick::bundle::list_leap_bundles()?;
            for entry in entries {
                println!("{}", display_bundle_id(&entry.name));
                if quants {
                    println!("  {}", entry.quants.join("  "));
                }
            }
        }
        Command::Chat {
            model,
            bundle_id,
            quant,
            cache_dir,
            cache_warm_mb,
            cache_disk_gb,
            no_cache,
            system,
            device,
            context_size,
            max_tokens,
            temperature,
            seed,
            no_tui,
        } => {
            use std::io::BufRead;

            let engine = resolve_engine(
                model.as_deref(),
                bundle_id.as_deref(),
                quant.as_deref(),
                cache_dir.as_deref(),
                &device,
                context_size,
            )?;
            let tokenizer = engine.tokenizer();

            // Configure the KV prefix cache (shared logic with `Run`).
            // Cross-restart disk-tier caching is the win for mobile / FFI
            // consumers whose process can be killed and resumed.
            configure_prefix_cache(
                &engine,
                cache_dir.as_deref(),
                no_cache,
                cache_warm_mb,
                cache_disk_gb,
            );

            // Up-front chat-template probe: fail before the user types
            // anything if the model has no template metadata. Without this,
            // the first per-turn `apply_chat_template` would `?`-return
            // through the entire CLI after the user already typed a
            // message — confusing UX.
            if tokenizer.chat_template().is_none() {
                anyhow::bail!(
                    "model has no chat template metadata; \
                     `wick chat` requires a chat-tuned model. \
                     Use `wick run --prompt <text>` for plain completion instead."
                );
            }

            let session = engine.new_session(wick::SessionConfig {
                seed,
                ..Default::default()
            });

            let mut history: Vec<wick::tokenizer::ChatMessage> = Vec::new();
            if let Some(sys) = system {
                history.push(wick::tokenizer::ChatMessage {
                    role: "system".into(),
                    content: sys,
                });
            }

            let opts = wick::GenerateOpts {
                max_tokens: max_tokens as u32,
                temperature,
                ..Default::default()
            };

            // Dispatch: inline TUI when both stdin AND stdout are TTYs
            // (so cursor positioning + raw-mode keystroke reads behave
            // sensibly) and the user didn't opt out via `--no-tui`.
            // Otherwise fall back to the line-based REPL — works for
            // pipes, log capture, dumb terminals, scripted tests.
            let use_tui =
                !no_tui && std::io::stdin().is_terminal() && std::io::stdout().is_terminal();
            if use_tui {
                // `tokenizer_arc()` shares the engine's existing
                // `Arc<BpeTokenizer>` with the worker thread instead
                // of deep-cloning the vocab + merge tables.
                let _final_history = chat_tui::run(session, engine.tokenizer_arc(), history, opts)?;
                return Ok(());
            }

            // Line-based REPL fallback below.
            let mut session = session;
            // The actual backend choice is reported by `load_engine` via
            // its own "Using ... backend" line above; don't echo `device`
            // here since under `--device auto` it'd misrepresent the
            // resolved backend.
            eprintln!(
                "wick chat (ctx={context_size}, max_tokens/turn={max_tokens}). \
                 Type `/help` for commands, `/exit` or EOF (Ctrl+D) to quit."
            );
            if !history.is_empty() {
                eprintln!("(system prompt active)");
            }

            let stdin = std::io::stdin();
            let cancel = session.cancel_handle();
            loop {
                eprint!("user> ");
                std::io::stderr().flush().ok();
                let mut line = String::new();
                let n = stdin.lock().read_line(&mut line)?;
                if n == 0 {
                    eprintln!();
                    break; // EOF
                }
                // Strip line terminators only; preserve any meaningful
                // leading/trailing whitespace the user actually typed
                // (e.g. indented code in a prompt).
                let user = line.trim_end_matches(['\r', '\n']).to_string();
                if user.trim().is_empty() {
                    continue;
                }
                // Slash-command dispatch. A leading `/` puts the line
                // into command mode; we never send it to the model.
                // Trade-off: legit user messages that genuinely start
                // with `/` (e.g. a Unix path) are unreachable today.
                // Mitigations would be `\\` escape or a `/say <text>`
                // form; not worth the complexity for v1.
                if user.starts_with('/') {
                    // Trim trailing whitespace so commands like "/help "
                    // or "/exit\t" still dispatch correctly. The user's
                    // line preserves leading/trailing spaces inside a
                    // chat message (intentional for indented prompts),
                    // but a bare command shouldn't be defeated by a
                    // stray space that's hard to see.
                    match user.trim_end() {
                        "/exit" | "/quit" => break,
                        "/help" => {
                            eprintln!("Commands:");
                            eprintln!(
                                "  /clear          Clear conversation history (system prompt is preserved)"
                            );
                            eprintln!("  /help           Show this help");
                            eprintln!("  /exit, /quit    Exit the REPL");
                            continue;
                        }
                        "/clear" => {
                            // Reset history but preserve the initial
                            // system message if one was set via
                            // `--system`. Drop the engine session's
                            // KV state so the next turn starts cold
                            // (the prefix cache will hit if the
                            // resulting render matches a cached
                            // prefill).
                            let had_system = history.first().is_some_and(|m| m.role == "system");
                            if had_system {
                                history.truncate(1);
                            } else {
                                history.clear();
                            }
                            session.reset();
                            eprintln!(
                                "(history cleared{})",
                                if had_system {
                                    "; system prompt preserved"
                                } else {
                                    ""
                                }
                            );
                            continue;
                        }
                        other => {
                            eprintln!(
                                "unknown command: {other}. Type /help for available commands."
                            );
                            continue;
                        }
                    }
                }

                history.push(wick::tokenizer::ChatMessage {
                    role: "user".into(),
                    content: user,
                });

                // Re-render the full conversation per turn and rely on
                // the engine's prefix cache for fast turn-N+1 prefill.
                // Delta-prefill is a future optimization; the simplicity
                // of "render fresh, reset, prefill" is worth the lookup.
                let formatted =
                    match wick::tokenizer::apply_chat_template(tokenizer, &history, true) {
                        Ok(s) => s,
                        Err(e) => {
                            eprintln!("error: chat-template render failed: {e}");
                            history.pop();
                            continue;
                        }
                    };
                let tokens = tokenizer.encode(&formatted);

                session.reset();
                if let Err(e) = session.append_tokens(&tokens) {
                    eprintln!("error: append_tokens failed: {e}");
                    history.pop();
                    continue;
                }

                eprint!("assistant> ");
                std::io::stderr().flush().ok();

                let mut sink = ChatSink::new(tokenizer, session.cancel_handle());
                let summary = match session.generate(&opts, &mut sink) {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!("\nerror: generate failed: {e}");
                        history.pop();
                        continue;
                    }
                };
                eprintln!();
                history.push(wick::tokenizer::ChatMessage {
                    role: "assistant".into(),
                    content: sink.into_text(),
                });

                // Stop the REPL if the sink flipped cancel because stdout
                // is gone (BrokenPipe — the user piped us into `head` or
                // similar). Without this we'd keep prompting + decoding
                // even though nothing reaches the terminal anymore.
                if matches!(summary.finish_reason, wick::FinishReason::Cancelled)
                    && cancel.load(Ordering::Relaxed)
                {
                    break;
                }
            }
        }
        Command::Bench {
            model,
            bundle_id,
            quant,
            cache_dir,
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

            let engine = resolve_engine(
                model.as_deref(),
                bundle_id.as_deref(),
                quant.as_deref(),
                cache_dir.as_deref(),
                &device,
                context_size,
            )?;
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
    use super::{
        Cli, display_bundle_id, normalize_bundle_id, read_wav_pcm16_mono, resample_linear,
        resolve_engine, split_at_marker, write_wav,
    };
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

    /// `chat` subcommand must parse cleanly with the minimal
    /// `--model` flag — the rest of its surface has defaults so a
    /// bare invocation is the documented entry point.
    #[test]
    fn chat_subcommand_parses_minimal() {
        let r = Cli::try_parse_from(["wick", "chat", "--model", "/tmp/x"]);
        assert!(
            r.is_ok(),
            "expected `chat --model <path>` alone to parse, got: {:?}",
            r.err()
        );
    }

    /// `chat` must also accept the full v1 flag surface together —
    /// guards against a flag rename / clap drift breaking the
    /// documented invocation. Includes the KV prefix cache flags
    /// (`--cache-warm-mb`, `--cache-disk-gb`, `--no-cache`) so the
    /// mobile-app "survive process restart" config path stays
    /// parsable.
    #[test]
    fn chat_subcommand_parses_full_flags() {
        let r = Cli::try_parse_from([
            "wick",
            "chat",
            "--model",
            "/tmp/x",
            "--system",
            "Be brief",
            "--device",
            "cpu",
            "--context-size",
            "2048",
            "--max-tokens",
            "256",
            "--temperature",
            "0.5",
            "--seed",
            "42",
            "--cache-dir",
            "/tmp/cache",
            "--cache-warm-mb",
            "128",
            "--cache-disk-gb",
            "5",
        ]);
        assert!(
            r.is_ok(),
            "expected full `chat` flag surface to parse, got: {:?}",
            r.err()
        );
    }

    /// `chat --no-cache` must parse cleanly — explicit opt-out of
    /// the KV prefix cache for users who'd rather not write to disk.
    #[test]
    fn chat_subcommand_parses_no_cache() {
        let r = Cli::try_parse_from(["wick", "chat", "--model", "/tmp/x", "--no-cache"]);
        assert!(
            r.is_ok(),
            "expected `chat --no-cache` to parse, got: {:?}",
            r.err()
        );
    }

    /// `chat --bundle-id <id> --quant <q>` parses cleanly without
    /// `--model`. This is the documented auto-download entry
    /// point — clap-rejecting it would make the LeapBundles flow
    /// unusable.
    #[test]
    fn chat_subcommand_parses_with_bundle_id() {
        let r = Cli::try_parse_from([
            "wick",
            "chat",
            "--bundle-id",
            "LFM2-1.2B-GGUF",
            "--quant",
            "Q4_0",
        ]);
        assert!(
            r.is_ok(),
            "expected `chat --bundle-id X --quant Y` to parse, got: {:?}",
            r.err()
        );
    }

    /// `--bundle-id` without `--quant` must be rejected by clap
    /// (and vice-versa) because the LeapBundles URL needs both.
    /// Surfacing this at parse time is friendlier than letting
    /// it through to a runtime error.
    #[test]
    fn chat_subcommand_rejects_bundle_id_without_quant() {
        for partial in [
            vec!["wick", "chat", "--bundle-id", "X"],
            vec!["wick", "chat", "--quant", "Q4_0"],
        ] {
            let r = Cli::try_parse_from(&partial);
            assert!(
                r.is_err(),
                "expected partial bundle args {partial:?} to be rejected by clap"
            );
        }
    }

    /// `--model` is mutually exclusive with `--bundle-id` /
    /// `--quant` — passing both is meaningless and we want the
    /// error at parse time, not after a wasted download attempt.
    #[test]
    fn chat_subcommand_rejects_model_with_bundle_id() {
        let r = Cli::try_parse_from([
            "wick",
            "chat",
            "--model",
            "/tmp/x",
            "--bundle-id",
            "LFM2-1.2B-GGUF",
            "--quant",
            "Q4_0",
        ]);
        assert!(
            r.is_err(),
            "expected `--model` + `--bundle-id` to be rejected by clap"
        );
    }

    /// Same auto-download entry point on `run`. Symmetric with
    /// `chat`'s parse test — flag drift on either subcommand
    /// breaks the documented form.
    #[test]
    fn run_subcommand_parses_with_bundle_id() {
        let r = Cli::try_parse_from([
            "wick",
            "run",
            "--bundle-id",
            "LFM2-1.2B-GGUF",
            "--quant",
            "Q4_0",
            "--prompt",
            "hi",
        ]);
        assert!(
            r.is_ok(),
            "expected `run --bundle-id X --quant Y` to parse, got: {:?}",
            r.err()
        );
    }

    /// `bench` is the third subcommand carrying the bundle-id
    /// surface; ensure it parses too.
    #[test]
    fn bench_subcommand_parses_with_bundle_id() {
        let r = Cli::try_parse_from([
            "wick",
            "bench",
            "--bundle-id",
            "LFM2-1.2B-GGUF",
            "--quant",
            "Q4_0",
        ]);
        assert!(
            r.is_ok(),
            "expected `bench --bundle-id X --quant Y` to parse, got: {:?}",
            r.err()
        );
    }

    /// `bench --model` + `--bundle-id` must be rejected at parse
    /// time (mutually exclusive). Mirrors the gate on `run` /
    /// `chat`.
    #[test]
    fn bench_subcommand_rejects_model_with_bundle_id() {
        let r = Cli::try_parse_from([
            "wick",
            "bench",
            "--model",
            "/tmp/x",
            "--bundle-id",
            "LFM2-1.2B-GGUF",
            "--quant",
            "Q4_0",
        ]);
        assert!(
            r.is_err(),
            "expected `bench --model` + `--bundle-id` to be rejected by clap"
        );
    }

    /// `resolve_engine` with no source flags errors out before
    /// touching disk or network — fast clear failure on a misuse
    /// that clap can't catch (both `--model` and `--bundle-id`
    /// being optional means clap accepts the empty case).
    #[test]
    fn resolve_engine_rejects_no_source() {
        let r = resolve_engine(None, None, None, None, "cpu", 1024);
        let err = match r {
            Ok(_) => panic!("expected an error when no source flags are set"),
            Err(e) => e,
        };
        let msg = format!("{err:#}");
        assert!(
            msg.contains("no model source"),
            "error should explain that no source was given; got: {msg}"
        );
    }

    /// `resolve_engine` rejects `--bundle-id` without `--quant`
    /// (and vice-versa). Clap's `requires` already catches this
    /// at parse time, but the helper has its own guard for
    /// programmatic callers (and as a defense-in-depth check).
    #[test]
    fn resolve_engine_rejects_partial_bundle_args() {
        let r = resolve_engine(None, Some("LFM2-1.2B-GGUF"), None, None, "cpu", 1024);
        assert!(
            r.is_err(),
            "expected partial bundle args (id only) to error"
        );
        let r = resolve_engine(None, None, Some("Q4_0"), None, "cpu", 1024);
        assert!(
            r.is_err(),
            "expected partial bundle args (quant only) to error"
        );
    }

    /// `normalize_bundle_id` appends `-GGUF` when missing and is
    /// idempotent when the suffix is already present. The CLI
    /// applies this in `resolve_engine` so users can type either
    /// form at `--bundle-id`.
    #[test]
    fn normalize_bundle_id_appends_when_missing() {
        assert_eq!(
            normalize_bundle_id("LFM2.5-1.2B-Instruct"),
            "LFM2.5-1.2B-Instruct-GGUF"
        );
        assert_eq!(normalize_bundle_id("Qwen3-1.7B"), "Qwen3-1.7B-GGUF");
    }

    #[test]
    fn normalize_bundle_id_is_idempotent() {
        assert_eq!(
            normalize_bundle_id("LFM2-1.2B-GGUF"),
            "LFM2-1.2B-GGUF",
            "must not double-append"
        );
    }

    /// `display_bundle_id` strips the trailing `-GGUF` for
    /// presentation (matches what users type at `--bundle-id`)
    /// and is a no-op for any future entry without the suffix.
    #[test]
    fn display_bundle_id_strips_gguf_suffix() {
        assert_eq!(display_bundle_id("LFM2-1.2B-GGUF"), "LFM2-1.2B");
        assert_eq!(
            display_bundle_id("LFM2.5-1.2B-Instruct-GGUF"),
            "LFM2.5-1.2B-Instruct"
        );
        // Hypothetical future non-GGUF bundle: passes through.
        assert_eq!(display_bundle_id("LFM3-MLX"), "LFM3-MLX");
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

    /// Stereo WAV must be down-mixed to mono by averaging across
    /// channels. Hand-craft a 4-frame stereo file with L=+0.5,
    /// R=-0.5 and verify the mono output is all zeros (avg of
    /// opposites = 0).
    #[test]
    fn read_wav_downmixes_stereo_to_average() {
        // Stereo PCM16 @ 16 kHz, 4 frames. L=+0.5 (16384), R=-0.5 (-16384).
        let l: i16 = 16_384;
        let r: i16 = -16_384;
        let mut data: Vec<u8> = Vec::new();
        for _ in 0..4 {
            data.extend_from_slice(&l.to_le_bytes());
            data.extend_from_slice(&r.to_le_bytes());
        }
        let data_sz = data.len() as u32;

        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&(36 + data_sz).to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
        buf.extend_from_slice(&2u16.to_le_bytes()); // 2 channels
        buf.extend_from_slice(&16_000u32.to_le_bytes());
        buf.extend_from_slice(&64_000u32.to_le_bytes());
        buf.extend_from_slice(&4u16.to_le_bytes()); // block align (2ch * 2B)
        buf.extend_from_slice(&16u16.to_le_bytes());
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&data_sz.to_le_bytes());
        buf.extend_from_slice(&data);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("stereo.wav");
        std::fs::write(&path, &buf).unwrap();

        let (samples, sr) = read_wav_pcm16_mono(path.to_str().unwrap()).unwrap();
        assert_eq!(sr, 16_000);
        assert_eq!(
            samples.len(),
            4,
            "stereo down-mix should yield 4 mono frames"
        );
        for (i, &s) in samples.iter().enumerate() {
            assert!(
                s.abs() < 1e-3,
                "frame {i}: avg of +0.5 and -0.5 should be ~0; got {s}"
            );
        }
    }

    /// 4-channel WAV must average all four channels per frame.
    #[test]
    fn read_wav_downmixes_quad_to_average() {
        // 4-channel @ 16 kHz, 2 frames. Channels = +1.0, +0.5, -0.5, -1.0.
        // Average per frame = 0.0.
        let s1: i16 = 32_767; // ~+1.0
        let s2: i16 = 16_384; // +0.5
        let s3: i16 = -16_384; // -0.5
        let s4: i16 = -32_768; // -1.0
        let mut data: Vec<u8> = Vec::new();
        for _ in 0..2 {
            data.extend_from_slice(&s1.to_le_bytes());
            data.extend_from_slice(&s2.to_le_bytes());
            data.extend_from_slice(&s3.to_le_bytes());
            data.extend_from_slice(&s4.to_le_bytes());
        }
        let data_sz = data.len() as u32;

        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&(36 + data_sz).to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.extend_from_slice(&4u16.to_le_bytes()); // 4 channels
        buf.extend_from_slice(&16_000u32.to_le_bytes());
        buf.extend_from_slice(&128_000u32.to_le_bytes());
        buf.extend_from_slice(&8u16.to_le_bytes()); // block align
        buf.extend_from_slice(&16u16.to_le_bytes());
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&data_sz.to_le_bytes());
        buf.extend_from_slice(&data);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("quad.wav");
        std::fs::write(&path, &buf).unwrap();

        let (samples, _sr) = read_wav_pcm16_mono(path.to_str().unwrap()).unwrap();
        assert_eq!(samples.len(), 2);
        // i16::MAX is 32767 vs i16::MIN = -32768 → asymmetric scale
        // means the +1.0/-1.0 pair averages to ~-1/(4*32768) ≈ -7.6e-6,
        // which combined with +0.5/-0.5 (exact 0) gives ~-1.9e-6.
        // Use a generous epsilon since the meaningful claim is "averages
        // to zero", not "byte-exact".
        for (i, &s) in samples.iter().enumerate() {
            assert!(
                s.abs() < 1e-3,
                "frame {i}: 4-channel avg should be ~0; got {s}"
            );
        }
    }

    /// channels=0 in the fmt header is malformed and must be rejected.
    #[test]
    fn read_wav_rejects_zero_channels() {
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&36u32.to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
        buf.extend_from_slice(&0u16.to_le_bytes()); // 0 channels (malformed)
        buf.extend_from_slice(&16_000u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.extend_from_slice(&16u16.to_le_bytes());
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&0u32.to_le_bytes());

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("zero_ch.wav");
        std::fs::write(&path, &buf).unwrap();

        let err = read_wav_pcm16_mono(path.to_str().unwrap()).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("channels=0"),
            "error should mention channels=0; got: {msg}"
        );
    }

    /// `resample_linear` with `sr_in == sr_out` must return the input
    /// unchanged. Hot path for the common 16 kHz → 16 kHz case (every
    /// invocation that doesn't actually need resampling); a regression
    /// here would silently corrupt every same-rate ASR call.
    #[test]
    fn resample_linear_same_rate_is_identity() {
        let samples: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.01).sin()).collect();
        let out = resample_linear(&samples, 16_000, 16_000);
        assert_eq!(out, samples);
    }

    /// Empty input must produce empty output without panicking — the
    /// caller's `n_samples` could be 0 if a fixture WAV's data chunk
    /// is missing and the assertion above is bypassed somehow.
    #[test]
    fn resample_linear_empty_input() {
        let out = resample_linear(&[], 44_100, 16_000);
        assert!(out.is_empty());
        let out = resample_linear(&[1.0, 2.0], 0, 16_000);
        assert!(out.is_empty());
        let out = resample_linear(&[1.0, 2.0], 16_000, 0);
        assert!(out.is_empty());
    }

    /// 2× upsample (8 kHz → 16 kHz). Output length doubles; output
    /// values reproduce the original at even indices and the
    /// midpoint between adjacent originals at odd indices.
    #[test]
    fn resample_linear_2x_upsample_interpolates_midpoints() {
        let input = vec![0.0, 1.0, 0.0, -1.0]; // 4 samples
        let out = resample_linear(&input, 8_000, 16_000);
        // n_out = 4 * 2 = 8 samples expected.
        assert_eq!(out.len(), 8);
        // Even indices recover originals (positions 0, 1, 2, 3).
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[2] - 1.0).abs() < 1e-6);
        assert!((out[4] - 0.0).abs() < 1e-6);
        assert!((out[6] - -1.0).abs() < 1e-6);
        // Odd indices interpolate midpoints (0.5, 0.5, -0.5).
        assert!((out[1] - 0.5).abs() < 1e-6);
        assert!((out[3] - 0.5).abs() < 1e-6);
        assert!((out[5] - -0.5).abs() < 1e-6);
        // Last odd index has no "next" sample — linear resamplers
        // typically hold the last value; we mirror that.
        assert!((out[7] - -1.0).abs() < 1e-6);
    }

    /// Tiny non-empty inputs must still produce ≥ 1 output sample.
    /// Without the `n_out.max(1)` clamp, a 1-sample input at 48 kHz
    /// → 16 kHz would round `n_out = 1 * 1/3 = 0.33 → 0` and hand
    /// back an empty buffer, which `Session::append_audio` surfaces
    /// as `EmptyInput` — degrading a degenerate-but-valid call into
    /// a confusing error several layers downstream.
    #[test]
    fn resample_linear_tiny_input_does_not_round_to_empty() {
        // 1 sample, 48 kHz → 16 kHz. Without the clamp, n_out = 0.
        let out = resample_linear(&[0.7], 48_000, 16_000);
        assert!(!out.is_empty(), "1 sample 48k → 16k should not be empty");
        assert_eq!(out.len(), 1);
        assert!((out[0] - 0.7).abs() < 1e-6);

        // 2 samples, 48 kHz → 16 kHz: n_out = round(2/3) = 1.
        // Different from the 1-sample case — clamp doesn't fire,
        // we get the natural rounded length.
        let out = resample_linear(&[0.5, 1.0], 48_000, 16_000);
        assert_eq!(out.len(), 1);
    }

    /// Output length scales by the rate ratio (within ±1 sample
    /// from rounding). 1 second of 44.1 kHz → ~16 000 samples at
    /// 16 kHz.
    #[test]
    fn resample_linear_output_length_matches_ratio() {
        let n_in = 44_100; // 1 s @ 44.1 kHz
        let input = vec![0.0; n_in];
        let out = resample_linear(&input, 44_100, 16_000);
        let expected = 16_000;
        let diff = (out.len() as i64 - expected as i64).abs();
        assert!(
            diff <= 1,
            "44.1k → 16k for {n_in} samples: got {} (expected {expected} ±1)",
            out.len()
        );
        // Spot-check 48 kHz → 16 kHz (exact 1/3 ratio).
        let n_in_48 = 48_000;
        let input48 = vec![0.0; n_in_48];
        let out48 = resample_linear(&input48, 48_000, 16_000);
        assert_eq!(out48.len(), 16_000);
    }
}
