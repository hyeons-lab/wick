#![cfg(all(feature = "metal", target_os = "macos"))]

//! Correctness parity between Metal attention kernel variants.
//!
//! The four single-token attention kernels — `attention.metal` (classic,
//! default), `flash_attention.metal` (seq_len > 4096 auto-switch, opt-in
//! via `WICK_FLASH=1`), `attention_gqa.metal` (`WICK_ATTN=gqa`), and
//! `attention_splitk.metal` (`WICK_ATTN=splitk`) — must all produce
//! identical greedy output for the same prompt on the same model. Each
//! previously bound K/V caches as `float*` while the caches are f16,
//! producing garbage output; the tests below caught that regression.
//!
//! Run with:
//!   cargo test -p wick --release --features metal --test attention_metal_parity -- --ignored --nocapture

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, OnceLock};

fn find_model(name: &str) -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var("HOME").expect("HOME not set"))
        .join(".leap/models")
        .join(name)
        .join(format!("{name}.gguf"));
    if p.exists() {
        Some(p)
    } else {
        eprintln!("model not found: {} — skipping", p.display());
        None
    }
}

/// Greedy-decode `max_tokens` after prefilling `prompt` and return the
/// generated-token slice. Constructs a fresh model so the `WICK_FLASH` env
/// var is picked up at load time (`metal_lfm2.rs:509`).
fn generate_greedy(model_path: &Path, prompt: &str, max_tokens: usize) -> Vec<u32> {
    use wick::kv_cache::KvCompression;
    use wick::model::metal_lfm2::MetalLfm2Model;
    use wick::{FinishReason, GenerateOpts, ModalitySink, Session, SessionConfig};

    let gguf = wick::gguf::GgufFile::open(model_path).unwrap();
    let tokenizer = wick::tokenizer::BpeTokenizer::from_gguf(&gguf).unwrap();
    let model = MetalLfm2Model::from_gguf(gguf, model_path, 4096).unwrap();
    let prompt_toks = tokenizer.encode(prompt);

    struct CollectSink(Vec<u32>);
    impl ModalitySink for CollectSink {
        fn on_text_tokens(&mut self, tokens: &[u32]) {
            self.0.extend_from_slice(tokens);
        }
        fn on_done(&mut self, _: FinishReason) {}
    }

    let mut session = Session::new(
        &model,
        &tokenizer,
        SessionConfig {
            kv_compression: KvCompression::None,
            seed: None,
            ..Default::default()
        },
    );
    session.append_tokens(&prompt_toks).unwrap();

    let opts = GenerateOpts {
        max_tokens: max_tokens as u32,
        temperature: 0.0,
        ..Default::default()
    };
    let mut sink = CollectSink(Vec::new());
    session.generate(&opts, &mut sink).unwrap();
    sink.0
}

const PROMPT: &str = "The capital of France is";
const N_TOKENS: usize = 12;

/// Process-wide mutex serializing env-var mutations across tests in this
/// binary. Required because `std::env::set_var` is `unsafe` in the 2024
/// edition — it races with any other thread reading the environment.
fn env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

/// RAII guard that sets an env var and restores the prior value on drop
/// (including on panic). Holds the process-wide env-var mutex for its
/// lifetime so concurrent tests can't interleave env-var mutations.
struct EnvVarGuard {
    var: &'static str,
    prev: Option<OsString>,
    // Underscore binding to keep the lock alive without triggering
    // unused-variable warnings. Dropped last (after prev restoration)
    // because fields drop in declaration order — we want the restore to
    // happen under the lock.
    _lock: MutexGuard<'static, ()>,
}

impl EnvVarGuard {
    fn set(var: &'static str, val: Option<&str>) -> Self {
        let lock = env_lock().lock().unwrap_or_else(|poisoned| {
            // Another test panicked while holding the lock. The env var was
            // restored by that test's Drop impl (unwinding still runs Drop),
            // so recovering is safe.
            poisoned.into_inner()
        });
        let prev = std::env::var_os(var);
        // SAFETY: we hold the process-wide env lock for the lifetime of this
        // guard, so no other Rust code in this process mutates or reads
        // environment variables concurrently.
        unsafe {
            match val {
                Some(v) => std::env::set_var(var, v),
                None => std::env::remove_var(var),
            }
        }
        EnvVarGuard {
            var,
            prev,
            _lock: lock,
        }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        // SAFETY: same as in `set` — we still hold `_lock` until Drop ends.
        unsafe {
            match &self.prev {
                Some(v) => std::env::set_var(self.var, v),
                None => std::env::remove_var(self.var),
            }
        }
    }
}

fn with_env<F: FnOnce() -> Vec<u32>>(var: &'static str, val: Option<&str>, f: F) -> Vec<u32> {
    let _guard = EnvVarGuard::set(var, val);
    f()
}

fn assert_parity(baseline: &[u32], variant: &[u32], variant_name: &str) {
    eprintln!("classic tokens:  {baseline:?}");
    eprintln!("{variant_name:<16} {variant:?}");
    assert_eq!(
        baseline, variant,
        "{variant_name} produced different greedy tokens than classic — likely a \
         kernel correctness bug (e.g. KV dtype mismatch: the cache is stored as \
         f16 but the kernel declares `float*`, so every load reinterprets two \
         halves as one f32).\nclassic = {baseline:?}\n{variant_name} = {variant:?}"
    );
}

/// Flash vs classic attention must produce the same greedy tokens. Caught
/// the f16/f32 dtype mismatch in `flash_attention.metal`.
#[test]
#[ignore]
fn test_classic_vs_flash_attention_parity() {
    let Some(path) = find_model("LFM2.5-VL-450M-Q4_0") else {
        return;
    };
    let classic = with_env("WICK_FLASH", None, || {
        generate_greedy(&path, PROMPT, N_TOKENS)
    });
    let flash = with_env("WICK_FLASH", Some("1"), || {
        generate_greedy(&path, PROMPT, N_TOKENS)
    });
    assert_parity(&classic, &flash, "flash");
}

/// GQA attention (`WICK_ATTN=gqa`) must match classic. Same f16/f32 bug
/// class as flash — `attention_gqa.metal` bound K/V as `float*` before
/// this fix.
#[test]
#[ignore]
fn test_classic_vs_gqa_attention_parity() {
    let Some(path) = find_model("LFM2.5-VL-450M-Q4_0") else {
        return;
    };
    // LFM2-450M has n_heads=16 / n_kv_heads=8 → group_size=2, which is within
    // the GQA kernel's supported range (2–4).
    let classic = with_env("WICK_ATTN", None, || {
        generate_greedy(&path, PROMPT, N_TOKENS)
    });
    let gqa = with_env("WICK_ATTN", Some("gqa"), || {
        generate_greedy(&path, PROMPT, N_TOKENS)
    });
    assert_parity(&classic, &gqa, "gqa");
}

/// Split-K attention (`WICK_ATTN=splitk`) must match classic. Same
/// f16/f32 bug class as flash — `attention_splitk.metal`'s compute kernel
/// bound K/V as `float*` before this fix.
#[test]
#[ignore]
fn test_classic_vs_splitk_attention_parity() {
    let Some(path) = find_model("LFM2.5-VL-450M-Q4_0") else {
        return;
    };
    let classic = with_env("WICK_ATTN", None, || {
        generate_greedy(&path, PROMPT, N_TOKENS)
    });
    let splitk = with_env("WICK_ATTN", Some("splitk"), || {
        generate_greedy(&path, PROMPT, N_TOKENS)
    });
    assert_parity(&classic, &splitk, "splitk");
}
