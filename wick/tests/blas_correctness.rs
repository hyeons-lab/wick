//! BLAS-path correctness oracle for `Lfm2Model::forward_prefill`.
//!
//! In the follow-up to PR #61 (this PR), the BLAS arm of
//! `forward_prefill` was lifted from `cfg(target_arch = "aarch64")`
//! to `cfg(any(target_arch = "aarch64", feature = "blas"))` so
//! x86_64+blas can compile + use it — PR #61's BLAS-on-CI premise,
//! redeemed here. That cfg surgery is mostly
//! mechanical, but a wrong-output bug specific to x86_64+blas (e.g.
//! transposed matrix layout, off-by-one stride, OpenBLAS
//! disagreement with Accelerate on edge cases) wouldn't be caught
//! by `shift_real_model` (only checks "shift fired + didn't panic")
//! or `wick-parity` (where every leg runs the same wick build, so
//! they'd all be wrong together).
//!
//! Strategy: hardcode a reference token vector that aarch64+blas
//! (macOS, where BLAS has shipped + worked for months) produces on a
//! fixed prompt. Assert that any target-with-`blas` produces the same
//! vector. Greedy decode (T=0, top_k=1) is robust to small float
//! diffs between BLAS implementations (Accelerate vs OpenBLAS) — if
//! the argmax order changes that's a real correctness regression.
//!
//! Brittleness — the reference is hard-tied to the LFM2 fixture
//! (`LFM2-350M-Extract-GGUF/Q4_0`) + the prompt + the greedy
//! constants. If any of those change, regenerate the reference:
//!
//! ```sh
//! WICK_TEST_DOWNLOAD=1 cargo test -p wick \
//!     --features remote,mmap,blas --test blas_correctness \
//!     -- --ignored --nocapture
//! ```
//!
//! On a clean run the test prints `tokens=[...]` to stdout (the
//! captured token vec); paste that into `EXPECTED` below.
//!
//! Gating: `cfg(all(remote, mmap, blas))` so a default `cargo test`
//! never compiles it; `#[ignore]` + `WICK_TEST_DOWNLOAD=1` so even an
//! explicit `--features ...` invocation skips unless opted in to the
//! ~210 MB download.

#![cfg(all(feature = "remote", feature = "mmap", feature = "blas"))]

mod common;

use wick::FinishReason;
use wick::bundle::BundleRepo;
use wick::engine::{BackendPreference, EngineConfig, WickEngine};
use wick::session::{GenerateOpts, ModalitySink, SessionConfig};

/// Captured on macOS arm64 (Accelerate-backed BLAS) at branch
/// `perf/blas-x86-ungate` HEAD against
/// `LFM2-350M-Extract-GGUF/Q4_0` with the constants below. See the
/// module docstring for the regeneration command.
///
/// All sixteen entries are the same token id — that's the LFM2-Extract
/// base model's actual greedy behavior on this short non-extract prompt
/// at T=0 (argmax cycles on a single high-probability token once the
/// post-prompt state stabilizes). Not a bug; it's still a valid
/// correctness oracle because (a) any wrong-output BLAS bug would
/// shift the argmax to a different token id, and (b) OpenBLAS and
/// Accelerate should agree on the SGEMM result at IEEE-754 precision
/// for inputs of this size, so cross-implementation drift shouldn't
/// flip the argmax.
const EXPECTED: &[u32] = &[
    856, 856, 856, 856, 856, 856, 856, 856, 856, 856, 856, 856, 856, 856, 856, 856,
];

const PROMPT: &str = "The capital of France is";
const MAX_TOKENS: u32 = 16;
const SEED: u64 = 0;
const CTX: usize = 256;

#[test]
#[ignore = "downloads ~210 MB; set WICK_TEST_DOWNLOAD=1 and pass --ignored"]
fn blas_prefill_matches_aarch64_reference() {
    if std::env::var("WICK_TEST_DOWNLOAD").is_err() {
        eprintln!("skipping: WICK_TEST_DOWNLOAD not set");
        return;
    }

    let repo = BundleRepo::new(common::download::cache_dir());
    let engine = WickEngine::from_bundle_id(
        "LFM2-350M-Extract-GGUF",
        "Q4_0",
        EngineConfig {
            context_size: CTX,
            backend: BackendPreference::Cpu,
            bundle_repo: Some(repo),
            ..Default::default()
        },
    )
    .expect("load engine");

    // `ubatch_size: 0` (monolithic prefill) → forward_prefill with
    // n = prompt-len → batched path → BLAS arm. Larger ubatch sizes
    // would still hit BLAS but the monolithic call exercises the
    // largest n the prompt allows, which is the layout most likely
    // to surface an SGEMM stride bug.
    let cfg = SessionConfig {
        seed: Some(SEED),
        ubatch_size: 0,
        ..Default::default()
    };
    let mut session = engine.new_session(cfg);
    session.append_text(PROMPT).expect("append prompt");

    struct Collect(Vec<u32>);
    impl ModalitySink for Collect {
        fn on_text_tokens(&mut self, t: &[u32]) {
            self.0.extend_from_slice(t);
        }
        fn on_done(&mut self, _reason: FinishReason) {}
    }
    let mut sink = Collect(Vec::new());
    session
        .generate(
            &GenerateOpts {
                max_tokens: MAX_TOKENS,
                temperature: 0.0,
                top_p: 1.0,
                top_k: 1,
                repetition_penalty: 1.0,
                stop_tokens: Vec::new(),
                flush_every_tokens: 1,
                flush_every_ms: 0,
            },
            &mut sink,
        )
        .expect("generate");

    let tokens = sink.0;
    println!("tokens={tokens:?}");

    if EXPECTED.is_empty() {
        panic!(
            "EXPECTED is empty — first run; capture the printed `tokens=[...]` \
             above and paste it into EXPECTED in this file."
        );
    }
    assert_eq!(
        tokens, EXPECTED,
        "BLAS produced different tokens than the aarch64+blas reference. \
         If the LFM2 fixture or greedy constants changed, regenerate \
         EXPECTED per the module docstring."
    );
}
