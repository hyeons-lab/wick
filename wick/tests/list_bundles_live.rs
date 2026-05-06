//! Live smoke test for `wick::bundle::list_leap_bundles` against the
//! real `huggingface.co/LiquidAI/LeapBundles` API.
//!
//! Proves end-to-end that the model-info endpoint still returns the
//! `siblings` shape we parse, and that at least one well-known bundle
//! is present. The unit test in `bundle::tests` covers the parser
//! against a synthetic body; this one catches schema drift on the
//! upstream API.
//!
//! Gating: `#[ignore]` + `WICK_TEST_DOWNLOAD=1` env var, matching
//! `bundle_download.rs`. Opt-in:
//!
//! ```sh
//! WICK_TEST_DOWNLOAD=1 cargo test -p wick --features remote \
//!     --test list_bundles_live -- --ignored
//! ```

#![cfg(feature = "remote")]

#[test]
#[ignore = "hits the real HF API; set WICK_TEST_DOWNLOAD=1 and pass --ignored"]
fn list_bundles_against_live_hf() {
    if std::env::var("WICK_TEST_DOWNLOAD").is_err() {
        eprintln!("skipping: WICK_TEST_DOWNLOAD not set");
        return;
    }
    let entries = wick::bundle::list_leap_bundles().expect("list_leap_bundles failed");
    assert!(
        !entries.is_empty(),
        "expected at least one bundle in LeapBundles"
    );

    // Sanity: at least one well-known bundle Liquid has shipped for
    // a long time should be present. If this name ever gets renamed
    // upstream, update the test rather than chasing the regression.
    let want = "LFM2.5-1.2B-Instruct-GGUF";
    let found = entries.iter().any(|e| e.name == want);
    assert!(
        found,
        "expected `{want}` in the catalog; got {:?}",
        entries.iter().map(|e| &e.name).collect::<Vec<_>>()
    );

    // Every entry's quants list should be non-empty (we filter on
    // `<bundle>/<quant>.json` shape, so an entry with zero quants
    // would mean the parse loop produced an empty BTreeSet — which
    // it can't, since insertion is the only path that creates the
    // set. Asserting it anyway as a regression tripwire.)
    for e in &entries {
        assert!(
            !e.quants.is_empty(),
            "bundle `{}` listed with no quants",
            e.name
        );
    }
}
