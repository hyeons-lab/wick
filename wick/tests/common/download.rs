//! Cached HTTP download helper for gated integration tests.
//!
//! Downloads a file from a given URL to a persistent cache directory and
//! returns the local path. Subsequent calls skip the network if the file
//! exists AND matches the server-advertised byte length (`HEAD` probe).
//!
//! Cache root: `$WICK_TEST_MODELS_DIR` if set, else `target/tmp/wick-test-models`
//! under the workspace root. A `target/tmp/` subdir is chosen (rather than
//! `target/debug/tmp/` or similar) so the CI cache stanza in `.github/`
//! can key on a single stable path across debug/release invocations.
//!
//! Used only by tests gated on `WICK_TEST_DOWNLOAD=1` — never runs in a
//! default `cargo test` invocation.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// How long to wait on HEAD / GET before giving up. Tests should fail
/// loudly rather than hang CI runners.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

/// Returns the resolved cache directory, creating it if missing.
fn cache_dir() -> PathBuf {
    if let Ok(override_path) = std::env::var("WICK_TEST_MODELS_DIR") {
        let p = PathBuf::from(override_path);
        fs::create_dir_all(&p).expect("create WICK_TEST_MODELS_DIR");
        return p;
    }
    // Workspace-root-relative. `CARGO_MANIFEST_DIR` points at the crate
    // (`wick/`), so one dir up is the workspace root.
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest
        .parent()
        .expect("CARGO_MANIFEST_DIR has no parent")
        .join("target")
        .join("tmp")
        .join("wick-test-models");
    fs::create_dir_all(&root).expect("create cache_dir");
    root
}

/// Ask the server for the expected content length via a `HEAD` request.
/// Returns `None` if the header is missing or the request failed — callers
/// treat that as "can't verify, assume cached file is OK if it exists."
fn head_content_length(url: &str) -> Option<u64> {
    let client = reqwest::blocking::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .build()
        .ok()?;
    let resp = client.head(url).send().ok()?;
    // HF redirects resolve/main URLs to a CDN — `reqwest` follows redirects
    // by default, so the final `Content-Length` is the object's real size.
    resp.headers()
        .get(reqwest::header::CONTENT_LENGTH)?
        .to_str()
        .ok()?
        .parse()
        .ok()
}

/// Stream `url` into `dest`. Writes to a sibling `<dest>.partial` file
/// first and renames on success, so a failure mid-transfer leaves no
/// half-written file that a later `ensure_cached` could mistake for a
/// finished download when the HEAD probe is also unavailable.
fn download_to(url: &str, dest: &Path) -> io::Result<()> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(600))
        .build()
        .map_err(io::Error::other)?;
    let mut resp = client
        .get(url)
        .send()
        .map_err(io::Error::other)?
        .error_for_status()
        .map_err(io::Error::other)?;

    let mut partial = dest.as_os_str().to_owned();
    partial.push(".partial");
    let partial = PathBuf::from(partial);
    // Best-effort remove of a leftover `.partial` from a prior crash.
    let _ = fs::remove_file(&partial);

    // Scope the file handle so it's closed before the rename.
    {
        let mut file = fs::File::create(&partial)?;
        io::copy(&mut resp, &mut file)?;
        file.sync_all()?;
    }
    fs::rename(&partial, dest)?;
    Ok(())
}

/// Ensure `url` is present under the cache as `filename`, returning the
/// local path.
///
/// - If the file exists AND its size matches `HEAD`'s `Content-Length`,
///   the download is skipped.
/// - If the file exists but the size differs (partial previous run, or
///   upstream file changed), it's re-downloaded.
/// - If `HEAD` fails (offline, server flaky) and the file exists, we
///   reuse whatever's there — intentional: CI cache hit shouldn't be
///   defeated by a transient upstream blip.
///
/// Panics on unrecoverable errors (no file + download failed). Tests that
/// call this are already gated on `WICK_TEST_DOWNLOAD=1`, so a panic here
/// is the correct failure mode.
pub fn ensure_cached(url: &str, filename: &str) -> PathBuf {
    let path = cache_dir().join(filename);
    let expected = head_content_length(url);

    if path.exists() {
        let actual = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        match expected {
            Some(exp) if exp == actual => return path,
            Some(exp) => {
                eprintln!("cached {filename} size {actual} != upstream {exp}; re-downloading");
            }
            None => {
                // Couldn't check — reuse the cached file rather than
                // burning bandwidth on a possibly-stale upstream probe.
                return path;
            }
        }
    }

    eprintln!("downloading {url} → {}", path.display());
    download_to(url, &path).unwrap_or_else(|e| panic!("download {url}: {e}"));
    path
}
