//! Image-input loader shared by `wick run --image`, the line-REPL `/image`
//! command, and the TUI `/image` command.
//!
//! Resolves a user-supplied argument — filesystem path or `http(s)://` URL —
//! to bytes, with the same 50 MB cap and same non-regular-file rejection
//! across all three entry points. URL fetches use `reqwest::blocking` with a
//! streaming size guard so an adversarial / mis-Content-Length'd server can't
//! exhaust memory.

use std::io::Read;

use anyhow::{Context, Result, bail, ensure};

/// Default byte cap applied at every `/image` and `--image` entry point.
/// LFM2-VL's 262 144-pixel input band is ~1 MB raw RGB; 50 MB rejects clearly
/// malicious / accidental multi-GB inputs without ever clipping a real image.
pub(crate) const MAX_IMAGE_BYTES: u64 = 50 * 1024 * 1024;

/// Per-fetch HTTP timeout. URL fetches block the foreground REPL/TUI; users
/// expect a few-second ceiling, not minute-long stalls behind a slow server.
const HTTP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// Cap on redirect chain length so a redirect loop (or chain to a malicious
/// host) can't burn the entire timeout window.
const MAX_REDIRECTS: usize = 10;

/// Streaming read chunk size. 64 KiB is the standard Linux page-aligned
/// network read buffer; larger doesn't speed up TLS reads measurably and
/// smaller bloats syscall count.
const READ_CHUNK_BYTES: usize = 64 * 1024;

/// Returns true iff `arg` looks like an HTTP(S) URL. Anything else (including
/// `file://`, relative paths, absolute paths, and bare strings) is treated
/// as a filesystem path by [`load`].
pub(crate) fn looks_like_url(arg: &str) -> bool {
    arg.starts_with("http://") || arg.starts_with("https://")
}

/// Resolves `arg` to image bytes, capped at `max_bytes`. URL inputs fetch
/// over HTTP; everything else reads from disk.
pub(crate) fn load(arg: &str, max_bytes: u64) -> Result<Vec<u8>> {
    if looks_like_url(arg) {
        load_url(arg, max_bytes)
    } else {
        load_path(arg, max_bytes)
    }
}

fn load_path(path: &str, max_bytes: u64) -> Result<Vec<u8>> {
    let meta = std::fs::metadata(path).with_context(|| format!("stat image source {path}"))?;
    ensure!(meta.is_file(), "image source {path}: not a regular file");
    ensure!(
        meta.len() <= max_bytes,
        "image source {path}: file is {} bytes, larger than the {} byte cap",
        meta.len(),
        max_bytes
    );
    std::fs::read(path).with_context(|| format!("reading image source {path}"))
}

fn load_url(url: &str, max_bytes: u64) -> Result<Vec<u8>> {
    let client = reqwest::blocking::Client::builder()
        .timeout(HTTP_TIMEOUT)
        .redirect(reqwest::redirect::Policy::limited(MAX_REDIRECTS))
        .user_agent(concat!("wick-cli/", env!("CARGO_PKG_VERSION")))
        .build()
        .context("build HTTP client for image fetch")?;

    let mut response = client
        .get(url)
        .send()
        .with_context(|| format!("fetch image source {url}"))?
        .error_for_status()
        .with_context(|| format!("HTTP error from image source {url}"))?;

    // Cheap pre-flight: if the server announces a length over the cap, fail
    // before reading any body bytes. Honest servers always set this; lying
    // or omitting servers fall through to the streaming guard below.
    if let Some(content_length) = response.content_length() {
        ensure!(
            content_length <= max_bytes,
            "image source {url}: Content-Length {content_length} exceeds the {max_bytes} byte cap"
        );
    }

    let initial_capacity =
        usize::try_from(response.content_length().unwrap_or(0).min(max_bytes)).unwrap_or(0);
    let mut buf: Vec<u8> = Vec::with_capacity(initial_capacity);
    let mut chunk = [0u8; READ_CHUNK_BYTES];
    loop {
        let n = response
            .read(&mut chunk)
            .with_context(|| format!("read body from image source {url}"))?;
        if n == 0 {
            break;
        }
        let new_len = buf.len().saturating_add(n);
        if new_len as u64 > max_bytes {
            bail!("image source {url}: response exceeds the {max_bytes} byte cap");
        }
        buf.extend_from_slice(&chunk[..n]);
    }
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    #[test]
    fn classifier_accepts_http_and_https() {
        assert!(looks_like_url("http://example.com/a.jpg"));
        assert!(looks_like_url("https://example.com/a.jpg"));
        assert!(looks_like_url("https://"));
        assert!(looks_like_url("http://"));
    }

    #[test]
    fn classifier_rejects_non_http_inputs() {
        assert!(!looks_like_url(""));
        assert!(!looks_like_url("/abs/path.jpg"));
        assert!(!looks_like_url("./rel/path.jpg"));
        assert!(!looks_like_url("path.jpg"));
        assert!(!looks_like_url("file:///abs/path.jpg"));
        assert!(!looks_like_url("ftp://example.com/x"));
        assert!(!looks_like_url("Http://example.com/x")); // case-sensitive
        assert!(!looks_like_url("httpfoo"));
    }

    #[test]
    fn load_path_reads_file_bytes() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(b"hello world").unwrap();
        let bytes = load(f.path().to_str().unwrap(), MAX_IMAGE_BYTES).unwrap();
        assert_eq!(bytes, b"hello world");
    }

    #[test]
    fn load_path_rejects_oversize() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(&[0u8; 32]).unwrap();
        let err = load(f.path().to_str().unwrap(), 16).unwrap_err();
        let s = format!("{err:#}");
        assert!(s.contains("byte cap"), "unexpected error: {s}");
    }

    #[test]
    fn load_path_rejects_non_regular() {
        // /dev/null exists on every supported target and is not a regular
        // file. /dev/zero also works but reads unbounded data — exactly the
        // hazard this guard prevents.
        let err = load("/dev/null", MAX_IMAGE_BYTES).unwrap_err();
        let s = format!("{err:#}");
        assert!(s.contains("not a regular file"), "unexpected error: {s}");
    }

    #[test]
    fn load_path_missing_file_errors_with_context() {
        let err = load("/this/path/should/not/exist.jpg", MAX_IMAGE_BYTES).unwrap_err();
        let s = format!("{err:#}");
        assert!(s.contains("stat image source"), "unexpected error: {s}");
    }
}
