//! HTTP downloader with streaming SHA-256 hashing + atomic cache writes.
//!
//! Used by `BundleRepo` to fetch manifest / GGUF files from a remote URL
//! into a local cache. Gated behind the `remote` feature so wasm and
//! minimal-footprint builds don't pull `reqwest`.
//!
//! ## Integrity policy
//!
//! Downloaded bytes are hashed on the fly (SHA-256) and compared against
//! one of:
//!
//! 1. **Caller-supplied** `expected_sha256` on `BundleRepo::resolve_url`.
//!    Lets a caller (e.g. a manifest with per-file hashes) pin an exact
//!    value regardless of what the server advertises.
//! 2. **`X-Linked-Etag`**: HuggingFace serves LFS objects with an
//!    `X-Linked-Etag: sha256:<hex>` header. **Critical detail:** this
//!    header is only present on the first-hop response (the 302
//!    redirect from `huggingface.co`). The final CDN response after
//!    the redirect carries a different `ETag` that is the CAS storage
//!    key, NOT the file's content SHA-256. [`head_info`] therefore
//!    uses a no-redirect client so it reads headers from the origin's
//!    302. `BundleRepo::resolve_url` then threads the captured
//!    linked-etag into [`download_to`] as `expected_sha256`, so the
//!    cache-miss path is integrity-verified even when the caller
//!    didn't pin a hash.
//!
//! When neither a caller hash nor a server etag is available, the
//! download succeeds without hash verification (HTTPS still protects
//! transport; callers can tighten by plumbing an explicit hash).
//!
//! On mismatch the partial file is deleted and the caller receives
//! `WickError::Backend(…)` describing expected vs. actual.
//!
//! ## Sidecar hash files
//!
//! After a successful download, the computed SHA-256 is persisted to
//! `<dest>.sha256` (just the hex digest, no trailing newline). On cache
//! hits, `BundleRepo::resolve_url` can read the sidecar and compare it
//! against the server's `X-Linked-Etag` in O(1) instead of re-hashing
//! the whole file (which is an I/O + CPU tax on every resolve for
//! multi-GB GGUFs). Missing or mismatched sidecars fall back to a full
//! `sha256_file` pass, which also repairs the sidecar on success.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Duration;

use reqwest::blocking::Client;
use sha2::{Digest, Sha256};

use crate::session::WickError;

/// HTTP timeout for the GET request. Downloads run in a single shot;
/// if a 10-minute window isn't enough for a 10 GB+ shard on a slow
/// connection the caller should split the download itself.
const DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(600);

/// HTTP timeout for HEAD probes. A slow HEAD means the server is
/// struggling; we'd rather fall back to a best-effort cache reuse
/// than block the caller for minutes.
const HEAD_TIMEOUT: Duration = Duration::from_secs(30);

/// HEAD-probe result.
pub(crate) struct HeadInfo {
    /// Expected total bytes. `None` means the header was absent or the
    /// request failed.
    pub content_length: Option<u64>,
    /// Content-addressed hash from `X-Linked-Etag: sha256:<hex>`. HF
    /// and LFS-compatible CDNs set this; origin servers usually don't.
    pub linked_sha256: Option<String>,
}

/// Sidecar filename for a given cache entry. `<dest>.sha256` holds the
/// bare hex digest (64 chars, no newline). See module docs for the
/// rationale — it turns a multi-GB rehash into a single small read on
/// every cache hit.
pub(crate) fn sidecar_path(dest: &Path) -> PathBuf {
    let mut s = dest.as_os_str().to_owned();
    s.push(".sha256");
    PathBuf::from(s)
}

/// Read a previously-persisted sidecar hex digest, if any. Returns
/// `None` on missing file, I/O error, or invalid content — callers
/// treat `None` as "full rehash required."
pub(crate) fn read_sidecar(dest: &Path) -> Option<String> {
    let text = fs::read_to_string(sidecar_path(dest)).ok()?;
    let hex = text.trim();
    if hex.len() == 64 && hex.chars().all(|c| c.is_ascii_hexdigit()) {
        Some(hex.to_ascii_lowercase())
    } else {
        None
    }
}

/// Persist `sha256_hex` alongside `dest` as a sidecar file. Best-effort
/// — a failure to write just means the next cache hit pays a rehash.
pub(crate) fn write_sidecar(dest: &Path, sha256_hex: &str) {
    let _ = fs::write(sidecar_path(dest), sha256_hex);
}

/// Issue a `HEAD` for `url` using `client` and extract size +
/// linked-etag. Swallows network errors into `None` fields — the caller
/// decides how strict to be.
///
/// **The `client` passed here MUST be configured with redirects
/// disabled.** HuggingFace serves `X-Linked-Etag` only on the first
/// response (the 302 redirect to the CDN); a redirect-following client
/// surfaces the CDN's unrelated `ETag` instead. `BundleRepo::new`
/// constructs a dedicated no-redirect client for this reason. On a
/// 3xx response from the non-following client we still read the
/// headers (that's the whole point); on any non-success, non-3xx
/// response we conservatively return `None` fields.
pub(crate) fn head_info(client: &Client, url: &str) -> HeadInfo {
    let none = HeadInfo {
        content_length: None,
        linked_sha256: None,
    };
    // Chain `.error_for_status()` so 4xx/5xx responses are rejected
    // (we don't trust headers on an error page). 3xx passes through —
    // which is the whole point, since HF's `X-Linked-Etag` lives on
    // the 302.
    let Ok(resp) = client
        .head(url)
        .timeout(HEAD_TIMEOUT)
        .send()
        .and_then(|r| r.error_for_status())
    else {
        return none;
    };
    // HF sets `x-linked-size` on the first hop; prefer that over
    // `Content-Length` because `Content-Length` on a 302 reflects the
    // redirect body (zero or a tiny HTML stub), not the file size.
    let headers = resp.headers();
    let content_length = headers
        .get("x-linked-size")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok())
        .or_else(|| {
            headers
                .get(reqwest::header::CONTENT_LENGTH)
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok())
        });
    let linked_sha256 = extract_linked_sha256(headers);
    HeadInfo {
        content_length,
        linked_sha256,
    }
}

/// Pull `X-Linked-Etag: "sha256:<hex>"` from a header map. Returns
/// `None` if the header is missing or not an `sha256:` scheme.
fn extract_linked_sha256(headers: &reqwest::header::HeaderMap) -> Option<String> {
    headers
        .get("x-linked-etag")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.trim_matches('"'))
        .and_then(|s| s.strip_prefix("sha256:"))
        .map(|h| h.to_ascii_lowercase())
}

/// SHA-256 a file that's already on disk. Used to verify a cached
/// entry when callers want stronger than size-only assurance and no
/// sidecar is available.
pub(crate) fn sha256_file(path: &Path) -> io::Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    io::copy(&mut file, &mut hasher)?;
    Ok(hex_encode(&hasher.finalize()))
}

/// Stream `url` into `dest` atomically + verify the content hash +
/// persist a sidecar hash file.
///
/// Writes to `<dest>.partial.<pid>.<unique>` first and renames on
/// success. On integrity failure the partial is deleted.
/// `expected_sha256` overrides any server-provided `X-Linked-Etag`.
///
/// `progress`, when `Some`, is called periodically during the byte
/// stream — at most once per ~256 KB written, plus one final
/// callback at end-of-stream with the total bytes written. `None`
/// makes downloads silent (and skips the writer-wrapping overhead).
pub(crate) fn download_to(
    client: &Client,
    url: &str,
    dest: &Path,
    expected_sha256: Option<&str>,
    progress: Option<&dyn crate::bundle::DownloadProgress>,
) -> Result<(), WickError> {
    let mut resp = client
        .get(url)
        .timeout(DOWNLOAD_TIMEOUT)
        .send()
        .map_err(|e| WickError::Backend(format!("GET {url}: {e}")))?
        .error_for_status()
        .map_err(|e| WickError::Backend(format!("GET {url}: {e}")))?;

    // Capture before consuming the response into io::copy; `progress`
    // wants this in every call so the consumer can compute a percent.
    let total_bytes = resp.content_length();

    // Server-side hash fallback when the caller didn't pin one.
    let server_hash = extract_linked_sha256(resp.headers());
    let expected = expected_sha256
        .map(|s| s.to_ascii_lowercase())
        .or(server_hash);

    // Unique temp suffix so two concurrent processes (or threads)
    // fetching the same URL don't race on one `.partial` name. Random
    // tail uses SystemTime nanos + thread id — cheap, no RNG dep.
    let mut partial_name = dest.as_os_str().to_owned();
    partial_name.push(format!(
        ".partial.{}.{}",
        std::process::id(),
        unique_suffix()
    ));
    let partial = PathBuf::from(partial_name);

    // Compute-and-close in an inner scope so the file handle is
    // dropped before any cleanup or rename. On Windows, `fs::remove_file`
    // (and `fs::rename` onto the destination) can fail if the handle
    // is still open — we must release it before touching the partial
    // path again. POSIX tolerates unlink-while-open, but closing early
    // is strictly safer.
    let copy_result: Result<String, WickError> = {
        let mut file = fs::File::create(&partial)?;
        let mut hashing = HashingWriter {
            inner: &mut file,
            hasher: Sha256::new(),
        };
        // Wrap the hashing writer to count + report bytes if a
        // progress callback is attached. The wrapper's own write impl
        // forwards to `hashing` then conditionally invokes the
        // callback with throttling. When `progress` is None the
        // wrapper is essentially a thin forwarder; the per-write
        // overhead is one Option-check + a u64 add. Scoped so the
        // mutable borrow on `hashing` ends before we finalize the
        // hasher + emit the end-of-stream callback below.
        let final_bytes = {
            let mut counting = ProgressingWriter::new(&mut hashing, progress, url, total_bytes);
            io::copy(&mut resp, &mut counting)
                .map_err(|e| WickError::Backend(format!("write {}: {e}", partial.display())))?;
            counting.bytes_written
        };
        if let Some(p) = progress {
            // Final 100% callback so consumers can flip a UI from
            // "downloading" to "verifying" / "done" deterministically.
            // The throttled in-loop callbacks may stop at e.g.
            // bytes - 256KB; this guarantees the last reported value
            // matches the actual stream length.
            p.on_progress(url, final_bytes, total_bytes);
        }
        let digest = hashing.hasher.finalize();
        file.sync_all()?;
        Ok(hex_encode(&digest))
    };
    let actual_hex = match copy_result {
        Ok(h) => h,
        Err(e) => {
            // File handle is out of scope now — safe to unlink on Windows.
            // Mid-stream failure must not leave a `.partial.<pid>.<hash>`
            // behind; unique suffixes mean a retry won't overwrite.
            let _ = fs::remove_file(&partial);
            return Err(e);
        }
    };

    if let Some(exp) = expected.as_deref() {
        if exp != actual_hex {
            let _ = fs::remove_file(&partial);
            return Err(WickError::Backend(format!(
                "integrity check failed for {url}: expected sha256:{exp}, got sha256:{actual_hex}"
            )));
        }
    }

    // `fs::rename` on Windows fails if `dest` exists (POSIX would
    // silently replace). Remove first so cache invalidation + re-
    // download works cross-platform. If the rename itself fails
    // (cross-filesystem move, permission issue, etc.), clean up the
    // partial before propagating — otherwise each retry leaves yet
    // another `.partial.<pid>.<unique>` file on disk since the
    // unique suffix means retries never overwrite.
    let _ = fs::remove_file(dest);
    if let Err(e) = fs::rename(&partial, dest) {
        let _ = fs::remove_file(&partial);
        return Err(e.into());
    }
    // Persist the sidecar hash so subsequent cache hits can skip
    // rehashing a multi-GB GGUF. Best-effort; write failure only
    // costs us one rehash on the next resolve.
    write_sidecar(dest, &actual_hex);
    Ok(())
}

/// Process-unique-ish suffix for the temp download filename. Combines
/// SystemTime nanos with a thread ID hash — cheap and avoids pulling a
/// dedicated RNG dep through just for this.
fn unique_suffix() -> u64 {
    use std::hash::{Hash, Hasher};
    use std::time::SystemTime;

    let mut h = std::collections::hash_map::DefaultHasher::new();
    std::thread::current().id().hash(&mut h);
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
        .hash(&mut h);
    h.finish()
}

/// `io::Write` adapter that fans bytes into both the wrapped writer
/// and a running SHA-256 digest. Avoids a second pass over the file
/// after writing.
struct HashingWriter<'a, W: io::Write> {
    inner: &'a mut W,
    hasher: Sha256,
}

impl<W: io::Write> io::Write for HashingWriter<'_, W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.hasher.update(&buf[..n]);
        Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

/// Counts bytes written + (optionally) reports them to a
/// `DownloadProgress` callback. Wraps another `io::Write` so the
/// hashing + counting + (maybe) callback layers stack cleanly:
/// `io::copy(resp, ProgressingWriter -> HashingWriter -> File)`.
///
/// Throttled at `PROGRESS_THROTTLE_BYTES` granularity to avoid
/// hammering a UI's main thread on multi-MB downloads — callers
/// targeting a progress bar typically can't repaint faster than
/// ~30 Hz anyway, and a 256 KB step at 10 MB/s is ~25 callbacks
/// per second, comfortably matched.
struct ProgressingWriter<'a, W: io::Write> {
    inner: &'a mut W,
    progress: Option<&'a dyn crate::bundle::DownloadProgress>,
    url: &'a str,
    total_bytes: Option<u64>,
    bytes_written: u64,
    last_callback_at: u64,
}

const PROGRESS_THROTTLE_BYTES: u64 = 256 * 1024;

impl<'a, W: io::Write> ProgressingWriter<'a, W> {
    fn new(
        inner: &'a mut W,
        progress: Option<&'a dyn crate::bundle::DownloadProgress>,
        url: &'a str,
        total_bytes: Option<u64>,
    ) -> Self {
        Self {
            inner,
            progress,
            url,
            total_bytes,
            bytes_written: 0,
            last_callback_at: 0,
        }
    }
}

impl<W: io::Write> io::Write for ProgressingWriter<'_, W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.bytes_written += n as u64;
        if let Some(p) = self.progress
            && self.bytes_written - self.last_callback_at >= PROGRESS_THROTTLE_BYTES
        {
            p.on_progress(self.url, self.bytes_written, self.total_bytes);
            self.last_callback_at = self.bytes_written;
        }
        Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sidecar_path_appends_extension() {
        assert_eq!(
            sidecar_path(Path::new("/cache/x.gguf")),
            PathBuf::from("/cache/x.gguf.sha256")
        );
    }

    /// `ProgressingWriter` throttles its callback to ~PROGRESS_THROTTLE_BYTES
    /// granularity. Driving it with three writes (small / large /
    /// small) verifies:
    /// - Writes that don't cross the threshold don't trigger a
    ///   callback (first 1 KB → no event).
    /// - A single write that crosses the threshold triggers exactly
    ///   one event with the cumulative byte count (300 KB → event at
    ///   301 KB total).
    /// - Subsequent small writes don't re-trigger until the next
    ///   threshold crossing (final 1 KB → no event).
    /// - The final emitted byte count matches the high-water mark.
    #[test]
    fn progressing_writer_throttles_callback() {
        use crate::bundle::DownloadProgress;
        use std::io::Write as _;
        use std::sync::Mutex;

        #[derive(Debug, Default)]
        struct Recorder {
            calls: Mutex<Vec<(String, u64, Option<u64>)>>,
        }
        impl DownloadProgress for Recorder {
            fn on_progress(&self, url: &str, bytes: u64, total: Option<u64>) {
                self.calls
                    .lock()
                    .unwrap()
                    .push((url.to_string(), bytes, total));
            }
        }

        let recorder = Recorder::default();
        let mut sink = std::io::sink();
        let mut writer = ProgressingWriter::new(
            &mut sink,
            Some(&recorder as &dyn DownloadProgress),
            "https://example.com/foo.gguf",
            Some(1024 * 1024),
        );

        // Below threshold: no callback.
        writer.write_all(&[0u8; 1024]).unwrap();
        assert_eq!(recorder.calls.lock().unwrap().len(), 0);

        // Crossing threshold (256 KB) inside one write: one callback,
        // bytes = 1 KB + 300 KB = 308224 = 301 KiB.
        writer.write_all(&[0u8; 300 * 1024]).unwrap();
        let calls_after_big = recorder.calls.lock().unwrap().clone();
        assert_eq!(calls_after_big.len(), 1);
        assert_eq!(calls_after_big[0].0, "https://example.com/foo.gguf");
        assert_eq!(calls_after_big[0].1, (1 + 300) * 1024);
        assert_eq!(calls_after_big[0].2, Some(1024 * 1024));

        // Below the next threshold: no new callback.
        writer.write_all(&[0u8; 1024]).unwrap();
        assert_eq!(recorder.calls.lock().unwrap().len(), 1);

        // bytes_written reflects everything written.
        assert_eq!(writer.bytes_written, (1 + 300 + 1) * 1024);
    }

    /// `progress = None` → ProgressingWriter is a thin forwarder, no
    /// allocation, no calls. Sanity-checks the `if let Some(p)` branch
    /// stays cold so the no-progress path doesn't pay for it.
    #[test]
    fn progressing_writer_with_none_progress_is_silent() {
        use std::io::Write as _;
        let mut sink = std::io::sink();
        let mut writer = ProgressingWriter::new(&mut sink, None, "https://example.com/x", None);
        writer.write_all(&[0u8; 1024 * 1024]).unwrap();
        assert_eq!(writer.bytes_written, 1024 * 1024);
        // Nothing to assert beyond "didn't panic / didn't dispatch
        // through a None pointer".
    }

    #[test]
    fn read_sidecar_accepts_valid_hex() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("x.gguf");
        let expected = "a".repeat(64);
        fs::write(sidecar_path(&dest), &expected).unwrap();
        assert_eq!(read_sidecar(&dest).as_deref(), Some(expected.as_str()));
    }

    #[test]
    fn read_sidecar_normalizes_case_and_whitespace() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("x.gguf");
        let hex = "AbCdEf".repeat(8) + "AbCdEfAbCdEfAbCd";
        assert_eq!(hex.len(), 64);
        fs::write(sidecar_path(&dest), format!("  {hex}  \n")).unwrap();
        assert_eq!(read_sidecar(&dest), Some(hex.to_ascii_lowercase()));
    }

    #[test]
    fn read_sidecar_rejects_wrong_length() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("x.gguf");
        fs::write(sidecar_path(&dest), "deadbeef").unwrap();
        assert!(read_sidecar(&dest).is_none());
    }

    #[test]
    fn read_sidecar_rejects_non_hex() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("x.gguf");
        let garbage = "z".repeat(64);
        fs::write(sidecar_path(&dest), garbage).unwrap();
        assert!(read_sidecar(&dest).is_none());
    }

    #[test]
    fn read_sidecar_missing_file_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("nonexistent.gguf");
        assert!(read_sidecar(&dest).is_none());
    }

    #[test]
    fn sha256_file_round_trips_known_input() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("x.bin");
        fs::write(&p, b"hello").unwrap();
        // Known SHA-256("hello") = 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
        assert_eq!(
            sha256_file(&p).unwrap(),
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
    }
}
