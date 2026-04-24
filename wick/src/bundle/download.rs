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
//! 1. **Caller-supplied** `expected_sha256`. Lets future callers (e.g.
//!    PR B's manifest-level hashes) enforce a specific value.
//! 2. **`X-Linked-Etag`**: HuggingFace serves LFS objects with an
//!    `X-Linked-Etag: sha256:<hex>` header — content-addressed, stable
//!    across revisions. When neither the caller supplies a hash nor the
//!    server provides this header, we succeed without verification
//!    (HTTPS still protects transport; callers can tighten this by
//!    plumbing an explicit hash through).
//!
//! On mismatch the `.partial` file is deleted and the caller receives
//! `WickError::Backend(…)` describing expected vs. actual.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Duration;

use sha2::{Digest, Sha256};

use crate::session::WickError;

/// HTTP timeout for the GET request. Downloads run in a single shot;
/// if a 10-minute window isn't enough for a 10 GB+ shard on a slow
/// connection the caller should split the download itself.
const DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(600);

/// HTTP timeout for HEAD probes. Short because a slow HEAD means the
/// server is struggling and we'd rather fall back to a best-effort
/// cache reuse than block the caller.
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

/// Issue a `HEAD` for `url` and extract size + linked-etag. Swallows
/// network errors into `None` fields — the caller decides how strict
/// to be.
pub(crate) fn head_info(url: &str) -> HeadInfo {
    let none = HeadInfo {
        content_length: None,
        linked_sha256: None,
    };
    let Ok(client) = reqwest::blocking::Client::builder()
        .timeout(HEAD_TIMEOUT)
        .build()
    else {
        return none;
    };
    let Ok(resp) = client.head(url).send() else {
        return none;
    };
    let content_length = resp
        .headers()
        .get(reqwest::header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok());
    // `X-Linked-Etag` on HF is formatted `"sha256:<hex>"` with quotes.
    // Strip quotes + scheme prefix to yield the hex digest.
    let linked_sha256 = resp
        .headers()
        .get("x-linked-etag")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.trim_matches('"'))
        .and_then(|s| s.strip_prefix("sha256:"))
        .map(|h| h.to_ascii_lowercase());
    HeadInfo {
        content_length,
        linked_sha256,
    }
}

/// SHA-256 a file that's already on disk. Used to verify a cached
/// entry when callers want stronger than size-only assurance.
pub(crate) fn sha256_file(path: &Path) -> io::Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    io::copy(&mut file, &mut hasher)?;
    Ok(hex_encode(&hasher.finalize()))
}

/// Stream `url` into `dest` atomically + verify the content hash.
///
/// Writes to `<dest>.partial` first and renames on success. On integrity
/// failure the partial is deleted. `expected_sha256` overrides any
/// server-provided `X-Linked-Etag`.
pub(crate) fn download_to(
    url: &str,
    dest: &Path,
    expected_sha256: Option<&str>,
) -> Result<(), WickError> {
    let client = reqwest::blocking::Client::builder()
        .timeout(DOWNLOAD_TIMEOUT)
        .build()
        .map_err(|e| WickError::Backend(format!("build http client: {e}")))?;

    let mut resp = client
        .get(url)
        .send()
        .map_err(|e| WickError::Backend(format!("GET {url}: {e}")))?
        .error_for_status()
        .map_err(|e| WickError::Backend(format!("GET {url}: {e}")))?;

    // Server-side hash fallback when the caller didn't pin one.
    let server_hash = resp
        .headers()
        .get("x-linked-etag")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.trim_matches('"'))
        .and_then(|s| s.strip_prefix("sha256:"))
        .map(|h| h.to_ascii_lowercase());
    let expected = expected_sha256
        .map(|s| s.to_ascii_lowercase())
        .or(server_hash);

    // Unique temp suffix so two concurrent processes (or threads)
    // fetching the same URL don't race on one `.partial` name. The
    // random tail uses SystemTime nanos + a PID — cheap, doesn't drag
    // in a new RNG dep, and collision probability is negligible for the
    // realistic case (human-initiated concurrent bundle downloads).
    let mut partial_name = dest.as_os_str().to_owned();
    partial_name.push(format!(
        ".partial.{}.{}",
        std::process::id(),
        unique_suffix()
    ));
    let partial = PathBuf::from(partial_name);

    // Scope so the file is closed before rename.
    let actual_hex = {
        let mut file = fs::File::create(&partial)?;
        let mut hashing = HashingWriter {
            inner: &mut file,
            hasher: Sha256::new(),
        };
        io::copy(&mut resp, &mut hashing)
            .map_err(|e| WickError::Backend(format!("write {}: {e}", partial.display())))?;
        let digest = hashing.hasher.finalize();
        file.sync_all()?;
        hex_encode(&digest)
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
    // download works cross-platform. Best-effort: if remove fails
    // (e.g., file never existed), the rename below still handles the
    // common path.
    let _ = fs::remove_file(dest);
    fs::rename(&partial, dest)?;
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

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}
