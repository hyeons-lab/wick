//! Bundle fetching + caching.
//!
//! `BundleRepo` resolves a remote URL to a file in a caller-chosen
//! cache directory. This is Phase 1.6 PR A scope: it does *not* resolve
//! bundle IDs like `"LiquidAI/LFM2-1.2B-GGUF"` to a manifest URL (PR B);
//! it takes a direct URL (typically pulled from a manifest's `files`
//! entries) and returns a local path.
//!
//! ## Caching
//!
//! Files are stored under `<store_dir>/<host>/<url-path>`, mirroring the
//! URL structure so contents are trivially inspectable and swappable
//! with a CDN mirror that preserves paths.
//!
//! ## `store_dir`, not `cache_dir`
//!
//! The directory the caller supplies is named `store_dir` on purpose.
//! On Android the consumer is expected to pass `Context.getFilesDir()`
//! (persistent storage), **not** `Context.getCacheDir()` — the latter
//! is OS-purgeable under storage pressure and would cause silent,
//! expensive re-downloads. Desktop and server callers typically pass
//! something like `~/.cache/wick/` but the crate never hardcodes a
//! default location; it's always caller-supplied.
//!
//! ## Integrity
//!
//! Each download is SHA-256'd on the fly and compared against either a
//! caller-supplied hash (via the `expected_sha256` argument to
//! [`BundleRepo::resolve_url`]) or the server's `X-Linked-Etag`
//! header (HuggingFace sets this for LFS objects — content-addressed,
//! stable across revisions). The successful hash is persisted as
//! `<dest>.sha256` alongside the cached file; subsequent cache hits
//! read the sidecar and compare it against the etag in O(1) rather
//! than re-hashing multi-GB files on every resolve. A missing or
//! stale sidecar triggers a full rehash (which also repairs the
//! sidecar on success).
//!
//! A cached file is considered valid when:
//! 1. A caller-supplied hash matches the sidecar (or full rehash
//!    fallback), or
//! 2. HEAD provides `X-Linked-Etag` and it matches the sidecar (or
//!    full rehash fallback), or
//! 3. HEAD provides only `Content-Length` and the sizes match, or
//! 4. HEAD fails entirely — reuse whatever's cached so a transient
//!    upstream blip doesn't defeat a CI cache hit.
//!
//! See [`download::head_info`] for the HEAD probe.

pub(crate) mod download;

use std::fs;
use std::path::{Path, PathBuf};

use reqwest::blocking::Client;

use crate::session::WickError;

/// Repository for remote bundle files cached to a caller-chosen
/// directory. Construction is cheap — create one per `WickEngine` at
/// most, or pass the same instance to multiple engines.
///
/// Holds two pooled `reqwest::blocking::Client`s:
/// - `http_client`: default redirect policy. Used for the `GET`
///   streaming-download path so HF's 302 to the CDN is followed
///   automatically.
/// - `head_client`: redirects **disabled**. Used for `HEAD` probes so
///   the code reads headers from HF's origin-hop 302 (which carries
///   `X-Linked-Etag`, the content SHA-256). A redirect-following
///   client would surface the CDN's unrelated `ETag` instead.
#[derive(Clone, Debug)]
pub struct BundleRepo {
    store_dir: PathBuf,
    http_client: Client,
    head_client: Client,
}

impl BundleRepo {
    /// Create a new repo rooted at `store_dir`. The directory does not
    /// need to exist yet — it will be created on the first download.
    ///
    /// Constructs two clients (see [`BundleRepo`] docs for the
    /// redirect-policy split). Both use reqwest's defaults otherwise;
    /// per-request timeouts override at the call site (30s for HEAD,
    /// 10min for GET). `Client::builder().build()` panics only on
    /// severe OS resource failure (can't create a Tokio runtime) —
    /// documented failure mode, acceptable for a process-startup path.
    pub fn new(store_dir: impl Into<PathBuf>) -> Self {
        let head_client = Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .expect("build no-redirect reqwest client");
        Self {
            store_dir: store_dir.into(),
            http_client: Client::new(),
            head_client,
        }
    }

    /// Root directory backing this repo.
    pub fn store_dir(&self) -> &Path {
        &self.store_dir
    }

    /// Resolve a remote URL to a local path, downloading if not cached.
    ///
    /// `expected_sha256`: if provided, the cached entry's hash (or the
    /// freshly-downloaded bytes' hash) must match this exactly. If
    /// `None`, integrity verification falls back to the server's
    /// `X-Linked-Etag` (when present) or a size check (when only
    /// `Content-Length` is available).
    ///
    /// ### Verification policy
    ///
    /// On cache hit:
    /// - If a sidecar `<dest>.sha256` exists, compare it against the
    ///   expected hash in O(1). This is the fast path for multi-GB
    ///   cached GGUFs.
    /// - Else fall back to `sha256_file` (full rehash) and repair the
    ///   sidecar on success.
    /// - If no hash is available anywhere, fall back to a
    ///   `Content-Length` size check.
    /// - If HEAD also fails, reuse the cached file (transient outage
    ///   shouldn't defeat a CI cache hit).
    ///
    /// On cache miss: download, hashing on the fly, verifying against
    /// `expected_sha256` or `X-Linked-Etag`. A mismatch deletes the
    /// partial and returns `WickError::Backend`. The sidecar is
    /// persisted on success.
    pub fn resolve_url(
        &self,
        url: &str,
        expected_sha256: Option<&str>,
    ) -> Result<PathBuf, WickError> {
        let dest = self.path_for_url(url)?;

        // Probe HEAD once up front (no-redirect client so HF's
        // `X-Linked-Etag` is captured from the first hop). Used both
        // to validate a cached file and — on cache miss — to hand the
        // server's advertised content hash to `download_to`, so the
        // first download is integrity-verified even when the caller
        // didn't pin a hash. Re-used across cache-hit and cache-miss
        // code paths so we issue at most one HEAD per resolve.
        let head = if expected_sha256.is_some() {
            // Caller pinned a hash — no need to consult the server.
            download::HeadInfo {
                content_length: None,
                linked_sha256: None,
            }
        } else {
            download::head_info(&self.head_client, url)
        };

        if dest.exists() && self.cache_hit_valid(&dest, url, expected_sha256, &head) {
            return Ok(dest);
        }

        // Cache miss: only now does the filesystem need to exist.
        // Deferred so cache-hit callers don't pay a `stat` on the
        // parent directory on every resolve.
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)?;
        }

        // Prefer caller's hash; else use the server's linked-etag
        // captured pre-redirect.
        let download_hash = expected_sha256
            .map(|s| s.to_ascii_lowercase())
            .or_else(|| head.linked_sha256.clone());

        tracing::info!(
            target: "wick::bundle",
            url,
            dest = %dest.display(),
            hash_source = match (expected_sha256.is_some(), head.linked_sha256.is_some()) {
                (true, _) => "caller",
                (false, true) => "x-linked-etag",
                (false, false) => "unverified",
            },
            "downloading bundle file"
        );
        download::download_to(&self.http_client, url, &dest, download_hash.as_deref())?;
        Ok(dest)
    }

    /// Decide whether an existing cached entry at `dest` is still
    /// valid. Verification prefers caller-supplied hash, then etag via
    /// sidecar, then etag via full rehash, then size, then reuse-on-
    /// HEAD-failure. Any mismatch returns `false` → caller re-downloads.
    fn cache_hit_valid(
        &self,
        dest: &Path,
        url: &str,
        expected_sha256: Option<&str>,
        head: &download::HeadInfo,
    ) -> bool {
        // Caller hash takes precedence over whatever the server
        // advertised — lets manifest-level hashes override an etag
        // that's been rotated.
        let expected_hash = expected_sha256
            .map(|s| s.to_ascii_lowercase())
            .or_else(|| head.linked_sha256.clone());

        if let Some(exp_hash) = expected_hash {
            return hash_matches(dest, url, &exp_hash);
        }

        if let Some(exp_len) = head.content_length {
            let actual = fs::metadata(dest).map(|m| m.len()).unwrap_or(0);
            if actual == exp_len {
                return true;
            }
            tracing::info!(
                target: "wick::bundle",
                url,
                expected = exp_len,
                actual,
                "cached file size mismatch; re-downloading"
            );
            return false;
        }

        // HEAD failed entirely — best-effort reuse so a transient
        // outage doesn't defeat a CI cache hit.
        true
    }

    /// Compute the on-disk cache location for `url`, rooted at
    /// `store_dir`. Mirrors `<host>/<path>` so the cache is inspectable
    /// and safely swappable with a host-preserving mirror.
    ///
    /// Rejects URLs whose host or path contain segments that could
    /// escape `store_dir` (e.g. `..`, null bytes, a bare `/` path
    /// component). This is a pre-`PathBuf::push` filter — `PathBuf`
    /// itself is not a validator; an attacker-controlled URL must not
    /// be able to write outside the cache root.
    ///
    /// The **host** is lowercased before it becomes a cache-dir name so
    /// URLs that differ only in host casing (per RFC 3986 §3.2.2,
    /// hosts are case-insensitive) share a cache entry on
    /// case-sensitive filesystems. Path segments stay as-is — they're
    /// content-addressable and different casings can legitimately
    /// resolve to different resources on the origin.
    fn path_for_url(&self, url: &str) -> Result<PathBuf, WickError> {
        let (host, path) = split_url(url)?;
        let host_lower = host.to_ascii_lowercase();
        validate_path_segment("url host", &host_lower)?;

        // Strip the leading `/` and any trailing query/fragment before
        // segmenting. `?` / `#` are URL syntax that don't belong in the
        // on-disk path. If the request actually depends on them we'd
        // need a caller to pass them through separately; today every
        // bundle URL is a clean path.
        let path_no_qs = path
            .trim_start_matches('/')
            .split(['?', '#'])
            .next()
            .unwrap_or("");
        if path_no_qs.is_empty() {
            return Err(WickError::Backend(format!(
                "url `{url}` has no path component"
            )));
        }

        let mut out = self.store_dir.clone();
        out.push(&host_lower);
        for segment in path_no_qs.split('/') {
            validate_path_segment("url path segment", segment)?;
            out.push(segment);
        }
        Ok(out)
    }
}

/// Check whether `dest` hashes to `expected_hash` (case-insensitive
/// hex), preferring the sidecar fast path. Logs a tracing event when
/// a full rehash is performed or a mismatch is detected.
fn hash_matches(dest: &Path, url: &str, expected_hash: &str) -> bool {
    let expected = expected_hash.to_ascii_lowercase();

    // Fast path: trust the sidecar. We wrote it ourselves after the
    // last successful verification, so it's at least as trustworthy
    // as the cached file itself. `read_sidecar` returns lowercase.
    if let Some(cached) = download::read_sidecar(dest) {
        if cached == expected {
            return true;
        }
        tracing::info!(
            target: "wick::bundle",
            url,
            expected = %expected,
            actual = %cached,
            "cached file sidecar hash mismatch; re-downloading"
        );
        return false;
    }

    // Slow path: full rehash. Only hits when the sidecar is absent
    // (e.g. cached before the sidecar feature shipped) or unreadable.
    // `sha256_file` returns lowercase hex. On a match, persist the
    // sidecar so the next cache hit skips straight to the fast path.
    match download::sha256_file(dest) {
        Ok(actual) if actual == expected => {
            download::write_sidecar(dest, &actual);
            true
        }
        Ok(actual) => {
            tracing::info!(
                target: "wick::bundle",
                url,
                expected = %expected,
                actual = %actual,
                "cached file hash mismatch; re-downloading"
            );
            false
        }
        Err(e) => {
            tracing::warn!(
                target: "wick::bundle",
                url,
                error = %e,
                "failed to hash cached file; re-downloading"
            );
            false
        }
    }
}

/// Strict allowlist for path segments: ASCII alphanumerics and
/// `-`, `_`, `.`. Everything else is rejected:
/// - `/`, `\` (path separators on any OS)
/// - `:` (Windows drive letters / NTFS alternate data streams)
/// - `*`, `?`, `"`, `<`, `>`, `|` (Windows-reserved)
/// - whitespace, null bytes, control chars (confusing / truncating)
/// - URL-reserved `?`, `#`, `%` (semantics-altering under URL parsing)
/// - non-ASCII (keeps paths portable across codepages; real bundle IDs
///   and URL segments in `LiquidAI/LeapBundles` are all ASCII today)
///
/// Used for both URL-derived cache-dir segments (in `path_for_url`,
/// where lax input could escape `store_dir` via `..`) and any future
/// caller that hand-assembles URLs from user input.
fn validate_path_segment(kind: &str, segment: &str) -> Result<(), WickError> {
    if segment.is_empty() {
        return Err(WickError::Backend(format!("{kind} must not be empty")));
    }
    if segment == "." || segment == ".." {
        return Err(WickError::Backend(format!(
            "{kind} `{segment}` is not a valid path component"
        )));
    }
    for ch in segment.chars() {
        let ok = ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.');
        if !ok {
            return Err(WickError::Backend(format!(
                "{kind} `{segment}` contains forbidden character {ch:?}"
            )));
        }
    }
    Ok(())
}

/// Minimal URL parser: extract `(host, path)` from `https://host/path`.
/// Scheme comparison is case-insensitive (RFC 3986 §3.1) to match the
/// case-insensitive check in `engine::is_remote_url` — otherwise a
/// `HTTPS://…` URL would be accepted by the remote-URL gate but
/// rejected here. We avoid pulling in a full `url` crate dep — this
/// is the only URL handling `wick` needs and the shape we accept is
/// narrow.
fn split_url(url: &str) -> Result<(&str, &str), WickError> {
    // Case-insensitive scheme match: find the `://` and check the
    // preceding label against our supported schemes.
    let scheme_end = url.find("://").ok_or_else(|| {
        WickError::Backend(format!("url `{url}` must start with https:// or http://"))
    })?;
    let scheme = &url[..scheme_end];
    let lower = scheme.to_ascii_lowercase();
    if lower != "http" && lower != "https" {
        return Err(WickError::Backend(format!(
            "url `{url}` must start with https:// or http://"
        )));
    }
    let after_scheme = &url[scheme_end + 3..]; // skip "://"
    let (host, path) = after_scheme
        .split_once('/')
        .ok_or_else(|| WickError::Backend(format!("url `{url}` has no path component")))?;
    if host.is_empty() {
        return Err(WickError::Backend(format!(
            "url `{url}` has empty host component"
        )));
    }
    // Return the path slice preserving its leading `/` so the caller
    // can detect an empty path after trimming.
    let path_start = url.len() - path.len() - 1;
    Ok((host, &url[path_start..]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_for_url_mirrors_host_and_path() {
        let repo = BundleRepo::new("/tmp/store");
        let p = repo
            .path_for_url("https://huggingface.co/LiquidAI/LFM2-1.2B-GGUF/resolve/main/x.gguf")
            .unwrap();
        assert_eq!(
            p,
            PathBuf::from("/tmp/store/huggingface.co/LiquidAI/LFM2-1.2B-GGUF/resolve/main/x.gguf")
        );
    }

    #[test]
    fn split_url_rejects_missing_scheme() {
        assert!(split_url("huggingface.co/x").is_err());
    }

    #[test]
    fn split_url_rejects_missing_path() {
        assert!(split_url("https://huggingface.co").is_err());
    }

    #[test]
    fn split_url_accepts_http_and_https() {
        assert!(split_url("http://example.com/x").is_ok());
        assert!(split_url("https://example.com/x").is_ok());
    }

    #[test]
    fn split_url_scheme_is_case_insensitive() {
        // Mixed-case schemes must be accepted (RFC 3986 §3.1) so we
        // don't drift from `engine::is_remote_url`, which already
        // does case-insensitive matching. Otherwise an `HTTPS://…`
        // URL would pass the remote-URL gate but fail here.
        assert!(split_url("HTTPS://example.com/x").is_ok());
        assert!(split_url("Http://example.com/x").is_ok());
        assert!(split_url("HTTP://example.com/x").is_ok());
    }

    #[test]
    fn path_for_url_lowercases_host_for_cache_consistency() {
        // Two URLs that differ only in host casing must share a cache
        // entry — hosts are case-insensitive per RFC 3986 §3.2.2, but
        // case-sensitive filesystems would otherwise double-cache.
        let repo = BundleRepo::new("/tmp/store");
        let a = repo
            .path_for_url("https://HuggingFace.co/LiquidAI/M/x.gguf")
            .unwrap();
        let b = repo
            .path_for_url("https://huggingface.co/LiquidAI/M/x.gguf")
            .unwrap();
        assert_eq!(a, b);
        assert_eq!(
            a,
            PathBuf::from("/tmp/store/huggingface.co/LiquidAI/M/x.gguf")
        );
    }

    #[test]
    fn path_for_url_rejects_parent_dir_segment() {
        let repo = BundleRepo::new("/tmp/store");
        // Attacker-controlled URL with `..` would otherwise escape
        // `store_dir` after PathBuf::push canonicalization.
        let e = repo
            .path_for_url("https://evil.example.com/a/../../etc/passwd")
            .expect_err("`..` segment must be rejected");
        assert!(format!("{e}").contains("not a valid path component"));
    }

    #[test]
    fn path_for_url_rejects_windows_reserved_chars() {
        // Chars that appear as segment content and are Windows-reserved:
        // `*`, `"`, `<`, `>`, `|`. (`?` and `#` are separately stripped
        // by `path_for_url` as URL syntax — see
        // `path_for_url_strips_query_and_fragment`.) Catching these up
        // front means a Windows consumer never sees a cryptic
        // filesystem error at `PathBuf::push` time.
        let repo = BundleRepo::new("/tmp/store");
        for bad in ["a*b", "a\"b", "a<b", "a>b", "a|b"] {
            let url = format!("https://example.com/{bad}");
            let e = repo
                .path_for_url(&url)
                .expect_err(&format!("{bad:?} must be rejected"));
            let msg = format!("{e}");
            assert!(
                msg.contains("forbidden"),
                "unexpected error for {bad:?}: {msg}"
            );
        }
    }

    #[test]
    fn path_for_url_rejects_empty_segment() {
        let repo = BundleRepo::new("/tmp/store");
        // Double slash produces an empty segment.
        let e = repo
            .path_for_url("https://example.com/a//b")
            .expect_err("empty path segment must be rejected");
        assert!(format!("{e}").contains("must not be empty"));
    }

    #[test]
    fn path_for_url_strips_query_and_fragment() {
        let repo = BundleRepo::new("/tmp/store");
        // Query / fragment are URL syntax; they must not appear in the
        // on-disk path or the filename becomes `model.gguf?x=1` etc.
        let p = repo
            .path_for_url("https://example.com/model.gguf?foo=bar#frag")
            .unwrap();
        assert_eq!(p, PathBuf::from("/tmp/store/example.com/model.gguf"));
    }

    #[test]
    fn path_for_url_rejects_null_byte_in_path() {
        let repo = BundleRepo::new("/tmp/store");
        let e = repo
            .path_for_url("https://example.com/a\0b")
            .expect_err("null byte in path must be rejected");
        assert!(format!("{e}").contains("forbidden"));
    }

    #[test]
    fn hash_matches_uses_sidecar_fast_path() {
        // Covers the first-class invariant behind the PR #37 review's
        // concern about rehashing large cached GGUFs: when a sidecar
        // exists and matches, `hash_matches` must return `true`
        // WITHOUT touching the file contents. We prove the latter by
        // never writing file contents at all — only a sidecar. If the
        // code ever regresses into full-rehash mode here, `sha256_file`
        // will error (file doesn't exist / is empty vs. expected hash)
        // and this test will fail.
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("x.gguf");
        // Create a dummy empty file so `dest.exists()` would be true
        // in a real resolve, but we're calling `hash_matches` directly
        // here so this isn't strictly necessary.
        std::fs::write(&dest, b"").unwrap();
        let hex = "0123456789abcdef".repeat(4);
        assert_eq!(hex.len(), 64);
        std::fs::write(download::sidecar_path(&dest), &hex).unwrap();

        assert!(hash_matches(&dest, "https://example.com/x", &hex));
        // Case-insensitive match.
        assert!(hash_matches(
            &dest,
            "https://example.com/x",
            &hex.to_uppercase()
        ));
        // Mismatch returns false without panicking.
        let wrong = "f".repeat(64);
        assert!(!hash_matches(&dest, "https://example.com/x", &wrong));
    }

    #[test]
    fn hash_matches_full_rehash_when_no_sidecar() {
        // When the sidecar is absent, `hash_matches` falls back to
        // hashing the file. This is a correctness test, not a
        // performance test — it just proves the fallback produces the
        // right answer for a known input.
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("x.bin");
        std::fs::write(&dest, b"hello").unwrap();
        let correct = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824";
        let wrong = "0".repeat(64);
        assert!(hash_matches(&dest, "https://example.com/x", correct));
        assert!(!hash_matches(&dest, "https://example.com/x", &wrong));
    }
}
