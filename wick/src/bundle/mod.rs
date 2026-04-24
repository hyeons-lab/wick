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
//! caller-supplied hash or the server's `X-Linked-Etag: sha256:<hex>`
//! header (HuggingFace sets this for LFS objects — content-addressed,
//! stable across revisions). A cached file is considered valid when
//! (a) its SHA-256 matches the server's `X-Linked-Etag`, or (b) its
//! size matches the server's `Content-Length` and no `X-Linked-Etag`
//! was offered, or (c) the HEAD probe failed entirely — in which case
//! we reuse whatever's cached rather than letting a transient upstream
//! blip force a re-download. See `download::head_info` for the probe.

pub(crate) mod download;

use std::fs;
use std::path::{Path, PathBuf};

use crate::session::WickError;

/// Repository for remote bundle files cached to a caller-chosen
/// directory. Construction is cheap — create one per `WickEngine` at
/// most, or pass the same instance to multiple engines.
#[derive(Clone, Debug)]
pub struct BundleRepo {
    store_dir: PathBuf,
}

impl BundleRepo {
    /// Create a new repo rooted at `store_dir`. The directory does not
    /// need to exist yet — it will be created on the first download.
    pub fn new(store_dir: impl Into<PathBuf>) -> Self {
        Self {
            store_dir: store_dir.into(),
        }
    }

    /// Root directory backing this repo.
    pub fn store_dir(&self) -> &Path {
        &self.store_dir
    }

    /// Resolve a remote URL to a local path, downloading if not cached.
    ///
    /// ### Verification policy
    ///
    /// On cache hit:
    /// - If HEAD provides `X-Linked-Etag: sha256:<hex>`, compare against
    ///   the cached file's SHA-256. Mismatch → re-download.
    /// - Else if HEAD provides `Content-Length`, compare sizes.
    /// - Else (HEAD unreachable or silent), reuse the cached file.
    ///
    /// On cache miss: download, verifying the hash against
    /// `X-Linked-Etag` if present. A mismatch deletes the partial and
    /// returns `WickError::Backend`.
    pub fn resolve_url(&self, url: &str) -> Result<PathBuf, WickError> {
        let dest = self.path_for_url(url)?;
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)?;
        }

        let head = download::head_info(url);

        if dest.exists() {
            let keep = match (head.linked_sha256.as_deref(), head.content_length) {
                (Some(exp_hash), _) => match download::sha256_file(&dest) {
                    Ok(actual) if actual == exp_hash => true,
                    Ok(actual) => {
                        tracing::info!(
                            target: "wick::bundle",
                            url,
                            expected = exp_hash,
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
                },
                (None, Some(exp_len)) => {
                    let actual = fs::metadata(&dest).map(|m| m.len()).unwrap_or(0);
                    if actual == exp_len {
                        true
                    } else {
                        tracing::info!(
                            target: "wick::bundle",
                            url,
                            expected = exp_len,
                            actual,
                            "cached file size mismatch; re-downloading"
                        );
                        false
                    }
                }
                (None, None) => {
                    // HEAD failed entirely — reuse. A CI cache hit
                    // shouldn't be defeated by a transient outage.
                    true
                }
            };
            if keep {
                return Ok(dest);
            }
        }

        tracing::info!(
            target: "wick::bundle",
            url,
            dest = %dest.display(),
            "downloading bundle file"
        );
        download::download_to(url, &dest, None)?;
        Ok(dest)
    }

    /// Compute the on-disk cache location for `url`, rooted at
    /// `store_dir`. Mirrors `<host>/<path>` so the cache is inspectable
    /// and safely swappable with a host-preserving mirror.
    fn path_for_url(&self, url: &str) -> Result<PathBuf, WickError> {
        let (host, path) = split_url(url)?;
        let mut out = self.store_dir.clone();
        out.push(host);
        // Strip the leading `/` before pushing — otherwise `PathBuf::push`
        // treats it as absolute and throws away the `host` prefix.
        let rel = path.trim_start_matches('/');
        if rel.is_empty() {
            return Err(WickError::Backend(format!(
                "url `{url}` has no path component"
            )));
        }
        out.push(rel);
        Ok(out)
    }
}

/// Minimal URL parser: extract `(host, path)` from `https://host/path`.
/// We avoid pulling in a full `url` crate dep — this is the only URL
/// handling `wick` needs and the shape we accept is narrow.
fn split_url(url: &str) -> Result<(&str, &str), WickError> {
    let rest = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))
        .ok_or_else(|| {
            WickError::Backend(format!("url `{url}` must start with https:// or http://"))
        })?;
    let (host, path) = rest
        .split_once('/')
        .ok_or_else(|| WickError::Backend(format!("url `{url}` has no path component")))?;
    if host.is_empty() {
        return Err(WickError::Backend(format!(
            "url `{url}` has empty host component"
        )));
    }
    Ok((host, &url[url.len() - path.len() - 1..]))
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
}
