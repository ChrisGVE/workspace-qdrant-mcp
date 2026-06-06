//! Pin grammar sources to immutable refs + tarball checksums (#88).
//!
//! For each language definition's preferred (first) grammar source — the one
//! the daemon's downloader actually uses — this module resolves an immutable
//! git ref and records the SHA256 of the source tarball at that ref:
//!
//! 1. Resolve the ref: the repo's latest release tag (via the unauthenticated
//!    `releases/latest` redirect — no API rate limit), falling back to the
//!    default-branch HEAD commit SHA via the GitHub API for repos without
//!    releases.
//! 2. Download `archive/{ref}.tar.gz` and hash it.
//!
//! Failures are logged and skipped — an entry that cannot be pinned simply
//! stays unpinned (the daemon then keeps its original moving-target fetch).

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use workspace_qdrant_core::language_registry::types::LanguageDefinition;

/// Outcome counters for a pinning pass.
#[derive(Debug, Default, Clone, Copy)]
pub struct PinStats {
    /// Sources successfully pinned (git_ref + sha256 recorded).
    pub pinned: usize,
    /// Sources where ref resolution or download failed (left unpinned).
    pub failed: usize,
    /// Definitions with no grammar source (nothing to pin).
    pub skipped: usize,
}

/// Pin the preferred grammar source of every definition in place.
pub async fn pin_definitions(
    definitions: &mut [LanguageDefinition],
    github_token: Option<&str>,
) -> Result<PinStats> {
    let client = reqwest::Client::builder()
        .user_agent("workspace-qdrant-mcp registry-updater")
        .build()
        .context("Building HTTP client")?;

    let mut stats = PinStats::default();

    for def in definitions.iter_mut() {
        let language = def.language.clone();
        let Some(source) = def.grammar.sources.first_mut() else {
            stats.skipped += 1;
            continue;
        };

        match pin_source(&client, &source.repo, github_token).await {
            Ok((git_ref, sha256)) => {
                tracing::info!(language, repo = %source.repo, git_ref, "pinned");
                source.git_ref = Some(git_ref);
                source.sha256 = Some(sha256);
                stats.pinned += 1;
            }
            Err(e) => {
                tracing::warn!(language, repo = %source.repo, error = %e, "pinning failed");
                stats.failed += 1;
            }
        }
    }

    Ok(stats)
}

/// Resolve an immutable ref for `owner/repo` and hash its archive tarball.
///
/// If the resolved release tag's archive is not fetchable — e.g. GitHub
/// answers HTTP 300 when a tag and branch share a name (seen with
/// briot/tree-sitter-ada's "master" release tag) — retry once with the
/// unambiguous HEAD commit SHA.
pub async fn pin_source(
    client: &reqwest::Client,
    repo: &str,
    github_token: Option<&str>,
) -> Result<(String, String)> {
    let git_ref = resolve_ref(client, repo, github_token).await?;
    match hash_archive(client, repo, &git_ref).await {
        Ok(sha256) => Ok((git_ref, sha256)),
        Err(tag_err) => {
            let sha_ref = head_commit_sha(client, repo, github_token)
                .await
                .with_context(|| format!("after tag archive failed: {tag_err}"))?;
            let sha256 = hash_archive(client, repo, &sha_ref).await?;
            Ok((sha_ref, sha256))
        }
    }
}

/// Download `archive/{git_ref}.tar.gz` and return its SHA256 (hex).
async fn hash_archive(client: &reqwest::Client, repo: &str, git_ref: &str) -> Result<String> {
    let url = format!("https://github.com/{repo}/archive/{git_ref}.tar.gz");
    let resp = client
        .get(&url)
        .send()
        .await
        .with_context(|| format!("Downloading {url}"))?
        .error_for_status()
        .with_context(|| format!("HTTP error for {url}"))?;
    let bytes = resp
        .bytes()
        .await
        .with_context(|| format!("Reading body of {url}"))?;
    Ok(format!("{:x}", Sha256::digest(&bytes)))
}

/// Resolve the immutable ref to pin: latest release tag, else default-branch
/// HEAD commit SHA.
async fn resolve_ref(
    client: &reqwest::Client,
    repo: &str,
    github_token: Option<&str>,
) -> Result<String> {
    if let Some(tag) = latest_release_tag(client, repo).await? {
        return Ok(tag);
    }
    head_commit_sha(client, repo, github_token).await
}

/// Latest release tag via the `releases/latest` redirect (rate-limit free).
///
/// GitHub answers `releases/latest` with a redirect to `releases/tag/{tag}`
/// when releases exist, and to the releases index (or 404) when none do.
async fn latest_release_tag(client: &reqwest::Client, repo: &str) -> Result<Option<String>> {
    let url = format!("https://github.com/{repo}/releases/latest");
    let resp = client
        .get(&url)
        // reqwest follows redirects by default; the FINAL url carries the tag.
        .send()
        .await
        .with_context(|| format!("Fetching {url}"))?;

    if !resp.status().is_success() {
        return Ok(None);
    }
    let final_url = resp.url().as_str();
    Ok(extract_tag_from_release_url(final_url))
}

/// Pull `{tag}` out of a `.../releases/tag/{tag}` URL, if present.
fn extract_tag_from_release_url(url: &str) -> Option<String> {
    let marker = "/releases/tag/";
    let idx = url.find(marker)?;
    let tag = &url[idx + marker.len()..];
    let tag = tag.split(['?', '#']).next().unwrap_or(tag);
    if tag.is_empty() {
        None
    } else {
        // The tag is percent-encoded in the redirect URL (e.g. "%2F" for "/").
        Some(percent_decode(tag))
    }
}

/// Minimal percent-decoding (GitHub tags only need a handful of escapes).
fn percent_decode(s: &str) -> String {
    let mut out = Vec::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            // Slice bytes (not the str) — avoids a panic on multi-byte UTF-8.
            if let Ok(hex) = std::str::from_utf8(&bytes[i + 1..i + 3]) {
                if let Ok(byte) = u8::from_str_radix(hex, 16) {
                    out.push(byte);
                    i += 3;
                    continue;
                }
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8(out).unwrap_or_else(|_| s.to_string())
}

/// Default-branch HEAD commit SHA via the GitHub API (needs a token to avoid
/// the 60-requests/hour unauthenticated limit on large runs).
async fn head_commit_sha(
    client: &reqwest::Client,
    repo: &str,
    github_token: Option<&str>,
) -> Result<String> {
    let url = format!("https://api.github.com/repos/{repo}/commits/HEAD");
    let mut req = client
        .get(&url)
        .header("Accept", "application/vnd.github.sha");
    if let Some(token) = github_token {
        req = req.bearer_auth(token);
    }
    let resp = req
        .send()
        .await
        .with_context(|| format!("Fetching {url}"))?
        .error_for_status()
        .with_context(|| format!("HTTP error for {url}"))?;
    let sha = resp.text().await.context("Reading commit SHA")?;
    let sha = sha.trim().to_string();
    anyhow::ensure!(
        sha.len() == 40 && sha.bytes().all(|b| b.is_ascii_hexdigit()),
        "unexpected commit SHA response for {repo}: {sha:?}"
    );
    Ok(sha)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_tag_from_redirect_url() {
        assert_eq!(
            extract_tag_from_release_url(
                "https://github.com/tree-sitter/tree-sitter-rust/releases/tag/v0.24.0"
            ),
            Some("v0.24.0".to_string())
        );
        // Percent-encoded slash in tag names (e.g. "grammar/v1.0").
        assert_eq!(
            extract_tag_from_release_url("https://github.com/x/y/releases/tag/grammar%2Fv1.0"),
            Some("grammar/v1.0".to_string())
        );
        // No-release repos redirect to the releases index.
        assert_eq!(
            extract_tag_from_release_url("https://github.com/x/y/releases"),
            None
        );
    }

    #[test]
    fn percent_decode_passthrough_and_escapes() {
        assert_eq!(percent_decode("v0.24.0"), "v0.24.0");
        assert_eq!(percent_decode("a%2Fb"), "a/b");
        assert_eq!(percent_decode("bad%zz"), "bad%zz");
    }
}
