//! HTTP fetch logic for grammar source tarballs.

use reqwest::Client;
use tracing::{debug, info};

use super::{DownloadError, DownloadResult};
use crate::tree_sitter::grammar_registry;

/// Fetch grammar source tarball.
///
/// When the registry pins an immutable `git_ref`, ONLY the pinned archive URL
/// is fetched — falling back to a moving target would defeat the pin (and any
/// sha256 over different bytes would fail anyway). Unpinned sources keep the
/// original behavior: GitHub release URL first, then branch-archive fallback.
pub(super) async fn fetch_grammar_source(
    client: &Client,
    language: &str,
    source: &grammar_registry::GrammarSource,
) -> DownloadResult<Vec<u8>> {
    if let Some(ref git_ref) = source.git_ref {
        let pinned_url = source.pinned_archive_url(git_ref);
        debug!(language, url = %pinned_url, "Fetching pinned archive tarball");
        let bytes = fetch_bytes(client, &pinned_url, language, git_ref).await?;
        info!(language, git_ref, "Downloaded pinned archive");
        return Ok(bytes);
    }

    // Try the GitHub release tarball first (most grammars have releases).
    // The redirect URL resolves to the latest release asset.
    let release_url = format!(
        "https://github.com/{}/{}/releases/latest/download/{}.tar.gz",
        source.owner, source.repo, source.repo
    );

    debug!(language, url = %release_url, "Trying release tarball");
    match fetch_bytes(client, &release_url, language, "latest").await {
        Ok(bytes) => {
            info!(language, "Downloaded from release tarball");
            return Ok(bytes);
        }
        Err(DownloadError::NotFound { .. }) | Err(DownloadError::HttpError { .. }) => {
            debug!(language, "No release tarball, trying archive fallback");
        }
        Err(e) => return Err(e),
    }

    // Fallback: download from a branch archive.
    // Some repos keep generated parser.c only on a specific branch.
    let mut branches: Vec<&str> = Vec::new();
    if let Some(ref branch) = source.archive_branch {
        branches.push(branch.as_str());
    }
    branches.extend_from_slice(&["main", "master"]);

    for branch in &branches {
        let archive_url = source.archive_tarball_url(branch);
        debug!(language, url = %archive_url, "Trying archive tarball");
        match fetch_bytes(client, &archive_url, language, branch).await {
            Ok(bytes) => {
                info!(language, branch, "Downloaded from archive");
                return Ok(bytes);
            }
            Err(DownloadError::NotFound { .. }) | Err(DownloadError::HttpError { .. }) => {
                continue;
            }
            Err(e) => return Err(e),
        }
    }

    Err(DownloadError::NotFound {
        language: language.to_string(),
        version: "latest".to_string(),
    })
}

/// Fetch raw bytes from a URL, mapping HTTP errors to [`DownloadError`].
pub(super) async fn fetch_bytes(
    client: &Client,
    url: &str,
    language: &str,
    version: &str,
) -> DownloadResult<Vec<u8>> {
    let response = client
        .get(url)
        .header("User-Agent", "workspace-qdrant-mcp grammar-downloader")
        .send()
        .await
        .map_err(|e| DownloadError::NetworkError(e.to_string()))?;

    let status = response.status();
    if status == reqwest::StatusCode::NOT_FOUND {
        return Err(DownloadError::NotFound {
            language: language.to_string(),
            version: version.to_string(),
        });
    }
    if !status.is_success() {
        return Err(DownloadError::HttpError {
            status: status.as_u16(),
            message: format!("Failed to download from {}", url),
        });
    }

    let bytes = response
        .bytes()
        .await
        .map_err(|e| DownloadError::NetworkError(format!("Failed to read response body: {}", e)))?;
    Ok(bytes.to_vec())
}
