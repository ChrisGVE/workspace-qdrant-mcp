//! GitHub API types and release fetching for update command

use anyhow::{bail, Context, Result};
use reqwest::Client;
use serde::Deserialize;

use super::GITHUB_REPO;

/// GitHub release API response
#[derive(Debug, Deserialize)]
#[allow(dead_code)] // Fields needed for serde JSON deserialization from GitHub API
pub struct GitHubRelease {
    pub tag_name: String,
    pub name: String,
    pub prerelease: bool,
    pub draft: bool,
    pub assets: Vec<GitHubAsset>,
    pub body: Option<String>,
}

/// GitHub release asset
#[derive(Debug, Deserialize)]
pub struct GitHubAsset {
    pub name: String,
    pub browser_download_url: String,
    pub size: u64,
}

/// Create HTTP client
pub fn create_http_client(current_version: &str) -> Result<Client> {
    Client::builder()
        .user_agent(format!("wqm-cli/{}", current_version))
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .context("Failed to create HTTP client")
}

/// Fetch the latest stable release from GitHub
pub async fn fetch_latest_release(client: &Client) -> Result<GitHubRelease> {
    let url = format!(
        "https://api.github.com/repos/{}/releases/latest",
        GITHUB_REPO
    );

    let response = client
        .get(&url)
        .header("Accept", "application/vnd.github+json")
        .send()
        .await
        .context("Failed to connect to GitHub")?;

    if !response.status().is_success() {
        bail!("GitHub API error: {}", response.status());
    }

    response
        .json()
        .await
        .context("Failed to parse GitHub response")
}

/// Check if a release matches the requested channel
pub fn matches_channel(release: &GitHubRelease, channel: &str) -> bool {
    match channel.to_lowercase().as_str() {
        "stable" => !release.prerelease && !release.draft,
        "beta" => release.tag_name.contains("-beta"),
        "rc" => release.tag_name.contains("-rc"),
        "alpha" => release.tag_name.contains("-alpha"),
        _ => !release.prerelease && !release.draft, // Default to stable
    }
}

/// Fetch the latest release for a specific channel
pub async fn fetch_latest_release_for_channel(
    client: &Client,
    channel: &str,
) -> Result<GitHubRelease> {
    // For stable channel, use the /latest endpoint (faster)
    if channel.to_lowercase() == "stable" {
        return fetch_latest_release(client).await;
    }

    // For other channels, fetch all releases and filter
    let url = format!(
        "https://api.github.com/repos/{}/releases?per_page=50",
        GITHUB_REPO
    );

    let response = client
        .get(&url)
        .header("Accept", "application/vnd.github+json")
        .send()
        .await
        .context("Failed to connect to GitHub")?;

    if !response.status().is_success() {
        bail!("GitHub API error: {}", response.status());
    }

    let releases: Vec<GitHubRelease> = response
        .json()
        .await
        .context("Failed to parse GitHub response")?;

    releases
        .into_iter()
        .filter(|r| !r.draft && matches_channel(r, channel))
        .next()
        .context(format!("No releases found for channel: {}", channel))
}

/// Fetch a specific release from GitHub by version tag
pub async fn fetch_specific_release(client: &Client, version: &str) -> Result<GitHubRelease> {
    // Ensure version has 'v' prefix
    let tag = if version.starts_with('v') {
        version.to_string()
    } else {
        format!("v{}", version)
    };

    let url = format!(
        "https://api.github.com/repos/{}/releases/tags/{}",
        GITHUB_REPO, tag
    );

    let response = client
        .get(&url)
        .header("Accept", "application/vnd.github+json")
        .send()
        .await
        .context("Failed to connect to GitHub")?;

    if response.status() == reqwest::StatusCode::NOT_FOUND {
        bail!("Version {} not found", tag);
    }

    if !response.status().is_success() {
        bail!("GitHub API error: {}", response.status());
    }

    response
        .json()
        .await
        .context("Failed to parse GitHub response")
}
