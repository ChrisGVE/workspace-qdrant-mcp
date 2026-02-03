//! Update command - daemon update management
//!
//! Phase 1 HIGH priority command for updating the daemon.
//! Subcommands: check, install, --force, --version

use anyhow::{Context, Result, bail};
use clap::{Args, Subcommand};
use reqwest::Client;
use semver::Version;
use serde::Deserialize;
use sha2::{Sha256, Digest};
use std::path::PathBuf;
use std::process::Command;
use tokio::io::AsyncWriteExt;

use crate::output;

/// GitHub repository for releases
const GITHUB_REPO: &str = "ChrisGVE/workspace-qdrant-mcp";

/// Current daemon version (embedded at compile time)
const CURRENT_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Update command arguments
#[derive(Args)]
pub struct UpdateArgs {
    #[command(subcommand)]
    command: Option<UpdateCommand>,

    /// Force reinstall even if already at latest version
    #[arg(short, long)]
    force: bool,

    /// Install a specific version
    #[arg(short = 'V', long)]
    version: Option<String>,

    /// Update channel (stable, beta, rc, alpha)
    #[arg(short, long, default_value = "stable")]
    channel: String,
}

/// Update subcommands
#[derive(Subcommand)]
enum UpdateCommand {
    /// Check for updates without installing
    Check {
        /// Update channel (stable, beta, rc, alpha)
        #[arg(short, long, default_value = "stable")]
        channel: String,
    },

    /// Install the latest version (or specified version)
    Install {
        /// Force reinstall even if already at latest version
        #[arg(short, long)]
        force: bool,

        /// Install a specific version
        #[arg(short = 'V', long)]
        version: Option<String>,

        /// Update channel (stable, beta, rc, alpha)
        #[arg(short, long, default_value = "stable")]
        channel: String,
    },
}

/// GitHub release API response
#[derive(Debug, Deserialize)]
struct GitHubRelease {
    tag_name: String,
    name: String,
    prerelease: bool,
    draft: bool,
    assets: Vec<GitHubAsset>,
    body: Option<String>,
}

/// GitHub release asset
#[derive(Debug, Deserialize)]
struct GitHubAsset {
    name: String,
    browser_download_url: String,
    size: u64,
}

/// Platform target triple
fn get_target_triple() -> &'static str {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    { "aarch64-apple-darwin" }
    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    { "x86_64-apple-darwin" }
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    { "x86_64-unknown-linux-gnu" }
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    { "aarch64-unknown-linux-gnu" }
    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    { "x86_64-pc-windows-msvc" }
    #[cfg(all(target_os = "windows", target_arch = "aarch64"))]
    { "aarch64-pc-windows-msvc" }
    #[cfg(not(any(
        all(target_os = "macos", target_arch = "aarch64"),
        all(target_os = "macos", target_arch = "x86_64"),
        all(target_os = "linux", target_arch = "x86_64"),
        all(target_os = "linux", target_arch = "aarch64"),
        all(target_os = "windows", target_arch = "x86_64"),
        all(target_os = "windows", target_arch = "aarch64"),
    )))]
    { "unknown" }
}

/// Get the binary filename for the current platform
fn get_binary_filename() -> String {
    let target = get_target_triple();
    #[cfg(target_os = "windows")]
    { format!("memexd-{}.exe", target) }
    #[cfg(not(target_os = "windows"))]
    { format!("memexd-{}", target) }
}

/// Get checksum filename for the current platform
fn get_checksum_filename() -> String {
    format!("{}.sha256", get_binary_filename())
}

/// Execute update command
pub async fn execute(args: UpdateArgs) -> Result<()> {
    match args.command {
        Some(UpdateCommand::Check { channel }) => check(&channel).await,
        Some(UpdateCommand::Install { force, version, channel }) => {
            install(force, version, &channel).await
        }
        None => {
            // Default: check and install if update available
            if args.version.is_some() || args.force {
                install(args.force, args.version, &args.channel).await
            } else {
                check_and_install(&args.channel).await
            }
        }
    }
}

/// Check for updates
async fn check(channel: &str) -> Result<()> {
    output::section("Update Check");

    output::kv("Current version", CURRENT_VERSION);
    output::kv("Platform", get_target_triple());
    output::kv("Channel", channel);
    output::separator();

    let client = create_http_client()?;
    let release = fetch_latest_release_for_channel(&client, channel).await?;

    let latest_version = parse_version(&release.tag_name)?;
    let current_version = Version::parse(CURRENT_VERSION)
        .context("Failed to parse current version")?;

    output::kv("Latest version", &release.tag_name);

    if latest_version > current_version {
        output::separator();
        output::success("Update available!");
        output::info(format!("Run 'wqm update install' to update to {}", release.tag_name));

        if let Some(body) = &release.body {
            output::separator();
            output::info("Release notes:");
            // Print first few lines of release notes
            for line in body.lines().take(10) {
                println!("  {}", line);
            }
        }
    } else if latest_version == current_version {
        output::success("Already at the latest version");
    } else {
        output::info("Current version is newer than the latest release");
    }

    Ok(())
}

/// Check and install if update available
async fn check_and_install(channel: &str) -> Result<()> {
    output::section("Update");

    output::kv("Current version", CURRENT_VERSION);
    output::kv("Platform", get_target_triple());
    output::kv("Channel", channel);
    output::separator();

    let client = create_http_client()?;
    let release = fetch_latest_release_for_channel(&client, channel).await?;

    let latest_version = parse_version(&release.tag_name)?;
    let current_version = Version::parse(CURRENT_VERSION)
        .context("Failed to parse current version")?;

    output::kv("Latest version", &release.tag_name);

    if latest_version > current_version {
        output::separator();
        output::info("Update available, installing...");
        perform_update(&client, &release, false).await?;
    } else {
        output::success("Already at the latest version");
    }

    Ok(())
}

/// Install update (with optional force and version)
async fn install(force: bool, version: Option<String>, channel: &str) -> Result<()> {
    output::section("Install Update");

    output::kv("Current version", CURRENT_VERSION);
    output::kv("Platform", get_target_triple());
    output::kv("Channel", channel);

    let client = create_http_client()?;

    let release = if let Some(ver) = version {
        output::kv("Target version", &ver);
        fetch_specific_release(&client, &ver).await?
    } else {
        output::info("Fetching latest release...");
        fetch_latest_release_for_channel(&client, channel).await?
    };

    output::separator();
    output::kv("Installing version", &release.tag_name);

    if !force {
        let target_version = parse_version(&release.tag_name)?;
        let current_version = Version::parse(CURRENT_VERSION)
            .context("Failed to parse current version")?;

        if target_version == current_version {
            output::info("Already at this version. Use --force to reinstall.");
            return Ok(());
        }
    }

    perform_update(&client, &release, force).await
}

/// Perform the actual update
async fn perform_update(client: &Client, release: &GitHubRelease, _force: bool) -> Result<()> {
    let binary_name = get_binary_filename();
    let checksum_name = get_checksum_filename();

    // Find the binary asset
    let binary_asset = release.assets.iter()
        .find(|a| a.name == binary_name)
        .context(format!("No binary found for platform: {}", get_target_triple()))?;

    // Find the checksum asset (optional)
    let checksum_asset = release.assets.iter()
        .find(|a| a.name == checksum_name);

    output::info(format!("Downloading {} ({} bytes)...", binary_name, binary_asset.size));

    // Download binary to temp file
    let temp_dir = std::env::temp_dir();
    let temp_binary = temp_dir.join(&binary_name);

    download_file(client, &binary_asset.browser_download_url, &temp_binary).await?;
    output::success("Download complete");

    // Verify checksum if available
    if let Some(checksum) = checksum_asset {
        output::info("Verifying checksum...");
        let expected = download_checksum(client, &checksum.browser_download_url).await?;
        let actual = compute_sha256(&temp_binary).await?;

        if expected != actual {
            // Clean up temp file
            let _ = tokio::fs::remove_file(&temp_binary).await;
            bail!("Checksum verification failed!\nExpected: {}\nActual: {}", expected, actual);
        }
        output::success("Checksum verified");
    } else {
        output::warning("No checksum available, skipping verification");
    }

    // Stop the daemon
    output::separator();
    output::info("Stopping daemon...");
    stop_daemon().await?;

    // Find installation location
    let install_path = find_install_location()?;
    output::kv("Install location", install_path.display().to_string());

    // Backup old binary
    let backup_path = install_path.with_extension("bak");
    if install_path.exists() {
        std::fs::rename(&install_path, &backup_path)
            .context("Failed to backup existing binary")?;
    }

    // Move new binary to install location
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let permissions = std::fs::Permissions::from_mode(0o755);
        std::fs::set_permissions(&temp_binary, permissions)?;
    }

    std::fs::rename(&temp_binary, &install_path)
        .context("Failed to install new binary")?;

    // Clean up backup
    let _ = std::fs::remove_file(&backup_path);

    output::success(format!("Installed {} to {}", release.tag_name, install_path.display()));

    // Restart daemon
    output::separator();
    output::info("Starting daemon...");
    start_daemon().await?;

    output::separator();
    output::success("Update complete!");

    Ok(())
}

/// Create HTTP client
fn create_http_client() -> Result<Client> {
    Client::builder()
        .user_agent(format!("wqm-cli/{}", CURRENT_VERSION))
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .context("Failed to create HTTP client")
}

/// Fetch the latest release from GitHub
async fn fetch_latest_release(client: &Client) -> Result<GitHubRelease> {
    let url = format!(
        "https://api.github.com/repos/{}/releases/latest",
        GITHUB_REPO
    );

    let response = client.get(&url)
        .header("Accept", "application/vnd.github+json")
        .send()
        .await
        .context("Failed to connect to GitHub")?;

    if !response.status().is_success() {
        bail!("GitHub API error: {}", response.status());
    }

    response.json()
        .await
        .context("Failed to parse GitHub response")
}

/// Check if a release matches the requested channel
fn matches_channel(release: &GitHubRelease, channel: &str) -> bool {
    match channel.to_lowercase().as_str() {
        "stable" => !release.prerelease && !release.draft,
        "beta" => release.tag_name.contains("-beta"),
        "rc" => release.tag_name.contains("-rc"),
        "alpha" => release.tag_name.contains("-alpha"),
        _ => !release.prerelease && !release.draft,  // Default to stable
    }
}

/// Fetch the latest release for a specific channel
async fn fetch_latest_release_for_channel(client: &Client, channel: &str) -> Result<GitHubRelease> {
    // For stable channel, use the /latest endpoint (faster)
    if channel.to_lowercase() == "stable" {
        return fetch_latest_release(client).await;
    }

    // For other channels, fetch all releases and filter
    let url = format!(
        "https://api.github.com/repos/{}/releases?per_page=50",
        GITHUB_REPO
    );

    let response = client.get(&url)
        .header("Accept", "application/vnd.github+json")
        .send()
        .await
        .context("Failed to connect to GitHub")?;

    if !response.status().is_success() {
        bail!("GitHub API error: {}", response.status());
    }

    let releases: Vec<GitHubRelease> = response.json()
        .await
        .context("Failed to parse GitHub response")?;

    // Find the first release matching the channel
    releases
        .into_iter()
        .filter(|r| !r.draft && matches_channel(r, channel))
        .next()
        .context(format!("No releases found for channel: {}", channel))
}

/// Fetch a specific release from GitHub
async fn fetch_specific_release(client: &Client, version: &str) -> Result<GitHubRelease> {
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

    let response = client.get(&url)
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

    response.json()
        .await
        .context("Failed to parse GitHub response")
}

/// Download a file from URL
async fn download_file(client: &Client, url: &str, path: &PathBuf) -> Result<()> {
    let response = client.get(url)
        .send()
        .await
        .context("Failed to start download")?;

    if !response.status().is_success() {
        bail!("Download failed: {}", response.status());
    }

    let bytes = response.bytes()
        .await
        .context("Failed to download file")?;

    let mut file = tokio::fs::File::create(path)
        .await
        .context("Failed to create file")?;

    file.write_all(&bytes)
        .await
        .context("Failed to write file")?;

    Ok(())
}

/// Download and parse checksum file
async fn download_checksum(client: &Client, url: &str) -> Result<String> {
    let response = client.get(url)
        .send()
        .await
        .context("Failed to download checksum")?;

    if !response.status().is_success() {
        bail!("Checksum download failed: {}", response.status());
    }

    let text = response.text()
        .await
        .context("Failed to read checksum")?;

    // Checksum file format: "hash  filename" or just "hash"
    let checksum = text.split_whitespace()
        .next()
        .context("Invalid checksum format")?
        .to_lowercase();

    Ok(checksum)
}

/// Compute SHA256 of a file
async fn compute_sha256(path: &PathBuf) -> Result<String> {
    let bytes = tokio::fs::read(path)
        .await
        .context("Failed to read file for checksum")?;

    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    let result = hasher.finalize();

    Ok(format!("{:x}", result))
}

/// Parse version string (with or without 'v' prefix)
fn parse_version(version_str: &str) -> Result<Version> {
    let clean = version_str.strip_prefix('v').unwrap_or(version_str);
    Version::parse(clean)
        .context(format!("Failed to parse version: {}", version_str))
}

/// Find the installation location of the daemon binary
fn find_install_location() -> Result<PathBuf> {
    // First, try to find existing installation
    if let Ok(path) = which::which("memexd") {
        return Ok(path);
    }

    // Default locations by platform
    #[cfg(target_os = "macos")]
    let default = PathBuf::from("/usr/local/bin/memexd");

    #[cfg(target_os = "linux")]
    let default = dirs::home_dir()
        .map(|h| h.join(".local/bin/memexd"))
        .unwrap_or_else(|| PathBuf::from("/usr/local/bin/memexd"));

    #[cfg(target_os = "windows")]
    let default = dirs::data_local_dir()
        .map(|d| d.join("workspace-qdrant").join("memexd.exe"))
        .unwrap_or_else(|| PathBuf::from("C:\\Program Files\\workspace-qdrant\\memexd.exe"));

    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    let default = PathBuf::from("./memexd");

    // Ensure parent directory exists
    if let Some(parent) = default.parent() {
        std::fs::create_dir_all(parent).ok();
    }

    Ok(default)
}

/// Stop the daemon
async fn stop_daemon() -> Result<()> {
    // Try to stop gracefully via service command
    let result = Command::new("wqm")
        .args(["service", "stop"])
        .status();

    match result {
        Ok(status) if status.success() => {
            output::success("Daemon stopped");
        }
        _ => {
            // Fallback: kill by name
            #[cfg(unix)]
            {
                let _ = Command::new("pkill")
                    .args(["-f", "memexd"])
                    .status();
            }
            #[cfg(windows)]
            {
                let _ = Command::new("taskkill")
                    .args(["/F", "/IM", "memexd.exe"])
                    .status();
            }
            output::info("Daemon stop attempted");
        }
    }

    // Wait for daemon to fully stop
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    Ok(())
}

/// Start the daemon
async fn start_daemon() -> Result<()> {
    let result = Command::new("wqm")
        .args(["service", "start"])
        .status();

    match result {
        Ok(status) if status.success() => {
            output::success("Daemon started");
        }
        _ => {
            output::warning("Could not start daemon automatically");
            output::info("Start manually with: wqm service start");
        }
    }

    Ok(())
}
