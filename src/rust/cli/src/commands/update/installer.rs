//! Binary download, checksum verification, and daemon lifecycle for update command

use anyhow::{Context, Result, bail};
use reqwest::Client;
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::process::Command;
use tokio::io::AsyncWriteExt;

use crate::output;

use super::github::GitHubRelease;
use super::platform::{
    find_install_location, get_binary_filename, get_checksum_filename, get_target_triple,
};

/// Download a file from URL to the given path
pub async fn download_file(client: &Client, url: &str, path: &PathBuf) -> Result<()> {
    let response = client
        .get(url)
        .send()
        .await
        .context("Failed to start download")?;

    if !response.status().is_success() {
        bail!("Download failed: {}", response.status());
    }

    let bytes = response.bytes().await.context("Failed to download file")?;

    let mut file = tokio::fs::File::create(path)
        .await
        .context("Failed to create file")?;

    file.write_all(&bytes).await.context("Failed to write file")?;

    Ok(())
}

/// Download and parse a checksum file, returning the hex hash string
pub async fn download_checksum(client: &Client, url: &str) -> Result<String> {
    let response = client
        .get(url)
        .send()
        .await
        .context("Failed to download checksum")?;

    if !response.status().is_success() {
        bail!("Checksum download failed: {}", response.status());
    }

    let text = response.text().await.context("Failed to read checksum")?;

    // Checksum file format: "hash  filename" or just "hash"
    let checksum = text
        .split_whitespace()
        .next()
        .context("Invalid checksum format")?
        .to_lowercase();

    Ok(checksum)
}

/// Compute SHA256 of a file and return it as a lowercase hex string
pub async fn compute_sha256(path: &PathBuf) -> Result<String> {
    let bytes = tokio::fs::read(path)
        .await
        .context("Failed to read file for checksum")?;

    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    let result = hasher.finalize();

    Ok(format!("{:x}", result))
}

/// Stop the daemon, waiting briefly for full shutdown
pub async fn stop_daemon() -> Result<()> {
    let result = Command::new("wqm").args(["service", "stop"]).status();

    match result {
        Ok(status) if status.success() => {
            output::success("Daemon stopped");
        }
        _ => {
            // Fallback: kill by name
            #[cfg(unix)]
            {
                let _ = Command::new("pkill").args(["-f", "memexd"]).status();
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
pub async fn start_daemon() -> Result<()> {
    let result = Command::new("wqm").args(["service", "start"]).status();

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

/// Backup the old binary and install the new one from a temp path.
fn replace_binary(temp_binary: &PathBuf, install_path: &PathBuf) -> Result<()> {
    let backup_path = install_path.with_extension("bak");
    if install_path.exists() {
        std::fs::rename(install_path, &backup_path)
            .context("Failed to backup existing binary")?;
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let permissions = std::fs::Permissions::from_mode(0o755);
        std::fs::set_permissions(temp_binary, permissions)?;
    }

    std::fs::rename(temp_binary, install_path)
        .context("Failed to install new binary")?;

    let _ = std::fs::remove_file(&backup_path);
    Ok(())
}

/// Perform the actual binary replacement: download, verify, replace, restart
pub async fn perform_update(client: &Client, release: &GitHubRelease, _force: bool) -> Result<()> {
    let binary_name = get_binary_filename();
    let checksum_name = get_checksum_filename();

    // Find the binary asset
    let binary_asset = release
        .assets
        .iter()
        .find(|a| a.name == binary_name)
        .context(format!("No binary found for platform: {}", get_target_triple()))?;

    // Find the checksum asset (optional)
    let checksum_asset = release.assets.iter().find(|a| a.name == checksum_name);

    output::info(format!(
        "Downloading {} ({} bytes)...",
        binary_name, binary_asset.size
    ));

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
            let _ = tokio::fs::remove_file(&temp_binary).await;
            bail!(
                "Checksum verification failed!\nExpected: {}\nActual: {}",
                expected,
                actual
            );
        }
        output::success("Checksum verified");
    } else {
        output::warning("No checksum available, skipping verification");
    }

    // Stop the daemon
    output::separator();
    output::info("Stopping daemon...");
    stop_daemon().await?;

    // Find installation location and install
    let install_path = find_install_location()?;
    output::kv("Install location", install_path.display().to_string());
    replace_binary(&temp_binary, &install_path)?;

    output::success(format!(
        "Installed {} to {}",
        release.tag_name,
        install_path.display()
    ));

    // Restart daemon
    output::separator();
    output::info("Starting daemon...");
    start_daemon().await?;

    output::separator();
    output::success("Update complete!");

    Ok(())
}
