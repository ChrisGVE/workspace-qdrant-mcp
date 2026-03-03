//! Subcommand handler functions for the update command

use anyhow::{Context, Result};
use semver::Version;

use crate::output;

use super::github::{create_http_client, fetch_latest_release_for_channel, fetch_specific_release};
use super::installer::perform_update;
use super::platform::{get_target_triple};
use super::CURRENT_VERSION;

/// Parse version string (with or without 'v' prefix)
pub fn parse_version(version_str: &str) -> Result<Version> {
    let clean = version_str.strip_prefix('v').unwrap_or(version_str);
    Version::parse(clean).context(format!("Failed to parse version: {}", version_str))
}

/// Check for available updates without installing
pub async fn check(channel: &str) -> Result<()> {
    output::section("Update Check");

    output::kv("Current version", CURRENT_VERSION);
    output::kv("Platform", get_target_triple());
    output::kv("Channel", channel);
    output::separator();

    let client = create_http_client(CURRENT_VERSION)?;
    let release = fetch_latest_release_for_channel(&client, channel).await?;

    let latest_version = parse_version(&release.tag_name)?;
    let current_version =
        Version::parse(CURRENT_VERSION).context("Failed to parse current version")?;

    output::kv("Latest version", &release.tag_name);

    if latest_version > current_version {
        output::separator();
        output::success("Update available!");
        output::info(format!(
            "Run 'wqm update install' to update to {}",
            release.tag_name
        ));

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

/// Check and install if an update is available
pub async fn check_and_install(channel: &str) -> Result<()> {
    output::section("Update");

    output::kv("Current version", CURRENT_VERSION);
    output::kv("Platform", get_target_triple());
    output::kv("Channel", channel);
    output::separator();

    let client = create_http_client(CURRENT_VERSION)?;
    let release = fetch_latest_release_for_channel(&client, channel).await?;

    let latest_version = parse_version(&release.tag_name)?;
    let current_version =
        Version::parse(CURRENT_VERSION).context("Failed to parse current version")?;

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

/// Install an update (optionally forced or pinned to a specific version)
pub async fn install(force: bool, version: Option<String>, channel: &str) -> Result<()> {
    output::section("Install Update");

    output::kv("Current version", CURRENT_VERSION);
    output::kv("Platform", get_target_triple());
    output::kv("Channel", channel);

    let client = create_http_client(CURRENT_VERSION)?;

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
        let current_version =
            Version::parse(CURRENT_VERSION).context("Failed to parse current version")?;

        if target_version == current_version {
            output::info("Already at this version. Use --force to reinstall.");
            return Ok(());
        }
    }

    perform_update(&client, &release, force).await
}
