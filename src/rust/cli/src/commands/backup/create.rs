//! Backup create subcommand handler.

use std::io::Write as _;
use std::path::PathBuf;

use anyhow::{Context, Result};

use crate::output;

use super::types::{build_client, qdrant_url, QdrantResponse, SnapshotInfo};

pub async fn create_backup(
    collection: &str,
    output_dir: Option<PathBuf>,
    _description: Option<String>,
    json: bool,
) -> Result<()> {
    if !json {
        output::section("Create Backup");
        output::kv("Qdrant", &qdrant_url());
        output::kv("Collection", collection);
        output::separator();
    }

    let client = build_client()?;
    let snapshot = trigger_snapshot(&client, collection, json).await?;

    if json {
        output::print_json(&snapshot);
    } else {
        output::separator();
        output::success(format!("Snapshot created: {}", snapshot.name));
        output::kv("Size", output::format_bytes(snapshot.size));
        if let Some(ref t) = snapshot.creation_time {
            output::kv("Created", t);
        }
        if let Some(ref c) = snapshot.checksum {
            output::kv("Checksum", c);
        }
    }

    if let Some(dir) = output_dir {
        download_snapshot(&client, collection, &snapshot, &dir, json).await?;
    }

    if !json {
        output::separator();
        output::info("To restore: wqm restore snapshot <name> --collection <collection>");
    }

    Ok(())
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

async fn trigger_snapshot(
    client: &reqwest::Client,
    collection: &str,
    json: bool,
) -> Result<SnapshotInfo> {
    let base = qdrant_url();

    let url = if collection == "all" {
        if !json {
            output::info("Creating full Qdrant snapshot (all collections)...");
        }
        format!("{}/snapshots", base)
    } else {
        if !json {
            output::info(&format!(
                "Creating snapshot for collection '{}'...",
                collection
            ));
        }
        format!("{}/collections/{}/snapshots", base, collection)
    };

    let resp = client
        .post(&url)
        .query(&[("wait", "true")])
        .send()
        .await
        .context("Failed to connect to Qdrant")?;

    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("Qdrant returned {}: {}", status, body);
    }

    let api_resp: QdrantResponse<SnapshotInfo> = resp
        .json()
        .await
        .context("Failed to parse Qdrant response")?;

    Ok(api_resp.result)
}

async fn download_snapshot(
    client: &reqwest::Client,
    collection: &str,
    snapshot: &SnapshotInfo,
    dir: &PathBuf,
    json: bool,
) -> Result<()> {
    if !json {
        output::separator();
        output::info(&format!("Downloading snapshot to {}...", dir.display()));
    }

    std::fs::create_dir_all(dir)
        .with_context(|| format!("Failed to create output directory: {}", dir.display()))?;

    let base = qdrant_url();
    let download_url = if collection == "all" {
        format!("{}/snapshots/{}", base, snapshot.name)
    } else {
        format!(
            "{}/collections/{}/snapshots/{}",
            base, collection, snapshot.name
        )
    };

    let resp = client
        .get(&download_url)
        .send()
        .await
        .context("Failed to download snapshot")?;

    if !resp.status().is_success() {
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("Download failed: {}", body);
    }

    let dest = dir.join(&snapshot.name);
    let bytes = resp.bytes().await.context("Failed to read snapshot data")?;
    let mut file = std::fs::File::create(&dest)
        .with_context(|| format!("Failed to create file: {}", dest.display()))?;
    file.write_all(&bytes)
        .context("Failed to write snapshot file")?;

    if !json {
        output::success(format!("Downloaded to: {}", dest.display()));
    }

    Ok(())
}
