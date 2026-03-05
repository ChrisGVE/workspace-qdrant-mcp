//! `restore verify` subcommand handler.

use anyhow::{Context, Result};

use crate::output;

use super::client::{build_client, qdrant_url, QdrantResponse, SnapshotInfo};

/// Verify that a snapshot exists and is accessible without restoring it.
pub async fn verify_snapshot(snapshot: &str, collection: &str) -> Result<()> {
    output::section("Verify Snapshot");

    let base = qdrant_url();
    let client = build_client()?;

    output::kv("Qdrant", &base);
    output::kv("Snapshot", snapshot);
    output::kv("Collection", collection);
    output::separator();
    output::info("Running verification checks...");

    let is_full = collection == "all";
    let list_url = if is_full {
        format!("{}/snapshots", base)
    } else {
        format!("{}/collections/{}/snapshots", base, collection)
    };

    let snapshots = fetch_snapshot_list(&client, &list_url).await?;
    let found = snapshots.iter().find(|s| s.name == snapshot);

    match found {
        Some(snap) => {
            print_snapshot_details(snap);
            let download_url = build_download_url(&base, snapshot, collection, is_full);
            verify_download_access(&client, &download_url).await;
            output::separator();
            output::success("Verification complete - snapshot is valid");
            output::info(
                "To restore: wqm restore snapshot <name> --collection <collection> --force",
            );
        }
        None => print_not_found(snapshot, collection, is_full, &snapshots),
    }

    Ok(())
}

/// Fetch the list of snapshots from Qdrant.
async fn fetch_snapshot_list(client: &reqwest::Client, url: &str) -> Result<Vec<SnapshotInfo>> {
    let resp = client
        .get(url)
        .send()
        .await
        .context("Failed to connect to Qdrant")?;
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("Qdrant returned {}: {}", status, body);
    }
    let api_resp: QdrantResponse<Vec<SnapshotInfo>> = resp
        .json()
        .await
        .context("Failed to parse Qdrant response")?;
    Ok(api_resp.result)
}

/// Print metadata for a found snapshot.
fn print_snapshot_details(snap: &SnapshotInfo) {
    output::success("Snapshot exists in Qdrant");
    output::kv("  Name", &snap.name);
    output::kv("  Size", output::format_bytes(snap.size));
    if let Some(ref t) = snap.creation_time {
        output::kv("  Created", t);
    }
    if let Some(ref c) = snap.checksum {
        output::kv("  Checksum", c);
        output::success("Checksum available for integrity verification");
    } else {
        output::warning("No checksum available - integrity cannot be verified");
    }
    output::separator();
}

/// Build the URL used to download (or HEAD-check) a snapshot.
fn build_download_url(base: &str, snapshot: &str, collection: &str, is_full: bool) -> String {
    if is_full {
        format!("{}/snapshots/{}", base, snapshot)
    } else {
        format!("{}/collections/{}/snapshots/{}", base, collection, snapshot)
    }
}

/// Issue a HEAD request to verify the snapshot is downloadable.
async fn verify_download_access(client: &reqwest::Client, url: &str) {
    match client.head(url).send().await {
        Ok(resp) if resp.status().is_success() => {
            output::success("Snapshot is accessible for download/restore");
            if let Some(len) = resp.content_length() {
                output::kv("  Download Size", output::format_bytes(len as i64));
            }
        }
        Ok(resp) => {
            output::warning(&format!(
                "Snapshot download check returned: {}",
                resp.status()
            ));
        }
        Err(e) => {
            output::warning(&format!("Could not verify download access: {}", e));
        }
    }
}

/// Print an error and list available snapshots when the target is not found.
fn print_not_found(snapshot: &str, collection: &str, is_full: bool, available: &[SnapshotInfo]) {
    output::error(format!(
        "Snapshot '{}' not found in {} snapshots",
        snapshot,
        if is_full { "full" } else { collection }
    ));
    output::separator();
    if !available.is_empty() {
        output::info("Available snapshots:");
        for s in available {
            output::info(&format!("  - {}", s.name));
        }
    } else {
        output::info("No snapshots available.");
    }
}
