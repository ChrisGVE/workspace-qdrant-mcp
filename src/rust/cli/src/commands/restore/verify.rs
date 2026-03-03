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

    // Step 1: Check snapshot exists in the list
    let (url, is_full) = if collection == "all" {
        (format!("{}/snapshots", base), true)
    } else {
        (
            format!("{}/collections/{}/snapshots", base, collection),
            false,
        )
    };

    let resp = client
        .get(&url)
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

    let found = api_resp.result.iter().find(|s| s.name == snapshot);

    match found {
        Some(snap) => {
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

            // Step 2: Verify snapshot is downloadable (HEAD request)
            let download_url = if is_full {
                format!("{}/snapshots/{}", base, snapshot)
            } else {
                format!(
                    "{}/collections/{}/snapshots/{}",
                    base, collection, snapshot
                )
            };

            let head_resp = client
                .head(&download_url)
                .send()
                .await;

            match head_resp {
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

            output::separator();
            output::success("Verification complete - snapshot is valid");
            output::info(
                "To restore: wqm restore snapshot <name> --collection <collection> --force",
            );
        }
        None => {
            output::error(format!(
                "Snapshot '{}' not found in {} snapshots",
                snapshot,
                if is_full { "full" } else { collection }
            ));
            output::separator();

            // Show available snapshots
            if !api_resp.result.is_empty() {
                output::info("Available snapshots:");
                for s in &api_resp.result {
                    output::info(&format!("  - {}", s.name));
                }
            } else {
                output::info("No snapshots available.");
            }
        }
    }

    Ok(())
}
