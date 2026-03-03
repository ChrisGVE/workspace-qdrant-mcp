//! `restore list` subcommand handler.

use anyhow::{Context, Result};

use crate::output;

use super::client::{
    build_client, qdrant_url, CollectionsResult, QdrantResponse, SnapshotInfo,
};

/// List available snapshots, optionally filtered to a specific collection.
pub async fn list_snapshots(collection: Option<String>) -> Result<()> {
    output::section("Available Snapshots");

    let base = qdrant_url();
    let client = build_client()?;

    match &collection {
        Some(coll) => {
            output::kv("Collection", coll);
            output::separator();

            let url = format!("{}/collections/{}/snapshots", base, coll);
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

            if api_resp.result.is_empty() {
                output::info("No snapshots found for this collection.");
            } else {
                output::info(&format!("Found {} snapshot(s):", api_resp.result.len()));
                for s in &api_resp.result {
                    output::kv("  Name", &s.name);
                    output::kv("  Size", output::format_bytes(s.size));
                    if let Some(ref t) = s.creation_time {
                        output::kv("  Created", t);
                    }
                    output::separator();
                }
            }
        }
        None => {
            // List full snapshots
            output::info("Full snapshots:");
            let url = format!("{}/snapshots", base);
            let resp = client
                .get(&url)
                .send()
                .await
                .context("Failed to connect to Qdrant")?;

            if resp.status().is_success() {
                let api_resp: QdrantResponse<Vec<SnapshotInfo>> = resp
                    .json()
                    .await
                    .context("Failed to parse full snapshot response")?;

                if api_resp.result.is_empty() {
                    output::info("  No full snapshots found.");
                } else {
                    for s in &api_resp.result {
                        output::kv("  Name", &s.name);
                        output::kv("  Size", output::format_bytes(s.size));
                        if let Some(ref t) = s.creation_time {
                            output::kv("  Created", t);
                        }
                        output::separator();
                    }
                }
            }

            output::separator();

            // List per-collection snapshots
            let coll_url = format!("{}/collections", base);
            let coll_resp = client
                .get(&coll_url)
                .send()
                .await
                .context("Failed to list collections")?;

            if coll_resp.status().is_success() {
                let coll_result: QdrantResponse<CollectionsResult> = coll_resp
                    .json()
                    .await
                    .context("Failed to parse collections response")?;

                for entry in &coll_result.result.collections {
                    let snap_url =
                        format!("{}/collections/{}/snapshots", base, entry.name);
                    if let Ok(resp) = client.get(&snap_url).send().await {
                        if resp.status().is_success() {
                            if let Ok(snap_result) =
                                resp.json::<QdrantResponse<Vec<SnapshotInfo>>>().await
                            {
                                output::info(&format!(
                                    "{}: {} snapshot(s)",
                                    entry.name,
                                    snap_result.result.len()
                                ));
                                for s in &snap_result.result {
                                    output::kv("  Name", &s.name);
                                    output::kv("  Size", output::format_bytes(s.size));
                                    if let Some(ref t) = s.creation_time {
                                        output::kv("  Created", t);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(())
}
