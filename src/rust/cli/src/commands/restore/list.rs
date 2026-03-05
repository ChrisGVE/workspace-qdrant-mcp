//! `restore list` subcommand handler.

use anyhow::{Context, Result};

use crate::output;

use super::client::{build_client, qdrant_url, CollectionsResult, QdrantResponse, SnapshotInfo};

/// List available snapshots, optionally filtered to a specific collection.
pub async fn list_snapshots(collection: Option<String>) -> Result<()> {
    output::section("Available Snapshots");

    let base = qdrant_url();
    let client = build_client()?;

    match &collection {
        Some(coll) => list_collection_snapshots(&client, &base, coll).await?,
        None => list_all_snapshots(&client, &base).await?,
    }

    Ok(())
}

/// List snapshots for a single named collection.
async fn list_collection_snapshots(client: &reqwest::Client, base: &str, coll: &str) -> Result<()> {
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
            print_snapshot_row(s);
        }
    }
    Ok(())
}

/// List full snapshots and per-collection snapshots.
async fn list_all_snapshots(client: &reqwest::Client, base: &str) -> Result<()> {
    list_full_snapshots(client, base).await;
    output::separator();
    list_per_collection_snapshots(client, base).await;
    Ok(())
}

/// Print full (non-collection-scoped) snapshots.
async fn list_full_snapshots(client: &reqwest::Client, base: &str) {
    output::info("Full snapshots:");
    let url = format!("{}/snapshots", base);
    if let Ok(resp) = client.get(&url).send().await {
        if resp.status().is_success() {
            if let Ok(api_resp) = resp.json::<QdrantResponse<Vec<SnapshotInfo>>>().await {
                if api_resp.result.is_empty() {
                    output::info("  No full snapshots found.");
                } else {
                    for s in &api_resp.result {
                        print_snapshot_row(s);
                    }
                }
            }
        }
    }
}

/// Enumerate all collections and print their snapshots.
async fn list_per_collection_snapshots(client: &reqwest::Client, base: &str) {
    let coll_url = format!("{}/collections", base);
    let Ok(coll_resp) = client.get(&coll_url).send().await else {
        return;
    };
    if !coll_resp.status().is_success() {
        return;
    }
    let Ok(coll_result) = coll_resp.json::<QdrantResponse<CollectionsResult>>().await else {
        return;
    };

    for entry in &coll_result.result.collections {
        let snap_url = format!("{}/collections/{}/snapshots", base, entry.name);
        if let Ok(resp) = client.get(&snap_url).send().await {
            if resp.status().is_success() {
                if let Ok(snap_result) = resp.json::<QdrantResponse<Vec<SnapshotInfo>>>().await {
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

/// Print one snapshot row with name, size, and optional creation time.
fn print_snapshot_row(s: &SnapshotInfo) {
    output::kv("  Name", &s.name);
    output::kv("  Size", output::format_bytes(s.size));
    if let Some(ref t) = s.creation_time {
        output::kv("  Created", t);
    }
    output::separator();
}
