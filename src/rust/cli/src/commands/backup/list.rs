//! Backup list subcommand handler.

use anyhow::{Context, Result};

use crate::output;

use super::types::{
    build_client, qdrant_url, CollectionsResult, QdrantResponse, SnapshotInfo, SnapshotRow,
};

pub async fn list_backups(
    collection: Option<String>,
    verbose: bool,
    json: bool,
    script: bool,
    no_headers: bool,
) -> Result<()> {
    if !json {
        output::section("List Backups");
    }

    let base = qdrant_url();
    let client = build_client()?;

    match &collection {
        Some(coll) => {
            if !json {
                output::kv("Collection", coll);
                output::separator();
            }
            let snapshots = fetch_collection_snapshots(&client, &base, coll).await?;
            display_collection_snapshots(&snapshots, verbose, json, script, no_headers);
        }
        None => {
            let all_snapshots = fetch_all_snapshots(&client, &base, json, verbose).await?;
            if json {
                output::print_json(&all_snapshots);
            } else if script {
                let rows: Vec<SnapshotRow> = all_snapshots.iter().map(SnapshotRow::from).collect();
                output::print_script(&rows, !no_headers);
            }
        }
    }

    Ok(())
}

// ─── Fetch helpers ───────────────────────────────────────────────────────────

async fn fetch_collection_snapshots(
    client: &reqwest::Client,
    base: &str,
    collection: &str,
) -> Result<Vec<SnapshotInfo>> {
    let url = format!("{}/collections/{}/snapshots", base, collection);
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
    Ok(api_resp.result)
}

async fn fetch_all_snapshots(
    client: &reqwest::Client,
    base: &str,
    json: bool,
    verbose: bool,
) -> Result<Vec<SnapshotInfo>> {
    if !json {
        output::info("Full snapshots (all collections):");
    }

    let mut all_snapshots: Vec<SnapshotInfo> = Vec::new();
    all_snapshots.extend(fetch_full_snapshots(client, base, json).await?);

    if !json {
        output::separator();
        output::info("Per-collection snapshots:");
    }
    fetch_per_collection_snapshots(client, base, json, verbose, &mut all_snapshots).await?;

    Ok(all_snapshots)
}

async fn fetch_full_snapshots(
    client: &reqwest::Client,
    base: &str,
    json: bool,
) -> Result<Vec<SnapshotInfo>> {
    let url = format!("{}/snapshots", base);
    let resp = client
        .get(&url)
        .send()
        .await
        .context("Failed to connect to Qdrant")?;

    if !resp.status().is_success() {
        return Ok(Vec::new());
    }

    let api_resp: QdrantResponse<Vec<SnapshotInfo>> = resp
        .json()
        .await
        .context("Failed to parse full snapshot response")?;

    if !json {
        if api_resp.result.is_empty() {
            output::info("  No full snapshots found.");
        } else {
            let rows: Vec<SnapshotRow> = api_resp.result.iter().map(SnapshotRow::from).collect();
            output::print_table_auto(&rows);
        }
    }

    Ok(api_resp.result)
}

async fn fetch_per_collection_snapshots(
    client: &reqwest::Client,
    base: &str,
    json: bool,
    verbose: bool,
    all_snapshots: &mut Vec<SnapshotInfo>,
) -> Result<()> {
    let coll_url = format!("{}/collections", base);
    let coll_resp = client
        .get(&coll_url)
        .send()
        .await
        .context("Failed to list collections")?;

    if !coll_resp.status().is_success() {
        return Ok(());
    }

    let coll_result: QdrantResponse<CollectionsResult> = coll_resp
        .json()
        .await
        .context("Failed to parse collections response")?;

    for entry in &coll_result.result.collections {
        let snap_url = format!("{}/collections/{}/snapshots", base, entry.name);
        let snap_resp = client.get(&snap_url).send().await;
        if let Ok(resp) = snap_resp {
            if resp.status().is_success() {
                if let Ok(snap_result) = resp.json::<QdrantResponse<Vec<SnapshotInfo>>>().await {
                    display_per_collection_entry(
                        &entry.name,
                        &snap_result.result,
                        json,
                        verbose,
                        all_snapshots,
                    );
                }
            }
        }
    }

    Ok(())
}

// ─── Display helpers ─────────────────────────────────────────────────────────

fn display_collection_snapshots(
    snapshots: &[SnapshotInfo],
    verbose: bool,
    json: bool,
    script: bool,
    no_headers: bool,
) {
    if json {
        output::print_json(snapshots);
    } else if script {
        let rows: Vec<SnapshotRow> = snapshots.iter().map(SnapshotRow::from).collect();
        output::print_script(&rows, !no_headers);
    } else if snapshots.is_empty() {
        output::info("No snapshots found for this collection.");
    } else {
        output::info(&format!("Found {} snapshot(s):", snapshots.len()));
        let rows: Vec<SnapshotRow> = snapshots.iter().map(SnapshotRow::from).collect();
        output::print_table_auto(&rows);
        if verbose {
            output::separator();
            for s in snapshots {
                if let Some(ref checksum) = s.checksum {
                    output::kv(&format!("{} checksum", s.name), checksum);
                }
            }
        }
    }
}

fn display_per_collection_entry(
    name: &str,
    snapshots: &[SnapshotInfo],
    json: bool,
    verbose: bool,
    all_snapshots: &mut Vec<SnapshotInfo>,
) {
    if json {
        all_snapshots.extend(snapshots.iter().cloned());
    } else if snapshots.is_empty() {
        output::info(&format!("  {}: no snapshots", name));
    } else {
        output::info(&format!("  {}: {} snapshot(s)", name, snapshots.len()));
        if verbose {
            let rows: Vec<SnapshotRow> = snapshots.iter().map(SnapshotRow::from).collect();
            output::print_table_auto(&rows);
        }
    }
}
