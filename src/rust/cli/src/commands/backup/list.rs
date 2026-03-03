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

            let snapshots = api_resp.result;
            if json {
                output::print_json(&snapshots);
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
                    for s in &snapshots {
                        if let Some(ref checksum) = s.checksum {
                            output::kv(&format!("{} checksum", s.name), checksum);
                        }
                    }
                }
            }
        }
        None => {
            if !json {
                // List full snapshots
                output::info("Full snapshots (all collections):");
            }
            // Collect all snapshots for JSON output
            let mut all_snapshots: Vec<SnapshotInfo> = Vec::new();

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

                if json {
                    all_snapshots.extend(api_resp.result.iter().cloned());
                } else if api_resp.result.is_empty() {
                    output::info("  No full snapshots found.");
                } else {
                    let rows: Vec<SnapshotRow> =
                        api_resp.result.iter().map(SnapshotRow::from).collect();
                    output::print_table_auto(&rows);
                }
            }

            if !json {
                output::separator();

                // List per-collection snapshots
                output::info("Per-collection snapshots:");
            }
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
                    let snap_resp = client.get(&snap_url).send().await;
                    if let Ok(resp) = snap_resp {
                        if resp.status().is_success() {
                            if let Ok(snap_result) =
                                resp.json::<QdrantResponse<Vec<SnapshotInfo>>>().await
                            {
                                if json {
                                    all_snapshots.extend(snap_result.result.iter().cloned());
                                } else if snap_result.result.is_empty() {
                                    output::info(&format!("  {}: no snapshots", entry.name));
                                } else {
                                    output::info(&format!(
                                        "  {}: {} snapshot(s)",
                                        entry.name,
                                        snap_result.result.len()
                                    ));
                                    if verbose {
                                        let rows: Vec<SnapshotRow> = snap_result
                                            .result
                                            .iter()
                                            .map(SnapshotRow::from)
                                            .collect();
                                        output::print_table_auto(&rows);
                                    }
                                }
                            }
                        }
                    }
                }
            }

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
