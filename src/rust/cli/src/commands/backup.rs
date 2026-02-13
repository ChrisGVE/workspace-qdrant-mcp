//! Backup command - Qdrant snapshot management
//!
//! Creates and manages Qdrant snapshots for data backup.
//! Subcommands: create, list, delete
//!
//! For restoration, use the separate 'restore' command.

use std::io::Write as _;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use serde::Deserialize;
use tabled::Tabled;

use crate::output;

/// Get Qdrant URL from environment or default
fn qdrant_url() -> String {
    std::env::var("QDRANT_URL")
        .unwrap_or_else(|_| wqm_common::constants::DEFAULT_QDRANT_URL.to_string())
}

/// Get optional Qdrant API key
fn qdrant_api_key() -> Option<String> {
    std::env::var("QDRANT_API_KEY").ok()
}

/// Build a reqwest client with optional API key header
fn build_client() -> Result<reqwest::Client> {
    let mut headers = reqwest::header::HeaderMap::new();
    if let Some(key) = qdrant_api_key() {
        headers.insert(
            "api-key",
            reqwest::header::HeaderValue::from_str(&key)
                .context("Invalid QDRANT_API_KEY value")?,
        );
    }
    reqwest::Client::builder()
        .default_headers(headers)
        .timeout(std::time::Duration::from_secs(300))
        .build()
        .context("Failed to build HTTP client")
}

/// Qdrant API response wrapper
#[derive(Debug, Deserialize)]
struct QdrantResponse<T> {
    #[allow(dead_code)]
    status: Option<String>,
    result: T,
    #[allow(dead_code)]
    time: Option<f64>,
}

/// Snapshot metadata from Qdrant
#[derive(Debug, Deserialize, Clone)]
struct SnapshotInfo {
    name: String,
    size: i64,
    creation_time: Option<String>,
    checksum: Option<String>,
}

/// Displayable snapshot row for table output
#[derive(Tabled)]
struct SnapshotRow {
    #[tabled(rename = "Name")]
    name: String,
    #[tabled(rename = "Size")]
    size: String,
    #[tabled(rename = "Created")]
    created: String,
    #[tabled(rename = "Checksum")]
    checksum: String,
}

impl From<&SnapshotInfo> for SnapshotRow {
    fn from(s: &SnapshotInfo) -> Self {
        Self {
            name: s.name.clone(),
            size: output::format_bytes(s.size),
            created: s.creation_time.clone().unwrap_or_else(|| "unknown".into()),
            checksum: s
                .checksum
                .clone()
                .unwrap_or_else(|| "none".into()),
        }
    }
}

/// Collection list entry from Qdrant
#[derive(Debug, Deserialize)]
struct CollectionEntry {
    name: String,
}

/// Collections result from Qdrant
#[derive(Debug, Deserialize)]
struct CollectionsResult {
    collections: Vec<CollectionEntry>,
}

/// Backup command arguments
#[derive(Args)]
pub struct BackupArgs {
    #[command(subcommand)]
    command: BackupCommand,
}

/// Backup subcommands
#[derive(Subcommand)]
enum BackupCommand {
    /// Create a new snapshot
    Create {
        /// Collection name (or 'all' for full backup)
        #[arg(short, long, default_value = "all")]
        collection: String,

        /// Output directory to download snapshot to
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Add description/label to the snapshot
        #[arg(short, long)]
        description: Option<String>,
    },

    /// List existing snapshots
    List {
        /// Collection name (optional, shows all if omitted)
        collection: Option<String>,

        /// Show detailed information
        #[arg(short, long)]
        verbose: bool,
    },

    /// Delete a snapshot
    Delete {
        /// Snapshot name
        snapshot: String,

        /// Collection the snapshot belongs to (use 'all' for full snapshots)
        #[arg(short, long)]
        collection: String,

        /// Force deletion without confirmation
        #[arg(short, long)]
        force: bool,
    },
}

/// Execute backup command
pub async fn execute(args: BackupArgs) -> Result<()> {
    match args.command {
        BackupCommand::Create {
            collection,
            output,
            description,
        } => create_backup(&collection, output, description).await,
        BackupCommand::List { collection, verbose } => list_backups(collection, verbose).await,
        BackupCommand::Delete {
            snapshot,
            collection,
            force,
        } => delete_backup(&snapshot, &collection, force).await,
    }
}

async fn create_backup(
    collection: &str,
    output_dir: Option<PathBuf>,
    _description: Option<String>,
) -> Result<()> {
    output::section("Create Backup");

    let base = qdrant_url();
    let client = build_client()?;

    output::kv("Qdrant", &base);
    output::kv("Collection", collection);
    output::separator();

    let snapshot: SnapshotInfo = if collection == "all" {
        output::info("Creating full Qdrant snapshot (all collections)...");
        let url = format!("{}/snapshots", base);
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
        api_resp.result
    } else {
        output::info(&format!("Creating snapshot for collection '{}'...", collection));
        let url = format!("{}/collections/{}/snapshots", base, collection);
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
        api_resp.result
    };

    output::separator();
    output::success(format!("Snapshot created: {}", snapshot.name));
    output::kv("Size", output::format_bytes(snapshot.size));
    if let Some(ref t) = snapshot.creation_time {
        output::kv("Created", t);
    }
    if let Some(ref c) = snapshot.checksum {
        output::kv("Checksum", c);
    }

    // Download snapshot if output directory specified
    if let Some(dir) = output_dir {
        output::separator();
        output::info(&format!("Downloading snapshot to {}...", dir.display()));

        std::fs::create_dir_all(&dir)
            .with_context(|| format!("Failed to create output directory: {}", dir.display()))?;

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

        output::success(format!("Downloaded to: {}", dest.display()));
    }

    output::separator();
    output::info("To restore: wqm restore snapshot <name> --collection <collection>");

    Ok(())
}

async fn list_backups(collection: Option<String>, verbose: bool) -> Result<()> {
    output::section("List Backups");

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

            let snapshots = api_resp.result;
            if snapshots.is_empty() {
                output::info("No snapshots found for this collection.");
            } else {
                output::info(&format!("Found {} snapshot(s):", snapshots.len()));
                let rows: Vec<SnapshotRow> = snapshots.iter().map(SnapshotRow::from).collect();
                output::print_table(&rows);

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
            // List full snapshots
            output::info("Full snapshots (all collections):");
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
                    let rows: Vec<SnapshotRow> =
                        api_resp.result.iter().map(SnapshotRow::from).collect();
                    output::print_table(&rows);
                }
            }

            output::separator();

            // List per-collection snapshots
            output::info("Per-collection snapshots:");
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
                                if snap_result.result.is_empty() {
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
                                        output::print_table(&rows);
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

async fn delete_backup(snapshot: &str, collection: &str, force: bool) -> Result<()> {
    output::section("Delete Backup");

    output::kv("Snapshot", snapshot);
    output::kv("Collection", collection);
    output::separator();

    if !force {
        output::warning("This will permanently delete the snapshot!");
        output::warning("Use --force to skip this warning.");
        output::separator();

        // Prompt for confirmation
        eprint!("Type 'yes' to confirm deletion: ");
        std::io::stderr().flush().ok();
        let mut input = String::new();
        std::io::stdin()
            .read_line(&mut input)
            .context("Failed to read input")?;
        if input.trim() != "yes" {
            output::info("Deletion cancelled.");
            return Ok(());
        }
    }

    let base = qdrant_url();
    let client = build_client()?;

    let url = if collection == "all" {
        format!("{}/snapshots/{}", base, snapshot)
    } else {
        format!(
            "{}/collections/{}/snapshots/{}",
            base, collection, snapshot
        )
    };

    output::info(&format!("Deleting snapshot '{}'...", snapshot));

    let resp = client
        .delete(&url)
        .query(&[("wait", "true")])
        .send()
        .await
        .context("Failed to connect to Qdrant")?;

    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("Qdrant returned {}: {}", status, body);
    }

    output::success(format!("Snapshot '{}' deleted successfully.", snapshot));

    Ok(())
}
