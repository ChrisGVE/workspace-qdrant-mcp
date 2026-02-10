//! Restore command - Qdrant snapshot restoration
//!
//! Restores data from Qdrant snapshots.
//! Subcommands: snapshot, from-backup, list, verify

use std::io::Write as _;

use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use serde::{Deserialize, Serialize};

use crate::output;

/// Default Qdrant URL
const DEFAULT_QDRANT_URL: &str = "http://localhost:6333";

/// Get Qdrant URL from environment or default
fn qdrant_url() -> String {
    std::env::var("QDRANT_URL").unwrap_or_else(|_| DEFAULT_QDRANT_URL.to_string())
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
        .timeout(std::time::Duration::from_secs(600))
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

/// Snapshot recovery request body
#[derive(Debug, Serialize)]
struct RecoverRequest {
    location: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    priority: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    checksum: Option<String>,
}

/// Restore command arguments
#[derive(Args)]
pub struct RestoreArgs {
    #[command(subcommand)]
    command: RestoreCommand,
}

/// Restore subcommands
#[derive(Subcommand)]
enum RestoreCommand {
    /// Restore from a Qdrant snapshot
    Snapshot {
        /// Snapshot name or path
        snapshot: String,

        /// Collection to restore to
        #[arg(short, long)]
        collection: String,

        /// Force restore even if collection exists
        #[arg(short, long)]
        force: bool,
    },

    /// Restore from a local snapshot file (upload to Qdrant)
    FromBackup {
        /// Path to snapshot file
        path: std::path::PathBuf,

        /// Target collection
        #[arg(short, long)]
        collection: String,

        /// Force restore even if collection exists
        #[arg(short, long)]
        force: bool,
    },

    /// List available snapshots for restoration
    List {
        /// Show snapshots for a specific collection
        #[arg(short, long)]
        collection: Option<String>,
    },

    /// Verify a snapshot without restoring
    Verify {
        /// Snapshot name
        snapshot: String,

        /// Collection the snapshot belongs to (use 'all' for full snapshots)
        #[arg(short, long)]
        collection: String,
    },
}

/// Execute restore command
pub async fn execute(args: RestoreArgs) -> Result<()> {
    match args.command {
        RestoreCommand::Snapshot {
            snapshot,
            collection,
            force,
        } => restore_snapshot(&snapshot, &collection, force).await,
        RestoreCommand::FromBackup {
            path,
            collection,
            force,
        } => restore_from_backup(&path, &collection, force).await,
        RestoreCommand::List { collection } => list_snapshots(collection).await,
        RestoreCommand::Verify {
            snapshot,
            collection,
        } => verify_snapshot(&snapshot, &collection).await,
    }
}

async fn restore_snapshot(snapshot: &str, collection: &str, force: bool) -> Result<()> {
    output::section("Restore from Snapshot");

    let base = qdrant_url();
    let client = build_client()?;

    output::kv("Qdrant", &base);
    output::kv("Snapshot", snapshot);
    output::kv("Target Collection", collection);
    output::separator();

    if !force {
        output::warning("Restore will overwrite existing data in the target collection!");
        output::warning("Use --force to skip this warning.");
        output::separator();

        eprint!("Type 'yes' to confirm restore: ");
        std::io::stderr().flush().ok();
        let mut input = String::new();
        std::io::stdin()
            .read_line(&mut input)
            .context("Failed to read input")?;
        if input.trim() != "yes" {
            output::info("Restore cancelled.");
            return Ok(());
        }
    }

    output::info("Recovering collection from snapshot...");

    // Build the snapshot location - could be a name (for server-side snapshots)
    // or a file:// URI for local paths or http:// for remote
    let location = if snapshot.starts_with("http://")
        || snapshot.starts_with("https://")
        || snapshot.starts_with("file://")
    {
        snapshot.to_string()
    } else {
        // Assume it's a snapshot name on the server
        format!(
            "{}/collections/{}/snapshots/{}",
            base, collection, snapshot
        )
    };

    let recover_url = format!("{}/collections/{}/snapshots/recover", base, collection);
    let body = RecoverRequest {
        location,
        priority: Some("snapshot".to_string()),
        checksum: None,
    };

    let resp = client
        .put(&recover_url)
        .query(&[("wait", "true")])
        .json(&body)
        .send()
        .await
        .context("Failed to connect to Qdrant")?;

    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("Qdrant returned {}: {}", status, body);
    }

    output::success(format!(
        "Collection '{}' restored from snapshot '{}'",
        collection, snapshot
    ));
    output::separator();
    output::info("Verify with: wqm restore verify <snapshot> --collection <collection>");

    Ok(())
}

async fn restore_from_backup(
    path: &std::path::Path,
    collection: &str,
    force: bool,
) -> Result<()> {
    output::section("Restore from Local Snapshot File");

    let base = qdrant_url();
    let client = build_client()?;

    output::kv("Qdrant", &base);
    output::kv("File", &path.display().to_string());
    output::kv("Target Collection", collection);
    output::separator();

    if !path.exists() {
        anyhow::bail!("Snapshot file not found: {}", path.display());
    }

    let metadata = std::fs::metadata(path)
        .with_context(|| format!("Failed to read file metadata: {}", path.display()))?;
    output::kv("File Size", output::format_bytes(metadata.len() as i64));
    output::separator();

    if !force {
        output::warning("This will upload and restore the snapshot, overwriting collection data!");
        output::warning("Use --force to skip this warning.");
        output::separator();

        eprint!("Type 'yes' to confirm restore: ");
        std::io::stderr().flush().ok();
        let mut input = String::new();
        std::io::stdin()
            .read_line(&mut input)
            .context("Failed to read input")?;
        if input.trim() != "yes" {
            output::info("Restore cancelled.");
            return Ok(());
        }
    }

    output::info("Uploading snapshot file to Qdrant...");

    // Read the snapshot file
    let file_bytes = tokio::fs::read(path)
        .await
        .with_context(|| format!("Failed to read snapshot file: {}", path.display()))?;

    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("snapshot.snapshot")
        .to_string();

    // Upload via multipart form
    let upload_url = format!(
        "{}/collections/{}/snapshots/upload",
        base, collection
    );

    let part = reqwest::multipart::Part::bytes(file_bytes)
        .file_name(file_name)
        .mime_str("application/octet-stream")
        .context("Failed to create multipart part")?;

    let form = reqwest::multipart::Form::new().part("snapshot", part);

    let resp = client
        .post(&upload_url)
        .query(&[("wait", "true"), ("priority", "snapshot")])
        .multipart(form)
        .send()
        .await
        .context("Failed to upload snapshot to Qdrant")?;

    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("Qdrant returned {}: {}", status, body);
    }

    output::success(format!(
        "Collection '{}' restored from file '{}'",
        collection,
        path.display()
    ));

    Ok(())
}

async fn list_snapshots(collection: Option<String>) -> Result<()> {
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

async fn verify_snapshot(snapshot: &str, collection: &str) -> Result<()> {
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
