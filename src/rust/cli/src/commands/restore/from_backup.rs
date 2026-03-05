//! `restore from-backup` subcommand handler.

use std::io::Write as _;

use anyhow::{Context, Result};

use crate::output;

use super::client::{build_client, qdrant_url};

/// Prompt the user to confirm a destructive restore.
///
/// Returns `Ok(true)` if confirmed, `Ok(false)` if cancelled.
fn prompt_confirmation() -> Result<bool> {
    output::warning("This will upload and restore the snapshot, overwriting collection data!");
    output::warning("Use --force to skip this warning.");
    output::separator();

    eprint!("Type 'yes' to confirm restore: ");
    std::io::stderr().flush().ok();
    let mut input = String::new();
    std::io::stdin()
        .read_line(&mut input)
        .context("Failed to read input")?;
    Ok(input.trim() == "yes")
}

/// Upload a local snapshot file to Qdrant and restore a collection from it.
pub async fn restore_from_backup(
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

    if !force && !prompt_confirmation()? {
        output::info("Restore cancelled.");
        return Ok(());
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
    let upload_url = format!("{}/collections/{}/snapshots/upload", base, collection);

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
