//! `restore snapshot` subcommand handler.

use std::io::Write as _;

use anyhow::{Context, Result};

use crate::output;

use super::client::{build_client, qdrant_url, RecoverRequest};

/// Restore a collection from a named Qdrant snapshot (server-side or URI).
pub async fn restore_snapshot(snapshot: &str, collection: &str, force: bool) -> Result<()> {
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
