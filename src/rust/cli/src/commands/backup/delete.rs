//! Backup delete subcommand handler.

use std::io::Write as _;

use anyhow::{Context, Result};

use crate::output;

use super::types::{build_client, qdrant_url};

pub async fn delete_backup(
    snapshot: &str,
    collection: &str,
    force: bool,
    json: bool,
) -> Result<()> {
    if !json {
        output::section("Delete Backup");

        output::kv("Snapshot", snapshot);
        output::kv("Collection", collection);
        output::separator();
    }

    if !force && !json {
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

    if !json {
        output::info(&format!("Deleting snapshot '{}'...", snapshot));
    }

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

    if json {
        output::print_json(&serde_json::json!({
            "deleted": true,
            "snapshot": snapshot,
            "collection": collection,
        }));
    } else {
        output::success(format!("Snapshot '{}' deleted successfully.", snapshot));
    }

    Ok(())
}
