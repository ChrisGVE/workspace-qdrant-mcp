//! Collections reset subcommand handler

use std::io::Write as _;

use anyhow::{Context as _, Result};

use crate::grpc::client::DaemonClient;
use crate::output;

use super::{VALID_COLLECTIONS, build_client, qdrant_url};

pub async fn reset_collections(
    names: Vec<String>,
    include_queue: bool,
    yes: bool,
) -> Result<()> {
    let validated = match validate_collection_names(&names) {
        Some(v) => v,
        None => return Ok(()),
    };

    print_reset_warning(&validated, include_queue);

    if !yes && !confirm_reset(&validated)? {
        return Ok(());
    }

    let daemon_running = try_pause_daemon().await;

    let client = build_client()?;
    let base_url = qdrant_url();
    perform_resets(&client, &base_url, &validated).await?;

    if include_queue {
        println!();
        output::info("Cleaning queue items...");
        match clean_queue_items(&validated).await {
            Ok(count) => output::success(format!("Removed {} queue items", count)),
            Err(e) => output::warning(format!("Queue cleanup failed: {}", e)),
        }
    }

    if daemon_running {
        try_resume_daemon().await;
    }

    println!();
    output::success("Collection reset complete");

    Ok(())
}

/// Validate and deduplicate collection names. Returns `None` on first invalid name.
fn validate_collection_names(names: &[String]) -> Option<Vec<String>> {
    let mut validated = Vec::new();
    for name in names {
        if !VALID_COLLECTIONS.contains(&name.as_str()) {
            output::error(format!(
                "Invalid collection name: '{}'. Valid names: {}",
                name,
                VALID_COLLECTIONS.join(", ")
            ));
            return None;
        }
        if !validated.contains(name) {
            validated.push(name.clone());
        }
    }
    Some(validated)
}

/// Print the destructive-action warning before reset.
fn print_reset_warning(validated: &[String], include_queue: bool) {
    output::section("Collection Reset");
    println!();
    output::warning("This will DELETE and RECREATE the following collections:");
    for name in validated {
        println!("  - {}", name);
    }
    println!();
    output::warning("ALL DATA IN THESE COLLECTIONS WILL BE PERMANENTLY LOST.");
    output::info("The daemon will re-populate from watched folders automatically.");
    if include_queue {
        output::info("Related pending/failed queue items will also be cleaned.");
    }
    println!();
}

/// Prompt the user to type each collection name to confirm. Returns `false` on abort.
fn confirm_reset(validated: &[String]) -> Result<bool> {
    output::info("To confirm, type each collection name exactly:");
    for name in validated {
        print!("  Reset '{}'? Type '{}' to confirm: ", name, name);
        std::io::stdout().flush()?;

        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input != name.as_str() {
            output::error(format!("Expected '{}', got '{}'. Aborting.", name, input));
            return Ok(false);
        }
    }
    println!();
    Ok(true)
}

/// Reset each validated collection in sequence.
async fn perform_resets(
    client: &reqwest::Client,
    base_url: &str,
    validated: &[String],
) -> Result<()> {
    for name in validated {
        print!("  Resetting '{}'... ", name);
        std::io::stdout().flush()?;

        match reset_single_collection(client, base_url, name).await {
            Ok(()) => println!("done."),
            Err(e) => {
                println!("FAILED: {}", e);
                output::error(format!("Failed to reset '{}': {}", name, e));
            }
        }
    }
    Ok(())
}

/// Delete and recreate a single collection with proper vector configuration
async fn reset_single_collection(
    client: &reqwest::Client,
    base_url: &str,
    name: &str,
) -> Result<()> {
    let collection_url = format!("{}/collections/{}", base_url, name);

    // 1. Delete collection (ignore 404 — may not exist)
    let delete_resp = client.delete(&collection_url).send().await?;
    if !delete_resp.status().is_success() && delete_resp.status().as_u16() != 404 {
        let status = delete_resp.status();
        let body = delete_resp.text().await.unwrap_or_default();
        anyhow::bail!("DELETE failed ({}): {}", status, body);
    }

    // 2. Create collection with the same vector config as daemon's create_multi_tenant_collection
    let create_body = serde_json::json!({
        "vectors": {
            "dense": {
                "size": 384,
                "distance": "Cosine",
                "hnsw_config": {
                    "m": 16,
                    "ef_construct": 100
                },
                "on_disk": false
            }
        },
        "sparse_vectors": {
            "sparse": {}
        },
        "shard_number": 1,
        "replication_factor": 1,
        "on_disk_payload": true
    });

    let create_resp = client
        .put(&collection_url)
        .json(&create_body)
        .send()
        .await?;

    if !create_resp.status().is_success() {
        let status = create_resp.status();
        let body = create_resp.text().await.unwrap_or_default();
        anyhow::bail!("CREATE failed ({}): {}", status, body);
    }

    // 3. Create payload indexes based on collection type
    let index_url = format!("{}/collections/{}/index", base_url, name);

    // All collections get project_id index (used for multi-tenancy)
    let project_id_index = serde_json::json!({
        "field_name": "project_id",
        "field_schema": "keyword"
    });
    let _ = client
        .put(&index_url)
        .json(&project_id_index)
        .send()
        .await;

    // Libraries also get library_name index
    if name == wqm_common::constants::COLLECTION_LIBRARIES {
        let library_index = serde_json::json!({
            "field_name": "library_name",
            "field_schema": "keyword"
        });
        let _ = client
            .put(&index_url)
            .json(&library_index)
            .send()
            .await;
    }

    Ok(())
}

/// Clean pending/failed queue items for specified collections
async fn clean_queue_items(collections: &[String]) -> Result<usize> {
    let db_path = crate::config::get_database_path()
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    if !db_path.exists() {
        return Ok(0);
    }

    let conn = rusqlite::Connection::open(&db_path)
        .context("Failed to open state database")?;

    // Build placeholders for IN clause
    let placeholders: Vec<String> = (1..=collections.len())
        .map(|i| format!("?{}", i))
        .collect();
    let sql = format!(
        "DELETE FROM unified_queue WHERE status IN ('pending', 'failed') AND collection IN ({})",
        placeholders.join(", ")
    );

    let params: Vec<&dyn rusqlite::types::ToSql> = collections
        .iter()
        .map(|s| s as &dyn rusqlite::types::ToSql)
        .collect();

    let count = conn
        .execute(&sql, params.as_slice())
        .context("Failed to clean queue items")?;

    Ok(count)
}

/// Try to pause daemon watchers via gRPC. Returns true if daemon was reachable.
async fn try_pause_daemon() -> bool {
    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            match client.system().pause_all_watchers(()).await {
                Ok(_) => {
                    output::info("Daemon watchers paused");
                    true
                }
                Err(e) => {
                    output::warning(format!("Failed to pause watchers: {}", e));
                    false
                }
            }
        }
        Err(_) => {
            output::info("Daemon not running, proceeding with direct Qdrant access");
            false
        }
    }
}

/// Try to resume daemon watchers via gRPC
async fn try_resume_daemon() {
    if let Ok(mut client) = DaemonClient::connect_default().await {
        match client.system().resume_all_watchers(()).await {
            Ok(_) => {
                output::info("Daemon watchers resumed");
            }
            Err(e) => {
                output::warning(format!("Failed to resume watchers: {}", e));
            }
        }
    }
}
