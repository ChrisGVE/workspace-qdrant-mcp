//! Collections command - Qdrant collection management
//!
//! Provides per-collection reset and listing of Qdrant collections.
//! Subcommands: list, reset
//!
//! Note: Tenant rename moved to `admin rename-tenant` command.

use std::io::Write as _;

use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use serde::Deserialize;

use crate::grpc::client::DaemonClient;
use crate::output;

/// Canonical collection names (validated against wqm-common constants)
const VALID_COLLECTIONS: &[&str] = &[
    wqm_common::constants::COLLECTION_PROJECTS,
    wqm_common::constants::COLLECTION_LIBRARIES,
    wqm_common::constants::COLLECTION_MEMORY,
];

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
        .timeout(std::time::Duration::from_secs(60))
        .build()
        .context("Failed to build HTTP client")
}

/// Collections command arguments
#[derive(Args)]
pub struct CollectionsArgs {
    #[command(subcommand)]
    command: CollectionsCommand,
}

/// Collections subcommands
#[derive(Subcommand)]
enum CollectionsCommand {
    /// List Qdrant collections
    List,

    /// Reset (delete and recreate) specific collection(s)
    Reset {
        /// Collection name(s) to reset (projects, libraries, memory)
        #[arg(required = true)]
        names: Vec<String>,

        /// Also clean related pending/failed queue items from SQLite
        #[arg(long)]
        include_queue: bool,

        /// Skip confirmation prompts
        #[arg(short, long)]
        yes: bool,
    },

}

/// Execute collections command
pub async fn execute(args: CollectionsArgs) -> Result<()> {
    match args.command {
        CollectionsCommand::List => list_collections().await,
        CollectionsCommand::Reset {
            names,
            include_queue,
            yes,
        } => reset_collections(names, include_queue, yes).await,
    }
}

/// Qdrant collection info from list endpoint
#[derive(Debug, Deserialize)]
struct CollectionDescription {
    name: String,
}

/// Qdrant list collections response
#[derive(Debug, Deserialize)]
struct CollectionsListResult {
    collections: Vec<CollectionDescription>,
}

/// Qdrant API response wrapper
#[derive(Debug, Deserialize)]
struct QdrantResponse<T> {
    #[allow(dead_code)]
    status: Option<String>,
    result: T,
}

async fn list_collections() -> Result<()> {
    output::section("Qdrant Collections");

    let client = build_client()?;
    let url = format!("{}/collections", qdrant_url());

    match client.get(&url).send().await {
        Ok(resp) if resp.status().is_success() => {
            let body: QdrantResponse<CollectionsListResult> = resp
                .json()
                .await
                .context("Failed to parse Qdrant response")?;

            if body.result.collections.is_empty() {
                output::info("No collections found");
                return Ok(());
            }

            for col in &body.result.collections {
                let is_canonical = VALID_COLLECTIONS.contains(&col.name.as_str());
                let label = if is_canonical { "(canonical)" } else { "" };
                output::kv(&col.name, label);
            }

            output::separator();
            output::info(format!(
                "Total: {} collections",
                body.result.collections.len()
            ));
        }
        Ok(resp) => {
            output::error(format!(
                "Qdrant returned status {}: {}",
                resp.status(),
                resp.text().await.unwrap_or_default()
            ));
        }
        Err(e) => {
            output::error(format!(
                "Failed to connect to Qdrant at {}: {}",
                qdrant_url(),
                e
            ));
        }
    }

    Ok(())
}

async fn reset_collections(names: Vec<String>, include_queue: bool, yes: bool) -> Result<()> {
    // Validate all collection names
    let mut validated = Vec::new();
    for name in &names {
        if !VALID_COLLECTIONS.contains(&name.as_str()) {
            output::error(format!(
                "Invalid collection name: '{}'. Valid names: {}",
                name,
                VALID_COLLECTIONS.join(", ")
            ));
            return Ok(());
        }
        if !validated.contains(name) {
            validated.push(name.clone());
        }
    }

    // Display warning
    output::section("Collection Reset");
    println!();
    output::warning("This will DELETE and RECREATE the following collections:");
    for name in &validated {
        println!("  - {}", name);
    }
    println!();
    output::warning("ALL DATA IN THESE COLLECTIONS WILL BE PERMANENTLY LOST.");
    output::info("The daemon will re-populate from watched folders automatically.");
    if include_queue {
        output::info("Related pending/failed queue items will also be cleaned.");
    }
    println!();

    // Confirmation
    if !yes {
        output::info("To confirm, type each collection name exactly:");
        for name in &validated {
            print!("  Reset '{}'? Type '{}' to confirm: ", name, name);
            std::io::stdout().flush()?;

            let mut input = String::new();
            std::io::stdin().read_line(&mut input)?;
            let input = input.trim();

            if input != name.as_str() {
                output::error(format!("Expected '{}', got '{}'. Aborting.", name, input));
                return Ok(());
            }
        }
        println!();
    }

    // Try to pause daemon watchers (non-fatal if daemon is not running)
    let daemon_running = try_pause_daemon().await;

    let client = build_client()?;
    let base_url = qdrant_url();

    // Reset each collection
    for name in &validated {
        print!("  Resetting '{}'... ", name);
        std::io::stdout().flush()?;

        match reset_single_collection(&client, &base_url, name).await {
            Ok(()) => {
                println!("done.");
            }
            Err(e) => {
                println!("FAILED: {}", e);
                output::error(format!("Failed to reset '{}': {}", name, e));
            }
        }
    }

    // Clean queue items if requested
    if include_queue {
        println!();
        output::info("Cleaning queue items...");
        match clean_queue_items(&validated).await {
            Ok(count) => {
                output::success(format!("Removed {} queue items", count));
            }
            Err(e) => {
                output::warning(format!("Queue cleanup failed: {}", e));
            }
        }
    }

    // Resume daemon watchers
    if daemon_running {
        try_resume_daemon().await;
    }

    println!();
    output::success("Collection reset complete");

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


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_collections() {
        assert!(VALID_COLLECTIONS.contains(&"projects"));
        assert!(VALID_COLLECTIONS.contains(&"libraries"));
        assert!(VALID_COLLECTIONS.contains(&"memory"));
        assert!(!VALID_COLLECTIONS.contains(&"invalid"));
    }

    #[test]
    fn test_qdrant_url_default() {
        // When QDRANT_URL is not set, should use default
        std::env::remove_var("QDRANT_URL");
        let url = qdrant_url();
        assert!(url.starts_with("http"));
        assert!(url.contains("6333"));
    }
}
