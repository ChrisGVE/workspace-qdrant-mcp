//! Cleanup-orphans subcommand handler

use anyhow::Result;

use super::super::qdrant_helpers;
use super::ALL_COLLECTIONS;
use crate::grpc::client::DaemonClient;
use crate::grpc::proto::{QueueType, RefreshSignalRequest};
use crate::output;
use crate::queue::{ItemType, QueueOperation, UnifiedQueueClient};

/// Detect and optionally delete orphaned tenants across collections.
pub async fn execute(delete: bool, collection_filter: Option<String>) -> Result<()> {
    output::section("Orphan Detection & Cleanup");

    let collections: Vec<&str> = if let Some(ref c) = collection_filter {
        if !ALL_COLLECTIONS.contains(&c.as_str()) {
            output::error(format!(
                "Unknown collection '{}'. Valid: {}",
                c,
                ALL_COLLECTIONS.join(", ")
            ));
            return Ok(());
        }
        vec![c.as_str()]
    } else {
        ALL_COLLECTIONS.to_vec()
    };

    let conn = match qdrant_helpers::open_state_db() {
        Ok(c) => Some(c),
        Err(_) => {
            output::warning(
                "Could not open state database. All Qdrant tenants will appear as orphans.",
            );
            None
        }
    };

    let http_client = qdrant_helpers::build_qdrant_http_client()?;
    let base_url = qdrant_helpers::qdrant_base_url();

    let total_orphans =
        scan_collections_for_orphans(&collections, &conn, &http_client, &base_url).await?;

    output::separator();

    if total_orphans.is_empty() {
        output::success("No orphaned tenants found across any collection.");
        return Ok(());
    }

    output::warning(format!("Found {} orphan(s):", total_orphans.len()));
    for (coll, tenant) in &total_orphans {
        output::kv(format!("  {}", coll), tenant);
    }

    if !delete {
        output::separator();
        output::info("To delete orphaned points, run: wqm admin cleanup-orphans --delete");
        return Ok(());
    }

    output::separator();
    output::warning("This will delete ALL Qdrant points for the orphaned tenants listed above.");
    if !output::confirm("Proceed with deletion?") {
        output::info("Aborted.");
        return Ok(());
    }

    enqueue_orphan_deletions(&total_orphans).await
}

/// Scan collections to find orphaned tenants (in Qdrant but not in SQLite).
async fn scan_collections_for_orphans(
    collections: &[&str],
    conn: &Option<rusqlite::Connection>,
    http_client: &reqwest::Client,
    base_url: &str,
) -> Result<Vec<(String, String)>> {
    use std::collections::HashSet;

    let mut total_orphans: Vec<(String, String)> = Vec::new();

    for collection in collections {
        let tenant_field = qdrant_helpers::tenant_field_for_collection(collection);
        output::info(format!(
            "Scanning {} (field: {})...",
            collection, tenant_field
        ));

        let qdrant_tenants = qdrant_helpers::scroll_unique_field_values(
            http_client,
            base_url,
            collection,
            tenant_field,
        )
        .await?;

        if qdrant_tenants.is_empty() {
            output::kv(format!("  {} tenants in Qdrant", collection), "0");
            continue;
        }

        let known_tenants = if let Some(ref c) = conn {
            qdrant_helpers::get_known_tenants_for_collection(c, collection)?
        } else {
            HashSet::new()
        };

        output::kv(
            format!("  {} Qdrant", collection),
            qdrant_tenants.len().to_string(),
        );
        output::kv(
            format!("  {} SQLite", collection),
            known_tenants.len().to_string(),
        );

        let mut orphans: Vec<&String> = qdrant_tenants
            .iter()
            .filter(|t| !known_tenants.contains(*t))
            .collect();
        orphans.sort();

        for tenant in &orphans {
            total_orphans.push((collection.to_string(), tenant.to_string()));
        }
    }

    Ok(total_orphans)
}

/// Enqueue deletion operations for orphaned tenants and signal the daemon.
async fn enqueue_orphan_deletions(orphans: &[(String, String)]) -> Result<()> {
    let queue_client = match UnifiedQueueClient::connect() {
        Ok(c) => c,
        Err(e) => {
            output::error(format!("Could not connect to queue: {}", e));
            return Ok(());
        }
    };

    let mut queued = 0;
    for (coll, tenant) in orphans {
        let payload_json = serde_json::json!({ "tenant_id_to_delete": tenant }).to_string();

        match queue_client.enqueue(
            ItemType::Tenant,
            QueueOperation::Delete,
            tenant,
            coll,
            &payload_json,
            "",
            None,
        ) {
            Ok(result) => {
                if result.was_duplicate {
                    output::info(format!("  {} / {} — already queued", coll, tenant));
                } else {
                    output::success(format!(
                        "  {} / {} — deletion queued ({})",
                        coll, tenant, result.queue_id
                    ));
                }
                queued += 1;
            }
            Err(e) => {
                output::error(format!(
                    "  {} / {} — failed to enqueue: {}",
                    coll, tenant, e
                ));
            }
        }
    }

    output::separator();
    output::success(format!(
        "Queued deletion for {} orphan(s). Daemon will process.",
        queued
    ));

    if let Ok(mut client) = DaemonClient::connect_default().await {
        let request = RefreshSignalRequest {
            queue_type: QueueType::IngestQueue as i32,
            lsp_languages: vec![],
            grammar_languages: vec![],
        };
        if client.system().send_refresh_signal(request).await.is_ok() {
            output::success("Daemon notified to process deletions");
        }
    }

    Ok(())
}
