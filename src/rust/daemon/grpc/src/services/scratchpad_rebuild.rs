//! Scratchpad reconciliation helpers for `rebuild_scratchpad`.
//!
//! Follows the same pattern as `rules_rebuild.rs`: scroll Qdrant, read
//! SQLite mirror, reconcile bidirectionally.

use std::collections::HashMap;
use std::sync::Arc;

use qdrant_client::qdrant::Filter;
use sqlx::SqlitePool;
use tracing::{info, warn};

use workspace_qdrant_core::StorageClient;

use super::rules_rebuild::{extract_point_id_str, extract_str};

/// A scratchpad entry from SQLite mirror.
pub(crate) struct MirrorEntry {
    pub id: String,
    pub title: String,
    pub content: String,
    pub tags: String,
    pub tenant_id: String,
}

/// Step 1: Scroll all scratchpad entries from Qdrant, indexed by scratchpad_id.
pub(crate) async fn scroll_qdrant_scratchpad(
    storage: &Arc<StorageClient>,
) -> Result<HashMap<String, String>, String> {
    match storage.collection_exists("scratchpad").await {
        Ok(false) => {
            info!("[rebuild:scratchpad] Collection does not exist — nothing to reconcile");
            return Err("no_collection".into());
        }
        Err(e) => return Err(format!("Failed to check scratchpad collection: {}", e)),
        Ok(true) => {}
    }

    let all_points = storage
        .scroll_with_filter("scratchpad", Filter::default(), 10000, None)
        .await
        .map_err(|e| format!("Failed to scroll scratchpad from Qdrant: {}", e))?;

    let mut by_id: HashMap<String, String> = HashMap::new();
    for point in &all_points {
        if let Some(point_id) = extract_point_id_str(point) {
            let content = extract_str(point, "content").unwrap_or_default();
            by_id.insert(point_id, content);
        }
    }

    info!(
        "[rebuild:scratchpad] Found {} Qdrant scratchpad entries",
        by_id.len()
    );
    Ok(by_id)
}

/// Step 2: Read all scratchpad entries from SQLite mirror.
pub(crate) async fn read_sqlite_scratchpad(pool: &SqlitePool) -> Result<Vec<MirrorEntry>, String> {
    let rows: Vec<(String, String, String, String, String)> = sqlx::query_as(
        "SELECT scratchpad_id, COALESCE(title, ''), content, \
         COALESCE(tags, '[]'), tenant_id \
         FROM scratchpad_mirror",
    )
    .fetch_all(pool)
    .await
    .map_err(|e| format!("Failed to read scratchpad_mirror: {}", e))?;

    Ok(rows
        .into_iter()
        .map(|(id, title, content, tags, tenant_id)| MirrorEntry {
            id,
            title,
            content,
            tags,
            tenant_id,
        })
        .collect())
}

/// Step 3: Reconcile — enqueue mirror entries missing from Qdrant.
pub(crate) async fn reconcile_scratchpad(
    pool: &SqlitePool,
    qdrant_entries: &HashMap<String, String>,
    mirror_entries: &[MirrorEntry],
) -> u64 {
    let now = wqm_common::timestamps::now_utc();
    let mut enqueued = 0u64;

    for entry in mirror_entries {
        // Check if content already exists in Qdrant (by content match, not ID)
        let already_in_qdrant = qdrant_entries.values().any(|q| q == &entry.content);

        if !already_in_qdrant {
            enqueue_scratchpad_ingestion(pool, entry, &now).await;
            enqueued += 1;
        }
    }

    if enqueued > 0 {
        info!(
            "[rebuild:scratchpad] Enqueued {} mirror entries for re-ingestion",
            enqueued
        );
    } else {
        info!("[rebuild:scratchpad] All mirror entries already in Qdrant");
    }

    enqueued
}

/// Enqueue a single scratchpad entry for re-ingestion into Qdrant.
async fn enqueue_scratchpad_ingestion(pool: &SqlitePool, entry: &MirrorEntry, now: &str) {
    let payload = serde_json::json!({
        "content": entry.content,
        "source_type": "scratchpad",
        "title": entry.title,
        "tags": entry.tags,
    });
    let payload_str = payload.to_string();
    let idem_key = wqm_common::hashing::compute_content_hash(&format!(
        "text|add|{}|scratchpad|{}",
        entry.tenant_id, payload_str
    ));

    if let Err(e) = sqlx::query(
        "INSERT OR IGNORE INTO unified_queue \
         (queue_id, idempotency_key, item_type, op, tenant_id, collection, \
          status, payload_json, created_at, updated_at) \
         VALUES (?1, ?2, 'text', 'add', ?3, 'scratchpad', 'pending', ?4, ?5, ?6)",
    )
    .bind(uuid::Uuid::new_v4().to_string())
    .bind(&idem_key[..32])
    .bind(&entry.tenant_id)
    .bind(&payload_str)
    .bind(now)
    .bind(now)
    .execute(pool)
    .await
    {
        warn!(
            "[rebuild:scratchpad] Failed to enqueue entry {}: {}",
            entry.id, e
        );
    }
}
