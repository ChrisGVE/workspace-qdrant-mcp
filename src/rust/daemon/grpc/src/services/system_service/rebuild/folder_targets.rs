//! Rebuild handlers for folder-backed targets: watch_folders (projects/libraries)
//! and components.

use tracing::{error, info};

/// Rescan watch folders by enqueuing scan operations.
pub(super) async fn rebuild_watch_folders(
    db_pool: Option<&sqlx::SqlitePool>,
    collection: &str,
    tenant_id: Option<&str>,
) {
    let Some(pool) = db_pool else {
        error!("[rebuild:{}] Database pool not configured", collection);
        return;
    };

    // Fetch watch folders for the given collection (and optional tenant)
    let folders: Vec<(String, String, String)> = if let Some(tid) = tenant_id {
        match sqlx::query_as::<_, (String, String, String)>(
            "SELECT watch_id, tenant_id, path FROM watch_folders \
             WHERE collection = ?1 AND tenant_id = ?2 AND enabled = 1",
        )
        .bind(collection)
        .bind(tid)
        .fetch_all(pool)
        .await
        {
            Ok(rows) => rows,
            Err(e) => {
                error!(
                    "[rebuild:{}] Failed to fetch watch folders: {}",
                    collection, e
                );
                return;
            }
        }
    } else {
        match sqlx::query_as::<_, (String, String, String)>(
            "SELECT watch_id, tenant_id, path FROM watch_folders \
             WHERE collection = ?1 AND enabled = 1",
        )
        .bind(collection)
        .fetch_all(pool)
        .await
        {
            Ok(rows) => rows,
            Err(e) => {
                error!(
                    "[rebuild:{}] Failed to fetch watch folders: {}",
                    collection, e
                );
                return;
            }
        }
    };

    if folders.is_empty() {
        info!("[rebuild:{}] No enabled watch folders found", collection);
        return;
    }

    // Enqueue scan operations for each watch folder
    let now = wqm_common::timestamps::now_utc();
    let mut enqueued = 0u64;
    for (watch_id, tid, path) in &folders {
        let payload = serde_json::json!({
            "path": path,
            "watch_id": watch_id,
            "rebuild": true,
        });
        let payload_str = payload.to_string();
        let idem_key = wqm_common::hashing::compute_content_hash(&format!(
            "tenant|scan|{}|{}|{}",
            tid, collection, payload_str
        ));

        let _ = sqlx::query(
            "INSERT OR IGNORE INTO unified_queue \
             (queue_id, idempotency_key, item_type, op, tenant_id, collection, priority, status, payload_json, created_at, updated_at) \
             VALUES (?1, ?2, 'tenant', 'scan', ?3, ?4, 5, 'pending', ?5, ?6, ?7)",
        )
        .bind(uuid::Uuid::new_v4().to_string())
        .bind(&idem_key[..32])
        .bind(tid)
        .bind(collection)
        .bind(&payload_str)
        .bind(&now)
        .bind(&now)
        .execute(pool)
        .await;
        enqueued += 1;
    }

    info!(
        "[rebuild:{}] Enqueued {} scan operations for watch folders",
        collection, enqueued
    );
}

/// Re-detect and assign workspace components to tracked files.
pub(super) async fn rebuild_components(
    db_pool: Option<&sqlx::SqlitePool>,
    tenant_id: Option<&str>,
    force: bool,
) {
    let Some(pool) = db_pool else {
        error!(target = "components", "Database pool not configured");
        return;
    };

    let start = std::time::Instant::now();

    match workspace_qdrant_core::component_detection::backfill_components(
        pool, 200, force, tenant_id,
    )
    .await
    {
        Ok(stats) => {
            info!(
                target = "components",
                folders = stats.folders_processed,
                updated = stats.files_updated,
                unmatched = stats.files_unmatched,
                errors = stats.errors,
                force,
                duration_ms = start.elapsed().as_millis() as u64,
                "Component rebuild complete"
            );
        }
        Err(e) => {
            error!(target = "components", error = %e, "Component rebuild failed");
        }
    }
}
