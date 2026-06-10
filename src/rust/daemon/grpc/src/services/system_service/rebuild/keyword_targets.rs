//! Rebuild handler for the keywords/tags target.
//!
//! Clears existing keyword and tag rows then re-enqueues uplift operations
//! so the pipeline re-extracts them on the next processing cycle.

use tracing::{error, info, warn};

/// Re-extract keywords/tags by enqueuing uplift operations.
pub(super) async fn rebuild_keywords(
    db_pool: Option<&sqlx::SqlitePool>,
    tenant_id: Option<&str>,
    collection: &str,
) {
    let Some(pool) = db_pool else {
        error!(target = "keywords", "Database pool not configured");
        return;
    };

    let start = std::time::Instant::now();

    let files = match fetch_keyword_files(pool, tenant_id, collection).await {
        Some(f) => f,
        None => return,
    };

    if files.is_empty() {
        info!(target = "keywords", collection, "No files found");
        return;
    }

    delete_keyword_data(pool, tenant_id, collection, &files).await;

    // Enqueue uplift operations for all files
    let now = wqm_common::timestamps::now_utc();
    let mut enqueued = 0u64;
    for (_file_id, relative_path, tid) in &files {
        let Some(payload_str) = build_uplift_payload(relative_path) else {
            continue;
        };
        let idem_key = wqm_common::hashing::compute_content_hash(&format!(
            "file|uplift|{}|{}|{}",
            tid, collection, payload_str
        ));

        let _ = sqlx::query(
            "INSERT OR IGNORE INTO unified_queue \
             (queue_id, idempotency_key, item_type, op, tenant_id, collection, priority, status, payload_json, created_at, updated_at) \
             VALUES (?1, ?2, 'file', 'uplift', ?3, ?4, 3, 'pending', ?5, ?6, ?7)",
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
        target = "keywords",
        enqueued,
        collection,
        duration_ms = start.elapsed().as_millis() as u64,
        "Cleared keyword/tag data and enqueued uplift operations"
    );
}

/// Build the uplift `payload_json` for one tracked file.
///
/// The queue consumer parses this as `FilePayload`, whose `file_path` field
/// is a validating `wqm_common::paths::RelativePath` — an absolute or
/// otherwise malformed path fails deserialization there and poisons the item
/// as permanent. Validate at the producer instead and skip bad rows.
fn build_uplift_payload(relative_path: &str) -> Option<String> {
    match wqm_common::paths::RelativePath::from_user_input(relative_path) {
        Ok(rel) => Some(serde_json::json!({ "file_path": rel, "rebuild": true }).to_string()),
        Err(e) => {
            warn!(
                target = "keywords",
                relative_path,
                error = %e,
                "Skipping uplift enqueue: invalid relative path in tracked_files"
            );
            None
        }
    }
}

/// Fetch all tracked files for the keyword rebuild scope.
///
/// Returns `None` on a query error (already logged), `Some(vec)` otherwise.
/// Rows are `(file_id, relative_path, tenant_id)` — `relative_path` is the
/// post-v37 canonical column, anchored to the owning watch_folder root.
async fn fetch_keyword_files(
    pool: &sqlx::SqlitePool,
    tenant_id: Option<&str>,
    collection: &str,
) -> Option<Vec<(i64, String, String)>> {
    if let Some(tid) = tenant_id {
        match sqlx::query_as::<_, (i64, String, String)>(
            "SELECT tf.file_id, tf.relative_path, wf.tenant_id \
             FROM tracked_files tf JOIN watch_folders wf ON tf.watch_folder_id = wf.watch_id \
             WHERE wf.tenant_id = ?1 AND wf.collection = ?2",
        )
        .bind(tid)
        .bind(collection)
        .fetch_all(pool)
        .await
        {
            Ok(rows) => Some(rows),
            Err(e) => {
                error!(target = "keywords", error = %e, "Failed to fetch files");
                None
            }
        }
    } else {
        match sqlx::query_as::<_, (i64, String, String)>(
            "SELECT tf.file_id, tf.relative_path, wf.tenant_id \
             FROM tracked_files tf JOIN watch_folders wf ON tf.watch_folder_id = wf.watch_id \
             WHERE wf.collection = ?1",
        )
        .bind(collection)
        .fetch_all(pool)
        .await
        {
            Ok(rows) => Some(rows),
            Err(e) => {
                error!(target = "keywords", error = %e, "Failed to fetch files");
                None
            }
        }
    }
}

/// Delete existing keyword/tag rows for the given scope before re-enqueue.
async fn delete_keyword_data(
    pool: &sqlx::SqlitePool,
    tenant_id: Option<&str>,
    collection: &str,
    files: &[(i64, String, String)],
) {
    if tenant_id.is_some() {
        // Tenant-scoped: delete by file IDs
        for (file_id, _, _) in files {
            let _ = sqlx::query("DELETE FROM keywords WHERE file_id = ?1")
                .bind(file_id)
                .execute(pool)
                .await;
            let _ = sqlx::query("DELETE FROM tags WHERE file_id = ?1")
                .bind(file_id)
                .execute(pool)
                .await;
        }
    } else {
        // Collection-wide: bulk delete
        let _ = sqlx::query("DELETE FROM keywords WHERE collection = ?1")
            .bind(collection)
            .execute(pool)
            .await;
        let _ = sqlx::query("DELETE FROM tags WHERE collection = ?1")
            .bind(collection)
            .execute(pool)
            .await;
        let _ = sqlx::query("DELETE FROM keyword_baskets WHERE collection = ?1")
            .bind(collection)
            .execute(pool)
            .await;
    }
}

#[cfg(test)]
mod tests {
    use super::build_uplift_payload;
    use wqm_common::payloads::FilePayload;

    #[test]
    fn uplift_payload_round_trips_through_file_payload() {
        // The consumer parses the enqueued payload as `FilePayload`; this
        // round-trip is the contract the producer must satisfy.
        let payload = build_uplift_payload("src/api/server.rs").expect("valid relative path");
        assert!(payload.contains("\"rebuild\":true"));

        let parsed: FilePayload =
            serde_json::from_str(&payload).expect("consumer must parse the produced payload");
        assert_eq!(parsed.file_path.as_str(), "src/api/server.rs");
    }

    #[test]
    fn uplift_payload_skips_absolute_path_rows() {
        // An absolute path would fail FilePayload deserialization at the
        // consumer and poison the item as permanent; the producer must skip
        // such rows instead of enqueuing them.
        assert!(build_uplift_payload("/home/user/repo/src/main.rs").is_none());
    }
}
