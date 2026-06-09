//! Filesystem reconciliation — detect tracked files that no longer exist on
//! disk and enqueue delete operations to clean them up.

use async_trait::async_trait;
use sqlx::Row;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use crate::idle::task::{MaintenanceContext, MaintenanceResult, MaintenanceTask};
use crate::idle::IdleState;
use crate::unified_queue_schema::{ItemType, QueueOperation};

/// Batch SELECT joining tracked files to their watch folder.
///
/// `watch_folders` is keyed by `watch_id` (not `id`); `tracked_files`
/// references it via `watch_folder_id`. Kept as a named const so the join
/// is exercised by a schema-backed test and cannot silently drift from the
/// real schema again.
const RECONCILE_BATCH_QUERY: &str = "SELECT tf.file_id, tf.relative_path, COALESCE(tf.primary_branch, 'default') AS branch, tf.collection,
                    COALESCE(tf.chunk_count, 0) AS chunk_count,
                    wf.tenant_id, wf.path AS watch_path
             FROM tracked_files tf
             JOIN watch_folders wf ON tf.watch_folder_id = wf.watch_id
             ORDER BY tf.file_id
             LIMIT ?1 OFFSET ?2";

/// Batch-checks tracked files against the filesystem.
///
/// Runs in `FullIdle` or `QdrantDownIdle` (only needs disk + SQLite).
/// Missing files get a delete operation enqueued so the normal pipeline
/// handles Qdrant cleanup when available.
pub struct FilesystemReconcileTask {
    batch_size: i64,
    offset: i64,
    total_checked: u64,
    files_missing: u64,
    files_oversized: u64,
}

impl FilesystemReconcileTask {
    pub fn new() -> Self {
        Self {
            batch_size: 100,
            offset: 0,
            total_checked: 0,
            files_missing: 0,
            files_oversized: 0,
        }
    }

    /// True when the on-disk file exceeds its per-extension ingestion size
    /// limit (the same gate the ingestion path applies, #121). Extensions
    /// without a configured limit are unbounded → never oversized. A failed
    /// stat returns false (the missing-file branch handles non-existent paths).
    fn is_oversized(
        &self,
        ctx: &MaintenanceContext<'_>,
        abs_path: &std::path::Path,
        relative_path: &str,
    ) -> bool {
        let Ok(size) = std::fs::metadata(abs_path).map(|m| m.len()) else {
            return false;
        };
        let ext = crate::file_classification::get_extension_for_storage(std::path::Path::new(
            relative_path,
        ))
        .unwrap_or_default();
        ctx.ingestion_limits
            .size_limit_bytes(&ext)
            .is_some_and(|limit| size > limit)
    }
}

#[async_trait]
impl MaintenanceTask for FilesystemReconcileTask {
    fn name(&self) -> &str {
        "filesystem_reconcile"
    }

    fn required_idle_states(&self) -> &[IdleState] {
        &[IdleState::FullIdle, IdleState::QdrantDownIdle]
    }

    fn idle_delay_secs(&self) -> u64 {
        60
    }

    fn cooldown_secs(&self) -> u64 {
        1800 // 30 minutes
    }

    fn reset(&mut self) {
        self.offset = 0;
        self.total_checked = 0;
        self.files_missing = 0;
        self.files_oversized = 0;
    }

    async fn run_batch(
        &mut self,
        ctx: &MaintenanceContext<'_>,
        cancel: &CancellationToken,
    ) -> MaintenanceResult {
        let rows = sqlx::query(RECONCILE_BATCH_QUERY)
            .bind(self.batch_size)
            .bind(self.offset)
            .fetch_all(ctx.pool)
            .await;

        let rows = match rows {
            Ok(r) => r,
            Err(e) => {
                // A query error here is structural (schema mismatch), not
                // transient. Returning `Yielded` would re-run on the very next
                // tick with no cooldown, hot-looping and flooding the log.
                // End the cycle so the task's cooldown applies before retry.
                warn!(
                    "Filesystem reconcile query failed: {} — backing off until next cycle",
                    e
                );
                return MaintenanceResult::Done;
            }
        };

        if rows.is_empty() {
            if self.files_missing > 0 || self.files_oversized > 0 {
                info!(
                    "Filesystem reconcile complete: checked={}, missing={}, oversized={}",
                    self.total_checked, self.files_missing, self.files_oversized
                );
            } else {
                debug!(
                    "Filesystem reconcile complete: checked={}, all present",
                    self.total_checked
                );
            }
            return MaintenanceResult::Done;
        }

        for row in &rows {
            if cancel.is_cancelled() {
                return MaintenanceResult::Yielded;
            }

            self.total_checked += 1;
            let relative_path: &str = row.try_get("relative_path").unwrap_or("");
            let watch_path: &str = row.try_get("watch_path").unwrap_or("");
            if relative_path.is_empty() || watch_path.is_empty() {
                continue;
            }

            // Reconstruct the absolute filesystem path from the canonical
            // watch-folder root and the validated relative path.
            let abs_path = std::path::Path::new(watch_path).join(relative_path);

            if abs_path.exists() {
                // Self-heal (#121): a tracked file that was indexed
                // (chunk_count > 0) but now exceeds its per-extension size limit
                // must be re-processed so the ingestion size gate records it as a
                // 0-chunk skip and purges its stale Qdrant points / FTS lines.
                // Files indexed before the size gate fired (or before their
                // enqueue path set `size_bytes`) are exactly this case.
                let chunk_count: i64 = row.try_get("chunk_count").unwrap_or(0);
                if chunk_count > 0 && self.is_oversized(ctx, &abs_path, relative_path) {
                    self.files_oversized += 1;
                    let tenant_id: String = row.try_get("tenant_id").unwrap_or_default();
                    let branch: String = row.try_get("branch").unwrap_or_default();
                    let collection: String = row.try_get("collection").unwrap_or_default();
                    let abs_path_str = abs_path.to_string_lossy();
                    let payload = serde_json::json!({ "file_path": abs_path_str });
                    if let Err(e) = ctx
                        .queue_manager
                        .enqueue_unified(
                            ItemType::File,
                            QueueOperation::Update,
                            &tenant_id,
                            &collection,
                            &payload.to_string(),
                            Some(&branch),
                            None,
                        )
                        .await
                    {
                        warn!(
                            "Failed to enqueue oversized re-process for {}: {}",
                            abs_path_str, e
                        );
                    } else {
                        info!(
                            "Enqueued oversized re-process (now exceeds size gate): {}",
                            abs_path_str
                        );
                    }
                }
                continue;
            }

            // File is missing from disk
            self.files_missing += 1;
            let tenant_id: String = row.try_get("tenant_id").unwrap_or_default();
            let branch: String = row.try_get("branch").unwrap_or_default();
            let collection: String = row.try_get("collection").unwrap_or_default();

            let abs_path_str = abs_path.to_string_lossy();
            let payload = serde_json::json!({ "file_path": abs_path_str });
            if let Err(e) = ctx
                .queue_manager
                .enqueue_unified(
                    ItemType::File,
                    QueueOperation::Delete,
                    &tenant_id,
                    &collection,
                    &payload.to_string(),
                    Some(&branch),
                    None,
                )
                .await
            {
                warn!(
                    "Failed to enqueue delete for missing file {}: {}",
                    abs_path_str, e
                );
            } else {
                info!("Enqueued delete for missing file: {}", abs_path_str);
            }
        }

        self.offset += self.batch_size;
        MaintenanceResult::Continue
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::queue_config::QueueConnectionConfig;
    use crate::schema_version::v40::CREATE_TRACKED_FILES_V40_SQL;

    /// The reconcile join must run against the real `watch_folders` schema,
    /// whose primary key is `watch_id` (regression guard for the `wf.id`
    /// typo that produced "no such column: wf.id" and flooded the log).
    #[tokio::test]
    async fn reconcile_query_matches_watch_folders_schema() {
        let temp_dir = tempfile::tempdir().unwrap();
        let db_path = temp_dir.path().join("reconcile_schema.db");
        let pool = QueueConnectionConfig::with_database_path(&db_path)
            .create_pool()
            .await
            .unwrap();

        for stmt in include_str!("../../schema/watch_folders_schema.sql").split(';') {
            let stmt = stmt.trim();
            if !stmt.is_empty() {
                sqlx::query(stmt).execute(&pool).await.unwrap();
            }
        }
        sqlx::query(CREATE_TRACKED_FILES_V40_SQL)
            .execute(&pool)
            .await
            .unwrap();

        sqlx::query(
            "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, created_at, updated_at)
             VALUES ('w1', '/tmp/proj', 'projects', 't1', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "INSERT INTO tracked_files (watch_folder_id, primary_branch, file_mtime, file_hash, collection, relative_path, created_at, updated_at)
             VALUES ('w1', 'main', '2025-01-01T00:00:00Z', 'h1', 'projects', 'src/main.rs', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')",
        )
        .execute(&pool)
        .await
        .unwrap();

        let rows = sqlx::query(RECONCILE_BATCH_QUERY)
            .bind(100_i64)
            .bind(0_i64)
            .fetch_all(&pool)
            .await
            .expect("reconcile query must execute against the real schema");

        assert_eq!(rows.len(), 1, "expected the single tracked file to join");
        let watch_path: String = rows[0].try_get("watch_path").unwrap();
        assert_eq!(watch_path, "/tmp/proj");
        // Guard the chunk_count column the oversized self-heal (#121) reads.
        let chunk_count: i64 = rows[0]
            .try_get("chunk_count")
            .expect("reconcile query must expose chunk_count");
        assert_eq!(chunk_count, 0, "default chunk_count is 0");
    }
}
