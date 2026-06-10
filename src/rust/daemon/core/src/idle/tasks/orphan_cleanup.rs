//! Qdrant mirror reconcile — repair drift between SQLite `qdrant_chunks`
//! and actual Qdrant point storage, in both directions:
//!
//! - **Cleanup**: rows whose points no longer exist in Qdrant are deleted
//!   (the original orphan cleanup).
//! - **Repair**: tracked files with `chunk_count > 0` whose mirror rows are
//!   missing entirely get them rebuilt from the live Qdrant points — see
//!   [`super::mirror_repair`] for why missing rows leak points.
//!
//! The pre-fix task had two compounding bugs (observed 2026-06-10 as ~51%
//! of all files losing their mirror rows after a reembed):
//! 1. `check_points_exist` compared the hyphenated canonical UUIDs Qdrant
//!    returns against the bare 32-hex IDs `compute_point_id` stores, so
//!    every checked point was declared an orphan and all of the file's rows
//!    were deleted while the points sat intact in Qdrant.
//! 2. OFFSET pagination over a JOIN the task itself was shrinking skipped
//!    every other batch of 50 files — producing perfect alternating
//!    wiped/kept blocks of exactly `batch_size` in `file_id` order.
//!    Pagination is now keyset (`file_id > cursor`), immune to mid-sweep
//!    row deletion.

use std::collections::HashSet;

use async_trait::async_trait;
use sqlx::Row;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use crate::idle::task::{MaintenanceContext, MaintenanceResult, MaintenanceTask};
use crate::idle::IdleState;

use super::mirror_repair;

/// Sweep phase: cleanup runs first, then repair, then the sweep is done.
enum Phase {
    Cleanup,
    Repair,
}

/// Reconciles `qdrant_chunks` (SQLite) against actual Qdrant point storage.
///
/// Runs only in `FullIdle` (needs both Qdrant and SQLite). Processes files
/// in batches with a keyset cursor on `file_id`. Orphaned SQLite entries
/// (points not in Qdrant) are deleted from `qdrant_chunks`; files missing
/// their mirror rows entirely get them rebuilt from Qdrant.
pub struct OrphanCleanupTask {
    batch_size: i64,
    phase: Phase,
    /// Keyset cursor: largest `file_id` already examined in the current phase.
    cursor: i64,
    total_checked: u64,
    orphans_cleaned: u64,
    files_repaired: u64,
    chunks_rebuilt: u64,
}

impl OrphanCleanupTask {
    pub fn new() -> Self {
        Self {
            batch_size: 50,
            phase: Phase::Cleanup,
            cursor: 0,
            total_checked: 0,
            orphans_cleaned: 0,
            files_repaired: 0,
            chunks_rebuilt: 0,
        }
    }
}

#[async_trait]
impl MaintenanceTask for OrphanCleanupTask {
    fn name(&self) -> &str {
        "orphan_cleanup"
    }

    fn required_idle_states(&self) -> &[IdleState] {
        &[IdleState::FullIdle]
    }

    fn idle_delay_secs(&self) -> u64 {
        120 // 2 minutes of idle
    }

    fn cooldown_secs(&self) -> u64 {
        3600 // once per hour
    }

    fn reset(&mut self) {
        self.phase = Phase::Cleanup;
        self.cursor = 0;
        self.total_checked = 0;
        self.orphans_cleaned = 0;
        self.files_repaired = 0;
        self.chunks_rebuilt = 0;
    }

    async fn run_batch(
        &mut self,
        ctx: &MaintenanceContext<'_>,
        cancel: &CancellationToken,
    ) -> MaintenanceResult {
        match self.phase {
            Phase::Cleanup => self.run_cleanup_batch(ctx, cancel).await,
            Phase::Repair => self.run_repair_batch(ctx, cancel).await,
        }
    }
}

impl OrphanCleanupTask {
    /// One batch of the cleanup phase: verify tracked point IDs against
    /// Qdrant and delete rows whose points are gone.
    async fn run_cleanup_batch(
        &mut self,
        ctx: &MaintenanceContext<'_>,
        cancel: &CancellationToken,
    ) -> MaintenanceResult {
        let rows = match fetch_cleanup_batch(ctx.pool, self.cursor, self.batch_size).await {
            Ok(r) => r,
            Err(e) => {
                warn!("Orphan cleanup query failed: {} — will retry", e);
                return MaintenanceResult::Yielded;
            }
        };

        if rows.is_empty() {
            self.phase = Phase::Repair;
            self.cursor = 0;
            return MaintenanceResult::Continue;
        }

        // Advance strictly by the batch's own file_ids: rows this sweep
        // deletes must never shift which files the next batch sees.
        let Some(next_cursor) = last_file_id(&rows) else {
            warn!("Orphan cleanup: batch rows missing file_id — aborting sweep");
            self.log_completion();
            return MaintenanceResult::Done;
        };

        for row in &rows {
            if cancel.is_cancelled() {
                // Cursor not advanced: the batch is re-examined on resume,
                // which is idempotent.
                return MaintenanceResult::Yielded;
            }

            let (file_id, file_path, collection) = match parse_row(row) {
                Some(tuple) => tuple,
                None => continue,
            };
            self.total_checked += 1;

            self.reconcile_file(ctx, file_id, &file_path, &collection)
                .await;
        }

        self.cursor = next_cursor;
        MaintenanceResult::Continue
    }

    /// One batch of the repair phase: rebuild missing mirror rows from the
    /// live Qdrant points.
    async fn run_repair_batch(
        &mut self,
        ctx: &MaintenanceContext<'_>,
        cancel: &CancellationToken,
    ) -> MaintenanceResult {
        let candidates =
            match mirror_repair::fetch_repair_batch(ctx.pool, self.cursor, self.batch_size).await {
                Ok(c) => c,
                Err(e) => {
                    warn!("Mirror repair query failed: {} — will retry", e);
                    return MaintenanceResult::Yielded;
                }
            };

        let Some((next_cursor, _, _, _)) = candidates.last() else {
            self.log_completion();
            return MaintenanceResult::Done;
        };
        let next_cursor = *next_cursor;

        for (file_id, relative_path, collection, base_point) in &candidates {
            if cancel.is_cancelled() {
                return MaintenanceResult::Yielded;
            }
            let rebuilt =
                mirror_repair::repair_file(ctx, *file_id, relative_path, collection, base_point)
                    .await;
            if rebuilt > 0 {
                self.files_repaired += 1;
                self.chunks_rebuilt += rebuilt;
            }
        }

        self.cursor = next_cursor;
        MaintenanceResult::Continue
    }

    /// Log a summary message when the sweep finishes.
    fn log_completion(&self) {
        if self.orphans_cleaned > 0 || self.files_repaired > 0 {
            info!(
                "Mirror reconcile complete: files_checked={}, orphans_cleaned={}, files_repaired={}, chunks_rebuilt={}",
                self.total_checked, self.orphans_cleaned, self.files_repaired, self.chunks_rebuilt
            );
        } else {
            debug!(
                "Mirror reconcile complete: checked={}, no drift",
                self.total_checked
            );
        }
    }

    /// Reconcile a single file's tracked points against Qdrant.
    async fn reconcile_file(
        &mut self,
        ctx: &MaintenanceContext<'_>,
        file_id: i64,
        file_path: &str,
        collection: &str,
    ) {
        let sqlite_points: Vec<String> =
            sqlx::query_scalar("SELECT point_id FROM qdrant_chunks WHERE file_id = ?1")
                .bind(file_id)
                .fetch_all(ctx.pool)
                .await
                .unwrap_or_default();

        if sqlite_points.is_empty() {
            return;
        }

        let qdrant_existing = match ctx
            .storage_client
            .check_points_exist(collection, &sqlite_points)
            .await
        {
            Ok(existing) => existing,
            Err(e) => {
                warn!(
                    "Orphan check failed for {} ({}): {}",
                    file_path, collection, e
                );
                return;
            }
        };

        let sqlite_set: HashSet<&str> = sqlite_points.iter().map(String::as_str).collect();
        let orphans: Vec<&str> = sqlite_set
            .iter()
            .filter(|id| !qdrant_existing.contains(**id))
            .copied()
            .collect();

        if orphans.is_empty() {
            return;
        }

        debug!(
            "File {} has {} orphaned qdrant_chunks entries",
            file_path,
            orphans.len()
        );

        for point_id in &orphans {
            if let Err(e) =
                sqlx::query("DELETE FROM qdrant_chunks WHERE point_id = ?1 AND file_id = ?2")
                    .bind(point_id)
                    .bind(file_id)
                    .execute(ctx.pool)
                    .await
            {
                warn!("Failed to delete orphan chunk {}: {}", point_id, e);
            } else {
                self.orphans_cleaned += 1;
            }
        }
    }
}

/// Fetch the next batch of files that have `qdrant_chunks` entries, strictly
/// after `after_file_id` (keyset pagination — see module docs for why OFFSET
/// is wrong here).
pub(crate) async fn fetch_cleanup_batch(
    pool: &sqlx::SqlitePool,
    after_file_id: i64,
    limit: i64,
) -> Result<Vec<sqlx::sqlite::SqliteRow>, sqlx::Error> {
    sqlx::query(
        "SELECT DISTINCT tf.file_id, tf.relative_path, tf.collection
         FROM tracked_files tf
         JOIN qdrant_chunks qc ON tf.file_id = qc.file_id
         WHERE tf.file_id > ?1
         ORDER BY tf.file_id
         LIMIT ?2",
    )
    .bind(after_file_id)
    .bind(limit)
    .fetch_all(pool)
    .await
}

/// Largest readable `file_id` in a batch (rows are ordered ascending).
fn last_file_id(rows: &[sqlx::sqlite::SqliteRow]) -> Option<i64> {
    rows.iter()
        .rev()
        .find_map(|r| r.try_get::<i64, _>("file_id").ok())
}

/// Extract `(file_id, file_path, collection)` from a row, logging on failure.
fn parse_row(row: &sqlx::sqlite::SqliteRow) -> Option<(i64, String, String)> {
    match (
        row.try_get::<i64, _>("file_id"),
        row.try_get::<String, _>("relative_path"),
        row.try_get::<String, _>("collection"),
    ) {
        (Ok(id), Ok(path), Ok(col)) => Some((id, path, col)),
        _ => {
            warn!("Orphan cleanup: skipping row with missing fields");
            None
        }
    }
}

/// Shared fixtures for this module's tests and `mirror_repair`'s.
#[cfg(test)]
pub(crate) mod tests_support {
    use sqlx::sqlite::SqlitePoolOptions;
    use sqlx::SqlitePool;

    use crate::tracked_files_schema::{self, ProcessingStatus};

    /// In-memory pool with the production `watch_folders`, `tracked_files`
    /// (v37) and `qdrant_chunks` DDL. sqlx turns `PRAGMA foreign_keys` ON by
    /// default, so the watch_folder row the tracked_files FK points at must
    /// exist.
    pub(crate) async fn test_pool() -> SqlitePool {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .expect("create in-memory pool");
        sqlx::query(crate::watch_folders_schema::CREATE_WATCH_FOLDERS_SQL)
            .execute(&pool)
            .await
            .expect("create watch_folders");
        sqlx::query(tracked_files_schema::CREATE_TRACKED_FILES_V37_SQL)
            .execute(&pool)
            .await
            .expect("create tracked_files");
        sqlx::query(tracked_files_schema::CREATE_QDRANT_CHUNKS_SQL)
            .execute(&pool)
            .await
            .expect("create qdrant_chunks");
        sqlx::query(
            "INSERT INTO watch_folders
                 (watch_id, path, collection, tenant_id, created_at, updated_at)
             VALUES ('wf-test', '/tmp/wf-test', 'projects', 'tenant-test',
                     '2026-06-10T00:00:00Z', '2026-06-10T00:00:00Z')",
        )
        .execute(&pool)
        .await
        .expect("insert watch_folder");
        pool
    }

    /// Insert a tracked file claiming 2 chunks; when `with_rows` is false
    /// the mirror rows are intentionally missing (drifted state).
    pub(crate) async fn seed_file(
        pool: &SqlitePool,
        idx: usize,
        with_rows: bool,
        with_base_point: bool,
    ) -> i64 {
        let file_id = tracked_files_schema::insert_tracked_file(
            pool,
            "wf-test",
            &format!("src/file{idx}.rs"),
            Some("main"),
            None,
            None,
            "2026-06-10T00:00:00Z",
            &format!("hash{idx}"),
            2,
            None,
            ProcessingStatus::None,
            ProcessingStatus::None,
            Some("projects"),
            None,
            false,
            with_base_point.then(|| format!("bp{idx}")).as_deref(),
            None,
        )
        .await
        .expect("insert tracked file");
        if with_rows {
            tracked_files_schema::insert_qdrant_chunks(
                pool,
                file_id,
                &[
                    (
                        format!("p{idx}a"),
                        0,
                        "ch".to_string(),
                        None,
                        None,
                        None,
                        None,
                    ),
                    (
                        format!("p{idx}b"),
                        1,
                        "ch".to_string(),
                        None,
                        None,
                        None,
                        None,
                    ),
                ],
            )
            .await
            .expect("insert chunks");
        }
        file_id
    }
}

#[cfg(test)]
mod tests {
    use sqlx::Row;

    use crate::tracked_files_schema;

    use super::fetch_cleanup_batch;
    use super::tests_support::{seed_file, test_pool};

    fn ids(rows: &[sqlx::sqlite::SqliteRow]) -> Vec<i64> {
        rows.iter().map(|r| r.get::<i64, _>("file_id")).collect()
    }

    /// Regression for the production ~51% wipe: the sweep deletes rows of
    /// the files it just examined, which shrinks the paginated JOIN. With
    /// OFFSET pagination the next page skipped the files that slid into the
    /// window (every other batch of 50 was never checked — and, with the
    /// existence check also broken, every *checked* batch was wiped,
    /// producing perfect alternating blocks of 50 in file_id order). Keyset
    /// pagination must return the immediately following files instead.
    #[tokio::test]
    async fn cleanup_batch_keyset_does_not_skip_files_after_mid_sweep_deletions() {
        let pool = test_pool().await;
        let mut file_ids = Vec::new();
        for idx in 0..6 {
            file_ids.push(seed_file(&pool, idx, true, true).await);
        }

        let b1 = fetch_cleanup_batch(&pool, 0, 2).await.unwrap();
        assert_eq!(ids(&b1), &file_ids[0..2]);

        // Simulate the sweep deleting every mirror row of the batch it just
        // examined (what the broken existence check did to each batch).
        for id in ids(&b1) {
            tracked_files_schema::delete_qdrant_chunks(&pool, id)
                .await
                .unwrap();
        }

        let b2 = fetch_cleanup_batch(&pool, file_ids[1], 2).await.unwrap();
        assert_eq!(
            ids(&b2),
            &file_ids[2..4],
            "keyset batch must continue with the next files; OFFSET skipped them"
        );

        for id in ids(&b2) {
            tracked_files_schema::delete_qdrant_chunks(&pool, id)
                .await
                .unwrap();
        }

        let b3 = fetch_cleanup_batch(&pool, file_ids[3], 2).await.unwrap();
        assert_eq!(ids(&b3), &file_ids[4..6]);

        let b4 = fetch_cleanup_batch(&pool, file_ids[5], 2).await.unwrap();
        assert!(b4.is_empty(), "sweep must terminate after the last file");
    }
}
