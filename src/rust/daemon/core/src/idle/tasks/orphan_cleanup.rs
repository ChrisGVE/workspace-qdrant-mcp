//! Qdrant orphan cleanup — detect and remove points that exist in SQLite
//! `qdrant_chunks` but not in Qdrant (or vice versa).

use std::collections::HashSet;

use async_trait::async_trait;
use sqlx::Row;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use crate::idle::task::{MaintenanceContext, MaintenanceResult, MaintenanceTask};
use crate::idle::IdleState;

/// Reconciles `qdrant_chunks` (SQLite) against actual Qdrant point storage.
///
/// Runs only in `FullIdle` (needs both Qdrant and SQLite). Processes files
/// in batches, checking each file's tracked point IDs against Qdrant.
/// Orphaned SQLite entries (points not in Qdrant) are deleted from
/// `qdrant_chunks` so the tracking state is accurate.
pub struct OrphanCleanupTask {
    batch_size: i64,
    offset: i64,
    total_checked: u64,
    orphans_cleaned: u64,
}

impl OrphanCleanupTask {
    pub fn new() -> Self {
        Self {
            batch_size: 50,
            offset: 0,
            total_checked: 0,
            orphans_cleaned: 0,
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
        self.offset = 0;
        self.total_checked = 0;
        self.orphans_cleaned = 0;
    }

    async fn run_batch(
        &mut self,
        ctx: &MaintenanceContext<'_>,
        cancel: &CancellationToken,
    ) -> MaintenanceResult {
        let rows = match self.fetch_file_batch(ctx).await {
            Ok(r) => r,
            Err(()) => return MaintenanceResult::Yielded,
        };

        if rows.is_empty() {
            self.log_completion();
            return MaintenanceResult::Done;
        }

        for row in &rows {
            if cancel.is_cancelled() {
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

        self.offset += self.batch_size;
        MaintenanceResult::Continue
    }
}

impl OrphanCleanupTask {
    /// Fetch the next batch of files that have qdrant_chunks entries.
    async fn fetch_file_batch(
        &self,
        ctx: &MaintenanceContext<'_>,
    ) -> Result<Vec<sqlx::sqlite::SqliteRow>, ()> {
        let rows = sqlx::query(
            "SELECT DISTINCT tf.file_id, tf.file_path, tf.collection
             FROM tracked_files tf
             JOIN qdrant_chunks qc ON tf.file_id = qc.file_id
             ORDER BY tf.file_id
             LIMIT ?1 OFFSET ?2",
        )
        .bind(self.batch_size)
        .bind(self.offset)
        .fetch_all(ctx.pool)
        .await;

        match rows {
            Ok(r) => Ok(r),
            Err(e) => {
                warn!("Orphan cleanup query failed: {} — will retry", e);
                Err(())
            }
        }
    }

    /// Log a summary message when cleanup finishes.
    fn log_completion(&self) {
        if self.orphans_cleaned > 0 {
            info!(
                "Orphan cleanup complete: files_checked={}, orphans_cleaned={}",
                self.total_checked, self.orphans_cleaned
            );
        } else {
            debug!(
                "Orphan cleanup complete: checked={}, no orphans",
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

/// Extract `(file_id, file_path, collection)` from a row, logging on failure.
fn parse_row(row: &sqlx::sqlite::SqliteRow) -> Option<(i64, String, String)> {
    match (
        row.try_get::<i64, _>("file_id"),
        row.try_get::<String, _>("file_path"),
        row.try_get::<String, _>("collection"),
    ) {
        (Ok(id), Ok(path), Ok(col)) => Some((id, path, col)),
        _ => {
            warn!("Orphan cleanup: skipping row with missing fields");
            None
        }
    }
}
