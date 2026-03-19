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
        // Get a batch of files that have qdrant_chunks entries
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

        let rows = match rows {
            Ok(r) => r,
            Err(e) => {
                warn!("Orphan cleanup query failed: {} — will retry", e);
                return MaintenanceResult::Yielded;
            }
        };

        if rows.is_empty() {
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
            return MaintenanceResult::Done;
        }

        for row in &rows {
            if cancel.is_cancelled() {
                return MaintenanceResult::Yielded;
            }

            let (file_id, file_path, collection) = match (
                row.try_get::<i64, _>("file_id"),
                row.try_get::<String, _>("file_path"),
                row.try_get::<String, _>("collection"),
            ) {
                (Ok(id), Ok(path), Ok(col)) => (id, path, col),
                _ => {
                    warn!("Orphan cleanup: skipping row with missing fields");
                    continue;
                }
            };
            self.total_checked += 1;

            // Get point IDs tracked in SQLite for this file
            let sqlite_points: Vec<String> =
                sqlx::query_scalar("SELECT point_id FROM qdrant_chunks WHERE file_id = ?1")
                    .bind(file_id)
                    .fetch_all(ctx.pool)
                    .await
                    .unwrap_or_default();

            if sqlite_points.is_empty() {
                continue;
            }

            // Check which points actually exist in Qdrant
            let qdrant_existing = match ctx
                .storage_client
                .check_points_exist(&collection, &sqlite_points)
                .await
            {
                Ok(existing) => existing,
                Err(e) => {
                    warn!(
                        "Orphan check failed for {} ({}): {}",
                        file_path, collection, e
                    );
                    continue;
                }
            };

            // Find orphans: in SQLite but not in Qdrant
            let sqlite_set: HashSet<&str> = sqlite_points.iter().map(String::as_str).collect();
            let orphans: Vec<&str> = sqlite_set
                .iter()
                .filter(|id| !qdrant_existing.contains(**id))
                .copied()
                .collect();

            if orphans.is_empty() {
                continue;
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

        self.offset += self.batch_size;
        MaintenanceResult::Continue
    }
}
