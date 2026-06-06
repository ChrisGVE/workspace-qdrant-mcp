//! Completed-queue cleanup — bound the `unified_queue` dedup window.
//!
//! `unified_queue` rows are deduped on a globally-unique `idempotency_key`
//! (plus a status-agnostic composite UNIQUE index). A `done` row therefore
//! keeps deduping any future re-enqueue of the same work for as long as it
//! survives. `QueueManager::cleanup_completed_unified_items` exists to age out
//! those rows but had no scheduled caller — so the dedup window was unbounded
//! and `unified_queue` grew without limit (this is what made reembed a no-op
//! against already-`done` rows; see #96). This task runs the cleanup on an
//! idle cadence so completed work is reclaimed and the dedup window stays
//! bounded.

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use crate::idle::task::{MaintenanceContext, MaintenanceResult, MaintenanceTask};
use crate::idle::IdleState;

/// Deletes `done` `unified_queue` rows older than `retention_hours`.
///
/// Only needs SQLite, so it runs even when Qdrant is down. Hourly cadence with
/// a short idle delay keeps it out of the way of active processing.
pub struct QueueCleanupTask {
    retention_hours: i64,
}

impl QueueCleanupTask {
    pub fn new() -> Self {
        Self {
            retention_hours: 24,
        }
    }
}

impl Default for QueueCleanupTask {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl MaintenanceTask for QueueCleanupTask {
    fn name(&self) -> &str {
        "queue_cleanup"
    }

    fn required_idle_states(&self) -> &[IdleState] {
        // SQLite-only — safe to run while Qdrant is unavailable.
        &[IdleState::FullIdle, IdleState::QdrantDownIdle]
    }

    fn idle_delay_secs(&self) -> u64 {
        300 // 5 minutes of idle before reclaiming
    }

    fn cooldown_secs(&self) -> u64 {
        3600 // hourly
    }

    async fn run_batch(
        &mut self,
        ctx: &MaintenanceContext<'_>,
        _cancel: &CancellationToken,
    ) -> MaintenanceResult {
        let query = format!(
            "DELETE FROM unified_queue \
             WHERE status = 'done' \
             AND updated_at < datetime('now', '-{} hours')",
            self.retention_hours
        );
        match sqlx::query(&query).execute(ctx.pool).await {
            Ok(r) if r.rows_affected() > 0 => {
                info!(
                    "queue_cleanup: reclaimed {} completed queue rows (older than {}h)",
                    r.rows_affected(),
                    self.retention_hours
                );
            }
            Ok(_) => {
                debug!("queue_cleanup: no completed rows past retention");
            }
            Err(e) => {
                warn!("queue_cleanup failed: {}", e);
            }
        }
        MaintenanceResult::Done
    }
}
