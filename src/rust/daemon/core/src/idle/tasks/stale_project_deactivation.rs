//! Stale project deactivation — deactivate projects with no activity for 7 days.
//!
//! A lightweight guardrail: if no file changes have been processed for a project
//! in a week, it is deactivated (is_active = 0). The project remains registered
//! and its data stays intact — the next MCP server connection or file change will
//! reactivate it immediately.

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info};

use crate::idle::task::{MaintenanceContext, MaintenanceResult, MaintenanceTask};
use crate::idle::IdleState;

/// Deactivates projects whose `last_activity_at` is older than 7 days.
///
/// Runs during any idle state (only needs SQLite, not Qdrant). Checks once
/// per day with a 5-minute idle delay to avoid interfering with active work.
pub struct StaleProjectDeactivationTask {
    staleness_days: i64,
}

impl StaleProjectDeactivationTask {
    pub fn new() -> Self {
        Self { staleness_days: 7 }
    }
}

#[async_trait]
impl MaintenanceTask for StaleProjectDeactivationTask {
    fn name(&self) -> &str {
        "stale_project_deactivation"
    }

    fn required_idle_states(&self) -> &[IdleState] {
        // Only needs SQLite — can run even when Qdrant is down
        &[IdleState::FullIdle, IdleState::QdrantDownIdle]
    }

    fn idle_delay_secs(&self) -> u64 {
        300 // 5 minutes of idle
    }

    fn cooldown_secs(&self) -> u64 {
        86400 // once per day
    }

    async fn run_batch(
        &mut self,
        ctx: &MaintenanceContext<'_>,
        _cancel: &CancellationToken,
    ) -> MaintenanceResult {
        let cutoff = chrono::Utc::now() - chrono::Duration::days(self.staleness_days);
        let cutoff_str = wqm_common::timestamps::format_utc(&cutoff);

        let result = sqlx::query(
            "UPDATE watch_folders \
             SET is_active = 0, \
                 updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') \
             WHERE is_active > 0 \
               AND last_activity_at IS NOT NULL \
               AND last_activity_at < ?1",
        )
        .bind(&cutoff_str)
        .execute(ctx.pool)
        .await;

        match result {
            Ok(r) if r.rows_affected() > 0 => {
                info!(
                    "Deactivated {} stale projects (no activity since {})",
                    r.rows_affected(),
                    cutoff_str
                );
            }
            Ok(_) => {
                debug!("No stale projects to deactivate (cutoff: {})", cutoff_str);
            }
            Err(e) => {
                tracing::warn!("Stale project deactivation failed: {}", e);
            }
        }

        MaintenanceResult::Done
    }
}
