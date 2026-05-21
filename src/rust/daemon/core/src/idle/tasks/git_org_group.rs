//! Git organization group maintenance -- recompute org-based project groups.
//!
//! During idle periods, recomputes git-org grouping for all registered projects.
//! This catches any groups that were missed during incremental updates (e.g.,
//! projects registered before their org peers, or remote URL changes that were
//! not propagated incrementally).

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info};

use crate::grouping::git_org;
use crate::idle::task::{MaintenanceContext, MaintenanceResult, MaintenanceTask};
use crate::idle::IdleState;

/// Recomputes git-org groups for all registered projects.
///
/// Runs during any idle state (only needs SQLite, not Qdrant). Checks once
/// per hour with a 2-minute idle delay to avoid interfering with active work.
pub struct GitOrgGroupTask;

impl GitOrgGroupTask {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl MaintenanceTask for GitOrgGroupTask {
    fn name(&self) -> &str {
        "git_org_group"
    }

    fn required_idle_states(&self) -> &[IdleState] {
        // Only needs SQLite -- can run even when Qdrant is down
        &[IdleState::FullIdle, IdleState::QdrantDownIdle]
    }

    fn idle_delay_secs(&self) -> u64 {
        120 // 2 minutes of idle
    }

    fn cooldown_secs(&self) -> u64 {
        3600 // once per hour
    }

    async fn run_batch(
        &mut self,
        ctx: &MaintenanceContext<'_>,
        _cancel: &CancellationToken,
    ) -> MaintenanceResult {
        match git_org::compute_git_org_groups(ctx.pool).await {
            Ok(groups) => {
                if groups > 0 {
                    info!("Git org group maintenance: created {} groups", groups);
                } else {
                    debug!("Git org group maintenance: no multi-project orgs found");
                }
            }
            Err(e) => {
                tracing::warn!("Git org group maintenance failed: {}", e);
            }
        }

        MaintenanceResult::Done
    }
}
