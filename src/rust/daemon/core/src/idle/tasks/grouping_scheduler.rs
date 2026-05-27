//! Unified grouping scheduler idle task.
//!
//! Replaces the individual `DependencyGroupingTask` and `GitOrgGroupTask` with
//! a single coordinator that runs all four grouping strategies in the correct
//! order. Phase 1 (dependency, workspace, git_org) runs first since they produce
//! input data, then phase 2 (affinity) runs because it derives from phase 1.
//!
//! Internal staleness tracking ensures each strategy only reruns when its
//! cooldown has expired, avoiding redundant work.

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info};

use crate::grouping::scheduler::GroupingScheduler;
use crate::idle::task::{MaintenanceContext, MaintenanceResult, MaintenanceTask};
use crate::idle::IdleState;

/// Unified grouping scheduler that coordinates all four grouping strategies.
///
/// Runs during `FullIdle` or `QdrantDownIdle` since all strategies only need
/// SQLite. Activates after 2 minutes of idle with a 30-minute outer cooldown.
/// Internal per-strategy cooldowns (1 hour each) provide finer-grained control.
pub struct GroupingSchedulerTask {
    scheduler: GroupingScheduler,
}

impl GroupingSchedulerTask {
    pub fn new() -> Self {
        Self {
            scheduler: GroupingScheduler::new(),
        }
    }
}

#[async_trait]
impl MaintenanceTask for GroupingSchedulerTask {
    fn name(&self) -> &str {
        "grouping_scheduler"
    }

    fn required_idle_states(&self) -> &[IdleState] {
        // All strategies only need SQLite, not Qdrant
        &[IdleState::FullIdle, IdleState::QdrantDownIdle]
    }

    fn idle_delay_secs(&self) -> u64 {
        120 // 2 minutes of idle before activating
    }

    fn cooldown_secs(&self) -> u64 {
        1800 // 30 minutes between scheduler ticks (inner cooldowns are 1 hour)
    }

    async fn run_batch(
        &mut self,
        ctx: &MaintenanceContext<'_>,
        _cancel: &CancellationToken,
    ) -> MaintenanceResult {
        if !self.scheduler.has_stale_strategies() {
            debug!("Grouping scheduler: no stale strategies, skipping");
            return MaintenanceResult::Done;
        }

        let result = self
            .scheduler
            .run_stale_with_storage(ctx.pool, Some(ctx.storage_client))
            .await;

        if result.failed_strategies.is_empty() {
            if result.total_groups() > 0 {
                info!(
                    total = result.total_groups(),
                    "Grouping scheduler: created {} groups",
                    result.total_groups()
                );
            } else {
                debug!("Grouping scheduler: no groups above thresholds");
            }
        } else {
            for (strategy, error) in &result.failed_strategies {
                tracing::warn!(
                    strategy = strategy.as_str(),
                    error = error.as_str(),
                    "Grouping strategy failed"
                );
            }
        }

        MaintenanceResult::Done
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_metadata() {
        let task = GroupingSchedulerTask::new();
        assert_eq!(task.name(), "grouping_scheduler");
        assert_eq!(task.idle_delay_secs(), 120);
        assert_eq!(task.cooldown_secs(), 1800);
        assert!(task.can_run_in(IdleState::FullIdle));
        assert!(task.can_run_in(IdleState::QdrantDownIdle));
        assert!(!task.can_run_in(IdleState::Active));
        assert!(!task.can_run_in(IdleState::ResourceConstrained));
    }
}
