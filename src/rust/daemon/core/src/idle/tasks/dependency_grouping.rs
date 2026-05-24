//! Dependency-based project grouping -- periodic idle recomputation.
//!
//! During idle periods, recomputes Jaccard-similarity-based project groups
//! from the cached dependency sets in `project_dependencies`. Groups projects
//! that share >= 30% of their dependency names into `project_groups` entries
//! with `group_type = "dependency"`.

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use crate::grouping::dependency;
use crate::idle::task::{MaintenanceContext, MaintenanceResult, MaintenanceTask};
use crate::idle::IdleState;

/// Recomputes dependency-based project groups during idle windows.
///
/// Only needs SQLite (no Qdrant) so it can run even when Qdrant is down.
/// Runs with a 2-minute idle delay and a 1-hour cooldown between cycles.
pub struct DependencyGroupingTask;

impl DependencyGroupingTask {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl MaintenanceTask for DependencyGroupingTask {
    fn name(&self) -> &str {
        "dependency_grouping"
    }

    fn required_idle_states(&self) -> &[IdleState] {
        &[IdleState::FullIdle, IdleState::QdrantDownIdle]
    }

    fn idle_delay_secs(&self) -> u64 {
        120 // 2 minutes of idle before running
    }

    fn cooldown_secs(&self) -> u64 {
        3600 // recompute at most once per hour
    }

    async fn run_batch(
        &mut self,
        ctx: &MaintenanceContext<'_>,
        _cancel: &CancellationToken,
    ) -> MaintenanceResult {
        match dependency::compute_dependency_groups(ctx.pool, None).await {
            Ok(groups) => {
                if groups > 0 {
                    info!("Dependency grouping complete: {} groups created", groups);
                } else {
                    debug!("Dependency grouping complete: no groups above threshold");
                }
            }
            Err(e) => {
                warn!("Dependency grouping failed: {}", e);
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
        let task = DependencyGroupingTask::new();
        assert_eq!(task.name(), "dependency_grouping");
        assert_eq!(task.idle_delay_secs(), 120);
        assert_eq!(task.cooldown_secs(), 3600);
        assert!(task.can_run_in(IdleState::FullIdle));
        assert!(task.can_run_in(IdleState::QdrantDownIdle));
        assert!(!task.can_run_in(IdleState::Active));
        assert!(!task.can_run_in(IdleState::ResourceConstrained));
    }
}
