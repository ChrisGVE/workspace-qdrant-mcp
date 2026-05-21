//! Tag-affinity grouping maintenance task (T24).
//!
//! Runs during idle periods to recompute tag-based project affinity groups.
//! Groups projects whose tag profiles (from Tier 1-3 tagging) share enough
//! overlap (Jaccard >= 0.25 by default).

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;
use tracing::warn;

use crate::grouping::affinity::tag_affinity::{compute_tag_affinity_groups, TagAffinityConfig};
use crate::idle::task::{MaintenanceContext, MaintenanceResult, MaintenanceTask};
use crate::idle::IdleState;

/// Maintenance task that recomputes tag-affinity project groups.
///
/// Runs in `FullIdle` or `QdrantDownIdle` (only needs SQLite). Executes the
/// full computation in a single batch since it is an O(n^2) pairwise
/// comparison over a small number of projects (typically < 100).
pub struct TagAffinityGroupingTask {
    config: TagAffinityConfig,
}

impl TagAffinityGroupingTask {
    pub fn new() -> Self {
        Self {
            config: TagAffinityConfig::default(),
        }
    }
}

#[async_trait]
impl MaintenanceTask for TagAffinityGroupingTask {
    fn name(&self) -> &str {
        "tag_affinity_grouping"
    }

    fn required_idle_states(&self) -> &[IdleState] {
        // Only needs SQLite, not Qdrant
        &[IdleState::FullIdle, IdleState::QdrantDownIdle]
    }

    fn idle_delay_secs(&self) -> u64 {
        180 // 3 minutes of idle before running
    }

    fn cooldown_secs(&self) -> u64 {
        7200 // recompute at most once every 2 hours
    }

    async fn run_batch(
        &mut self,
        ctx: &MaintenanceContext<'_>,
        _cancel: &CancellationToken,
    ) -> MaintenanceResult {
        match compute_tag_affinity_groups(ctx.pool, &self.config).await {
            Ok(_groups) => MaintenanceResult::Done,
            Err(e) => {
                warn!("Tag affinity grouping failed: {} — will retry", e);
                MaintenanceResult::Yielded
            }
        }
    }
}
