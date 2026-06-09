//! MaintenanceTask trait and supporting types.

use std::sync::Arc;

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use super::IdleState;
use crate::config::IngestionLimitsConfig;
use crate::graph::GraphStore;
use crate::queue_operations::QueueManager;
use crate::search_db::SearchDbManager;
use crate::storage::StorageClient;

/// Result of a single maintenance batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaintenanceResult {
    /// More work to do — call `run_batch` again on the next idle window.
    Continue,
    /// All work complete for this cycle.
    Done,
    /// Interrupted by cancellation — progress saved internally.
    Yielded,
}

/// Shared references provided to each maintenance task.
pub struct MaintenanceContext<'a> {
    pub pool: &'a sqlx::SqlitePool,
    pub storage_client: &'a Arc<StorageClient>,
    pub search_db: Option<&'a Arc<SearchDbManager>>,
    pub queue_manager: &'a QueueManager,
    /// Graph store for tasks that need to read/write graph edges (e.g. ELABORATES).
    /// `None` when graph support is disabled.
    pub graph_store: Option<&'a Arc<dyn GraphStore>>,
    /// Per-extension ingestion size limits, shared with the ingestion path.
    /// Used by `filesystem_reconcile` to detect already-ingested files that now
    /// exceed the size gate and re-process them as skips (#121 self-heal).
    pub ingestion_limits: &'a Arc<IngestionLimitsConfig>,
}

/// A maintenance task that runs during idle periods.
///
/// Implementations must be resumable: `run_batch` processes a fixed-size
/// chunk of work, then returns `Continue` (more to do), `Done` (finished),
/// or `Yielded` (cancelled mid-batch). The scheduler calls `run_batch`
/// repeatedly across idle windows until `Done`.
#[async_trait]
pub trait MaintenanceTask: Send + Sync {
    /// Human-readable name for logging.
    fn name(&self) -> &str;

    /// Which idle states this task can run in.
    fn required_idle_states(&self) -> &[IdleState];

    /// Minimum seconds of continuous idle before this task activates.
    fn idle_delay_secs(&self) -> u64;

    /// Minimum seconds between full cycles (cooldown after `Done`).
    fn cooldown_secs(&self) -> u64;

    /// Run one batch. Check `cancel.is_cancelled()` between items to yield
    /// promptly when real work arrives.
    async fn run_batch(
        &mut self,
        ctx: &MaintenanceContext<'_>,
        cancel: &CancellationToken,
    ) -> MaintenanceResult;

    /// Check if this task can run in the given idle state.
    fn can_run_in(&self, state: IdleState) -> bool {
        self.required_idle_states().contains(&state)
    }

    /// Reset internal progress for a new cycle (called before first batch).
    fn reset(&mut self) {}
}
