//! Fairness Scheduler Module
//!
//! Implements asymmetric anti-starvation alternation.
//! Uses different batch sizes for high-priority (default 10) and low-priority (default 3)
//! directions, flipping between priority DESC (active first) and priority ASC
//! (inactive projects get a turn) to prevent starvation while preserving priority advantage.

use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;
use tracing::{debug, info};

use crate::queue_operations::{QueueManager, QueueError};
use crate::unified_queue_schema::UnifiedQueueItem;

/// Fairness scheduler errors
#[derive(Error, Debug)]
pub enum FairnessError {
    #[error("Queue error: {0}")]
    Queue(#[from] QueueError),

    #[error("No items available")]
    NoItems,

    #[error("Scheduler not initialized")]
    NotInitialized,
}

/// Result type for fairness scheduler operations
pub type FairnessResult<T> = Result<T, FairnessError>;

/// Configuration for the fairness scheduler
#[derive(Debug, Clone)]
pub struct FairnessSchedulerConfig {
    /// Whether fairness scheduling is enabled (if disabled, falls back to priority DESC always)
    pub enabled: bool,

    /// Batch size when processing high-priority items (priority DESC direction)
    pub high_priority_batch: u64,

    /// Batch size when processing low-priority items (priority ASC / anti-starvation direction)
    pub low_priority_batch: u64,

    /// Worker ID for lease acquisition
    pub worker_id: String,

    /// Lease duration in seconds
    pub lease_duration_secs: i64,
}

impl Default for FairnessSchedulerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            high_priority_batch: 10, // Spec: process 10 high-priority items per cycle
            low_priority_batch: 3,   // Spec: process 3 low-priority items per anti-starvation cycle
            worker_id: format!("fairness-worker-{}", uuid::Uuid::new_v4()),
            lease_duration_secs: 300, // 5 minutes
        }
    }
}

/// Metrics tracked by the fairness scheduler
#[derive(Debug, Clone, Default)]
pub struct FairnessMetrics {
    /// Number of times priority direction was flipped
    pub direction_flips_total: u64,

    /// Items dequeued with high priority first (DESC)
    pub high_priority_first_items: u64,

    /// Items dequeued with low priority first (ASC)
    pub low_priority_first_items: u64,

    /// Total items dequeued via fairness scheduling
    pub total_items_dequeued: u64,

    /// Current priority direction (true = DESC, false = ASC)
    pub current_priority_descending: bool,

    /// Items processed since last flip
    pub items_since_flip: u64,
}

/// Anti-starvation state for the scheduler
struct AlternationState {
    /// Whether to use priority DESC (true) or ASC (false)
    use_priority_descending: bool,

    /// Items processed since last flip
    items_since_flip: u64,
}

impl Default for AlternationState {
    fn default() -> Self {
        Self {
            use_priority_descending: true, // Start with high priority first
            items_since_flip: 0,
        }
    }
}

/// Fairness scheduler for balanced queue processing
///
/// Implements the spec's anti-starvation mechanism with asymmetric batch sizes:
/// processes `high_priority_batch` items in priority DESC (active projects first),
/// then flips to `low_priority_batch` items in priority ASC (inactive projects get a turn).
pub struct FairnessScheduler {
    /// Queue manager for database operations
    queue_manager: QueueManager,

    /// Scheduler configuration
    config: FairnessSchedulerConfig,

    /// Anti-starvation alternation state
    alternation_state: Arc<RwLock<AlternationState>>,

    /// Metrics for monitoring
    metrics: Arc<RwLock<FairnessMetrics>>,
}

impl FairnessScheduler {
    /// Create a new fairness scheduler
    pub fn new(queue_manager: QueueManager, config: FairnessSchedulerConfig) -> Self {
        info!(
            "Creating fairness scheduler (enabled={}, high_priority_batch={}, low_priority_batch={})",
            config.enabled, config.high_priority_batch, config.low_priority_batch
        );

        Self {
            queue_manager,
            config,
            alternation_state: Arc::new(RwLock::new(AlternationState::default())),
            metrics: Arc::new(RwLock::new(FairnessMetrics {
                current_priority_descending: true,
                ..Default::default()
            })),
        }
    }

    /// Get current fairness metrics
    pub async fn get_metrics(&self) -> FairnessMetrics {
        self.metrics.read().await.clone()
    }

    /// Dequeue items for a specific project with current priority direction
    async fn dequeue_project_batch(
        &self,
        project_id: &str,
        max_items: i32,
        priority_descending: bool,
    ) -> FairnessResult<Vec<UnifiedQueueItem>> {
        let items = self.queue_manager
            .dequeue_unified(
                max_items,
                &self.config.worker_id,
                Some(self.config.lease_duration_secs),
                Some(project_id),
                None,
                Some(priority_descending),
            )
            .await?;

        Ok(items)
    }

    /// Dequeue items globally with current priority direction
    async fn dequeue_global_batch(
        &self,
        max_items: i32,
        priority_descending: bool,
    ) -> FairnessResult<Vec<UnifiedQueueItem>> {
        let items = self.queue_manager
            .dequeue_unified(
                max_items,
                &self.config.worker_id,
                Some(self.config.lease_duration_secs),
                None,
                None,
                Some(priority_descending),
            )
            .await?;

        Ok(items)
    }

    /// Main scheduling method: dequeue the next batch of items
    ///
    /// Algorithm (asymmetric anti-starvation alternation):
    /// 1. Get current priority direction (DESC or ASC)
    /// 2. Dequeue up to max_batch_size items with that direction
    /// 3. Track items processed
    /// 4. After high_priority_batch items in DESC, flip to ASC
    /// 5. After low_priority_batch items in ASC, flip back to DESC
    ///
    /// This ensures:
    /// - Most of the time, high priority (active projects/memory) go first
    /// - Periodically, low priority (inactive projects/libraries) get a turn
    /// - Asymmetric batches (~77% high, ~23% low) prevent large library files
    ///   from neutralizing the priority advantage
    /// - No items starve indefinitely
    pub async fn dequeue_next_batch(&self, max_batch_size: i32) -> FairnessResult<Vec<UnifiedQueueItem>> {
        // If fairness is disabled, always use priority DESC
        if !self.config.enabled {
            debug!("Fairness disabled, using priority DESC");
            return self.dequeue_global_batch(max_batch_size, true).await;
        }

        // Get current direction and determine batch size for this direction
        let (priority_descending, current_batch_limit) = {
            let state = self.alternation_state.read().await;
            let limit = if state.use_priority_descending {
                self.config.high_priority_batch
            } else {
                self.config.low_priority_batch
            };
            (state.use_priority_descending, limit)
        };

        debug!(
            "Dequeuing batch with priority {} (batch_limit={})",
            if priority_descending { "DESC (high first)" } else { "ASC (low first)" },
            current_batch_limit
        );

        // Dequeue items with current direction
        let items = self.dequeue_global_batch(max_batch_size, priority_descending).await?;
        let items_count = items.len() as u64;

        if items_count > 0 {
            // Update alternation state and metrics
            let mut state = self.alternation_state.write().await;
            let mut metrics = self.metrics.write().await;

            state.items_since_flip += items_count;
            metrics.total_items_dequeued += items_count;
            metrics.items_since_flip = state.items_since_flip;

            if priority_descending {
                metrics.high_priority_first_items += items_count;
            } else {
                metrics.low_priority_first_items += items_count;
            }

            // Check if we should flip direction (using direction-appropriate batch limit)
            if state.items_since_flip >= current_batch_limit {
                state.use_priority_descending = !state.use_priority_descending;
                state.items_since_flip = 0;
                metrics.direction_flips_total += 1;
                metrics.current_priority_descending = state.use_priority_descending;
                metrics.items_since_flip = 0;

                info!(
                    "Anti-starvation flip: switching to priority {} (flip #{})",
                    if state.use_priority_descending { "DESC" } else { "ASC" },
                    metrics.direction_flips_total
                );
            }

            info!(
                "Fairness scheduler dequeued {} items (priority {}, {}/{} until flip)",
                items_count,
                if priority_descending { "DESC" } else { "ASC" },
                state.items_since_flip,
                current_batch_limit
            );
        } else {
            debug!("No items available in queue");
        }

        Ok(items)
    }

    /// Dequeue next batch for a specific project
    ///
    /// Uses the asymmetric anti-starvation alternation for the priority direction.
    pub async fn dequeue_project_next_batch(
        &self,
        project_id: &str,
        max_batch_size: i32,
    ) -> FairnessResult<Vec<UnifiedQueueItem>> {
        let (priority_descending, current_batch_limit) = if !self.config.enabled {
            (true, self.config.high_priority_batch)
        } else {
            let state = self.alternation_state.read().await;
            let limit = if state.use_priority_descending {
                self.config.high_priority_batch
            } else {
                self.config.low_priority_batch
            };
            (state.use_priority_descending, limit)
        };

        let items = self.dequeue_project_batch(project_id, max_batch_size, priority_descending).await?;
        let items_count = items.len() as u64;

        if items_count > 0 && self.config.enabled {
            let mut state = self.alternation_state.write().await;
            let mut metrics = self.metrics.write().await;

            state.items_since_flip += items_count;
            metrics.total_items_dequeued += items_count;

            if priority_descending {
                metrics.high_priority_first_items += items_count;
            } else {
                metrics.low_priority_first_items += items_count;
            }

            if state.items_since_flip >= current_batch_limit {
                state.use_priority_descending = !state.use_priority_descending;
                state.items_since_flip = 0;
                metrics.direction_flips_total += 1;
                metrics.current_priority_descending = state.use_priority_descending;

                info!(
                    "Anti-starvation flip: switching to priority {} (flip #{})",
                    if state.use_priority_descending { "DESC" } else { "ASC" },
                    metrics.direction_flips_total
                );
            }
        }

        Ok(items)
    }

    /// Reset the alternation state
    pub async fn reset(&self) {
        let mut state = self.alternation_state.write().await;
        *state = AlternationState::default();

        let mut metrics = self.metrics.write().await;
        metrics.current_priority_descending = true;
        metrics.items_since_flip = 0;

        debug!("Fairness scheduler state reset");
    }

    /// Force a specific priority direction (for testing or manual override)
    pub async fn set_priority_direction(&self, descending: bool) {
        let mut state = self.alternation_state.write().await;
        state.use_priority_descending = descending;
        state.items_since_flip = 0;

        let mut metrics = self.metrics.write().await;
        metrics.current_priority_descending = descending;
        metrics.items_since_flip = 0;

        debug!(
            "Priority direction manually set to {}",
            if descending { "DESC" } else { "ASC" }
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fairness_scheduler_config_default() {
        let config = FairnessSchedulerConfig::default();
        assert!(config.enabled);
        assert_eq!(config.high_priority_batch, 10);
        assert_eq!(config.low_priority_batch, 3);
        assert_eq!(config.lease_duration_secs, 300);
        assert!(config.worker_id.starts_with("fairness-worker-"));
    }

    #[test]
    fn test_alternation_state_default() {
        let state = AlternationState::default();
        assert!(state.use_priority_descending);
        assert_eq!(state.items_since_flip, 0);
    }

    #[test]
    fn test_fairness_metrics_default() {
        let metrics = FairnessMetrics::default();
        assert_eq!(metrics.direction_flips_total, 0);
        assert_eq!(metrics.high_priority_first_items, 0);
        assert_eq!(metrics.low_priority_first_items, 0);
        assert_eq!(metrics.total_items_dequeued, 0);
        assert!(!metrics.current_priority_descending); // Default bool is false
        assert_eq!(metrics.items_since_flip, 0);
    }
}
