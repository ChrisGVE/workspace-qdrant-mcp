//! Fairness Scheduler Module
//!
//! Implements Task 21: 10-item anti-starvation alternation.
//! Every 10 items, flips between priority DESC (active first) and priority ASC
//! (inactive projects get a turn) to prevent starvation.

use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;
use tracing::{debug, info};

use crate::queue_operations::{QueueManager, QueueError};
use crate::unified_queue_schema::{ItemType, UnifiedQueueItem};

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

    /// Number of items between priority direction flips (spec: 10)
    pub items_per_flip: u64,

    /// Worker ID for lease acquisition
    pub worker_id: String,

    /// Lease duration in seconds
    pub lease_duration_secs: i64,
}

impl Default for FairnessSchedulerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            items_per_flip: 10, // Spec: every 10 items
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
/// Implements the spec's anti-starvation mechanism: every 10 items,
/// alternate between priority DESC (active projects first) and
/// priority ASC (inactive projects get a turn).
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
            "Creating fairness scheduler (enabled={}, items_per_flip={})",
            config.enabled, config.items_per_flip
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

    /// Dequeue items by type with current priority direction
    async fn dequeue_by_type(
        &self,
        max_items: i32,
        item_type: ItemType,
        priority_descending: bool,
    ) -> FairnessResult<Vec<UnifiedQueueItem>> {
        let items = self.queue_manager
            .dequeue_unified(
                max_items,
                &self.config.worker_id,
                Some(self.config.lease_duration_secs),
                None,
                Some(item_type),
                Some(priority_descending),
            )
            .await?;

        Ok(items)
    }

    /// Main scheduling method: dequeue the next batch of items
    ///
    /// Algorithm (Task 21 - Anti-starvation alternation):
    /// 1. Get current priority direction (DESC or ASC)
    /// 2. Dequeue up to batch_size items with that direction
    /// 3. Track items processed
    /// 4. Every items_per_flip items (default 10), flip the direction
    ///
    /// This ensures:
    /// - Most of the time, high priority (active projects/memory) go first
    /// - Every 10 items, low priority (inactive projects/libraries) get a turn
    /// - No items starve indefinitely
    pub async fn dequeue_next_batch(&self, max_batch_size: i32) -> FairnessResult<Vec<UnifiedQueueItem>> {
        // If fairness is disabled, always use priority DESC
        if !self.config.enabled {
            debug!("Fairness disabled, using priority DESC");
            return self.dequeue_global_batch(max_batch_size, true).await;
        }

        // Get current direction
        let priority_descending = {
            let state = self.alternation_state.read().await;
            state.use_priority_descending
        };

        debug!(
            "Dequeuing batch with priority {} (items_per_flip={})",
            if priority_descending { "DESC (high first)" } else { "ASC (low first)" },
            self.config.items_per_flip
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

            // Check if we should flip direction
            if state.items_since_flip >= self.config.items_per_flip {
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
                self.config.items_per_flip
            );
        } else {
            debug!("No items available in queue");
        }

        Ok(items)
    }

    /// Dequeue next batch for a specific project
    ///
    /// Uses the anti-starvation alternation for the priority direction.
    pub async fn dequeue_project_next_batch(
        &self,
        project_id: &str,
        max_batch_size: i32,
    ) -> FairnessResult<Vec<UnifiedQueueItem>> {
        let priority_descending = if !self.config.enabled {
            true
        } else {
            let state = self.alternation_state.read().await;
            state.use_priority_descending
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

            if state.items_since_flip >= self.config.items_per_flip {
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
        assert_eq!(config.items_per_flip, 10);
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
