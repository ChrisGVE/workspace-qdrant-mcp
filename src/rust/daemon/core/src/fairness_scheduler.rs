//! Fairness Scheduler Module
//!
//! Implements Task 34: Weighted scheduling algorithm that processes N items per active
//! project before processing global oldest items. Provides fairness across projects
//! while preventing starvation.

use chrono::{DateTime, Duration as ChronoDuration, Utc};
use std::collections::HashMap;
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
    /// Whether fairness scheduling is enabled (if disabled, falls back to FIFO)
    pub enabled: bool,

    /// Maximum items to dequeue per active project before moving to next
    pub items_per_active_project: i32,

    /// Maximum age in seconds for global items before starvation guard triggers
    pub max_global_item_age_secs: i64,

    /// Threshold in hours for considering a project "active"
    pub active_project_threshold_hours: i64,

    /// Worker ID for lease acquisition
    pub worker_id: String,

    /// Lease duration in seconds
    pub lease_duration_secs: i64,
}

impl Default for FairnessSchedulerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            items_per_active_project: 5,
            max_global_item_age_secs: 600, // 10 minutes
            active_project_threshold_hours: 1,
            worker_id: format!("fairness-worker-{}", uuid::Uuid::new_v4()),
            lease_duration_secs: 300, // 5 minutes
        }
    }
}

/// Local tracking state for an active project during scheduling
#[derive(Debug, Clone)]
struct ProjectSchedulingState {
    /// Project ID (tenant_id)
    project_id: String,

    /// Number of items processed in the current round
    items_processed_this_round: i32,

    /// Last time we processed an item for this project
    last_processed_at: DateTime<Utc>,
}

impl ProjectSchedulingState {
    fn new(project_id: &str) -> Self {
        Self {
            project_id: project_id.to_string(),
            items_processed_this_round: 0,
            last_processed_at: Utc::now(),
        }
    }

    fn increment(&mut self) {
        self.items_processed_this_round += 1;
        self.last_processed_at = Utc::now();
    }

    fn reset_round(&mut self) {
        self.items_processed_this_round = 0;
    }
}

/// Metrics tracked by the fairness scheduler
#[derive(Debug, Clone, Default)]
pub struct FairnessMetrics {
    /// Number of times starvation guard was triggered
    pub starvation_overrides_total: u64,

    /// Current count of active projects
    pub active_projects_count: u64,

    /// Items dequeued per project in current round
    pub items_per_project: HashMap<String, u64>,

    /// Total items dequeued via fairness scheduling
    pub total_items_dequeued: u64,

    /// Total global (non-project) items dequeued
    pub global_items_dequeued: u64,

    /// Last refresh of active projects
    pub last_active_projects_refresh: Option<DateTime<Utc>>,
}

/// Fairness scheduler for balanced queue processing
///
/// Implements a weighted scheduling algorithm that ensures all active projects
/// get fair processing time while preventing starvation of older items.
pub struct FairnessScheduler {
    /// Queue manager for database operations
    queue_manager: QueueManager,

    /// Scheduler configuration
    config: FairnessSchedulerConfig,

    /// Local tracking of active projects for scheduling
    project_states: Arc<RwLock<HashMap<String, ProjectSchedulingState>>>,

    /// Current position in round-robin (index into active projects)
    current_project_index: Arc<RwLock<usize>>,

    /// Cached list of active project IDs (refreshed periodically)
    active_project_ids: Arc<RwLock<Vec<String>>>,

    /// Metrics for monitoring
    metrics: Arc<RwLock<FairnessMetrics>>,
}

impl FairnessScheduler {
    /// Create a new fairness scheduler
    pub fn new(queue_manager: QueueManager, config: FairnessSchedulerConfig) -> Self {
        info!(
            "Creating fairness scheduler (enabled={}, items_per_project={}, max_age={}s)",
            config.enabled, config.items_per_active_project, config.max_global_item_age_secs
        );

        Self {
            queue_manager,
            config,
            project_states: Arc::new(RwLock::new(HashMap::new())),
            current_project_index: Arc::new(RwLock::new(0)),
            active_project_ids: Arc::new(RwLock::new(Vec::new())),
            metrics: Arc::new(RwLock::new(FairnessMetrics::default())),
        }
    }

    /// Get current fairness metrics
    pub async fn get_metrics(&self) -> FairnessMetrics {
        self.metrics.read().await.clone()
    }

    /// Refresh the list of active projects from the database
    ///
    /// Active projects are those with:
    /// - items_in_queue > 0, OR
    /// - activity within the threshold hours
    pub async fn refresh_active_projects(&self) -> FairnessResult<Vec<String>> {
        let threshold = Utc::now() - ChronoDuration::hours(self.config.active_project_threshold_hours);

        // Get all active projects from the database
        let projects = self.queue_manager
            .list_active_projects(None, false, None)
            .await?;

        // Filter to those that are truly active (have queue items or recent activity)
        let active_ids: Vec<String> = projects
            .into_iter()
            .filter(|p| p.items_in_queue > 0 || p.last_activity_at >= threshold)
            .map(|p| p.project_id)
            .collect();

        // Update cached list
        {
            let mut ids = self.active_project_ids.write().await;
            *ids = active_ids.clone();
        }

        // Update project states - add new, keep existing
        {
            let mut states = self.project_states.write().await;

            // Add states for new projects
            for project_id in &active_ids {
                states
                    .entry(project_id.clone())
                    .or_insert_with(|| ProjectSchedulingState::new(project_id));
            }

            // Remove states for projects no longer active
            states.retain(|id, _| active_ids.contains(id));
        }

        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.active_projects_count = active_ids.len() as u64;
            metrics.last_active_projects_refresh = Some(Utc::now());
        }

        debug!(
            "Refreshed active projects: {} active",
            active_ids.len()
        );

        Ok(active_ids)
    }

    /// Check if there's a stale global item that should be prioritized
    ///
    /// Returns the item if found, along with how old it is in seconds.
    pub async fn check_starvation_guard(&self) -> FairnessResult<Option<(UnifiedQueueItem, i64)>> {
        let max_age_secs = self.config.max_global_item_age_secs;
        let threshold = Utc::now() - ChronoDuration::seconds(max_age_secs);

        // Query for oldest pending item across all projects
        let oldest_item = self.queue_manager
            .get_oldest_pending_unified_item()
            .await?;

        if let Some(item) = oldest_item {
            // Parse the created_at timestamp
            if let Ok(created_at) = DateTime::parse_from_rfc3339(&item.created_at) {
                let age_secs = (Utc::now() - created_at.with_timezone(&Utc)).num_seconds();

                if created_at.with_timezone(&Utc) < threshold {
                    debug!(
                        "Starvation guard triggered: item {} is {}s old (threshold: {}s)",
                        item.queue_id, age_secs, max_age_secs
                    );
                    return Ok(Some((item, age_secs)));
                }
            }
        }

        Ok(None)
    }

    /// Dequeue items for a specific project
    ///
    /// Returns up to N items for the given project.
    async fn dequeue_project_batch(
        &self,
        project_id: &str,
        max_items: i32,
    ) -> FairnessResult<Vec<UnifiedQueueItem>> {
        let items = self.queue_manager
            .dequeue_unified(
                max_items,
                &self.config.worker_id,
                Some(self.config.lease_duration_secs),
                Some(project_id),
                None,
            )
            .await?;

        Ok(items)
    }

    /// Dequeue the oldest global item (regardless of project)
    async fn dequeue_global_oldest(&self) -> FairnessResult<Option<UnifiedQueueItem>> {
        let items = self.queue_manager
            .dequeue_unified(
                1,
                &self.config.worker_id,
                Some(self.config.lease_duration_secs),
                None,
                None,
            )
            .await?;

        Ok(items.into_iter().next())
    }

    /// Main scheduling method: dequeue the next batch of items
    ///
    /// Algorithm:
    /// 1. Check starvation guard - if triggered, prioritize oldest global item
    /// 2. For each active project (round-robin):
    ///    - Dequeue up to N items
    ///    - Track how many processed this round
    /// 3. After all active projects get their items, dequeue 1 global oldest
    /// 4. Repeat
    pub async fn dequeue_next_batch(&self, max_batch_size: i32) -> FairnessResult<Vec<UnifiedQueueItem>> {
        // If fairness is disabled, fall back to simple FIFO
        if !self.config.enabled {
            debug!("Fairness disabled, using FIFO dequeue");
            return Ok(self.queue_manager
                .dequeue_unified(
                    max_batch_size,
                    &self.config.worker_id,
                    Some(self.config.lease_duration_secs),
                    None,
                    None,
                )
                .await?);
        }

        let mut batch = Vec::new();
        let items_per_project = self.config.items_per_active_project;

        // Step 1: Check starvation guard
        if let Some((stale_item, age_secs)) = self.check_starvation_guard().await? {
            info!(
                "Starvation guard: prioritizing item {} (age: {}s)",
                stale_item.queue_id, age_secs
            );

            // Dequeue this specific item
            let items = self.queue_manager
                .dequeue_unified(
                    1,
                    &self.config.worker_id,
                    Some(self.config.lease_duration_secs),
                    Some(&stale_item.tenant_id),
                    None,
                )
                .await?;

            if !items.is_empty() {
                batch.extend(items);

                // Update starvation metrics
                {
                    let mut metrics = self.metrics.write().await;
                    metrics.starvation_overrides_total += 1;
                    metrics.global_items_dequeued += 1;
                }

                // If batch is full, return
                if batch.len() as i32 >= max_batch_size {
                    return Ok(batch);
                }
            }
        }

        // Step 2: Refresh active projects periodically
        let active_ids = {
            let last_refresh = self.metrics.read().await.last_active_projects_refresh;
            let should_refresh = match last_refresh {
                Some(t) => (Utc::now() - t).num_seconds() > 60, // Refresh every minute
                None => true,
            };

            if should_refresh {
                self.refresh_active_projects().await?
            } else {
                self.active_project_ids.read().await.clone()
            }
        };

        if active_ids.is_empty() {
            // No active projects, fall back to global dequeue
            debug!("No active projects, dequeuing globally");
            let remaining = max_batch_size - batch.len() as i32;
            if remaining > 0 {
                let items = self.dequeue_global_oldest().await?;
                if let Some(item) = items {
                    batch.push(item);
                }
            }
            return Ok(batch);
        }

        // Step 3: Round-robin through active projects
        let num_projects = active_ids.len();
        let mut projects_visited = 0;
        let mut all_projects_exhausted = false;

        while batch.len() < max_batch_size as usize && !all_projects_exhausted {
            let current_idx = {
                let idx = self.current_project_index.read().await;
                *idx
            };

            let project_id = &active_ids[current_idx % num_projects];

            // Check how many items this project has gotten this round
            let items_this_round = {
                let states = self.project_states.read().await;
                states
                    .get(project_id)
                    .map(|s| s.items_processed_this_round)
                    .unwrap_or(0)
            };

            if items_this_round < items_per_project {
                // Still have quota for this project
                let remaining_quota = items_per_project - items_this_round;
                let remaining_batch = max_batch_size - batch.len() as i32;
                let to_fetch = remaining_quota.min(remaining_batch);

                let items = self.dequeue_project_batch(project_id, to_fetch).await?;

                if !items.is_empty() {
                    let count = items.len();
                    batch.extend(items);

                    // Update project state
                    {
                        let mut states = self.project_states.write().await;
                        if let Some(state) = states.get_mut(project_id) {
                            for _ in 0..count {
                                state.increment();
                            }
                        }
                    }

                    // Update metrics
                    {
                        let mut metrics = self.metrics.write().await;
                        *metrics.items_per_project.entry(project_id.clone()).or_insert(0) += count as u64;
                        metrics.total_items_dequeued += count as u64;
                    }

                    debug!(
                        "Dequeued {} items for project {} (this round: {})",
                        count,
                        project_id,
                        items_this_round + count as i32
                    );
                }
            }

            // Move to next project
            {
                let mut idx = self.current_project_index.write().await;
                *idx = (*idx + 1) % num_projects;
            }
            projects_visited += 1;

            // Check if we've completed a full round
            if projects_visited >= num_projects {
                // Reset all project round counters
                {
                    let mut states = self.project_states.write().await;
                    for state in states.values_mut() {
                        state.reset_round();
                    }
                }

                // After a full round, dequeue 1 global oldest item
                if batch.len() < max_batch_size as usize {
                    if let Some(item) = self.dequeue_global_oldest().await? {
                        debug!("Dequeued global oldest item: {}", item.queue_id);
                        batch.push(item);

                        let mut metrics = self.metrics.write().await;
                        metrics.global_items_dequeued += 1;
                        metrics.total_items_dequeued += 1;
                    }
                }

                // Start a new round
                projects_visited = 0;

                // Check if all projects are exhausted
                // (no items were dequeued in the last full round)
                if batch.is_empty() {
                    all_projects_exhausted = true;
                }
            }
        }

        if batch.is_empty() {
            debug!("No items available in queue");
        } else {
            info!(
                "Fairness scheduler dequeued {} items (from {} active projects)",
                batch.len(),
                active_ids.len()
            );
        }

        Ok(batch)
    }

    /// Reset the round-robin position and project states
    pub async fn reset(&self) {
        let mut idx = self.current_project_index.write().await;
        *idx = 0;

        let mut states = self.project_states.write().await;
        for state in states.values_mut() {
            state.reset_round();
        }

        debug!("Fairness scheduler state reset");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fairness_scheduler_config_default() {
        let config = FairnessSchedulerConfig::default();
        assert!(config.enabled);
        assert_eq!(config.items_per_active_project, 5);
        assert_eq!(config.max_global_item_age_secs, 600);
        assert_eq!(config.active_project_threshold_hours, 1);
        assert_eq!(config.lease_duration_secs, 300);
        assert!(config.worker_id.starts_with("fairness-worker-"));
    }

    #[test]
    fn test_project_scheduling_state() {
        let mut state = ProjectSchedulingState::new("test-project");
        assert_eq!(state.project_id, "test-project");
        assert_eq!(state.items_processed_this_round, 0);

        state.increment();
        assert_eq!(state.items_processed_this_round, 1);

        state.increment();
        assert_eq!(state.items_processed_this_round, 2);

        state.reset_round();
        assert_eq!(state.items_processed_this_round, 0);
    }

    #[test]
    fn test_fairness_metrics_default() {
        let metrics = FairnessMetrics::default();
        assert_eq!(metrics.starvation_overrides_total, 0);
        assert_eq!(metrics.active_projects_count, 0);
        assert!(metrics.items_per_project.is_empty());
        assert_eq!(metrics.total_items_dequeued, 0);
        assert_eq!(metrics.global_items_dequeued, 0);
        assert!(metrics.last_active_projects_refresh.is_none());
    }
}
