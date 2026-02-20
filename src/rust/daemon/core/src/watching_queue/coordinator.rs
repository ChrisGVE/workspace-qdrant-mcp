//! Watch-queue coordination protocol (Task 461.9).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;
use tracing::{debug, info, warn};

/// Configuration for the watch-queue coordinator
#[derive(Debug, Clone)]
pub struct CoordinatorConfig {
    /// Total capacity for all watchers combined (default: 10000)
    pub total_capacity: usize,
    /// Capacity reserved per watch for fair distribution
    pub min_per_watch: usize,
    /// Maximum capacity any single watch can hold
    pub max_per_watch: usize,
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        Self {
            total_capacity: 10000,
            min_per_watch: 100,
            max_per_watch: 2000,
        }
    }
}

/// Capacity allocation for a watch
#[derive(Debug, Clone)]
struct WatchAllocation {
    /// Currently held capacity
    held: usize,
    /// Total requested (including held)
    requested: usize,
    /// Last activity timestamp
    last_activity: SystemTime,
}

impl Default for WatchAllocation {
    fn default() -> Self {
        Self {
            held: 0,
            requested: 0,
            last_activity: SystemTime::now(),
        }
    }
}

/// Coordinator for watch-queue flow control (Task 461.9)
///
/// This coordinator manages capacity allocation between multiple file watchers
/// and the queue processor. It uses a semaphore-based approach where watchers
/// request capacity before enqueuing and the processor releases capacity
/// when items are dequeued.
///
/// The coordinator ensures:
/// - No single watcher can overwhelm the queue
/// - Fair distribution of capacity across watchers
/// - Backpressure when the queue is full
pub struct WatchQueueCoordinator {
    /// Configuration
    config: CoordinatorConfig,
    /// Total capacity currently allocated
    allocated: Arc<std::sync::atomic::AtomicUsize>,
    /// Per-watch allocations
    allocations: Arc<tokio::sync::RwLock<HashMap<String, WatchAllocation>>>,
    /// SQLite pool for optional persistence
    pool: Option<SqlitePool>,
    /// Metrics callback
    metrics_callback: Option<Box<dyn Fn(&str, usize, bool) + Send + Sync>>,
}

impl std::fmt::Debug for WatchQueueCoordinator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WatchQueueCoordinator")
            .field("config", &self.config)
            .field("allocated", &self.allocated.load(std::sync::atomic::Ordering::SeqCst))
            .field("has_pool", &self.pool.is_some())
            .finish()
    }
}

impl WatchQueueCoordinator {
    /// Create a new coordinator with default configuration
    pub fn new() -> Self {
        Self::with_config(CoordinatorConfig::default())
    }

    /// Create a coordinator with custom configuration
    pub fn with_config(config: CoordinatorConfig) -> Self {
        Self {
            config,
            allocated: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            allocations: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            pool: None,
            metrics_callback: None,
        }
    }

    /// Create a coordinator with SQLite persistence
    pub fn with_pool(pool: SqlitePool) -> Self {
        Self {
            config: CoordinatorConfig::default(),
            allocated: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            allocations: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            pool: Some(pool),
            metrics_callback: None,
        }
    }

    /// Set a metrics callback for capacity changes
    pub fn with_metrics_callback<F>(mut self, callback: F) -> Self
    where
        F: Fn(&str, usize, bool) + Send + Sync + 'static,
    {
        self.metrics_callback = Some(Box::new(callback));
        self
    }

    /// Request capacity for a watch
    ///
    /// Returns true if the requested capacity was granted, false if it was
    /// denied due to insufficient available capacity.
    pub async fn request_capacity(&self, watch_id: &str, num_items: usize) -> bool {
        // Check available capacity
        let current_allocated = self.allocated.load(std::sync::atomic::Ordering::SeqCst);
        if current_allocated + num_items > self.config.total_capacity {
            debug!(
                "Capacity request denied for {}: current={}, requested={}, total={}",
                watch_id, current_allocated, num_items, self.config.total_capacity
            );
            return false;
        }

        // Check per-watch limits
        let mut allocations = self.allocations.write().await;
        let allocation = allocations
            .entry(watch_id.to_string())
            .or_insert_with(Default::default);

        if allocation.held + num_items > self.config.max_per_watch {
            debug!(
                "Capacity request denied for {}: per-watch limit (held={}, requested={}, max={})",
                watch_id, allocation.held, num_items, self.config.max_per_watch
            );
            return false;
        }

        // Grant capacity
        allocation.held += num_items;
        allocation.requested += num_items;
        allocation.last_activity = SystemTime::now();
        self.allocated.fetch_add(num_items, std::sync::atomic::Ordering::SeqCst);

        // Invoke metrics callback
        if let Some(ref callback) = self.metrics_callback {
            callback(watch_id, num_items, true);
        }

        debug!(
            "Capacity granted for {}: {} items (total held: {})",
            watch_id, num_items, allocation.held
        );

        true
    }

    /// Release capacity after items are processed
    pub async fn release_capacity(&self, watch_id: &str, num_items: usize) {
        let mut allocations = self.allocations.write().await;

        if let Some(allocation) = allocations.get_mut(watch_id) {
            let to_release = num_items.min(allocation.held);
            allocation.held = allocation.held.saturating_sub(to_release);
            allocation.last_activity = SystemTime::now();

            self.allocated.fetch_sub(to_release, std::sync::atomic::Ordering::SeqCst);

            // Invoke metrics callback
            if let Some(ref callback) = self.metrics_callback {
                callback(watch_id, to_release, false);
            }

            debug!(
                "Capacity released for {}: {} items (remaining: {})",
                watch_id, to_release, allocation.held
            );
        } else {
            warn!("Release capacity called for unknown watch: {}", watch_id);
        }
    }

    /// Get available capacity
    pub fn get_available_capacity(&self) -> usize {
        let allocated = self.allocated.load(std::sync::atomic::Ordering::SeqCst);
        self.config.total_capacity.saturating_sub(allocated)
    }

    /// Get total capacity
    pub fn get_total_capacity(&self) -> usize {
        self.config.total_capacity
    }

    /// Get currently allocated capacity
    pub fn get_allocated_capacity(&self) -> usize {
        self.allocated.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Get allocation for a specific watch
    pub async fn get_watch_allocation(&self, watch_id: &str) -> Option<usize> {
        let allocations = self.allocations.read().await;
        allocations.get(watch_id).map(|a| a.held)
    }

    /// Get allocation summary for all watches
    pub async fn get_allocation_summary(&self) -> CoordinatorSummary {
        let allocations = self.allocations.read().await;
        let per_watch: HashMap<String, usize> = allocations
            .iter()
            .map(|(k, v)| (k.clone(), v.held))
            .collect();

        CoordinatorSummary {
            total_capacity: self.config.total_capacity,
            allocated_capacity: self.allocated.load(std::sync::atomic::Ordering::SeqCst),
            available_capacity: self.get_available_capacity(),
            num_watches: allocations.len(),
            per_watch_allocation: per_watch,
        }
    }

    /// Reset allocation for a watch (e.g., when watch is stopped)
    pub async fn reset_watch(&self, watch_id: &str) {
        let mut allocations = self.allocations.write().await;
        if let Some(allocation) = allocations.remove(watch_id) {
            self.allocated.fetch_sub(allocation.held, std::sync::atomic::Ordering::SeqCst);
            info!("Reset allocation for watch {}: released {} items", watch_id, allocation.held);
        }
    }

    /// Clean up stale allocations (watches that haven't been active recently)
    pub async fn cleanup_stale(&self, max_age: Duration) {
        let now = SystemTime::now();
        let mut allocations = self.allocations.write().await;

        let stale: Vec<String> = allocations
            .iter()
            .filter(|(_, alloc)| {
                now.duration_since(alloc.last_activity)
                    .map(|d| d > max_age)
                    .unwrap_or(false)
            })
            .map(|(k, _)| k.clone())
            .collect();

        for watch_id in stale {
            if let Some(allocation) = allocations.remove(&watch_id) {
                self.allocated.fetch_sub(allocation.held, std::sync::atomic::Ordering::SeqCst);
                info!("Cleaned up stale allocation for {}: released {} items", watch_id, allocation.held);
            }
        }
    }
}

impl Default for WatchQueueCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of coordinator state for telemetry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinatorSummary {
    pub total_capacity: usize,
    pub allocated_capacity: usize,
    pub available_capacity: usize,
    pub num_watches: usize,
    pub per_watch_allocation: HashMap<String, usize>,
}
