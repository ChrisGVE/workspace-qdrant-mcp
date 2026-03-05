//! Queue depth monitoring and adaptive throttling (Task 461.8).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::queue_operations::QueueManager;

/// Queue load level for adaptive throttling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueueLoadLevel {
    /// Normal load - no throttling needed
    Normal,
    /// High load - moderate throttling recommended
    High,
    /// Critical load - aggressive throttling required
    Critical,
}

impl QueueLoadLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            QueueLoadLevel::Normal => "normal",
            QueueLoadLevel::High => "high",
            QueueLoadLevel::Critical => "critical",
        }
    }
}

/// Configuration for queue depth throttling
#[derive(Debug, Clone)]
pub struct QueueThrottleConfig {
    /// Queue depth threshold for high load (default: 1000)
    pub high_threshold: i64,
    /// Queue depth threshold for critical load (default: 5000)
    pub critical_threshold: i64,
    /// How often to check queue depth in milliseconds (default: 5000)
    pub check_interval_ms: u64,
    /// Skip ratio when in high load (skip 1 in N events, default: 2)
    pub high_skip_ratio: u64,
    /// Skip ratio when in critical load (skip 1 in N events, default: 4)
    pub critical_skip_ratio: u64,
}

impl Default for QueueThrottleConfig {
    fn default() -> Self {
        Self {
            high_threshold: 1000,
            critical_threshold: 5000,
            check_interval_ms: 5000,
            high_skip_ratio: 2,
            critical_skip_ratio: 4,
        }
    }
}

/// State for queue depth throttling
#[derive(Debug)]
pub struct QueueThrottleState {
    /// Current queue depth (periodically updated)
    current_depth: Arc<tokio::sync::RwLock<i64>>,
    /// Current load level
    load_level: Arc<tokio::sync::RwLock<QueueLoadLevel>>,
    /// Per-collection depths
    collection_depths: Arc<tokio::sync::RwLock<HashMap<String, i64>>>,
    /// Event counter for skip ratio calculation
    event_counter: Arc<std::sync::atomic::AtomicU64>,
    /// Configuration
    config: QueueThrottleConfig,
    /// Last check timestamp
    last_check: Arc<tokio::sync::RwLock<SystemTime>>,
}

impl QueueThrottleState {
    /// Create a new throttle state with default configuration
    pub fn new() -> Self {
        Self::with_config(QueueThrottleConfig::default())
    }

    /// Create a new throttle state with custom configuration
    pub fn with_config(config: QueueThrottleConfig) -> Self {
        Self {
            current_depth: Arc::new(tokio::sync::RwLock::new(0)),
            load_level: Arc::new(tokio::sync::RwLock::new(QueueLoadLevel::Normal)),
            collection_depths: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            event_counter: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            config,
            last_check: Arc::new(tokio::sync::RwLock::new(SystemTime::UNIX_EPOCH)),
        }
    }

    /// Update queue depth from queue manager (using unified_queue)
    pub async fn update_from_queue(&self, queue_manager: &QueueManager) {
        match queue_manager.get_unified_queue_depth(None, None).await {
            Ok(depth) => {
                let mut current = self.current_depth.write().await;
                *current = depth;

                // Update load level
                let new_level = if depth >= self.config.critical_threshold {
                    QueueLoadLevel::Critical
                } else if depth >= self.config.high_threshold {
                    QueueLoadLevel::High
                } else {
                    QueueLoadLevel::Normal
                };

                let mut level = self.load_level.write().await;
                if *level != new_level {
                    info!(
                        "Queue load level changed: {:?} -> {:?} (depth: {})",
                        *level, new_level, depth
                    );
                }
                *level = new_level;

                // Update last check time
                let mut last = self.last_check.write().await;
                *last = SystemTime::now();
            }
            Err(e) => {
                warn!("Failed to get queue depth: {}", e);
            }
        }

        // Also update per-collection depths (using unified_queue)
        match queue_manager
            .get_unified_queue_depth_all_collections()
            .await
        {
            Ok(depths) => {
                let mut collection_depths = self.collection_depths.write().await;
                *collection_depths = depths;
            }
            Err(e) => {
                warn!("Failed to get per-collection queue depths: {}", e);
            }
        }
    }

    /// Check if we should throttle (skip this event)
    pub async fn should_throttle(&self) -> bool {
        let level = *self.load_level.read().await;
        match level {
            QueueLoadLevel::Normal => false,
            QueueLoadLevel::High => {
                let count = self
                    .event_counter
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                count % self.config.high_skip_ratio != 0
            }
            QueueLoadLevel::Critical => {
                let count = self
                    .event_counter
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                count % self.config.critical_skip_ratio != 0
            }
        }
    }

    /// Check if we need to refresh queue depth (time-based)
    pub async fn needs_refresh(&self) -> bool {
        let last = *self.last_check.read().await;
        let elapsed = SystemTime::now()
            .duration_since(last)
            .unwrap_or(Duration::ZERO);
        elapsed >= Duration::from_millis(self.config.check_interval_ms)
    }

    /// Get current queue depth
    pub async fn get_depth(&self) -> i64 {
        *self.current_depth.read().await
    }

    /// Get current load level
    pub async fn get_load_level(&self) -> QueueLoadLevel {
        *self.load_level.read().await
    }

    /// Get queue depth for a specific collection
    pub async fn get_collection_depth(&self, collection: &str) -> i64 {
        let depths = self.collection_depths.read().await;
        depths.get(collection).copied().unwrap_or(0)
    }

    /// Get throttle summary for telemetry
    pub async fn get_summary(&self) -> QueueThrottleSummary {
        QueueThrottleSummary {
            total_depth: *self.current_depth.read().await,
            load_level: *self.load_level.read().await,
            events_processed: self.event_counter.load(std::sync::atomic::Ordering::SeqCst),
            high_threshold: self.config.high_threshold,
            critical_threshold: self.config.critical_threshold,
        }
    }
}

impl Default for QueueThrottleState {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of throttle state for telemetry (Task 461.8)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueThrottleSummary {
    pub total_depth: i64,
    pub load_level: QueueLoadLevel,
    pub events_processed: u64,
    pub high_threshold: i64,
    pub critical_threshold: i64,
}
