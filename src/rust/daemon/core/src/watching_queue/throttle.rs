//! Queue depth monitoring and adaptive throttling (Task 461.8).

use std::collections::HashMap;
use std::sync::atomic::AtomicBool;
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
    /// Set when events are dropped under critical pressure (F-045).
    /// The reconciliation loop should check and clear this flag.
    needs_full_reconcile: Arc<AtomicBool>,
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
            needs_full_reconcile: Arc::new(AtomicBool::new(false)),
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

    /// Check if we should throttle (skip this event).
    ///
    /// At Critical load, all throttled events set `needs_full_reconcile` so
    /// the watcher reconciliation loop can catch up later (F-045). At High
    /// load, the skip ratio reduces throughput without losing events that
    /// matter — the unthrottled events provide enough coverage.
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
                let throttled = count % self.config.critical_skip_ratio != 0;
                if throttled {
                    // Mark that a full reconcile is needed so dropped events
                    // are eventually recovered (F-045).
                    self.needs_full_reconcile
                        .store(true, std::sync::atomic::Ordering::Release);
                }
                throttled
            }
        }
    }

    /// Returns `true` and clears the flag if a full reconcile was requested
    /// because events were dropped under critical queue pressure.
    pub fn take_needs_full_reconcile(&self) -> bool {
        self.needs_full_reconcile
            .swap(false, std::sync::atomic::Ordering::AcqRel)
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Force the state to a given load level + depth. Production code only
    /// transitions via `update_from_queue`, which needs a real QueueManager;
    /// these helpers let unit tests exercise the pure throttling math.
    async fn force_state(state: &QueueThrottleState, depth: i64, level: QueueLoadLevel) {
        *state.current_depth.write().await = depth;
        *state.load_level.write().await = level;
    }

    #[test]
    fn load_level_str_round_trip() {
        assert_eq!(QueueLoadLevel::Normal.as_str(), "normal");
        assert_eq!(QueueLoadLevel::High.as_str(), "high");
        assert_eq!(QueueLoadLevel::Critical.as_str(), "critical");
    }

    #[test]
    fn default_config_uses_documented_thresholds() {
        let c = QueueThrottleConfig::default();
        assert_eq!(c.high_threshold, 1000);
        assert_eq!(c.critical_threshold, 5000);
        assert_eq!(c.check_interval_ms, 5000);
        assert_eq!(c.high_skip_ratio, 2);
        assert_eq!(c.critical_skip_ratio, 4);
    }

    #[tokio::test]
    async fn initial_state_is_normal_load_zero_depth() {
        let s = QueueThrottleState::new();
        assert_eq!(s.get_depth().await, 0);
        assert_eq!(s.get_load_level().await, QueueLoadLevel::Normal);
        assert!(!s.take_needs_full_reconcile());
    }

    #[tokio::test]
    async fn should_throttle_normal_always_false() {
        let s = QueueThrottleState::new();
        for _ in 0..10 {
            assert!(!s.should_throttle().await);
        }
        // F-045 flag stays clear under normal load.
        assert!(!s.take_needs_full_reconcile());
    }

    #[tokio::test]
    async fn should_throttle_high_skips_one_in_n() {
        // With high_skip_ratio=2 the pattern is throttle/keep/throttle/keep...
        // (count starts at 0 → 0 % 2 == 0 → keep; 1 % 2 != 0 → throttle).
        let s = QueueThrottleState::new();
        force_state(&s, 1500, QueueLoadLevel::High).await;

        let mut kept = 0usize;
        let mut throttled = 0usize;
        for _ in 0..100 {
            if s.should_throttle().await {
                throttled += 1;
            } else {
                kept += 1;
            }
        }
        assert_eq!(kept, 50);
        assert_eq!(throttled, 50);
        // High load alone never sets the reconcile flag — that's
        // reserved for Critical.
        assert!(!s.take_needs_full_reconcile());
    }

    #[tokio::test]
    async fn should_throttle_critical_skips_three_in_four_and_flags_reconcile() {
        let s = QueueThrottleState::new();
        force_state(&s, 6000, QueueLoadLevel::Critical).await;

        let mut throttled = 0usize;
        for _ in 0..100 {
            if s.should_throttle().await {
                throttled += 1;
            }
        }
        // skip_ratio = 4 → throttle when count % 4 != 0 → 75/100.
        assert_eq!(throttled, 75);
        // F-045: any throttled event under critical load arms the flag.
        assert!(s.take_needs_full_reconcile());
    }

    #[tokio::test]
    async fn take_needs_full_reconcile_is_consumed_on_read() {
        let s = QueueThrottleState::new();
        force_state(&s, 6000, QueueLoadLevel::Critical).await;
        for _ in 0..4 {
            let _ = s.should_throttle().await;
        }
        assert!(s.take_needs_full_reconcile());
        // Second read after consumption returns false (flag was cleared).
        assert!(!s.take_needs_full_reconcile());
    }

    #[tokio::test]
    async fn needs_refresh_initially_true() {
        // last_check starts at UNIX_EPOCH, so any positive interval has
        // already elapsed and a refresh is owed.
        let s = QueueThrottleState::new();
        assert!(s.needs_refresh().await);
    }

    #[tokio::test]
    async fn get_collection_depth_returns_zero_for_unknown() {
        let s = QueueThrottleState::new();
        assert_eq!(s.get_collection_depth("never-seen").await, 0);
    }

    #[tokio::test]
    async fn get_summary_reflects_state_and_config() {
        let cfg = QueueThrottleConfig {
            high_threshold: 100,
            critical_threshold: 500,
            check_interval_ms: 1000,
            high_skip_ratio: 3,
            critical_skip_ratio: 5,
        };
        let s = QueueThrottleState::with_config(cfg);
        force_state(&s, 250, QueueLoadLevel::High).await;
        // Drive the event counter so the summary's events_processed is
        // distinguishable from a fresh state.
        for _ in 0..6 {
            let _ = s.should_throttle().await;
        }

        let summary = s.get_summary().await;
        assert_eq!(summary.total_depth, 250);
        assert_eq!(summary.load_level, QueueLoadLevel::High);
        assert_eq!(summary.events_processed, 6);
        assert_eq!(summary.high_threshold, 100);
        assert_eq!(summary.critical_threshold, 500);
    }
}
