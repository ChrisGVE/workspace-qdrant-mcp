//! Configuration, warmup state, and metrics types for the unified queue processor.

use chrono::{DateTime, Duration as ChronoDuration, Utc};
use std::time::Instant;

/// Warmup state tracker for startup throttling (Task 577)
///
/// Tracks whether the daemon is still in the warmup window after startup.
/// During warmup, the queue processor uses reduced resource limits to avoid
/// CPU spikes.
#[derive(Debug, Clone)]
pub struct WarmupState {
    daemon_start: Instant,
    warmup_window_secs: u64,
}

impl WarmupState {
    /// Create a new warmup state tracker
    pub fn new(warmup_window_secs: u64) -> Self {
        Self {
            daemon_start: Instant::now(),
            warmup_window_secs,
        }
    }

    /// Check if the daemon is still in the warmup window
    pub fn is_in_warmup(&self) -> bool {
        self.daemon_start.elapsed().as_secs() < self.warmup_window_secs
    }

    /// Get the elapsed time since daemon start
    pub fn elapsed_secs(&self) -> u64 {
        self.daemon_start.elapsed().as_secs()
    }
}

/// Processing metrics for unified queue monitoring
#[derive(Debug, Clone, Default)]
pub struct UnifiedProcessingMetrics {
    /// Total items processed by type
    pub items_processed_by_type: std::collections::HashMap<String, u64>,
    /// Total items failed
    pub items_failed: u64,
    /// Current queue depth
    pub queue_depth: i64,
    /// Average processing time (milliseconds)
    pub avg_processing_time_ms: f64,
    /// Items processed per second
    pub items_per_second: f64,
    /// Last metrics update time
    pub last_update: DateTime<Utc>,
    /// Total errors by type
    pub error_counts: std::collections::HashMap<String, u64>,
}

/// Configuration for the unified queue processor
#[derive(Debug, Clone)]
pub struct UnifiedProcessorConfig {
    /// Number of items to process in each batch
    pub batch_size: i32,
    /// Polling interval in milliseconds
    pub poll_interval_ms: u64,
    /// Worker ID for lease acquisition
    pub worker_id: String,
    /// Lease duration in seconds
    pub lease_duration_secs: i64,
    /// Maximum retries before marking as failed
    pub max_retries: i32,
    /// Retry delays (exponential backoff)
    pub retry_delays: Vec<ChronoDuration>,

    // Fairness scheduler settings (asymmetric anti-starvation alternation)
    /// Whether fairness scheduling is enabled (if disabled, falls back to priority DESC always)
    pub fairness_enabled: bool,
    /// Batch size when processing high-priority items (priority DESC direction, default: 10)
    pub high_priority_batch: u64,
    /// Batch size when processing low-priority items (priority ASC / anti-starvation, default: 3)
    pub low_priority_batch: u64,

    // Resource limits (Task 504)
    /// Delay in milliseconds between processing items
    pub inter_item_delay_ms: u64,
    /// Maximum concurrent embedding operations
    pub max_concurrent_embeddings: usize,
    /// Pause processing when memory usage exceeds this percentage
    pub max_memory_percent: u8,

    // Warmup throttling (Task 577)
    /// Duration in seconds of the warmup window with reduced limits
    pub warmup_window_secs: u64,
    /// Max concurrent embeddings during warmup
    pub warmup_max_concurrent_embeddings: usize,
    /// Inter-item delay in ms during warmup
    pub warmup_inter_item_delay_ms: u64,

    // ONNX thread tuning
    /// Number of ONNX intra-op threads per embedding session (default: 2)
    pub onnx_intra_threads: usize,

    // Failed item resurrection
    /// How often (seconds) to scan for failed transient items and reset them to pending.
    /// Default: 3600 (1 hour). Set to 0 to disable.
    pub failed_resurrection_interval_secs: u64,
}

impl Default for UnifiedProcessorConfig {
    fn default() -> Self {
        Self {
            batch_size: 10,
            poll_interval_ms: 500,
            worker_id: format!("unified-worker-{}", uuid::Uuid::new_v4()),
            lease_duration_secs: 300, // 5 minutes
            max_retries: 3,
            retry_delays: vec![
                ChronoDuration::minutes(1),
                ChronoDuration::minutes(5),
                ChronoDuration::minutes(15),
                ChronoDuration::hours(1),
            ],
            // Fairness scheduler defaults (asymmetric anti-starvation)
            fairness_enabled: true,
            high_priority_batch: 10, // Spec: process 10 high-priority items per cycle
            low_priority_batch: 3,   // Spec: process 3 low-priority items per anti-starvation cycle
            // Resource limits defaults (Task 504)
            inter_item_delay_ms: 50,
            max_concurrent_embeddings: 2,
            max_memory_percent: 70,
            // Warmup throttling defaults (Task 577)
            warmup_window_secs: 30,
            warmup_max_concurrent_embeddings: 1,
            warmup_inter_item_delay_ms: 200,
            // ONNX thread tuning
            onnx_intra_threads: 2,
            // Failed item resurrection
            failed_resurrection_interval_secs: 3600,
        }
    }
}
