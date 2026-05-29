//! Shared types for queue processing
//!
//! Types used across multiple queue-related modules.

use chrono::Duration as ChronoDuration;

/// Configuration for the queue processor
#[derive(Debug, Clone)]
pub struct ProcessorConfig {
    /// Number of items to dequeue in each batch
    pub batch_size: i32,

    /// Poll interval between batches (milliseconds)
    pub poll_interval_ms: u64,

    /// Maximum number of retry attempts
    pub max_retries: i32,

    /// Retry delay intervals (exponential backoff)
    pub retry_delays: Vec<ChronoDuration>,

    /// Target processing throughput (docs per minute)
    pub target_throughput: u64,

    /// Enable performance monitoring
    pub enable_metrics: bool,

    /// Number of parallel workers for batch processing (Task 21)
    /// Higher values increase throughput but use more resources
    pub worker_count: usize,

    /// Maximum queue depth before enabling backpressure (Task 21)
    /// When exceeded, enqueue operations may be slowed
    pub backpressure_threshold: i64,
}

impl Default for ProcessorConfig {
    fn default() -> Self {
        Self {
            batch_size: 10,
            poll_interval_ms: 500,
            max_retries: 5,
            retry_delays: vec![
                ChronoDuration::minutes(1),
                ChronoDuration::minutes(5),
                ChronoDuration::minutes(15),
                ChronoDuration::hours(1),
            ],
            target_throughput: 1000, // 1000+ docs/min
            enable_metrics: true,
            worker_count: 4,              // Default to 4 parallel workers
            backpressure_threshold: 1000, // Start backpressure at 1000 items
        }
    }
}
