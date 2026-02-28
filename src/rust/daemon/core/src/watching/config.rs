//! Configuration types for the file watching system

use serde::{Deserialize, Serialize};

use crate::processing::TaskPriority;

/// File watching configuration with comprehensive options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatcherConfig {
    /// Patterns to include (glob patterns)
    pub include_patterns: Vec<String>,

    /// Patterns to exclude (glob patterns)
    pub exclude_patterns: Vec<String>,

    /// Whether to watch directories recursively
    pub recursive: bool,

    /// Maximum recursion depth (-1 for unlimited)
    pub max_depth: i32,

    /// Debounce time in milliseconds (minimum time between events for the same file)
    pub debounce_ms: u64,

    /// Polling interval in milliseconds (for polling-based watching)
    /// Recommended: 3000-5000ms for balanced performance and low idle CPU usage
    /// - Lower values (1000-2000ms): More responsive but higher CPU usage
    /// - Higher values (5000-10000ms): Lowest CPU usage but less responsive
    /// Default: 3000ms (optimized for low idle CPU usage)
    pub polling_interval_ms: u64,

    /// Minimum polling interval in milliseconds (safety bound)
    /// Prevents overly aggressive polling that wastes CPU
    pub min_polling_interval_ms: u64,

    /// Maximum polling interval in milliseconds (safety bound)
    /// Prevents overly slow polling that misses rapid changes
    pub max_polling_interval_ms: u64,

    /// Maximum number of events to queue before dropping
    pub max_queue_size: usize,

    /// Priority for tasks generated from file watching
    pub task_priority: TaskPriority,

    /// Collection name for processed documents
    pub default_collection: String,

    /// Whether to process existing files on startup
    pub process_existing: bool,

    /// File size limit in bytes (files larger than this are ignored)
    pub max_file_size: Option<u64>,

    /// Whether to use polling mode (useful for network drives)
    pub use_polling: bool,

    /// Batch processing settings
    pub batch_processing: BatchConfig,

    /// Maximum number of events to store in debouncer (memory limit)
    pub max_debouncer_capacity: usize,

    /// Maximum total events to store in batcher (memory limit)
    pub max_batcher_capacity: usize,

    /// Telemetry configuration
    pub telemetry: TelemetryConfig,
}

/// Configuration for telemetry collection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryConfig {
    /// Enable telemetry collection
    pub enabled: bool,

    /// Number of historical snapshots to retain
    pub history_retention: usize,

    /// Collection interval in seconds
    pub collection_interval_secs: u64,

    /// Individual metric toggles
    pub cpu_usage: bool,
    pub memory_usage: bool,
    pub latency: bool,
    pub queue_depth: bool,
    pub throughput: bool,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            history_retention: 120,
            collection_interval_secs: 60,
            cpu_usage: true,
            memory_usage: true,
            latency: true,
            queue_depth: true,
            throughput: true,
        }
    }
}

/// Configuration for batch processing of file events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    /// Enable batch processing
    pub enabled: bool,

    /// Maximum batch size
    pub max_batch_size: usize,

    /// Maximum time to wait for batch to fill (in milliseconds)
    pub max_batch_wait_ms: u64,

    /// Whether to group batches by file type
    pub group_by_type: bool,
}

impl WatcherConfig {
    /// Validate and clamp polling interval to safe bounds
    ///
    /// Ensures polling_interval_ms is within [min_polling_interval_ms, max_polling_interval_ms]
    /// to prevent CPU waste (too fast) or missing changes (too slow)
    pub fn validate_polling_interval(&mut self) {
        if self.polling_interval_ms < self.min_polling_interval_ms {
            tracing::warn!(
                "Polling interval {}ms is below minimum {}ms, clamping to minimum",
                self.polling_interval_ms,
                self.min_polling_interval_ms
            );
            self.polling_interval_ms = self.min_polling_interval_ms;
        }

        if self.polling_interval_ms > self.max_polling_interval_ms {
            tracing::warn!(
                "Polling interval {}ms exceeds maximum {}ms, clamping to maximum",
                self.polling_interval_ms,
                self.max_polling_interval_ms
            );
            self.polling_interval_ms = self.max_polling_interval_ms;
        }
    }
}

impl Default for WatcherConfig {
    fn default() -> Self {
        Self {
            include_patterns: vec![
                "*.txt".to_string(),
                "*.md".to_string(),
                "*.pdf".to_string(),
                "*.epub".to_string(),
                "*.docx".to_string(),
                "*.py".to_string(),
                "*.rs".to_string(),
                "*.js".to_string(),
                "*.ts".to_string(),
                "*.json".to_string(),
                "*.yaml".to_string(),
                "*.yml".to_string(),
                "*.toml".to_string(),
            ],
            exclude_patterns: vec![
                "*.tmp".to_string(),
                "*.swp".to_string(),
                "*.bak".to_string(),
                "*~".to_string(),
                ".git/**".to_string(),
                ".svn/**".to_string(),
                "node_modules/**".to_string(),
                "target/**".to_string(),
                "__pycache__/**".to_string(),
                ".pytest_cache/**".to_string(),
                ".DS_Store".to_string(),
                "Thumbs.db".to_string(),
            ],
            recursive: true,
            max_depth: -1,
            debounce_ms: 1000, // 1 second debounce
            polling_interval_ms: 3000, // 3 second polling (optimized for low idle CPU usage)
            min_polling_interval_ms: 100, // 100ms minimum (prevents CPU waste)
            max_polling_interval_ms: 60000, // 60 seconds maximum (prevents missing changes)
            max_queue_size: 10000,
            task_priority: TaskPriority::BackgroundWatching,
            default_collection: "documents".to_string(),
            process_existing: false,
            max_file_size: Some(100 * 1024 * 1024), // 100MB limit
            use_polling: false,
            batch_processing: BatchConfig {
                enabled: true,
                max_batch_size: 10,
                max_batch_wait_ms: 5000, // 5 seconds
                group_by_type: true,
            },
            max_debouncer_capacity: 10_000, // 10K events max in debouncer
            max_batcher_capacity: 5_000,    // 5K events max in batcher
            telemetry: TelemetryConfig::default(),
        }
    }
}
