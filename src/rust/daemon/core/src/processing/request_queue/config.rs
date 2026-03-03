//! Queue configuration types and builder

use serde::{Deserialize, Serialize};

/// Request queuing configuration and limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueConfig {
    /// Maximum number of queued requests per priority level
    pub max_queued_per_priority: usize,
    /// Default timeout for queued requests in milliseconds
    pub default_queue_timeout_ms: u64,
    /// Enable request deduplication based on content hash
    pub enable_deduplication: bool,
    /// Maximum time to wait for queue space in milliseconds
    pub queue_wait_timeout_ms: u64,
    /// Enable priority boost for aged requests
    pub enable_priority_boost: bool,
    /// Age threshold for priority boost in milliseconds
    pub priority_boost_age_ms: u64,
    /// Enable rate limiting for task submission
    pub enable_rate_limiting: bool,
    /// Maximum tasks per second (None = unlimited)
    pub max_tasks_per_second: Option<u64>,
    /// Enable backpressure signaling
    pub enable_backpressure: bool,
    /// Queue utilization threshold for backpressure warning (0.0-1.0, default: 0.8)
    pub backpressure_threshold: f64,
}

impl Default for QueueConfig {
    fn default() -> Self {
        Self {
            max_queued_per_priority: 100,
            default_queue_timeout_ms: 30_000,
            enable_deduplication: true,
            queue_wait_timeout_ms: 5_000,
            enable_priority_boost: true,
            priority_boost_age_ms: 10_000,
            enable_rate_limiting: true,
            max_tasks_per_second: Some(100),
            enable_backpressure: true,
            backpressure_threshold: 0.8,
        }
    }
}

/// Configuration builder for creating optimized queue configurations
pub struct QueueConfigBuilder {
    config: QueueConfig,
}

impl QueueConfigBuilder {
    /// Start with default configuration
    pub fn new() -> Self {
        Self {
            config: QueueConfig::default(),
        }
    }

    /// Set maximum queued requests per priority level
    pub fn max_queued_per_priority(mut self, max: usize) -> Self {
        self.config.max_queued_per_priority = max;
        self
    }

    /// Set default queue timeout
    pub fn default_queue_timeout(mut self, timeout_ms: u64) -> Self {
        self.config.default_queue_timeout_ms = timeout_ms;
        self
    }

    /// Enable or disable deduplication
    pub fn deduplication(mut self, enable: bool) -> Self {
        self.config.enable_deduplication = enable;
        self
    }

    /// Set queue wait timeout
    pub fn queue_wait_timeout(mut self, timeout_ms: u64) -> Self {
        self.config.queue_wait_timeout_ms = timeout_ms;
        self
    }

    /// Enable or disable priority boosting
    pub fn priority_boost(mut self, enable: bool, age_threshold_ms: u64) -> Self {
        self.config.enable_priority_boost = enable;
        self.config.priority_boost_age_ms = age_threshold_ms;
        self
    }

    /// Build the configuration for MCP servers (low latency)
    pub fn for_mcp_server(mut self) -> Self {
        self.config.max_queued_per_priority = 50;
        self.config.default_queue_timeout_ms = 5_000;
        self.config.enable_deduplication = true;
        self.config.queue_wait_timeout_ms = 1_000;
        self.config.enable_priority_boost = true;
        self.config.priority_boost_age_ms = 2_000;
        self
    }

    /// Build the configuration for batch processing (high throughput)
    pub fn for_batch_processing(mut self) -> Self {
        self.config.max_queued_per_priority = 1000;
        self.config.default_queue_timeout_ms = 60_000;
        self.config.enable_deduplication = true;
        self.config.queue_wait_timeout_ms = 10_000;
        self.config.enable_priority_boost = false;
        self.config.priority_boost_age_ms = 30_000;
        self
    }

    /// Build the configuration for resource-constrained environments
    pub fn for_low_resource(mut self) -> Self {
        self.config.max_queued_per_priority = 10;
        self.config.default_queue_timeout_ms = 120_000;
        self.config.enable_deduplication = false;
        self.config.queue_wait_timeout_ms = 30_000;
        self.config.enable_priority_boost = false;
        self.config.priority_boost_age_ms = 60_000;
        self
    }

    /// Build the final configuration
    pub fn build(self) -> QueueConfig {
        self.config
    }
}

impl Default for QueueConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}
