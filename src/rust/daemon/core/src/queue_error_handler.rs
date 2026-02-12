// Queue Error Handling and Retry System
//
// Provides comprehensive error classification, retry strategies with exponential
// backoff, dead letter queue management, and circuit breaker pattern for queue operations.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use rand::Rng;
use serde::{Deserialize, Serialize};
use tracing::{debug, error, info, warn};

use crate::queue_operations::QueueManager;

/// Error category classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorCategory {
    /// Temporary errors, retry possible
    Transient,
    /// Permanent errors, no retry
    Permanent,
    /// Rate limiting, retry with backoff
    RateLimit,
    /// Resource exhaustion, retry with delay
    Resource,
}

/// Specific error types with categorization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorType {
    // Transient errors (retry with backoff)
    NetworkTimeout,
    ConnectionRefused,
    TemporaryFailure,
    DatabaseLocked,

    // Rate limit errors (retry with longer backoff)
    RateLimitExceeded,
    TooManyRequests,

    // Resource errors (retry with delay)
    OutOfMemory,
    DiskFull,
    QuotaExceeded,

    // Permanent errors (no retry)
    FileNotFound,
    InvalidFormat,
    PermissionDenied,
    InvalidConfiguration,
    ValidationError,
    MalformedData,
}

impl ErrorType {
    pub fn category(&self) -> ErrorCategory {
        match self {
            ErrorType::NetworkTimeout
            | ErrorType::ConnectionRefused
            | ErrorType::TemporaryFailure
            | ErrorType::DatabaseLocked => ErrorCategory::Transient,

            ErrorType::RateLimitExceeded | ErrorType::TooManyRequests => {
                ErrorCategory::RateLimit
            }

            ErrorType::OutOfMemory | ErrorType::DiskFull | ErrorType::QuotaExceeded => {
                ErrorCategory::Resource
            }

            ErrorType::FileNotFound
            | ErrorType::InvalidFormat
            | ErrorType::PermissionDenied
            | ErrorType::InvalidConfiguration
            | ErrorType::ValidationError
            | ErrorType::MalformedData => ErrorCategory::Permanent,
        }
    }

    pub fn max_retries(&self) -> i32 {
        match self {
            ErrorType::NetworkTimeout => 5,
            ErrorType::ConnectionRefused => 5,
            ErrorType::TemporaryFailure => 3,
            ErrorType::DatabaseLocked => 10,
            ErrorType::RateLimitExceeded => 10,
            ErrorType::TooManyRequests => 8,
            ErrorType::OutOfMemory => 3,
            ErrorType::DiskFull => 3,
            ErrorType::QuotaExceeded => 5,
            _ => 0, // Permanent errors
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            ErrorType::NetworkTimeout => "network_timeout",
            ErrorType::ConnectionRefused => "connection_refused",
            ErrorType::TemporaryFailure => "temporary_failure",
            ErrorType::DatabaseLocked => "database_locked",
            ErrorType::RateLimitExceeded => "rate_limit_exceeded",
            ErrorType::TooManyRequests => "too_many_requests",
            ErrorType::OutOfMemory => "out_of_memory",
            ErrorType::DiskFull => "disk_full",
            ErrorType::QuotaExceeded => "quota_exceeded",
            ErrorType::FileNotFound => "file_not_found",
            ErrorType::InvalidFormat => "invalid_format",
            ErrorType::PermissionDenied => "permission_denied",
            ErrorType::InvalidConfiguration => "invalid_configuration",
            ErrorType::ValidationError => "validation_error",
            ErrorType::MalformedData => "malformed_data",
        }
    }
}

/// Retry strategy configuration
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Base retry delay in seconds
    pub base_delay: f64,
    /// Maximum delay between retries
    pub max_delay: f64,
    /// Exponential backoff multiplier
    pub backoff_multiplier: f64,
    /// Jitter factor (0-1) to randomize delays
    pub jitter_factor: f64,
    /// Default max retries (when error type not specified)
    pub default_max_retries: i32,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            base_delay: 1.0,
            max_delay: 300.0,  // 5 minutes
            backoff_multiplier: 2.0,
            jitter_factor: 0.1,
            default_max_retries: 3,
        }
    }
}

/// Circuit breaker pattern configuration
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Number of failures before opening circuit
    pub failure_threshold: usize,
    /// Time window for counting failures (seconds)
    pub failure_window: u64,
    /// Time to wait before trying half-open state (seconds)
    pub recovery_timeout: u64,
    /// Success count needed in half-open state to close circuit
    pub success_threshold: usize,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            failure_window: 60,
            recovery_timeout: 300,
            success_threshold: 2,
        }
    }
}

/// Circuit breaker state
#[derive(Debug, Clone)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

impl CircuitState {
    pub fn as_str(&self) -> &'static str {
        match self {
            CircuitState::Closed => "closed",
            CircuitState::Open => "open",
            CircuitState::HalfOpen => "half-open",
        }
    }
}

/// Circuit breaker state tracking
#[derive(Debug, Clone)]
pub struct CircuitBreakerState {
    pub state: CircuitState,
    pub failure_count: usize,
    pub success_count: usize,
    pub last_failure_time: Option<u64>,
    pub opened_at: Option<u64>,
    pub failures: Vec<u64>,
}

impl Default for CircuitBreakerState {
    fn default() -> Self {
        Self {
            state: CircuitState::Closed,
            failure_count: 0,
            success_count: 0,
            last_failure_time: None,
            opened_at: None,
            failures: Vec::new(),
        }
    }
}

/// Error metrics for monitoring
#[derive(Debug, Clone, Default, Serialize)]
pub struct ErrorMetrics {
    pub total_errors: u64,
    pub transient_errors: u64,
    pub permanent_errors: u64,
    pub rate_limit_errors: u64,
    pub resource_errors: u64,
    pub retry_count: u64,
    pub circuit_breaker_opens: u64,
    pub dead_letter_items: u64,
    pub successful_retries: u64,
    pub failed_retries: u64,
}

/// Comprehensive error handler for queue operations
pub struct ErrorHandler {
    queue_manager: QueueManager,
    retry_config: RetryConfig,
    circuit_breaker_config: CircuitBreakerConfig,
    circuit_breakers: HashMap<String, CircuitBreakerState>,
    metrics: ErrorMetrics,
}

impl ErrorHandler {
    /// Create new error handler
    pub fn new(
        queue_manager: QueueManager,
        retry_config: Option<RetryConfig>,
        circuit_breaker_config: Option<CircuitBreakerConfig>,
    ) -> Self {
        Self {
            queue_manager,
            retry_config: retry_config.unwrap_or_default(),
            circuit_breaker_config: circuit_breaker_config.unwrap_or_default(),
            circuit_breakers: HashMap::new(),
            metrics: ErrorMetrics::default(),
        }
    }

    /// Classify error by analyzing error message
    pub fn classify_error(&self, error_message: &str) -> ErrorType {
        let error_lower = error_message.to_lowercase();

        if error_lower.contains("timeout") || error_lower.contains("timed out") {
            ErrorType::NetworkTimeout
        } else if error_lower.contains("connection refused") {
            ErrorType::ConnectionRefused
        } else if error_lower.contains("database is locked") {
            ErrorType::DatabaseLocked
        } else if error_lower.contains("rate limit") {
            ErrorType::RateLimitExceeded
        } else if error_lower.contains("too many requests") || error_lower.contains("429") {
            ErrorType::TooManyRequests
        } else if error_lower.contains("out of memory") {
            ErrorType::OutOfMemory
        } else if error_lower.contains("disk full") || error_lower.contains("no space left") {
            ErrorType::DiskFull
        } else if error_lower.contains("quota exceeded") {
            ErrorType::QuotaExceeded
        } else if error_lower.contains("file not found") || error_lower.contains("no such file") {
            ErrorType::FileNotFound
        } else if error_lower.contains("invalid format") {
            ErrorType::InvalidFormat
        } else if error_lower.contains("permission denied")
            || error_lower.contains("access denied")
        {
            ErrorType::PermissionDenied
        } else if error_lower.contains("validation error") {
            ErrorType::ValidationError
        } else if error_lower.contains("malformed") {
            ErrorType::MalformedData
        } else {
            warn!("Unknown error type, classifying as temporary: {}", error_message);
            ErrorType::TemporaryFailure
        }
    }

    /// Calculate retry delay with exponential backoff and jitter
    pub fn calculate_retry_delay(&self, retry_count: i32, error_type: ErrorType) -> f64 {
        // Base delay with exponential backoff
        let mut delay = self.retry_config.base_delay
            * self.retry_config.backoff_multiplier.powi(retry_count);

        // Apply different multipliers based on error category
        match error_type.category() {
            ErrorCategory::RateLimit => delay *= 3.0,  // Longer delays for rate limits
            ErrorCategory::Resource => delay *= 2.0,   // Moderate delays for resource issues
            _ => {}
        }

        // Cap at max delay
        delay = delay.min(self.retry_config.max_delay);

        // Add jitter to prevent thundering herd
        let mut rng = rand::thread_rng();
        let jitter = delay * self.retry_config.jitter_factor * rng.gen::<f64>();
        delay += jitter;

        debug!(
            "Calculated retry delay for {}: {:.2}s (attempt {})",
            error_type.as_str(),
            delay,
            retry_count + 1
        );

        delay
    }

    /// Determine if error should be retried
    pub fn should_retry(&self, error_type: ErrorType, retry_count: i32) -> bool {
        // Permanent errors never retry
        if error_type.category() == ErrorCategory::Permanent {
            return false;
        }

        // Check if max retries exceeded
        let max_retries = if error_type.max_retries() > 0 {
            error_type.max_retries()
        } else {
            self.retry_config.default_max_retries
        };

        retry_count < max_retries
    }

    /// Get or create circuit breaker state for collection
    pub fn get_circuit_breaker(&mut self, collection: &str) -> &mut CircuitBreakerState {
        self.circuit_breakers
            .entry(collection.to_string())
            .or_default()
    }

    /// Check circuit breaker state
    pub fn check_circuit_breaker(&mut self, collection: &str) -> (bool, &str) {
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let recovery_timeout = self.circuit_breaker_config.recovery_timeout;

        let breaker = self.get_circuit_breaker(collection);

        match breaker.state {
            CircuitState::Open => {
                if let Some(opened_at) = breaker.opened_at {
                    if (current_time - opened_at) > recovery_timeout {
                        // Try half-open state
                        breaker.state = CircuitState::HalfOpen;
                        breaker.success_count = 0;
                        info!("Circuit breaker {} entering half-open state", collection);
                        return (true, "half-open");
                    }
                }
                (false, "circuit_open")
            }
            _ => (true, breaker.state.as_str()),
        }
    }

    /// Record failure in circuit breaker
    pub fn record_circuit_breaker_failure(&mut self, collection: &str) {
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let failure_window = self.circuit_breaker_config.failure_window;
        let failure_threshold = self.circuit_breaker_config.failure_threshold;

        let breaker = self.get_circuit_breaker(collection);
        breaker.failure_count += 1;
        breaker.last_failure_time = Some(current_time);
        breaker.failures.push(current_time);

        // Remove old failures outside time window
        let window_start = current_time - failure_window;
        breaker.failures.retain(|&f| f > window_start);

        // Check if should open circuit
        let failure_count = breaker.failures.len();
        match breaker.state {
            CircuitState::Closed => {
                if failure_count >= failure_threshold {
                    breaker.state = CircuitState::Open;
                    breaker.opened_at = Some(current_time);
                    self.metrics.circuit_breaker_opens += 1;
                    warn!(
                        "Circuit breaker opened for {} after {} failures",
                        collection,
                        failure_count
                    );
                }
            }
            CircuitState::HalfOpen => {
                // Failed in half-open, back to open
                breaker.state = CircuitState::Open;
                breaker.opened_at = Some(current_time);
                breaker.success_count = 0;
                warn!("Circuit breaker {} reopened after half-open failure", collection);
            }
            _ => {}
        }
    }

    /// Record success in circuit breaker
    pub fn record_circuit_breaker_success(&mut self, collection: &str) {
        let success_threshold = self.circuit_breaker_config.success_threshold;
        let breaker = self.get_circuit_breaker(collection);

        if matches!(breaker.state, CircuitState::HalfOpen) {
            breaker.success_count += 1;

            if breaker.success_count >= success_threshold {
                breaker.state = CircuitState::Closed;
                breaker.failure_count = 0;
                breaker.failures.clear();
                info!("Circuit breaker {} closed after successful recovery", collection);
            }
        }
    }

    /// Handle error with classification, retry logic, and circuit breaker
    pub async fn handle_error(
        &mut self,
        file_path: &str,
        error_message: &str,
        collection: &str,
        context: Option<&HashMap<String, serde_json::Value>>,
    ) -> (bool, ErrorType) {
        // Classify error
        let error_type = self.classify_error(error_message);

        // Update metrics
        self.metrics.total_errors += 1;
        match error_type.category() {
            ErrorCategory::Transient => self.metrics.transient_errors += 1,
            ErrorCategory::Permanent => self.metrics.permanent_errors += 1,
            ErrorCategory::RateLimit => self.metrics.rate_limit_errors += 1,
            ErrorCategory::Resource => self.metrics.resource_errors += 1,
        }

        // Check circuit breaker
        let (can_proceed, _breaker_reason) = self.check_circuit_breaker(collection);

        if !can_proceed {
            warn!(
                "Circuit breaker open for {}, moving to dead letter queue",
                collection
            );
            self.move_to_dead_letter_queue(
                file_path,
                error_type,
                error_message,
                Some(&HashMap::from([("circuit_breaker".to_string(), serde_json::json!("open"))])),
            )
            .await;
            return (false, error_type);
        }

        // Get current retry count (placeholder, should be fetched from queue)
        let retry_count = 0;

        // Check if should retry
        if !self.should_retry(error_type, retry_count) {
            info!(
                "Error {} for {} is not retryable, moving to dead letter queue",
                error_type.as_str(),
                file_path
            );
            self.move_to_dead_letter_queue(file_path, error_type, error_message, context)
                .await;
            self.record_circuit_breaker_failure(collection);
            return (false, error_type);
        }

        // Record error and determine retry delay
        let max_retries = if error_type.max_retries() > 0 {
            error_type.max_retries()
        } else {
            self.retry_config.default_max_retries
        };

        let result = self
            .queue_manager
            .mark_error(
                file_path,
                error_type.as_str(),
                error_message,
                context,
                max_retries,
            )
            .await;

        match result {
            Ok((should_retry_result, _error_message_id)) => {
                if should_retry_result {
                    self.metrics.retry_count += 1;

                    let delay = self.calculate_retry_delay(retry_count, error_type);
                    info!(
                        "Retrying {} after {:.2}s (attempt {}/{})",
                        file_path,
                        delay,
                        retry_count + 1,
                        max_retries
                    );

                    (true, error_type)
                } else {
                    warn!(
                        "Max retries exceeded for {}, moving to dead letter queue",
                        file_path
                    );
                    self.move_to_dead_letter_queue(file_path, error_type, error_message, context)
                        .await;
                    self.record_circuit_breaker_failure(collection);
                    (false, error_type)
                }
            }
            Err(e) => {
                error!("Failed to record error: {}", e);
                (false, error_type)
            }
        }
    }

    /// Move permanently failed item to dead letter queue
    async fn move_to_dead_letter_queue(
        &mut self,
        file_path: &str,
        error_type: ErrorType,
        error_message: &str,
        context: Option<&HashMap<String, serde_json::Value>>,
    ) {
        let mut error_details = context.cloned().unwrap_or_default();
        error_details.insert(
            "dead_letter_reason".to_string(),
            serde_json::json!(error_type.as_str()),
        );
        error_details.insert(
            "moved_to_dlq_at".to_string(),
            serde_json::json!(wqm_common::timestamps::now_utc()),
        );

        // Record in messages table
        let _ = self
            .queue_manager
            .mark_error(
                file_path,
                &format!("DEAD_LETTER_{}", error_type.as_str()),
                error_message,
                Some(&error_details),
                0, // Force removal from queue
            )
            .await;

        self.metrics.dead_letter_items += 1;

        error!(
            "Moved {} to dead letter queue: {}",
            file_path,
            error_type.as_str()
        );
    }

    /// Get error handling metrics
    pub fn get_metrics(&self) -> HashMap<String, serde_json::Value> {
        let mut metrics = HashMap::new();
        metrics.insert("total_errors".to_string(), serde_json::json!(self.metrics.total_errors));
        metrics.insert("transient_errors".to_string(), serde_json::json!(self.metrics.transient_errors));
        metrics.insert("permanent_errors".to_string(), serde_json::json!(self.metrics.permanent_errors));
        metrics.insert("rate_limit_errors".to_string(), serde_json::json!(self.metrics.rate_limit_errors));
        metrics.insert("resource_errors".to_string(), serde_json::json!(self.metrics.resource_errors));
        metrics.insert("retry_count".to_string(), serde_json::json!(self.metrics.retry_count));
        metrics.insert("circuit_breaker_opens".to_string(), serde_json::json!(self.metrics.circuit_breaker_opens));
        metrics.insert("dead_letter_items".to_string(), serde_json::json!(self.metrics.dead_letter_items));
        metrics.insert("successful_retries".to_string(), serde_json::json!(self.metrics.successful_retries));
        metrics.insert("failed_retries".to_string(), serde_json::json!(self.metrics.failed_retries));

        let active_breakers: HashMap<String, &str> = self
            .circuit_breakers
            .iter()
            .filter(|(_, state)| !matches!(state.state, CircuitState::Closed))
            .map(|(name, state)| (name.clone(), state.state.as_str()))
            .collect();
        metrics.insert("active_circuit_breakers".to_string(), serde_json::json!(active_breakers));

        metrics
    }

    /// Reset error metrics
    pub fn reset_metrics(&mut self) {
        self.metrics = ErrorMetrics::default();
        info!("Error metrics reset");
    }
}
