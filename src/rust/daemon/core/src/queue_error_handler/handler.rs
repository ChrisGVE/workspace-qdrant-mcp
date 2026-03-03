// Comprehensive error handler for queue operations

use std::collections::HashMap;
use tracing::{error, info, warn};
use serde::Serialize;

use crate::queue_operations::QueueManager;

use super::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
use super::error_types::{ErrorCategory, ErrorType};
use super::retry::RetryConfig;

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
    circuit_breakers: HashMap<String, CircuitBreaker>,
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
        self.retry_config.calculate_delay(retry_count, error_type)
    }

    /// Determine if error should be retried
    pub fn should_retry(&self, error_type: ErrorType, retry_count: i32) -> bool {
        self.retry_config.should_retry(error_type, retry_count)
    }

    /// Check circuit breaker state for a collection.
    /// Returns `(can_proceed, reason_str)`.
    pub fn check_circuit_breaker(&mut self, collection: &str) -> (bool, &'static str) {
        self.get_or_create_breaker(collection).check()
    }

    /// Record a failure in the circuit breaker for a collection
    pub fn record_circuit_breaker_failure(&mut self, collection: &str) {
        let opened = self.get_or_create_breaker(collection).record_failure();
        if opened {
            self.metrics.circuit_breaker_opens += 1;
        }
    }

    /// Record a success in the circuit breaker for a collection
    pub fn record_circuit_breaker_success(&mut self, collection: &str) {
        self.get_or_create_breaker(collection).record_success();
    }

    /// Handle error with classification, retry logic, and circuit breaker
    pub async fn handle_error(
        &mut self,
        file_path: &str,
        error_message: &str,
        collection: &str,
        context: Option<&HashMap<String, serde_json::Value>>,
    ) -> (bool, ErrorType) {
        let error_type = self.classify_and_update_metrics(error_message);

        if !self.check_and_handle_circuit_breaker(file_path, error_type, error_message, collection).await {
            return (false, error_type);
        }

        let retry_count = 0; // placeholder, should be fetched from queue
        if !self.should_retry(error_type, retry_count) {
            info!(
                "Error {} for {} is not retryable, moving to dead letter queue",
                error_type.as_str(),
                file_path
            );
            self.move_to_dead_letter_queue(file_path, error_type, error_message, context).await;
            self.record_circuit_breaker_failure(collection);
            return (false, error_type);
        }

        self.record_error_and_retry(file_path, error_type, error_message, collection, context, retry_count).await
    }

    fn classify_and_update_metrics(&mut self, error_message: &str) -> ErrorType {
        let error_type = self.classify_error(error_message);
        self.metrics.total_errors += 1;
        match error_type.category() {
            ErrorCategory::Transient => self.metrics.transient_errors += 1,
            ErrorCategory::Permanent => self.metrics.permanent_errors += 1,
            ErrorCategory::RateLimit => self.metrics.rate_limit_errors += 1,
            ErrorCategory::Resource => self.metrics.resource_errors += 1,
        }
        error_type
    }

    async fn check_and_handle_circuit_breaker(
        &mut self,
        file_path: &str,
        error_type: ErrorType,
        error_message: &str,
        collection: &str,
    ) -> bool {
        let (can_proceed, _) = self.check_circuit_breaker(collection);
        if !can_proceed {
            warn!("Circuit breaker open for {}, moving to dead letter queue", collection);
            self.move_to_dead_letter_queue(
                file_path,
                error_type,
                error_message,
                Some(&HashMap::from([(
                    "circuit_breaker".to_string(),
                    serde_json::json!("open"),
                )])),
            )
            .await;
        }
        can_proceed
    }

    async fn record_error_and_retry(
        &mut self,
        file_path: &str,
        error_type: ErrorType,
        error_message: &str,
        collection: &str,
        context: Option<&HashMap<String, serde_json::Value>>,
        retry_count: i32,
    ) -> (bool, ErrorType) {
        let max_retries = if error_type.max_retries() > 0 {
            error_type.max_retries()
        } else {
            self.retry_config.default_max_retries
        };

        let result = self
            .queue_manager
            .mark_error(file_path, error_type.as_str(), error_message, context, max_retries)
            .await;

        match result {
            Ok((should_retry_result, _error_message_id)) => {
                if should_retry_result {
                    self.metrics.retry_count += 1;
                    let delay = self.calculate_retry_delay(retry_count, error_type);
                    info!(
                        "Retrying {} after {:.2}s (attempt {}/{})",
                        file_path, delay, retry_count + 1, max_retries
                    );
                    (true, error_type)
                } else {
                    warn!("Max retries exceeded for {}, moving to dead letter queue", file_path);
                    self.move_to_dead_letter_queue(file_path, error_type, error_message, context).await;
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

    /// Get error handling metrics
    pub fn get_metrics(&self) -> HashMap<String, serde_json::Value> {
        let mut metrics = HashMap::new();
        metrics.insert(
            "total_errors".to_string(),
            serde_json::json!(self.metrics.total_errors),
        );
        metrics.insert(
            "transient_errors".to_string(),
            serde_json::json!(self.metrics.transient_errors),
        );
        metrics.insert(
            "permanent_errors".to_string(),
            serde_json::json!(self.metrics.permanent_errors),
        );
        metrics.insert(
            "rate_limit_errors".to_string(),
            serde_json::json!(self.metrics.rate_limit_errors),
        );
        metrics.insert(
            "resource_errors".to_string(),
            serde_json::json!(self.metrics.resource_errors),
        );
        metrics.insert(
            "retry_count".to_string(),
            serde_json::json!(self.metrics.retry_count),
        );
        metrics.insert(
            "circuit_breaker_opens".to_string(),
            serde_json::json!(self.metrics.circuit_breaker_opens),
        );
        metrics.insert(
            "dead_letter_items".to_string(),
            serde_json::json!(self.metrics.dead_letter_items),
        );
        metrics.insert(
            "successful_retries".to_string(),
            serde_json::json!(self.metrics.successful_retries),
        );
        metrics.insert(
            "failed_retries".to_string(),
            serde_json::json!(self.metrics.failed_retries),
        );

        let active_breakers: HashMap<String, &'static str> = self
            .circuit_breakers
            .iter()
            .filter(|(_, breaker)| !breaker.is_closed())
            .map(|(name, breaker)| (name.clone(), breaker.state_str()))
            .collect();
        metrics.insert(
            "active_circuit_breakers".to_string(),
            serde_json::json!(active_breakers),
        );

        metrics
    }

    /// Reset error metrics
    pub fn reset_metrics(&mut self) {
        self.metrics = ErrorMetrics::default();
        info!("Error metrics reset");
    }

    // --- private helpers ---

    fn get_or_create_breaker(&mut self, collection: &str) -> &mut CircuitBreaker {
        let config = self.circuit_breaker_config.clone();
        self.circuit_breakers
            .entry(collection.to_string())
            .or_insert_with(|| CircuitBreaker::new(collection, config))
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
}
