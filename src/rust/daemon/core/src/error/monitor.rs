//! Error monitoring and alerting.

use tracing::{error, info, warn};

use super::{ErrorSeverity, WorkspaceError};

/// Error monitoring and alerting hooks
pub trait ErrorMonitor: Send + Sync {
    /// Report an error for monitoring
    fn report_error(&self, error: &WorkspaceError, context: Option<&str>);

    /// Report error recovery
    fn report_recovery(&self, error_category: &str, attempt: u32);

    /// Report circuit breaker state change
    fn report_circuit_breaker_state(&self, name: &str, state: &str);

    /// Get error statistics
    fn get_error_stats(&self) -> ErrorStats;
}

/// Error statistics for monitoring
#[derive(Debug, Default, Clone)]
pub struct ErrorStats {
    pub total_errors: u64,
    pub errors_by_category: std::collections::HashMap<String, u64>,
    pub retryable_errors: u64,
    pub non_retryable_errors: u64,
    pub recovery_successes: u64,
    pub circuit_breaker_opens: u64,
}

/// Default error monitor implementation
pub struct DefaultErrorMonitor {
    stats: std::sync::Arc<tokio::sync::Mutex<ErrorStats>>,
}

impl DefaultErrorMonitor {
    pub fn new() -> Self {
        Self {
            stats: std::sync::Arc::new(tokio::sync::Mutex::new(ErrorStats::default())),
        }
    }
}

impl ErrorMonitor for DefaultErrorMonitor {
    fn report_error(&self, error: &WorkspaceError, context: Option<&str>) {
        let category = error.category();
        let severity = error.severity();
        let is_retryable = error.is_retryable();

        match severity {
            ErrorSeverity::Low => info!(
                error_category = category,
                retryable = is_retryable,
                context = context.unwrap_or("none"),
                "Error reported: {}", error
            ),
            ErrorSeverity::Medium => warn!(
                error_category = category,
                retryable = is_retryable,
                context = context.unwrap_or("none"),
                "Error reported: {}", error
            ),
            ErrorSeverity::High | ErrorSeverity::Critical => error!(
                error_category = category,
                retryable = is_retryable,
                context = context.unwrap_or("none"),
                "Error reported: {}", error
            ),
        }

        // Update statistics (in production, this would be async)
        if let Ok(mut stats) = self.stats.try_lock() {
            stats.total_errors += 1;
            *stats.errors_by_category.entry(category.to_string()).or_insert(0) += 1;
            if is_retryable {
                stats.retryable_errors += 1;
            } else {
                stats.non_retryable_errors += 1;
            }
        }
    }

    fn report_recovery(&self, error_category: &str, attempt: u32) {
        info!(
            error_category = error_category,
            attempt = attempt,
            "Error recovery succeeded"
        );

        if let Ok(mut stats) = self.stats.try_lock() {
            stats.recovery_successes += 1;
        }
    }

    fn report_circuit_breaker_state(&self, name: &str, state: &str) {
        match state {
            "open" => {
                warn!(circuit_breaker = name, "Circuit breaker opened");
                if let Ok(mut stats) = self.stats.try_lock() {
                    stats.circuit_breaker_opens += 1;
                }
            }
            "closed" => info!(circuit_breaker = name, "Circuit breaker closed"),
            "half-open" => info!(circuit_breaker = name, "Circuit breaker half-open"),
            _ => warn!(circuit_breaker = name, state = state, "Unknown circuit breaker state"),
        }
    }

    fn get_error_stats(&self) -> ErrorStats {
        if let Ok(stats) = self.stats.try_lock() {
            stats.clone()
        } else {
            ErrorStats::default()
        }
    }
}

impl Default for DefaultErrorMonitor {
    fn default() -> Self {
        Self::new()
    }
}
