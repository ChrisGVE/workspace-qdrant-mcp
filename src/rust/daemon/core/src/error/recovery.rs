//! Error recovery strategies and retry logic.

use std::time::Duration;

use tracing::warn;

use super::circuit_breaker::CircuitBreaker;
use super::monitor::{ErrorMonitor, ErrorStats};
use super::{CircuitBreakerStatus, WorkspaceError};

/// Error recovery strategies
#[derive(Debug, Clone)]
pub struct ErrorRecoveryStrategy {
    pub max_retries: u32,
    pub base_delay: Duration,
    pub max_delay: Duration,
    pub exponential_backoff: bool,
    pub circuit_breaker_threshold: Option<u32>,
}

impl Default for ErrorRecoveryStrategy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
            exponential_backoff: true,
            circuit_breaker_threshold: Some(5),
        }
    }
}

impl ErrorRecoveryStrategy {
    /// Create a recovery strategy for network operations
    pub fn network() -> Self {
        Self {
            max_retries: 5,
            base_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(10),
            exponential_backoff: true,
            circuit_breaker_threshold: Some(3),
        }
    }

    /// Create a recovery strategy for database operations
    pub fn database() -> Self {
        Self {
            max_retries: 3,
            base_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(30),
            exponential_backoff: true,
            circuit_breaker_threshold: Some(5),
        }
    }

    /// Create a recovery strategy for file operations
    pub fn file_operations() -> Self {
        Self {
            max_retries: 2,
            base_delay: Duration::from_millis(50),
            max_delay: Duration::from_secs(1),
            exponential_backoff: false,
            circuit_breaker_threshold: None,
        }
    }

    /// Create a recovery strategy for embedding generation
    pub fn embedding() -> Self {
        Self {
            max_retries: 3,
            base_delay: Duration::from_millis(1000),
            max_delay: Duration::from_secs(60),
            exponential_backoff: true,
            circuit_breaker_threshold: Some(10),
        }
    }

    /// Calculate delay for given attempt
    pub fn calculate_delay(&self, attempt: u32) -> Duration {
        if !self.exponential_backoff {
            return self.base_delay;
        }

        let delay = self.base_delay.as_millis() as u64 * (2_u64.pow(attempt.saturating_sub(1)));
        Duration::from_millis(delay.min(self.max_delay.as_millis() as u64))
    }
}

/// Error recovery utility with built-in retry logic
pub struct ErrorRecovery {
    monitor: Box<dyn ErrorMonitor>,
    circuit_breakers: std::collections::HashMap<String, CircuitBreaker>,
}

impl ErrorRecovery {
    /// Create a new error recovery system
    pub fn new(monitor: Box<dyn ErrorMonitor>) -> Self {
        Self {
            monitor,
            circuit_breakers: std::collections::HashMap::new(),
        }
    }

    /// Execute an operation with automatic retry and circuit breaker protection
    pub async fn execute_with_retry<F, T, E>(
        &mut self,
        operation_name: &str,
        operation: F,
        strategy: ErrorRecoveryStrategy,
    ) -> std::result::Result<T, WorkspaceError>
    where
        F: Fn() -> std::pin::Pin<
            Box<dyn std::future::Future<Output = std::result::Result<T, E>> + Send + 'static>,
        >,
        E: std::error::Error + Send + Sync + 'static,
    {
        let mut attempt = 1;

        loop {
            // Get or create circuit breaker
            if !self.circuit_breakers.contains_key(operation_name) {
                if let Some(threshold) = strategy.circuit_breaker_threshold {
                    self.circuit_breakers.insert(
                        operation_name.to_string(),
                        CircuitBreaker::new(operation_name, threshold),
                    );
                }
            }

            let result =
                if let Some(circuit_breaker) = self.circuit_breakers.get_mut(operation_name) {
                    circuit_breaker.execute(operation()).await
                } else {
                    match operation().await {
                        Ok(value) => Ok(value),
                        Err(error) => Err(WorkspaceError::Internal {
                            message: format!("Operation failed: {}", error),
                            component: operation_name.to_string(),
                            source: Some(Box::new(error)),
                        }),
                    }
                };

            match result {
                Ok(value) => {
                    if attempt > 1 {
                        self.monitor.report_recovery(operation_name, attempt);
                    }
                    return Ok(value);
                }
                Err(error) => {
                    self.monitor.report_error(&error, Some(operation_name));

                    if !error.is_retryable() || attempt >= strategy.max_retries {
                        return Err(error);
                    }

                    let delay = strategy.calculate_delay(attempt);
                    warn!(
                        operation = operation_name,
                        attempt = attempt,
                        max_attempts = strategy.max_retries,
                        delay_ms = delay.as_millis(),
                        "Retrying operation after error: {}",
                        error
                    );

                    tokio::time::sleep(delay).await;
                    attempt += 1;
                }
            }
        }
    }

    /// Get circuit breaker status for monitoring
    pub fn get_circuit_breaker_status(&self, name: &str) -> Option<CircuitBreakerStatus> {
        self.circuit_breakers.get(name).map(|cb| cb.status())
    }

    /// Get all circuit breaker statuses
    pub fn get_all_circuit_breaker_statuses(&self) -> Vec<CircuitBreakerStatus> {
        self.circuit_breakers
            .values()
            .map(|cb| cb.status())
            .collect()
    }

    /// Get error statistics
    pub fn get_error_stats(&self) -> ErrorStats {
        self.monitor.get_error_stats()
    }
}

impl Default for ErrorRecovery {
    fn default() -> Self {
        Self::new(Box::new(super::DefaultErrorMonitor::new()))
    }
}
