//! Retry logic with exponential backoff and jitter for gRPC operations
//!
//! This module provides configurable retry mechanisms for handling transient
//! failures in gRPC communications.

use crate::error::{DaemonError, DaemonResult};
use rand::Rng;
use std::future::Future;
use std::time::Duration;
use tokio::time::sleep;
use tracing::{debug, warn};

/// Configuration for retry behavior
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts
    pub max_attempts: u32,
    /// Initial delay before first retry
    pub initial_delay: Duration,
    /// Maximum delay between retries
    pub max_delay: Duration,
    /// Multiplier for exponential backoff
    pub backoff_multiplier: f64,
    /// Maximum jitter factor (0.0 to 1.0)
    pub jitter_factor: f64,
    /// Whether to enable jitter
    pub enable_jitter: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            jitter_factor: 0.1,
            enable_jitter: true,
        }
    }
}

impl RetryConfig {
    /// Create a new retry configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum retry attempts
    pub fn max_attempts(mut self, attempts: u32) -> Self {
        self.max_attempts = attempts;
        self
    }

    /// Set initial delay
    pub fn initial_delay(mut self, delay: Duration) -> Self {
        self.initial_delay = delay;
        self
    }

    /// Set maximum delay
    pub fn max_delay(mut self, delay: Duration) -> Self {
        self.max_delay = delay;
        self
    }

    /// Set backoff multiplier
    pub fn backoff_multiplier(mut self, multiplier: f64) -> Self {
        self.backoff_multiplier = multiplier;
        self
    }

    /// Set jitter factor
    pub fn jitter_factor(mut self, factor: f64) -> Self {
        self.jitter_factor = factor.clamp(0.0, 1.0);
        self
    }

    /// Enable or disable jitter
    pub fn enable_jitter(mut self, enable: bool) -> Self {
        self.enable_jitter = enable;
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> DaemonResult<()> {
        if self.max_attempts == 0 {
            return Err(DaemonError::RetryConfigInvalid {
                message: "max_attempts must be greater than 0".to_string(),
            });
        }

        if self.initial_delay.is_zero() {
            return Err(DaemonError::RetryConfigInvalid {
                message: "initial_delay must be greater than 0".to_string(),
            });
        }

        if self.max_delay < self.initial_delay {
            return Err(DaemonError::RetryConfigInvalid {
                message: "max_delay must be greater than or equal to initial_delay".to_string(),
            });
        }

        if self.backoff_multiplier <= 1.0 {
            return Err(DaemonError::RetryConfigInvalid {
                message: "backoff_multiplier must be greater than 1.0".to_string(),
            });
        }

        if !(0.0..=1.0).contains(&self.jitter_factor) {
            return Err(DaemonError::RetryConfigInvalid {
                message: "jitter_factor must be between 0.0 and 1.0".to_string(),
            });
        }

        Ok(())
    }
}

/// Trait to determine if an error should trigger a retry
pub trait RetryPredicate {
    /// Returns true if the error should trigger a retry
    fn should_retry(&self, error: &DaemonError, attempt: u32) -> bool;
}

/// Default retry predicate for network and transient errors
#[derive(Debug, Clone)]
pub struct DefaultRetryPredicate;

impl RetryPredicate for DefaultRetryPredicate {
    fn should_retry(&self, error: &DaemonError, _attempt: u32) -> bool {
        match error {
            // Network errors that should be retried
            DaemonError::NetworkConnection { .. } |
            DaemonError::NetworkTimeout { .. } |
            DaemonError::NetworkUnavailable { .. } |
            DaemonError::DnsResolution { .. } |
            DaemonError::TlsHandshake { .. } => true,

            // Service health issues that should be retried
            DaemonError::ServiceHealthCheck { .. } |
            DaemonError::NoHealthyInstances { .. } => true,

            // HTTP errors that might be transient
            DaemonError::Http(ref http_err) => {
                if let Some(status) = http_err.status() {
                    // Retry on server errors and some client errors
                    matches!(status.as_u16(), 408 | 429 | 500..=599)
                } else {
                    // Retry on connection errors
                    true
                }
            },

            // Database connection issues
            DaemonError::Database(ref db_err) => {
                match db_err {
                    sqlx::Error::PoolClosed |
                    sqlx::Error::PoolTimedOut |
                    sqlx::Error::Io(_) => true,
                    _ => false,
                }
            },

            // I/O errors that might be transient
            DaemonError::Io(ref io_err) => {
                matches!(io_err.kind(),
                    std::io::ErrorKind::ConnectionRefused |
                    std::io::ErrorKind::ConnectionAborted |
                    std::io::ErrorKind::ConnectionReset |
                    std::io::ErrorKind::TimedOut |
                    std::io::ErrorKind::Interrupted
                )
            },

            // Circuit breaker in half-open state might be worth retrying
            DaemonError::CircuitBreakerHalfOpen { .. } => true,

            // Other errors should not be retried by default
            _ => false,
        }
    }
}

/// Retry strategy implementation
#[derive(Debug)]
pub struct RetryStrategy {
    config: RetryConfig,
    predicate: Box<dyn RetryPredicate + Send + Sync>,
}

impl RetryStrategy {
    /// Create a new retry strategy with default configuration
    pub fn new() -> Self {
        Self {
            config: RetryConfig::default(),
            predicate: Box::new(DefaultRetryPredicate),
        }
    }

    /// Create a retry strategy with custom configuration
    pub fn with_config(config: RetryConfig) -> DaemonResult<Self> {
        config.validate()?;
        Ok(Self {
            config,
            predicate: Box::new(DefaultRetryPredicate),
        })
    }

    /// Set a custom retry predicate
    pub fn with_predicate<P>(mut self, predicate: P) -> Self
    where
        P: RetryPredicate + Send + Sync + 'static,
    {
        self.predicate = Box::new(predicate);
        self
    }

    /// Execute an operation with retry logic
    pub async fn execute<F, Fut, T>(&self, operation: F) -> DaemonResult<T>
    where
        F: Fn() -> Fut,
        Fut: Future<Output = DaemonResult<T>>,
    {
        let mut last_error = None;
        let mut rng = rand::thread_rng();

        for attempt in 0..self.config.max_attempts {
            match operation().await {
                Ok(result) => {
                    if attempt > 0 {
                        debug!("Operation succeeded after {} retries", attempt);
                    }
                    return Ok(result);
                }
                Err(error) => {
                    last_error = Some(error.clone());

                    // Check if we should retry this error
                    if !self.predicate.should_retry(&error, attempt) {
                        debug!("Error not retryable: {}", error);
                        return Err(error);
                    }

                    // Don't sleep after the last attempt
                    if attempt + 1 < self.config.max_attempts {
                        let delay = self.calculate_delay(attempt, &mut rng)?;
                        warn!(
                            "Operation failed (attempt {}/{}), retrying in {:?}: {}",
                            attempt + 1,
                            self.config.max_attempts,
                            delay,
                            error
                        );
                        sleep(delay).await;
                    } else {
                        debug!("Maximum retry attempts reached");
                    }
                }
            }
        }

        // All retries exhausted
        let final_error = last_error.unwrap_or_else(|| DaemonError::Internal {
            message: "No error recorded during retry attempts".to_string(),
        });

        Err(DaemonError::RetryLimitExceeded {
            attempts: self.config.max_attempts,
        })
    }

    /// Calculate delay for the next retry attempt with exponential backoff and jitter
    fn calculate_delay(&self, attempt: u32, rng: &mut impl Rng) -> DaemonResult<Duration> {
        if attempt >= self.config.max_attempts {
            return Err(DaemonError::BackoffCalculation {
                message: format!("Attempt {} exceeds max attempts {}", attempt, self.config.max_attempts),
            });
        }

        // Calculate exponential backoff
        let base_delay = self.config.initial_delay.as_millis() as f64
            * self.config.backoff_multiplier.powi(attempt as i32);

        // Cap at max_delay
        let capped_delay = base_delay.min(self.config.max_delay.as_millis() as f64);

        // Apply jitter if enabled
        let final_delay = if self.config.enable_jitter {
            let jitter_range = capped_delay * self.config.jitter_factor;
            let jitter = rng.gen_range(-jitter_range..=jitter_range);
            (capped_delay + jitter).max(0.0)
        } else {
            capped_delay
        };

        Ok(Duration::from_millis(final_delay as u64))
    }

    /// Get the current configuration
    pub fn config(&self) -> &RetryConfig {
        &self.config
    }
}

impl Default for RetryStrategy {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use tokio::time::Instant;

    #[test]
    fn test_retry_config_validation() {
        // Valid config
        let config = RetryConfig::new()
            .max_attempts(3)
            .initial_delay(Duration::from_millis(100))
            .max_delay(Duration::from_secs(10))
            .backoff_multiplier(2.0)
            .jitter_factor(0.1);
        assert!(config.validate().is_ok());

        // Invalid max_attempts
        let config = RetryConfig::new().max_attempts(0);
        assert!(config.validate().is_err());

        // Invalid initial_delay
        let config = RetryConfig::new().initial_delay(Duration::ZERO);
        assert!(config.validate().is_err());

        // Invalid max_delay vs initial_delay
        let config = RetryConfig::new()
            .initial_delay(Duration::from_secs(10))
            .max_delay(Duration::from_millis(100));
        assert!(config.validate().is_err());

        // Invalid backoff_multiplier
        let config = RetryConfig::new().backoff_multiplier(0.5);
        assert!(config.validate().is_err());

        // Invalid jitter_factor
        let config = RetryConfig::new().jitter_factor(1.5);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_default_retry_predicate() {
        let predicate = DefaultRetryPredicate;

        // Should retry network errors
        assert!(predicate.should_retry(&DaemonError::NetworkConnection {
            message: "test".to_string()
        }, 0));

        assert!(predicate.should_retry(&DaemonError::NetworkTimeout {
            timeout_ms: 5000
        }, 0));

        // Should not retry invalid input
        assert!(!predicate.should_retry(&DaemonError::InvalidInput {
            message: "test".to_string()
        }, 0));

        // Should not retry not found
        assert!(!predicate.should_retry(&DaemonError::NotFound {
            resource: "test".to_string()
        }, 0));
    }

    #[tokio::test]
    async fn test_retry_strategy_success_immediate() {
        let strategy = RetryStrategy::new();
        let counter = Arc::new(Mutex::new(0));
        let counter_clone = counter.clone();

        let result = strategy.execute(|| {
            let counter = counter_clone.clone();
            async move {
                let mut count = counter.lock().unwrap();
                *count += 1;
                Ok::<i32, DaemonError>(42)
            }
        }).await;

        assert_eq!(result.unwrap(), 42);
        assert_eq!(*counter.lock().unwrap(), 1);
    }

    #[tokio::test]
    async fn test_retry_strategy_success_after_retries() {
        let config = RetryConfig::new()
            .max_attempts(3)
            .initial_delay(Duration::from_millis(10))
            .enable_jitter(false);

        let strategy = RetryStrategy::with_config(config).unwrap();
        let counter = Arc::new(Mutex::new(0));
        let counter_clone = counter.clone();

        let result = strategy.execute(|| {
            let counter = counter_clone.clone();
            async move {
                let mut count = counter.lock().unwrap();
                *count += 1;
                if *count < 3 {
                    Err(DaemonError::NetworkConnection {
                        message: "connection failed".to_string()
                    })
                } else {
                    Ok::<i32, DaemonError>(42)
                }
            }
        }).await;

        assert_eq!(result.unwrap(), 42);
        assert_eq!(*counter.lock().unwrap(), 3);
    }

    #[tokio::test]
    async fn test_retry_strategy_non_retryable_error() {
        let strategy = RetryStrategy::new();
        let counter = Arc::new(Mutex::new(0));
        let counter_clone = counter.clone();

        let result = strategy.execute(|| {
            let counter = counter_clone.clone();
            async move {
                let mut count = counter.lock().unwrap();
                *count += 1;
                Err::<i32, DaemonError>(DaemonError::InvalidInput {
                    message: "bad input".to_string()
                })
            }
        }).await;

        assert!(result.is_err());
        assert_eq!(*counter.lock().unwrap(), 1); // Should not retry
    }

    #[tokio::test]
    async fn test_retry_strategy_max_attempts_exceeded() {
        let config = RetryConfig::new()
            .max_attempts(2)
            .initial_delay(Duration::from_millis(10))
            .enable_jitter(false);

        let strategy = RetryStrategy::with_config(config).unwrap();
        let counter = Arc::new(Mutex::new(0));
        let counter_clone = counter.clone();

        let result = strategy.execute(|| {
            let counter = counter_clone.clone();
            async move {
                let mut count = counter.lock().unwrap();
                *count += 1;
                Err::<i32, DaemonError>(DaemonError::NetworkConnection {
                    message: "always fails".to_string()
                })
            }
        }).await;

        assert!(matches!(result, Err(DaemonError::RetryLimitExceeded { attempts: 2 })));
        assert_eq!(*counter.lock().unwrap(), 2);
    }

    #[tokio::test]
    async fn test_delay_calculation() {
        let config = RetryConfig::new()
            .initial_delay(Duration::from_millis(100))
            .backoff_multiplier(2.0)
            .max_delay(Duration::from_secs(5))
            .enable_jitter(false);

        let strategy = RetryStrategy::with_config(config).unwrap();
        let mut rng = rand::thread_rng();

        // Test exponential backoff
        let delay0 = strategy.calculate_delay(0, &mut rng).unwrap();
        let delay1 = strategy.calculate_delay(1, &mut rng).unwrap();
        let delay2 = strategy.calculate_delay(2, &mut rng).unwrap();

        assert_eq!(delay0, Duration::from_millis(100));
        assert_eq!(delay1, Duration::from_millis(200));
        assert_eq!(delay2, Duration::from_millis(400));
    }

    #[tokio::test]
    async fn test_delay_calculation_with_jitter() {
        let config = RetryConfig::new()
            .initial_delay(Duration::from_millis(100))
            .backoff_multiplier(2.0)
            .jitter_factor(0.1)
            .enable_jitter(true);

        let strategy = RetryStrategy::with_config(config).unwrap();
        let mut rng = rand::thread_rng();

        let delay = strategy.calculate_delay(0, &mut rng).unwrap();

        // With 10% jitter, delay should be between 90ms and 110ms
        assert!(delay >= Duration::from_millis(90));
        assert!(delay <= Duration::from_millis(110));
    }

    #[tokio::test]
    async fn test_delay_capping() {
        let config = RetryConfig::new()
            .initial_delay(Duration::from_millis(100))
            .backoff_multiplier(10.0)
            .max_delay(Duration::from_millis(500))
            .enable_jitter(false);

        let strategy = RetryStrategy::with_config(config).unwrap();
        let mut rng = rand::thread_rng();

        // After a few attempts, delay should be capped at max_delay
        let delay = strategy.calculate_delay(5, &mut rng).unwrap();
        assert_eq!(delay, Duration::from_millis(500));
    }

    #[tokio::test]
    async fn test_timing_of_retries() {
        let config = RetryConfig::new()
            .max_attempts(3)
            .initial_delay(Duration::from_millis(50))
            .backoff_multiplier(2.0)
            .enable_jitter(false);

        let strategy = RetryStrategy::with_config(config).unwrap();
        let start = Instant::now();

        let result = strategy.execute(|| async {
            Err::<i32, DaemonError>(DaemonError::NetworkConnection {
                message: "always fails".to_string()
            })
        }).await;

        let elapsed = start.elapsed();

        // Should take at least 50ms + 100ms = 150ms for 2 delays
        assert!(elapsed >= Duration::from_millis(140)); // Allow some tolerance
        assert!(result.is_err());
    }

    #[test]
    fn test_custom_retry_predicate() {
        struct CustomPredicate;

        impl RetryPredicate for CustomPredicate {
            fn should_retry(&self, error: &DaemonError, attempt: u32) -> bool {
                // Only retry network errors on first attempt
                matches!(error, DaemonError::NetworkConnection { .. }) && attempt == 0
            }
        }

        let predicate = CustomPredicate;

        // Should retry network error on first attempt
        assert!(predicate.should_retry(&DaemonError::NetworkConnection {
            message: "test".to_string()
        }, 0));

        // Should not retry network error on second attempt
        assert!(!predicate.should_retry(&DaemonError::NetworkConnection {
            message: "test".to_string()
        }, 1));
    }
}