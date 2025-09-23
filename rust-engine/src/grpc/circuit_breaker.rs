//! Circuit breaker pattern implementation for fault tolerance
//!
//! This module provides circuit breaker functionality to prevent cascading
//! failures by monitoring service health and cutting connections when failures
//! exceed configured thresholds.

use crate::error::{DaemonError, DaemonResult};
use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Circuit breaker state
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CircuitState {
    /// Circuit is closed, requests are allowed through
    Closed,
    /// Circuit is open, requests are immediately rejected
    Open,
    /// Circuit is half-open, allowing limited requests to test service recovery
    HalfOpen,
}

/// Configuration for circuit breaker behavior
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Failure threshold to trigger circuit opening
    pub failure_threshold: u32,
    /// Success threshold to close circuit from half-open state
    pub success_threshold: u32,
    /// Time to wait before attempting recovery (moving to half-open)
    pub recovery_timeout: Duration,
    /// Window size for failure rate calculation
    pub failure_window_size: u32,
    /// Minimum number of requests before considering failure rate
    pub minimum_requests: u32,
    /// Timeout for individual requests
    pub request_timeout: Duration,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 3,
            recovery_timeout: Duration::from_secs(60),
            failure_window_size: 10,
            minimum_requests: 5,
            request_timeout: Duration::from_secs(30),
        }
    }
}

impl CircuitBreakerConfig {
    /// Create a new circuit breaker configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set failure threshold
    pub fn failure_threshold(mut self, threshold: u32) -> Self {
        self.failure_threshold = threshold;
        self
    }

    /// Set success threshold
    pub fn success_threshold(mut self, threshold: u32) -> Self {
        self.success_threshold = threshold;
        self
    }

    /// Set recovery timeout
    pub fn recovery_timeout(mut self, timeout: Duration) -> Self {
        self.recovery_timeout = timeout;
        self
    }

    /// Set failure window size
    pub fn failure_window_size(mut self, size: u32) -> Self {
        self.failure_window_size = size;
        self
    }

    /// Set minimum requests threshold
    pub fn minimum_requests(mut self, minimum: u32) -> Self {
        self.minimum_requests = minimum;
        self
    }

    /// Set request timeout
    pub fn request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = timeout;
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> DaemonResult<()> {
        if self.failure_threshold == 0 {
            return Err(DaemonError::CircuitBreakerConfig {
                message: "failure_threshold must be greater than 0".to_string(),
            });
        }

        if self.success_threshold == 0 {
            return Err(DaemonError::CircuitBreakerConfig {
                message: "success_threshold must be greater than 0".to_string(),
            });
        }

        if self.failure_window_size == 0 {
            return Err(DaemonError::CircuitBreakerConfig {
                message: "failure_window_size must be greater than 0".to_string(),
            });
        }

        if self.minimum_requests == 0 {
            return Err(DaemonError::CircuitBreakerConfig {
                message: "minimum_requests must be greater than 0".to_string(),
            });
        }

        if self.recovery_timeout.is_zero() {
            return Err(DaemonError::CircuitBreakerConfig {
                message: "recovery_timeout must be greater than 0".to_string(),
            });
        }

        if self.request_timeout.is_zero() {
            return Err(DaemonError::CircuitBreakerConfig {
                message: "request_timeout must be greater than 0".to_string(),
            });
        }

        Ok(())
    }
}

/// Request outcome for circuit breaker metrics
#[derive(Debug, Clone, Copy)]
enum RequestOutcome {
    Success,
    Failure,
}

/// Circuit breaker metrics
#[derive(Debug, Clone)]
struct CircuitBreakerMetrics {
    /// Recent request outcomes in a sliding window
    recent_outcomes: Vec<RequestOutcome>,
    /// Current failure count in the window
    failure_count: u32,
    /// Current success count (used in half-open state)
    success_count: u32,
    /// Total requests processed
    total_requests: u64,
    /// Total failures encountered
    total_failures: u64,
    /// Last state transition time
    last_state_change: Instant,
}

impl CircuitBreakerMetrics {
    fn new() -> Self {
        Self {
            recent_outcomes: Vec::new(),
            failure_count: 0,
            success_count: 0,
            total_requests: 0,
            total_failures: 0,
            last_state_change: Instant::now(),
        }
    }

    /// Record a request outcome
    fn record_outcome(&mut self, outcome: RequestOutcome, window_size: u32) {
        self.total_requests += 1;

        // Add to sliding window
        self.recent_outcomes.push(outcome);

        // Maintain window size
        if self.recent_outcomes.len() > window_size as usize {
            let removed = self.recent_outcomes.remove(0);
            match removed {
                RequestOutcome::Failure => self.failure_count = self.failure_count.saturating_sub(1),
                RequestOutcome::Success => self.success_count = self.success_count.saturating_sub(1),
            }
        }

        // Update counters
        match outcome {
            RequestOutcome::Failure => {
                self.failure_count += 1;
                self.total_failures += 1;
            }
            RequestOutcome::Success => {
                self.success_count += 1;
            }
        }
    }

    /// Get current failure rate
    fn failure_rate(&self) -> f64 {
        if self.recent_outcomes.is_empty() {
            0.0
        } else {
            self.failure_count as f64 / self.recent_outcomes.len() as f64
        }
    }

    /// Check if we have enough requests for meaningful statistics
    fn has_minimum_requests(&self, minimum: u32) -> bool {
        self.recent_outcomes.len() >= minimum as usize
    }

    /// Reset success count (used when transitioning states)
    fn reset_success_count(&mut self) {
        self.success_count = 0;
    }

    /// Update last state change time
    fn update_state_change_time(&mut self) {
        self.last_state_change = Instant::now();
    }
}

/// Circuit breaker implementation
#[derive(Debug)]
pub struct CircuitBreaker {
    /// Circuit breaker configuration
    config: CircuitBreakerConfig,
    /// Current circuit state
    state: Arc<RwLock<CircuitState>>,
    /// Circuit breaker metrics
    metrics: Arc<RwLock<CircuitBreakerMetrics>>,
    /// Service name for logging
    service_name: String,
}

impl CircuitBreaker {
    /// Create a new circuit breaker
    pub fn new(service_name: String, config: CircuitBreakerConfig) -> DaemonResult<Self> {
        config.validate()?;

        Ok(Self {
            config,
            state: Arc::new(RwLock::new(CircuitState::Closed)),
            metrics: Arc::new(RwLock::new(CircuitBreakerMetrics::new())),
            service_name,
        })
    }

    /// Create a circuit breaker with default configuration
    pub fn new_default(service_name: String) -> DaemonResult<Self> {
        Self::new(service_name, CircuitBreakerConfig::default())
    }

    /// Execute an operation through the circuit breaker
    pub async fn execute<F, Fut, T>(&self, operation: F) -> DaemonResult<T>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = DaemonResult<T>>,
    {
        // Check if circuit allows request
        self.check_circuit().await?;

        // Execute operation with timeout
        let result = tokio::time::timeout(self.config.request_timeout, operation()).await;

        match result {
            Ok(Ok(success)) => {
                self.record_success().await;
                Ok(success)
            }
            Ok(Err(error)) => {
                self.record_failure().await;
                Err(error)
            }
            Err(_) => {
                // Timeout occurred
                let timeout_error = DaemonError::NetworkTimeout {
                    timeout_ms: self.config.request_timeout.as_millis() as u64,
                };
                self.record_failure().await;
                Err(timeout_error)
            }
        }
    }

    /// Check if the circuit allows requests
    async fn check_circuit(&self) -> DaemonResult<()> {
        let state = self.state.read().await;
        match *state {
            CircuitState::Closed => Ok(()),
            CircuitState::Open => {
                // Check if enough time has passed for recovery attempt
                let metrics = self.metrics.read().await;
                if metrics.last_state_change.elapsed() >= self.config.recovery_timeout {
                    drop(metrics);
                    drop(state);

                    // Transition to half-open
                    self.transition_to_half_open().await;
                    Ok(())
                } else {
                    Err(DaemonError::CircuitBreakerOpen {
                        service: self.service_name.clone(),
                    })
                }
            }
            CircuitState::HalfOpen => Ok(()),
        }
    }

    /// Record a successful operation
    async fn record_success(&self) {
        let mut metrics = self.metrics.write().await;
        metrics.record_outcome(RequestOutcome::Success, self.config.failure_window_size);

        let state = self.state.read().await;
        if *state == CircuitState::HalfOpen {
            // Check if we have enough successes to close the circuit
            if metrics.success_count >= self.config.success_threshold {
                drop(state);
                drop(metrics);
                self.transition_to_closed().await;
            }
        }
    }

    /// Record a failed operation
    async fn record_failure(&self) {
        let mut metrics = self.metrics.write().await;
        metrics.record_outcome(RequestOutcome::Failure, self.config.failure_window_size);

        let state = self.state.read().await;
        match *state {
            CircuitState::Closed => {
                // Check if we should open the circuit
                if metrics.has_minimum_requests(self.config.minimum_requests)
                    && metrics.failure_count >= self.config.failure_threshold {
                    drop(state);
                    drop(metrics);
                    self.transition_to_open().await;
                }
            }
            CircuitState::HalfOpen => {
                // Any failure in half-open state should open the circuit
                drop(state);
                drop(metrics);
                self.transition_to_open().await;
            }
            CircuitState::Open => {
                // Already open, nothing to do
            }
        }
    }

    /// Transition to closed state
    async fn transition_to_closed(&self) {
        let mut state = self.state.write().await;
        let mut metrics = self.metrics.write().await;

        *state = CircuitState::Closed;
        metrics.reset_success_count();
        metrics.update_state_change_time();

        info!("Circuit breaker for {} transitioned to CLOSED", self.service_name);
    }

    /// Transition to open state
    async fn transition_to_open(&self) {
        let mut state = self.state.write().await;
        let mut metrics = self.metrics.write().await;

        *state = CircuitState::Open;
        metrics.reset_success_count();
        metrics.update_state_change_time();

        warn!("Circuit breaker for {} transitioned to OPEN", self.service_name);
    }

    /// Transition to half-open state
    async fn transition_to_half_open(&self) {
        let mut state = self.state.write().await;
        let mut metrics = self.metrics.write().await;

        *state = CircuitState::HalfOpen;
        metrics.reset_success_count();
        metrics.update_state_change_time();

        info!("Circuit breaker for {} transitioned to HALF-OPEN", self.service_name);
    }

    /// Get current circuit state
    pub async fn state(&self) -> CircuitState {
        *self.state.read().await
    }

    /// Get circuit breaker statistics
    pub async fn stats(&self) -> CircuitBreakerStats {
        let state = *self.state.read().await;
        let metrics = self.metrics.read().await;

        CircuitBreakerStats {
            state,
            failure_rate: metrics.failure_rate(),
            total_requests: metrics.total_requests,
            total_failures: metrics.total_failures,
            current_failures: metrics.failure_count,
            current_successes: metrics.success_count,
            time_since_state_change: metrics.last_state_change.elapsed(),
        }
    }

    /// Force circuit to open (for testing or manual intervention)
    pub async fn force_open(&self) {
        self.transition_to_open().await;
    }

    /// Force circuit to close (for testing or manual intervention)
    pub async fn force_close(&self) {
        self.transition_to_closed().await;
    }

    /// Get service name
    pub fn service_name(&self) -> &str {
        &self.service_name
    }

    /// Get configuration
    pub fn config(&self) -> &CircuitBreakerConfig {
        &self.config
    }
}

/// Circuit breaker statistics
#[derive(Debug, Clone)]
pub struct CircuitBreakerStats {
    /// Current circuit state
    pub state: CircuitState,
    /// Current failure rate (0.0 to 1.0)
    pub failure_rate: f64,
    /// Total requests processed
    pub total_requests: u64,
    /// Total failures encountered
    pub total_failures: u64,
    /// Current failures in the sliding window
    pub current_failures: u32,
    /// Current successes in half-open state
    pub current_successes: u32,
    /// Time since last state change
    pub time_since_state_change: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use tokio::time::{sleep, Duration};

    #[test]
    fn test_circuit_breaker_config_validation() {
        // Valid config
        let config = CircuitBreakerConfig::new()
            .failure_threshold(5)
            .success_threshold(3)
            .recovery_timeout(Duration::from_secs(60))
            .failure_window_size(10)
            .minimum_requests(5)
            .request_timeout(Duration::from_secs(30));
        assert!(config.validate().is_ok());

        // Invalid failure_threshold
        let config = CircuitBreakerConfig::new().failure_threshold(0);
        assert!(config.validate().is_err());

        // Invalid success_threshold
        let config = CircuitBreakerConfig::new().success_threshold(0);
        assert!(config.validate().is_err());

        // Invalid failure_window_size
        let config = CircuitBreakerConfig::new().failure_window_size(0);
        assert!(config.validate().is_err());

        // Invalid minimum_requests
        let config = CircuitBreakerConfig::new().minimum_requests(0);
        assert!(config.validate().is_err());

        // Invalid recovery_timeout
        let config = CircuitBreakerConfig::new().recovery_timeout(Duration::ZERO);
        assert!(config.validate().is_err());

        // Invalid request_timeout
        let config = CircuitBreakerConfig::new().request_timeout(Duration::ZERO);
        assert!(config.validate().is_err());
    }

    #[tokio::test]
    async fn test_circuit_breaker_closed_state() {
        let config = CircuitBreakerConfig::new()
            .failure_threshold(3)
            .minimum_requests(2);

        let cb = CircuitBreaker::new("test-service".to_string(), config).unwrap();

        // Should start in closed state
        assert_eq!(cb.state().await, CircuitState::Closed);

        // Successful operations should keep circuit closed
        let result = cb.execute(|| async { Ok::<i32, DaemonError>(42) }).await;
        assert_eq!(result.unwrap(), 42);
        assert_eq!(cb.state().await, CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_circuit_breaker_opening() {
        let config = CircuitBreakerConfig::new()
            .failure_threshold(2)
            .minimum_requests(2)
            .failure_window_size(5);

        let cb = CircuitBreaker::new("test-service".to_string(), config).unwrap();

        // Generate enough failures to open circuit
        for _ in 0..3 {
            let _ = cb.execute(|| async {
                Err::<i32, DaemonError>(DaemonError::NetworkConnection {
                    message: "test failure".to_string()
                })
            }).await;
        }

        // Circuit should be open now
        assert_eq!(cb.state().await, CircuitState::Open);

        // Next request should be immediately rejected
        let result = cb.execute(|| async { Ok::<i32, DaemonError>(42) }).await;
        assert!(matches!(result, Err(DaemonError::CircuitBreakerOpen { .. })));
    }

    #[tokio::test]
    async fn test_circuit_breaker_half_open_to_closed() {
        let config = CircuitBreakerConfig::new()
            .failure_threshold(2)
            .success_threshold(2)
            .minimum_requests(2)
            .recovery_timeout(Duration::from_millis(10))
            .failure_window_size(5);

        let cb = CircuitBreaker::new("test-service".to_string(), config).unwrap();

        // Open the circuit
        for _ in 0..3 {
            let _ = cb.execute(|| async {
                Err::<i32, DaemonError>(DaemonError::NetworkConnection {
                    message: "test failure".to_string()
                })
            }).await;
        }
        assert_eq!(cb.state().await, CircuitState::Open);

        // Wait for recovery timeout
        sleep(Duration::from_millis(20)).await;

        // Next request should transition to half-open
        let result = cb.execute(|| async { Ok::<i32, DaemonError>(42) }).await;
        assert_eq!(result.unwrap(), 42);

        // Should be in half-open state after first success
        let state = cb.state().await;
        // Could be either half-open or closed depending on timing
        assert!(state == CircuitState::HalfOpen || state == CircuitState::Closed);

        // Another success should close the circuit
        let result = cb.execute(|| async { Ok::<i32, DaemonError>(42) }).await;
        assert_eq!(result.unwrap(), 42);
        assert_eq!(cb.state().await, CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_circuit_breaker_half_open_to_open() {
        let config = CircuitBreakerConfig::new()
            .failure_threshold(2)
            .success_threshold(3)
            .minimum_requests(2)
            .recovery_timeout(Duration::from_millis(10))
            .failure_window_size(5);

        let cb = CircuitBreaker::new("test-service".to_string(), config).unwrap();

        // Open the circuit
        for _ in 0..3 {
            let _ = cb.execute(|| async {
                Err::<i32, DaemonError>(DaemonError::NetworkConnection {
                    message: "test failure".to_string()
                })
            }).await;
        }
        assert_eq!(cb.state().await, CircuitState::Open);

        // Wait for recovery timeout
        sleep(Duration::from_millis(20)).await;

        // First request should succeed and transition to half-open
        let result = cb.execute(|| async { Ok::<i32, DaemonError>(42) }).await;
        assert_eq!(result.unwrap(), 42);

        // Next request fails, should open circuit again
        let result = cb.execute(|| async {
            Err::<i32, DaemonError>(DaemonError::NetworkConnection {
                message: "test failure".to_string()
            })
        }).await;
        assert!(result.is_err());
        assert_eq!(cb.state().await, CircuitState::Open);
    }

    #[tokio::test]
    async fn test_circuit_breaker_timeout() {
        let config = CircuitBreakerConfig::new()
            .request_timeout(Duration::from_millis(50))
            .failure_threshold(1)
            .minimum_requests(1);

        let cb = CircuitBreaker::new("test-service".to_string(), config).unwrap();

        // Operation that takes too long
        let result = cb.execute(|| async {
            sleep(Duration::from_millis(100)).await;
            Ok::<i32, DaemonError>(42)
        }).await;

        assert!(matches!(result, Err(DaemonError::NetworkTimeout { .. })));
        assert_eq!(cb.state().await, CircuitState::Open);
    }

    #[tokio::test]
    async fn test_circuit_breaker_stats() {
        let config = CircuitBreakerConfig::new()
            .failure_threshold(2)
            .minimum_requests(2);

        let cb = CircuitBreaker::new("test-service".to_string(), config).unwrap();

        // Execute some operations
        let _ = cb.execute(|| async { Ok::<i32, DaemonError>(42) }).await;
        let _ = cb.execute(|| async {
            Err::<i32, DaemonError>(DaemonError::NetworkConnection {
                message: "test failure".to_string()
            })
        }).await;

        let stats = cb.stats().await;
        assert_eq!(stats.total_requests, 2);
        assert_eq!(stats.total_failures, 1);
        assert_eq!(stats.current_failures, 1);
        assert!(stats.failure_rate > 0.0);
    }

    #[tokio::test]
    async fn test_circuit_breaker_force_open_close() {
        let cb = CircuitBreaker::new_default("test-service".to_string()).unwrap();

        // Should start closed
        assert_eq!(cb.state().await, CircuitState::Closed);

        // Force open
        cb.force_open().await;
        assert_eq!(cb.state().await, CircuitState::Open);

        // Force close
        cb.force_close().await;
        assert_eq!(cb.state().await, CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_circuit_breaker_metrics_sliding_window() {
        let config = CircuitBreakerConfig::new()
            .failure_window_size(3)
            .failure_threshold(2)
            .minimum_requests(2);

        let cb = CircuitBreaker::new("test-service".to_string(), config).unwrap();

        // Fill up the window with successes
        for _ in 0..3 {
            let _ = cb.execute(|| async { Ok::<i32, DaemonError>(42) }).await;
        }

        let stats = cb.stats().await;
        assert_eq!(stats.current_failures, 0);
        assert_eq!(cb.state().await, CircuitState::Closed);

        // Add failures, but they should push out successes
        for _ in 0..2 {
            let _ = cb.execute(|| async {
                Err::<i32, DaemonError>(DaemonError::NetworkConnection {
                    message: "test failure".to_string()
                })
            }).await;
        }

        let stats = cb.stats().await;
        assert_eq!(stats.current_failures, 2);
    }

    #[tokio::test]
    async fn test_circuit_breaker_concurrent_access() {
        let config = CircuitBreakerConfig::new()
            .failure_threshold(10)
            .minimum_requests(5);

        let cb = Arc::new(CircuitBreaker::new("test-service".to_string(), config).unwrap());

        let counter = Arc::new(Mutex::new(0));
        let mut handles = vec![];

        // Spawn multiple concurrent operations
        for _ in 0..10 {
            let cb_clone = cb.clone();
            let counter_clone = counter.clone();
            let handle = tokio::spawn(async move {
                let result = cb_clone.execute(|| async {
                    let mut count = counter_clone.lock().unwrap();
                    *count += 1;
                    Ok::<i32, DaemonError>(*count)
                }).await;
                result.is_ok()
            });
            handles.push(handle);
        }

        // Wait for all operations to complete
        for handle in handles {
            let _ = handle.await;
        }

        let stats = cb.stats().await;
        assert_eq!(stats.total_requests, 10);
        assert_eq!(*counter.lock().unwrap(), 10);
    }
}