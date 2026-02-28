//! Circuit breaker implementation for external service calls.

use std::time::{Duration, Instant};

use tracing::{info, warn};

use super::WorkspaceError;

/// Circuit breaker implementation for external service calls
#[derive(Debug)]
pub struct CircuitBreaker {
    name: String,
    failure_threshold: u32,
    timeout: Duration,
    reset_timeout: Duration,
    state: CircuitBreakerState,
    failure_count: u32,
    last_failure_time: Option<Instant>,
    last_success_time: Option<Instant>,
}

#[derive(Debug, PartialEq)]
enum CircuitBreakerState {
    Closed,
    Open,
    HalfOpen,
}

impl CircuitBreaker {
    /// Create a new circuit breaker
    pub fn new(name: impl Into<String>, failure_threshold: u32) -> Self {
        Self {
            name: name.into(),
            failure_threshold,
            timeout: Duration::from_secs(10),
            reset_timeout: Duration::from_secs(60),
            state: CircuitBreakerState::Closed,
            failure_count: 0,
            last_failure_time: None,
            last_success_time: None,
        }
    }

    /// Execute an operation with circuit breaker protection
    pub async fn execute<F, T, E>(&mut self, operation: F) -> std::result::Result<T, WorkspaceError>
    where
        F: std::future::Future<Output = std::result::Result<T, E>>,
        E: std::error::Error + Send + Sync + 'static,
    {
        match self.state {
            CircuitBreakerState::Open => {
                if let Some(last_failure) = self.last_failure_time {
                    if last_failure.elapsed() > self.reset_timeout {
                        info!("Circuit breaker transitioning to half-open: {}", self.name);
                        self.state = CircuitBreakerState::HalfOpen;
                    } else {
                        return Err(WorkspaceError::circuit_breaker_open(
                            &self.name,
                            self.failure_count,
                            "Circuit breaker is open",
                        ));
                    }
                } else {
                    return Err(WorkspaceError::circuit_breaker_open(
                        &self.name,
                        self.failure_count,
                        "Circuit breaker is open",
                    ));
                }
            }
            CircuitBreakerState::HalfOpen => {
                // Allow one test request
            }
            CircuitBreakerState::Closed => {
                // Normal operation
            }
        }

        // Execute the operation with timeout
        let result = tokio::time::timeout(self.timeout, operation).await;

        match result {
            Ok(Ok(value)) => {
                self.on_success();
                Ok(value)
            }
            Ok(Err(error)) => {
                self.on_failure();
                Err(WorkspaceError::Internal {
                    message: format!("Operation failed: {}", error),
                    component: self.name.clone(),
                    source: Some(Box::new(error)),
                })
            }
            Err(_) => {
                self.on_failure();
                Err(WorkspaceError::timeout(
                    format!("Operation timed out in circuit breaker: {}", self.name),
                    self.timeout,
                    &self.name,
                ))
            }
        }
    }

    fn on_success(&mut self) {
        self.failure_count = 0;
        self.last_success_time = Some(Instant::now());
        self.state = CircuitBreakerState::Closed;

        if self.state == CircuitBreakerState::HalfOpen {
            info!("Circuit breaker recovered: {}", self.name);
        }
    }

    fn on_failure(&mut self) {
        self.failure_count += 1;
        self.last_failure_time = Some(Instant::now());

        if self.failure_count >= self.failure_threshold {
            warn!(
                "Circuit breaker opened due to {} failures: {}",
                self.failure_count, self.name
            );
            self.state = CircuitBreakerState::Open;
        }
    }

    /// Get circuit breaker status
    pub fn status(&self) -> CircuitBreakerStatus {
        CircuitBreakerStatus {
            name: self.name.clone(),
            state: match self.state {
                CircuitBreakerState::Closed => "closed".to_string(),
                CircuitBreakerState::Open => "open".to_string(),
                CircuitBreakerState::HalfOpen => "half-open".to_string(),
            },
            failure_count: self.failure_count,
            failure_threshold: self.failure_threshold,
            last_failure_time: self.last_failure_time,
            last_success_time: self.last_success_time,
        }
    }
}

/// Circuit breaker status for monitoring
#[derive(Debug, Clone)]
pub struct CircuitBreakerStatus {
    pub name: String,
    pub state: String,
    pub failure_count: u32,
    pub failure_threshold: u32,
    pub last_failure_time: Option<Instant>,
    pub last_success_time: Option<Instant>,
}
