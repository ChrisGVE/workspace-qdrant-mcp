// Circuit breaker pattern for queue operations

use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{info, warn};

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

/// Manages circuit breaker state for a single collection
pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    state: CircuitBreakerState,
    collection: String,
}

impl CircuitBreaker {
    pub fn new(collection: impl Into<String>, config: CircuitBreakerConfig) -> Self {
        Self {
            config,
            state: CircuitBreakerState::default(),
            collection: collection.into(),
        }
    }

    /// Check whether the circuit allows a request through.
    /// Returns `(can_proceed, reason_str)`.
    pub fn check(&mut self) -> (bool, &'static str) {
        let current_time = Self::now_secs();

        match self.state.state {
            CircuitState::Open => {
                if let Some(opened_at) = self.state.opened_at {
                    if (current_time - opened_at) > self.config.recovery_timeout {
                        self.state.state = CircuitState::HalfOpen;
                        self.state.success_count = 0;
                        info!(
                            "Circuit breaker {} entering half-open state",
                            self.collection
                        );
                        return (true, "half-open");
                    }
                }
                (false, "circuit_open")
            }
            _ => (true, self.state.state.as_str()),
        }
    }

    /// Record a failure, potentially opening the circuit.
    /// Returns `true` if the circuit was just opened.
    pub fn record_failure(&mut self) -> bool {
        let current_time = Self::now_secs();

        self.state.failure_count += 1;
        self.state.last_failure_time = Some(current_time);
        self.state.failures.push(current_time);

        // Remove old failures outside the time window
        let window_start = current_time - self.config.failure_window;
        self.state.failures.retain(|&f| f > window_start);

        let failure_count = self.state.failures.len();
        match self.state.state {
            CircuitState::Closed => {
                if failure_count >= self.config.failure_threshold {
                    self.state.state = CircuitState::Open;
                    self.state.opened_at = Some(current_time);
                    warn!(
                        "Circuit breaker opened for {} after {} failures",
                        self.collection, failure_count
                    );
                    return true;
                }
            }
            CircuitState::HalfOpen => {
                // Failed in half-open, back to open
                self.state.state = CircuitState::Open;
                self.state.opened_at = Some(current_time);
                self.state.success_count = 0;
                warn!(
                    "Circuit breaker {} reopened after half-open failure",
                    self.collection
                );
                return true;
            }
            _ => {}
        }
        false
    }

    /// Record a success. When in half-open state this may close the circuit.
    pub fn record_success(&mut self) {
        if matches!(self.state.state, CircuitState::HalfOpen) {
            self.state.success_count += 1;

            if self.state.success_count >= self.config.success_threshold {
                self.state.state = CircuitState::Closed;
                self.state.failure_count = 0;
                self.state.failures.clear();
                info!(
                    "Circuit breaker {} closed after successful recovery",
                    self.collection
                );
            }
        }
    }

    /// Return the current state string for use in metrics/logging.
    pub fn state_str(&self) -> &'static str {
        self.state.state.as_str()
    }

    /// Return `true` if the circuit is currently closed (normal operation).
    pub fn is_closed(&self) -> bool {
        matches!(self.state.state, CircuitState::Closed)
    }

    fn now_secs() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }
}
