//! Thread-safe Qdrant circuit breaker for the StorageClient.
//!
//! Wraps the existing `CircuitBreaker` from `queue_error_handler` in an
//! `Arc<Mutex<>>` so it can be shared across async tasks and the queue
//! processor's health-check loop.

use std::sync::Mutex;
use tracing::{debug, info};

use crate::queue_error_handler::{CircuitBreaker, CircuitBreakerConfig};

/// Thread-safe Qdrant availability tracker.
///
/// Intended to be stored as `Arc<QdrantCircuitBreaker>` inside `StorageClient`
/// and shared with the queue processor and health-check loop.
pub struct QdrantCircuitBreaker {
    inner: Mutex<CircuitBreaker>,
}

impl QdrantCircuitBreaker {
    /// Create a new circuit breaker with the given configuration.
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            inner: Mutex::new(CircuitBreaker::new("qdrant", config)),
        }
    }

    /// Create with defaults tuned for Qdrant:
    /// - Open after 5 failures within 60 seconds
    /// - Stay open for 30 seconds
    /// - Close after 2 successes in half-open
    pub fn with_defaults() -> Self {
        Self::new(CircuitBreakerConfig {
            failure_threshold: 5,
            failure_window: 60,
            recovery_timeout: 30,
            success_threshold: 2,
        })
    }

    /// Check if the circuit allows a request. Returns `true` if requests
    /// should proceed, `false` if the circuit is open (Qdrant presumed down).
    pub fn is_available(&self) -> bool {
        let mut cb = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let (can_proceed, _reason) = cb.check();
        can_proceed
    }

    /// Record a successful Qdrant operation.
    pub fn record_success(&self) {
        let mut cb = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        cb.record_success();
    }

    /// Record a failed Qdrant operation. Returns `true` if the circuit
    /// just transitioned to open.
    pub fn record_failure(&self) -> bool {
        let mut cb = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        cb.record_failure()
    }

    /// Return the current state as a string ("closed", "open", "half-open").
    pub fn state_str(&self) -> &'static str {
        let cb = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        cb.state_str()
    }

    /// Return `true` if the circuit is closed (normal operation).
    pub fn is_closed(&self) -> bool {
        let cb = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        cb.is_closed()
    }

    /// Log the current state at debug level.
    pub fn log_state(&self) {
        let state = self.state_str();
        if state != "closed" {
            info!("Qdrant circuit breaker state: {}", state);
        } else {
            debug!("Qdrant circuit breaker state: closed");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_starts_closed() {
        let cb = QdrantCircuitBreaker::with_defaults();
        assert!(cb.is_available());
        assert!(cb.is_closed());
        assert_eq!(cb.state_str(), "closed");
    }

    #[test]
    fn test_opens_after_threshold_failures() {
        let cb = QdrantCircuitBreaker::new(CircuitBreakerConfig {
            failure_threshold: 3,
            failure_window: 60,
            recovery_timeout: 30,
            success_threshold: 2,
        });

        // 2 failures: still closed
        cb.record_failure();
        cb.record_failure();
        assert!(cb.is_available());

        // 3rd failure: opens
        let just_opened = cb.record_failure();
        assert!(just_opened);
        assert!(!cb.is_available());
        assert_eq!(cb.state_str(), "open");
    }

    #[test]
    fn test_success_resets_in_closed_state() {
        let cb = QdrantCircuitBreaker::new(CircuitBreakerConfig {
            failure_threshold: 3,
            failure_window: 60,
            recovery_timeout: 30,
            success_threshold: 2,
        });

        cb.record_failure();
        cb.record_failure();
        cb.record_success(); // should not change state (still closed)
        assert!(cb.is_available());
        assert!(cb.is_closed());
    }
}
