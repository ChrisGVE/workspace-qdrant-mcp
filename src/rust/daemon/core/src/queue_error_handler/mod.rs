//! Circuit breaker for queue and storage operations.

mod circuit_breaker;

pub use circuit_breaker::{CircuitBreaker, CircuitBreakerConfig, CircuitBreakerState, CircuitState};
