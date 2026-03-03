// Queue Error Handling and Retry System
//
// Provides comprehensive error classification, retry strategies with exponential
// backoff, dead letter queue management, and circuit breaker pattern for queue operations.

mod circuit_breaker;
mod error_types;
mod handler;
mod retry;

// Re-export the full public API so that all existing import paths remain valid.
pub use circuit_breaker::{
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerState, CircuitState,
};
pub use error_types::{ErrorCategory, ErrorType};
pub use handler::{ErrorHandler, ErrorMetrics};
pub use retry::RetryConfig;
