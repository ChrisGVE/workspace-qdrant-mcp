//! gRPC server and client implementations for all daemon services

pub mod server;
pub mod services;
pub mod middleware;
pub mod retry;
pub mod circuit_breaker;
pub mod message_validation;
pub mod security;

// Modules only used in tests or not used at all
#[cfg(any(test, feature = "test-utils"))]
pub mod client;
#[cfg(any(test, feature = "test-utils"))]
pub mod service_discovery;
#[cfg(any(test, feature = "test-utils"))]
pub mod streaming;
#[cfg(any(test, feature = "test-utils"))]
pub mod health;
#[cfg(any(test, feature = "test-utils"))]
pub mod health_service;
#[cfg(any(test, feature = "test-utils"))]
pub mod service_monitors;
#[cfg(any(test, feature = "test-utils"))]
pub mod alerting;
#[cfg(any(test, feature = "test-utils"))]
pub mod health_integration;
#[cfg(any(test, feature = "test-utils"))]
pub mod transport;

#[cfg(test)]
pub mod shared_test_utils;

// Re-export types used by external binaries
// NOTE: These are currently unused but kept for potential future use
#[allow(unused_imports)]
pub use retry::RetryStrategy;
#[allow(unused_imports)]
pub use circuit_breaker::CircuitBreaker;

#[cfg(test)]
pub use shared_test_utils::*;