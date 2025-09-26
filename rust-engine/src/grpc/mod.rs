//! gRPC server and client implementations for all daemon services

pub mod server;
pub mod client;
pub mod services;
pub mod middleware;
pub mod service_discovery;
pub mod retry;
pub mod circuit_breaker;
pub mod message_validation;
pub mod streaming;
pub mod health;
pub mod health_service;
pub mod service_monitors;
pub mod alerting;
pub mod health_integration;
pub mod security;
pub mod transport;

#[cfg(test)]
pub mod shared_test_utils;

// Re-export only actually used types
pub use server::GrpcServer;
pub use retry::RetryStrategy;
pub use circuit_breaker::{CircuitBreaker, CircuitBreakerStats};

#[cfg(test)]
pub use shared_test_utils::*;