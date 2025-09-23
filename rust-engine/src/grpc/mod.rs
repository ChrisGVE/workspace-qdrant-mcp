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

#[cfg(test)]
pub mod shared_test_utils;

pub use server::GrpcServer;
pub use client::{WorkspaceDaemonClient, ConnectionPool, ConnectionStats};
pub use service_discovery::{ServiceRegistry, ServiceInstance, ServiceHealth, LoadBalancingStrategy, ServiceDiscoveryConfig, ServiceDiscoveryStats};
pub use retry::{RetryConfig, RetryStrategy, RetryPredicate, DefaultRetryPredicate};
pub use circuit_breaker::{CircuitBreaker, CircuitBreakerConfig, CircuitState, CircuitBreakerStats};
pub use message_validation::{MessageValidator, MessageStats, StreamHandle};
pub use streaming::{StreamingDocumentHandler, DocumentChunk, DocumentUploadResult, SearchRequest, SearchResult};

#[cfg(test)]
pub use shared_test_utils::*;