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

#[cfg(test)]
pub mod shared_test_utils;

pub use server::GrpcServer;
pub use client::{WorkspaceDaemonClient, ConnectionPool, ConnectionStats};
pub use service_discovery::{ServiceRegistry, ServiceInstance, LoadBalancingStrategy, ServiceDiscoveryConfig, ServiceDiscoveryStats};
// ServiceHealth is provided by health module
pub use retry::{RetryConfig, RetryStrategy, RetryPredicate, DefaultRetryPredicate};
pub use circuit_breaker::{CircuitBreaker, CircuitBreakerConfig, CircuitState, CircuitBreakerStats};
pub use message_validation::{MessageValidator, MessageStats, StreamHandle};
pub use streaming::{StreamingDocumentHandler, DocumentChunk, DocumentUploadResult, SearchRequest, SearchResult};
pub use health::{HealthStatus, ServiceHealth, ServiceMetrics, ServiceHealthMonitor, HealthMonitoringSystem, AlertConfig, ExternalMonitoring};
pub use health_service::{HealthService, ServiceStatus, ComponentHealth, SystemMetrics, Metric, ConnectionStatsProvider};
pub use service_monitors::{ServiceMonitors, DocumentProcessorMonitor, SearchServiceMonitor, MemoryServiceMonitor, DocumentProcessingStats, SearchStats, MemoryStats, ServiceStats};
pub use alerting::{AlertManager, AlertChannel, Alert, AlertSeverity, AlertType, RecoverySystem, RecoveryProcedure, RecoveryResult, LogAlertChannel, WebhookAlertChannel, EmailAlertChannel};
pub use health_integration::{HealthIntegration, HealthSystemStatus, HealthConfigurationFactory};

#[cfg(test)]
pub use shared_test_utils::*;