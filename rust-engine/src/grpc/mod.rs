//! gRPC server implementations for all daemon services

pub mod server;
pub mod services;
pub mod middleware;
pub mod service_discovery;

pub use server::GrpcServer;
pub use service_discovery::{ServiceRegistry, ServiceInstance, ServiceHealth, LoadBalancingStrategy, ServiceDiscoveryConfig, ServiceDiscoveryStats};