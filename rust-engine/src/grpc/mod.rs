//! gRPC server and client implementations for all daemon services

pub mod server;
pub mod client;
pub mod services;
pub mod middleware;
pub mod service_discovery;

#[cfg(test)]
pub mod shared_test_utils;

pub use server::GrpcServer;
pub use client::{WorkspaceDaemonClient, ConnectionPool, ConnectionStats};
pub use service_discovery::{ServiceRegistry, ServiceInstance, ServiceHealth, LoadBalancingStrategy, ServiceDiscoveryConfig, ServiceDiscoveryStats};

#[cfg(test)]
pub use shared_test_utils::*;