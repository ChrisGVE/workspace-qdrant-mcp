//! Service Discovery Module
//!
//! This module exports all service discovery components.

pub mod registry;
pub mod network;
pub mod health;
pub mod manager;

pub use manager::DiscoveryManager;
pub use registry::{ServiceRegistry, ServiceInfo, ServiceStatus};
pub use network::{NetworkDiscovery, DiscoveryMessage, DiscoveryMessageType};
pub use health::{HealthChecker, HealthStatus};