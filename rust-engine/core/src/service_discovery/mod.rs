//! Service Discovery Module
//!
//! This module exports all service discovery components.

use thiserror::Error;
use serde::{Deserialize, Serialize};
use std::time::Duration;

pub mod registry;
pub mod network;
pub mod health;
pub mod manager;

/// Service discovery errors
#[derive(Error, Debug)]
pub enum DiscoveryError {
    #[error("Registry error: {0}")]
    Registry(String),
    
    #[error("Network discovery failed: {0}")]
    Network(String),
    
    #[error("Health check failed: {0}")]
    HealthCheck(String),
    
    #[error("Configuration error: {0}")]
    Configuration(String),
    
    #[error("Timeout: {0}")]
    Timeout(String),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Service discovery result type
pub type DiscoveryResult<T> = Result<T, DiscoveryError>;

/// Service discovery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryConfig {
    /// Enable registry-based discovery
    pub enable_registry: bool,
    
    /// Enable network multicast discovery
    pub enable_network: bool,
    
    /// Enable health checking
    pub enable_health_check: bool,
    
    /// Discovery timeout
    pub discovery_timeout: Duration,
    
    /// Registry file path
    pub registry_path: Option<String>,
    
    /// Network discovery port
    pub network_port: Option<u16>,
    
    /// Health check configuration
    pub health_config: Option<health::HealthConfig>,
    
    /// Cleanup interval for stale entries
    pub cleanup_interval: Duration,
    
    /// Enable authentication
    pub enable_authentication: bool,
    
    /// Multicast address for network discovery
    pub multicast_address: String,
    
    /// Multicast port for network discovery
    pub multicast_port: u16,
    
    /// Health check interval
    pub health_check_interval: Duration,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            enable_registry: true,
            enable_network: true,
            enable_health_check: true,
            discovery_timeout: Duration::from_secs(30),
            registry_path: None,
            network_port: Some(8765),
            health_config: None,
            cleanup_interval: Duration::from_secs(300), // 5 minutes
            enable_authentication: false,
            multicast_address: "224.0.0.1".to_string(),
            multicast_port: 8765,
            health_check_interval: Duration::from_secs(30),
        }
    }
}

// Error conversions
impl From<registry::RegistryError> for DiscoveryError {
    fn from(error: registry::RegistryError) -> Self {
        DiscoveryError::Registry(error.to_string())
    }
}

impl From<network::NetworkError> for DiscoveryError {
    fn from(error: network::NetworkError) -> Self {
        DiscoveryError::Network(error.to_string())
    }
}

impl From<health::HealthError> for DiscoveryError {
    fn from(error: health::HealthError) -> Self {
        DiscoveryError::HealthCheck(error.to_string())
    }
}

pub use manager::DiscoveryManager;
pub use registry::{ServiceRegistry, ServiceInfo, ServiceStatus};
pub use network::{NetworkDiscovery, DiscoveryMessage, DiscoveryMessageType};
pub use health::{HealthChecker, HealthStatus, HealthConfig};