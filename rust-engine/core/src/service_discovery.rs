//! Service Discovery System
//!
//! This module provides automatic service discovery between the Rust daemon
//! and Python MCP server components. It supports multiple discovery strategies
//! with fallback mechanisms for robust component communication.
//!
//! # Discovery Strategies
//!
//! 1. **File-based Registry** - Primary method using shared services.json
//! 2. **Network Discovery** - UDP multicast for local network discovery  
//! 3. **Configuration Fallback** - Use unified config system endpoints
//! 4. **Default Endpoints** - Standard ports as last resort
//!
//! # Features
//!
//! - Cross-platform service registry management
//! - Health checking with automatic re-discovery
//! - Process lifecycle tracking and cleanup
//! - Security through authentication tokens
//! - Graceful degradation and fallback mechanisms

pub mod registry;
pub mod network;
pub mod health;
pub mod manager;

pub use manager::DiscoveryManager;
pub use registry::{ServiceRegistry, ServiceInfo, ServiceStatus};
pub use network::{NetworkDiscovery, DiscoveryMessage, DiscoveryMessageType};
pub use health::{HealthChecker, HealthStatus};

use std::time::Duration;
use thiserror::Error;

/// Service discovery errors
#[derive(Error, Debug)]
pub enum DiscoveryError {
    #[error("Registry error: {0}")]
    RegistryError(#[from] registry::RegistryError),
    
    #[error("Network discovery error: {0}")]
    NetworkError(#[from] network::NetworkError),
    
    #[error("Health check error: {0}")]
    HealthError(#[from] health::HealthError),
    
    #[error("Service not found: {0}")]
    ServiceNotFound(String),
    
    #[error("Authentication failed")]
    AuthenticationFailed,
    
    #[error("Timeout waiting for service discovery")]
    Timeout,
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("JSON serialization error: {0}")]
    JsonError(#[from] serde_json::Error),
}

/// Service discovery configuration
#[derive(Debug, Clone)]
pub struct DiscoveryConfig {
    /// Registry file location (defaults to ~/.workspace-qdrant/services.json)
    pub registry_path: Option<std::path::PathBuf>,
    
    /// Network discovery multicast address
    pub multicast_address: String,
    
    /// Network discovery port
    pub multicast_port: u16,
    
    /// Discovery timeout for network operations
    pub discovery_timeout: Duration,
    
    /// Health check interval
    pub health_check_interval: Duration,
    
    /// Registry cleanup interval for stale entries
    pub cleanup_interval: Duration,
    
    /// Enable network discovery
    pub enable_network_discovery: bool,
    
    /// Enable authentication tokens
    pub enable_authentication: bool,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            registry_path: None,
            multicast_address: "239.255.42.42".to_string(),
            multicast_port: 9999,
            discovery_timeout: Duration::from_secs(10),
            health_check_interval: Duration::from_secs(30),
            cleanup_interval: Duration::from_secs(60),
            enable_network_discovery: true,
            enable_authentication: true,
        }
    }
}

/// Known service names in the discovery system
pub mod service_names {
    pub const RUST_DAEMON: &str = "rust-daemon";
    pub const PYTHON_MCP: &str = "python-mcp";
}

/// Result type for discovery operations
pub type DiscoveryResult<T> = Result<T, DiscoveryError>;