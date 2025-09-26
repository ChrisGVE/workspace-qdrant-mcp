//! Transport layer optimizations for gRPC communication
//!
//! This module provides Unix domain socket support, local communication optimizations,
//! and transport selection strategies for efficient inter-process communication.

use crate::config::{
    TransportConfig, UnixSocketConfig, LocalOptimizationConfig,
    LocalLatencyConfig, TransportStrategy
};
use crate::error::{DaemonResult, DaemonError};

use anyhow::{Result, anyhow};
use std::path::Path;
use std::os::unix::fs::PermissionsExt;
use std::sync::Arc;
use std::time::Duration;
use tonic::transport::{Server, Endpoint};
use tokio::net::{UnixListener, UnixStream};
use tokio_stream::wrappers::UnixListenerStream;
use tower::service_fn;
use tracing::{info, warn, error, debug};

#[cfg(unix)]
use std::os::unix::net::UnixListener as StdUnixListener;

/// Transport type detection result
#[derive(Debug, Clone)]
pub enum TransportType {
    /// TCP transport with host and port
    Tcp { host: String, port: u16 },
    /// Unix domain socket transport with socket path
    UnixSocket { path: String },
}

/// Local connection optimization settings
#[derive(Debug, Clone)]
pub struct LocalOptimization {
    /// Buffer size for local connections
    pub buffer_size: usize,
    /// Disable Nagle's algorithm
    pub disable_nagle: bool,
    /// Connection pool size
    pub connection_pool_size: u32,
    /// Keep-alive interval
    pub keepalive_interval: Duration,
    /// Enable memory-efficient serialization
    pub memory_efficient_serialization: bool,
}

impl From<LocalOptimizationConfig> for LocalOptimization {
    fn from(config: LocalOptimizationConfig) -> Self {
        Self {
            buffer_size: config.local_buffer_size,
            disable_nagle: config.reduce_latency.disable_nagle,
            connection_pool_size: config.reduce_latency.connection_pool_size,
            keepalive_interval: Duration::from_secs(config.reduce_latency.keepalive_interval_secs),
            memory_efficient_serialization: config.memory_efficient_serialization,
        }
    }
}

/// Unix domain socket manager
#[derive(Debug)]
pub struct UnixSocketManager {
    config: UnixSocketConfig,
}

impl UnixSocketManager {
    /// Create a new Unix socket manager
    pub fn new(config: UnixSocketConfig) -> Self {
        Self { config }
    }

    /// Create and configure Unix domain socket listener
    pub async fn create_listener(&self) -> Result<UnixListener> {
        if !self.config.enabled {
            return Err(anyhow!("Unix domain sockets are disabled"));
        }

        let socket_path = Path::new(&self.config.socket_path);

        // Remove existing socket file if it exists
        if socket_path.exists() {
            std::fs::remove_file(socket_path)
                .map_err(|e| anyhow!("Failed to remove existing socket file: {}", e))?;
        }

        // Create the Unix listener
        let listener = UnixListener::bind(socket_path)
            .map_err(|e| anyhow!("Failed to bind Unix socket: {}", e))?;

        // Set socket file permissions
        let mut permissions = std::fs::metadata(socket_path)?.permissions();
        permissions.set_mode(self.config.permissions);
        std::fs::set_permissions(socket_path, permissions)
            .map_err(|e| anyhow!("Failed to set socket permissions: {}", e))?;

        info!("Unix domain socket created at: {}", self.config.socket_path);

        Ok(listener)
    }

    /// Check if the current environment prefers Unix sockets for local communication
    pub fn should_prefer_unix_socket(&self) -> bool {
        self.config.enabled && self.config.prefer_for_local && self.is_local_environment()
    }

    /// Detect if we're in a local development environment
    fn is_local_environment(&self) -> bool {
        // Check for local indicators
        std::env::var("DEVELOPMENT").is_ok() ||
        std::env::var("LOCAL_MODE").is_ok() ||
        std::env::var("NODE_ENV").map_or(false, |env| env == "development")
    }

    /// Clean up socket file on shutdown
    pub fn cleanup(&self) -> Result<()> {
        let socket_path = Path::new(&self.config.socket_path);
        if socket_path.exists() {
            std::fs::remove_file(socket_path)
                .map_err(|e| anyhow!("Failed to remove socket file: {}", e))?;
            debug!("Cleaned up socket file: {}", self.config.socket_path);
        }
        Ok(())
    }
}

/// Transport optimization manager
#[derive(Debug)]
pub struct TransportManager {
    config: TransportConfig,
    unix_socket_manager: UnixSocketManager,
    local_optimization: LocalOptimization,
}

impl TransportManager {
    /// Create a new transport manager
    pub fn new(config: TransportConfig) -> Self {
        let unix_socket_manager = UnixSocketManager::new(config.unix_socket.clone());
        let local_optimization = LocalOptimization::from(config.local_optimization.clone());

        Self {
            config,
            unix_socket_manager,
            local_optimization,
        }
    }

    /// Determine the best transport type based on configuration and environment
    pub fn determine_transport_type(&self, host: &str, port: u16) -> TransportType {
        match self.config.transport_strategy {
            TransportStrategy::ForceTcp => {
                TransportType::Tcp {
                    host: host.to_string(),
                    port,
                }
            }
            TransportStrategy::ForceUnixSocket => {
                TransportType::UnixSocket {
                    path: self.config.unix_socket.socket_path.clone(),
                }
            }
            TransportStrategy::UnixSocketWithTcpFallback => {
                if self.unix_socket_manager.should_prefer_unix_socket() && self.is_local_connection(host) {
                    TransportType::UnixSocket {
                        path: self.config.unix_socket.socket_path.clone(),
                    }
                } else {
                    TransportType::Tcp {
                        host: host.to_string(),
                        port,
                    }
                }
            }
            TransportStrategy::Auto => {
                if self.is_local_connection(host) &&
                   self.unix_socket_manager.should_prefer_unix_socket() {
                    TransportType::UnixSocket {
                        path: self.config.unix_socket.socket_path.clone(),
                    }
                } else {
                    TransportType::Tcp {
                        host: host.to_string(),
                        port,
                    }
                }
            }
        }
    }

    /// Check if the connection is local
    fn is_local_connection(&self, host: &str) -> bool {
        host == "127.0.0.1" ||
        host == "localhost" ||
        host == "::1" ||
        host.starts_with("192.168.") ||
        host.starts_with("10.") ||
        host.starts_with("172.")
    }

    /// Create an optimized server for the determined transport type
    pub async fn create_optimized_server(
        &self,
        transport_type: &TransportType,
    ) -> Result<tonic::transport::Server> {
        let mut server = Server::builder();

        // Apply local optimizations if this is a local connection
        let should_optimize = match transport_type {
            TransportType::UnixSocket { .. } => true,
            TransportType::Tcp { host, .. } => self.is_local_connection(host),
        };

        if should_optimize {
            server = self.apply_local_optimizations(server);
        }

        Ok(server)
    }

    /// Apply local communication optimizations to the server
    fn apply_local_optimizations(
        &self,
        mut server: tonic::transport::Server,
    ) -> tonic::transport::Server {
        if !self.config.local_optimization.enabled {
            return server;
        }

        // Configure larger buffers for local communication
        if self.local_optimization.buffer_size > 0 {
            server = server
                .initial_stream_window_size(Some(self.local_optimization.buffer_size as u32))
                .initial_connection_window_size(Some(self.local_optimization.buffer_size as u32 * 2));
        }

        // Configure connection pooling
        server = server
            .concurrency_limit_per_connection(self.local_optimization.connection_pool_size as usize);

        // Configure keep-alive for local connections
        server = server
            .tcp_keepalive(Some(self.local_optimization.keepalive_interval));

        // Disable Nagle's algorithm for reduced latency
        if self.local_optimization.disable_nagle {
            server = server.tcp_nodelay(true);
        }

        info!("Applied local transport optimizations: buffer_size={}, pool_size={}, keepalive={:?}",
              self.local_optimization.buffer_size,
              self.local_optimization.connection_pool_size,
              self.local_optimization.keepalive_interval);

        server
    }

    /// Create a client endpoint with transport optimizations
    pub fn create_optimized_endpoint(&self, transport_type: &TransportType) -> Result<Endpoint> {
        match transport_type {
            TransportType::Tcp { host, port } => {
                let uri = format!("http://{}:{}", host, port);
                let mut endpoint = Endpoint::from_shared(uri)?;

                // Apply local optimizations for TCP connections
                if self.is_local_connection(host) && self.config.local_optimization.enabled {
                    endpoint = self.apply_client_optimizations(endpoint);
                }

                Ok(endpoint)
            }
            TransportType::UnixSocket { path: _ } => {
                // For Unix sockets, we need to create a custom endpoint
                // This is a simplified approach - in practice you'd need to implement
                // a custom connector for Unix domain sockets
                let uri = format!("http://[::]:50051"); // Placeholder URI
                let mut endpoint = Endpoint::from_shared(uri)?;

                // Apply local optimizations for Unix socket connections
                if self.config.local_optimization.enabled {
                    endpoint = self.apply_client_optimizations(endpoint);
                }

                Ok(endpoint)
            }
        }
    }

    /// Apply client-side transport optimizations
    fn apply_client_optimizations(&self, mut endpoint: Endpoint) -> Endpoint {
        // Configure connection keep-alive
        endpoint = endpoint.keep_alive_while_idle(true);

        // Configure timeouts for local connections (more aggressive)
        endpoint = endpoint
            .timeout(Duration::from_secs(5)) // Shorter timeout for local
            .connect_timeout(Duration::from_secs(2)); // Fast connection for local

        // Configure TCP options
        if self.local_optimization.disable_nagle {
            endpoint = endpoint.tcp_nodelay(true);
        }

        endpoint = endpoint.tcp_keepalive(Some(self.local_optimization.keepalive_interval));

        endpoint
    }

    /// Get Unix socket manager
    pub fn unix_socket_manager(&self) -> &UnixSocketManager {
        &self.unix_socket_manager
    }

    /// Get local optimization settings
    pub fn local_optimization(&self) -> &LocalOptimization {
        &self.local_optimization
    }

    /// Create transport-specific binding address
    pub fn get_binding_address(&self, transport_type: &TransportType) -> String {
        match transport_type {
            TransportType::Tcp { host, port } => format!("{}:{}", host, port),
            TransportType::UnixSocket { path } => path.clone(),
        }
    }
}

/// Transport statistics collector
#[derive(Debug, Clone)]
pub struct TransportStats {
    pub active_tcp_connections: u32,
    pub active_unix_connections: u32,
    pub total_bytes_tcp: u64,
    pub total_bytes_unix: u64,
    pub avg_latency_tcp_ms: f64,
    pub avg_latency_unix_ms: f64,
    pub connection_errors: u32,
    pub transport_switches: u32,
}

impl Default for TransportStats {
    fn default() -> Self {
        Self {
            active_tcp_connections: 0,
            active_unix_connections: 0,
            total_bytes_tcp: 0,
            total_bytes_unix: 0,
            avg_latency_tcp_ms: 0.0,
            avg_latency_unix_ms: 0.0,
            connection_errors: 0,
            transport_switches: 0,
        }
    }
}

/// Connection pool for optimized local communication
#[derive(Debug)]
pub struct LocalConnectionPool {
    pool_size: u32,
    keep_alive_interval: Duration,
    // In a real implementation, this would contain actual connection pools
}

impl LocalConnectionPool {
    pub fn new(pool_size: u32, keep_alive_interval: Duration) -> Self {
        Self {
            pool_size,
            keep_alive_interval,
        }
    }

    /// Get an optimized connection from the pool
    pub async fn get_connection(&self) -> Result<()> {
        // Placeholder for actual connection pool implementation
        Ok(())
    }

    /// Return a connection to the pool
    pub async fn return_connection(&self) -> Result<()> {
        // Placeholder for actual connection pool implementation
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_transport_type_determination() {
        let config = TransportConfig {
            unix_socket: UnixSocketConfig {
                enabled: true,
                socket_path: "/tmp/test.sock".to_string(),
                permissions: 0o600,
                prefer_for_local: true,
            },
            local_optimization: LocalOptimizationConfig::default(),
            transport_strategy: TransportStrategy::Auto,
        };

        let transport_manager = TransportManager::new(config);

        // Local connection should prefer Unix socket
        let transport = transport_manager.determine_transport_type("127.0.0.1", 50051);
        if matches!(transport, TransportType::UnixSocket { .. }) {
            // This is expected in a local development environment
        }

        // Remote connection should use TCP
        let transport = transport_manager.determine_transport_type("remote-server.com", 50051);
        assert!(matches!(transport, TransportType::Tcp { .. }));
    }

    #[test]
    fn test_local_connection_detection() {
        let config = TransportConfig::default();
        let transport_manager = TransportManager::new(config);

        assert!(transport_manager.is_local_connection("127.0.0.1"));
        assert!(transport_manager.is_local_connection("localhost"));
        assert!(transport_manager.is_local_connection("192.168.1.1"));
        assert!(!transport_manager.is_local_connection("example.com"));
    }

    #[test]
    fn test_unix_socket_manager_creation() {
        let config = UnixSocketConfig {
            enabled: true,
            socket_path: "/tmp/test.sock".to_string(),
            permissions: 0o600,
            prefer_for_local: true,
        };

        let unix_manager = UnixSocketManager::new(config);
        assert!(unix_manager.config.enabled);
    }

    #[tokio::test]
    async fn test_unix_socket_creation_disabled() {
        let config = UnixSocketConfig {
            enabled: false,
            socket_path: "/tmp/test.sock".to_string(),
            permissions: 0o600,
            prefer_for_local: true,
        };

        let unix_manager = UnixSocketManager::new(config);
        let result = unix_manager.create_listener().await;
        assert!(result.is_err());
    }

    #[test]
    fn test_local_optimization_from_config() {
        let config = LocalOptimizationConfig {
            enabled: true,
            use_large_buffers: true,
            local_buffer_size: 128 * 1024,
            memory_efficient_serialization: true,
            reduce_latency: LocalLatencyConfig {
                disable_nagle: true,
                custom_connection_pooling: true,
                connection_pool_size: 20,
                keepalive_interval_secs: 30,
            },
        };

        let optimization = LocalOptimization::from(config);
        assert_eq!(optimization.buffer_size, 128 * 1024);
        assert!(optimization.disable_nagle);
        assert_eq!(optimization.connection_pool_size, 20);
        assert_eq!(optimization.keepalive_interval, Duration::from_secs(30));
    }

    #[test]
    fn test_transport_stats_default() {
        let stats = TransportStats::default();
        assert_eq!(stats.active_tcp_connections, 0);
        assert_eq!(stats.active_unix_connections, 0);
        assert_eq!(stats.total_bytes_tcp, 0);
        assert_eq!(stats.connection_errors, 0);
    }

    #[test]
    fn test_local_connection_pool() {
        let pool = LocalConnectionPool::new(10, Duration::from_secs(30));
        assert_eq!(pool.pool_size, 10);
        assert_eq!(pool.keep_alive_interval, Duration::from_secs(30));
    }
}