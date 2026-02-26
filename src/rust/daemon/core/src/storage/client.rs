//! Qdrant client connection management
//!
//! Handles client creation with transport fallback, retry logic with
//! exponential backoff, connection statistics, and daemon-mode output
//! suppression.

use std::collections::HashMap;
use std::env;
use std::sync::Arc;
use std::time::Duration;

use qdrant_client::Qdrant;
use qdrant_client::config::QdrantConfig;
use tokio::time::sleep;
use tracing::{debug, info, warn, error};

use super::config::{StorageConfig, TransportMode};
use super::types::StorageError;

/// Connection pool statistics
#[derive(Debug, Default)]
pub(crate) struct ConnectionStats {
    pub(crate) successful_connections: u64,
    pub(crate) failed_connections: u64,
    pub(crate) active_connections: u32,
    pub(crate) total_requests: u64,
    pub(crate) total_errors: u64,
}

/// Storage client with Qdrant integration
pub struct StorageClient {
    /// Qdrant client instance
    pub(crate) client: Arc<Qdrant>,
    /// Client configuration
    pub(crate) config: StorageConfig,
    /// Connection pool statistics
    pub(crate) stats: Arc<tokio::sync::Mutex<ConnectionStats>>,
}

impl StorageClient {
    /// Create a new storage client with default configuration
    pub fn new() -> Self {
        Self::with_config(StorageConfig::default())
    }

    /// Create a storage client with custom configuration
    pub fn with_config(config: StorageConfig) -> Self {
        // Debug: write to a file to verify code is running
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("/tmp/storage_debug.log")
        {
            use std::io::Write;
            let _ = writeln!(f, "StorageClient::with_config called: url={}, check_compat={}",
                config.url, config.check_compatibility);
        }

        info!("Initializing Qdrant client with transport: {:?}", config.transport);

        let connection_url = build_connection_url(&config);
        info!("Connecting to Qdrant at: {}", connection_url);

        let qdrant_config = build_qdrant_config(&config, &connection_url);

        // Try to build the client with fallback to HTTP on gRPC failure
        // In daemon mode, temporarily suppress stdout/stderr during client creation
        let client = if is_daemon_mode() && !config.check_compatibility {
            suppress_output_temporarily(|| Qdrant::new(qdrant_config.clone()))
        } else {
            Qdrant::new(qdrant_config.clone())
        };

        let client = match client {
            Ok(client) => {
                info!("Successfully created Qdrant client with {:?} transport", config.transport);
                Arc::new(client)
            },
            Err(e) if matches!(config.transport, TransportMode::Grpc) => {
                create_fallback_client(&config, &connection_url, e)
            },
            Err(e) => {
                error!("Failed to build Qdrant client: {}", e);
                panic!("Failed to build Qdrant client: {}", e);
            }
        };

        Self {
            client,
            config,
            stats: Arc::new(tokio::sync::Mutex::new(ConnectionStats::default())),
        }
    }

    /// Test connection to Qdrant server
    pub async fn test_connection(&self) -> Result<bool, StorageError> {
        debug!("Testing connection to Qdrant server: {}", self.config.url);

        match self.client.health_check().await {
            Ok(_) => {
                info!("Successfully connected to Qdrant server");
                self.update_stats(|stats| stats.successful_connections += 1).await;
                Ok(true)
            },
            Err(e) => {
                error!("Failed to connect to Qdrant server: {}", e);
                self.update_stats(|stats| stats.failed_connections += 1).await;
                Err(StorageError::Connection(e.to_string()))
            }
        }
    }

    /// Get connection statistics
    pub async fn get_stats(&self) -> Result<HashMap<String, u64>, StorageError> {
        let stats = self.stats.lock().await;
        let mut result = HashMap::new();

        result.insert("successful_connections".to_string(), stats.successful_connections);
        result.insert("failed_connections".to_string(), stats.failed_connections);
        result.insert("active_connections".to_string(), stats.active_connections as u64);
        result.insert("total_requests".to_string(), stats.total_requests);
        result.insert("total_errors".to_string(), stats.total_errors);

        Ok(result)
    }

    /// Retry operation with exponential backoff
    pub(crate) async fn retry_operation<F, Fut, T>(&self, operation: F) -> Result<T, StorageError>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<T, StorageError>>,
    {
        let mut attempt = 0;
        let max_retries = self.config.max_retries;
        let base_delay = Duration::from_millis(self.config.retry_delay_ms);

        loop {
            match operation().await {
                Ok(result) => {
                    if attempt > 0 {
                        info!("Operation succeeded after {} retries", attempt);
                    }
                    self.update_stats(|stats| stats.total_requests += 1).await;
                    return Ok(result);
                },
                Err(e) => {
                    attempt += 1;
                    self.update_stats(|stats| {
                        stats.total_requests += 1;
                        stats.total_errors += 1;
                    }).await;

                    if attempt >= max_retries {
                        error!("Operation failed after {} attempts: {}", attempt, e);
                        return Err(e);
                    }

                    // Exponential backoff with jitter
                    let delay = base_delay * (2_u32.pow(attempt - 1));
                    let jitter = Duration::from_millis(fastrand::u64(0..=100));
                    let total_delay = delay + jitter;

                    warn!("Operation failed (attempt {}), retrying in {:?}: {}", attempt, total_delay, e);
                    sleep(total_delay).await;
                }
            }
        }
    }

    /// Update connection statistics
    pub(crate) async fn update_stats<F>(&self, update_fn: F)
    where
        F: FnOnce(&mut ConnectionStats),
    {
        if let Ok(mut stats) = self.stats.try_lock() {
            update_fn(&mut stats);
        }
    }
}

impl Default for StorageClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Build the connection URL based on transport mode
fn build_connection_url(config: &StorageConfig) -> String {
    match config.transport {
        TransportMode::Grpc => {
            // For gRPC, use port 6334 (Qdrant's gRPC port)
            // Auto-convert port 6333 to 6334 if specified
            if config.url.contains(":6333") {
                config.url.replace(":6333", ":6334")
            } else if !config.url.contains(":6334") && !config.url.contains(':') {
                format!("{}:6334", config.url.trim_end_matches('/'))
            } else {
                config.url.clone()
            }
        },
        TransportMode::Http => {
            if config.url.starts_with("http://") || config.url.starts_with("https://") {
                config.url.clone()
            } else {
                format!("http://{}", config.url)
            }
        }
    }
}

/// Build the Qdrant client configuration
fn build_qdrant_config(config: &StorageConfig, connection_url: &str) -> QdrantConfig {
    let mut qdrant_config = QdrantConfig::from_url(connection_url);

    // Configure authentication
    if let Some(api_key) = &config.api_key {
        info!("Configuring API key authentication");
        qdrant_config = qdrant_config.api_key(api_key.clone());
    }

    // Configure timeout
    qdrant_config = qdrant_config.timeout(Duration::from_millis(config.timeout_ms));

    // Configure connection timeout for better reliability
    qdrant_config = qdrant_config.connect_timeout(Duration::from_millis(config.timeout_ms / 2));

    // Enable keep-alive for better connection stability
    qdrant_config = qdrant_config.keep_alive_while_idle();

    // Configure compatibility checking - disable in daemon mode for silence
    if !config.check_compatibility {
        qdrant_config = qdrant_config.skip_compatibility_check();
    }

    // Log and apply HTTP/2 configuration for gRPC transport
    if matches!(config.transport, TransportMode::Grpc) {
        log_http2_config(config);

        if config.http2.tcp_keepalive {
            qdrant_config = qdrant_config.keep_alive_while_idle();
        }

        if let Some(_frame_size) = config.http2.max_frame_size {
            warn!("HTTP/2 max frame size configuration requires lower-level gRPC configuration");
            warn!("Consider using HTTP transport if frame size errors persist");
        }
    }

    qdrant_config
}

/// Log HTTP/2 configuration for gRPC transport
fn log_http2_config(config: &StorageConfig) {
    info!("gRPC transport configured with HTTP/2 settings:");
    if let Some(frame_size) = config.http2.max_frame_size {
        info!("  - Max frame size: {} bytes", frame_size);
    }
    if let Some(window_size) = config.http2.initial_window_size {
        info!("  - Initial window size: {} bytes", window_size);
    }
    if let Some(header_size) = config.http2.max_header_list_size {
        info!("  - Max header list size: {} bytes", header_size);
    }
    info!("  - Server push: {}", config.http2.enable_push);
    info!("  - TCP keepalive: {}", config.http2.tcp_keepalive);
}

/// Create a fallback HTTP client when gRPC fails
fn create_fallback_client(config: &StorageConfig, connection_url: &str, original_error: qdrant_client::QdrantError) -> Arc<Qdrant> {
    error!("Failed to create gRPC client: {}", original_error);
    warn!("Attempting fallback to HTTP transport...");

    let fallback_url = connection_url.replace("grpc://", "http://");
    let mut fallback_config = QdrantConfig::from_url(&fallback_url)
        .timeout(Duration::from_millis(config.timeout_ms))
        .connect_timeout(Duration::from_millis(config.timeout_ms / 2))
        .keep_alive_while_idle();

    if let Some(api_key) = &config.api_key {
        fallback_config = fallback_config.api_key(api_key.clone());
    }

    match Qdrant::new(fallback_config) {
        Ok(client) => {
            warn!("Successfully created fallback HTTP client");
            Arc::new(client)
        },
        Err(fallback_error) => {
            error!("Fallback to HTTP also failed: {}", fallback_error);
            panic!(
                "Failed to build Qdrant client with both gRPC and HTTP: original error: {}, fallback error: {}",
                original_error, fallback_error
            );
        }
    }
}

/// Detect if running in daemon mode for MCP stdio compliance
fn is_daemon_mode() -> bool {
    // Primary explicit indicator
    if env::var("WQM_SERVICE_MODE").map(|v| v == "true").unwrap_or(false) {
        return true;
    }

    // macOS LaunchAgent/LaunchDaemon - XPC_SERVICE_NAME is set to "0" in regular
    // terminal sessions, so we check that it's not empty and not "0"
    if let Ok(xpc_name) = env::var("XPC_SERVICE_NAME") {
        if !xpc_name.is_empty() && xpc_name != "0" {
            return true;
        }
    }

    // Linux systemd indicators
    env::var("SYSTEMD_EXEC_PID").is_ok() ||
        env::var("SYSLOG_IDENTIFIER").is_ok() ||
        env::var("LOGNAME").map(|v| v == "root").unwrap_or(false)
}

/// Temporarily suppress stdout and stderr during a function call for MCP compliance
#[cfg(unix)]
fn suppress_output_temporarily<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    use std::fs::OpenOptions;
    use std::os::unix::io::AsRawFd;

    // Save original file descriptors
    let original_stdout = unsafe { libc::dup(libc::STDOUT_FILENO) };
    let original_stderr = unsafe { libc::dup(libc::STDERR_FILENO) };

    let result = if let Ok(null_file) = OpenOptions::new().write(true).open("/dev/null") {
        let null_fd = null_file.as_raw_fd();

        // Redirect stdout and stderr to /dev/null
        unsafe {
            libc::dup2(null_fd, libc::STDOUT_FILENO);
            libc::dup2(null_fd, libc::STDERR_FILENO);
        }

        // Execute the function
        let result = f();

        // Restore original file descriptors
        unsafe {
            libc::dup2(original_stdout, libc::STDOUT_FILENO);
            libc::dup2(original_stderr, libc::STDERR_FILENO);
            libc::close(original_stdout);
            libc::close(original_stderr);
        }

        result
    } else {
        // If we can't open /dev/null, just run the function normally
        f()
    };

    result
}

/// Windows version of output suppression
#[cfg(windows)]
fn suppress_output_temporarily<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    use std::fs::OpenOptions;
    use std::os::windows::io::AsRawHandle;

    if let Ok(null_file) = OpenOptions::new().write(true).open("NUL") {
        unsafe {
            winapi::um::processenv::SetStdHandle(
                winapi::um::winbase::STD_OUTPUT_HANDLE,
                null_file.as_raw_handle() as *mut std::ffi::c_void
            );
            winapi::um::processenv::SetStdHandle(
                winapi::um::winbase::STD_ERROR_HANDLE,
                null_file.as_raw_handle() as *mut std::ffi::c_void
            );
        }
    }

    f()
}
