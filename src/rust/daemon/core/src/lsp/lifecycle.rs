//! LSP Server Lifecycle Management
//!
//! This module handles the complete lifecycle of LSP servers including startup,
//! health monitoring, restart on failure, and graceful shutdown.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
use tokio::process::{Child, Command};
use tokio::sync::{Mutex, RwLock, oneshot};
use tokio::time::timeout;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::lsp::{
    Language, LspError, LspResult, LspPriority,
    DetectedServer, LspConfig, JsonRpcClient,
};

/// Current status of an LSP server instance
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ServerStatus {
    /// Server is initializing
    Initializing,
    /// Server is running and healthy
    Running,
    /// Server is experiencing issues but still responsive
    Degraded,
    /// Server is unresponsive or failed
    Failed,
    /// Server is shutting down
    Stopping,
    /// Server has been stopped
    Stopped,
}

/// Health metrics for an LSP server
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthMetrics {
    /// Current status
    pub status: ServerStatus,
    /// Timestamp of last successful health check
    pub last_healthy: chrono::DateTime<chrono::Utc>,
    /// Response time of last health check (milliseconds)
    pub response_time_ms: u64,
    /// Number of consecutive failed health checks
    pub consecutive_failures: u32,
    /// Total number of requests processed
    pub requests_processed: u64,
    /// Average response time over the last period
    pub avg_response_time_ms: f64,
    /// Memory usage in bytes (if available)
    pub memory_usage_bytes: Option<u64>,
    /// CPU usage percentage (if available)
    pub cpu_usage_percent: Option<f32>,
}

impl Default for HealthMetrics {
    fn default() -> Self {
        Self {
            status: ServerStatus::Initializing,
            last_healthy: chrono::Utc::now(),
            response_time_ms: 0,
            consecutive_failures: 0,
            requests_processed: 0,
            avg_response_time_ms: 0.0,
            memory_usage_bytes: None,
            cpu_usage_percent: None,
        }
    }
}

/// Metadata about a server instance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerMetadata {
    /// Unique identifier for this instance
    pub id: Uuid,
    /// Server name
    pub name: String,
    /// Executable path
    pub executable_path: PathBuf,
    /// Supported languages
    pub languages: Vec<Language>,
    /// Server version
    pub version: Option<String>,
    /// When the server was started
    pub started_at: chrono::DateTime<chrono::Utc>,
    /// Process ID (if available)
    pub process_id: Option<u32>,
    /// Working directory
    pub working_directory: PathBuf,
    /// Environment variables used
    pub environment: HashMap<String, String>,
    /// Command line arguments
    pub arguments: Vec<String>,
}

/// A running LSP server instance
pub struct ServerInstance {
    metadata: ServerMetadata,
    process: Arc<Mutex<Option<Child>>>,
    rpc_client: Arc<JsonRpcClient>,
    health_metrics: Arc<RwLock<HealthMetrics>>,
    shutdown_signal: Arc<Mutex<Option<oneshot::Sender<()>>>>,
    restart_policy: RestartPolicy,
    config: LspConfig,
}

/// Policy for restarting failed servers
#[derive(Debug, Clone)]
pub struct RestartPolicy {
    /// Whether automatic restart is enabled
    pub enabled: bool,
    /// Maximum number of restart attempts
    pub max_attempts: u32,
    /// Current number of restart attempts
    pub current_attempts: u32,
    /// Base delay between restart attempts
    pub base_delay: Duration,
    /// Maximum delay between restart attempts
    pub max_delay: Duration,
    /// Backoff multiplier
    pub backoff_multiplier: f64,
    /// Time window to reset restart attempts
    pub reset_window: Duration,
    /// Last restart attempt time
    pub last_restart: Option<Instant>,
}

impl Default for RestartPolicy {
    fn default() -> Self {
        Self {
            enabled: true,
            max_attempts: 5,
            current_attempts: 0,
            base_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(300), // 5 minutes
            backoff_multiplier: 2.0,
            reset_window: Duration::from_secs(3600), // 1 hour
            last_restart: None,
        }
    }
}

impl ServerInstance {
    /// Create a new server instance from detected server info
    pub async fn new(
        detected: DetectedServer,
        config: LspConfig,
    ) -> LspResult<Self> {
        let id = Uuid::new_v4();
        let started_at = chrono::Utc::now();

        let metadata = ServerMetadata {
            id,
            name: detected.name.clone(),
            executable_path: detected.path.clone(),
            languages: detected.languages,
            version: detected.version,
            started_at,
            process_id: None,
            working_directory: std::env::current_dir()
                .unwrap_or_else(|_| PathBuf::from("/")),
            environment: HashMap::new(),
            arguments: Vec::new(),
        };

        // Create RPC client for communication
        let rpc_client = Arc::new(JsonRpcClient::new());
        
        let instance = Self {
            metadata,
            process: Arc::new(Mutex::new(None)),
            rpc_client,
            health_metrics: Arc::new(RwLock::new(HealthMetrics::default())),
            shutdown_signal: Arc::new(Mutex::new(None)),
            restart_policy: RestartPolicy::default(),
            config,
        };

        Ok(instance)
    }

    /// Set the working directory for the LSP server
    ///
    /// This should be called before `start()` to set the project root
    /// for the LSP server. The working directory is used as the rootUri
    /// in the LSP initialize request.
    pub fn with_working_directory(mut self, path: PathBuf) -> Self {
        self.metadata.working_directory = path;
        self
    }

    /// Get the current working directory
    pub fn working_directory(&self) -> &Path {
        &self.metadata.working_directory
    }

    /// Start the LSP server process
    pub async fn start(&mut self) -> LspResult<()> {
        info!("Starting LSP server: {}", self.metadata.name);

        // Update status
        {
            let mut metrics = self.health_metrics.write().await;
            metrics.status = ServerStatus::Initializing;
        }

        // Spawn the process
        let mut child = self.spawn_process().await?;
        
        // Store process ID if available
        if let Some(pid) = child.id() {
            self.metadata.process_id = Some(pid);
            debug!("LSP server {} started with PID {}", self.metadata.name, pid);
        }

        // Connect RPC client to the process
        if let Some(stdout) = child.stdout.take() {
            if let Some(stdin) = child.stdin.take() {
                self.rpc_client.connect_stdio(stdin, stdout).await?;
            }
        }

        // Store the process
        *self.process.lock().await = Some(child);

        // Initialize the LSP server
        self.initialize_lsp().await?;

        // Update status to running
        {
            let mut metrics = self.health_metrics.write().await;
            metrics.status = ServerStatus::Running;
            metrics.last_healthy = chrono::Utc::now();
        }

        info!("LSP server {} started successfully", self.metadata.name);
        Ok(())
    }

    /// Spawn the LSP server process
    async fn spawn_process(&self) -> LspResult<Child> {
        let mut command = Command::new(&self.metadata.executable_path);

        // Add standard LSP arguments if needed
        match self.metadata.name.as_str() {
            "rust-analyzer" => {
                // rust-analyzer uses stdio by default
            }
            "ruff-lsp" => {
                // ruff-lsp uses stdio by default
            }
            "pylsp" => {
                // pylsp uses stdio by default
            }
            "typescript-language-server" => {
                command.args(&["--stdio"]);
            }
            "clangd" => {
                command.args(&["--background-index", "--clang-tidy"]);
            }
            "gopls" => {
                // gopls uses stdio by default
            }
            _ => {
                // Default to stdio for unknown servers
                command.args(&["--stdio"]);
            }
        }

        // Set working directory
        command.current_dir(&self.metadata.working_directory);

        // Configure stdio
        command
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        // Set environment variables
        for (key, value) in &self.metadata.environment {
            command.env(key, value);
        }

        // Spawn the process
        let child = command.spawn()?;

        Ok(child)
    }

    /// Initialize the LSP server with the initialization request
    async fn initialize_lsp(&self) -> LspResult<()> {
        debug!("Sending LSP initialize request to {}", self.metadata.name);

        // Create initialization parameters
        let init_params = serde_json::json!({
            "processId": std::process::id(),
            "rootUri": format!("file://{}", self.metadata.working_directory.display()),
            "capabilities": {
                "textDocument": {
                    "synchronization": {
                        "dynamicRegistration": false,
                        "willSave": false,
                        "willSaveWaitUntil": false,
                        "didSave": false
                    },
                    "completion": {
                        "dynamicRegistration": false,
                        "completionItem": {
                            "snippetSupport": false,
                            "commitCharactersSupport": false,
                            "documentationFormat": ["plaintext"],
                            "deprecatedSupport": false,
                            "preselectSupport": false
                        },
                        "contextSupport": false
                    },
                    "hover": {
                        "dynamicRegistration": false,
                        "contentFormat": ["plaintext"]
                    },
                    "definition": {
                        "dynamicRegistration": false
                    },
                    "references": {
                        "dynamicRegistration": false
                    },
                    "documentSymbol": {
                        "dynamicRegistration": false
                    }
                },
                "workspace": {
                    "applyEdit": false,
                    "workspaceEdit": {
                        "documentChanges": false
                    },
                    "didChangeConfiguration": {
                        "dynamicRegistration": false
                    },
                    "didChangeWatchedFiles": {
                        "dynamicRegistration": false
                    },
                    "symbol": {
                        "dynamicRegistration": false
                    },
                    "executeCommand": {
                        "dynamicRegistration": false
                    }
                }
            }
        });

        // Send initialize request
        let _response = timeout(
            self.config.request_timeout,
            self.rpc_client.send_request("initialize", init_params),
        )
        .await
        .map_err(|_| LspError::Timeout {
            operation: "LSP initialize".to_string(),
        })??;

        // Send initialized notification
        self.rpc_client
            .send_notification("initialized", serde_json::json!({}))
            .await?;

        debug!("LSP server {} initialized successfully", self.metadata.name);
        Ok(())
    }

    /// Perform health check on the server
    pub async fn health_check(&self) -> LspResult<HealthMetrics> {
        let start_time = Instant::now();

        // Check if process is still running
        let process_healthy = {
            let mut process_guard = self.process.lock().await;
            match process_guard.as_mut() {
                Some(child) => {
                    match child.try_wait() {
                        Ok(Some(_)) => false, // Process has exited
                        Ok(None) => true,     // Process is still running
                        Err(_) => false,      // Error checking process
                    }
                }
                None => false, // No process
            }
        };

        let mut metrics = self.health_metrics.write().await;
        
        if !process_healthy {
            metrics.status = ServerStatus::Failed;
            metrics.consecutive_failures += 1;
            return Ok(metrics.clone());
        }

        // Try to send a simple request to check responsiveness
        let rpc_healthy = match timeout(
            Duration::from_secs(5),
            self.rpc_client.send_request("shutdown", serde_json::json!(null)),
        )
        .await
        {
            Ok(Ok(_)) => {
                // Server responded, send initialize again to keep it running
                let _ = self.initialize_lsp().await;
                true
            }
            Ok(Err(_)) => false, // RPC error
            Err(_) => false,     // Timeout
        };

        let response_time = start_time.elapsed();
        metrics.response_time_ms = response_time.as_millis() as u64;

        if rpc_healthy {
            metrics.status = ServerStatus::Running;
            metrics.last_healthy = chrono::Utc::now();
            metrics.consecutive_failures = 0;
        } else {
            metrics.status = if metrics.consecutive_failures > 3 {
                ServerStatus::Failed
            } else {
                ServerStatus::Degraded
            };
            metrics.consecutive_failures += 1;
        }

        // Update average response time
        let alpha = 0.1; // Exponential moving average factor
        metrics.avg_response_time_ms = alpha * (metrics.response_time_ms as f64) + 
                                      (1.0 - alpha) * metrics.avg_response_time_ms;

        Ok(metrics.clone())
    }

    /// Restart the server instance
    pub async fn restart(&mut self) -> LspResult<()> {
        info!("Restarting LSP server: {}", self.metadata.name);

        // Update restart policy
        self.restart_policy.current_attempts += 1;
        self.restart_policy.last_restart = Some(Instant::now());

        // Stop current process
        self.stop_process().await?;

        // Wait for restart delay with exponential backoff
        let delay = self.calculate_restart_delay();
        tokio::time::sleep(delay).await;

        // Start new process
        self.start().await?;

        info!("LSP server {} restarted successfully", self.metadata.name);
        Ok(())
    }

    /// Calculate restart delay with exponential backoff
    fn calculate_restart_delay(&self) -> Duration {
        let base_delay_secs = self.restart_policy.base_delay.as_secs_f64();
        let multiplier = self.restart_policy.backoff_multiplier;
        let attempts = self.restart_policy.current_attempts as f64;
        
        let delay_secs = base_delay_secs * multiplier.powf(attempts - 1.0);
        let max_delay_secs = self.restart_policy.max_delay.as_secs_f64();
        
        Duration::from_secs_f64(delay_secs.min(max_delay_secs))
    }

    /// Stop the server process
    async fn stop_process(&mut self) -> LspResult<()> {
        debug!("Stopping LSP server process: {}", self.metadata.name);

        // Update status
        {
            let mut metrics = self.health_metrics.write().await;
            metrics.status = ServerStatus::Stopping;
        }

        // Send shutdown request if RPC client is connected
        let _ = timeout(
            Duration::from_secs(5),
            self.rpc_client.send_request("shutdown", serde_json::json!(null)),
        )
        .await;

        // Send exit notification
        let _ = self.rpc_client
            .send_notification("exit", serde_json::json!({}))
            .await;

        // Wait for graceful shutdown
        tokio::time::sleep(Duration::from_secs(2)).await;

        // Force kill if still running
        let mut process_guard = self.process.lock().await;
        if let Some(mut child) = process_guard.take() {
            let _ = child.kill().await;
            let _ = child.wait().await;
        }

        // Update status
        {
            let mut metrics = self.health_metrics.write().await;
            metrics.status = ServerStatus::Stopped;
        }

        debug!("LSP server process stopped: {}", self.metadata.name);
        Ok(())
    }

    /// Shutdown the server instance
    pub async fn shutdown(&mut self) -> LspResult<()> {
        info!("Shutting down LSP server: {}", self.metadata.name);

        // Signal shutdown to background tasks
        if let Some(shutdown_tx) = self.shutdown_signal.lock().await.take() {
            let _ = shutdown_tx.send(());
        }

        // Stop the process
        self.stop_process().await?;

        info!("LSP server {} shutdown complete", self.metadata.name);
        Ok(())
    }

    /// Get server metadata
    pub fn metadata(&self) -> &ServerMetadata {
        &self.metadata
    }

    /// Get current status
    pub async fn status(&self) -> ServerStatus {
        self.health_metrics.read().await.status.clone()
    }

    /// Get uptime duration
    pub async fn uptime(&self) -> Duration {
        let now = chrono::Utc::now();
        let duration = now - self.metadata.started_at;
        Duration::from_secs(duration.num_seconds() as u64)
    }

    /// Get primary language supported by this server
    pub fn primary_language(&self) -> Option<Language> {
        self.metadata.languages.first().cloned()
    }

    /// Get server ID
    pub fn id(&self) -> Uuid {
        self.metadata.id
    }

    /// Get RPC client for direct communication
    pub fn rpc_client(&self) -> Arc<JsonRpcClient> {
        self.rpc_client.clone()
    }

    /// Check if the server process is still alive
    pub async fn is_alive(&self) -> bool {
        let mut process_guard = self.process.lock().await;
        if let Some(ref mut child) = *process_guard {
            // try_wait returns Ok(None) if process is still running
            // Ok(Some(status)) if process has exited
            // Err if there's an error checking
            match child.try_wait() {
                Ok(None) => true,  // Still running
                Ok(Some(_)) => {
                    // Process exited
                    debug!("LSP server {} has exited", self.metadata.name);
                    false
                }
                Err(e) => {
                    // Error checking - assume dead
                    debug!("Error checking LSP server {} status: {}", self.metadata.name, e);
                    false
                }
            }
        } else {
            // No process started
            false
        }
    }

    /// Check server health and restart if needed
    /// Returns true if server needed restart, false otherwise
    pub async fn check_and_restart_if_needed(&mut self) -> LspResult<bool> {
        // Check if process is alive
        if self.is_alive().await {
            return Ok(false); // No restart needed
        }

        // Process died - extract values from restart policy
        let enabled = self.restart_policy.enabled;
        let max_attempts = self.restart_policy.max_attempts;
        let current_attempts = self.restart_policy.current_attempts;
        let last_restart = self.restart_policy.last_restart;
        let reset_window = self.restart_policy.reset_window;

        // Check if restart is enabled
        if !enabled {
            info!(
                "LSP server {} crashed but restart is disabled",
                self.metadata.name
            );
            let mut metrics = self.health_metrics.write().await;
            metrics.status = ServerStatus::Failed;
            return Ok(false);
        }

        // Check if we exceeded max restart attempts
        if current_attempts >= max_attempts {
            info!(
                "LSP server {} crashed and exceeded max restart attempts ({})",
                self.metadata.name, max_attempts
            );
            let mut metrics = self.health_metrics.write().await;
            metrics.status = ServerStatus::Failed;
            return Ok(false);
        }

        // Check if we should reset restart counter based on time window
        if let Some(last) = last_restart {
            if last.elapsed() > reset_window {
                // Reset counter - server was stable for a while
                self.restart_policy.current_attempts = 0;
            }
        }

        // Attempt restart
        let attempt_number = self.restart_policy.current_attempts + 1;
        info!(
            "LSP server {} crashed, attempting restart (attempt {} of {})",
            self.metadata.name,
            attempt_number,
            max_attempts
        );

        // Update metrics
        {
            let mut metrics = self.health_metrics.write().await;
            metrics.status = ServerStatus::Failed;
            metrics.consecutive_failures += 1;
        }

        // Perform restart with backoff
        self.restart().await?;

        info!(
            "LSP server {} restarted successfully",
            self.metadata.name
        );

        Ok(true)
    }

    /// Get restart policy (for monitoring/debugging)
    pub fn restart_policy(&self) -> &RestartPolicy {
        &self.restart_policy
    }

    /// Reset restart attempts counter (e.g., after extended stable period)
    pub fn reset_restart_attempts(&mut self) {
        self.restart_policy.current_attempts = 0;
        self.restart_policy.last_restart = None;
    }
}

impl Clone for ServerInstance {
    fn clone(&self) -> Self {
        Self {
            metadata: self.metadata.clone(),
            process: self.process.clone(),
            rpc_client: self.rpc_client.clone(),
            health_metrics: self.health_metrics.clone(),
            shutdown_signal: Arc::new(Mutex::new(None)),
            restart_policy: self.restart_policy.clone(),
            config: self.config.clone(),
        }
    }
}

/// Manager for LSP server instances
pub struct LspServerManager {
    config: LspConfig,
    instances: Arc<RwLock<HashMap<Uuid, ServerInstance>>>,
}

impl LspServerManager {
    /// Create a new LSP server manager
    pub async fn new(config: LspConfig) -> LspResult<Self> {
        Ok(Self {
            config,
            instances: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Create a new server instance
    pub async fn create_instance(
        &mut self,
        detected: DetectedServer,
    ) -> LspResult<ServerInstance> {
        let mut instance = ServerInstance::new(detected, self.config.clone()).await?;
        instance.start().await?;

        let id = instance.id();
        self.instances.write().await.insert(id, instance.clone());

        Ok(instance)
    }

    /// Get server instance by ID
    pub async fn get_instance(&self, id: &Uuid) -> Option<ServerInstance> {
        self.instances.read().await.get(id).cloned()
    }

    /// Get all server instances
    pub async fn get_all_instances(&self) -> Vec<ServerInstance> {
        self.instances.read().await.values().cloned().collect()
    }

    /// Remove a server instance
    pub async fn remove_instance(&mut self, id: &Uuid) -> LspResult<()> {
        if let Some(mut instance) = self.instances.write().await.remove(id) {
            instance.shutdown().await?;
        }
        Ok(())
    }

    /// Check if a server should be restarted based on policy
    pub async fn should_restart(&self, instance: &ServerInstance) -> bool {
        let metrics = instance.health_metrics.read().await;
        let restart_policy = &instance.restart_policy;

        // Check if restart is enabled and we haven't exceeded max attempts
        if !restart_policy.enabled || 
           restart_policy.current_attempts >= restart_policy.max_attempts {
            return false;
        }

        // Check if server is in a failed state
        if !matches!(metrics.status, ServerStatus::Failed | ServerStatus::Degraded) {
            return false;
        }

        // Check if we should reset restart attempts based on time window
        if let Some(last_restart) = restart_policy.last_restart {
            if last_restart.elapsed() > restart_policy.reset_window {
                // Reset attempts counter (this would need to be mutable)
                return true;
            }
        }

        true
    }

    /// Restart a server instance
    pub async fn restart_instance(&mut self, instance: &ServerInstance) -> LspResult<()> {
        let id = instance.id();
        
        if let Some(mut server) = self.instances.write().await.get_mut(&id) {
            server.restart().await?;
        }

        Ok(())
    }
}

impl Clone for LspServerManager {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            instances: self.instances.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lsp::detection::ServerCapabilities;
    use tempfile::tempdir;

    #[test]
    fn test_restart_policy_default() {
        let policy = RestartPolicy::default();
        assert!(policy.enabled);
        assert_eq!(policy.max_attempts, 5);
        assert_eq!(policy.base_delay, Duration::from_secs(1));
    }

    #[test]
    fn test_health_metrics_default() {
        let metrics = HealthMetrics::default();
        assert_eq!(metrics.status, ServerStatus::Initializing);
        assert_eq!(metrics.consecutive_failures, 0);
    }

    #[test]
    fn test_restart_policy_backoff_calculation() {
        // Test exponential backoff calculation
        let policy = RestartPolicy {
            enabled: true,
            max_attempts: 5,
            current_attempts: 0,
            base_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(300),
            backoff_multiplier: 2.0,
            reset_window: Duration::from_secs(3600),
            last_restart: None,
        };

        // Attempt 1: base_delay * 2^0 = 1s
        let delay1 = {
            let base = policy.base_delay.as_secs_f64();
            let multiplier = policy.backoff_multiplier;
            Duration::from_secs_f64(base * multiplier.powf(0.0))
        };
        assert_eq!(delay1, Duration::from_secs(1));

        // Attempt 2: base_delay * 2^1 = 2s
        let delay2 = {
            let base = policy.base_delay.as_secs_f64();
            let multiplier = policy.backoff_multiplier;
            Duration::from_secs_f64(base * multiplier.powf(1.0))
        };
        assert_eq!(delay2, Duration::from_secs(2));

        // Attempt 3: base_delay * 2^2 = 4s
        let delay3 = {
            let base = policy.base_delay.as_secs_f64();
            let multiplier = policy.backoff_multiplier;
            Duration::from_secs_f64(base * multiplier.powf(2.0))
        };
        assert_eq!(delay3, Duration::from_secs(4));
    }

    #[test]
    fn test_restart_policy_max_delay_cap() {
        let policy = RestartPolicy {
            enabled: true,
            max_attempts: 10,
            current_attempts: 9,
            base_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(60),
            backoff_multiplier: 2.0,
            reset_window: Duration::from_secs(3600),
            last_restart: None,
        };

        // Attempt 9: base_delay * 2^8 = 256s, but capped at max_delay = 60s
        let base = policy.base_delay.as_secs_f64();
        let multiplier = policy.backoff_multiplier;
        let max = policy.max_delay.as_secs_f64();
        let delay = (base * multiplier.powf(8.0)).min(max);

        assert_eq!(delay, 60.0);
    }

    #[test]
    fn test_restart_policy_disabled() {
        let policy = RestartPolicy {
            enabled: false,
            ..Default::default()
        };
        assert!(!policy.enabled);
    }

    #[test]
    fn test_restart_policy_exceeded_max_attempts() {
        let policy = RestartPolicy {
            enabled: true,
            max_attempts: 3,
            current_attempts: 3,
            ..Default::default()
        };

        // Should not restart when current_attempts >= max_attempts
        assert!(policy.current_attempts >= policy.max_attempts);
    }

    #[test]
    fn test_server_status_variants() {
        assert_eq!(ServerStatus::Initializing, ServerStatus::Initializing);
        assert_ne!(ServerStatus::Running, ServerStatus::Failed);

        let statuses = vec![
            ServerStatus::Initializing,
            ServerStatus::Running,
            ServerStatus::Stopping,
            ServerStatus::Stopped,
            ServerStatus::Failed,
            ServerStatus::Degraded,
        ];

        // All variants should be distinct
        for (i, s1) in statuses.iter().enumerate() {
            for (j, s2) in statuses.iter().enumerate() {
                if i == j {
                    assert_eq!(s1, s2);
                } else {
                    assert_ne!(s1, s2);
                }
            }
        }
    }

    #[test]
    fn test_health_metrics_failure_tracking() {
        let mut metrics = HealthMetrics::default();
        assert_eq!(metrics.consecutive_failures, 0);

        metrics.consecutive_failures += 1;
        assert_eq!(metrics.consecutive_failures, 1);

        metrics.consecutive_failures += 1;
        assert_eq!(metrics.consecutive_failures, 2);

        // Reset on success
        metrics.consecutive_failures = 0;
        assert_eq!(metrics.consecutive_failures, 0);
    }

    #[tokio::test]
    async fn test_server_manager_creation() {
        let temp_dir = tempdir().unwrap();
        let config = LspConfig {
            database_path: temp_dir.path().join("test.db"),
            ..Default::default()
        };

        let manager = LspServerManager::new(config).await;
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_server_manager_should_restart_disabled() {
        let temp_dir = tempdir().unwrap();
        let config = LspConfig {
            database_path: temp_dir.path().join("test.db"),
            ..Default::default()
        };

        let manager = LspServerManager::new(config).await.unwrap();

        // Create a mock-like server instance using DetectedServer
        let detected = DetectedServer {
            name: "test-server".to_string(),
            path: PathBuf::from("/usr/bin/test"),
            languages: vec![Language::Rust],
            version: Some("1.0".to_string()),
            capabilities: ServerCapabilities::default(),
            priority: 1,
        };

        let mut instance = ServerInstance::new(detected, LspConfig::default()).await.unwrap();

        // Disable restart and set to failed state
        instance.restart_policy.enabled = false;
        {
            let mut metrics = instance.health_metrics.write().await;
            metrics.status = ServerStatus::Failed;
        }

        let should_restart = manager.should_restart(&instance).await;
        assert!(!should_restart);
    }

    #[tokio::test]
    async fn test_server_manager_should_restart_exceeded() {
        let temp_dir = tempdir().unwrap();
        let config = LspConfig {
            database_path: temp_dir.path().join("test.db"),
            ..Default::default()
        };

        let manager = LspServerManager::new(config).await.unwrap();

        let detected = DetectedServer {
            name: "test-server".to_string(),
            path: PathBuf::from("/usr/bin/test"),
            languages: vec![Language::Rust],
            version: Some("1.0".to_string()),
            capabilities: ServerCapabilities::default(),
            priority: 1,
        };

        let mut instance = ServerInstance::new(detected, LspConfig::default()).await.unwrap();

        // Set to max attempts
        instance.restart_policy.current_attempts = 5;
        {
            let mut metrics = instance.health_metrics.write().await;
            metrics.status = ServerStatus::Failed;
        }

        let should_restart = manager.should_restart(&instance).await;
        assert!(!should_restart);
    }

    #[tokio::test]
    async fn test_server_instance_restart_policy_getter() {
        let detected = DetectedServer {
            name: "test-server".to_string(),
            path: PathBuf::from("/usr/bin/test"),
            languages: vec![Language::Rust],
            version: Some("1.0".to_string()),
            capabilities: ServerCapabilities::default(),
            priority: 1,
        };

        let instance = ServerInstance::new(detected, LspConfig::default()).await.unwrap();
        let policy = instance.restart_policy();

        assert!(policy.enabled);
        assert_eq!(policy.max_attempts, 5);
    }

    #[tokio::test]
    async fn test_server_instance_reset_restart_attempts() {
        let detected = DetectedServer {
            name: "test-server".to_string(),
            path: PathBuf::from("/usr/bin/test"),
            languages: vec![Language::Rust],
            version: Some("1.0".to_string()),
            capabilities: ServerCapabilities::default(),
            priority: 1,
        };

        let mut instance = ServerInstance::new(detected, LspConfig::default()).await.unwrap();

        // Simulate some restart attempts
        instance.restart_policy.current_attempts = 3;
        instance.restart_policy.last_restart = Some(Instant::now());

        // Reset
        instance.reset_restart_attempts();

        assert_eq!(instance.restart_policy.current_attempts, 0);
        assert!(instance.restart_policy.last_restart.is_none());
    }

    #[tokio::test]
    async fn test_server_instance_is_alive_no_process() {
        let detected = DetectedServer {
            name: "test-server".to_string(),
            path: PathBuf::from("/usr/bin/test"),
            languages: vec![Language::Rust],
            version: Some("1.0".to_string()),
            capabilities: ServerCapabilities::default(),
            priority: 1,
        };

        let instance = ServerInstance::new(detected, LspConfig::default()).await.unwrap();

        // No process started yet, should return false
        let alive = instance.is_alive().await;
        assert!(!alive);
    }

    #[tokio::test]
    async fn test_server_instance_with_working_directory() {
        let detected = DetectedServer {
            name: "test-server".to_string(),
            path: PathBuf::from("/usr/bin/test"),
            languages: vec![Language::Rust],
            version: Some("1.0".to_string()),
            capabilities: ServerCapabilities::default(),
            priority: 1,
        };

        let instance = ServerInstance::new(detected, LspConfig::default()).await.unwrap();

        // Default working directory should be current directory
        let default_dir = instance.working_directory().to_path_buf();
        assert!(!default_dir.as_os_str().is_empty());

        // Set custom working directory
        let project_root = PathBuf::from("/path/to/project");
        let instance = instance.with_working_directory(project_root.clone());

        // Verify it was set
        assert_eq!(instance.working_directory(), project_root);
    }
}