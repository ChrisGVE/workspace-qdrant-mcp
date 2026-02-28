//! Server process management
//!
//! Handles spawning LSP server processes with stdio transport,
//! process lifecycle (start, stop, shutdown), and accessor methods.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use tokio::process::{Child, Command};
use tokio::sync::{Mutex, RwLock, oneshot};
use tokio::time::timeout;
use tracing::{debug, info};
use uuid::Uuid;

use crate::lsp::{DetectedServer, JsonRpcClient, Language, LspConfig, LspResult};

use super::{HealthMetrics, RestartPolicy, ServerMetadata, ServerStatus};

/// A running LSP server instance
pub struct ServerInstance {
    pub(super) metadata: ServerMetadata,
    pub(super) process: Arc<Mutex<Option<Child>>>,
    pub(super) rpc_client: Arc<JsonRpcClient>,
    pub(super) health_metrics: Arc<RwLock<HealthMetrics>>,
    pub(super) shutdown_signal: Arc<Mutex<Option<oneshot::Sender<()>>>>,
    pub(super) restart_policy: RestartPolicy,
    pub(super) config: LspConfig,
}

impl ServerInstance {
    /// Create a new server instance from detected server info
    pub async fn new(detected: DetectedServer, config: LspConfig) -> LspResult<Self> {
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
            working_directory: std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/")),
            environment: HashMap::new(),
            arguments: Vec::new(),
        };

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

        // Add standard LSP arguments based on server type
        self.configure_server_args(&mut command);

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

        let child = command.spawn()?;

        Ok(child)
    }

    /// Configure server-specific command line arguments
    fn configure_server_args(&self, command: &mut Command) {
        match self.metadata.name.as_str() {
            "rust-analyzer" | "ruff-lsp" | "pylsp" | "gopls" => {
                // These servers use stdio by default
            }
            "typescript-language-server" => {
                command.args(["--stdio"]);
            }
            "clangd" => {
                command.args(["--background-index", "--clang-tidy"]);
            }
            _ => {
                // Default to stdio for unknown servers
                command.args(["--stdio"]);
            }
        }
    }

    /// Stop the server process
    pub(super) async fn stop_process(&mut self) -> LspResult<()> {
        debug!("Stopping LSP server process: {}", self.metadata.name);

        // Update status
        {
            let mut metrics = self.health_metrics.write().await;
            metrics.status = ServerStatus::Stopping;
        }

        // Send shutdown request if RPC client is connected
        let _ = timeout(
            Duration::from_secs(5),
            self.rpc_client
                .send_request("shutdown", serde_json::json!(null)),
        )
        .await;

        // Send exit notification
        let _ = self
            .rpc_client
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

    /// Get health metrics (read-write access for manager/restart logic)
    pub fn health_metrics(&self) -> &Arc<RwLock<HealthMetrics>> {
        &self.health_metrics
    }

    /// Get restart policy (for monitoring/debugging)
    pub fn restart_policy(&self) -> &RestartPolicy {
        &self.restart_policy
    }

    /// Get mutable restart policy (for tests and manager)
    pub fn restart_policy_mut(&mut self) -> &mut RestartPolicy {
        &mut self.restart_policy
    }

    /// Check if the server process is still alive
    pub async fn is_alive(&self) -> bool {
        let mut process_guard = self.process.lock().await;
        if let Some(ref mut child) = *process_guard {
            match child.try_wait() {
                Ok(None) => true, // Still running
                Ok(Some(_)) => {
                    debug!("LSP server {} has exited", self.metadata.name);
                    false
                }
                Err(e) => {
                    debug!(
                        "Error checking LSP server {} status: {}",
                        self.metadata.name, e
                    );
                    false
                }
            }
        } else {
            false
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lsp::detection::ServerCapabilities;

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
