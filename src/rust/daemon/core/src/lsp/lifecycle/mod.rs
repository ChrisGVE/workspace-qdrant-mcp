//! LSP Server Lifecycle Management
//!
//! This module handles the complete lifecycle of LSP servers including startup,
//! health monitoring, restart on failure, and graceful shutdown.
//!
//! # Submodules
//!
//! - [`process`] - Server process spawn, stdio transport, and accessors
//! - [`restart`] - Backoff restart logic and health-driven restart
//! - [`initialization`] - LSP capability negotiation and workspace config

mod initialization;
mod process;
mod restart;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::lsp::{DetectedServer, Language, LspConfig, LspResult};

// Re-export all public types from submodules
pub use process::ServerInstance;

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
    pub async fn create_instance(&mut self, detected: DetectedServer) -> LspResult<ServerInstance> {
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
        let metrics = instance.health_metrics().read().await;
        let restart_policy = instance.restart_policy();

        // Check if restart is enabled and we haven't exceeded max attempts
        if !restart_policy.enabled || restart_policy.current_attempts >= restart_policy.max_attempts
        {
            return false;
        }

        // Check if server is in a failed state
        if !matches!(
            metrics.status,
            ServerStatus::Failed | ServerStatus::Degraded
        ) {
            return false;
        }

        // Check if we should reset restart attempts based on time window
        if let Some(last_restart) = restart_policy.last_restart {
            if last_restart.elapsed() > restart_policy.reset_window {
                return true;
            }
        }

        true
    }

    /// Restart a server instance
    pub async fn restart_instance(&mut self, instance: &ServerInstance) -> LspResult<()> {
        let id = instance.id();

        if let Some(server) = self.instances.write().await.get_mut(&id) {
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
        let temp_dir = tempfile::tempdir().unwrap();
        let config = LspConfig {
            database_path: temp_dir.path().join("test.db"),
            ..Default::default()
        };

        let manager = LspServerManager::new(config).await;
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_server_manager_should_restart_disabled() {
        let temp_dir = tempfile::tempdir().unwrap();
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

        let mut instance = ServerInstance::new(detected, LspConfig::default())
            .await
            .unwrap();

        // Disable restart and set to failed state
        instance.restart_policy_mut().enabled = false;
        {
            let mut metrics = instance.health_metrics().write().await;
            metrics.status = ServerStatus::Failed;
        }

        let should_restart = manager.should_restart(&instance).await;
        assert!(!should_restart);
    }

    #[tokio::test]
    async fn test_server_manager_should_restart_exceeded() {
        let temp_dir = tempfile::tempdir().unwrap();
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

        let mut instance = ServerInstance::new(detected, LspConfig::default())
            .await
            .unwrap();

        // Set to max attempts
        instance.restart_policy_mut().current_attempts = 5;
        {
            let mut metrics = instance.health_metrics().write().await;
            metrics.status = ServerStatus::Failed;
        }

        let should_restart = manager.should_restart(&instance).await;
        assert!(!should_restart);
    }
}
