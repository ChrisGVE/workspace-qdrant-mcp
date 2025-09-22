//! Core daemon implementation for workspace document processing

pub mod core;
pub mod state;
pub mod processing;
pub mod watcher;

use crate::config::DaemonConfig;
use crate::error::{DaemonError, DaemonResult};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

/// Main daemon coordinator
#[derive(Debug, Clone)]
pub struct WorkspaceDaemon {
    config: DaemonConfig,
    state: Arc<RwLock<state::DaemonState>>,
    processing: Arc<processing::DocumentProcessor>,
    watcher: Option<Arc<watcher::FileWatcher>>,
}

impl WorkspaceDaemon {
    /// Create a new daemon instance
    pub async fn new(config: DaemonConfig) -> DaemonResult<Self> {
        // Validate configuration
        config.validate()?;

        info!("Initializing Workspace Daemon with config: {:?}", config);

        // Initialize state management
        let state = Arc::new(RwLock::new(
            state::DaemonState::new(&config.database).await?
        ));

        // Initialize document processor
        let processing = Arc::new(
            processing::DocumentProcessor::new(&config.processing, &config.qdrant).await?
        );

        // Initialize file watcher if enabled
        let watcher = if config.file_watcher.enabled {
            Some(Arc::new(
                watcher::FileWatcher::new(&config.file_watcher, Arc::clone(&processing)).await?
            ))
        } else {
            None
        };

        Ok(Self {
            config,
            state,
            processing,
            watcher,
        })
    }

    /// Start all daemon services
    pub async fn start(&mut self) -> DaemonResult<()> {
        info!("Starting daemon services");

        // Start file watcher if enabled
        if let Some(ref watcher) = self.watcher {
            watcher.start().await?;
            info!("File watcher started");
        }

        info!("All daemon services started successfully");
        Ok(())
    }

    /// Stop all daemon services
    pub async fn stop(&mut self) -> DaemonResult<()> {
        info!("Stopping daemon services");

        // Stop file watcher
        if let Some(ref watcher) = self.watcher {
            watcher.stop().await?;
            info!("File watcher stopped");
        }

        info!("All daemon services stopped");
        Ok(())
    }

    /// Get daemon configuration
    pub fn config(&self) -> &DaemonConfig {
        &self.config
    }

    /// Get daemon state (read-only)
    pub async fn state(&self) -> tokio::sync::RwLockReadGuard<state::DaemonState> {
        self.state.read().await
    }

    /// Get daemon state (read-write)
    pub async fn state_mut(&self) -> tokio::sync::RwLockWriteGuard<state::DaemonState> {
        self.state.write().await
    }

    /// Get document processor
    pub fn processor(&self) -> &Arc<processing::DocumentProcessor> {
        &self.processing
    }

    /// Get file watcher
    pub fn watcher(&self) -> Option<&Arc<watcher::FileWatcher>> {
        self.watcher.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::*;
    use tempfile::TempDir;
    use tokio_test;
    use std::path::PathBuf;
    use uuid::Uuid;

    fn create_test_config() -> DaemonConfig {
        // Use in-memory SQLite database for tests
        let db_path = ":memory:".to_string();

        DaemonConfig {
            server: ServerConfig {
                host: "127.0.0.1".to_string(),
                port: 50051,
                max_connections: 1000,
                connection_timeout_secs: 30,
                request_timeout_secs: 300,
                enable_tls: false,
            },
            database: DatabaseConfig {
                sqlite_path: db_path,
                max_connections: 5,
                connection_timeout_secs: 30,
                enable_wal: true,
            },
            qdrant: QdrantConfig {
                url: "http://localhost:6333".to_string(),
                api_key: None,
                timeout_secs: 30,
                max_retries: 3,
                default_collection: crate::config::CollectionConfig {
                    vector_size: 384,
                    distance_metric: "Cosine".to_string(),
                    enable_indexing: true,
                    replication_factor: 1,
                    shard_number: 1,
                },
            },
            processing: ProcessingConfig {
                max_concurrent_tasks: 4,
                default_chunk_size: 1000,
                default_chunk_overlap: 100,
                max_file_size_bytes: 1000000,
                supported_extensions: vec!["txt".to_string(), "md".to_string()],
                enable_lsp: false,
                lsp_timeout_secs: 10,
            },
            file_watcher: FileWatcherConfig {
                enabled: false,
                debounce_ms: 500,
                max_watched_dirs: 100,
                ignore_patterns: vec![],
                recursive: true,
            },
            metrics: crate::config::MetricsConfig {
                enabled: false,
                collection_interval_secs: 60,
                retention_days: 30,
                enable_prometheus: false,
                prometheus_port: 9090,
            },
            logging: crate::config::LoggingConfig {
                level: "info".to_string(),
                file_path: None,
                json_format: false,
                max_file_size_mb: 100,
                max_files: 5,
            },
        }
    }

    fn create_test_config_with_watcher() -> DaemonConfig {
        let mut config = create_test_config();
        config.file_watcher.enabled = true;
        config
    }

    #[tokio::test]
    async fn test_workspace_daemon_new_success() {
        let config = create_test_config();
        let result = WorkspaceDaemon::new(config).await;

        assert!(result.is_ok());
        let daemon = result.unwrap();
        assert_eq!(daemon.config().database.max_connections, 5);
        assert_eq!(daemon.config().qdrant.url, "http://localhost:6333");
        assert!(daemon.watcher().is_none());
    }

    #[tokio::test]
    async fn test_workspace_daemon_new_with_watcher() {
        let config = create_test_config_with_watcher();
        let result = WorkspaceDaemon::new(config).await;

        assert!(result.is_ok());
        let daemon = result.unwrap();
        assert!(daemon.watcher().is_some());
    }

    #[tokio::test]
    async fn test_workspace_daemon_debug_format() {
        let config = create_test_config();
        let daemon = WorkspaceDaemon::new(config).await.unwrap();

        let debug_str = format!("{:?}", daemon);
        assert!(debug_str.contains("WorkspaceDaemon"));
    }

    #[tokio::test]
    async fn test_daemon_config_access() {
        let config = create_test_config();
        let original_max_connections = config.database.max_connections;
        let daemon = WorkspaceDaemon::new(config).await.unwrap();

        assert_eq!(daemon.config().database.max_connections, original_max_connections);
    }

    #[tokio::test]
    async fn test_daemon_state_access() {
        let config = create_test_config();
        let daemon = WorkspaceDaemon::new(config).await.unwrap();

        let state = daemon.state().await;
        // Test that we can access state
        drop(state);

        let state_mut = daemon.state_mut().await;
        // Test that we can access mutable state
        drop(state_mut);
    }

    #[tokio::test]
    async fn test_daemon_processor_access() {
        let config = create_test_config();
        let daemon = WorkspaceDaemon::new(config).await.unwrap();

        let processor1 = daemon.processor();
        let processor2 = daemon.processor();
        assert!(Arc::ptr_eq(&processor1, &processor2)); // Should be same Arc
    }

    #[tokio::test]
    async fn test_daemon_start_stop_cycle() {
        let config = create_test_config();
        let mut daemon = WorkspaceDaemon::new(config).await.unwrap();

        // Test start
        let start_result = daemon.start().await;
        assert!(start_result.is_ok());

        // Test stop
        let stop_result = daemon.stop().await;
        assert!(stop_result.is_ok());
    }

    #[tokio::test]
    async fn test_daemon_start_stop_with_watcher() {
        let config = create_test_config_with_watcher();
        let mut daemon = WorkspaceDaemon::new(config).await.unwrap();

        // Test start with watcher
        let start_result = daemon.start().await;
        assert!(start_result.is_ok());

        // Test stop with watcher
        let stop_result = daemon.stop().await;
        assert!(stop_result.is_ok());
    }

    #[tokio::test]
    async fn test_daemon_start_disabled_watcher() {
        let config = create_test_config(); // watcher disabled
        let mut daemon = WorkspaceDaemon::new(config).await.unwrap();

        let start_result = daemon.start().await;
        assert!(start_result.is_ok());
    }

    #[tokio::test]
    async fn test_daemon_multiple_start_stop_cycles() {
        let config = create_test_config();
        let mut daemon = WorkspaceDaemon::new(config).await.unwrap();

        // Multiple start/stop cycles
        for _ in 0..3 {
            assert!(daemon.start().await.is_ok());
            assert!(daemon.stop().await.is_ok());
        }
    }

    #[tokio::test]
    async fn test_daemon_watcher_option_handling() {
        // Test with watcher disabled
        let config_disabled = create_test_config();
        let daemon_disabled = WorkspaceDaemon::new(config_disabled).await.unwrap();
        assert!(daemon_disabled.watcher().is_none());

        // Test with watcher enabled
        let config_enabled = create_test_config_with_watcher();
        let daemon_enabled = WorkspaceDaemon::new(config_enabled).await.unwrap();
        assert!(daemon_enabled.watcher().is_some());
    }

    #[tokio::test]
    async fn test_daemon_concurrent_state_access() {
        let config = create_test_config();
        let daemon = Arc::new(WorkspaceDaemon::new(config).await.unwrap());

        let daemon1 = Arc::clone(&daemon);
        let daemon2 = Arc::clone(&daemon);

        let handle1 = tokio::spawn(async move {
            let _state = daemon1.state().await;
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        });

        let handle2 = tokio::spawn(async move {
            let _state = daemon2.state().await;
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        });

        let (r1, r2) = tokio::join!(handle1, handle2);
        assert!(r1.is_ok());
        assert!(r2.is_ok());
    }

    #[tokio::test]
    async fn test_daemon_processor_arc_sharing() {
        let config = create_test_config();
        let daemon = WorkspaceDaemon::new(config).await.unwrap();

        let processor1 = daemon.processor();
        let processor2 = daemon.processor();

        // Both should point to the same Arc<DocumentProcessor>
        assert!(Arc::ptr_eq(processor1, processor2));
    }

    #[tokio::test]
    async fn test_daemon_error_handling_invalid_config() {
        let mut config = create_test_config();

        // Make config invalid by setting empty URL
        config.qdrant.url = String::new();

        let result = WorkspaceDaemon::new(config).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_daemon_struct_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<WorkspaceDaemon>();
        assert_sync::<WorkspaceDaemon>();
    }
}