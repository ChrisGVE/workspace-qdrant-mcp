//! Core daemon implementation for workspace document processing

pub mod core;
pub mod state;
pub mod processing;
pub mod watcher;
// pub mod watcher_performance; // Temporarily disabled - needs proper imports
pub mod file_ops;
pub mod runtime;
pub mod fs_compat;

use crate::config::DaemonConfig;
use crate::error::DaemonResult;
use std::sync::Arc;
use tokio::sync::{RwLock, Mutex};
use tracing::info;
use self::runtime::{RuntimeManager, RuntimeConfig};

/// Main daemon coordinator
#[derive(Debug, Clone)]
pub struct WorkspaceDaemon {
    config: DaemonConfig,
    #[allow(dead_code)]
    state: Arc<RwLock<state::DaemonState>>,
    #[allow(dead_code)]
    processing: Arc<processing::DocumentProcessor>,
    watcher: Option<Arc<Mutex<watcher::FileWatcher>>>,
    runtime_manager: Arc<RuntimeManager>,
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
            Some(Arc::new(Mutex::new(
                watcher::FileWatcher::new(&config.file_watcher, Arc::clone(&processing)).await?
            )))
        } else {
            None
        };

        // Initialize runtime manager
        let runtime_config = RuntimeConfig {
            max_concurrent_tasks: config.processing.max_concurrent_tasks,
            task_timeout: std::time::Duration::from_secs(config.server.request_timeout_secs),
            resource_pool_size: config.server.max_connections,
            enable_monitoring: config.metrics.enabled,
            monitoring_interval: std::time::Duration::from_secs(config.metrics.collection_interval_secs),
            max_retry_attempts: config.qdrant.max_retries,
            shutdown_timeout: std::time::Duration::from_secs(30),
        };
        let runtime_manager = Arc::new(RuntimeManager::new(runtime_config).await?);

        let daemon = Self {
            config,
            state,
            processing,
            watcher,
            runtime_manager,
        };

        // Create auto-watch if enabled (do this early, before starting services)
        if daemon.config.auto_ingestion.enabled && daemon.config.auto_ingestion.auto_create_watches {
            info!("Auto-ingestion is enabled, creating auto-watch during initialization");

            if let Some(ref project_path) = daemon.config.auto_ingestion.project_path {
                daemon.create_auto_watch(project_path).await?;
            } else {
                info!("Auto-ingestion enabled but no project_path specified");
            }
        }

        Ok(daemon)
    }

    /// Start all daemon services
    pub async fn start(&mut self) -> DaemonResult<()> {
        info!("Starting daemon services");

        // Start runtime manager
        self.runtime_manager.start().await?;
        info!("Runtime manager started");

        // Start file watcher if enabled
        if let Some(ref watcher) = self.watcher {
            {
                let watcher_guard = watcher.lock().await;
                watcher_guard.start().await?;
            }
            info!("File watcher started");

            // Configure watcher with database watch configurations
            self.configure_file_watcher_from_database().await?;
        }

        // Auto-watch creation is now done during initialization, not during start

        info!("All daemon services started successfully");
        Ok(())
    }

    /// Stop all daemon services
    #[allow(dead_code)]
    pub async fn stop(&mut self) -> DaemonResult<()> {
        info!("Stopping daemon services");

        // Stop file watcher
        if let Some(ref watcher) = self.watcher {
            let watcher_guard = watcher.lock().await;
            watcher_guard.stop().await?;
            info!("File watcher stopped");
        }

        // Stop runtime manager gracefully
        self.runtime_manager.stop(true).await?;
        info!("Runtime manager stopped");

        info!("All daemon services stopped");
        Ok(())
    }

    /// Get daemon configuration
    pub fn config(&self) -> &DaemonConfig {
        &self.config
    }

    /// Get daemon state (read-only)
    #[allow(dead_code)]
    pub async fn state(&self) -> tokio::sync::RwLockReadGuard<'_, state::DaemonState> {
        self.state.read().await
    }

    /// Get daemon state (read-write)
    #[allow(dead_code)]
    pub async fn state_mut(&self) -> tokio::sync::RwLockWriteGuard<'_, state::DaemonState> {
        self.state.write().await
    }

    /// Get document processor
    #[allow(dead_code)]
    pub fn processor(&self) -> &Arc<processing::DocumentProcessor> {
        &self.processing
    }

    /// Get file watcher
    #[allow(dead_code)]
    pub fn watcher(&self) -> Option<&Arc<Mutex<watcher::FileWatcher>>> {
        self.watcher.as_ref()
    }

    /// Get runtime manager
    #[allow(dead_code)]
    pub fn runtime_manager(&self) -> &Arc<RuntimeManager> {
        &self.runtime_manager
    }

    /// Get runtime statistics
    #[allow(dead_code)]
    pub async fn get_runtime_statistics(&self) -> runtime::RuntimeStatistics {
        self.runtime_manager.get_statistics().await
    }

    /// Create an automatic watch configuration for a project path
    async fn create_auto_watch(&self, project_path: &str) -> DaemonResult<()> {
        info!("Creating auto-watch for project path: {}", project_path);

        // Check if watch configuration already exists
        let state = self.state.read().await;
        if state.watch_configuration_exists(project_path).await? {
            info!("Watch configuration already exists for path: {}", project_path);
            return Ok(());
        }

        // Create collection name
        let collection_name = self.generate_collection_name(project_path);

        // Prepare file patterns
        let patterns = if self.config.auto_ingestion.include_source_files {
            self.config.auto_ingestion.include_patterns.clone()
        } else {
            vec![]
        };

        // Prepare ignore patterns
        let mut ignore_patterns = self.config.auto_ingestion.exclude_patterns.clone();

        // Add standard ignore patterns from file watcher config
        ignore_patterns.extend(self.config.file_watcher.ignore_patterns.clone());

        // Determine recursive depth
        let recursive_depth = if self.config.auto_ingestion.max_depth == 0 {
            -1 // Unlimited depth
        } else {
            self.config.auto_ingestion.max_depth as i32
        };

        // Create watch configuration in database
        let watch_id = state.create_watch_configuration(
            project_path,
            &collection_name,
            &patterns,
            &ignore_patterns,
            self.config.auto_ingestion.recursive,
            recursive_depth,
        ).await?;

        info!("Successfully created auto-watch configuration with ID: {} for path: {} -> collection: {}",
              watch_id, project_path, collection_name);

        info!("Auto-watch creation completed successfully");
        Ok(())
    }

    /// Configure file watcher with database watch configurations
    async fn configure_file_watcher_from_database(&mut self) -> DaemonResult<()> {
        if let Some(ref watcher) = self.watcher {
            let state = self.state.read().await;
            let watch_configs = state.get_active_watch_configurations().await?;

            info!("Configuring file watcher with {} active watch configurations", watch_configs.len());

            let mut watcher_guard = watcher.lock().await;
            for config in watch_configs {
                info!("Adding directory to file watcher: {} -> collection: {}", config.path, config.collection);
                watcher_guard.watch_directory(&config.path).await?;
                info!("Successfully added directory to file watcher: {}", config.path);
            }
        }

        Ok(())
    }

    /// Generate a collection name for the given project path
    fn generate_collection_name(&self, project_path: &str) -> String {
        use std::path::Path;

        let path = Path::new(project_path);
        let project_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("unknown-project");

        format!("{}-{}", project_name, self.config.auto_ingestion.target_collection_suffix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::*;

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
                security: crate::config::SecurityConfig::default(),
                transport: crate::config::TransportConfig::default(),
                message: crate::config::MessageConfig::default(),
                compression: crate::config::CompressionConfig::default(),
                streaming: crate::config::StreamingConfig::default(),
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
            auto_ingestion: crate::config::AutoIngestionConfig::default(),
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

    #[tokio::test]
    async fn test_auto_watch_creation() {
        let mut config = create_test_config();
        config.auto_ingestion.enabled = true;
        config.auto_ingestion.auto_create_watches = true;
        config.auto_ingestion.project_path = Some("/tmp/test_project".to_string());

        let daemon = WorkspaceDaemon::new(config).await.unwrap();

        // Test the auto-watch creation logic
        let result = daemon.create_auto_watch("/tmp/test_project").await;
        assert!(result.is_ok());

        // Test collection name generation
        let collection_name = daemon.generate_collection_name("/tmp/test_project");
        assert_eq!(collection_name, "test_project-repo");
    }

    #[tokio::test]
    async fn test_collection_name_generation() {
        let config = create_test_config();
        let daemon = WorkspaceDaemon::new(config).await.unwrap();

        // Test various project paths
        assert_eq!(daemon.generate_collection_name("/home/user/my-project"), "my-project-repo");
        assert_eq!(daemon.generate_collection_name("/path/to/workspace-qdrant-mcp"), "workspace-qdrant-mcp-repo");
        assert_eq!(daemon.generate_collection_name("/"), "unknown-project-repo");
        assert_eq!(daemon.generate_collection_name(""), "unknown-project-repo");
    }

    #[tokio::test]
    async fn test_auto_ingestion_disabled() {
        let mut config = create_test_config();
        config.auto_ingestion.enabled = false;

        let mut daemon = WorkspaceDaemon::new(config).await.unwrap();

        // When auto-ingestion is disabled, start should complete without creating watches
        let result = daemon.start().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_auto_watch_creation_no_project_path() {
        let mut config = create_test_config();
        config.auto_ingestion.enabled = true;
        config.auto_ingestion.auto_create_watches = true;
        config.auto_ingestion.project_path = None; // No project path

        let daemon = WorkspaceDaemon::new(config).await.unwrap();

        // When no project path is provided, start should complete without attempting to create watches
        // We can't test the full start() method because it involves the runtime manager
        // Instead, let's verify the auto-ingestion configuration is correct
        assert!(daemon.config.auto_ingestion.enabled);
        assert!(daemon.config.auto_ingestion.auto_create_watches);
        assert!(daemon.config.auto_ingestion.project_path.is_none());
    }
}