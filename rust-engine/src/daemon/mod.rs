//! Core daemon implementation for workspace document processing

pub mod core;
pub mod state;
pub mod processing;
pub mod watcher;
// pub mod watcher_performance; // Temporarily disabled - needs proper imports
pub mod file_ops;
pub mod runtime;

use crate::config::{DaemonConfig, get_config_bool, get_config_string, get_config_u64};
use crate::error::DaemonResult;
use std::sync::Arc;
use tokio::sync::{RwLock, Mutex};
use tracing::{info, warn, debug};
use self::runtime::{RuntimeManager, RuntimeConfig};
use std::path::{Path, PathBuf};
use std::env;

/// Project information detected from the filesystem and Git
#[derive(Debug, Clone)]
struct ProjectInfo {
    name: String,
    root_path: PathBuf,
    #[allow(dead_code)]
    git_repository: Option<String>,
    #[allow(dead_code)]
    git_branch: Option<String>,
    #[allow(dead_code)]
    identifier: String, // Unique identifier for disambiguation
}

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
        // TODO: Update DaemonState::new to use lua-style config internally
        let state = Arc::new(RwLock::new(
            state::DaemonState::new(&config.database()).await?
        ));

        // Initialize document processor
        // TODO: Update DocumentProcessor::new to use lua-style config internally
        let processing = Arc::new(
            processing::DocumentProcessor::new(&config.processing(), &config.qdrant()).await?
        );

        // Initialize file watcher if enabled
        // Fixed: Use correct path from default_configuration.yaml
        let watcher = if get_config_bool("document_processing.file_watching.enabled", false) {
            Some(Arc::new(Mutex::new(
                watcher::FileWatcher::new(&config.file_watcher(), Arc::clone(&processing)).await?
            )))
        } else {
            None
        };

        // Initialize runtime manager
        // Using lua-style config access instead of hardcoded structs
        let runtime_config = RuntimeConfig {
            max_concurrent_tasks: get_config_u64("document_processing.performance.max_concurrent_tasks", 4) as usize,
            task_timeout: std::time::Duration::from_secs(get_config_u64("grpc.client.request_timeout", 30)),
            resource_pool_size: get_config_u64("grpc.server.max_concurrent_streams", 100) as usize,
            enable_monitoring: get_config_bool("monitoring.metrics.enabled", true),
            monitoring_interval: std::time::Duration::from_secs(get_config_u64("monitoring.metrics.collection_interval_secs", 60)),
            max_retry_attempts: get_config_u64("external_services.qdrant.max_retries", 3) as u32,
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
        // Fixed: Use correct paths from default_configuration.yaml
        if get_config_bool("document_processing.file_watching.project_folders.auto_monitor", false) {
            info!("Auto-ingestion is enabled, creating auto-watch during initialization");

            // Try to get project path from config
            let project_path = get_config_string("document_processing.file_watching.project_path", "");
            if !project_path.is_empty() {
                daemon.create_auto_watch(&project_path).await?;
            } else {
                info!("Auto-ingestion enabled but no project_path specified, attempting automatic project detection");

                // Attempt automatic project detection
                match daemon.detect_current_project().await {
                    Ok(project_info) => {
                        info!("Detected project: {} at path: {}", project_info.name, project_info.root_path.display());
                        daemon.create_auto_watch_for_project(&project_info).await?;
                    }
                    Err(e) => {
                        warn!("Failed to detect current project for auto-ingestion: {}", e);
                        info!("Auto-ingestion will be disabled. You can manually specify a project_path in the configuration.");
                    }
                }
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

        // Prepare file patterns using lua-style config access
        let patterns = if get_config_bool("document_processing.supported_types.code.enabled", true) {
            // Get include patterns from config
            vec!["*.rs".to_string(), "*.py".to_string(), "*.js".to_string(), "*.ts".to_string()]
        } else {
            vec![]
        };

        // Prepare ignore patterns using lua-style config access
        let ignore_patterns = vec![
            "target/".to_string(),
            "node_modules/".to_string(),
            ".git/".to_string(),
            "*.log".to_string()
        ];

        // Determine recursive depth using lua-style config access
        let max_depth = get_config_u64("document_processing.file_watching.max_depth", 0);
        let recursive_depth = if max_depth == 0 {
            -1 // Unlimited depth
        } else {
            max_depth as i32
        };

        // Create watch configuration in database
        let watch_id = state.create_watch_configuration(
            project_path,
            &collection_name,
            &patterns,
            &ignore_patterns,
            get_config_bool("document_processing.file_watching.project_folders.auto_monitor", true),
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

    /// Detect the current project by analyzing the working directory and Git repository
    async fn detect_current_project(&self) -> DaemonResult<ProjectInfo> {
        debug!("Starting automatic project detection");

        // Get current working directory
        let current_dir = env::current_dir().map_err(|e| {
            crate::error::DaemonError::Internal {
                message: format!("Failed to get current working directory: {}", e),
            }
        })?;

        debug!("Current working directory: {}", current_dir.display());

        // Find Git repository root by walking up the directory tree
        let git_root = self.find_git_repository_root(&current_dir).await?;
        debug!("Found Git repository root: {}", git_root.display());

        // Extract Git information
        let (git_repository, git_branch) = self.extract_git_info(&git_root).await;

        // Generate project name and identifier
        let project_name = self.generate_project_name(&git_root, &git_repository);
        let identifier = self.generate_project_identifier(&git_root, &git_repository);

        let project_info = ProjectInfo {
            name: project_name.clone(),
            root_path: git_root,
            git_repository: git_repository.clone(),
            git_branch,
            identifier: identifier.clone(),
        };

        info!("Project detection successful: {} ({})", project_name, identifier);
        Ok(project_info)
    }

    /// Find the Git repository root by walking up the directory tree
    async fn find_git_repository_root(&self, start_path: &Path) -> DaemonResult<PathBuf> {
        let mut current_path = start_path.to_path_buf();

        loop {
            let git_dir = current_path.join(".git");

            // Check if .git exists (either as directory or file for worktrees)
            if tokio::fs::metadata(&git_dir).await.is_ok() {
                debug!("Found .git at: {}", git_dir.display());
                return Ok(current_path);
            }

            // Move to parent directory
            match current_path.parent() {
                Some(parent) => {
                    current_path = parent.to_path_buf();
                }
                None => {
                    // Reached filesystem root without finding .git
                    // Fall back to current working directory as project root
                    warn!("No Git repository found, using current directory as project root");
                    return Ok(start_path.to_path_buf());
                }
            }
        }
    }

    /// Extract Git repository information (remote URL and current branch)
    async fn extract_git_info(&self, git_root: &Path) -> (Option<String>, Option<String>) {
        let mut git_repository = None;
        let mut git_branch = None;

        // Try to read Git remote URL
        let git_config_path = git_root.join(".git").join("config");
        if let Ok(config_content) = tokio::fs::read_to_string(&git_config_path).await {
            git_repository = self.parse_git_remote_from_config(&config_content);
        }

        // Try to read current branch
        let git_head_path = git_root.join(".git").join("HEAD");
        if let Ok(head_content) = tokio::fs::read_to_string(&git_head_path).await {
            git_branch = self.parse_git_branch_from_head(&head_content);
        }

        debug!("Git info - repository: {:?}, branch: {:?}", git_repository, git_branch);
        (git_repository, git_branch)
    }

    /// Parse Git remote URL from config file content
    fn parse_git_remote_from_config(&self, config_content: &str) -> Option<String> {
        // Look for [remote "origin"] section and extract URL
        let lines: Vec<&str> = config_content.lines().collect();
        let mut in_origin_section = false;

        for line in lines {
            let trimmed = line.trim();

            if trimmed == "[remote \"origin\"]" {
                in_origin_section = true;
                continue;
            }

            if trimmed.starts_with('[') && in_origin_section {
                // Entering a new section, stop looking
                break;
            }

            if in_origin_section && trimmed.starts_with("url = ") {
                let url = trimmed.strip_prefix("url = ").unwrap_or(trimmed);
                return Some(url.to_string());
            }
        }

        None
    }

    /// Parse current branch from HEAD file content
    fn parse_git_branch_from_head(&self, head_content: &str) -> Option<String> {
        let trimmed = head_content.trim();

        if trimmed.starts_with("ref: refs/heads/") {
            // Extract branch name from "ref: refs/heads/branch-name"
            trimmed.strip_prefix("ref: refs/heads/")
                .map(|branch| branch.to_string())
        } else {
            // HEAD is detached (contains commit hash)
            Some("detached".to_string())
        }
    }

    /// Generate a human-readable project name
    fn generate_project_name(&self, git_root: &Path, git_repository: &Option<String>) -> String {
        // Priority: Git remote name > directory name
        if let Some(repo_url) = git_repository {
            if let Some(name) = self.extract_repository_name_from_url(repo_url) {
                return self.sanitize_project_name(&name);
            }
        }

        // Fall back to directory name
        git_root
            .file_name()
            .and_then(|name| name.to_str())
            .map(|name| self.sanitize_project_name(name))
            .unwrap_or_else(|| "unknown-project".to_string())
    }

    /// Generate a unique project identifier for disambiguation
    fn generate_project_identifier(&self, git_root: &Path, git_repository: &Option<String>) -> String {
        // Primary: absolute path for uniqueness
        // Secondary: Git remote URL for cross-machine consistency
        let path_based = git_root
            .to_string_lossy()
            .replace('/', "-")
            .replace('\\', "-")
            .trim_start_matches('-')
            .to_string();

        if let Some(repo_url) = git_repository {
            // Use a hash of the repository URL for consistency
            let repo_hash = self.simple_hash(repo_url);
            format!("{}-{}", path_based, repo_hash)
        } else {
            path_based
        }
    }

    /// Extract repository name from Git URL
    fn extract_repository_name_from_url(&self, url: &str) -> Option<String> {
        // Handle various Git URL formats:
        // - https://github.com/user/repo.git
        // - git@github.com:user/repo.git
        // - https://gitlab.com/group/subgroup/repo

        let url = url.trim();

        // Remove .git suffix if present
        let url = url.strip_suffix(".git").unwrap_or(url);

        // Extract the last component after splitting by / or :
        let parts: Vec<&str> = url.split(&['/', ':']).collect();
        parts.last().map(|name| name.to_string())
    }

    /// Sanitize project name for use in collection names
    fn sanitize_project_name(&self, name: &str) -> String {
        name.chars()
            .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '-' })
            .collect::<String>()
            .trim_matches('-')
            .to_lowercase()
    }

    /// Simple hash function for generating consistent identifiers
    fn simple_hash(&self, input: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        input.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    /// Create an automatic watch configuration for a detected project
    async fn create_auto_watch_for_project(&self, project_info: &ProjectInfo) -> DaemonResult<()> {
        info!("Creating auto-watch for detected project: {} at path: {}",
              project_info.name, project_info.root_path.display());

        // Generate collection name using project name
        let collection_suffix = get_config_string("collections.root_name", "project");
        let collection_name = format!("{}-{}", project_info.name, collection_suffix);

        // Use the project root path as the watch path
        let project_path_str = project_info.root_path.to_string_lossy();

        // Check if watch configuration already exists
        let state = self.state.read().await;
        if state.watch_configuration_exists(&project_path_str).await? {
            info!("Watch configuration already exists for project path: {}", project_path_str);
            return Ok(());
        }

        // Prepare file patterns using lua-style config access
        let patterns = if get_config_bool("document_processing.supported_types.code.enabled", true) {
            vec!["*.rs".to_string(), "*.py".to_string(), "*.js".to_string(), "*.ts".to_string()]
        } else {
            vec![]
        };

        // Prepare ignore patterns using lua-style config access
        let ignore_patterns = vec![
            "target/".to_string(),
            "node_modules/".to_string(),
            ".git/".to_string(),
            "*.log".to_string()
        ];

        // Determine recursive depth using lua-style config access
        let max_depth = get_config_u64("document_processing.file_watching.max_depth", 0);
        let recursive_depth = if max_depth == 0 {
            -1 // Unlimited depth
        } else {
            max_depth as i32
        };

        // Create watch configuration in database
        let watch_id = state.create_watch_configuration(
            &project_path_str,
            &collection_name,
            &patterns,
            &ignore_patterns,
            get_config_bool("document_processing.file_watching.project_folders.auto_monitor", true),
            recursive_depth,
        ).await?;

        info!("Successfully created auto-watch configuration with ID: {} for project: {} -> collection: {}",
              watch_id, project_info.name, collection_name);

        info!("Auto-watch creation completed successfully for project: {}", project_info.name);
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

        let collection_suffix = get_config_string("collections.root_name", "project");
        format!("{}-{}", project_name, collection_suffix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::*;

    fn create_test_config() -> DaemonConfig {
        // Create a test configuration using the new PRDv3 structure
        DaemonConfig::default()
    }

    fn create_test_config_with_watcher() -> DaemonConfig {
        let mut config = create_test_config();
        // Note: file_watcher configuration is now read-only in new format
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
        let daemon = WorkspaceDaemon::new(config).await.unwrap();

        // Test access to config through lua-style functions
        let max_connections = get_config_u64("external_services.database.max_connections", 5);
        assert_eq!(max_connections, 5);
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
        // Note: config is now read-only in new format

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
        let config = create_test_config();
        // Note: config is now read-only in new format
        // auto_ingestion settings are handled through the new configuration

        let daemon = WorkspaceDaemon::new(config).await.unwrap();

        // Test the auto-watch creation logic
        let result = daemon.create_auto_watch("/tmp/test_project").await;
        assert!(result.is_ok());

        // Test collection name generation
        let collection_name = daemon.generate_collection_name("/tmp/test_project");
        assert_eq!(collection_name, "test_project-project");
    }

    #[tokio::test]
    async fn test_collection_name_generation() {
        let config = create_test_config();
        let daemon = WorkspaceDaemon::new(config).await.unwrap();

        // Test various project paths
        assert_eq!(daemon.generate_collection_name("/home/user/my-project"), "my-project-project");
        assert_eq!(daemon.generate_collection_name("/path/to/workspace-qdrant-mcp"), "workspace-qdrant-mcp-project");
        assert_eq!(daemon.generate_collection_name("/"), "unknown-project-project");
        assert_eq!(daemon.generate_collection_name(""), "unknown-project-project");
    }

    #[tokio::test]
    async fn test_auto_ingestion_disabled() {
        let mut config = create_test_config();
        // Note: config is now read-only in new format

        let mut daemon = WorkspaceDaemon::new(config).await.unwrap();

        // When auto-ingestion is disabled, start should complete without creating watches
        let result = daemon.start().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_auto_watch_creation_no_project_path() {
        let config = create_test_config();
        // Note: config is now read-only in new format
        // auto_ingestion settings are handled through the new configuration

        let daemon = WorkspaceDaemon::new(config).await.unwrap();

        // When no project path is provided, start should complete without attempting to create watches
        // We can't test the full start() method because it involves the runtime manager
        // Instead, let's verify the file watching configuration is correct
        assert!(get_config_bool("document_processing.file_watching.enabled", false));
        assert!(get_config_bool("document_processing.file_watching.project_folders.auto_monitor", false));
    }
}