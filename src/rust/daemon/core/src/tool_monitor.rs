//! Tool Availability Monitoring Module
//!
//! **DEPRECATED**: This module is deprecated because it violates the 3-table SQLite
//! compliance requirement. The tool_availability table and missing_metadata_queue
//! table are not part of the allowed schema (only schema_version, unified_queue,
//! and watch_folders are permitted).
//!
//! ## Future Direction
//!
//! Tool availability tracking should be refactored to use:
//! 1. In-memory state for runtime tool availability
//! 2. unified_queue for file requeuing (instead of missing_metadata_queue)
//! 3. Structured logging (tracing) for tool check events
//!
//! ## Original Design (Deprecated)
//!
//! The tool monitor was designed to run as a background task that:
//! 1. Periodically checks for tool availability (tree-sitter parsers, LSP servers)
//! 2. Queries missing_metadata_queue for files waiting on tools
//! 3. Requeues files when their required tools become available
//! 4. Tracks tool availability state in tool_availability table
//!
//! ## Tool Detection (Still Useful)
//!
//! Tool detection logic is implemented in pure Rust with no external dependencies:
//! - Tree-sitter: Checks for `lib<language>.so/dylib/dll` in standard locations
//! - LSP servers: Scans PATH for `*-language-server` executables

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use thiserror::Error;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

use crate::config::MonitoringConfig;
use crate::queue_operations::{QueueManager, QueueError};
use crate::queue_types::MissingTool;

/// Tool monitoring errors
#[derive(Error, Debug)]
pub enum MonitoringError {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Queue error: {0}")]
    Queue(#[from] QueueError),

    #[error("Tool detection failed: {0}")]
    ToolDetection(String),

    #[error("Configuration error: {0}")]
    Configuration(String),

    #[error("Already running")]
    AlreadyRunning,

    #[error("Not started")]
    NotStarted,
}

/// Result type for monitoring operations
pub type MonitoringResult<T> = Result<T, MonitoringError>;

/// Requeue statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RequeueStats {
    /// Number of unique tools checked
    pub tools_checked: usize,

    /// Number of tools that became available
    pub tools_now_available: usize,

    /// Number of files requeued
    pub items_requeued: usize,

    /// Timestamp of the check
    pub check_timestamp: DateTime<Utc>,

    /// Duration of check in milliseconds
    pub check_duration_ms: u64,
}

/// Tool availability state
#[derive(Debug, Clone, Serialize, Deserialize)]
struct _ToolAvailability {
    tool_type: String,
    language: String,
    tool_path: Option<String>,
    last_checked_at: i64,
    is_available: bool,
}

/// Tool monitor for tracking tool availability and requeuing files
pub struct ToolMonitor {
    /// Monitoring configuration
    config: MonitoringConfig,

    /// SQLite database pool
    db_pool: SqlitePool,

    /// Queue manager for requeuing operations
    queue_manager: QueueManager,

    /// Cancellation token for graceful shutdown
    cancellation_token: CancellationToken,

    /// Background task handle
    task_handle: Option<JoinHandle<()>>,
}

impl ToolMonitor {
    /// Create a new tool monitor
    pub fn new(
        config: MonitoringConfig,
        db_pool: SqlitePool,
        queue_manager: QueueManager,
    ) -> Self {
        Self {
            config,
            db_pool,
            queue_manager,
            cancellation_token: CancellationToken::new(),
            task_handle: None,
        }
    }

    /// Initialize database schema for tool availability tracking
    ///
    /// **DEPRECATED**: This method creates the tool_availability table which
    /// violates 3-table SQLite compliance. Do not call this method.
    /// Tool availability should be tracked in-memory instead.
    #[deprecated(since = "0.4.0", note = "Violates 3-table SQLite compliance. Use in-memory state instead.")]
    pub async fn initialize_schema(&self) -> MonitoringResult<()> {
        warn!("initialize_schema() is deprecated - tool_availability table violates 3-table compliance");
        // Return Ok without creating the table to avoid schema violations
        Ok(())
    }

    /// Start the monitoring background task
    pub fn start(&mut self) -> MonitoringResult<()> {
        if self.task_handle.is_some() {
            return Err(MonitoringError::AlreadyRunning);
        }

        if !self.config.enable_monitoring {
            info!("Tool monitoring disabled by configuration");
            return Ok(());
        }

        let config = self.config.clone();
        let db_pool = self.db_pool.clone();
        let queue_manager = self.queue_manager.clone();
        let cancellation_token = self.cancellation_token.clone();

        let task_handle = tokio::spawn(async move {
            Self::monitoring_loop(config, db_pool, queue_manager, cancellation_token).await;
        });

        self.task_handle = Some(task_handle);

        info!(
            "Tool monitor started (check_interval: {}h, check_on_startup: {})",
            self.config.check_interval_hours, self.config.check_on_startup
        );

        Ok(())
    }

    /// Stop the monitoring background task
    pub async fn stop(&mut self) -> MonitoringResult<()> {
        if let Some(task_handle) = self.task_handle.take() {
            info!("Stopping tool monitor...");

            // Signal cancellation
            self.cancellation_token.cancel();

            // Wait for task to complete (with timeout)
            match tokio::time::timeout(std::time::Duration::from_secs(5), task_handle).await {
                Ok(Ok(())) => {
                    info!("Tool monitor stopped successfully");
                    Ok(())
                }
                Ok(Err(e)) => {
                    error!("Tool monitor task panicked: {}", e);
                    Err(MonitoringError::Configuration(format!(
                        "Task panicked: {}",
                        e
                    )))
                }
                Err(_) => {
                    warn!("Tool monitor shutdown timed out");
                    Ok(()) // Still consider it stopped
                }
            }
        } else {
            Err(MonitoringError::NotStarted)
        }
    }

    /// Main monitoring loop
    async fn monitoring_loop(
        config: MonitoringConfig,
        db_pool: SqlitePool,
        queue_manager: QueueManager,
        cancellation_token: CancellationToken,
    ) {
        // Perform initial check on startup if configured
        if config.check_on_startup {
            info!("Performing initial tool availability check on startup...");
            match Self::check_and_requeue_impl(&config, &db_pool, &queue_manager).await {
                Ok(stats) => {
                    info!(
                        "Startup check complete: {} tools checked, {} now available, {} items requeued",
                        stats.tools_checked, stats.tools_now_available, stats.items_requeued
                    );
                }
                Err(e) => {
                    error!("Startup tool check failed: {}", e);
                }
            }
        }

        // Create periodic interval
        let interval_duration =
            std::time::Duration::from_secs(config.check_interval_hours * 3600);
        let mut interval = tokio::time::interval(interval_duration);

        // Skip first tick (we already did startup check)
        interval.tick().await;

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    info!("Performing periodic tool availability check...");

                    match Self::check_and_requeue_impl(&config, &db_pool, &queue_manager).await {
                        Ok(stats) => {
                            info!(
                                "Periodic check complete: {} tools checked, {} now available, {} items requeued (duration: {}ms)",
                                stats.tools_checked,
                                stats.tools_now_available,
                                stats.items_requeued,
                                stats.check_duration_ms
                            );
                        }
                        Err(e) => {
                            error!("Periodic tool check failed: {}", e);
                        }
                    }
                }
                _ = cancellation_token.cancelled() => {
                    info!("Tool monitor received cancellation signal");
                    break;
                }
            }
        }

        info!("Tool monitor loop exited");
    }

    /// Check tool availability and requeue files (implementation)
    async fn check_and_requeue_impl(
        _config: &MonitoringConfig,
        db_pool: &SqlitePool,
        queue_manager: &QueueManager,
    ) -> MonitoringResult<RequeueStats> {
        let start_time = std::time::Instant::now();
        let mut stats = RequeueStats {
            check_timestamp: Utc::now(),
            ..Default::default()
        };

        // Get all items from missing_metadata_queue
        let missing_items = queue_manager.get_missing_metadata_items(1000).await?;

        if missing_items.is_empty() {
            debug!("No items in missing_metadata_queue to check");
            stats.check_duration_ms = start_time.elapsed().as_millis() as u64;
            return Ok(stats);
        }

        info!(
            "Checking {} items from missing_metadata_queue",
            missing_items.len()
        );

        // Extract unique tools to check
        let mut tools_to_check: HashMap<(String, String), Vec<String>> = HashMap::new();

        for item in &missing_items {
            for missing_tool in &item.missing_tools {
                let (tool_type, language) = match missing_tool {
                    MissingTool::TreeSitterParser { language } => {
                        ("tree-sitter".to_string(), language.clone())
                    }
                    MissingTool::LspServer { language } => ("lsp".to_string(), language.clone()),
                    _ => continue, // Skip non-tool errors
                };

                tools_to_check
                    .entry((tool_type, language))
                    .or_default()
                    .push(item.queue_id.clone());
            }
        }

        stats.tools_checked = tools_to_check.len();
        info!("Checking {} unique tools", stats.tools_checked);

        // Check each tool's availability
        for ((tool_type, language), queue_ids) in tools_to_check {
            match Self::check_tool_availability(db_pool, &tool_type, &language).await {
                Ok(Some(tool_path)) => {
                    info!(
                        "Tool now available: {} for {} ({}), requeuing {} items",
                        tool_type,
                        language,
                        tool_path,
                        queue_ids.len()
                    );

                    stats.tools_now_available += 1;

                    // Requeue all items that were waiting for this tool
                    for queue_id in queue_ids {
                        match queue_manager
                            .retry_missing_metadata_item(&queue_id)
                            .await
                        {
                            Ok(true) => {
                                stats.items_requeued += 1;
                            }
                            Ok(false) => {
                                warn!("Item {} not found in missing_metadata_queue", queue_id);
                            }
                            Err(e) => {
                                error!("Failed to requeue item {}: {}", queue_id, e);
                            }
                        }
                    }
                }
                Ok(None) => {
                    debug!(
                        "Tool still unavailable: {} for {}",
                        tool_type, language
                    );
                }
                Err(e) => {
                    warn!(
                        "Failed to check tool availability for {} ({}): {}",
                        tool_type, language, e
                    );
                }
            }
        }

        stats.check_duration_ms = start_time.elapsed().as_millis() as u64;

        Ok(stats)
    }

    /// Check if a specific tool is available
    ///
    /// Returns Ok(Some(path)) if available, Ok(None) if not available
    async fn check_tool_availability(
        db_pool: &SqlitePool,
        tool_type: &str,
        language: &str,
    ) -> MonitoringResult<Option<String>> {
        // Check tool availability based on type
        let tool_path = match tool_type {
            "tree-sitter" => Self::find_tree_sitter_parser(language),
            "lsp" => Self::find_lsp_server(language),
            _ => {
                warn!("Unknown tool type: {}", tool_type);
                None
            }
        };

        let is_available = tool_path.is_some();

        // NOTE: Previously this wrote to tool_availability table, but that table
        // violates 3-table SQLite compliance. Tool availability is now checked
        // without persistence - state should be tracked in-memory if needed.
        let _ = db_pool; // Suppress unused warning

        debug!(
            "Tool check: {} for {} - {}",
            tool_type,
            language,
            if is_available {
                "AVAILABLE"
            } else {
                "NOT AVAILABLE"
            }
        );

        Ok(tool_path)
    }

    /// Find tree-sitter parser for a language
    ///
    /// Searches standard locations for tree-sitter dynamic libraries
    fn find_tree_sitter_parser(language: &str) -> Option<String> {
        // Standard search paths for tree-sitter parsers
        let search_paths = Self::get_tree_sitter_search_paths();

        // Library filename patterns
        #[cfg(target_os = "macos")]
        let lib_patterns = vec![
            format!("libtree-sitter-{}.dylib", language),
            format!("tree-sitter-{}.dylib", language),
        ];

        #[cfg(target_os = "linux")]
        let lib_patterns = vec![
            format!("libtree-sitter-{}.so", language),
            format!("tree-sitter-{}.so", language),
        ];

        #[cfg(target_os = "windows")]
        let lib_patterns = vec![
            format!("tree-sitter-{}.dll", language),
            format!("libtree-sitter-{}.dll", language),
        ];

        // Search for library
        for search_path in search_paths {
            for pattern in &lib_patterns {
                let lib_path = search_path.join(pattern);
                if lib_path.exists() && lib_path.is_file() {
                    return Some(lib_path.to_string_lossy().to_string());
                }
            }
        }

        None
    }

    /// Get standard tree-sitter search paths
    fn get_tree_sitter_search_paths() -> Vec<PathBuf> {
        let mut paths = Vec::new();

        // User-local tree-sitter parsers
        if let Ok(home) = std::env::var("HOME") {
            let home_path = PathBuf::from(&home);
            paths.push(home_path.join(".tree-sitter/bin"));
            paths.push(PathBuf::from(&home).join(".local/lib"));
        }

        // System-wide locations
        #[cfg(unix)]
        {
            paths.push(PathBuf::from("/usr/local/lib"));
            paths.push(PathBuf::from("/usr/lib"));
            paths.push(PathBuf::from("/opt/homebrew/lib")); // macOS ARM
        }

        // Current working directory
        if let Ok(cwd) = std::env::current_dir() {
            paths.push(cwd.join("lib"));
        }

        paths
    }

    /// Find LSP server for a language
    ///
    /// Searches PATH for language server executables
    fn find_lsp_server(language: &str) -> Option<String> {
        // Common LSP server naming patterns
        let server_names = Self::get_lsp_server_names(language);

        // Search PATH for executables
        if let Ok(path_var) = std::env::var("PATH") {
            for path_dir in path_var.split(':') {
                let path = Path::new(path_dir);
                if !path.exists() || !path.is_dir() {
                    continue;
                }

                for server_name in &server_names {
                    let executable = path.join(server_name);

                    // Check if executable exists
                    #[cfg(unix)]
                    {
                        if executable.exists() && executable.is_file() {
                            // Check if file is executable
                            use std::os::unix::fs::PermissionsExt;
                            if let Ok(metadata) = std::fs::metadata(&executable) {
                                let permissions = metadata.permissions();
                                if permissions.mode() & 0o111 != 0 {
                                    return Some(executable.to_string_lossy().to_string());
                                }
                            }
                        }
                    }

                    #[cfg(not(unix))]
                    {
                        if executable.exists() && executable.is_file() {
                            return Some(executable.to_string_lossy().to_string());
                        }
                    }
                }
            }
        }

        None
    }

    /// Get LSP server name patterns for a language
    fn get_lsp_server_names(language: &str) -> Vec<String> {
        match language {
            "rust" => vec!["rust-analyzer".to_string(), "rls".to_string()],
            "python" => vec![
                "pylsp".to_string(),
                "pyls".to_string(),
                "pyright-langserver".to_string(),
            ],
            "javascript" | "typescript" => vec![
                "typescript-language-server".to_string(),
                "tsserver".to_string(),
            ],
            "go" => vec!["gopls".to_string()],
            "java" => vec!["jdtls".to_string(), "java-language-server".to_string()],
            "c" | "cpp" => vec!["clangd".to_string(), "ccls".to_string()],
            "ruby" => vec!["solargraph".to_string()],
            "php" => vec!["phpactor".to_string(), "intelephense".to_string()],
            _ => vec![
                // Generic patterns
                format!("{}-language-server", language),
                format!("{}-lsp", language),
                format!("{}ls", language),
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_lsp_server_names() {
        let names = ToolMonitor::get_lsp_server_names("rust");
        assert!(names.contains(&"rust-analyzer".to_string()));

        let names = ToolMonitor::get_lsp_server_names("python");
        assert!(names.contains(&"pylsp".to_string()));

        let names = ToolMonitor::get_lsp_server_names("unknown");
        assert!(names.contains(&"unknown-language-server".to_string()));
    }

    #[test]
    fn test_get_tree_sitter_search_paths() {
        let paths = ToolMonitor::get_tree_sitter_search_paths();
        assert!(!paths.is_empty());

        // Should include user home directory path if HOME is set
        if std::env::var("HOME").is_ok() {
            assert!(paths.iter().any(|p| p.to_string_lossy().contains(".tree-sitter")));
        }
    }

    #[tokio::test]
    async fn test_requeue_stats_default() {
        let stats = RequeueStats::default();
        assert_eq!(stats.tools_checked, 0);
        assert_eq!(stats.tools_now_available, 0);
        assert_eq!(stats.items_requeued, 0);
    }
}
