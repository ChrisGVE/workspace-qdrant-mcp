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
use thiserror::Error;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

use crate::config::MonitoringConfig;
use crate::queue_operations::{MissingMetadataItem, QueueError, QueueManager};
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

/// Group missing metadata items by their required (tool_type, language) pair.
///
/// Returns a map from `(tool_type, language)` to the list of queue IDs that
/// depend on that tool. Non-tool missing entries are skipped.
fn group_items_by_tool(items: &[MissingMetadataItem]) -> HashMap<(String, String), Vec<String>> {
    let mut tools_to_check: HashMap<(String, String), Vec<String>> = HashMap::new();

    for item in items {
        for missing_tool in &item.missing_tools {
            let (tool_type, language) = match missing_tool {
                MissingTool::TreeSitterParser { language } => {
                    ("tree-sitter".to_string(), language.clone())
                }
                MissingTool::LspServer { language } => ("lsp".to_string(), language.clone()),
                _ => continue,
            };

            tools_to_check
                .entry((tool_type, language))
                .or_default()
                .push(item.queue_id.clone());
        }
    }

    tools_to_check
}

/// Requeue all items waiting on a tool that is now available.
///
/// Iterates over the given queue IDs, calling `retry_missing_metadata_item` for
/// each one, and accumulates the count of successfully requeued items.
async fn requeue_items_for_tool(
    queue_manager: &QueueManager,
    queue_ids: Vec<String>,
    stats: &mut RequeueStats,
) {
    for queue_id in queue_ids {
        match queue_manager.retry_missing_metadata_item(&queue_id).await {
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
    pub fn new(config: MonitoringConfig, db_pool: SqlitePool, queue_manager: QueueManager) -> Self {
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
    #[deprecated(
        since = "0.1.0-beta1",
        note = "Violates 3-table SQLite compliance. Use in-memory state instead."
    )]
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
        let interval_duration = std::time::Duration::from_secs(config.check_interval_hours * 3600);
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

        let tools_to_check = group_items_by_tool(&missing_items);
        stats.tools_checked = tools_to_check.len();
        info!("Checking {} unique tools", stats.tools_checked);

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
                    requeue_items_for_tool(queue_manager, queue_ids, &mut stats).await;
                }
                Ok(None) => {
                    debug!("Tool still unavailable: {} for {}", tool_type, language);
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
        let tool_path = match tool_type {
            "tree-sitter" => super::tool_detection::find_tree_sitter_parser(language),
            "lsp" => super::tool_detection::find_lsp_server(language),
            _ => {
                warn!("Unknown tool type: {}", tool_type);
                None
            }
        };

        let is_available = tool_path.is_some();
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_requeue_stats_default() {
        let stats = RequeueStats::default();
        assert_eq!(stats.tools_checked, 0);
        assert_eq!(stats.tools_now_available, 0);
        assert_eq!(stats.items_requeued, 0);
    }
}
