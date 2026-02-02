//! Priority Manager Module
//!
//! Manages dynamic priority adjustments for queue items based on server lifecycle events.
//! When an MCP server starts for a project, related files are bumped to high priority (1).
//! When a server stops, related files are demoted to normal priority (3).
//!
//! ## Session Tracking
//!
//! The priority manager now includes session tracking with heartbeat mechanism:
//! - `register_session`: Increments active_sessions, updates last_active, bumps priority to HIGH
//! - `heartbeat`: Updates last_active timestamp for active sessions
//! - `unregister_session`: Decrements active_sessions, demotes priority when no active sessions
//! - `cleanup_orphaned_sessions`: Detects sessions without heartbeat for >60s (configurable)
//!
//! ## Priority Levels
//!
//! - HIGH (1): Active agent sessions - items processed first
//! - NORMAL (3): Registered projects without active sessions
//! - LOW (5): Background/inactive projects

use chrono::{DateTime, Utc, Duration as ChronoDuration};
use sqlx::SqlitePool;
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::sync::RwLock;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn, error};

use crate::metrics::METRICS;

/// Priority levels for the queue system
pub mod priority {
    /// HIGH priority: Active agent sessions - items processed first
    pub const HIGH: u8 = 1;
    /// NORMAL priority: Registered projects without active sessions
    pub const NORMAL: u8 = 3;
    /// LOW priority: Background/inactive projects
    pub const LOW: u8 = 5;
}

/// Session monitoring configuration
#[derive(Debug, Clone)]
pub struct SessionMonitorConfig {
    /// Heartbeat timeout in seconds (default: 60)
    pub heartbeat_timeout_secs: u64,
    /// Check interval in seconds (default: 30)
    pub check_interval_secs: u64,
}

impl Default for SessionMonitorConfig {
    fn default() -> Self {
        Self {
            heartbeat_timeout_secs: 60,
            check_interval_secs: 30,
        }
    }
}

/// Priority management errors
#[derive(Error, Debug)]
pub enum PriorityError {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("Invalid priority value: {0}")]
    InvalidPriority(i32),

    #[error("Empty tenant_id or branch")]
    EmptyParameter,

    #[error("Project not found: {0}")]
    ProjectNotFound(String),

    #[error("Session monitor already running")]
    MonitorAlreadyRunning,

    #[error("Session monitor not running")]
    MonitorNotRunning,
}

/// Result type for priority operations
pub type PriorityResult<T> = Result<T, PriorityError>;

/// Priority transition information
#[derive(Debug, Clone)]
pub struct PriorityTransition {
    /// Priority before transition
    pub from_priority: u8,

    /// Priority after transition
    pub to_priority: u8,

    /// Number of items affected in ingestion_queue
    pub ingestion_queue_affected: usize,

    /// Number of items affected in missing_metadata_queue
    pub missing_metadata_queue_affected: usize,

    /// Total items affected across all queues
    pub total_affected: usize,
}

impl PriorityTransition {
    /// Create a new transition record
    pub fn new(from_priority: u8, to_priority: u8) -> Self {
        Self {
            from_priority,
            to_priority,
            ingestion_queue_affected: 0,
            missing_metadata_queue_affected: 0,
            total_affected: 0,
        }
    }

    /// Add counts from queue updates
    pub fn add_counts(&mut self, ingestion: usize, missing_metadata: usize) {
        self.ingestion_queue_affected = ingestion;
        self.missing_metadata_queue_affected = missing_metadata;
        self.total_affected = ingestion + missing_metadata;
    }
}

/// Session information for tracking active MCP server connections
#[derive(Debug, Clone)]
pub struct SessionInfo {
    /// Project ID (12-char hex)
    pub project_id: String,
    /// Number of active sessions for this project
    pub active_sessions: i32,
    /// Last heartbeat timestamp
    pub last_active: DateTime<Utc>,
    /// Current priority level
    pub priority: String,
}

/// Result of orphaned session cleanup
#[derive(Debug, Clone)]
pub struct OrphanedSessionCleanup {
    /// Number of projects with orphaned sessions detected
    pub projects_affected: usize,
    /// Total sessions cleaned up
    pub sessions_cleaned: i32,
    /// Project IDs that were demoted
    pub demoted_projects: Vec<String>,
}

/// Priority Manager for server lifecycle-driven priority adjustments
#[derive(Clone)]
pub struct PriorityManager {
    db_pool: SqlitePool,
}

impl PriorityManager {
    /// Create a new PriorityManager with existing database pool
    pub fn new(db_pool: SqlitePool) -> Self {
        Self { db_pool }
    }

    /// Handle server start event - bump priority from 3 to 1 (urgent)
    ///
    /// When an MCP server starts for a project, all pending items for that project
    /// should be processed urgently to provide fresh context.
    ///
    /// # Arguments
    /// * `tenant_id` - Project/tenant identifier
    /// * `branch` - Git branch name
    ///
    /// # Returns
    /// PriorityTransition with count of affected items
    pub async fn on_server_start(
        &self,
        tenant_id: &str,
        branch: &str,
    ) -> PriorityResult<PriorityTransition> {
        // Validate inputs
        if tenant_id.is_empty() || branch.is_empty() {
            warn!(
                "Empty tenant_id or branch in on_server_start: tenant_id='{}', branch='{}'",
                tenant_id, branch
            );
            return Err(PriorityError::EmptyParameter);
        }

        const FROM_PRIORITY: u8 = 3; // Normal priority
        const TO_PRIORITY: u8 = 1; // Urgent priority

        info!(
            "Server START: Bumping priority {} → {} for tenant_id='{}', branch='{}'",
            FROM_PRIORITY, TO_PRIORITY, tenant_id, branch
        );

        self.bulk_update_priority(tenant_id, branch, FROM_PRIORITY, TO_PRIORITY)
            .await
    }

    /// Handle server stop event - demote priority from 1 to 3 (normal)
    ///
    /// When an MCP server stops, pending items can be deprioritized as they're
    /// no longer urgently needed for active development.
    ///
    /// # Arguments
    /// * `tenant_id` - Project/tenant identifier
    /// * `branch` - Git branch name
    ///
    /// # Returns
    /// PriorityTransition with count of affected items
    pub async fn on_server_stop(
        &self,
        tenant_id: &str,
        branch: &str,
    ) -> PriorityResult<PriorityTransition> {
        // Validate inputs
        if tenant_id.is_empty() || branch.is_empty() {
            warn!(
                "Empty tenant_id or branch in on_server_stop: tenant_id='{}', branch='{}'",
                tenant_id, branch
            );
            return Err(PriorityError::EmptyParameter);
        }

        const FROM_PRIORITY: u8 = 1; // Urgent priority
        const TO_PRIORITY: u8 = 3; // Normal priority

        info!(
            "Server STOP: Demoting priority {} → {} for tenant_id='{}', branch='{}'",
            FROM_PRIORITY, TO_PRIORITY, tenant_id, branch
        );

        self.bulk_update_priority(tenant_id, branch, FROM_PRIORITY, TO_PRIORITY)
            .await
    }

    /// Bulk update priorities for all matching queue items
    ///
    /// Updates both ingestion_queue and missing_metadata_queue in a single transaction.
    /// Only items with the exact from_priority are affected, preventing unintended changes.
    ///
    /// # Arguments
    /// * `tenant_id` - Filter by tenant/project
    /// * `branch` - Filter by git branch
    /// * `from_priority` - Only update items with this priority
    /// * `to_priority` - Set priority to this value
    ///
    /// # Returns
    /// PriorityTransition with counts of affected items
    async fn bulk_update_priority(
        &self,
        tenant_id: &str,
        branch: &str,
        from_priority: u8,
        to_priority: u8,
    ) -> PriorityResult<PriorityTransition> {
        // Validate priority range
        if from_priority > 10 {
            return Err(PriorityError::InvalidPriority(from_priority as i32));
        }
        if to_priority > 10 {
            return Err(PriorityError::InvalidPriority(to_priority as i32));
        }

        // Start transaction for atomic updates
        let mut tx = self.db_pool.begin().await?;

        // Update ingestion_queue
        let ingestion_query = r#"
            UPDATE ingestion_queue
            SET priority = ?1
            WHERE tenant_id = ?2
              AND branch = ?3
              AND priority = ?4
        "#;

        let ingestion_result = sqlx::query(ingestion_query)
            .bind(to_priority as i32)
            .bind(tenant_id)
            .bind(branch)
            .bind(from_priority as i32)
            .execute(&mut *tx)
            .await?;

        let ingestion_affected = ingestion_result.rows_affected() as usize;

        // Update missing_metadata_queue
        let missing_query = r#"
            UPDATE missing_metadata_queue
            SET priority = ?1
            WHERE tenant_id = ?2
              AND branch = ?3
              AND priority = ?4
        "#;

        let missing_result = sqlx::query(missing_query)
            .bind(to_priority as i32)
            .bind(tenant_id)
            .bind(branch)
            .bind(from_priority as i32)
            .execute(&mut *tx)
            .await?;

        let missing_affected = missing_result.rows_affected() as usize;

        // Commit transaction
        tx.commit().await?;

        // Create transition record
        let mut transition = PriorityTransition::new(from_priority, to_priority);
        transition.add_counts(ingestion_affected, missing_affected);

        // Log results
        if transition.total_affected == 0 {
            debug!(
                "No items found with priority {} for tenant_id='{}', branch='{}'",
                from_priority, tenant_id, branch
            );
        } else {
            info!(
                "Priority transition complete: {} items updated (ingestion: {}, missing_metadata: {})",
                transition.total_affected,
                transition.ingestion_queue_affected,
                transition.missing_metadata_queue_affected
            );

            // Warn if unusually large batch
            if transition.total_affected > 1000 {
                warn!(
                    "Large priority update: {} items affected for tenant_id='{}', branch='{}'",
                    transition.total_affected, tenant_id, branch
                );
            }
        }

        Ok(transition)
    }

    /// Get count of items with specific priority for a tenant/branch
    ///
    /// Utility method for testing and monitoring.
    pub async fn count_items_with_priority(
        &self,
        tenant_id: &str,
        branch: &str,
        priority: u8,
    ) -> PriorityResult<usize> {
        if priority > 10 {
            return Err(PriorityError::InvalidPriority(priority as i32));
        }

        let query = r#"
            SELECT
                (SELECT COUNT(*) FROM ingestion_queue
                 WHERE tenant_id = ?1 AND branch = ?2 AND priority = ?3) +
                (SELECT COUNT(*) FROM missing_metadata_queue
                 WHERE tenant_id = ?1 AND branch = ?2 AND priority = ?3) as total
        "#;

        let count: i64 = sqlx::query_scalar(query)
            .bind(tenant_id)
            .bind(branch)
            .bind(priority as i32)
            .fetch_one(&self.db_pool)
            .await?;

        Ok(count as usize)
    }

    // =========================================================================
    // Session Tracking Methods
    // =========================================================================

    /// Register a new session for a project
    ///
    /// Increments active_sessions counter, updates last_active timestamp,
    /// and bumps queue priority to HIGH if this is the first session.
    ///
    /// # Arguments
    /// * `project_id` - 12-character hex project identifier
    /// * `branch` - Git branch name (for queue priority updates)
    ///
    /// # Returns
    /// Updated session count for the project
    pub async fn register_session(
        &self,
        project_id: &str,
        _branch: &str,
    ) -> PriorityResult<i32> {
        if project_id.is_empty() {
            return Err(PriorityError::EmptyParameter);
        }

        let now = Utc::now();

        // Start transaction
        let mut tx = self.db_pool.begin().await?;

        // Update projects table
        let update_query = r#"
            UPDATE projects
            SET active_sessions = active_sessions + 1,
                last_active = ?1,
                priority = 'high'
            WHERE project_id = ?2
        "#;

        let result = sqlx::query(update_query)
            .bind(now.to_rfc3339())
            .bind(project_id)
            .execute(&mut *tx)
            .await?;

        if result.rows_affected() == 0 {
            tx.rollback().await?;
            return Err(PriorityError::ProjectNotFound(project_id.to_string()));
        }

        // Get updated session count
        let count: i32 = sqlx::query_scalar(
            "SELECT active_sessions FROM projects WHERE project_id = ?1"
        )
            .bind(project_id)
            .fetch_one(&mut *tx)
            .await?;

        // If this is the first session, bump queue priorities
        if count == 1 {
            // Update ingestion_queue priorities for this project
            sqlx::query(
                r#"
                UPDATE ingestion_queue
                SET priority = ?1
                WHERE tenant_id = ?2
                  AND priority > ?1
                "#,
            )
            .bind(priority::HIGH as i32)
            .bind(project_id)
            .execute(&mut *tx)
            .await?;

            // Update missing_metadata_queue priorities
            sqlx::query(
                r#"
                UPDATE missing_metadata_queue
                SET priority = ?1
                WHERE tenant_id = ?2
                  AND priority > ?1
                "#,
            )
            .bind(priority::HIGH as i32)
            .bind(project_id)
            .execute(&mut *tx)
            .await?;

            info!(
                "First session registered for project {}, priority bumped to HIGH",
                project_id
            );
        }

        tx.commit().await?;

        // Record session metrics (Task 412.6)
        METRICS.session_started(project_id, "high");

        info!(
            "Session registered for project {}: {} active sessions",
            project_id, count
        );

        Ok(count)
    }

    /// Unregister a session for a project
    ///
    /// Decrements active_sessions counter. If no sessions remain,
    /// demotes queue priority from HIGH to NORMAL.
    ///
    /// # Arguments
    /// * `project_id` - 12-character hex project identifier
    /// * `branch` - Git branch name (for queue priority updates)
    ///
    /// # Returns
    /// Updated session count for the project
    pub async fn unregister_session(
        &self,
        project_id: &str,
        _branch: &str,
    ) -> PriorityResult<i32> {
        if project_id.is_empty() {
            return Err(PriorityError::EmptyParameter);
        }

        // Start transaction
        let mut tx = self.db_pool.begin().await?;

        // Get current session count
        let current_count: i32 = sqlx::query_scalar(
            "SELECT active_sessions FROM projects WHERE project_id = ?1"
        )
            .bind(project_id)
            .fetch_optional(&mut *tx)
            .await?
            .ok_or_else(|| PriorityError::ProjectNotFound(project_id.to_string()))?;

        // Don't go below 0
        let new_count = (current_count - 1).max(0);

        // Update projects table
        let priority = if new_count == 0 { "normal" } else { "high" };

        sqlx::query(
            r#"
            UPDATE projects
            SET active_sessions = ?1,
                priority = ?2
            WHERE project_id = ?3
            "#,
        )
        .bind(new_count)
        .bind(priority)
        .bind(project_id)
        .execute(&mut *tx)
        .await?;

        // If no sessions remain, demote queue priorities
        if new_count == 0 {
            // Demote ingestion_queue priorities from HIGH to NORMAL
            sqlx::query(
                r#"
                UPDATE ingestion_queue
                SET priority = ?1
                WHERE tenant_id = ?2
                  AND priority = ?3
                "#,
            )
            .bind(priority::NORMAL as i32)
            .bind(project_id)
            .bind(priority::HIGH as i32)
            .execute(&mut *tx)
            .await?;

            // Demote missing_metadata_queue priorities
            sqlx::query(
                r#"
                UPDATE missing_metadata_queue
                SET priority = ?1
                WHERE tenant_id = ?2
                  AND priority = ?3
                "#,
            )
            .bind(priority::NORMAL as i32)
            .bind(project_id)
            .bind(priority::HIGH as i32)
            .execute(&mut *tx)
            .await?;

            info!(
                "Last session unregistered for project {}, priority demoted to NORMAL",
                project_id
            );
        }

        tx.commit().await?;

        // Record session end metrics (Task 412.6)
        // Note: duration is not tracked (session start time not persisted)
        METRICS.session_ended(project_id, priority, 0.0);

        info!(
            "Session unregistered for project {}: {} active sessions",
            project_id, new_count
        );

        Ok(new_count)
    }

    /// Update heartbeat timestamp for a project
    ///
    /// Called periodically by MCP servers to indicate they're still alive.
    /// Updates the last_active timestamp to the current time.
    ///
    /// # Arguments
    /// * `project_id` - 12-character hex project identifier
    ///
    /// # Returns
    /// true if heartbeat was recorded, false if project not found
    pub async fn heartbeat(&self, project_id: &str) -> PriorityResult<bool> {
        if project_id.is_empty() {
            return Err(PriorityError::EmptyParameter);
        }

        // Measure heartbeat latency (Task 412.6)
        let start = Instant::now();

        let now = Utc::now();

        let result = sqlx::query(
            r#"
            UPDATE projects
            SET last_active = ?1
            WHERE project_id = ?2
              AND active_sessions > 0
            "#,
        )
        .bind(now.to_rfc3339())
        .bind(project_id)
        .execute(&self.db_pool)
        .await?;

        let updated = result.rows_affected() > 0;

        // Record heartbeat latency metric (Task 412.6)
        let latency_secs = start.elapsed().as_secs_f64();
        if updated {
            METRICS.heartbeat_processed(project_id, latency_secs);
            debug!("Heartbeat received for project {} (latency: {:.3}s)", project_id, latency_secs);
        } else {
            warn!(
                "Heartbeat for project {} ignored (no active sessions or not found)",
                project_id
            );
        }

        Ok(updated)
    }

    /// Get session info for a project
    ///
    /// # Arguments
    /// * `project_id` - 12-character hex project identifier
    ///
    /// # Returns
    /// SessionInfo if project exists, None otherwise
    pub async fn get_session_info(&self, project_id: &str) -> PriorityResult<Option<SessionInfo>> {
        let query = r#"
            SELECT project_id, active_sessions, last_active, priority
            FROM projects
            WHERE project_id = ?1
        "#;

        let row = sqlx::query(query)
            .bind(project_id)
            .fetch_optional(&self.db_pool)
            .await?;

        if let Some(row) = row {
            use sqlx::Row;
            let last_active_str: Option<String> = row.try_get("last_active")?;
            let last_active = last_active_str
                .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(Utc::now);

            Ok(Some(SessionInfo {
                project_id: row.try_get("project_id")?,
                active_sessions: row.try_get("active_sessions")?,
                last_active,
                priority: row.try_get("priority")?,
            }))
        } else {
            Ok(None)
        }
    }

    /// Cleanup orphaned sessions
    ///
    /// Detects projects with active_sessions > 0 but last_active older than
    /// the heartbeat timeout. These are sessions where the MCP server died
    /// without sending a proper shutdown notification.
    ///
    /// # Arguments
    /// * `timeout_secs` - Heartbeat timeout in seconds (default: 60)
    ///
    /// # Returns
    /// OrphanedSessionCleanup with cleanup statistics
    pub async fn cleanup_orphaned_sessions(
        &self,
        timeout_secs: u64,
    ) -> PriorityResult<OrphanedSessionCleanup> {
        let cutoff = Utc::now() - ChronoDuration::seconds(timeout_secs as i64);
        let cutoff_str = cutoff.to_rfc3339();

        // Start transaction
        let mut tx = self.db_pool.begin().await?;

        // Find orphaned projects (active sessions but stale heartbeat)
        let orphaned_query = r#"
            SELECT project_id, active_sessions
            FROM projects
            WHERE active_sessions > 0
              AND last_active IS NOT NULL
              AND last_active < ?1
        "#;

        let rows = sqlx::query(orphaned_query)
            .bind(&cutoff_str)
            .fetch_all(&mut *tx)
            .await?;

        let mut demoted_projects = Vec::new();
        let mut total_sessions = 0i32;

        for row in &rows {
            use sqlx::Row;
            let project_id: String = row.try_get("project_id")?;
            let sessions: i32 = row.try_get("active_sessions")?;

            total_sessions += sessions;
            demoted_projects.push(project_id.clone());

            // Record orphaned session cleanup metrics (Task 412.6)
            // Decrement active sessions for each orphaned session
            for _ in 0..sessions {
                METRICS.session_ended(&project_id, "high", 0.0);
            }

            // Reset active_sessions and demote priority
            sqlx::query(
                r#"
                UPDATE projects
                SET active_sessions = 0,
                    priority = 'normal'
                WHERE project_id = ?1
                "#,
            )
            .bind(&project_id)
            .execute(&mut *tx)
            .await?;

            // Demote queue priorities for this project
            sqlx::query(
                r#"
                UPDATE ingestion_queue
                SET priority = ?1
                WHERE tenant_id = ?2
                  AND priority = ?3
                "#,
            )
            .bind(priority::NORMAL as i32)
            .bind(&project_id)
            .bind(priority::HIGH as i32)
            .execute(&mut *tx)
            .await?;

            sqlx::query(
                r#"
                UPDATE missing_metadata_queue
                SET priority = ?1
                WHERE tenant_id = ?2
                  AND priority = ?3
                "#,
            )
            .bind(priority::NORMAL as i32)
            .bind(&project_id)
            .bind(priority::HIGH as i32)
            .execute(&mut *tx)
            .await?;
        }

        tx.commit().await?;

        let cleanup = OrphanedSessionCleanup {
            projects_affected: demoted_projects.len(),
            sessions_cleaned: total_sessions,
            demoted_projects: demoted_projects.clone(),
        };

        if cleanup.projects_affected > 0 {
            warn!(
                "Cleaned up {} orphaned sessions across {} projects: {:?}",
                cleanup.sessions_cleaned,
                cleanup.projects_affected,
                cleanup.demoted_projects
            );
        } else {
            debug!("No orphaned sessions found (timeout: {}s)", timeout_secs);
        }

        Ok(cleanup)
    }

    /// Get all projects with high priority (active sessions)
    pub async fn get_high_priority_projects(&self) -> PriorityResult<Vec<SessionInfo>> {
        let query = r#"
            SELECT project_id, active_sessions, last_active, priority
            FROM projects
            WHERE priority = 'high'
              AND active_sessions > 0
            ORDER BY last_active DESC
        "#;

        let rows = sqlx::query(query)
            .fetch_all(&self.db_pool)
            .await?;

        let mut projects = Vec::new();
        for row in rows {
            use sqlx::Row;
            let last_active_str: Option<String> = row.try_get("last_active")?;
            let last_active = last_active_str
                .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(Utc::now);

            projects.push(SessionInfo {
                project_id: row.try_get("project_id")?,
                active_sessions: row.try_get("active_sessions")?,
                last_active,
                priority: row.try_get("priority")?,
            });
        }

        Ok(projects)
    }
}

/// Session Monitor - Background task for orphaned session cleanup
///
/// Runs periodically to detect and cleanup orphaned sessions where
/// MCP servers died without sending shutdown notifications.
pub struct SessionMonitor {
    priority_manager: PriorityManager,
    config: SessionMonitorConfig,
    cancellation_token: CancellationToken,
    task_handle: Arc<RwLock<Option<JoinHandle<()>>>>,
}

impl SessionMonitor {
    /// Create a new session monitor
    pub fn new(priority_manager: PriorityManager, config: SessionMonitorConfig) -> Self {
        Self {
            priority_manager,
            config,
            cancellation_token: CancellationToken::new(),
            task_handle: Arc::new(RwLock::new(None)),
        }
    }

    /// Create with default configuration
    pub fn with_defaults(priority_manager: PriorityManager) -> Self {
        Self::new(priority_manager, SessionMonitorConfig::default())
    }

    /// Start the session monitor background task
    pub async fn start(&self) -> PriorityResult<()> {
        let mut handle = self.task_handle.write().await;
        if handle.is_some() {
            return Err(PriorityError::MonitorAlreadyRunning);
        }

        info!(
            "Starting session monitor (heartbeat_timeout={}s, check_interval={}s)",
            self.config.heartbeat_timeout_secs, self.config.check_interval_secs
        );

        let priority_manager = self.priority_manager.clone();
        let timeout_secs = self.config.heartbeat_timeout_secs;
        let check_interval = Duration::from_secs(self.config.check_interval_secs);
        let cancellation_token = self.cancellation_token.clone();

        let task = tokio::spawn(async move {
            loop {
                // Check for cancellation
                if cancellation_token.is_cancelled() {
                    info!("Session monitor shutting down");
                    break;
                }

                // Wait for check interval or cancellation
                tokio::select! {
                    _ = tokio::time::sleep(check_interval) => {
                        // Perform cleanup
                        match priority_manager.cleanup_orphaned_sessions(timeout_secs).await {
                            Ok(cleanup) => {
                                if cleanup.projects_affected > 0 {
                                    info!(
                                        "Session monitor cleanup: {} orphaned sessions across {} projects",
                                        cleanup.sessions_cleaned,
                                        cleanup.projects_affected
                                    );
                                }
                            }
                            Err(e) => {
                                error!("Session monitor cleanup failed: {}", e);
                            }
                        }
                    }
                    _ = cancellation_token.cancelled() => {
                        info!("Session monitor received cancellation signal");
                        break;
                    }
                }
            }
        });

        *handle = Some(task);
        Ok(())
    }

    /// Stop the session monitor
    pub async fn stop(&self) -> PriorityResult<()> {
        let mut handle = self.task_handle.write().await;

        if handle.is_none() {
            return Err(PriorityError::MonitorNotRunning);
        }

        info!("Stopping session monitor...");
        self.cancellation_token.cancel();

        if let Some(task) = handle.take() {
            match tokio::time::timeout(Duration::from_secs(5), task).await {
                Ok(Ok(())) => {
                    info!("Session monitor stopped cleanly");
                }
                Ok(Err(e)) => {
                    error!("Session monitor task panicked: {}", e);
                }
                Err(_) => {
                    warn!("Session monitor did not stop within timeout, aborting");
                }
            }
        }

        Ok(())
    }

    /// Check if the monitor is running
    pub async fn is_running(&self) -> bool {
        self.task_handle.read().await.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::queue_config::QueueConnectionConfig;
    use crate::queue_operations::{QueueManager, QueueOperation};
    use tempfile::tempdir;

    /// Helper to create test database with schema
    async fn setup_test_db() -> (SqlitePool, tempfile::TempDir) {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_priority.db");

        let config = QueueConnectionConfig::with_database_path(&db_path);
        let pool = config.create_pool().await.unwrap();

        // Initialize schema
        sqlx::query(include_str!("schema/legacy/queue_schema.sql"))
            .execute(&pool)
            .await
            .unwrap();

        sqlx::query(include_str!("schema/legacy/missing_metadata_queue_schema.sql"))
            .execute(&pool)
            .await
            .unwrap();

        (pool, temp_dir)
    }

    /// Helper to create test database with projects table for session tracking tests
    async fn setup_test_db_with_projects() -> (SqlitePool, tempfile::TempDir) {
        let (pool, temp_dir) = setup_test_db().await;

        // Add projects table schema (v7)
        let projects_schema = r#"
            CREATE TABLE IF NOT EXISTS projects (
                project_id TEXT PRIMARY KEY,
                project_name TEXT,
                project_root TEXT NOT NULL UNIQUE,
                priority TEXT DEFAULT 'normal' CHECK (priority IN ('high', 'normal', 'low')),
                active_sessions INTEGER DEFAULT 0,
                git_remote TEXT,
                registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_active TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        "#;

        sqlx::query(projects_schema)
            .execute(&pool)
            .await
            .unwrap();

        (pool, temp_dir)
    }

    /// Helper to create a test project
    async fn create_test_project(pool: &SqlitePool, project_id: &str, project_root: &str) {
        sqlx::query(
            r#"
            INSERT INTO projects (project_id, project_root, priority, active_sessions)
            VALUES (?1, ?2, 'normal', 0)
            "#,
        )
        .bind(project_id)
        .bind(project_root)
        .execute(pool)
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn test_server_start_bumps_priority() {
        let (pool, _temp_dir) = setup_test_db().await;
        let queue_manager = QueueManager::new(pool.clone());
        let priority_manager = PriorityManager::new(pool);

        // Enqueue items with normal priority (3)
        queue_manager
            .enqueue_file(
                "/test/file1.rs",
                "test-collection",
                "test-tenant",
                "main",
                QueueOperation::Ingest,
                3,
                None,
            )
            .await
            .unwrap();

        queue_manager
            .enqueue_file(
                "/test/file2.rs",
                "test-collection",
                "test-tenant",
                "main",
                QueueOperation::Ingest,
                3,
                None,
            )
            .await
            .unwrap();

        // Trigger server start - should bump priority to 1
        let transition = priority_manager
            .on_server_start("test-tenant", "main")
            .await
            .unwrap();

        assert_eq!(transition.from_priority, 3);
        assert_eq!(transition.to_priority, 1);
        assert_eq!(transition.ingestion_queue_affected, 2);
        assert_eq!(transition.total_affected, 2);

        // Verify items now have priority 1
        let count = priority_manager
            .count_items_with_priority("test-tenant", "main", 1)
            .await
            .unwrap();
        assert_eq!(count, 2);

        // Verify no items remain with priority 3
        let count = priority_manager
            .count_items_with_priority("test-tenant", "main", 3)
            .await
            .unwrap();
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn test_server_stop_demotes_priority() {
        let (pool, _temp_dir) = setup_test_db().await;
        let queue_manager = QueueManager::new(pool.clone());
        let priority_manager = PriorityManager::new(pool);

        // Enqueue items with urgent priority (1)
        queue_manager
            .enqueue_file(
                "/test/file1.rs",
                "test-collection",
                "test-tenant",
                "main",
                QueueOperation::Ingest,
                1,
                None,
            )
            .await
            .unwrap();

        // Trigger server stop - should demote priority to 3
        let transition = priority_manager
            .on_server_stop("test-tenant", "main")
            .await
            .unwrap();

        assert_eq!(transition.from_priority, 1);
        assert_eq!(transition.to_priority, 3);
        assert_eq!(transition.ingestion_queue_affected, 1);
        assert_eq!(transition.total_affected, 1);

        // Verify item now has priority 3
        let count = priority_manager
            .count_items_with_priority("test-tenant", "main", 3)
            .await
            .unwrap();
        assert_eq!(count, 1);
    }

    #[tokio::test]
    async fn test_branch_isolation() {
        let (pool, _temp_dir) = setup_test_db().await;
        let queue_manager = QueueManager::new(pool.clone());
        let priority_manager = PriorityManager::new(pool);

        // Enqueue items on different branches
        queue_manager
            .enqueue_file(
                "/test/file1.rs",
                "test-collection",
                "test-tenant",
                "main",
                QueueOperation::Ingest,
                3,
                None,
            )
            .await
            .unwrap();

        queue_manager
            .enqueue_file(
                "/test/file2.rs",
                "test-collection",
                "test-tenant",
                "feature-branch",
                QueueOperation::Ingest,
                3,
                None,
            )
            .await
            .unwrap();

        // Trigger server start for main branch only
        let transition = priority_manager
            .on_server_start("test-tenant", "main")
            .await
            .unwrap();

        assert_eq!(transition.total_affected, 1);

        // Verify main branch item has priority 1
        let count = priority_manager
            .count_items_with_priority("test-tenant", "main", 1)
            .await
            .unwrap();
        assert_eq!(count, 1);

        // Verify feature branch item still has priority 3
        let count = priority_manager
            .count_items_with_priority("test-tenant", "feature-branch", 3)
            .await
            .unwrap();
        assert_eq!(count, 1);
    }

    #[tokio::test]
    async fn test_empty_parameters_error() {
        let (pool, _temp_dir) = setup_test_db().await;
        let priority_manager = PriorityManager::new(pool);

        // Empty tenant_id
        let result = priority_manager.on_server_start("", "main").await;
        assert!(matches!(result, Err(PriorityError::EmptyParameter)));

        // Empty branch
        let result = priority_manager.on_server_start("test-tenant", "").await;
        assert!(matches!(result, Err(PriorityError::EmptyParameter)));
    }

    #[tokio::test]
    async fn test_no_matching_items() {
        let (pool, _temp_dir) = setup_test_db().await;
        let priority_manager = PriorityManager::new(pool);

        // No items in queue - should succeed with 0 affected
        let transition = priority_manager
            .on_server_start("nonexistent-tenant", "main")
            .await
            .unwrap();

        assert_eq!(transition.total_affected, 0);
    }

    #[tokio::test]
    async fn test_transaction_rollback_on_error() {
        let (pool, _temp_dir) = setup_test_db().await;
        let queue_manager = QueueManager::new(pool.clone());
        let priority_manager = PriorityManager::new(pool.clone());

        // Enqueue an item
        queue_manager
            .enqueue_file(
                "/test/file1.rs",
                "test-collection",
                "test-tenant",
                "main",
                QueueOperation::Ingest,
                3,
                None,
            )
            .await
            .unwrap();

        // Close the pool to force an error
        pool.close().await;

        // Attempt priority update - should fail
        let result = priority_manager
            .on_server_start("test-tenant", "main")
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_missing_metadata_queue_update() {
        let (pool, _temp_dir) = setup_test_db().await;
        let queue_manager = QueueManager::new(pool.clone());
        let priority_manager = PriorityManager::new(pool);

        // Enqueue item in ingestion_queue
        queue_manager
            .enqueue_file(
                "/test/file1.rs",
                "test-collection",
                "test-tenant",
                "main",
                QueueOperation::Ingest,
                3,
                None,
            )
            .await
            .unwrap();

        // Manually add item to missing_metadata_queue with priority 3
        let insert_query = r#"
            INSERT INTO missing_metadata_queue (
                queue_id, file_absolute_path, collection_name, tenant_id, branch,
                operation, priority, missing_tools, queued_timestamp
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
        "#;

        sqlx::query(insert_query)
            .bind("test-queue-id")
            .bind("/test/file2.rs")
            .bind("test-collection")
            .bind("test-tenant")
            .bind("main")
            .bind("ingest")
            .bind(3)
            .bind(r#"[{"LspServer": {"language": "rust"}}]"#)
            .bind("2024-01-01T00:00:00Z")
            .execute(&priority_manager.db_pool)
            .await
            .unwrap();

        // Trigger server start
        let transition = priority_manager
            .on_server_start("test-tenant", "main")
            .await
            .unwrap();

        // Should affect both queues
        assert_eq!(transition.ingestion_queue_affected, 1);
        assert_eq!(transition.missing_metadata_queue_affected, 1);
        assert_eq!(transition.total_affected, 2);
    }

    // =========================================================================
    // Session Tracking Tests
    // =========================================================================

    #[tokio::test]
    async fn test_register_session_increments_count() {
        let (pool, _temp_dir) = setup_test_db_with_projects().await;
        let priority_manager = PriorityManager::new(pool.clone());

        // Create test project
        create_test_project(&pool, "abcd12345678", "/test/project").await;

        // Register first session
        let count = priority_manager
            .register_session("abcd12345678", "main")
            .await
            .unwrap();
        assert_eq!(count, 1);

        // Register second session
        let count = priority_manager
            .register_session("abcd12345678", "main")
            .await
            .unwrap();
        assert_eq!(count, 2);

        // Verify session info
        let info = priority_manager
            .get_session_info("abcd12345678")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(info.active_sessions, 2);
        assert_eq!(info.priority, "high");
    }

    #[tokio::test]
    async fn test_unregister_session_decrements_count() {
        let (pool, _temp_dir) = setup_test_db_with_projects().await;
        let priority_manager = PriorityManager::new(pool.clone());

        // Create test project and register sessions
        create_test_project(&pool, "abcd12345678", "/test/project").await;
        priority_manager.register_session("abcd12345678", "main").await.unwrap();
        priority_manager.register_session("abcd12345678", "main").await.unwrap();

        // Unregister one session
        let count = priority_manager
            .unregister_session("abcd12345678", "main")
            .await
            .unwrap();
        assert_eq!(count, 1);

        // Verify still high priority
        let info = priority_manager
            .get_session_info("abcd12345678")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(info.priority, "high");

        // Unregister last session
        let count = priority_manager
            .unregister_session("abcd12345678", "main")
            .await
            .unwrap();
        assert_eq!(count, 0);

        // Verify demoted to normal priority
        let info = priority_manager
            .get_session_info("abcd12345678")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(info.priority, "normal");
    }

    #[tokio::test]
    async fn test_register_session_bumps_queue_priority() {
        let (pool, _temp_dir) = setup_test_db_with_projects().await;
        let queue_manager = QueueManager::new(pool.clone());
        let priority_manager = PriorityManager::new(pool.clone());

        // Create test project
        create_test_project(&pool, "abcd12345678", "/test/project").await;

        // Enqueue items with normal priority
        queue_manager
            .enqueue_file(
                "/test/file1.rs",
                "test-collection",
                "abcd12345678",  // Use project_id as tenant_id
                "main",
                QueueOperation::Ingest,
                priority::NORMAL as i32,
                None,
            )
            .await
            .unwrap();

        // Register session - should bump queue priority
        priority_manager
            .register_session("abcd12345678", "main")
            .await
            .unwrap();

        // Verify queue items now have HIGH priority
        let count = priority_manager
            .count_items_with_priority("abcd12345678", "main", priority::HIGH)
            .await
            .unwrap();
        assert_eq!(count, 1);
    }

    #[tokio::test]
    async fn test_heartbeat_updates_timestamp() {
        let (pool, _temp_dir) = setup_test_db_with_projects().await;
        let priority_manager = PriorityManager::new(pool.clone());

        // Create test project and register session
        create_test_project(&pool, "abcd12345678", "/test/project").await;
        priority_manager.register_session("abcd12345678", "main").await.unwrap();

        // Get initial timestamp
        let info_before = priority_manager
            .get_session_info("abcd12345678")
            .await
            .unwrap()
            .unwrap();

        // Wait briefly and send heartbeat
        tokio::time::sleep(Duration::from_millis(10)).await;
        let updated = priority_manager
            .heartbeat("abcd12345678")
            .await
            .unwrap();
        assert!(updated);

        // Verify timestamp updated
        let info_after = priority_manager
            .get_session_info("abcd12345678")
            .await
            .unwrap()
            .unwrap();
        assert!(info_after.last_active >= info_before.last_active);
    }

    #[tokio::test]
    async fn test_heartbeat_ignored_without_active_session() {
        let (pool, _temp_dir) = setup_test_db_with_projects().await;
        let priority_manager = PriorityManager::new(pool.clone());

        // Create test project without active sessions
        create_test_project(&pool, "abcd12345678", "/test/project").await;

        // Heartbeat should be ignored
        let updated = priority_manager
            .heartbeat("abcd12345678")
            .await
            .unwrap();
        assert!(!updated);
    }

    #[tokio::test]
    async fn test_cleanup_orphaned_sessions() {
        let (pool, _temp_dir) = setup_test_db_with_projects().await;
        let priority_manager = PriorityManager::new(pool.clone());

        // Create test project
        create_test_project(&pool, "abcd12345678", "/test/project").await;

        // Register session
        priority_manager.register_session("abcd12345678", "main").await.unwrap();

        // Manually set last_active to old timestamp to simulate orphaned session
        let old_time = Utc::now() - ChronoDuration::minutes(5);
        sqlx::query("UPDATE projects SET last_active = ?1 WHERE project_id = ?2")
            .bind(old_time.to_rfc3339())
            .bind("abcd12345678")
            .execute(&pool)
            .await
            .unwrap();

        // Cleanup with 60 second timeout - should detect orphaned session
        let cleanup = priority_manager
            .cleanup_orphaned_sessions(60)
            .await
            .unwrap();

        assert_eq!(cleanup.projects_affected, 1);
        assert_eq!(cleanup.sessions_cleaned, 1);
        assert!(cleanup.demoted_projects.contains(&"abcd12345678".to_string()));

        // Verify session cleaned up
        let info = priority_manager
            .get_session_info("abcd12345678")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(info.active_sessions, 0);
        assert_eq!(info.priority, "normal");
    }

    #[tokio::test]
    async fn test_no_orphaned_sessions_with_recent_heartbeat() {
        let (pool, _temp_dir) = setup_test_db_with_projects().await;
        let priority_manager = PriorityManager::new(pool.clone());

        // Create test project and register session (sets last_active to now)
        create_test_project(&pool, "abcd12345678", "/test/project").await;
        priority_manager.register_session("abcd12345678", "main").await.unwrap();

        // Cleanup with 60 second timeout - should NOT detect orphaned session
        let cleanup = priority_manager
            .cleanup_orphaned_sessions(60)
            .await
            .unwrap();

        assert_eq!(cleanup.projects_affected, 0);
        assert_eq!(cleanup.sessions_cleaned, 0);

        // Verify session still active
        let info = priority_manager
            .get_session_info("abcd12345678")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(info.active_sessions, 1);
        assert_eq!(info.priority, "high");
    }

    #[tokio::test]
    async fn test_get_high_priority_projects() {
        let (pool, _temp_dir) = setup_test_db_with_projects().await;
        let priority_manager = PriorityManager::new(pool.clone());

        // Create multiple test projects
        create_test_project(&pool, "project1aaaa", "/test/project1").await;
        create_test_project(&pool, "project2bbbb", "/test/project2").await;
        create_test_project(&pool, "project3cccc", "/test/project3").await;

        // Register sessions for some projects
        priority_manager.register_session("project1aaaa", "main").await.unwrap();
        priority_manager.register_session("project2bbbb", "main").await.unwrap();

        // Get high priority projects
        let high_priority = priority_manager
            .get_high_priority_projects()
            .await
            .unwrap();

        assert_eq!(high_priority.len(), 2);
        let project_ids: Vec<_> = high_priority.iter().map(|p| &p.project_id).collect();
        assert!(project_ids.contains(&&"project1aaaa".to_string()));
        assert!(project_ids.contains(&&"project2bbbb".to_string()));
        assert!(!project_ids.contains(&&"project3cccc".to_string()));
    }

    #[tokio::test]
    async fn test_register_nonexistent_project_fails() {
        let (pool, _temp_dir) = setup_test_db_with_projects().await;
        let priority_manager = PriorityManager::new(pool);

        // Try to register session for non-existent project
        let result = priority_manager
            .register_session("nonexistent12", "main")
            .await;

        assert!(matches!(result, Err(PriorityError::ProjectNotFound(_))));
    }

    #[tokio::test]
    async fn test_unregister_nonexistent_project_fails() {
        let (pool, _temp_dir) = setup_test_db_with_projects().await;
        let priority_manager = PriorityManager::new(pool);

        // Try to unregister session for non-existent project
        let result = priority_manager
            .unregister_session("nonexistent12", "main")
            .await;

        assert!(matches!(result, Err(PriorityError::ProjectNotFound(_))));
    }

    #[tokio::test]
    async fn test_session_count_does_not_go_negative() {
        let (pool, _temp_dir) = setup_test_db_with_projects().await;
        let priority_manager = PriorityManager::new(pool.clone());

        // Create test project
        create_test_project(&pool, "abcd12345678", "/test/project").await;

        // Unregister without registering first
        let count = priority_manager
            .unregister_session("abcd12345678", "main")
            .await
            .unwrap();

        // Should not go negative
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn test_priority_constants() {
        assert_eq!(priority::HIGH, 1);
        assert_eq!(priority::NORMAL, 3);
        assert_eq!(priority::LOW, 5);
    }
}
