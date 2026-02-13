//! Priority Manager Module
//!
//! Manages dynamic priority adjustments for queue items based on server lifecycle events.
//! When an MCP server starts for a project, related files are bumped to high priority (1).
//! When a server stops, related files are demoted to normal priority (3).
//!
//! ## Session Tracking
//!
//! Session tracking uses the `watch_folders` table with `is_active` and `last_activity_at`:
//! - `activate_project`: Sets is_active=1, updates last_activity_at, bumps priority to HIGH
//! - `heartbeat`: Updates last_activity_at timestamp for active projects
//! - `deactivate_project`: Sets is_active=0, demotes priority when deactivated
//! - `cleanup_orphaned_sessions`: Detects projects without heartbeat for >60s (configurable)
//!
//! ## Priority Levels
//!
//! - HIGH (1): Active agent sessions - items processed first
//! - NORMAL (3): Registered projects without active sessions
//! - LOW (5): Background/inactive projects
//!
//! ## Schema Compliance (WORKSPACE_QDRANT_MCP.md v1.6.7)
//!
//! This module uses only the spec-defined tables:
//! - `watch_folders`: For activity tracking via `is_active` and `last_activity_at`
//! - `unified_queue`: For queue priority management

use chrono::{DateTime, Utc, Duration as ChronoDuration};
use sqlx::SqlitePool;
use wqm_common::timestamps;
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::sync::RwLock;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn, error};

use wqm_common::constants::COLLECTION_PROJECTS;
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

    /// Number of items affected in unified_queue
    pub unified_queue_affected: usize,

    /// Total items affected (same as unified_queue_affected)
    pub total_affected: usize,
}

impl PriorityTransition {
    /// Create a new transition record
    pub fn new(from_priority: u8, to_priority: u8) -> Self {
        Self {
            from_priority,
            to_priority,
            unified_queue_affected: 0,
            total_affected: 0,
        }
    }

    /// Add count from queue update
    pub fn set_count(&mut self, count: usize) {
        self.unified_queue_affected = count;
        self.total_affected = count;
    }
}

/// Session information for tracking active MCP server connections
///
/// Uses `watch_folders.is_active` for activity state per spec.
#[derive(Debug, Clone)]
pub struct SessionInfo {
    /// Watch ID (tenant identifier)
    pub watch_id: String,
    /// Tenant ID (project_id for projects)
    pub tenant_id: String,
    /// Whether this project is currently active (has active sessions)
    pub is_active: bool,
    /// Last heartbeat timestamp
    pub last_activity_at: Option<DateTime<Utc>>,
    /// Current priority level (derived from is_active)
    pub priority: String,
}

/// Result of orphaned session cleanup
#[derive(Debug, Clone)]
pub struct OrphanedSessionCleanup {
    /// Number of projects with orphaned sessions detected
    pub projects_affected: usize,
    /// Total sessions cleaned up (same as projects_affected with boolean model)
    pub sessions_cleaned: i32,
    /// Tenant IDs that were demoted
    pub demoted_projects: Vec<String>,
}

/// Priority Manager for server lifecycle-driven priority adjustments
///
/// Uses only spec-compliant tables:
/// - `watch_folders` for activity tracking
/// - `unified_queue` for priority management
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
    /// Updates unified_queue in a single transaction.
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

        // Update unified_queue (the only queue table per spec)
        let query = r#"
            UPDATE unified_queue
            SET priority = ?1,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE tenant_id = ?2
              AND branch = ?3
              AND priority = ?4
              AND status = 'pending'
        "#;

        let result = sqlx::query(query)
            .bind(to_priority as i32)
            .bind(tenant_id)
            .bind(branch)
            .bind(from_priority as i32)
            .execute(&self.db_pool)
            .await?;

        let affected = result.rows_affected() as usize;

        // Create transition record
        let mut transition = PriorityTransition::new(from_priority, to_priority);
        transition.set_count(affected);

        // Log results
        if transition.total_affected == 0 {
            debug!(
                "No items found with priority {} for tenant_id='{}', branch='{}'",
                from_priority, tenant_id, branch
            );
        } else {
            info!(
                "Priority transition complete: {} items updated in unified_queue",
                transition.total_affected
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
            SELECT COUNT(*) as total
            FROM unified_queue
            WHERE tenant_id = ?1
              AND branch = ?2
              AND priority = ?3
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
    // Session Tracking Methods (using watch_folders.is_active)
    // =========================================================================

    /// Activate a project (mark as having active sessions)
    ///
    /// Sets is_active=1 and updates last_activity_at timestamp.
    /// Also bumps queue priority to HIGH for this project.
    ///
    /// # Arguments
    /// * `tenant_id` - Tenant identifier (project_id)
    /// * `_branch` - Git branch name (for queue priority updates)
    ///
    /// # Returns
    /// true if project was activated, false if not found
    pub async fn register_session(
        &self,
        tenant_id: &str,
        _branch: &str,
    ) -> PriorityResult<i32> {
        if tenant_id.is_empty() {
            return Err(PriorityError::EmptyParameter);
        }

        let now = Utc::now();

        // Start transaction
        let mut tx = self.db_pool.begin().await?;

        // Update watch_folders to mark as active
        let update_query = r#"
            UPDATE watch_folders
            SET is_active = 1,
                last_activity_at = ?1,
                updated_at = ?1
            WHERE tenant_id = ?2
              AND collection = ?3
        "#;

        let result = sqlx::query(update_query)
            .bind(timestamps::format_utc(&now))
            .bind(tenant_id)
            .bind(COLLECTION_PROJECTS)
            .execute(&mut *tx)
            .await?;

        if result.rows_affected() == 0 {
            tx.rollback().await?;
            return Err(PriorityError::ProjectNotFound(tenant_id.to_string()));
        }

        // Bump queue priorities for this project
        sqlx::query(
            r#"
            UPDATE unified_queue
            SET priority = ?1,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE tenant_id = ?2
              AND priority > ?1
              AND status = 'pending'
            "#,
        )
        .bind(priority::HIGH as i32)
        .bind(tenant_id)
        .execute(&mut *tx)
        .await?;

        tx.commit().await?;

        // Record session metrics
        METRICS.session_started(tenant_id, "high");

        info!(
            "Session registered for project {}: marked as active",
            tenant_id
        );

        // Return 1 to indicate active (maintains API compatibility)
        Ok(1)
    }

    /// Deactivate a project (mark as no active sessions)
    ///
    /// Sets is_active=0 and demotes queue priority from HIGH to NORMAL.
    ///
    /// # Arguments
    /// * `tenant_id` - Tenant identifier (project_id)
    /// * `_branch` - Git branch name (for queue priority updates)
    ///
    /// # Returns
    /// 0 to indicate no active sessions
    pub async fn unregister_session(
        &self,
        tenant_id: &str,
        _branch: &str,
    ) -> PriorityResult<i32> {
        if tenant_id.is_empty() {
            return Err(PriorityError::EmptyParameter);
        }

        // Start transaction
        let mut tx = self.db_pool.begin().await?;

        // Check if project exists
        let exists: Option<i32> = sqlx::query_scalar(
            "SELECT 1 FROM watch_folders WHERE tenant_id = ?1 AND collection = ?2 LIMIT 1"
        )
            .bind(tenant_id)
            .bind(COLLECTION_PROJECTS)
            .fetch_optional(&mut *tx)
            .await?;

        if exists.is_none() {
            tx.rollback().await?;
            return Err(PriorityError::ProjectNotFound(tenant_id.to_string()));
        }

        // Update watch_folders to mark as inactive
        sqlx::query(
            r#"
            UPDATE watch_folders
            SET is_active = 0,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE tenant_id = ?1
              AND collection = ?2
            "#,
        )
        .bind(tenant_id)
        .bind(COLLECTION_PROJECTS)
        .execute(&mut *tx)
        .await?;

        // Demote queue priorities from HIGH to NORMAL
        sqlx::query(
            r#"
            UPDATE unified_queue
            SET priority = ?1,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE tenant_id = ?2
              AND priority = ?3
              AND status = 'pending'
            "#,
        )
        .bind(priority::NORMAL as i32)
        .bind(tenant_id)
        .bind(priority::HIGH as i32)
        .execute(&mut *tx)
        .await?;

        tx.commit().await?;

        // Record session end metrics
        METRICS.session_ended(tenant_id, "normal", 0.0);

        info!(
            "Session unregistered for project {}: marked as inactive",
            tenant_id
        );

        Ok(0)
    }

    /// Update heartbeat timestamp for a project
    ///
    /// Called periodically by MCP servers to indicate they're still alive.
    /// Updates the last_activity_at timestamp to the current time.
    ///
    /// # Arguments
    /// * `tenant_id` - Tenant identifier (project_id)
    ///
    /// # Returns
    /// true if heartbeat was recorded, false if project not found or not active
    pub async fn heartbeat(&self, tenant_id: &str) -> PriorityResult<bool> {
        if tenant_id.is_empty() {
            return Err(PriorityError::EmptyParameter);
        }

        // Measure heartbeat latency
        let start = Instant::now();

        let now = Utc::now();

        let result = sqlx::query(
            r#"
            UPDATE watch_folders
            SET last_activity_at = ?1,
                updated_at = ?1
            WHERE tenant_id = ?2
              AND collection = ?3
              AND is_active = 1
            "#,
        )
        .bind(timestamps::format_utc(&now))
        .bind(tenant_id)
        .bind(COLLECTION_PROJECTS)
        .execute(&self.db_pool)
        .await?;

        let updated = result.rows_affected() > 0;

        // Record heartbeat latency metric
        let latency_secs = start.elapsed().as_secs_f64();
        if updated {
            METRICS.heartbeat_processed(tenant_id, latency_secs);
            debug!("Heartbeat received for project {} (latency: {:.3}s)", tenant_id, latency_secs);
        } else {
            warn!(
                "Heartbeat for project {} ignored (not active or not found)",
                tenant_id
            );
        }

        Ok(updated)
    }

    /// Get session info for a project
    ///
    /// # Arguments
    /// * `tenant_id` - Tenant identifier (project_id)
    ///
    /// # Returns
    /// SessionInfo if project exists, None otherwise
    pub async fn get_session_info(&self, tenant_id: &str) -> PriorityResult<Option<SessionInfo>> {
        let query = r#"
            SELECT watch_id, tenant_id, is_active, last_activity_at
            FROM watch_folders
            WHERE tenant_id = ?1
              AND collection = ?2
            LIMIT 1
        "#;

        let row = sqlx::query(query)
            .bind(tenant_id)
            .bind(COLLECTION_PROJECTS)
            .fetch_optional(&self.db_pool)
            .await?;

        if let Some(row) = row {
            use sqlx::Row;
            let is_active: i32 = row.try_get("is_active").unwrap_or(0);
            let last_activity_str: Option<String> = row.try_get("last_activity_at").ok();
            let last_activity_at = last_activity_str
                .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
                .map(|dt| dt.with_timezone(&Utc));

            let priority = if is_active != 0 { "high" } else { "normal" };

            Ok(Some(SessionInfo {
                watch_id: row.try_get("watch_id")?,
                tenant_id: row.try_get("tenant_id")?,
                is_active: is_active != 0,
                last_activity_at,
                priority: priority.to_string(),
            }))
        } else {
            Ok(None)
        }
    }

    /// Cleanup orphaned sessions
    ///
    /// Detects projects with is_active=1 but last_activity_at older than
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
        let cutoff_str = timestamps::format_utc(&cutoff);

        // Start transaction
        let mut tx = self.db_pool.begin().await?;

        // Find orphaned projects (active but stale heartbeat)
        let orphaned_query = r#"
            SELECT tenant_id
            FROM watch_folders
            WHERE is_active = 1
              AND collection = ?1
              AND last_activity_at IS NOT NULL
              AND last_activity_at < ?2
        "#;

        let rows = sqlx::query(orphaned_query)
            .bind(COLLECTION_PROJECTS)
            .bind(&cutoff_str)
            .fetch_all(&mut *tx)
            .await?;

        let mut demoted_projects = Vec::new();

        for row in &rows {
            use sqlx::Row;
            let tenant_id: String = row.try_get("tenant_id")?;
            demoted_projects.push(tenant_id.clone());

            // Record orphaned session cleanup metrics
            METRICS.session_ended(&tenant_id, "high", 0.0);

            // Reset is_active to 0
            sqlx::query(
                r#"
                UPDATE watch_folders
                SET is_active = 0,
                    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE tenant_id = ?1
                  AND collection = ?2
                "#,
            )
            .bind(&tenant_id)
            .bind(COLLECTION_PROJECTS)
            .execute(&mut *tx)
            .await?;

            // Demote queue priorities for this project
            sqlx::query(
                r#"
                UPDATE unified_queue
                SET priority = ?1,
                    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE tenant_id = ?2
                  AND priority = ?3
                  AND status = 'pending'
                "#,
            )
            .bind(priority::NORMAL as i32)
            .bind(&tenant_id)
            .bind(priority::HIGH as i32)
            .execute(&mut *tx)
            .await?;
        }

        tx.commit().await?;

        let cleanup = OrphanedSessionCleanup {
            projects_affected: demoted_projects.len(),
            sessions_cleaned: demoted_projects.len() as i32,
            demoted_projects: demoted_projects.clone(),
        };

        if cleanup.projects_affected > 0 {
            warn!(
                "Cleaned up {} orphaned sessions: {:?}",
                cleanup.sessions_cleaned,
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
            SELECT watch_id, tenant_id, is_active, last_activity_at
            FROM watch_folders
            WHERE is_active = 1
              AND collection = ?1
            ORDER BY last_activity_at DESC
        "#;

        let rows = sqlx::query(query)
            .bind(COLLECTION_PROJECTS)
            .fetch_all(&self.db_pool)
            .await?;

        let mut projects = Vec::new();
        for row in rows {
            use sqlx::Row;
            let last_activity_str: Option<String> = row.try_get("last_activity_at").ok();
            let last_activity_at = last_activity_str
                .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
                .map(|dt| dt.with_timezone(&Utc));

            projects.push(SessionInfo {
                watch_id: row.try_get("watch_id")?,
                tenant_id: row.try_get("tenant_id")?,
                is_active: true,
                last_activity_at,
                priority: "high".to_string(),
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
                                        "Session monitor cleanup: {} orphaned sessions",
                                        cleanup.sessions_cleaned
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
    use crate::watch_folders_schema::CREATE_WATCH_FOLDERS_SQL;
    use crate::unified_queue_schema::CREATE_UNIFIED_QUEUE_SQL;
    use tempfile::tempdir;

    /// Helper to create test database with spec-compliant schema
    async fn setup_test_db() -> (SqlitePool, tempfile::TempDir) {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_priority.db");
        let db_url = format!("sqlite://{}?mode=rwc", db_path.display());

        let pool = SqlitePool::connect(&db_url).await.unwrap();

        // Initialize spec-compliant schema
        sqlx::query(CREATE_UNIFIED_QUEUE_SQL)
            .execute(&pool)
            .await
            .unwrap();

        sqlx::query(CREATE_WATCH_FOLDERS_SQL)
            .execute(&pool)
            .await
            .unwrap();

        (pool, temp_dir)
    }

    /// Helper to create a test watch folder (project)
    async fn create_test_project(pool: &SqlitePool, tenant_id: &str, path: &str) {
        let watch_id = format!("watch_{}", tenant_id);
        sqlx::query(
            r#"
            INSERT INTO watch_folders (watch_id, path, collection, tenant_id, is_active, created_at, updated_at)
            VALUES (?1, ?2, 'projects', ?3, 0, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            "#,
        )
        .bind(&watch_id)
        .bind(path)
        .bind(tenant_id)
        .execute(pool)
        .await
        .unwrap();
    }

    /// Helper to enqueue a test item
    async fn enqueue_test_item(pool: &SqlitePool, tenant_id: &str, branch: &str, priority: i32) {
        let queue_id = uuid::Uuid::new_v4().to_string();
        let idempotency_key = format!("test_{}_{}", tenant_id, queue_id);
        sqlx::query(
            r#"
            INSERT INTO unified_queue (
                queue_id, item_type, op, tenant_id, collection, priority, status, branch, idempotency_key, payload_json
            ) VALUES (?1, 'file', 'ingest', ?2, 'projects', ?3, 'pending', ?4, ?5, '{}')
            "#,
        )
        .bind(&queue_id)
        .bind(tenant_id)
        .bind(priority)
        .bind(branch)
        .bind(&idempotency_key)
        .execute(pool)
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn test_server_start_bumps_priority() {
        let (pool, _temp_dir) = setup_test_db().await;
        let priority_manager = PriorityManager::new(pool.clone());

        // Enqueue items with normal priority (3)
        enqueue_test_item(&pool, "test-tenant", "main", 3).await;
        enqueue_test_item(&pool, "test-tenant", "main", 3).await;

        // Trigger server start - should bump priority to 1
        let transition = priority_manager
            .on_server_start("test-tenant", "main")
            .await
            .unwrap();

        assert_eq!(transition.from_priority, 3);
        assert_eq!(transition.to_priority, 1);
        assert_eq!(transition.unified_queue_affected, 2);
        assert_eq!(transition.total_affected, 2);

        // Verify items now have priority 1
        let count = priority_manager
            .count_items_with_priority("test-tenant", "main", 1)
            .await
            .unwrap();
        assert_eq!(count, 2);
    }

    #[tokio::test]
    async fn test_server_stop_demotes_priority() {
        let (pool, _temp_dir) = setup_test_db().await;
        let priority_manager = PriorityManager::new(pool.clone());

        // Enqueue items with urgent priority (1)
        enqueue_test_item(&pool, "test-tenant", "main", 1).await;

        // Trigger server stop - should demote priority to 3
        let transition = priority_manager
            .on_server_stop("test-tenant", "main")
            .await
            .unwrap();

        assert_eq!(transition.from_priority, 1);
        assert_eq!(transition.to_priority, 3);
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
        let priority_manager = PriorityManager::new(pool.clone());

        // Enqueue items on different branches
        enqueue_test_item(&pool, "test-tenant", "main", 3).await;
        enqueue_test_item(&pool, "test-tenant", "feature-branch", 3).await;

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

    // =========================================================================
    // Session Tracking Tests (using watch_folders)
    // =========================================================================

    #[tokio::test]
    async fn test_register_session_activates_project() {
        let (pool, _temp_dir) = setup_test_db().await;
        let priority_manager = PriorityManager::new(pool.clone());

        // Create test project
        create_test_project(&pool, "abcd12345678", "/test/project").await;

        // Register session
        let count = priority_manager
            .register_session("abcd12345678", "main")
            .await
            .unwrap();
        assert_eq!(count, 1); // Returns 1 to indicate active

        // Verify session info
        let info = priority_manager
            .get_session_info("abcd12345678")
            .await
            .unwrap()
            .unwrap();
        assert!(info.is_active);
        assert_eq!(info.priority, "high");
    }

    #[tokio::test]
    async fn test_unregister_session_deactivates_project() {
        let (pool, _temp_dir) = setup_test_db().await;
        let priority_manager = PriorityManager::new(pool.clone());

        // Create test project and register session
        create_test_project(&pool, "abcd12345678", "/test/project").await;
        priority_manager.register_session("abcd12345678", "main").await.unwrap();

        // Unregister session
        let count = priority_manager
            .unregister_session("abcd12345678", "main")
            .await
            .unwrap();
        assert_eq!(count, 0); // Returns 0 to indicate inactive

        // Verify demoted to normal priority
        let info = priority_manager
            .get_session_info("abcd12345678")
            .await
            .unwrap()
            .unwrap();
        assert!(!info.is_active);
        assert_eq!(info.priority, "normal");
    }

    #[tokio::test]
    async fn test_register_session_bumps_queue_priority() {
        let (pool, _temp_dir) = setup_test_db().await;
        let priority_manager = PriorityManager::new(pool.clone());

        // Create test project
        create_test_project(&pool, "abcd12345678", "/test/project").await;

        // Enqueue items with normal priority
        enqueue_test_item(&pool, "abcd12345678", "main", priority::NORMAL as i32).await;

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
        let (pool, _temp_dir) = setup_test_db().await;
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
        assert!(info_after.last_activity_at >= info_before.last_activity_at);
    }

    #[tokio::test]
    async fn test_heartbeat_ignored_without_active_session() {
        let (pool, _temp_dir) = setup_test_db().await;
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
        let (pool, _temp_dir) = setup_test_db().await;
        let priority_manager = PriorityManager::new(pool.clone());

        // Create test project
        create_test_project(&pool, "abcd12345678", "/test/project").await;

        // Register session
        priority_manager.register_session("abcd12345678", "main").await.unwrap();

        // Manually set last_activity_at to old timestamp to simulate orphaned session
        let old_time = Utc::now() - ChronoDuration::minutes(5);
        sqlx::query("UPDATE watch_folders SET last_activity_at = ?1 WHERE tenant_id = ?2")
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
        assert!(!info.is_active);
        assert_eq!(info.priority, "normal");
    }

    #[tokio::test]
    async fn test_no_orphaned_sessions_with_recent_heartbeat() {
        let (pool, _temp_dir) = setup_test_db().await;
        let priority_manager = PriorityManager::new(pool.clone());

        // Create test project and register session (sets last_activity_at to now)
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
        assert!(info.is_active);
        assert_eq!(info.priority, "high");
    }

    #[tokio::test]
    async fn test_get_high_priority_projects() {
        let (pool, _temp_dir) = setup_test_db().await;
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
        let tenant_ids: Vec<_> = high_priority.iter().map(|p| &p.tenant_id).collect();
        assert!(tenant_ids.contains(&&"project1aaaa".to_string()));
        assert!(tenant_ids.contains(&&"project2bbbb".to_string()));
        assert!(!tenant_ids.contains(&&"project3cccc".to_string()));
    }

    #[tokio::test]
    async fn test_register_nonexistent_project_fails() {
        let (pool, _temp_dir) = setup_test_db().await;
        let priority_manager = PriorityManager::new(pool);

        // Try to register session for non-existent project
        let result = priority_manager
            .register_session("nonexistent12", "main")
            .await;

        assert!(matches!(result, Err(PriorityError::ProjectNotFound(_))));
    }

    #[tokio::test]
    async fn test_unregister_nonexistent_project_fails() {
        let (pool, _temp_dir) = setup_test_db().await;
        let priority_manager = PriorityManager::new(pool);

        // Try to unregister session for non-existent project
        let result = priority_manager
            .unregister_session("nonexistent12", "main")
            .await;

        assert!(matches!(result, Err(PriorityError::ProjectNotFound(_))));
    }

    #[tokio::test]
    async fn test_priority_constants() {
        assert_eq!(priority::HIGH, 1);
        assert_eq!(priority::NORMAL, 3);
        assert_eq!(priority::LOW, 5);
    }
}
