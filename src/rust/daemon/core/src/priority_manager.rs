//! Priority Manager Module
//!
//! Manages project activation state based on MCP server lifecycle events.
//! Priority ordering is computed at dequeue time by JOINing `watch_folders.is_active`
//! and collection type — the stored `priority` column in `unified_queue` is NOT used
//! for dequeue ordering (see `queue_operations::dequeue_unified`).
//!
//! This module manages only `watch_folders.is_active` and `last_activity_at`:
//! - `register_session`: Sets is_active=1, updates last_activity_at
//! - `heartbeat`: Updates last_activity_at timestamp for active projects
//! - `unregister_session`: Sets is_active=0
//! - `set_priority`: Maps "high"/"normal" to is_active=1/0
//! - `cleanup_orphaned_sessions`: Detects stale active projects (>60s without heartbeat)
//!
//! ## Schema Compliance (WORKSPACE_QDRANT_MCP.md v1.6.7)
//!
//! This module uses only the `watch_folders` table for activity state.
//! Queue ordering is computed at dequeue time, not stored.

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

/// Priority levels for the queue system — re-exported from wqm_common
pub use wqm_common::constants::priority;

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

    // =========================================================================
    // Session Tracking Methods (using watch_folders.is_active)
    // =========================================================================

    /// Activate a project (mark as having active sessions)
    ///
    /// Sets is_active=1 and updates last_activity_at timestamp.
    /// Queue ordering is computed at dequeue time based on is_active,
    /// so no queue updates are needed here.
    ///
    /// # Arguments
    /// * `tenant_id` - Tenant identifier (project_id)
    /// * `_branch` - Git branch name (unused, kept for API compatibility)
    ///
    /// # Returns
    /// 1 if project was activated, error if not found
    pub async fn register_session(
        &self,
        tenant_id: &str,
        _branch: &str,
    ) -> PriorityResult<i32> {
        if tenant_id.is_empty() {
            return Err(PriorityError::EmptyParameter);
        }

        let now = Utc::now();

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
            .execute(&self.db_pool)
            .await?;

        if result.rows_affected() == 0 {
            return Err(PriorityError::ProjectNotFound(tenant_id.to_string()));
        }

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
    /// Sets is_active=0. Queue ordering is computed at dequeue time based on
    /// is_active, so no queue updates are needed here.
    ///
    /// # Arguments
    /// * `tenant_id` - Tenant identifier (project_id)
    /// * `_branch` - Git branch name (unused, kept for API compatibility)
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

        // Check if project exists
        let exists: Option<i32> = sqlx::query_scalar(
            "SELECT 1 FROM watch_folders WHERE tenant_id = ?1 AND collection = ?2 LIMIT 1"
        )
            .bind(tenant_id)
            .bind(COLLECTION_PROJECTS)
            .fetch_optional(&self.db_pool)
            .await?;

        if exists.is_none() {
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
        .execute(&self.db_pool)
        .await?;

        // Record session end metrics
        METRICS.session_ended(tenant_id, "normal", 0.0);

        info!(
            "Session unregistered for project {}: marked as inactive",
            tenant_id
        );

        Ok(0)
    }

    /// Set project priority explicitly
    ///
    /// Maps a priority string ("high"/"normal") to watch_folders.is_active (1/0).
    /// Queue ordering is computed at dequeue time based on is_active.
    ///
    /// # Arguments
    /// * `tenant_id` - Tenant identifier (project_id)
    /// * `priority_str` - "high" or "normal"
    ///
    /// # Returns
    /// (previous_priority_string, 0) — queue_items_updated is always 0 since
    /// queue ordering is computed at dequeue time, not stored.
    pub async fn set_priority(
        &self,
        tenant_id: &str,
        priority_str: &str,
    ) -> PriorityResult<(String, i32)> {
        if tenant_id.is_empty() {
            return Err(PriorityError::EmptyParameter);
        }

        let new_is_active = match priority_str {
            "high" => 1,
            "normal" => 0,
            other => return Err(PriorityError::InvalidPriority(
                other.parse::<i32>().unwrap_or(-1)
            )),
        };

        // Get current state
        let current_active: Option<i32> = sqlx::query_scalar(
            "SELECT is_active FROM watch_folders WHERE tenant_id = ?1 AND collection = ?2 LIMIT 1"
        )
            .bind(tenant_id)
            .bind(COLLECTION_PROJECTS)
            .fetch_optional(&self.db_pool)
            .await?;

        let current_active = match current_active {
            Some(v) => v,
            None => {
                return Err(PriorityError::ProjectNotFound(tenant_id.to_string()));
            }
        };

        let previous_priority = if current_active == 1 { "high" } else { "normal" };

        // Update watch_folders
        let now = timestamps::format_utc(&chrono::Utc::now());
        sqlx::query(
            r#"
            UPDATE watch_folders
            SET is_active = ?1,
                last_activity_at = ?2,
                updated_at = ?2
            WHERE tenant_id = ?3
              AND collection = ?4
            "#,
        )
        .bind(new_is_active)
        .bind(&now)
        .bind(tenant_id)
        .bind(COLLECTION_PROJECTS)
        .execute(&self.db_pool)
        .await?;

        info!(
            "Set priority for project {}: {} -> {}",
            tenant_id, previous_priority, priority_str
        );

        Ok((previous_priority.to_string(), 0))
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

            // Reset is_active to 0 (dequeue ordering is computed at query time)
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

    #[tokio::test]
    async fn test_empty_parameters_error() {
        let (pool, _temp_dir) = setup_test_db().await;
        let priority_manager = PriorityManager::new(pool);

        // Empty tenant_id
        let result = priority_manager.register_session("", "main").await;
        assert!(matches!(result, Err(PriorityError::EmptyParameter)));
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
    async fn test_register_session_does_not_modify_queue_priority() {
        let (pool, _temp_dir) = setup_test_db().await;
        let priority_manager = PriorityManager::new(pool.clone());

        // Create test project
        create_test_project(&pool, "abcd12345678", "/test/project").await;

        // Enqueue items with normal priority
        let queue_id = uuid::Uuid::new_v4().to_string();
        sqlx::query(
            r#"INSERT INTO unified_queue (
                queue_id, item_type, op, tenant_id, collection, priority, status, branch, idempotency_key, payload_json
            ) VALUES (?1, 'file', 'ingest', 'abcd12345678', 'projects', ?2, 'pending', 'main', ?3, '{}')"#,
        )
        .bind(&queue_id)
        .bind(priority::NORMAL)
        .bind(format!("test_{}", queue_id))
        .execute(&pool)
        .await
        .unwrap();

        // Register session — should NOT modify stored queue priority
        priority_manager.register_session("abcd12345678", "main").await.unwrap();

        // Verify queue item still has its original stored priority (unchanged)
        let stored_priority: i32 = sqlx::query_scalar(
            "SELECT priority FROM unified_queue WHERE queue_id = ?1"
        )
        .bind(&queue_id)
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(stored_priority, priority::NORMAL);
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

    #[tokio::test]
    async fn test_set_priority_normal_to_high() {
        let (pool, _temp_dir) = setup_test_db().await;
        let priority_manager = PriorityManager::new(pool.clone());

        // Create test project (starts inactive/normal)
        create_test_project(&pool, "abcd12345678", "/test/project").await;

        // Set priority to high
        let (previous, queue_updated) = priority_manager
            .set_priority("abcd12345678", "high")
            .await
            .unwrap();

        assert_eq!(previous, "normal");
        // Queue items are not updated — ordering is computed at dequeue time
        assert_eq!(queue_updated, 0);

        // Verify project is now active
        let info = priority_manager.get_session_info("abcd12345678").await.unwrap().unwrap();
        assert!(info.is_active);
        assert_eq!(info.priority, "high");
    }

    #[tokio::test]
    async fn test_set_priority_high_to_normal() {
        let (pool, _temp_dir) = setup_test_db().await;
        let priority_manager = PriorityManager::new(pool.clone());

        // Create test project and activate it
        create_test_project(&pool, "abcd12345678", "/test/project").await;
        priority_manager.register_session("abcd12345678", "main").await.unwrap();

        // Set priority to normal
        let (previous, queue_updated) = priority_manager
            .set_priority("abcd12345678", "normal")
            .await
            .unwrap();

        assert_eq!(previous, "high");
        // Queue items are not updated — ordering is computed at dequeue time
        assert_eq!(queue_updated, 0);

        // Verify project is now inactive
        let info = priority_manager.get_session_info("abcd12345678").await.unwrap().unwrap();
        assert!(!info.is_active);
        assert_eq!(info.priority, "normal");
    }

    #[tokio::test]
    async fn test_set_priority_nonexistent_project() {
        let (pool, _temp_dir) = setup_test_db().await;
        let priority_manager = PriorityManager::new(pool);

        let result = priority_manager.set_priority("nonexistent12", "high").await;
        assert!(matches!(result, Err(PriorityError::ProjectNotFound(_))));
    }
}
