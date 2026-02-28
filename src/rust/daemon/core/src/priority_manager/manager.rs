//! PriorityManager: session lifecycle and priority management.

use chrono::{Duration as ChronoDuration, Utc};
use sqlx::SqlitePool;
use std::time::Instant;
use tracing::{debug, info, warn};
use wqm_common::timestamps;

use wqm_common::constants::COLLECTION_PROJECTS;
use crate::lifecycle::WatchFolderLifecycle;
use crate::metrics::METRICS;

use super::{OrphanedSessionCleanup, PriorityError, PriorityResult, SessionInfo};

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
    pub async fn register_session(
        &self,
        tenant_id: &str,
        _branch: &str,
    ) -> PriorityResult<i32> {
        if tenant_id.is_empty() {
            return Err(PriorityError::EmptyParameter);
        }

        // Delegate is_active mutation to WatchFolderLifecycle
        let lifecycle = WatchFolderLifecycle::new(self.db_pool.clone());
        let rows = lifecycle
            .activate_by_tenant(tenant_id, COLLECTION_PROJECTS)
            .await?;

        if rows == 0 {
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

        // Delegate is_active mutation to WatchFolderLifecycle
        let lifecycle = WatchFolderLifecycle::new(self.db_pool.clone());
        lifecycle
            .deactivate_by_tenant(tenant_id, COLLECTION_PROJECTS)
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

        // Delegate is_active mutation to WatchFolderLifecycle
        let lifecycle = WatchFolderLifecycle::new(self.db_pool.clone());
        lifecycle
            .set_active_by_tenant(tenant_id, COLLECTION_PROJECTS, new_is_active == 1)
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
            use chrono::DateTime;
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
    pub async fn cleanup_orphaned_sessions(
        &self,
        timeout_secs: u64,
    ) -> PriorityResult<OrphanedSessionCleanup> {
        let cutoff = Utc::now() - ChronoDuration::seconds(timeout_secs as i64);
        let cutoff_str = timestamps::format_utc(&cutoff);

        // Delegate finding and deactivation to WatchFolderLifecycle
        let lifecycle = WatchFolderLifecycle::new(self.db_pool.clone());

        let demoted_projects = lifecycle
            .find_stale_active_tenants(COLLECTION_PROJECTS, &cutoff_str)
            .await?;

        // Record metrics for each orphaned session
        for tenant_id in &demoted_projects {
            METRICS.session_ended(tenant_id, "high", 0.0);
        }

        // Bulk deactivate in a single transaction
        lifecycle
            .deactivate_orphaned_tenants(&demoted_projects, COLLECTION_PROJECTS)
            .await?;

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
            use chrono::DateTime;
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
