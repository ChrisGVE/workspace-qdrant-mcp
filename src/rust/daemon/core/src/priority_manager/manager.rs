//! PriorityManager: session lifecycle and priority management.

use chrono::{Duration as ChronoDuration, Utc};
use sqlx::SqlitePool;
use std::time::Instant;
use tracing::{debug, info, warn};
use wqm_common::timestamps;

use crate::lifecycle::WatchFolderLifecycle;
use crate::metrics::METRICS;
use wqm_common::constants::COLLECTION_PROJECTS;

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

    /// Increment is_active for a project session.
    ///
    /// Increments the session counter and updates last_activity_at.
    /// Queue ordering is computed at dequeue time based on is_active,
    /// so no queue updates are needed here.
    pub async fn register_session(&self, tenant_id: &str, _session_tag: &str) -> PriorityResult<i32> {
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

    /// Decrement is_active for a project session.
    ///
    /// Returns the is_active value after decrement. The caller uses this
    /// to decide whether side effects (LSP shutdown, watch refresh) should
    /// fire — they only fire when the count reaches 0.
    pub async fn unregister_session(&self, tenant_id: &str, _session_tag: &str) -> PriorityResult<i32> {
        if tenant_id.is_empty() {
            return Err(PriorityError::EmptyParameter);
        }

        // Check if project exists
        let exists: Option<i32> = sqlx::query_scalar(
            "SELECT 1 FROM watch_folders WHERE tenant_id = ?1 AND collection = ?2 LIMIT 1",
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

        // Read back the updated value to inform the caller
        let remaining: i32 = sqlx::query_scalar(
            "SELECT COALESCE(MAX(is_active), 0) FROM watch_folders \
             WHERE tenant_id = ?1 AND collection = ?2",
        )
        .bind(tenant_id)
        .bind(COLLECTION_PROJECTS)
        .fetch_one(&self.db_pool)
        .await?;

        // Record session end metrics
        METRICS.session_ended(tenant_id, "normal", 0.0);

        info!(
            "Session unregistered for project {}: is_active={}",
            tenant_id, remaining
        );

        Ok(remaining)
    }

    /// Deactivate a single watch folder by `(tenant_id, path)`.
    ///
    /// Decrements `is_active` by 1 (clamped to 0) for only the watch folder
    /// at the specified path, leaving other entries for the same tenant
    /// untouched. Returns the `is_active` value after the decrement.
    pub async fn unregister_session_by_path(
        &self,
        tenant_id: &str,
        path: &str,
    ) -> PriorityResult<i32> {
        if tenant_id.is_empty() || path.is_empty() {
            return Err(PriorityError::EmptyParameter);
        }

        let lifecycle = WatchFolderLifecycle::new(self.db_pool.clone());

        // Check existence before mutating
        let current = lifecycle
            .get_is_active_by_tenant_and_path(tenant_id, path)
            .await?;

        if current.is_none() {
            return Err(PriorityError::ProjectNotFound(format!(
                "{tenant_id} at path {path}"
            )));
        }

        lifecycle
            .deactivate_by_tenant_and_path(tenant_id, path)
            .await?;

        // Read back the updated value
        let updated = lifecycle
            .get_is_active_by_tenant_and_path(tenant_id, path)
            .await?
            .unwrap_or(0);

        METRICS.session_ended(tenant_id, "normal", 0.0);

        info!(
            "Session unregistered for project {} at path {}: is_active={}",
            tenant_id, path, updated
        );

        Ok(updated)
    }

    /// Set project priority explicitly
    ///
    /// Maps a priority string ("high"/"normal") to a session count increment/decrement.
    /// "high" increments is_active, "normal" decrements it (floor 0).
    /// Queue ordering is computed at dequeue time based on is_active.
    pub async fn set_priority(
        &self,
        tenant_id: &str,
        priority_str: &str,
    ) -> PriorityResult<(String, i32)> {
        if tenant_id.is_empty() {
            return Err(PriorityError::EmptyParameter);
        }

        // Validate priority string early
        if priority_str != "high" && priority_str != "normal" {
            return Err(PriorityError::InvalidPriority(
                priority_str.parse::<i32>().unwrap_or(-1),
            ));
        }

        // Get current state
        let current_active: Option<i32> = sqlx::query_scalar(
            "SELECT is_active FROM watch_folders WHERE tenant_id = ?1 AND collection = ?2 LIMIT 1",
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

        let previous_priority = if current_active > 0 { "high" } else { "normal" };

        // Delegate is_active mutation to WatchFolderLifecycle
        let lifecycle = WatchFolderLifecycle::new(self.db_pool.clone());
        if priority_str == "high" {
            lifecycle
                .activate_by_tenant(tenant_id, COLLECTION_PROJECTS)
                .await?;
        } else {
            lifecycle
                .deactivate_by_tenant(tenant_id, COLLECTION_PROJECTS)
                .await?;
        }

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

        // A heartbeat refreshes the activity timestamp for a project that has at least
        // one active session (is_active > 0). With reference counting, is_active
        // accurately reflects the number of live sessions — a heartbeat cannot
        // resurrect a project with 0 sessions, because that would bypass the register/
        // unregister lifecycle. The race condition that required `SET is_active = 1`
        // (Task 518) is resolved by reference counting: one session ending no longer
        // drops the count to 0 while another session is still alive.
        let result = sqlx::query(
            r#"
            UPDATE watch_folders
            SET last_activity_at = ?1,
                updated_at = ?1
            WHERE tenant_id = ?2
              AND collection = ?3
              AND is_active > 0
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
            debug!(
                "Heartbeat received for project {} (latency: {:.3}s)",
                tenant_id, latency_secs
            );
        } else {
            warn!(
                "Heartbeat for project {} not found (tenant_id={})",
                tenant_id, tenant_id
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
                cleanup.sessions_cleaned, cleanup.demoted_projects
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
            WHERE is_active > 0
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
