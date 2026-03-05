//! Project lifecycle operations: activate, deactivate, heartbeat.

use tracing::{info, warn};
use wqm_common::constants::COLLECTION_PROJECTS;

use super::{DaemonStateManager, DaemonStateResult};
use crate::lifecycle::WatchFolderLifecycle;

impl DaemonStateManager {
    /// Activate a project and all descendant submodules via recursive junction table traversal (Task 14)
    ///
    /// Delegates to `WatchFolderLifecycle::activate_project_group` -- the single
    /// code path for all `is_active` mutations.
    pub async fn activate_project_group(&self, watch_id: &str) -> DaemonStateResult<u64> {
        let lifecycle = WatchFolderLifecycle::new(self.pool.clone());
        Ok(lifecycle.activate_project_group(watch_id).await?)
    }

    /// Deactivate a project and all descendant submodules via recursive junction table traversal (Task 14)
    ///
    /// Delegates to `WatchFolderLifecycle::deactivate_project_group` -- the single
    /// code path for all `is_active` mutations.
    pub async fn deactivate_project_group(&self, watch_id: &str) -> DaemonStateResult<u64> {
        let lifecycle = WatchFolderLifecycle::new(self.pool.clone());
        Ok(lifecycle.deactivate_project_group(watch_id).await?)
    }

    /// Activate a project by tenant_id (project_id)
    /// Finds the watch folder and activates the entire project group
    /// Returns (rows_affected, watch_id used)
    pub async fn activate_project_by_tenant_id(
        &self,
        tenant_id: &str,
    ) -> DaemonStateResult<(u64, Option<String>)> {
        // Find the main watch folder for this tenant
        let watch_folder = self
            .get_watch_folder_by_tenant_id(tenant_id, COLLECTION_PROJECTS)
            .await?;

        match watch_folder {
            Some(folder) => {
                let affected = self.activate_project_group(&folder.watch_id).await?;
                Ok((affected, Some(folder.watch_id)))
            }
            None => Ok((0, None)),
        }
    }

    /// Deactivate a project by tenant_id (project_id)
    /// Finds the watch folder and deactivates the entire project group
    /// Returns (rows_affected, watch_id used)
    pub async fn deactivate_project_by_tenant_id(
        &self,
        tenant_id: &str,
    ) -> DaemonStateResult<(u64, Option<String>)> {
        // Find the main watch folder for this tenant
        let watch_folder = self
            .get_watch_folder_by_tenant_id(tenant_id, COLLECTION_PROJECTS)
            .await?;

        match watch_folder {
            Some(folder) => {
                let affected = self.deactivate_project_group(&folder.watch_id).await?;
                Ok((affected, Some(folder.watch_id)))
            }
            None => Ok((0, None)),
        }
    }

    /// Update last_activity_at for a project and all descendant submodules via recursive junction table traversal (Task 14)
    pub async fn heartbeat_project_group(&self, watch_id: &str) -> DaemonStateResult<u64> {
        let result = sqlx::query(
            r#"
            WITH RECURSIVE descendants AS (
                SELECT ?1 AS watch_id
                UNION
                SELECT j.child_watch_id FROM watch_folder_submodules j
                JOIN descendants d ON j.parent_watch_id = d.watch_id
            )
            UPDATE watch_folders
            SET last_activity_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE watch_id IN (SELECT watch_id FROM descendants)
            "#,
        )
        .bind(watch_id)
        .execute(&self.pool)
        .await?;

        Ok(result.rows_affected())
    }

    /// Update heartbeat by tenant_id (project_id)
    /// Finds the watch folder and updates last_activity_at for the entire project group
    /// Returns (rows_affected, watch_id used)
    pub async fn heartbeat_project_by_tenant_id(
        &self,
        tenant_id: &str,
    ) -> DaemonStateResult<(u64, Option<String>)> {
        // Find the main watch folder for this tenant
        let watch_folder = self
            .get_watch_folder_by_tenant_id(tenant_id, COLLECTION_PROJECTS)
            .await?;

        match watch_folder {
            Some(folder) => {
                let affected = self.heartbeat_project_group(&folder.watch_id).await?;
                Ok((affected, Some(folder.watch_id)))
            }
            None => Ok((0, None)),
        }
    }

    /// Deactivate projects that have been inactive for longer than the timeout.
    ///
    /// Queries for active projects whose `last_activity_at` is older than
    /// `timeout_secs` seconds ago. For each timed-out project, deactivates
    /// the entire project group (including submodules) via `deactivate_project_group`.
    ///
    /// Returns the number of project groups deactivated.
    pub async fn deactivate_inactive_projects(&self, timeout_secs: i64) -> DaemonStateResult<u64> {
        // Find active parent projects (not submodules) whose last activity exceeds timeout
        let stale_watches: Vec<String> = sqlx::query_scalar(
            r#"
            SELECT watch_id FROM watch_folders
            WHERE is_active = 1
              AND collection = ?1
              AND parent_watch_id IS NULL
              AND last_activity_at IS NOT NULL
              AND (julianday('now') - julianday(last_activity_at)) * 86400 > ?2
            "#,
        )
        .bind(COLLECTION_PROJECTS)
        .bind(timeout_secs)
        .fetch_all(&self.pool)
        .await?;

        if stale_watches.is_empty() {
            return Ok(0);
        }

        info!(
            "Inactivity timeout: deactivating {} project(s) inactive for >{}s",
            stale_watches.len(),
            timeout_secs
        );

        let mut deactivated = 0u64;
        for watch_id in &stale_watches {
            match self.deactivate_project_group(watch_id).await {
                Ok(affected) => {
                    info!(
                        "Deactivated project group {} ({} watch folders)",
                        watch_id, affected
                    );
                    deactivated += 1;
                }
                Err(e) => {
                    warn!("Failed to deactivate project group {}: {}", watch_id, e);
                }
            }
        }

        Ok(deactivated)
    }
}
