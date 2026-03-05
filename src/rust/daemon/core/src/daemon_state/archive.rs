//! Archive/unarchive operations for watch folders (Task 582).

use tracing::info;

use super::{DaemonStateManager, DaemonStateResult, WatchFolderRecord};

impl DaemonStateManager {
    /// Get all submodule watch folders for a given parent project via junction table (Task 14).
    pub async fn get_submodules_for_project(
        &self,
        parent_watch_id: &str,
    ) -> DaemonStateResult<Vec<WatchFolderRecord>> {
        let rows = sqlx::query(
            r#"
            SELECT wf.watch_id, wf.path, wf.collection, wf.tenant_id,
                   wf.parent_watch_id, wf.submodule_path,
                   wf.git_remote_url, wf.remote_hash, wf.disambiguation_path, wf.is_active, wf.last_activity_at,
                   wf.is_paused, wf.pause_start_time, wf.is_archived, wf.last_commit_hash, wf.is_git_tracked,
                   wf.library_mode,
                   wf.follow_symlinks, wf.enabled, wf.cleanup_on_disable,
                   wf.created_at, wf.updated_at, wf.last_scan
            FROM watch_folders wf
            INNER JOIN watch_folder_submodules j ON wf.watch_id = j.child_watch_id
            WHERE j.parent_watch_id = ?1
            "#,
        )
        .bind(parent_watch_id)
        .fetch_all(&self.pool)
        .await?;

        let mut results = Vec::new();
        for row in rows {
            results.push(self.row_to_watch_folder(&row)?);
        }
        Ok(results)
    }

    /// Count how many OTHER active (non-archived) projects reference the same submodule.
    ///
    /// Used to determine if a submodule can be safely archived when its parent
    /// is archived. Only counts references whose parent project is also active
    /// (not archived). This ensures that when a parent is archived but its
    /// submodule was skipped, it doesn't block archiving the same submodule
    /// under a different parent.
    pub async fn count_active_submodule_references(
        &self,
        remote_hash: &str,
        git_remote_url: &str,
        excluding_parent_watch_id: &str,
    ) -> DaemonStateResult<i64> {
        let count: i64 = sqlx::query_scalar(
            r#"
            SELECT COUNT(*) FROM watch_folders sub
            INNER JOIN watch_folder_submodules j ON sub.watch_id = j.child_watch_id
            WHERE sub.remote_hash = ?1
              AND sub.git_remote_url = ?2
              AND j.parent_watch_id != ?3
              AND COALESCE(sub.is_archived, 0) = 0
              AND EXISTS (
                SELECT 1 FROM watch_folders parent
                WHERE parent.watch_id = j.parent_watch_id
                  AND COALESCE(parent.is_archived, 0) = 0
              )
            "#,
        )
        .bind(remote_hash)
        .bind(git_remote_url)
        .bind(excluding_parent_watch_id)
        .fetch_one(&self.pool)
        .await?;
        Ok(count)
    }

    /// Archive a single watch folder by setting is_archived = 1.
    ///
    /// Does NOT modify parent_watch_id (preserved as historical fact).
    /// Does NOT delete any Qdrant data (archived projects remain searchable).
    /// Returns true if a row was updated, false if already archived or not found.
    pub async fn archive_watch_folder(&self, watch_id: &str) -> DaemonStateResult<bool> {
        let result = sqlx::query(
            r#"
            UPDATE watch_folders
            SET is_archived = 1,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE watch_id = ?1 AND COALESCE(is_archived, 0) = 0
            "#,
        )
        .bind(watch_id)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected() > 0)
    }

    /// Unarchive a single watch folder by setting is_archived = 0.
    ///
    /// Returns true if a row was updated, false if not archived or not found.
    pub async fn unarchive_watch_folder(&self, watch_id: &str) -> DaemonStateResult<bool> {
        let result = sqlx::query(
            r#"
            UPDATE watch_folders
            SET is_archived = 0,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE watch_id = ?1 AND COALESCE(is_archived, 0) = 1
            "#,
        )
        .bind(watch_id)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected() > 0)
    }

    /// Archive a project and its submodules with cross-reference safety.
    ///
    /// For each submodule of the parent project:
    /// - If other active projects reference the same remote, skip archiving
    /// - If no other active references exist, archive alongside parent
    ///
    /// Returns (archived_ids, skipped_ids).
    pub async fn archive_project_with_submodules(
        &self,
        parent_watch_id: &str,
    ) -> DaemonStateResult<(Vec<String>, Vec<String>)> {
        // Archive the parent first
        self.archive_watch_folder(parent_watch_id).await?;

        let submodules = self.get_submodules_for_project(parent_watch_id).await?;
        let mut archived = Vec::new();
        let mut skipped = Vec::new();

        for sub in &submodules {
            // Check if submodule has matching remote_hash and git_remote_url for cross-ref check
            let remote_hash = sub.remote_hash.as_deref().unwrap_or("");
            let git_remote_url = sub.git_remote_url.as_deref().unwrap_or("");

            if remote_hash.is_empty() && git_remote_url.is_empty() {
                // No remote info, archive with parent
                if self.archive_watch_folder(&sub.watch_id).await? {
                    archived.push(sub.watch_id.clone());
                }
                continue;
            }

            let other_refs = self
                .count_active_submodule_references(remote_hash, git_remote_url, parent_watch_id)
                .await?;

            if other_refs > 0 {
                info!(
                    "Skipping archive of shared submodule {}: {} other active reference(s)",
                    sub.watch_id, other_refs
                );
                skipped.push(sub.watch_id.clone());
            } else {
                if self.archive_watch_folder(&sub.watch_id).await? {
                    info!("Archived submodule {} with parent", sub.watch_id);
                    archived.push(sub.watch_id.clone());
                }
            }
        }

        Ok((archived, skipped))
    }
}
