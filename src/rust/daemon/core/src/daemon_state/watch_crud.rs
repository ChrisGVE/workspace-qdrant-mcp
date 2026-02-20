//! Watch folder CRUD operations.

use wqm_common::constants::COLLECTION_PROJECTS;
use wqm_common::timestamps;

use super::{DaemonStateManager, DaemonStateResult, WatchFolderRecord};

impl DaemonStateManager {
    // ========================================================================
    // Watch Folders methods (spec-defined watch_folders table)
    // ========================================================================

    /// Store or update a watch folder record
    pub async fn store_watch_folder(&self, record: &WatchFolderRecord) -> DaemonStateResult<()> {
        sqlx::query(
            r#"
            INSERT OR REPLACE INTO watch_folders (
                watch_id, path, collection, tenant_id,
                parent_watch_id, submodule_path,
                git_remote_url, remote_hash, disambiguation_path, is_active, last_activity_at,
                is_paused, pause_start_time, is_archived, last_commit_hash, is_git_tracked,
                library_mode,
                follow_symlinks, enabled, cleanup_on_disable,
                created_at, updated_at, last_scan
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23)
            "#,
        )
        .bind(&record.watch_id)
        .bind(&record.path)
        .bind(&record.collection)
        .bind(&record.tenant_id)
        .bind(&record.parent_watch_id)
        .bind(&record.submodule_path)
        .bind(&record.git_remote_url)
        .bind(&record.remote_hash)
        .bind(&record.disambiguation_path)
        .bind(record.is_active as i32)
        .bind(wqm_common::timestamps::format_optional_utc(&record.last_activity_at))
        .bind(record.is_paused as i32)
        .bind(wqm_common::timestamps::format_optional_utc(&record.pause_start_time))
        .bind(record.is_archived as i32)
        .bind(&record.last_commit_hash)
        .bind(record.is_git_tracked as i32)
        .bind(&record.library_mode)
        .bind(record.follow_symlinks as i32)
        .bind(record.enabled as i32)
        .bind(record.cleanup_on_disable as i32)
        .bind(timestamps::format_utc(&record.created_at))
        .bind(timestamps::format_utc(&record.updated_at))
        .bind(wqm_common::timestamps::format_optional_utc(&record.last_scan))
        .execute(&self.pool)
        .await?;

        // Task 14: Auto-populate junction table when parent_watch_id is set
        if let Some(ref parent_id) = record.parent_watch_id {
            sqlx::query(
                r#"
                INSERT OR IGNORE INTO watch_folder_submodules
                    (parent_watch_id, child_watch_id, submodule_path, created_at)
                VALUES (?1, ?2, ?3, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                "#,
            )
            .bind(parent_id)
            .bind(&record.watch_id)
            .bind(record.submodule_path.as_deref().unwrap_or(""))
            .execute(&self.pool)
            .await?;
        }

        Ok(())
    }

    /// Get a watch folder by ID
    pub async fn get_watch_folder(&self, watch_id: &str) -> DaemonStateResult<Option<WatchFolderRecord>> {
        let row = sqlx::query(
            r#"
            SELECT watch_id, path, collection, tenant_id,
                   parent_watch_id, submodule_path,
                   git_remote_url, remote_hash, disambiguation_path, is_active, last_activity_at,
                   is_paused, pause_start_time, is_archived, last_commit_hash, is_git_tracked,
                   library_mode,
                   follow_symlinks, enabled, cleanup_on_disable,
                   created_at, updated_at, last_scan
            FROM watch_folders WHERE watch_id = ?1
            "#,
        )
        .bind(watch_id)
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = row {
            Ok(Some(self.row_to_watch_folder(&row)?))
        } else {
            Ok(None)
        }
    }

    /// List watch folders with optional collection filter
    pub async fn list_watch_folders(&self, collection_filter: Option<&str>, enabled_only: bool) -> DaemonStateResult<Vec<WatchFolderRecord>> {
        let base_query = r#"
            SELECT watch_id, path, collection, tenant_id,
                   parent_watch_id, submodule_path,
                   git_remote_url, remote_hash, disambiguation_path, is_active, last_activity_at,
                   is_paused, pause_start_time, is_archived, last_commit_hash, is_git_tracked,
                   library_mode,
                   follow_symlinks, enabled, cleanup_on_disable,
                   created_at, updated_at, last_scan
            FROM watch_folders
        "#;

        let query = match (collection_filter, enabled_only) {
            (Some(_), true) => format!("{} WHERE collection = ?1 AND enabled = 1 ORDER BY created_at", base_query),
            (Some(_), false) => format!("{} WHERE collection = ?1 ORDER BY created_at", base_query),
            (None, true) => format!("{} WHERE enabled = 1 ORDER BY created_at", base_query),
            (None, false) => format!("{} ORDER BY created_at", base_query),
        };

        let rows = if let Some(collection) = collection_filter {
            sqlx::query(&query)
                .bind(collection)
                .fetch_all(&self.pool)
                .await?
        } else {
            sqlx::query(&query)
                .fetch_all(&self.pool)
                .await?
        };

        let mut results = Vec::new();
        for row in rows {
            results.push(self.row_to_watch_folder(&row)?);
        }

        Ok(results)
    }

    /// List active project watch folders
    pub async fn list_active_projects(&self) -> DaemonStateResult<Vec<WatchFolderRecord>> {
        let rows = sqlx::query(
            r#"
            SELECT watch_id, path, collection, tenant_id,
                   parent_watch_id, submodule_path,
                   git_remote_url, remote_hash, disambiguation_path, is_active, last_activity_at,
                   is_paused, pause_start_time, is_archived, last_commit_hash, is_git_tracked,
                   library_mode,
                   follow_symlinks, enabled, cleanup_on_disable,
                   created_at, updated_at, last_scan
            FROM watch_folders
            WHERE collection = ?1 AND is_active = 1 AND enabled = 1
            ORDER BY last_activity_at DESC
            "#,
        )
        .bind(COLLECTION_PROJECTS)
        .fetch_all(&self.pool)
        .await?;

        let mut results = Vec::new();
        for row in rows {
            results.push(self.row_to_watch_folder(&row)?);
        }

        Ok(results)
    }

    /// Get watch folder by tenant_id (project_id for projects, library_name for libraries)
    /// Returns the first matching top-level watch (not a submodule)
    pub async fn get_watch_folder_by_tenant_id(
        &self,
        tenant_id: &str,
        collection: &str,
    ) -> DaemonStateResult<Option<WatchFolderRecord>> {
        let row = sqlx::query(
            r#"
            SELECT watch_id, path, collection, tenant_id,
                   parent_watch_id, submodule_path,
                   git_remote_url, remote_hash, disambiguation_path, is_active, last_activity_at,
                   is_paused, pause_start_time, is_archived, last_commit_hash, is_git_tracked,
                   library_mode,
                   follow_symlinks, enabled, cleanup_on_disable,
                   created_at, updated_at, last_scan
            FROM watch_folders
            WHERE tenant_id = ?1 AND collection = ?2 AND parent_watch_id IS NULL
            LIMIT 1
            "#,
        )
        .bind(tenant_id)
        .bind(collection)
        .fetch_optional(&self.pool)
        .await?;

        match row {
            Some(row) => Ok(Some(self.row_to_watch_folder(&row)?)),
            None => Ok(None),
        }
    }

    /// Update watch folder enabled status
    pub async fn set_watch_folder_enabled(&self, watch_id: &str, enabled: bool) -> DaemonStateResult<bool> {
        let result = sqlx::query(
            r#"
            UPDATE watch_folders
            SET enabled = ?1, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE watch_id = ?2
            "#,
        )
        .bind(enabled as i32)
        .bind(watch_id)
        .execute(&self.pool)
        .await?;

        Ok(result.rows_affected() > 0)
    }

    /// Update last scan timestamp for a watch folder
    pub async fn update_watch_folder_last_scan(&self, watch_id: &str) -> DaemonStateResult<bool> {
        let result = sqlx::query(
            r#"
            UPDATE watch_folders
            SET last_scan = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE watch_id = ?1
            "#,
        )
        .bind(watch_id)
        .execute(&self.pool)
        .await?;

        Ok(result.rows_affected() > 0)
    }

    /// Remove watch folder
    pub async fn remove_watch_folder(&self, watch_id: &str) -> DaemonStateResult<bool> {
        let result = sqlx::query("DELETE FROM watch_folders WHERE watch_id = ?1")
            .bind(watch_id)
            .execute(&self.pool)
            .await?;

        Ok(result.rows_affected() > 0)
    }

    /// Check if a path is already registered as a project
    pub async fn is_path_registered(&self, path: &str) -> DaemonStateResult<bool> {
        let count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM watch_folders WHERE path = ?1 AND collection = ?2"
        )
        .bind(path)
        .bind(COLLECTION_PROJECTS)
        .fetch_one(&self.pool)
        .await?;

        Ok(count > 0)
    }
}
