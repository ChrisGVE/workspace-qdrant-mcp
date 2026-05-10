//! Worktree auto-registration and metadata lookup
//!
//! Handles detection and registration of git worktrees as additional watch roots
//! for existing projects. When a session starts in a worktree whose main project
//! is already registered, the worktree is automatically registered under the same
//! tenant_id.

use std::path::Path;

use chrono::Utc;
use tonic::Status;
use tracing::{debug, error, info};
use uuid::Uuid;

use workspace_qdrant_core::{
    daemon_state::WatchFolderRecord,
    git::{detect_git_status, find_main_worktree_path, resolve_git_dir},
    project_disambiguation::ProjectIdCalculator,
};
use wqm_common::constants::COLLECTION_PROJECTS;
use wqm_common::project_id::detect_git_remote;

use super::ProjectServiceImpl;

/// Worktree metadata for populating response fields on existing entries.
pub(super) struct WatchMetadata {
    pub is_worktree: bool,
    pub watch_path: Option<String>,
}

/// Result of a worktree auto-registration attempt.
pub(super) enum WorktreeResult {
    /// The path is a worktree and was successfully auto-registered.
    Registered {
        canonical_path: String,
        is_high_priority: bool,
    },
    /// The path is not a worktree, or the main project is not registered.
    NotApplicable,
}

impl ProjectServiceImpl {
    /// Attempt worktree auto-registration when the project is not found.
    ///
    /// Checks if the path is a git worktree and whether the main working tree
    /// is already registered. If so, creates a new watch folder entry for the
    /// worktree that shares the same tenant_id as the main project.
    pub(super) async fn try_worktree_auto_register(
        &self,
        effective_path: &Path,
        is_high_priority: bool,
    ) -> Result<WorktreeResult, Status> {
        let git_status = detect_git_status(effective_path);
        if !git_status.is_worktree {
            return Ok(WorktreeResult::NotApplicable);
        }

        // Resolve the worktree's git directory and find the main worktree root
        let git_dir = resolve_git_dir(effective_path).ok_or_else(|| {
            debug!(
                path = %effective_path.display(),
                "Worktree detected but could not resolve git dir"
            );
            Status::not_found("Worktree git directory not found")
        })?;

        let main_root = match find_main_worktree_path(&git_dir) {
            Some(p) => p,
            None => {
                debug!(
                    path = %effective_path.display(),
                    "Could not resolve main worktree path"
                );
                return Ok(WorktreeResult::NotApplicable);
            }
        };

        // Compute the tenant_id for the main worktree
        let main_remote = detect_git_remote(&main_root);
        let calculator = ProjectIdCalculator::new();
        let main_tenant_id = calculator.calculate(&main_root, main_remote.as_deref(), None);

        // Look up the main project's watch folder by tenant_id
        let main_record = match self.find_watch_folder_by_tenant(&main_tenant_id).await? {
            Some(r) => r,
            None => {
                info!(
                    path = %effective_path.display(),
                    main_tenant_id = %main_tenant_id,
                    "Worktree detected but main project not registered"
                );
                return Ok(WorktreeResult::NotApplicable);
            }
        };

        let canonical_path =
            std::fs::canonicalize(effective_path).unwrap_or_else(|_| effective_path.to_path_buf());
        let canonical_str = canonical_path.to_string_lossy().to_string();

        self.store_worktree_record(
            &canonical_str,
            &main_tenant_id,
            &main_record,
            git_status.commit_hash,
            is_high_priority,
        )
        .await?;

        info!(
            "Auto-registering worktree: {} for project {}",
            canonical_str, main_tenant_id
        );

        Ok(WorktreeResult::Registered {
            canonical_path: canonical_str,
            is_high_priority,
        })
    }

    async fn store_worktree_record(
        &self,
        canonical_str: &str,
        main_tenant_id: &str,
        main_record: &WatchFolderRecord,
        commit_hash: Option<String>,
        is_high_priority: bool,
    ) -> Result<(), Status> {
        let now = Utc::now();
        let record = WatchFolderRecord {
            watch_id: Uuid::new_v4().to_string(),
            path: canonical_str.to_string(),
            collection: COLLECTION_PROJECTS.to_string(),
            tenant_id: main_tenant_id.to_string(),
            parent_watch_id: None,
            submodule_path: None,
            git_remote_url: main_record.git_remote_url.clone(),
            remote_hash: main_record.remote_hash.clone(),
            disambiguation_path: main_record.disambiguation_path.clone(),
            is_active: is_high_priority,
            last_activity_at: if is_high_priority { Some(now) } else { None },
            is_paused: false,
            pause_start_time: None,
            is_archived: false,
            last_commit_hash: commit_hash,
            is_git_tracked: true,
            is_worktree: true,
            main_worktree_watch_id: Some(main_record.watch_id.clone()),
            library_mode: None,
            follow_symlinks: main_record.follow_symlinks,
            enabled: true,
            cleanup_on_disable: main_record.cleanup_on_disable,
            created_at: now,
            updated_at: now,
            last_scan: None,
        };
        self.state_manager
            .store_watch_folder(&record)
            .await
            .map_err(|e| {
                error!("Failed to store worktree watch folder: {e}");
                Status::internal(format!("Failed to store worktree watch folder: {e}"))
            })
    }

    /// Find a watch folder entry by tenant_id in the projects collection.
    pub(super) async fn find_watch_folder_by_tenant(
        &self,
        tenant_id: &str,
    ) -> Result<Option<WatchFolderRecord>, Status> {
        self.state_manager
            .get_watch_folder_by_tenant_id(tenant_id, COLLECTION_PROJECTS)
            .await
            .map_err(|e| {
                error!("Database error looking up watch folder by tenant: {e}");
                Status::internal(format!("Database error: {e}"))
            })
    }

    /// Look up worktree metadata (is_worktree, watch_path) for an existing project.
    ///
    /// Queries the watch_folders table for the given path or tenant_id to populate
    /// the `is_worktree` and `watch_path` response fields.
    pub(super) async fn lookup_watch_metadata(
        &self,
        project_id: &str,
        path: &str,
    ) -> WatchMetadata {
        let row: Option<(i32, String)> = sqlx::query_as(
            r#"SELECT is_worktree, path FROM watch_folders
               WHERE collection = ?1
                 AND (path = ?2 OR tenant_id = ?3)
               ORDER BY CASE WHEN path = ?2 THEN 0 ELSE 1 END
               LIMIT 1"#,
        )
        .bind(COLLECTION_PROJECTS)
        .bind(path)
        .bind(project_id)
        .fetch_optional(&self.db_pool)
        .await
        .unwrap_or(None);

        match row {
            Some((is_wt, watch_path)) => WatchMetadata {
                is_worktree: is_wt != 0,
                watch_path: Some(watch_path),
            },
            None => WatchMetadata {
                is_worktree: false,
                watch_path: None,
            },
        }
    }
}
