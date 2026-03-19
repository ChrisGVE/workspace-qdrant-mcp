//! Project registration logic
//!
//! Handles the RegisterProject gRPC flow: validation, project ID generation,
//! queue enqueue for new projects, session registration for existing projects,
//! LSP server startup, activity inheritance, and worktree auto-registration.

use std::path::{Path, PathBuf};

use chrono::Utc;
use tonic::Status;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::proto::{RegisterProjectRequest, RegisterProjectResponse};

use workspace_qdrant_core::{
    daemon_state::WatchFolderRecord,
    git::{detect_git_status, find_main_worktree_path, resolve_git_dir},
    project_disambiguation::ProjectIdCalculator,
    ItemType, ProjectPayload, QueueManager, UnifiedQueueOp,
};
use wqm_common::constants::COLLECTION_PROJECTS;
use wqm_common::project_id::detect_git_remote;

use super::ProjectServiceImpl;

/// Result of determining what action to take for a registration request
enum RegistrationAction {
    /// Project exists, session registered (high priority)
    ExistingActivated,
    /// Project exists, no activation change (normal priority)
    ExistingNoop,
    /// Project not found, caller did not request auto-registration
    NotFoundSkipped,
    /// New project enqueued for creation
    NewlyEnqueued { is_high_priority: bool },
    /// Worktree auto-registered (main project exists, worktree path is new)
    WorktreeAutoRegistered {
        canonical_path: String,
        is_high_priority: bool,
    },
}

/// Resolve the git repository root from a given path by walking upward.
/// Returns the original path if no `.git` directory is found.
fn resolve_git_root(path: &Path) -> PathBuf {
    let mut current = path.to_path_buf();
    loop {
        if current.join(".git").exists() {
            return current;
        }
        match current.parent() {
            Some(parent) if parent != current => {
                current = parent.to_path_buf();
            }
            _ => break,
        }
    }
    // No .git found, return original path
    path.to_path_buf()
}

impl ProjectServiceImpl {
    /// Execute the register_project business logic
    pub(crate) async fn handle_register_project(
        &self,
        req: RegisterProjectRequest,
    ) -> Result<RegisterProjectResponse, Status> {
        if req.path.is_empty() {
            return Err(Status::invalid_argument("path cannot be empty"));
        }

        // Resolve the git repository root so that subfolder registrations
        // use the correct project root and tenant ID.
        let effective_path = resolve_git_root(Path::new(&req.path));
        let effective_path_str = effective_path.to_string_lossy().to_string();

        if effective_path_str != req.path {
            info!("Resolved git root: {} -> {}", req.path, effective_path_str);
        }

        // Detect git remote from the resolved path when the request lacks one
        let effective_git_remote = if req.git_remote.as_ref().map_or(true, |r| r.is_empty()) {
            detect_git_remote(&effective_path)
        } else {
            req.git_remote.clone()
        };

        let effective_priority = req
            .priority
            .as_deref()
            .filter(|p| !p.is_empty())
            .unwrap_or("normal");
        let is_high_priority = effective_priority == "high";

        let project_id =
            self.resolve_project_id(&req, &effective_path, effective_git_remote.as_deref())?;

        info!(
            "Registering project: id={}, path={}, name={:?}, priority={}",
            project_id, effective_path_str, req.name, effective_priority
        );

        let action = self
            .determine_registration_action(&project_id, &req, is_high_priority, &effective_path)
            .await?;

        // Look up worktree metadata for existing entries
        let watch_meta = self
            .lookup_watch_metadata(&project_id, &effective_path_str)
            .await;

        match action {
            RegistrationAction::NotFoundSkipped => {
                info!(
                    "Project not registered, skipping auto-registration: {}",
                    project_id
                );
                Ok(RegisterProjectResponse {
                    created: false,
                    project_id,
                    priority: "none".to_string(),
                    is_active: false,
                    newly_registered: false,
                    is_worktree: false,
                    watch_path: None,
                })
            }
            RegistrationAction::ExistingActivated => {
                self.activate_project(&project_id, &effective_path_str)
                    .await;
                self.signal_watch_refresh(&project_id);
                Ok(RegisterProjectResponse {
                    created: false,
                    project_id,
                    priority: effective_priority.to_string(),
                    is_active: true,
                    newly_registered: false,
                    is_worktree: watch_meta.is_worktree,
                    watch_path: watch_meta.watch_path,
                })
            }
            RegistrationAction::ExistingNoop => Ok(RegisterProjectResponse {
                created: false,
                project_id,
                priority: effective_priority.to_string(),
                is_active: false,
                newly_registered: false,
                is_worktree: watch_meta.is_worktree,
                watch_path: watch_meta.watch_path,
            }),
            RegistrationAction::NewlyEnqueued {
                is_high_priority: hp,
            } => {
                if hp {
                    self.activate_project(&project_id, &effective_path_str)
                        .await;
                }
                self.signal_watch_refresh(&project_id);
                Ok(RegisterProjectResponse {
                    created: true,
                    project_id,
                    priority: effective_priority.to_string(),
                    is_active: hp,
                    newly_registered: true,
                    is_worktree: false,
                    watch_path: None,
                })
            }
            RegistrationAction::WorktreeAutoRegistered {
                canonical_path,
                is_high_priority: hp,
            } => {
                if hp {
                    self.activate_project(&project_id, &canonical_path).await;
                }
                self.signal_watch_refresh(&project_id);
                Ok(RegisterProjectResponse {
                    created: true,
                    project_id,
                    priority: if hp {
                        "high".to_string()
                    } else {
                        effective_priority.to_string()
                    },
                    is_active: hp,
                    newly_registered: true,
                    is_worktree: true,
                    watch_path: Some(canonical_path),
                })
            }
        }
    }

    /// Signal WatchManager to refresh watch configuration
    fn signal_watch_refresh(&self, project_id: &str) {
        if let Some(ref signal) = self.watch_refresh_signal {
            signal.notify_one();
            debug!(
                project_id = %project_id,
                "Signaled WatchManager refresh for project registration/activation"
            );
        }
    }

    /// Resolve or validate the project_id from the request.
    ///
    /// Uses the resolved git root path and detected remote for ID calculation
    /// so that subfolder registrations produce the same tenant ID as root
    /// registrations.
    fn resolve_project_id(
        &self,
        req: &RegisterProjectRequest,
        effective_path: &Path,
        effective_git_remote: Option<&str>,
    ) -> Result<String, Status> {
        if req.project_id.is_empty() {
            let calculator = ProjectIdCalculator::new();
            let generated = calculator.calculate(effective_path, effective_git_remote, None);
            info!(
                "Generated project_id for {}: {}",
                effective_path.display(),
                generated
            );
            Ok(generated)
        } else {
            let is_local = req.project_id.starts_with("local_");
            let is_hex =
                req.project_id.len() == 12 && req.project_id.chars().all(|c| c.is_ascii_hexdigit());
            if !is_local && !is_hex {
                return Err(Status::invalid_argument(
                    "project_id must be a 12-character hexadecimal string or start with 'local_'",
                ));
            }
            Ok(req.project_id.clone())
        }
    }

    /// Check if a project exists in watch_folders
    async fn project_exists(&self, project_id: &str) -> Result<bool, Status> {
        let existing: Option<(i32,)> = sqlx::query_as(
            "SELECT 1 FROM watch_folders WHERE tenant_id = ?1 AND collection = ?2 LIMIT 1",
        )
        .bind(project_id)
        .bind(COLLECTION_PROJECTS)
        .fetch_optional(&self.db_pool)
        .await
        .map_err(|e| {
            error!("Database error checking project: {e}");
            Status::internal(format!("Database error: {e}"))
        })?;

        Ok(existing.is_some())
    }

    /// Determine registration action based on project state and request parameters
    async fn determine_registration_action(
        &self,
        project_id: &str,
        req: &RegisterProjectRequest,
        is_high_priority: bool,
        effective_path: &Path,
    ) -> Result<RegistrationAction, Status> {
        if self.project_exists(project_id).await? {
            if is_high_priority {
                match self
                    .priority_manager
                    .register_session(project_id, "main")
                    .await
                {
                    Ok(_) => Ok(RegistrationAction::ExistingActivated),
                    Err(e) => {
                        error!("Failed to register session: {e}");
                        Err(Status::internal(format!("Failed to register session: {e}")))
                    }
                }
            } else {
                Ok(RegistrationAction::ExistingNoop)
            }
        } else if !req.register_if_new {
            // Project not found and caller did not request auto-registration.
            // Check if this path is a worktree whose main project is registered.
            self.try_worktree_auto_register(effective_path, is_high_priority)
                .await
        } else {
            self.enqueue_new_project(project_id, req, is_high_priority)
                .await
        }
    }

    /// Attempt worktree auto-registration when the project is not found.
    ///
    /// Checks if the path is a git worktree and whether the main working tree
    /// is already registered. If so, creates a new watch folder entry for the
    /// worktree that shares the same tenant_id as the main project.
    async fn try_worktree_auto_register(
        &self,
        effective_path: &Path,
        is_high_priority: bool,
    ) -> Result<RegistrationAction, Status> {
        let git_status = detect_git_status(effective_path);
        if !git_status.is_worktree {
            return Ok(RegistrationAction::NotFoundSkipped);
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
                return Ok(RegistrationAction::NotFoundSkipped);
            }
        };

        // Compute the tenant_id for the main worktree
        let main_remote = detect_git_remote(&main_root);
        let calculator = ProjectIdCalculator::new();
        let main_tenant_id = calculator.calculate(&main_root, main_remote.as_deref(), None);

        // Look up the main project's watch folder by tenant_id
        let main_watch = self.find_watch_folder_by_tenant(&main_tenant_id).await?;

        let main_record = match main_watch {
            Some(r) => r,
            None => {
                info!(
                    path = %effective_path.display(),
                    main_tenant_id = %main_tenant_id,
                    "Worktree detected but main project not registered"
                );
                return Ok(RegistrationAction::NotFoundSkipped);
            }
        };

        // Canonicalize the worktree path
        let canonical_path =
            std::fs::canonicalize(effective_path).unwrap_or_else(|_| effective_path.to_path_buf());
        let canonical_str = canonical_path.to_string_lossy().to_string();

        // Create the worktree watch folder record
        let now = Utc::now();
        let worktree_record = WatchFolderRecord {
            watch_id: Uuid::new_v4().to_string(),
            path: canonical_str.clone(),
            collection: COLLECTION_PROJECTS.to_string(),
            tenant_id: main_tenant_id.clone(),
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
            last_commit_hash: git_status.commit_hash,
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

        // Insert the worktree record
        self.state_manager
            .store_watch_folder(&worktree_record)
            .await
            .map_err(|e| {
                error!("Failed to store worktree watch folder: {e}");
                Status::internal(format!("Failed to store worktree watch folder: {e}"))
            })?;

        info!(
            "Auto-registering worktree: {} for project {}",
            canonical_str, main_tenant_id
        );

        Ok(RegistrationAction::WorktreeAutoRegistered {
            canonical_path: canonical_str,
            is_high_priority,
        })
    }

    /// Find a watch folder entry by tenant_id in the projects collection.
    async fn find_watch_folder_by_tenant(
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

    /// Enqueue a new project registration via (Tenant, Add)
    async fn enqueue_new_project(
        &self,
        project_id: &str,
        req: &RegisterProjectRequest,
        is_high_priority: bool,
    ) -> Result<RegistrationAction, Status> {
        // Use the resolved git root path for project_root, not req.path.
        // Re-resolve here to stay consistent with handle_register_project.
        let effective_root = resolve_git_root(Path::new(&req.path));
        let effective_root_str = effective_root.to_string_lossy().to_string();

        let effective_git_remote = if req.git_remote.as_ref().map_or(true, |r| r.is_empty()) {
            detect_git_remote(&effective_root)
        } else {
            req.git_remote.clone()
        };

        let queue_manager = QueueManager::new(self.db_pool.clone());
        let payload = ProjectPayload {
            project_root: effective_root_str.clone(),
            git_remote: effective_git_remote,
            project_type: None,
            old_tenant_id: None,
            is_active: Some(is_high_priority),
        };
        let payload_json = serde_json::to_string(&payload)
            .unwrap_or_else(|_| format!(r#"{{"project_root":"{}"}}"#, effective_root_str));

        match queue_manager
            .enqueue_unified(
                ItemType::Tenant,
                UnifiedQueueOp::Add,
                project_id,
                "projects",
                &payload_json,
                None,
                None,
            )
            .await
        {
            Ok((queue_id, _is_new)) => {
                info!(
                    project_id = %project_id,
                    queue_id = %queue_id,
                    "Enqueued project registration (Tenant, Add)"
                );
                Ok(RegistrationAction::NewlyEnqueued { is_high_priority })
            }
            Err(e) => {
                error!("Failed to enqueue project registration: {e}");
                Err(Status::internal(format!("Failed to enqueue project: {e}")))
            }
        }
    }

    /// Activate project: cancel deferred shutdown, start LSP, set watch folders active
    async fn activate_project(&self, project_id: &str, path: &str) {
        if super::lsp_lifecycle::cancel_deferred_shutdown(&self.pending_shutdowns, project_id).await
        {
            debug!(
                project_id = %project_id,
                "Cancelled pending deferred shutdown on project reactivation"
            );
        }

        let project_root = PathBuf::from(path);
        if let Err(e) = super::lsp_lifecycle::start_project_lsp_servers(
            &self.lsp_manager,
            &self.language_detector,
            project_id,
            &project_root,
        )
        .await
        {
            warn!(
                project_id = %project_id,
                error = %e,
                "Failed to start LSP servers (non-critical)"
            );
        }

        match self
            .state_manager
            .activate_project_by_tenant_id(project_id)
            .await
        {
            Ok((affected, watch_id)) => {
                if affected > 0 {
                    info!(
                        project_id = %project_id,
                        watch_id = ?watch_id,
                        affected_folders = affected,
                        "Activated project watch folders (activity inheritance)"
                    );
                } else {
                    warn!(
                        project_id = %project_id,
                        "No watch folders matched tenant_id, trying path-based fallback"
                    );
                    match self.activate_project_by_path(path).await {
                        Ok(true) => info!(
                            project_id = %project_id,
                            path = %path,
                            "Activated project via path fallback"
                        ),
                        Ok(false) => warn!(
                            project_id = %project_id,
                            path = %path,
                            "Could not activate project by path either"
                        ),
                        Err(e) => warn!(
                            project_id = %project_id,
                            path = %path,
                            error = %e,
                            "Path-based activation fallback failed"
                        ),
                    }
                }
            }
            Err(e) => {
                warn!(
                    project_id = %project_id,
                    error = %e,
                    "Failed to activate project watch folders (non-critical)"
                );
            }
        }
    }

    /// Fallback: activate project by looking up watch_folders by path.
    ///
    /// When `activate_project_by_tenant_id` finds no matching rows (e.g. because
    /// the tenant_id in the database differs from the one computed at registration
    /// time), this method searches for a watch folder whose `path` matches or is
    /// a parent of the given `path`, then activates the entire project group.
    async fn activate_project_by_path(&self, path: &str) -> Result<bool, Status> {
        let watch_folder: Option<(String, String)> = sqlx::query_as(
            r#"SELECT watch_id, tenant_id FROM watch_folders
               WHERE collection = ?1
                 AND (?2 = path OR ?2 LIKE path || '/' || '%')
               ORDER BY length(path) DESC
               LIMIT 1"#,
        )
        .bind(COLLECTION_PROJECTS)
        .bind(path)
        .fetch_optional(&self.db_pool)
        .await
        .map_err(|e| {
            error!("Database error in path-based activation: {e}");
            Status::internal(format!("Database error: {e}"))
        })?;

        if let Some((watch_id, existing_tenant)) = watch_folder {
            info!(
                "Found watch folder by path fallback: watch_id={}, tenant_id={}",
                watch_id, existing_tenant
            );
            match self.state_manager.activate_project_group(&watch_id).await {
                Ok(affected) => {
                    info!("Activated {} watch folders via path fallback", affected);
                    Ok(affected > 0)
                }
                Err(e) => {
                    warn!("Failed to activate via path fallback: {}", e);
                    Ok(false)
                }
            }
        } else {
            Ok(false)
        }
    }

    /// Look up worktree metadata (is_worktree, watch_path) for an existing project.
    ///
    /// Queries the watch_folders table for the given path or tenant_id to populate
    /// the `is_worktree` and `watch_path` response fields.
    async fn lookup_watch_metadata(&self, project_id: &str, path: &str) -> WatchMetadata {
        // Try path-exact match first, then fallback to tenant_id
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

/// Worktree metadata for populating response fields on existing entries.
struct WatchMetadata {
    is_worktree: bool,
    watch_path: Option<String>,
}
