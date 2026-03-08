//! Project registration logic
//!
//! Handles the RegisterProject gRPC flow: validation, project ID generation,
//! queue enqueue for new projects, session registration for existing projects,
//! LSP server startup, and activity inheritance.

use std::path::{Path, PathBuf};

use tonic::Status;
use tracing::{debug, error, info, warn};

use crate::proto::{RegisterProjectRequest, RegisterProjectResponse};

use workspace_qdrant_core::{
    project_disambiguation::ProjectIdCalculator, ItemType, ProjectPayload, QueueManager,
    UnifiedQueueOp,
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
            info!(
                "Resolved git root: {} -> {}",
                req.path, effective_path_str
            );
        }

        // Detect git remote from the resolved path when the request lacks one
        let effective_git_remote = if req
            .git_remote
            .as_ref()
            .map_or(true, |r| r.is_empty())
        {
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

        let project_id = self.resolve_project_id(
            &req,
            &effective_path,
            effective_git_remote.as_deref(),
        )?;

        info!(
            "Registering project: id={}, path={}, name={:?}, priority={}",
            project_id, effective_path_str, req.name, effective_priority
        );

        let action = self
            .determine_registration_action(&project_id, &req, is_high_priority)
            .await?;

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
                })
            }
            RegistrationAction::ExistingNoop => Ok(RegisterProjectResponse {
                created: false,
                project_id,
                priority: effective_priority.to_string(),
                is_active: false,
                newly_registered: false,
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
            let generated =
                calculator.calculate(effective_path, effective_git_remote, None);
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
            Ok(RegistrationAction::NotFoundSkipped)
        } else {
            self.enqueue_new_project(project_id, req, is_high_priority)
                .await
        }
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

        let effective_git_remote = if req
            .git_remote
            .as_ref()
            .map_or(true, |r| r.is_empty())
        {
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
}
