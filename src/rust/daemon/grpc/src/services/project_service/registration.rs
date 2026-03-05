//! Project registration logic
//!
//! Handles the RegisterProject gRPC flow: validation, project ID generation,
//! queue enqueue for new projects, session registration for existing projects,
//! LSP server startup, and activity inheritance.

use std::path::PathBuf;

use tonic::Status;
use tracing::{debug, error, info, warn};

use crate::proto::{RegisterProjectRequest, RegisterProjectResponse};

use workspace_qdrant_core::{
    project_disambiguation::ProjectIdCalculator, ItemType, ProjectPayload, QueueManager,
    UnifiedQueueOp,
};
use wqm_common::constants::COLLECTION_PROJECTS;

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

impl ProjectServiceImpl {
    /// Execute the register_project business logic
    pub(crate) async fn handle_register_project(
        &self,
        req: RegisterProjectRequest,
    ) -> Result<RegisterProjectResponse, Status> {
        if req.path.is_empty() {
            return Err(Status::invalid_argument("path cannot be empty"));
        }

        let effective_priority = req
            .priority
            .as_deref()
            .filter(|p| !p.is_empty())
            .unwrap_or("normal");
        let is_high_priority = effective_priority == "high";

        let project_id = self.resolve_project_id(&req)?;

        info!(
            "Registering project: id={}, path={}, name={:?}, priority={}",
            project_id, req.path, req.name, effective_priority
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
                self.activate_project(&project_id, &req.path).await;
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
                    self.activate_project(&project_id, &req.path).await;
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

    /// Resolve or validate the project_id from the request
    fn resolve_project_id(&self, req: &RegisterProjectRequest) -> Result<String, Status> {
        if req.project_id.is_empty() {
            let calculator = ProjectIdCalculator::new();
            let path = std::path::Path::new(&req.path);
            let git_remote = req.git_remote.as_deref();
            let generated = calculator.calculate(path, git_remote, None);
            info!("Generated project_id for {}: {}", req.path, generated);
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
        let queue_manager = QueueManager::new(self.db_pool.clone());
        let payload = ProjectPayload {
            project_root: req.path.clone(),
            git_remote: req.git_remote.clone(),
            project_type: None,
            old_tenant_id: None,
            is_active: Some(is_high_priority),
        };
        let payload_json = serde_json::to_string(&payload)
            .unwrap_or_else(|_| format!(r#"{{"project_root":"{}"}}"#, req.path));

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
                    debug!(
                        project_id = %project_id,
                        "No watch folders found for activity inheritance (project may not be watched yet)"
                    );
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
}
