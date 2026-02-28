//! Project deactivation logic
//!
//! Handles deprioritization: unregistering sessions, updating watch_folders
//! activity state, and triggering LSP shutdown (immediate or deferred).

use tonic::Status;
use tracing::{debug, error, info, warn};

use crate::proto::{DeprioritizeProjectRequest, DeprioritizeProjectResponse};

use super::ProjectServiceImpl;

impl ProjectServiceImpl {
    /// Execute the deprioritize_project business logic
    pub(crate) async fn handle_deprioritize_project(
        &self,
        req: DeprioritizeProjectRequest,
    ) -> Result<DeprioritizeProjectResponse, Status> {
        if req.project_id.is_empty() {
            return Err(Status::invalid_argument("project_id cannot be empty"));
        }

        info!("Deprioritizing project: {}", req.project_id);

        match self
            .priority_manager
            .unregister_session(&req.project_id, "main")
            .await
        {
            Ok(active_flag) => {
                let is_active = active_flag > 0;
                let new_priority = if is_active { "high" } else { "normal" };

                if !is_active {
                    self.deactivate_project(&req.project_id).await;
                    self.handle_lsp_shutdown(&req.project_id).await;

                    if let Some(ref signal) = self.watch_refresh_signal {
                        signal.notify_one();
                        debug!(
                            project_id = %req.project_id,
                            "Signaled WatchManager refresh for project deactivation"
                        );
                    }
                }

                Ok(DeprioritizeProjectResponse {
                    success: true,
                    is_active,
                    new_priority: new_priority.to_string(),
                })
            }
            Err(workspace_qdrant_core::PriorityError::ProjectNotFound(id)) => {
                warn!("Project not found for deprioritization: {}", id);
                Err(Status::not_found(format!("Project not found: {id}")))
            }
            Err(e) => {
                error!("Failed to deprioritize project: {e}");
                Err(Status::internal(format!("Failed to deprioritize: {e}")))
            }
        }
    }

    /// Set watch_folders is_active=false for project and all submodules
    async fn deactivate_project(&self, project_id: &str) {
        match self
            .state_manager
            .deactivate_project_by_tenant_id(project_id)
            .await
        {
            Ok((affected, watch_id)) => {
                if affected > 0 {
                    info!(
                        project_id = %project_id,
                        watch_id = ?watch_id,
                        affected_folders = affected,
                        "Deactivated project watch folders (activity inheritance)"
                    );
                }
            }
            Err(e) => {
                warn!(
                    project_id = %project_id,
                    error = %e,
                    "Failed to deactivate project watch folders (non-critical)"
                );
            }
        }
    }

    /// Handle LSP shutdown: immediate if queue empty and no delay, otherwise deferred
    async fn handle_lsp_shutdown(&self, project_id: &str) {
        let queue_depth = super::lsp_lifecycle::get_project_queue_depth(&self.db_pool, project_id)
            .await
            .unwrap_or(0);

        let has_queue_items = queue_depth > 0;
        let has_delay = self.deactivation_delay_secs > 0;

        if !has_queue_items && !has_delay {
            info!(
                project_id = %project_id,
                "No active sessions, queue empty, no delay - stopping LSP servers immediately"
            );
            if let Err(e) = super::lsp_lifecycle::stop_project_lsp_servers(
                &self.lsp_manager,
                &self.language_detector,
                project_id,
            )
            .await
            {
                warn!(
                    project_id = %project_id,
                    error = %e,
                    "Failed to stop LSP servers (non-critical)"
                );
            }
        } else {
            info!(
                project_id = %project_id,
                queue_depth = queue_depth,
                deactivation_delay_secs = self.deactivation_delay_secs,
                "Scheduling deferred LSP shutdown"
            );
            super::lsp_lifecycle::schedule_deferred_shutdown(
                &self.pending_shutdowns,
                self.deactivation_delay_secs,
                project_id,
                has_queue_items,
            )
            .await;
        }
    }
}
