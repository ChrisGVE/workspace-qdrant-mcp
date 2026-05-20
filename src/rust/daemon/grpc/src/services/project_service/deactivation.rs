//! Project deactivation logic
//!
//! Handles deprioritization: unregistering sessions, updating watch_folders
//! activity state, and triggering LSP shutdown (immediate or deferred).
//!
//! Supports path-scoped deactivation: when `watch_path` is set, only the
//! specific watch folder at that path is decremented rather than all entries
//! for the tenant.

use tonic::Status;
use tracing::{debug, error, info, warn};

use crate::proto::{DeprioritizeProjectRequest, DeprioritizeProjectResponse};
use crate::validation::extract_canonical_path;

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

        // Validate watch_path as CanonicalPath when present and non-empty.
        let watch_path = match req.watch_path.as_deref().filter(|p| !p.is_empty()) {
            Some(raw) => {
                let canonical = extract_canonical_path!(raw.to_string(), "watch_path")?;
                Some(canonical.into_string())
            }
            None => None,
        };

        if let Some(ref path) = watch_path {
            self.deprioritize_by_path(&req.project_id, path).await
        } else {
            self.deprioritize_tenant_wide(&req.project_id).await
        }
    }

    /// Tenant-wide deprioritization (original behaviour).
    ///
    /// Decrements `is_active` for every watch folder sharing the `tenant_id`.
    async fn deprioritize_tenant_wide(
        &self,
        project_id: &str,
    ) -> Result<DeprioritizeProjectResponse, Status> {
        info!("Deprioritizing project (tenant-wide): {}", project_id);

        match self
            .priority_manager
            .unregister_session(project_id, "main")
            .await
        {
            Ok(active_flag) => {
                let is_active = active_flag > 0;
                let new_priority = if is_active { "high" } else { "normal" };

                // unregister_session already decremented is_active;
                // only handle side effects when all sessions are gone
                if !is_active {
                    self.handle_lsp_shutdown(project_id).await;

                    if let Some(ref signal) = self.watch_refresh_signal {
                        signal.notify_one();
                        debug!(
                            project_id = %project_id,
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

    /// Path-scoped deprioritization for worktree sessions.
    ///
    /// Decrements `is_active` only for the watch folder at `watch_path`,
    /// leaving other entries for the same tenant untouched.
    async fn deprioritize_by_path(
        &self,
        project_id: &str,
        watch_path: &str,
    ) -> Result<DeprioritizeProjectResponse, Status> {
        info!(
            "Deprioritizing project (path-scoped): {} at {}",
            project_id, watch_path
        );

        match self
            .priority_manager
            .unregister_session_by_path(project_id, watch_path)
            .await
        {
            Ok(active_flag) => {
                let is_active = active_flag > 0;
                let new_priority = if is_active { "high" } else { "normal" };

                // When this specific path reaches zero, signal a watch refresh
                // so the watcher can stop monitoring it if needed.
                if !is_active {
                    if let Some(ref signal) = self.watch_refresh_signal {
                        signal.notify_one();
                        debug!(
                            project_id = %project_id,
                            watch_path = %watch_path,
                            "Signaled WatchManager refresh for path deactivation"
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
                warn!(
                    "Watch folder not found for deprioritization: {} at {}",
                    id, watch_path
                );
                Err(Status::not_found(format!("Watch folder not found: {id}")))
            }
            Err(e) => {
                error!("Failed to deprioritize project by path: {e}");
                Err(Status::internal(format!("Failed to deprioritize: {e}")))
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
