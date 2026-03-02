//! Project mutation operations
//!
//! Handles tenant renaming and project deletion.

use tonic::Status;
use tracing::{debug, error, info, warn};

use crate::proto::{
    DeleteProjectRequest, DeleteProjectResponse, RenameTenantRequest, RenameTenantResponse,
};

use wqm_common::constants::COLLECTION_PROJECTS;
use workspace_qdrant_core::{ItemType, ProjectPayload, QueueManager, UnifiedQueueOp};

use super::ProjectServiceImpl;

impl ProjectServiceImpl {
    /// Execute the rename_tenant business logic
    pub(crate) async fn handle_rename_tenant(
        &self,
        req: RenameTenantRequest,
    ) -> Result<RenameTenantResponse, Status> {
        if req.old_tenant_id.is_empty() {
            return Err(Status::invalid_argument("old_tenant_id cannot be empty"));
        }
        if req.new_tenant_id.is_empty() {
            return Err(Status::invalid_argument("new_tenant_id cannot be empty"));
        }
        if req.old_tenant_id == req.new_tenant_id {
            return Err(Status::invalid_argument(
                "old and new tenant_id are the same",
            ));
        }

        info!(
            "RenameTenant: '{}' -> '{}'",
            req.old_tenant_id, req.new_tenant_id
        );

        let mut tx = self.db_pool.begin().await.map_err(|e| {
            error!("Failed to begin transaction: {e}");
            Status::internal(format!("Transaction failed: {e}"))
        })?;

        let mut total_rows = 0i32;

        total_rows +=
            Self::rename_table(&mut tx, "watch_folders", &req.old_tenant_id, &req.new_tenant_id)
                .await?;
        total_rows +=
            Self::rename_table(&mut tx, "unified_queue", &req.old_tenant_id, &req.new_tenant_id)
                .await?;

        // tracked_files may not exist in all deployments
        match Self::rename_table(
            &mut tx,
            "tracked_files",
            &req.old_tenant_id,
            &req.new_tenant_id,
        )
        .await
        {
            Ok(rows) => total_rows += rows,
            Err(_) => {
                warn!("Failed to update tracked_files (non-fatal)");
            }
        }

        tx.commit().await.map_err(|e| {
            error!("Failed to commit rename transaction: {e}");
            Status::internal(format!("Failed to commit transaction: {e}"))
        })?;

        let message = format!(
            "Renamed tenant '{}' -> '{}': {} SQLite rows updated",
            req.old_tenant_id, req.new_tenant_id, total_rows
        );
        info!("{}", message);

        Ok(RenameTenantResponse {
            success: true,
            sqlite_rows_updated: total_rows,
            message,
        })
    }

    /// Rename tenant_id in a single table within a transaction
    async fn rename_table(
        tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
        table: &str,
        old_id: &str,
        new_id: &str,
    ) -> Result<i32, Status> {
        let query = format!("UPDATE {table} SET tenant_id = ?1 WHERE tenant_id = ?2");
        match sqlx::query(&query)
            .bind(new_id)
            .bind(old_id)
            .execute(&mut **tx)
            .await
        {
            Ok(result) => {
                let rows = result.rows_affected() as i32;
                info!("Updated {} {table} rows", rows);
                Ok(rows)
            }
            Err(e) => {
                error!("Failed to update {table}: {e}");
                Err(Status::internal(format!("Failed to update {table}: {e}")))
            }
        }
    }

    /// Execute the delete_project business logic
    pub(crate) async fn handle_delete_project(
        &self,
        req: DeleteProjectRequest,
    ) -> Result<DeleteProjectResponse, Status> {
        if req.project_id.is_empty() {
            return Err(Status::invalid_argument("project_id cannot be empty"));
        }

        info!(
            "Deleting project: {} (delete_qdrant_data={})",
            req.project_id, req.delete_qdrant_data
        );

        let exists: Option<(String,)> = sqlx::query_as(
            "SELECT watch_id FROM watch_folders WHERE tenant_id = ?1 AND collection = ?2 LIMIT 1",
        )
        .bind(&req.project_id)
        .bind(COLLECTION_PROJECTS)
        .fetch_optional(&self.db_pool)
        .await
        .map_err(|e| {
            error!("Database error checking project: {e}");
            Status::internal(format!("Database error: {e}"))
        })?;

        if exists.is_none() {
            return Err(Status::not_found(format!(
                "Project not found: {}",
                req.project_id
            )));
        }

        super::lsp_lifecycle::cancel_deferred_shutdown(&self.pending_shutdowns, &req.project_id)
            .await;
        if let Err(e) = super::lsp_lifecycle::stop_project_lsp_servers(
            &self.lsp_manager,
            &self.language_detector,
            &req.project_id,
        )
        .await
        {
            debug!("LSP cleanup for deleted project: {e}");
        }

        let queue_manager = QueueManager::new(self.db_pool.clone());
        let payload = ProjectPayload {
            project_root: String::new(),
            git_remote: None,
            project_type: None,
            old_tenant_id: None,
            is_active: None,
        };
        let payload_json =
            serde_json::to_string(&payload).unwrap_or_else(|_| "{}".to_string());

        match queue_manager
            .enqueue_unified(
                ItemType::Tenant,
                UnifiedQueueOp::Delete,
                &req.project_id,
                COLLECTION_PROJECTS,
                &payload_json,
                None,
                None,
            )
            .await
        {
            Ok((queue_id, _is_new)) => {
                let message = format!(
                    "Project {} deletion enqueued (queue_id={})",
                    req.project_id, queue_id
                );
                info!("{}", message);

                Ok(DeleteProjectResponse {
                    success: true,
                    watch_folders_deleted: 0,
                    tracked_files_deleted: 0,
                    qdrant_points_deleted: 0,
                    queue_items_deleted: 0,
                    message,
                })
            }
            Err(e) => {
                error!("Failed to enqueue project deletion: {e}");
                Err(Status::internal(format!(
                    "Failed to enqueue deletion: {e}"
                )))
            }
        }
    }
}
