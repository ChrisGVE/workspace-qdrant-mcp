//! LibraryWriteService gRPC implementation
//!
//! Daemon-exclusive writes for library management.
//! Replaces direct SQLite writes from CLI library commands.

use sqlx::SqlitePool;
use tonic::{Request, Response, Status};
use wqm_common::timestamps;

use crate::proto::{
    library_write_service_server::LibraryWriteService, AddLibraryRequest, AddLibraryResponse,
    ConfigureLibraryRequest, RemoveLibraryRequest, RemoveLibraryResponse, SetIncrementalRequest,
    SetIncrementalResponse, UnwatchLibraryRequest, WatchLibraryRequest, WatchLibraryResponse,
    WatchMutationResponse,
};

pub struct LibraryWriteServiceImpl {
    pool: SqlitePool,
}

impl LibraryWriteServiceImpl {
    pub fn new(pool: SqlitePool) -> Self {
        Self { pool }
    }
}

#[tonic::async_trait]
impl LibraryWriteService for LibraryWriteServiceImpl {
    async fn add_library(
        &self,
        request: Request<AddLibraryRequest>,
    ) -> Result<Response<AddLibraryResponse>, Status> {
        let req = request.into_inner();
        let watch_id = format!("lib-{}", req.tag);
        let now = timestamps::now_utc();

        // Check for duplicate watch_id
        let exists =
            sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM watch_folders WHERE watch_id = ?1")
                .bind(&watch_id)
                .fetch_one(&self.pool)
                .await
                .map_err(|e| Status::internal(format!("database error: {}", e)))?;

        if exists > 0 {
            return Ok(Response::new(AddLibraryResponse {
                success: false,
                watch_id,
                message: format!("Library '{}' already exists", req.tag),
            }));
        }

        // Check for duplicate path
        let path_exists =
            sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM watch_folders WHERE path = ?1")
                .bind(&req.path)
                .fetch_one(&self.pool)
                .await
                .map_err(|e| Status::internal(format!("database error: {}", e)))?;

        if path_exists > 0 {
            return Ok(Response::new(AddLibraryResponse {
                success: false,
                watch_id,
                message: format!("Path '{}' is already registered", req.path),
            }));
        }

        sqlx::query(
            "INSERT INTO watch_folders \
             (watch_id, path, collection, tenant_id, library_mode, enabled, is_active, \
              follow_symlinks, cleanup_on_disable, created_at, updated_at) \
             VALUES (?1, ?2, 'libraries', ?3, ?4, 0, 0, 0, 0, ?5, ?5)",
        )
        .bind(&watch_id)
        .bind(&req.path)
        .bind(&req.tag)
        .bind(&req.mode)
        .bind(&now)
        .execute(&self.pool)
        .await
        .map_err(|e| Status::internal(format!("database error: {}", e)))?;

        Ok(Response::new(AddLibraryResponse {
            success: true,
            watch_id,
            message: format!("Library '{}' added (not watching yet)", req.tag),
        }))
    }

    async fn remove_library(
        &self,
        request: Request<RemoveLibraryRequest>,
    ) -> Result<Response<RemoveLibraryResponse>, Status> {
        let req = request.into_inner();
        let watch_id = format!("lib-{}", req.tag);

        let exists =
            sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM watch_folders WHERE watch_id = ?1")
                .bind(&watch_id)
                .fetch_one(&self.pool)
                .await
                .map_err(|e| Status::internal(format!("database error: {}", e)))?;

        if exists == 0 {
            return Err(Status::not_found(format!(
                "library '{}' not found",
                req.tag
            )));
        }

        // Atomic deletion with FK disabled (matches CLI behavior)
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| Status::internal(format!("transaction error: {}", e)))?;

        // Disable FK for atomic cascading delete
        sqlx::query("PRAGMA foreign_keys = OFF")
            .execute(&mut *tx)
            .await
            .map_err(|e| Status::internal(format!("database error: {}", e)))?;

        let cancelled = sqlx::query(
            "DELETE FROM unified_queue WHERE tenant_id = ?1 AND collection = 'libraries'",
        )
        .bind(&req.tag)
        .execute(&mut *tx)
        .await
        .map_err(|e| Status::internal(format!("database error: {}", e)))?
        .rows_affected() as u32;

        let components = sqlx::query("DELETE FROM project_components WHERE watch_folder_id = ?1")
            .bind(&watch_id)
            .execute(&mut *tx)
            .await
            .map_err(|e| Status::internal(format!("database error: {}", e)))?
            .rows_affected() as u32;

        let tracked = sqlx::query("DELETE FROM tracked_files WHERE watch_folder_id = ?1")
            .bind(&watch_id)
            .execute(&mut *tx)
            .await
            .map_err(|e| Status::internal(format!("database error: {}", e)))?
            .rows_affected() as u32;

        sqlx::query("DELETE FROM watch_folders WHERE watch_id = ?1")
            .bind(&watch_id)
            .execute(&mut *tx)
            .await
            .map_err(|e| Status::internal(format!("database error: {}", e)))?;

        sqlx::query("PRAGMA foreign_keys = ON")
            .execute(&mut *tx)
            .await
            .map_err(|e| Status::internal(format!("database error: {}", e)))?;

        tx.commit()
            .await
            .map_err(|e| Status::internal(format!("commit error: {}", e)))?;

        Ok(Response::new(RemoveLibraryResponse {
            success: true,
            queue_items_cancelled: cancelled,
            tracked_files_deleted: tracked,
            components_deleted: components,
            message: format!("Library '{}' removed", req.tag),
        }))
    }

    async fn watch_library(
        &self,
        request: Request<WatchLibraryRequest>,
    ) -> Result<Response<WatchLibraryResponse>, Status> {
        let req = request.into_inner();
        let watch_id = format!("lib-{}", req.tag);
        let now = timestamps::now_utc();

        let exists =
            sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM watch_folders WHERE watch_id = ?1")
                .bind(&watch_id)
                .fetch_one(&self.pool)
                .await
                .map_err(|e| Status::internal(format!("database error: {}", e)))?;

        if exists > 0 {
            // Enable watching on existing library
            sqlx::query(
                "UPDATE watch_folders SET enabled = 1, library_mode = ?1, path = ?2, \
                 updated_at = ?3, last_activity_at = ?3 WHERE watch_id = ?4",
            )
            .bind(&req.mode)
            .bind(&req.path)
            .bind(&now)
            .bind(&watch_id)
            .execute(&self.pool)
            .await
            .map_err(|e| Status::internal(format!("database error: {}", e)))?;

            Ok(Response::new(WatchLibraryResponse {
                success: true,
                is_new: false,
                watch_id,
                message: format!("Library '{}' watching enabled", req.tag),
            }))
        } else {
            // Insert new library with watching enabled
            sqlx::query(
                "INSERT INTO watch_folders \
                 (watch_id, path, collection, tenant_id, library_mode, enabled, is_active, \
                  follow_symlinks, cleanup_on_disable, created_at, updated_at, last_activity_at) \
                 VALUES (?1, ?2, 'libraries', ?3, ?4, 1, 0, 0, 0, ?5, ?5, ?5)",
            )
            .bind(&watch_id)
            .bind(&req.path)
            .bind(&req.tag)
            .bind(&req.mode)
            .bind(&now)
            .execute(&self.pool)
            .await
            .map_err(|e| Status::internal(format!("database error: {}", e)))?;

            Ok(Response::new(WatchLibraryResponse {
                success: true,
                is_new: true,
                watch_id,
                message: format!("Library '{}' added and watching enabled", req.tag),
            }))
        }
    }

    async fn unwatch_library(
        &self,
        request: Request<UnwatchLibraryRequest>,
    ) -> Result<Response<WatchMutationResponse>, Status> {
        let watch_id = format!("lib-{}", request.into_inner().tag);
        let now = timestamps::now_utc();

        let result = sqlx::query(
            "UPDATE watch_folders SET enabled = 0, updated_at = ?1 WHERE watch_id = ?2",
        )
        .bind(&now)
        .bind(&watch_id)
        .execute(&self.pool)
        .await
        .map_err(|e| Status::internal(format!("database error: {}", e)))?;

        Ok(Response::new(WatchMutationResponse {
            affected_count: result.rows_affected() as u32,
        }))
    }

    async fn configure_library(
        &self,
        request: Request<ConfigureLibraryRequest>,
    ) -> Result<Response<WatchMutationResponse>, Status> {
        let req = request.into_inner();
        let watch_id = format!("lib-{}", req.tag);
        let now = timestamps::now_utc();
        let mut affected = 0u32;

        if let Some(ref mode) = req.mode {
            let result = sqlx::query(
                "UPDATE watch_folders SET library_mode = ?1, updated_at = ?2 WHERE watch_id = ?3",
            )
            .bind(mode)
            .bind(&now)
            .bind(&watch_id)
            .execute(&self.pool)
            .await
            .map_err(|e| Status::internal(format!("database error: {}", e)))?;
            affected += result.rows_affected() as u32;
        }

        if req.enable == Some(true) {
            let result = sqlx::query(
                "UPDATE watch_folders SET enabled = 1, updated_at = ?1 WHERE watch_id = ?2",
            )
            .bind(&now)
            .bind(&watch_id)
            .execute(&self.pool)
            .await
            .map_err(|e| Status::internal(format!("database error: {}", e)))?;
            affected += result.rows_affected() as u32;
        }

        if req.disable == Some(true) {
            let result = sqlx::query(
                "UPDATE watch_folders SET enabled = 0, updated_at = ?1 WHERE watch_id = ?2",
            )
            .bind(&now)
            .bind(&watch_id)
            .execute(&self.pool)
            .await
            .map_err(|e| Status::internal(format!("database error: {}", e)))?;
            affected += result.rows_affected() as u32;
        }

        Ok(Response::new(WatchMutationResponse {
            affected_count: affected,
        }))
    }

    async fn set_incremental(
        &self,
        request: Request<SetIncrementalRequest>,
    ) -> Result<Response<SetIncrementalResponse>, Status> {
        let req = request.into_inner();
        let value: i32 = if req.clear { 0 } else { 1 };
        let now = timestamps::now_utc();
        let mut updated = 0u32;
        let mut not_found = 0u32;

        for path in &req.file_paths {
            let result = sqlx::query(
                "UPDATE tracked_files SET incremental = ?1, updated_at = ?2 WHERE file_path = ?3",
            )
            .bind(value)
            .bind(&now)
            .bind(path)
            .execute(&self.pool)
            .await
            .map_err(|e| Status::internal(format!("database error: {}", e)))?;

            if result.rows_affected() > 0 {
                updated += 1;
            } else {
                not_found += 1;
            }
        }

        Ok(Response::new(SetIncrementalResponse { updated, not_found }))
    }
}
