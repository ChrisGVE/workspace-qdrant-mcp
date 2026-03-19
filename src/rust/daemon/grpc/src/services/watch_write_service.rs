//! WatchWriteService gRPC implementation
//!
//! Daemon-exclusive writes to the watch_folders table.
//! Replaces direct SQLite writes from CLI watch commands.

use sqlx::SqlitePool;
use tonic::{Request, Response, Status};
use wqm_common::timestamps;

use crate::proto::{
    watch_write_service_server::WatchWriteService, ArchiveWatchRequest, ArchiveWatchResponse,
    WatchIdRequest, WatchMutationResponse,
};

pub struct WatchWriteServiceImpl {
    pool: SqlitePool,
}

impl WatchWriteServiceImpl {
    pub fn new(pool: SqlitePool) -> Self {
        Self { pool }
    }
}

#[tonic::async_trait]
impl WatchWriteService for WatchWriteServiceImpl {
    async fn pause_watchers(
        &self,
        _request: Request<()>,
    ) -> Result<Response<WatchMutationResponse>, Status> {
        let now = timestamps::now_utc();
        let result = sqlx::query(
            "UPDATE watch_folders SET is_paused = 1, \
             pause_start_time = ?1, updated_at = ?1 \
             WHERE enabled = 1 AND is_paused = 0",
        )
        .bind(&now)
        .execute(&self.pool)
        .await
        .map_err(|e| Status::internal(format!("database error: {}", e)))?;

        Ok(Response::new(WatchMutationResponse {
            affected_count: result.rows_affected() as u32,
        }))
    }

    async fn resume_watchers(
        &self,
        _request: Request<()>,
    ) -> Result<Response<WatchMutationResponse>, Status> {
        let now = timestamps::now_utc();
        let result = sqlx::query(
            "UPDATE watch_folders SET is_paused = 0, \
             pause_start_time = NULL, updated_at = ?1 \
             WHERE enabled = 1 AND is_paused = 1",
        )
        .bind(&now)
        .execute(&self.pool)
        .await
        .map_err(|e| Status::internal(format!("database error: {}", e)))?;

        Ok(Response::new(WatchMutationResponse {
            affected_count: result.rows_affected() as u32,
        }))
    }

    async fn enable_watch(
        &self,
        request: Request<WatchIdRequest>,
    ) -> Result<Response<WatchMutationResponse>, Status> {
        let watch_id = resolve_watch_id(&self.pool, &request.into_inner().watch_id).await?;
        let now = timestamps::now_utc();

        let result = sqlx::query(
            "UPDATE watch_folders SET enabled = 1, updated_at = ?1 WHERE watch_id = ?2",
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

    async fn disable_watch(
        &self,
        request: Request<WatchIdRequest>,
    ) -> Result<Response<WatchMutationResponse>, Status> {
        let watch_id = resolve_watch_id(&self.pool, &request.into_inner().watch_id).await?;
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

    async fn archive_watch(
        &self,
        request: Request<ArchiveWatchRequest>,
    ) -> Result<Response<ArchiveWatchResponse>, Status> {
        let req = request.into_inner();
        let watch_id = resolve_watch_id(&self.pool, &req.watch_id).await?;
        let now = timestamps::now_utc();

        // Archive the parent
        let result = sqlx::query(
            "UPDATE watch_folders SET is_archived = 1, updated_at = ?1 \
             WHERE watch_id = ?2 AND COALESCE(is_archived, 0) = 0",
        )
        .bind(&now)
        .bind(&watch_id)
        .execute(&self.pool)
        .await
        .map_err(|e| Status::internal(format!("database error: {}", e)))?;

        let affected = result.rows_affected() as u32;
        if affected == 0 {
            return Ok(Response::new(ArchiveWatchResponse {
                affected_count: 0,
                submodules_archived: 0,
                submodules_skipped: 0,
            }));
        }

        // Cascade to submodules with cross-reference safety
        let (archived, skipped) = if req.cascade_submodules {
            archive_submodules_safely(&self.pool, &watch_id, &now).await?
        } else {
            (0, 0)
        };

        Ok(Response::new(ArchiveWatchResponse {
            affected_count: affected,
            submodules_archived: archived,
            submodules_skipped: skipped,
        }))
    }

    async fn unarchive_watch(
        &self,
        request: Request<WatchIdRequest>,
    ) -> Result<Response<WatchMutationResponse>, Status> {
        let watch_id = resolve_watch_id(&self.pool, &request.into_inner().watch_id).await?;
        let now = timestamps::now_utc();

        let result = sqlx::query(
            "UPDATE watch_folders SET is_archived = 0, updated_at = ?1 \
             WHERE watch_id = ?2 AND COALESCE(is_archived, 0) = 1",
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
}

/// Resolve a watch_id that may be an ID or a filesystem path.
async fn resolve_watch_id(pool: &SqlitePool, input: &str) -> Result<String, Status> {
    // Try exact watch_id
    let exists =
        sqlx::query_scalar::<_, String>("SELECT watch_id FROM watch_folders WHERE watch_id = ?1")
            .bind(input)
            .fetch_optional(pool)
            .await
            .map_err(|e| Status::internal(format!("database error: {}", e)))?;

    if let Some(id) = exists {
        return Ok(id);
    }

    // Try prefix match
    let prefix = format!("{}%", input);
    let matches = sqlx::query_scalar::<_, String>(
        "SELECT watch_id FROM watch_folders WHERE watch_id LIKE ?1",
    )
    .bind(&prefix)
    .fetch_all(pool)
    .await
    .map_err(|e| Status::internal(format!("database error: {}", e)))?;

    if matches.len() == 1 {
        return Ok(matches.into_iter().next().unwrap());
    }

    // Try as filesystem path
    let canonical = std::path::Path::new(input)
        .canonicalize()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| input.to_string());

    let by_path =
        sqlx::query_scalar::<_, String>("SELECT watch_id FROM watch_folders WHERE path = ?1")
            .bind(&canonical)
            .fetch_optional(pool)
            .await
            .map_err(|e| Status::internal(format!("database error: {}", e)))?;

    by_path.ok_or_else(|| Status::not_found(format!("watch folder not found: {}", input)))
}

/// Archive submodules with cross-reference safety checks.
async fn archive_submodules_safely(
    pool: &SqlitePool,
    parent_watch_id: &str,
    now: &str,
) -> Result<(u32, u32), Status> {
    let submodules = sqlx::query_as::<_, (String, Option<String>, Option<String>)>(
        "SELECT wf.watch_id, wf.remote_hash, wf.git_remote_url \
         FROM watch_folders wf \
         INNER JOIN watch_folder_submodules j ON wf.watch_id = j.child_watch_id \
         WHERE j.parent_watch_id = ?1",
    )
    .bind(parent_watch_id)
    .fetch_all(pool)
    .await
    .map_err(|e| Status::internal(format!("database error: {}", e)))?;

    let mut archived = 0u32;
    let mut skipped = 0u32;

    for (sub_id, remote_hash, git_remote_url) in &submodules {
        let rh = remote_hash.as_deref().unwrap_or("");
        let url = git_remote_url.as_deref().unwrap_or("");

        if rh.is_empty() && url.is_empty() {
            // No remote info — archive with parent
            let result = sqlx::query(
                "UPDATE watch_folders SET is_archived = 1, updated_at = ?1 \
                 WHERE watch_id = ?2 AND COALESCE(is_archived, 0) = 0",
            )
            .bind(now)
            .bind(sub_id)
            .execute(pool)
            .await
            .map_err(|e| Status::internal(format!("database error: {}", e)))?;
            if result.rows_affected() > 0 {
                archived += 1;
            }
            continue;
        }

        // Check if other active projects reference this submodule
        let other_refs = sqlx::query_scalar::<_, i64>(
            "SELECT COUNT(*) FROM watch_folders sub \
             WHERE sub.remote_hash = ?1 AND sub.git_remote_url = ?2 \
             AND sub.parent_watch_id != ?3 \
             AND COALESCE(sub.is_archived, 0) = 0 \
             AND EXISTS ( \
               SELECT 1 FROM watch_folders parent \
               WHERE parent.watch_id = sub.parent_watch_id \
               AND COALESCE(parent.is_archived, 0) = 0 \
             )",
        )
        .bind(rh)
        .bind(url)
        .bind(parent_watch_id)
        .fetch_one(pool)
        .await
        .map_err(|e| Status::internal(format!("database error: {}", e)))?;

        if other_refs > 0 {
            skipped += 1;
        } else {
            let result = sqlx::query(
                "UPDATE watch_folders SET is_archived = 1, updated_at = ?1 \
                 WHERE watch_id = ?2 AND COALESCE(is_archived, 0) = 0",
            )
            .bind(now)
            .bind(sub_id)
            .execute(pool)
            .await
            .map_err(|e| Status::internal(format!("database error: {}", e)))?;
            if result.rows_affected() > 0 {
                archived += 1;
            }
        }
    }

    Ok((archived, skipped))
}
