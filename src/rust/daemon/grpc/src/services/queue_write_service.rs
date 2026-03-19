//! QueueWriteService gRPC implementation
//!
//! Daemon-exclusive writes to the unified_queue table.
//! Replaces direct SQLite writes from CLI and MCP server.

use sqlx::SqlitePool;
use tonic::{Request, Response, Status};
use tracing::warn;
use uuid::Uuid;
use wqm_common::hashing::generate_idempotency_key;
use wqm_common::queue_types::{ItemType, QueueOperation};
use wqm_common::timestamps;

use crate::proto::{
    queue_write_service_server::QueueWriteService, CancelItemsRequest, CancelItemsResponse,
    CleanQueueByCollectionRequest, CleanQueueRequest, CleanQueueResponse, EnqueueItemRequest,
    EnqueueItemResponse, RemoveItemRequest, RemoveItemResponse, RetryAllResponse, RetryItemRequest,
    RetryItemResponse,
};

pub struct QueueWriteServiceImpl {
    pool: SqlitePool,
}

impl QueueWriteServiceImpl {
    pub fn new(pool: SqlitePool) -> Self {
        Self { pool }
    }
}

#[tonic::async_trait]
impl QueueWriteService for QueueWriteServiceImpl {
    async fn enqueue_item(
        &self,
        request: Request<EnqueueItemRequest>,
    ) -> Result<Response<EnqueueItemResponse>, Status> {
        let req = request.into_inner();

        let item_type = ItemType::parse_str(&req.item_type).ok_or_else(|| {
            Status::invalid_argument(format!("invalid item_type: {}", req.item_type))
        })?;
        let op = QueueOperation::parse_str(&req.op)
            .ok_or_else(|| Status::invalid_argument(format!("invalid op: {}", req.op)))?;

        if req.tenant_id.is_empty() {
            return Err(Status::invalid_argument("tenant_id cannot be empty"));
        }
        if req.collection.is_empty() {
            return Err(Status::invalid_argument("collection cannot be empty"));
        }

        let idempotency_key = generate_idempotency_key(
            item_type,
            op,
            &req.tenant_id,
            &req.collection,
            &req.payload_json,
        )
        .map_err(|e| Status::invalid_argument(format!("idempotency key error: {}", e)))?;

        let queue_id = Uuid::new_v4().to_string();
        let now = timestamps::now_utc();
        let branch = if req.branch.is_empty() {
            "main"
        } else {
            &req.branch
        };
        let metadata = req.metadata_json.as_deref().unwrap_or("{}");

        let result = sqlx::query(
            r#"INSERT OR IGNORE INTO unified_queue (
                queue_id, idempotency_key, item_type, op, tenant_id, collection,
                status, branch, payload_json, metadata, created_at, updated_at, retry_count
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'pending', ?7, ?8, ?9, ?10, ?10, 0)"#,
        )
        .bind(&queue_id)
        .bind(&idempotency_key)
        .bind(item_type.as_str())
        .bind(op.as_str())
        .bind(&req.tenant_id)
        .bind(&req.collection)
        .bind(branch)
        .bind(&req.payload_json)
        .bind(metadata)
        .bind(&now)
        .execute(&self.pool)
        .await
        .map_err(|e| Status::internal(format!("database error: {}", e)))?;

        let is_new = result.rows_affected() > 0;

        let response_queue_id = if is_new {
            queue_id
        } else {
            // Duplicate — look up existing queue_id
            sqlx::query_scalar::<_, String>(
                "SELECT queue_id FROM unified_queue WHERE idempotency_key = ?1",
            )
            .bind(&idempotency_key)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| Status::internal(format!("database error: {}", e)))?
            .unwrap_or_default()
        };

        Ok(Response::new(EnqueueItemResponse {
            queue_id: response_queue_id,
            idempotency_key,
            is_new,
        }))
    }

    async fn retry_all(&self, _request: Request<()>) -> Result<Response<RetryAllResponse>, Status> {
        let now = timestamps::now_utc();

        let result = sqlx::query(
            r#"UPDATE unified_queue
            SET status = 'pending', retry_count = 0, error_message = NULL,
                last_error_at = NULL, lease_until = NULL, worker_id = NULL,
                qdrant_status = NULL, search_status = NULL, updated_at = ?1
            WHERE status = 'failed'"#,
        )
        .bind(&now)
        .execute(&self.pool)
        .await
        .map_err(|e| Status::internal(format!("database error: {}", e)))?;

        Ok(Response::new(RetryAllResponse {
            reset_count: result.rows_affected() as u32,
        }))
    }

    async fn retry_item(
        &self,
        request: Request<RetryItemRequest>,
    ) -> Result<Response<RetryItemResponse>, Status> {
        let req = request.into_inner();
        let prefix = format!("{}%", req.queue_id);

        let row = sqlx::query_as::<_, (String, String, i32)>(
            "SELECT queue_id, status, retry_count FROM unified_queue \
             WHERE queue_id = ?1 OR queue_id LIKE ?2 LIMIT 1",
        )
        .bind(&req.queue_id)
        .bind(&prefix)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| Status::internal(format!("database error: {}", e)))?;

        let Some((found_id, status, retry_count)) = row else {
            return Ok(Response::new(RetryItemResponse {
                found: false,
                resolved_id: String::new(),
                previous_status: String::new(),
                previous_retry_count: 0,
                reset: false,
            }));
        };

        if status != "failed" {
            return Ok(Response::new(RetryItemResponse {
                found: true,
                resolved_id: found_id,
                previous_status: status,
                previous_retry_count: retry_count,
                reset: false,
            }));
        }

        let now = timestamps::now_utc();
        sqlx::query(
            r#"UPDATE unified_queue
            SET status = 'pending', retry_count = 0, error_message = NULL,
                last_error_at = NULL, lease_until = NULL, worker_id = NULL,
                qdrant_status = NULL, search_status = NULL, updated_at = ?1
            WHERE queue_id = ?2"#,
        )
        .bind(&now)
        .bind(&found_id)
        .execute(&self.pool)
        .await
        .map_err(|e| Status::internal(format!("database error: {}", e)))?;

        Ok(Response::new(RetryItemResponse {
            found: true,
            resolved_id: found_id,
            previous_status: status,
            previous_retry_count: retry_count,
            reset: true,
        }))
    }

    async fn clean_queue(
        &self,
        request: Request<CleanQueueRequest>,
    ) -> Result<Response<CleanQueueResponse>, Status> {
        let req = request.into_inner();

        let statuses: Vec<&str> = if req.statuses.is_empty() {
            vec!["done", "failed"]
        } else {
            req.statuses.iter().map(|s| s.as_str()).collect()
        };

        for s in &statuses {
            if !["done", "failed"].contains(s) {
                return Err(Status::invalid_argument(format!(
                    "invalid status '{}': only 'done' and 'failed' are cleanable",
                    s
                )));
            }
        }

        let days = req.older_than_days;
        if days <= 0 {
            return Err(Status::invalid_argument("older_than_days must be positive"));
        }

        // Build dynamic query
        let placeholders: String = statuses
            .iter()
            .enumerate()
            .map(|(i, _)| format!("?{}", i + 2))
            .collect::<Vec<_>>()
            .join(", ");

        let sql = format!(
            "DELETE FROM unified_queue \
             WHERE status IN ({}) AND updated_at < datetime('now', '-' || ?1 || ' days')",
            placeholders
        );

        let mut query = sqlx::query(&sql).bind(days);
        for s in &statuses {
            query = query.bind(*s);
        }

        let result = query
            .execute(&self.pool)
            .await
            .map_err(|e| Status::internal(format!("database error: {}", e)))?;

        Ok(Response::new(CleanQueueResponse {
            deleted_count: result.rows_affected() as u32,
        }))
    }

    async fn cancel_items(
        &self,
        request: Request<CancelItemsRequest>,
    ) -> Result<Response<CancelItemsResponse>, Status> {
        let req = request.into_inner();

        if req.tenant_id.is_empty() {
            return Err(Status::invalid_argument("tenant_id cannot be empty"));
        }

        // Filter out in_progress (safety invariant)
        let statuses: Vec<&str> = if req.statuses.is_empty() {
            vec!["pending", "failed"]
        } else {
            req.statuses
                .iter()
                .map(|s| s.as_str())
                .filter(|s| *s != "in_progress")
                .collect()
        };

        if statuses.is_empty() {
            return Err(Status::invalid_argument(
                "no cancellable statuses specified (in_progress cannot be cancelled)",
            ));
        }

        let (resolved_tenant, project_path) = resolve_tenant(&self.pool, &req.tenant_id)
            .await
            .map_err(|e| Status::not_found(format!("tenant resolution failed: {}", e)))?;

        let placeholders: String = statuses
            .iter()
            .enumerate()
            .map(|(i, _)| format!("?{}", i + 2))
            .collect::<Vec<_>>()
            .join(", ");

        if req.dry_run {
            let count_sql = format!(
                "SELECT COUNT(*) FROM unified_queue WHERE tenant_id = ?1 AND status IN ({})",
                placeholders
            );
            let mut query = sqlx::query_scalar::<_, i64>(&count_sql).bind(&resolved_tenant);
            for s in &statuses {
                query = query.bind(*s);
            }
            let count = query
                .fetch_one(&self.pool)
                .await
                .map_err(|e| Status::internal(format!("database error: {}", e)))?;

            return Ok(Response::new(CancelItemsResponse {
                count: count as u32,
                tenant_id: resolved_tenant,
                project_path,
                is_dry_run: true,
            }));
        }

        let delete_sql = format!(
            "DELETE FROM unified_queue WHERE tenant_id = ?1 AND status IN ({})",
            placeholders
        );
        let mut query = sqlx::query(&delete_sql).bind(&resolved_tenant);
        for s in &statuses {
            query = query.bind(*s);
        }
        let result = query
            .execute(&self.pool)
            .await
            .map_err(|e| Status::internal(format!("database error: {}", e)))?;

        Ok(Response::new(CancelItemsResponse {
            count: result.rows_affected() as u32,
            tenant_id: resolved_tenant,
            project_path,
            is_dry_run: false,
        }))
    }

    async fn remove_item(
        &self,
        request: Request<RemoveItemRequest>,
    ) -> Result<Response<RemoveItemResponse>, Status> {
        let req = request.into_inner();
        let prefix = format!("{}%", req.queue_id);

        let row = sqlx::query_as::<_, (String, String, String, String, String)>(
            "SELECT queue_id, item_type, op, collection, status \
             FROM unified_queue \
             WHERE queue_id = ?1 OR queue_id LIKE ?2 OR idempotency_key LIKE ?2 \
             LIMIT 1",
        )
        .bind(&req.queue_id)
        .bind(&prefix)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| Status::internal(format!("database error: {}", e)))?;

        let Some((found_id, item_type, op, collection, status)) = row else {
            return Ok(Response::new(RemoveItemResponse {
                found: false,
                resolved_id: String::new(),
                item_type: String::new(),
                op: String::new(),
                collection: String::new(),
                status: String::new(),
            }));
        };

        if status == "in_progress" {
            warn!(
                queue_id = %found_id,
                "removing in_progress item — processor may error"
            );
        }

        sqlx::query("DELETE FROM unified_queue WHERE queue_id = ?1")
            .bind(&found_id)
            .execute(&self.pool)
            .await
            .map_err(|e| Status::internal(format!("database error: {}", e)))?;

        Ok(Response::new(RemoveItemResponse {
            found: true,
            resolved_id: found_id,
            item_type,
            op,
            collection,
            status,
        }))
    }

    async fn clean_queue_by_collection(
        &self,
        request: Request<CleanQueueByCollectionRequest>,
    ) -> Result<Response<CleanQueueResponse>, Status> {
        let req = request.into_inner();

        if req.collections.is_empty() {
            return Err(Status::invalid_argument(
                "at least one collection name is required",
            ));
        }

        // Default statuses: pending + failed. Never allow in_progress.
        let statuses: Vec<&str> = if req.statuses.is_empty() {
            vec!["pending", "failed"]
        } else {
            req.statuses
                .iter()
                .map(|s| s.as_str())
                .filter(|s| *s != "in_progress")
                .collect()
        };

        if statuses.is_empty() {
            return Err(Status::invalid_argument(
                "no deletable statuses specified (in_progress cannot be deleted)",
            ));
        }

        // Build dynamic placeholders for collections and statuses
        let col_placeholders: String = req
            .collections
            .iter()
            .enumerate()
            .map(|(i, _)| format!("?{}", i + 1))
            .collect::<Vec<_>>()
            .join(", ");

        let status_placeholders: String = statuses
            .iter()
            .enumerate()
            .map(|(i, _)| format!("?{}", i + 1 + req.collections.len()))
            .collect::<Vec<_>>()
            .join(", ");

        let sql = format!(
            "DELETE FROM unified_queue WHERE collection IN ({}) AND status IN ({})",
            col_placeholders, status_placeholders
        );

        let mut query = sqlx::query(&sql);
        for c in &req.collections {
            query = query.bind(c);
        }
        for s in &statuses {
            query = query.bind(*s);
        }

        let result = query
            .execute(&self.pool)
            .await
            .map_err(|e| Status::internal(format!("database error: {}", e)))?;

        Ok(Response::new(CleanQueueResponse {
            deleted_count: result.rows_affected() as u32,
        }))
    }
}

/// Resolve a tenant hint to (tenant_id, project_path).
///
/// Tries exact tenant_id match, then exact path match (canonicalized),
/// then case-insensitive path substring match.
async fn resolve_tenant(pool: &SqlitePool, hint: &str) -> Result<(String, String), String> {
    // 1. Exact tenant_id match
    let row = sqlx::query_as::<_, (String, String)>(
        "SELECT tenant_id, path FROM watch_folders WHERE tenant_id = ?1 LIMIT 1",
    )
    .bind(hint)
    .fetch_optional(pool)
    .await
    .map_err(|e| e.to_string())?;

    if let Some((tid, path)) = row {
        return Ok((tid, path));
    }

    // 2. Exact path match
    let canonical = std::path::Path::new(hint)
        .canonicalize()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| hint.to_string());

    let row = sqlx::query_as::<_, (String, String)>(
        "SELECT tenant_id, path FROM watch_folders WHERE path = ?1 LIMIT 1",
    )
    .bind(&canonical)
    .fetch_optional(pool)
    .await
    .map_err(|e| e.to_string())?;

    if let Some((tid, path)) = row {
        return Ok((tid, path));
    }

    // 3. Case-insensitive substring match
    let pattern = format!("%{}%", hint);
    let rows = sqlx::query_as::<_, (String, String)>(
        "SELECT tenant_id, path FROM watch_folders WHERE path LIKE ?1",
    )
    .bind(&pattern)
    .fetch_all(pool)
    .await
    .map_err(|e| e.to_string())?;

    match rows.len() {
        0 => Err(format!("no project found matching '{}'", hint)),
        1 => Ok(rows.into_iter().next().unwrap()),
        n => Err(format!("ambiguous: '{}' matches {} projects", hint, n)),
    }
}
