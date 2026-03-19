//! TrackingWriteService gRPC implementation
//!
//! Daemon-exclusive writes for observability and mirror tables.
//! Replaces direct SQLite writes from MCP server and CLI.
//! Errors are logged but not propagated — instrumentation must never block.

use sqlx::SqlitePool;
use tonic::{Request, Response, Status};
use tracing::warn;
use wqm_common::timestamps;

use crate::proto::{
    tracking_write_service_server::TrackingWriteService, DeleteRuleMirrorRequest,
    LogSearchEventRequest, UpdateSearchEventRequest, UpsertRuleMirrorRequest,
};

pub struct TrackingWriteServiceImpl {
    pool: SqlitePool,
}

impl TrackingWriteServiceImpl {
    pub fn new(pool: SqlitePool) -> Self {
        Self { pool }
    }
}

#[tonic::async_trait]
impl TrackingWriteService for TrackingWriteServiceImpl {
    async fn log_search_event(
        &self,
        request: Request<LogSearchEventRequest>,
    ) -> Result<Response<()>, Status> {
        let req = request.into_inner();
        let now = timestamps::now_utc();

        if let Err(e) = sqlx::query(
            "INSERT INTO search_events (
                id, ts, session_id, project_id, actor, tool, op,
                query_text, filters, top_k, result_count, latency_ms,
                top_result_refs, outcome, parent_event_id, created_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?2)",
        )
        .bind(&req.id)
        .bind(&now)
        .bind(&req.session_id)
        .bind(&req.project_id)
        .bind(&req.actor)
        .bind(&req.tool)
        .bind(&req.op)
        .bind(&req.query_text)
        .bind(&req.filters)
        .bind(req.top_k)
        .bind(req.result_count)
        .bind(req.latency_ms)
        .bind(&req.top_result_refs)
        .bind(&req.outcome)
        .bind(&req.parent_event_id)
        .execute(&self.pool)
        .await
        {
            warn!("failed to log search event: {}", e);
        }

        Ok(Response::new(()))
    }

    async fn update_search_event(
        &self,
        request: Request<UpdateSearchEventRequest>,
    ) -> Result<Response<()>, Status> {
        let req = request.into_inner();

        if let Err(e) = sqlx::query(
            "UPDATE search_events \
             SET result_count = ?1, latency_ms = ?2, top_result_refs = ?3, outcome = ?4 \
             WHERE id = ?5",
        )
        .bind(req.result_count)
        .bind(req.latency_ms)
        .bind(&req.top_result_refs)
        .bind(&req.outcome)
        .bind(&req.event_id)
        .execute(&self.pool)
        .await
        {
            warn!("failed to update search event: {}", e);
        }

        Ok(Response::new(()))
    }

    async fn upsert_rule_mirror(
        &self,
        request: Request<UpsertRuleMirrorRequest>,
    ) -> Result<Response<()>, Status> {
        let req = request.into_inner();

        if let Err(e) = sqlx::query(
            "INSERT INTO rules_mirror (rule_id, rule_text, scope, tenant_id, created_at, updated_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6) \
             ON CONFLICT(rule_id) DO UPDATE SET \
                 rule_text = excluded.rule_text, \
                 scope = excluded.scope, \
                 tenant_id = excluded.tenant_id, \
                 updated_at = excluded.updated_at",
        )
        .bind(&req.rule_id)
        .bind(&req.rule_text)
        .bind(&req.scope)
        .bind(&req.tenant_id)
        .bind(&req.created_at)
        .bind(&req.updated_at)
        .execute(&self.pool)
        .await
        {
            warn!("failed to upsert rules mirror: {}", e);
        }

        Ok(Response::new(()))
    }

    async fn delete_rule_mirror(
        &self,
        request: Request<DeleteRuleMirrorRequest>,
    ) -> Result<Response<()>, Status> {
        let req = request.into_inner();

        if let Err(e) = sqlx::query("DELETE FROM rules_mirror WHERE rule_id = ?1")
            .bind(&req.rule_id)
            .execute(&self.pool)
            .await
        {
            warn!("failed to delete rules mirror: {}", e);
        }

        Ok(Response::new(()))
    }
}
