//! TrackingWriteService gRPC implementation
//!
//! Delegates all state.db mutations to the WriteActor via WriteActorHandle.
//! Errors are logged but not propagated — instrumentation must never block.

use tonic::{Request, Response, Status};
use workspace_qdrant_core::search_events_schema::{ALLOWED_ACTORS, ALLOWED_OPS, ALLOWED_TOOLS};
use workspace_qdrant_core::write_actor::{
    DeleteRuleMirrorData, DeleteScratchpadMirrorData, LogSearchEventData, UpdateSearchEventData,
    UpsertRuleMirrorData, UpsertScratchpadMirrorData, WriteActorHandle,
};

use crate::proto::{
    tracking_write_service_server::TrackingWriteService, DeleteRuleMirrorRequest,
    DeleteScratchpadMirrorRequest, LogSearchEventRequest, UpdateSearchEventRequest,
    UpsertRuleMirrorRequest, UpsertScratchpadMirrorRequest,
};

pub struct TrackingWriteServiceImpl {
    write_actor: WriteActorHandle,
}

impl TrackingWriteServiceImpl {
    pub fn new(write_actor: WriteActorHandle) -> Self {
        Self { write_actor }
    }
}

/// Reject a value the `search_events` CHECK would refuse, before it reaches the
/// fire-and-forget write path.
///
/// `LogSearchEvent` hands `actor`/`tool`/`op` straight to an INSERT whose only
/// guard is the SQLite CHECK; on the fire-and-forget path a CHECK violation is
/// swallowed with a `warn!`, so the caller never learns its event was dropped.
/// Validating against the canonical allow-list here turns that silent drop into
/// an explicit `invalid_argument` the caller can see and fix (L6/#135). The
/// allow-lists are the single source shared with the schema CHECK clauses
/// (`search_events_schema`).
fn check_allowed(field: &str, value: &str, allowed: &[&str]) -> Result<(), Status> {
    if allowed.contains(&value) {
        return Ok(());
    }
    Err(Status::invalid_argument(format!(
        "unrecognized {field} '{value}'; expected one of [{}]",
        allowed.join(", ")
    )))
}

#[tonic::async_trait]
impl TrackingWriteService for TrackingWriteServiceImpl {
    async fn log_search_event(
        &self,
        request: Request<LogSearchEventRequest>,
    ) -> Result<Response<()>, Status> {
        let req = request.into_inner();

        // Validate up front so a bad actor/tool/op is reported to the caller
        // rather than silently dropped by the SQLite CHECK on the fire-and-forget
        // write path (L6/#135).
        check_allowed("actor", &req.actor, ALLOWED_ACTORS)?;
        check_allowed("tool", &req.tool, ALLOWED_TOOLS)?;
        check_allowed("op", &req.op, ALLOWED_OPS)?;

        // Fire-and-forget: log errors but never fail the gRPC call
        if let Err(e) = self
            .write_actor
            .log_search_event(LogSearchEventData {
                id: req.id,
                session_id: req.session_id,
                project_id: req.project_id,
                actor: req.actor,
                tool: req.tool,
                op: req.op,
                query_text: req.query_text,
                filters: req.filters,
                top_k: req.top_k,
                result_count: req.result_count,
                latency_ms: req.latency_ms,
                top_result_refs: req.top_result_refs,
                outcome: req.outcome,
                parent_event_id: req.parent_event_id,
            })
            .await
        {
            tracing::warn!("log_search_event failed (fire-and-forget): {}", e);
        }
        Ok(Response::new(()))
    }

    async fn update_search_event(
        &self,
        request: Request<UpdateSearchEventRequest>,
    ) -> Result<Response<()>, Status> {
        let req = request.into_inner();
        if let Err(e) = self
            .write_actor
            .update_search_event(UpdateSearchEventData {
                event_id: req.event_id,
                result_count: Some(req.result_count),
                latency_ms: Some(req.latency_ms),
                top_result_refs: req.top_result_refs,
                outcome: req.outcome,
            })
            .await
        {
            tracing::warn!("update_search_event failed (fire-and-forget): {}", e);
        }
        Ok(Response::new(()))
    }

    async fn upsert_rule_mirror(
        &self,
        request: Request<UpsertRuleMirrorRequest>,
    ) -> Result<Response<()>, Status> {
        let req = request.into_inner();
        if let Err(e) = self
            .write_actor
            .upsert_rule_mirror(UpsertRuleMirrorData {
                rule_id: req.rule_id,
                rule_text: req.rule_text,
                scope: req.scope.unwrap_or_default(),
                tenant_id: req.tenant_id.unwrap_or_default(),
                created_at: req.created_at,
                updated_at: req.updated_at,
            })
            .await
        {
            tracing::warn!("upsert_rule_mirror failed (fire-and-forget): {}", e);
        }
        Ok(Response::new(()))
    }

    async fn delete_rule_mirror(
        &self,
        request: Request<DeleteRuleMirrorRequest>,
    ) -> Result<Response<()>, Status> {
        let req = request.into_inner();
        if let Err(e) = self
            .write_actor
            .delete_rule_mirror(DeleteRuleMirrorData {
                rule_id: req.rule_id,
            })
            .await
        {
            tracing::warn!("delete_rule_mirror failed (fire-and-forget): {}", e);
        }
        Ok(Response::new(()))
    }

    async fn upsert_scratchpad_mirror(
        &self,
        request: Request<UpsertScratchpadMirrorRequest>,
    ) -> Result<Response<()>, Status> {
        let req = request.into_inner();
        if let Err(e) = self
            .write_actor
            .upsert_scratchpad_mirror(UpsertScratchpadMirrorData {
                scratchpad_id: req.scratchpad_id,
                content: req.content,
                title: req.title.unwrap_or_default(),
                tags: req.tags.unwrap_or_else(|| "[]".to_string()),
                tenant_id: req.tenant_id,
                created_at: req.created_at,
                updated_at: req.updated_at,
            })
            .await
        {
            tracing::warn!("upsert_scratchpad_mirror failed (fire-and-forget): {}", e);
        }
        Ok(Response::new(()))
    }

    async fn delete_scratchpad_mirror(
        &self,
        request: Request<DeleteScratchpadMirrorRequest>,
    ) -> Result<Response<()>, Status> {
        let req = request.into_inner();
        if let Err(e) = self
            .write_actor
            .delete_scratchpad_mirror(DeleteScratchpadMirrorData {
                scratchpad_id: req.scratchpad_id,
            })
            .await
        {
            tracing::warn!("delete_scratchpad_mirror failed (fire-and-forget): {}", e);
        }
        Ok(Response::new(()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_allowed_accepts_canonical_values() {
        check_allowed("actor", "benchmark", ALLOWED_ACTORS).unwrap();
        check_allowed("tool", "grep", ALLOWED_TOOLS).unwrap();
        check_allowed("op", "search", ALLOWED_OPS).unwrap();
    }

    #[test]
    fn check_allowed_rejects_unknown_with_invalid_argument() {
        let err = check_allowed("actor", "robot", ALLOWED_ACTORS).unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
        assert!(err.message().contains("robot"));
        assert!(err.message().contains("actor"));

        assert_eq!(
            check_allowed("tool", "nope", ALLOWED_TOOLS)
                .unwrap_err()
                .code(),
            tonic::Code::InvalidArgument
        );
        assert_eq!(
            check_allowed("op", "delete", ALLOWED_OPS)
                .unwrap_err()
                .code(),
            tonic::Code::InvalidArgument
        );
    }
}
