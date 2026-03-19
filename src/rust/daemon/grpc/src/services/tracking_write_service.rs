//! TrackingWriteService gRPC implementation
//!
//! Delegates all state.db mutations to the WriteActor via WriteActorHandle.
//! Errors are logged but not propagated — instrumentation must never block.

use tonic::{Request, Response, Status};
use workspace_qdrant_core::write_actor::{
    DeleteRuleMirrorData, LogSearchEventData, UpdateSearchEventData, UpsertRuleMirrorData,
    WriteActorHandle,
};

use crate::proto::{
    tracking_write_service_server::TrackingWriteService, DeleteRuleMirrorRequest,
    LogSearchEventRequest, UpdateSearchEventRequest, UpsertRuleMirrorRequest,
};

pub struct TrackingWriteServiceImpl {
    write_actor: WriteActorHandle,
}

impl TrackingWriteServiceImpl {
    pub fn new(write_actor: WriteActorHandle) -> Self {
        Self { write_actor }
    }
}

#[tonic::async_trait]
impl TrackingWriteService for TrackingWriteServiceImpl {
    async fn log_search_event(
        &self,
        request: Request<LogSearchEventRequest>,
    ) -> Result<Response<()>, Status> {
        let req = request.into_inner();
        // Fire-and-forget: errors logged inside WriteActor
        let _ = self
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
            .await;
        Ok(Response::new(()))
    }

    async fn update_search_event(
        &self,
        request: Request<UpdateSearchEventRequest>,
    ) -> Result<Response<()>, Status> {
        let req = request.into_inner();
        let _ = self
            .write_actor
            .update_search_event(UpdateSearchEventData {
                event_id: req.event_id,
                result_count: Some(req.result_count),
                latency_ms: Some(req.latency_ms),
                top_result_refs: req.top_result_refs,
                outcome: req.outcome,
            })
            .await;
        Ok(Response::new(()))
    }

    async fn upsert_rule_mirror(
        &self,
        request: Request<UpsertRuleMirrorRequest>,
    ) -> Result<Response<()>, Status> {
        let req = request.into_inner();
        let _ = self
            .write_actor
            .upsert_rule_mirror(UpsertRuleMirrorData {
                rule_id: req.rule_id,
                rule_text: req.rule_text,
                scope: req.scope.unwrap_or_default(),
                tenant_id: req.tenant_id.unwrap_or_default(),
                created_at: req.created_at,
                updated_at: req.updated_at,
            })
            .await;
        Ok(Response::new(()))
    }

    async fn delete_rule_mirror(
        &self,
        request: Request<DeleteRuleMirrorRequest>,
    ) -> Result<Response<()>, Status> {
        let req = request.into_inner();
        let _ = self
            .write_actor
            .delete_rule_mirror(DeleteRuleMirrorData {
                rule_id: req.rule_id,
            })
            .await;
        Ok(Response::new(()))
    }
}
