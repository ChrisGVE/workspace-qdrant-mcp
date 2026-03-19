//! TrackingWriteService gRPC implementation
//!
//! Daemon-exclusive writes for observability and mirror tables.
//! Replaces direct SQLite writes from MCP server and CLI.

use sqlx::SqlitePool;
use tonic::{Request, Response, Status};

use crate::proto::{
    tracking_write_service_server::TrackingWriteService, DeleteRuleMirrorRequest,
    LogSearchEventRequest, UpdateSearchEventRequest, UpsertRuleMirrorRequest,
};

pub struct TrackingWriteServiceImpl {
    #[allow(dead_code)]
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
        _request: Request<LogSearchEventRequest>,
    ) -> Result<Response<()>, Status> {
        Err(Status::unimplemented("LogSearchEvent not yet implemented"))
    }

    async fn update_search_event(
        &self,
        _request: Request<UpdateSearchEventRequest>,
    ) -> Result<Response<()>, Status> {
        Err(Status::unimplemented(
            "UpdateSearchEvent not yet implemented",
        ))
    }

    async fn upsert_rule_mirror(
        &self,
        _request: Request<UpsertRuleMirrorRequest>,
    ) -> Result<Response<()>, Status> {
        Err(Status::unimplemented(
            "UpsertRuleMirror not yet implemented",
        ))
    }

    async fn delete_rule_mirror(
        &self,
        _request: Request<DeleteRuleMirrorRequest>,
    ) -> Result<Response<()>, Status> {
        Err(Status::unimplemented(
            "DeleteRuleMirror not yet implemented",
        ))
    }
}
