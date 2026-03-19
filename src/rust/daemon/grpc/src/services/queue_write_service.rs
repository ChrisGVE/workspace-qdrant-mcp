//! QueueWriteService gRPC implementation
//!
//! Daemon-exclusive writes to the unified_queue table.
//! Replaces direct SQLite writes from CLI and MCP server.

use sqlx::SqlitePool;
use tonic::{Request, Response, Status};

use crate::proto::{
    queue_write_service_server::QueueWriteService, CancelItemsRequest, CancelItemsResponse,
    CleanQueueRequest, CleanQueueResponse, EnqueueItemRequest, EnqueueItemResponse,
    RemoveItemRequest, RemoveItemResponse, RetryAllResponse, RetryItemRequest, RetryItemResponse,
};

pub struct QueueWriteServiceImpl {
    #[allow(dead_code)]
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
        _request: Request<EnqueueItemRequest>,
    ) -> Result<Response<EnqueueItemResponse>, Status> {
        Err(Status::unimplemented("EnqueueItem not yet implemented"))
    }

    async fn retry_all(&self, _request: Request<()>) -> Result<Response<RetryAllResponse>, Status> {
        Err(Status::unimplemented("RetryAll not yet implemented"))
    }

    async fn retry_item(
        &self,
        _request: Request<RetryItemRequest>,
    ) -> Result<Response<RetryItemResponse>, Status> {
        Err(Status::unimplemented("RetryItem not yet implemented"))
    }

    async fn clean_queue(
        &self,
        _request: Request<CleanQueueRequest>,
    ) -> Result<Response<CleanQueueResponse>, Status> {
        Err(Status::unimplemented("CleanQueue not yet implemented"))
    }

    async fn cancel_items(
        &self,
        _request: Request<CancelItemsRequest>,
    ) -> Result<Response<CancelItemsResponse>, Status> {
        Err(Status::unimplemented("CancelItems not yet implemented"))
    }

    async fn remove_item(
        &self,
        _request: Request<RemoveItemRequest>,
    ) -> Result<Response<RemoveItemResponse>, Status> {
        Err(Status::unimplemented("RemoveItem not yet implemented"))
    }
}
