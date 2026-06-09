//! QueueWriteService gRPC implementation
//!
//! Delegates all state.db mutations to the WriteActor via WriteActorHandle.

use tonic::{Request, Response, Status};
use workspace_qdrant_core::write_actor::{
    CancelItemsData, CleanQueueByCollectionData, CleanQueueData, EnqueueItemData, RemoveItemData,
    RetryItemData, WriteActorHandle,
};

use crate::proto::{
    queue_write_service_server::QueueWriteService, CancelItemsRequest, CancelItemsResponse,
    CleanQueueByCollectionRequest, CleanQueueRequest, CleanQueueResponse, EnqueueItemRequest,
    EnqueueItemResponse, PurgeDlqRequest, PurgeDlqResponse, RemoveItemRequest, RemoveItemResponse,
    ReplayDlqItemRequest, ReplayDlqItemResponse, RetryAllResponse, RetryItemRequest,
    RetryItemResponse,
};

pub struct QueueWriteServiceImpl {
    write_actor: WriteActorHandle,
}

impl QueueWriteServiceImpl {
    pub fn new(write_actor: WriteActorHandle) -> Self {
        Self { write_actor }
    }
}

/// Convert a WriteActor error string into a tonic Status.
fn to_status(err: String) -> Status {
    if err.contains("not found") || err.contains("no project found") {
        Status::not_found(err)
    } else if err.starts_with("invalid")
        || err.contains("cannot be empty")
        || err.contains("must be positive")
        || err.contains("no cancellable")
        || err.contains("no deletable")
        || err.contains("at least one")
    {
        Status::invalid_argument(err)
    } else {
        Status::internal(err)
    }
}

#[tonic::async_trait]
impl QueueWriteService for QueueWriteServiceImpl {
    async fn enqueue_item(
        &self,
        request: Request<EnqueueItemRequest>,
    ) -> Result<Response<EnqueueItemResponse>, Status> {
        let req = request.into_inner();
        let result = self
            .write_actor
            .enqueue_item(EnqueueItemData {
                item_type: req.item_type,
                op: req.op,
                tenant_id: req.tenant_id,
                collection: req.collection,
                payload_json: req.payload_json,
                branch: req.branch,
                metadata_json: req.metadata_json,
            })
            .await
            .map_err(to_status)?;

        Ok(Response::new(EnqueueItemResponse {
            queue_id: result.queue_id,
            idempotency_key: result.idempotency_key,
            is_new: result.is_new,
        }))
    }

    async fn retry_all(&self, _request: Request<()>) -> Result<Response<RetryAllResponse>, Status> {
        let result = self.write_actor.retry_all().await.map_err(to_status)?;
        Ok(Response::new(RetryAllResponse {
            reset_count: result.reset_count,
        }))
    }

    async fn retry_item(
        &self,
        request: Request<RetryItemRequest>,
    ) -> Result<Response<RetryItemResponse>, Status> {
        let req = request.into_inner();
        let result = self
            .write_actor
            .retry_item(RetryItemData {
                queue_id: req.queue_id,
            })
            .await
            .map_err(to_status)?;

        Ok(Response::new(RetryItemResponse {
            found: result.found,
            resolved_id: result.resolved_id,
            previous_status: result.previous_status,
            previous_retry_count: result.previous_retry_count,
            reset: result.reset,
        }))
    }

    async fn clean_queue(
        &self,
        request: Request<CleanQueueRequest>,
    ) -> Result<Response<CleanQueueResponse>, Status> {
        let req = request.into_inner();
        let deleted_count = self
            .write_actor
            .clean_queue(CleanQueueData {
                older_than_days: req.older_than_days,
                statuses: req.statuses,
            })
            .await
            .map_err(to_status)?;

        Ok(Response::new(CleanQueueResponse { deleted_count }))
    }

    async fn cancel_items(
        &self,
        request: Request<CancelItemsRequest>,
    ) -> Result<Response<CancelItemsResponse>, Status> {
        let req = request.into_inner();
        let result = self
            .write_actor
            .cancel_items(CancelItemsData {
                tenant_id: req.tenant_id,
                statuses: req.statuses,
                dry_run: req.dry_run,
            })
            .await
            .map_err(to_status)?;

        Ok(Response::new(CancelItemsResponse {
            count: result.count,
            tenant_id: result.tenant_id,
            project_path: result.project_path,
            is_dry_run: result.is_dry_run,
        }))
    }

    async fn remove_item(
        &self,
        request: Request<RemoveItemRequest>,
    ) -> Result<Response<RemoveItemResponse>, Status> {
        let req = request.into_inner();
        let result = self
            .write_actor
            .remove_item(RemoveItemData {
                queue_id: req.queue_id,
            })
            .await
            .map_err(to_status)?;

        Ok(Response::new(RemoveItemResponse {
            found: result.found,
            resolved_id: result.resolved_id,
            item_type: result.item_type,
            op: result.op,
            collection: result.collection,
            status: result.status,
        }))
    }

    async fn clean_queue_by_collection(
        &self,
        request: Request<CleanQueueByCollectionRequest>,
    ) -> Result<Response<CleanQueueResponse>, Status> {
        let req = request.into_inner();
        let deleted_count = self
            .write_actor
            .clean_queue_by_collection(CleanQueueByCollectionData {
                collections: req.collections,
                statuses: req.statuses,
            })
            .await
            .map_err(to_status)?;

        Ok(Response::new(CleanQueueResponse { deleted_count }))
    }

    async fn replay_dlq_item(
        &self,
        request: Request<ReplayDlqItemRequest>,
    ) -> Result<Response<ReplayDlqItemResponse>, Status> {
        let req = request.into_inner();
        match self
            .write_actor
            .replay_dlq_item(req.dlq_id, req.force)
            .await
        {
            Ok(new_queue_id) => Ok(Response::new(ReplayDlqItemResponse {
                success: true,
                new_queue_id,
                error: String::new(),
            })),
            Err(e) => Ok(Response::new(ReplayDlqItemResponse {
                success: false,
                new_queue_id: String::new(),
                error: e,
            })),
        }
    }

    async fn purge_dlq(
        &self,
        request: Request<PurgeDlqRequest>,
    ) -> Result<Response<PurgeDlqResponse>, Status> {
        let req = request.into_inner();
        // Use the retention verbatim — the default (30 days when the flag is
        // absent) is owned by the CLI's `--older-than` arg, so a value reaching
        // here is always an explicit operator choice. `0` means "older than now"
        // → purge every entry, which is exactly how an operator clears a recent
        // flood (#119). Only a negative value (cutoff in the future → would also
        // purge everything) is rejected, so a typo cannot silently empty the DLQ.
        if req.retention_days < 0 {
            return Err(Status::invalid_argument(format!(
                "retention_days must be >= 0 (0 purges all entries); got {}",
                req.retention_days
            )));
        }
        let retention_days = req.retention_days as u32;
        match self.write_actor.purge_dlq(retention_days).await {
            Ok((deleted, has_more)) => Ok(Response::new(PurgeDlqResponse {
                rows_deleted: deleted as i32,
                has_more,
            })),
            Err(e) => Err(Status::internal(e)),
        }
    }
}
