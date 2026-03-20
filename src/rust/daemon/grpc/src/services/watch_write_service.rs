//! WatchWriteService gRPC implementation
//!
//! Delegates all state.db mutations to the WriteActor via WriteActorHandle.

use tonic::{Request, Response, Status};
use workspace_qdrant_core::write_actor::{ArchiveWatchData, WatchIdData, WriteActorHandle};

use crate::proto::{
    watch_write_service_server::WatchWriteService, ArchiveWatchRequest, ArchiveWatchResponse,
    WatchIdRequest, WatchMutationResponse,
};

pub struct WatchWriteServiceImpl {
    write_actor: WriteActorHandle,
}

impl WatchWriteServiceImpl {
    pub fn new(write_actor: WriteActorHandle) -> Self {
        Self { write_actor }
    }
}

fn to_status(err: String) -> Status {
    if err.contains("not found") {
        Status::not_found(err)
    } else {
        Status::internal(err)
    }
}

#[tonic::async_trait]
impl WatchWriteService for WatchWriteServiceImpl {
    async fn pause_watchers(
        &self,
        _request: Request<()>,
    ) -> Result<Response<WatchMutationResponse>, Status> {
        let affected_count = self.write_actor.pause_watchers().await.map_err(to_status)?;
        Ok(Response::new(WatchMutationResponse { affected_count }))
    }

    async fn resume_watchers(
        &self,
        _request: Request<()>,
    ) -> Result<Response<WatchMutationResponse>, Status> {
        let affected_count = self
            .write_actor
            .resume_watchers()
            .await
            .map_err(to_status)?;
        Ok(Response::new(WatchMutationResponse { affected_count }))
    }

    async fn enable_watch(
        &self,
        request: Request<WatchIdRequest>,
    ) -> Result<Response<WatchMutationResponse>, Status> {
        let req = request.into_inner();
        let affected_count = self
            .write_actor
            .enable_watch(WatchIdData {
                watch_id: req.watch_id,
            })
            .await
            .map_err(to_status)?;
        Ok(Response::new(WatchMutationResponse { affected_count }))
    }

    async fn disable_watch(
        &self,
        request: Request<WatchIdRequest>,
    ) -> Result<Response<WatchMutationResponse>, Status> {
        let req = request.into_inner();
        let affected_count = self
            .write_actor
            .disable_watch(WatchIdData {
                watch_id: req.watch_id,
            })
            .await
            .map_err(to_status)?;
        Ok(Response::new(WatchMutationResponse { affected_count }))
    }

    async fn archive_watch(
        &self,
        request: Request<ArchiveWatchRequest>,
    ) -> Result<Response<ArchiveWatchResponse>, Status> {
        let req = request.into_inner();
        let result = self
            .write_actor
            .archive_watch(ArchiveWatchData {
                watch_id: req.watch_id,
                cascade_submodules: req.cascade_submodules,
            })
            .await
            .map_err(to_status)?;

        Ok(Response::new(ArchiveWatchResponse {
            affected_count: result.affected_count,
            submodules_archived: result.submodules_archived,
            submodules_skipped: result.submodules_skipped,
        }))
    }

    async fn unarchive_watch(
        &self,
        request: Request<WatchIdRequest>,
    ) -> Result<Response<WatchMutationResponse>, Status> {
        let req = request.into_inner();
        let affected_count = self
            .write_actor
            .unarchive_watch(WatchIdData {
                watch_id: req.watch_id,
            })
            .await
            .map_err(to_status)?;
        Ok(Response::new(WatchMutationResponse { affected_count }))
    }
}
