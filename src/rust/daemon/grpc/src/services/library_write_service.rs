//! LibraryWriteService gRPC implementation
//!
//! Delegates all state.db mutations to the WriteActor via WriteActorHandle.

use tonic::{Request, Response, Status};
use workspace_qdrant_core::write_actor::{
    AddLibraryData, ConfigureLibraryData, RemoveLibraryData, SetIncrementalData,
    UnwatchLibraryData, WatchLibraryData, WriteActorHandle,
};

use crate::proto::{
    library_write_service_server::LibraryWriteService, AddLibraryRequest, AddLibraryResponse,
    ConfigureLibraryRequest, RemoveLibraryRequest, RemoveLibraryResponse, SetIncrementalRequest,
    SetIncrementalResponse, UnwatchLibraryRequest, WatchLibraryRequest, WatchLibraryResponse,
    WatchMutationResponse,
};

pub struct LibraryWriteServiceImpl {
    write_actor: WriteActorHandle,
}

impl LibraryWriteServiceImpl {
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
impl LibraryWriteService for LibraryWriteServiceImpl {
    async fn add_library(
        &self,
        request: Request<AddLibraryRequest>,
    ) -> Result<Response<AddLibraryResponse>, Status> {
        let req = request.into_inner();
        let result = self
            .write_actor
            .add_library(AddLibraryData {
                tag: req.tag,
                path: req.path,
                mode: req.mode,
            })
            .await
            .map_err(to_status)?;

        Ok(Response::new(AddLibraryResponse {
            success: result.success,
            watch_id: result.watch_id,
            message: result.message,
        }))
    }

    async fn remove_library(
        &self,
        request: Request<RemoveLibraryRequest>,
    ) -> Result<Response<RemoveLibraryResponse>, Status> {
        let req = request.into_inner();
        let result = self
            .write_actor
            .remove_library(RemoveLibraryData { tag: req.tag })
            .await
            .map_err(to_status)?;

        Ok(Response::new(RemoveLibraryResponse {
            success: result.success,
            queue_items_cancelled: result.queue_items_cancelled,
            tracked_files_deleted: result.tracked_files_deleted,
            components_deleted: result.components_deleted,
            message: result.message,
        }))
    }

    async fn watch_library(
        &self,
        request: Request<WatchLibraryRequest>,
    ) -> Result<Response<WatchLibraryResponse>, Status> {
        let req = request.into_inner();
        let result = self
            .write_actor
            .watch_library(WatchLibraryData {
                tag: req.tag,
                path: req.path,
                mode: req.mode,
            })
            .await
            .map_err(to_status)?;

        Ok(Response::new(WatchLibraryResponse {
            success: result.success,
            is_new: result.is_new,
            watch_id: result.watch_id,
            message: result.message,
        }))
    }

    async fn unwatch_library(
        &self,
        request: Request<UnwatchLibraryRequest>,
    ) -> Result<Response<WatchMutationResponse>, Status> {
        let req = request.into_inner();
        let affected_count = self
            .write_actor
            .unwatch_library(UnwatchLibraryData { tag: req.tag })
            .await
            .map_err(to_status)?;

        Ok(Response::new(WatchMutationResponse { affected_count }))
    }

    async fn configure_library(
        &self,
        request: Request<ConfigureLibraryRequest>,
    ) -> Result<Response<WatchMutationResponse>, Status> {
        let req = request.into_inner();
        let affected_count = self
            .write_actor
            .configure_library(ConfigureLibraryData {
                tag: req.tag,
                mode: req.mode,
                enable: req.enable,
                disable: req.disable,
            })
            .await
            .map_err(to_status)?;

        Ok(Response::new(WatchMutationResponse { affected_count }))
    }

    async fn set_incremental(
        &self,
        request: Request<SetIncrementalRequest>,
    ) -> Result<Response<SetIncrementalResponse>, Status> {
        let req = request.into_inner();
        let result = self
            .write_actor
            .set_incremental(SetIncrementalData {
                file_paths: req.file_paths,
                clear: req.clear,
            })
            .await
            .map_err(to_status)?;

        Ok(Response::new(SetIncrementalResponse {
            updated: result.updated,
            not_found: result.not_found,
        }))
    }
}
