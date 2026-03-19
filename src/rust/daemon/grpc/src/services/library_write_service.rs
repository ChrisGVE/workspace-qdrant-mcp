//! LibraryWriteService gRPC implementation
//!
//! Daemon-exclusive writes for library management.
//! Replaces direct SQLite writes from CLI library commands.

use sqlx::SqlitePool;
use tonic::{Request, Response, Status};

use crate::proto::{
    library_write_service_server::LibraryWriteService, AddLibraryRequest, AddLibraryResponse,
    ConfigureLibraryRequest, RemoveLibraryRequest, RemoveLibraryResponse, SetIncrementalRequest,
    SetIncrementalResponse, UnwatchLibraryRequest, WatchLibraryRequest, WatchLibraryResponse,
    WatchMutationResponse,
};

pub struct LibraryWriteServiceImpl {
    #[allow(dead_code)]
    pool: SqlitePool,
}

impl LibraryWriteServiceImpl {
    pub fn new(pool: SqlitePool) -> Self {
        Self { pool }
    }
}

#[tonic::async_trait]
impl LibraryWriteService for LibraryWriteServiceImpl {
    async fn add_library(
        &self,
        _request: Request<AddLibraryRequest>,
    ) -> Result<Response<AddLibraryResponse>, Status> {
        Err(Status::unimplemented("AddLibrary not yet implemented"))
    }

    async fn remove_library(
        &self,
        _request: Request<RemoveLibraryRequest>,
    ) -> Result<Response<RemoveLibraryResponse>, Status> {
        Err(Status::unimplemented("RemoveLibrary not yet implemented"))
    }

    async fn watch_library(
        &self,
        _request: Request<WatchLibraryRequest>,
    ) -> Result<Response<WatchLibraryResponse>, Status> {
        Err(Status::unimplemented("WatchLibrary not yet implemented"))
    }

    async fn unwatch_library(
        &self,
        _request: Request<UnwatchLibraryRequest>,
    ) -> Result<Response<WatchMutationResponse>, Status> {
        Err(Status::unimplemented("UnwatchLibrary not yet implemented"))
    }

    async fn configure_library(
        &self,
        _request: Request<ConfigureLibraryRequest>,
    ) -> Result<Response<WatchMutationResponse>, Status> {
        Err(Status::unimplemented(
            "ConfigureLibrary not yet implemented",
        ))
    }

    async fn set_incremental(
        &self,
        _request: Request<SetIncrementalRequest>,
    ) -> Result<Response<SetIncrementalResponse>, Status> {
        Err(Status::unimplemented("SetIncremental not yet implemented"))
    }
}
