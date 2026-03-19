//! WatchWriteService gRPC implementation
//!
//! Daemon-exclusive writes to the watch_folders table.
//! Replaces direct SQLite writes from CLI watch commands.

use sqlx::SqlitePool;
use tonic::{Request, Response, Status};

use crate::proto::{
    watch_write_service_server::WatchWriteService, ArchiveWatchRequest, ArchiveWatchResponse,
    WatchIdRequest, WatchMutationResponse,
};

pub struct WatchWriteServiceImpl {
    #[allow(dead_code)]
    pool: SqlitePool,
}

impl WatchWriteServiceImpl {
    pub fn new(pool: SqlitePool) -> Self {
        Self { pool }
    }
}

#[tonic::async_trait]
impl WatchWriteService for WatchWriteServiceImpl {
    async fn pause_watchers(
        &self,
        _request: Request<()>,
    ) -> Result<Response<WatchMutationResponse>, Status> {
        Err(Status::unimplemented("PauseWatchers not yet implemented"))
    }

    async fn resume_watchers(
        &self,
        _request: Request<()>,
    ) -> Result<Response<WatchMutationResponse>, Status> {
        Err(Status::unimplemented("ResumeWatchers not yet implemented"))
    }

    async fn enable_watch(
        &self,
        _request: Request<WatchIdRequest>,
    ) -> Result<Response<WatchMutationResponse>, Status> {
        Err(Status::unimplemented("EnableWatch not yet implemented"))
    }

    async fn disable_watch(
        &self,
        _request: Request<WatchIdRequest>,
    ) -> Result<Response<WatchMutationResponse>, Status> {
        Err(Status::unimplemented("DisableWatch not yet implemented"))
    }

    async fn archive_watch(
        &self,
        _request: Request<ArchiveWatchRequest>,
    ) -> Result<Response<ArchiveWatchResponse>, Status> {
        Err(Status::unimplemented("ArchiveWatch not yet implemented"))
    }

    async fn unarchive_watch(
        &self,
        _request: Request<WatchIdRequest>,
    ) -> Result<Response<WatchMutationResponse>, Status> {
        Err(Status::unimplemented("UnarchiveWatch not yet implemented"))
    }
}
