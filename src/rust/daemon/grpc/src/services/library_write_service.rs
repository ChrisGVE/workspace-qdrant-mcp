//! LibraryWriteService gRPC implementation
//!
//! Delegates all state.db mutations to the WriteActor via WriteActorHandle.

use tonic::{Request, Response, Status};
use workspace_qdrant_core::write_actor::{
    AddLibraryData, ConfigureLibraryData, RemoveLibraryData, SetIncrementalData,
    UnwatchLibraryData, WatchLibraryData, WriteActorHandle,
};

use std::sync::Arc;

use sqlx::SqlitePool;
use workspace_qdrant_core::StorageClient;

use crate::proto::{
    library_write_service_server::LibraryWriteService, AddLibraryRequest, AddLibraryResponse,
    ConfigureLibraryRequest, RecoverLibraryRequest, RecoverLibraryResponse, RemoveLibraryRequest,
    RemoveLibraryResponse, SetIncrementalRequest, SetIncrementalResponse, UnwatchLibraryRequest,
    WatchLibraryRequest, WatchLibraryResponse, WatchMutationResponse,
};
use crate::validation::{extract_canonical_path, extract_relative_paths};

mod recover_library;

pub struct LibraryWriteServiceImpl {
    write_actor: WriteActorHandle,
    /// state.db pool + Qdrant client, wired for the recover cascade (#140).
    /// Optional so existing constructors (path-validation tests) stay simple;
    /// when absent, RecoverLibrary returns an internal error rather than a stub.
    db_pool: Option<SqlitePool>,
    storage: Option<Arc<StorageClient>>,
}

impl LibraryWriteServiceImpl {
    pub fn new(write_actor: WriteActorHandle) -> Self {
        Self {
            write_actor,
            db_pool: None,
            storage: None,
        }
    }

    /// Attach the state.db pool and Qdrant client needed by RecoverLibrary
    /// (#140). All other RPCs delegate to the WriteActor and ignore these.
    pub fn with_recover_deps(
        mut self,
        db_pool: SqlitePool,
        storage: Arc<StorageClient>,
    ) -> Self {
        self.db_pool = Some(db_pool);
        self.storage = Some(storage);
        self
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

        // Validate path as CanonicalPath (absolute library root).
        let canonical = extract_canonical_path!(req.path, "path")?;

        let result = self
            .write_actor
            .add_library(AddLibraryData {
                tag: req.tag,
                path: canonical.into_string(),
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

        // Validate path as CanonicalPath (absolute watch root).
        let canonical = extract_canonical_path!(req.path, "path")?;

        let result = self
            .write_actor
            .watch_library(WatchLibraryData {
                tag: req.tag,
                path: canonical.into_string(),
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

    async fn recover_library(
        &self,
        request: Request<RecoverLibraryRequest>,
    ) -> Result<Response<RecoverLibraryResponse>, Status> {
        let req = request.into_inner();
        self.handle_recover_library(req).await.map(Response::new)
    }

    async fn set_incremental(
        &self,
        request: Request<SetIncrementalRequest>,
    ) -> Result<Response<SetIncrementalResponse>, Status> {
        let req = request.into_inner();

        // Validate each file_path as RelativePath (content paths relative to project root).
        let validated = extract_relative_paths!(req.file_paths, "file_paths")?;
        let validated_strings: Vec<String> =
            validated.into_iter().map(|rp| rp.into_string()).collect();

        if req.watch_folder_id.is_empty() {
            return Err(Status::invalid_argument(
                "watch_folder_id is required to scope SetIncremental updates",
            ));
        }
        let watch_folder_id = Some(req.watch_folder_id);

        let result = self
            .write_actor
            .set_incremental(SetIncrementalData {
                file_paths: validated_strings,
                clear: req.clear,
                watch_folder_id,
            })
            .await
            .map_err(to_status)?;

        Ok(Response::new(SetIncrementalResponse {
            updated: result.updated,
            not_found: result.not_found,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use workspace_qdrant_core::write_actor::WriteActor;

    use crate::proto::library_write_service_server::LibraryWriteService;

    /// Create a minimal WriteActorHandle backed by an in-memory SQLite pool.
    /// Path validation rejects before the actor is ever called, so schema
    /// does not need to be fully initialized.
    async fn test_write_actor() -> WriteActorHandle {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        WriteActor::spawn(pool)
    }

    // ── AddLibraryRequest.path (canonical) ───────────────────────────

    #[tokio::test]
    async fn test_add_library_relative_path_rejected() {
        let service = LibraryWriteServiceImpl::new(test_write_actor().await);

        let request = Request::new(AddLibraryRequest {
            tag: "test-lib".to_string(),
            path: "relative/library".to_string(),
            mode: "full".to_string(),
        });

        let result = service.add_library(request).await;
        assert!(result.is_err());
        let status = result.unwrap_err();
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert!(
            status.message().contains("path"),
            "error should mention field name, got: {}",
            status.message()
        );
    }

    #[tokio::test]
    async fn test_add_library_parent_dir_rejected() {
        let service = LibraryWriteServiceImpl::new(test_write_actor().await);

        let request = Request::new(AddLibraryRequest {
            tag: "test-lib".to_string(),
            path: "/Users/chris/../escape".to_string(),
            mode: "full".to_string(),
        });

        let result = service.add_library(request).await;
        assert!(result.is_err());
        let status = result.unwrap_err();
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert!(status.message().contains(".."));
    }

    #[tokio::test]
    async fn test_add_library_empty_path_rejected() {
        let service = LibraryWriteServiceImpl::new(test_write_actor().await);

        let request = Request::new(AddLibraryRequest {
            tag: "test-lib".to_string(),
            path: String::new(),
            mode: "full".to_string(),
        });

        let result = service.add_library(request).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code(), tonic::Code::InvalidArgument);
    }

    // ── WatchLibraryRequest.path (canonical) ─────────────────────────

    #[tokio::test]
    async fn test_watch_library_relative_path_rejected() {
        let service = LibraryWriteServiceImpl::new(test_write_actor().await);

        let request = Request::new(WatchLibraryRequest {
            tag: "test-lib".to_string(),
            path: "not/absolute".to_string(),
            mode: "incremental".to_string(),
            patterns: vec![],
        });

        let result = service.watch_library(request).await;
        assert!(result.is_err());
        let status = result.unwrap_err();
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert!(status.message().contains("path"));
    }

    #[tokio::test]
    async fn test_watch_library_parent_dir_rejected() {
        let service = LibraryWriteServiceImpl::new(test_write_actor().await);

        let request = Request::new(WatchLibraryRequest {
            tag: "test-lib".to_string(),
            path: "/a/b/../c".to_string(),
            mode: "incremental".to_string(),
            patterns: vec![],
        });

        let result = service.watch_library(request).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code(), tonic::Code::InvalidArgument);
    }

    // ── SetIncrementalRequest.file_paths (relative, repeated) ────────

    #[tokio::test]
    async fn test_set_incremental_absolute_path_rejected() {
        let service = LibraryWriteServiceImpl::new(test_write_actor().await);

        let request = Request::new(SetIncrementalRequest {
            file_paths: vec!["src/ok.rs".to_string(), "/absolute/bad.rs".to_string()],
            clear: false,
            watch_folder_id: String::new(),
        });

        let result = service.set_incremental(request).await;
        assert!(result.is_err());
        let status = result.unwrap_err();
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert!(
            status.message().contains("file_paths[1]"),
            "error should include element index, got: {}",
            status.message()
        );
    }

    #[tokio::test]
    async fn test_set_incremental_parent_dir_rejected() {
        let service = LibraryWriteServiceImpl::new(test_write_actor().await);

        let request = Request::new(SetIncrementalRequest {
            file_paths: vec!["../escape.rs".to_string()],
            clear: false,
            watch_folder_id: String::new(),
        });

        let result = service.set_incremental(request).await;
        assert!(result.is_err());
        let status = result.unwrap_err();
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert!(status.message().contains("file_paths[0]"));
        assert!(status.message().contains(".."));
    }

    #[tokio::test]
    async fn test_set_incremental_empty_element_rejected() {
        let service = LibraryWriteServiceImpl::new(test_write_actor().await);

        let request = Request::new(SetIncrementalRequest {
            file_paths: vec![String::new()],
            clear: false,
            watch_folder_id: String::new(),
        });

        let result = service.set_incremental(request).await;
        assert!(result.is_err());
        let status = result.unwrap_err();
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert!(status.message().contains("file_paths[0]"));
    }

    #[tokio::test]
    async fn test_set_incremental_tilde_path_rejected() {
        let service = LibraryWriteServiceImpl::new(test_write_actor().await);

        let request = Request::new(SetIncrementalRequest {
            file_paths: vec!["~/not/relative".to_string()],
            clear: false,
            watch_folder_id: String::new(),
        });

        let result = service.set_incremental(request).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code(), tonic::Code::InvalidArgument);
    }

    #[tokio::test]
    async fn test_set_incremental_empty_watch_folder_id_rejected() {
        let service = LibraryWriteServiceImpl::new(test_write_actor().await);

        let request = Request::new(SetIncrementalRequest {
            file_paths: vec!["src/valid.rs".to_string()],
            clear: false,
            watch_folder_id: String::new(),
        });

        let result = service.set_incremental(request).await;
        assert!(result.is_err());
        let status = result.unwrap_err();
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert!(
            status.message().contains("watch_folder_id"),
            "error should mention watch_folder_id, got: {}",
            status.message()
        );
    }
}
