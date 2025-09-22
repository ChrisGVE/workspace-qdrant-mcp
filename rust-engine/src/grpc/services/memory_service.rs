//! Memory management gRPC service implementation

use crate::daemon::WorkspaceDaemon;
use crate::proto::{
    memory_service_server::MemoryService,
    AddDocumentRequest, AddDocumentResponse,
    UpdateDocumentRequest, UpdateDocumentResponse,
    RemoveDocumentRequest, GetDocumentRequest, GetDocumentResponse,
    ListDocumentsRequest, ListDocumentsResponse,
    CreateCollectionRequest, CreateCollectionResponse,
    DeleteCollectionRequest, ListCollectionsRequest, ListCollectionsResponse,
    DocumentInfo, CollectionInfo, DocumentContent, CollectionConfig,
};
use std::sync::Arc;
use tonic::{Request, Response, Status};
use tracing::{debug, error, info};

/// Memory service implementation
#[derive(Debug)]
pub struct MemoryServiceImpl {
    daemon: Arc<WorkspaceDaemon>,
}

impl MemoryServiceImpl {
    pub fn new(daemon: Arc<WorkspaceDaemon>) -> Self {
        Self { daemon }
    }
}

#[tonic::async_trait]
impl MemoryService for MemoryServiceImpl {
    async fn add_document(
        &self,
        request: Request<AddDocumentRequest>,
    ) -> Result<Response<AddDocumentResponse>, Status> {
        let req = request.into_inner();
        debug!("Adding document: {}", req.file_path);

        // TODO: Implement actual document addition
        let response = AddDocumentResponse {
            document_id: uuid::Uuid::new_v4().to_string(),
            success: true,
            error_message: String::new(),
        };

        Ok(Response::new(response))
    }

    async fn update_document(
        &self,
        request: Request<UpdateDocumentRequest>,
    ) -> Result<Response<UpdateDocumentResponse>, Status> {
        let req = request.into_inner();
        debug!("Updating document: {}", req.document_id);

        // TODO: Implement actual document update
        let response = UpdateDocumentResponse {
            success: true,
            error_message: String::new(),
            updated_at: Some(prost_types::Timestamp {
                seconds: chrono::Utc::now().timestamp(),
                nanos: 0,
            }),
        };

        Ok(Response::new(response))
    }

    async fn remove_document(
        &self,
        request: Request<RemoveDocumentRequest>,
    ) -> Result<Response<()>, Status> {
        let req = request.into_inner();
        info!("Removing document: {} from collection: {}", req.document_id, req.collection_name);

        // TODO: Implement actual document removal

        Ok(Response::new(()))
    }

    async fn get_document(
        &self,
        request: Request<GetDocumentRequest>,
    ) -> Result<Response<GetDocumentResponse>, Status> {
        let req = request.into_inner();
        debug!("Getting document: {} from collection: {}", req.document_id, req.collection_name);

        // TODO: Implement actual document retrieval
        let response = GetDocumentResponse {
            document_id: req.document_id.clone(),
            content: Some(DocumentContent {
                text: "Example document content".to_string(),
                chunks: vec![],
                extracted_metadata: std::collections::HashMap::new(),
            }),
            metadata: std::collections::HashMap::new(),
            created_at: Some(prost_types::Timestamp {
                seconds: chrono::Utc::now().timestamp() - 86400,
                nanos: 0,
            }),
            updated_at: Some(prost_types::Timestamp {
                seconds: chrono::Utc::now().timestamp(),
                nanos: 0,
            }),
        };

        Ok(Response::new(response))
    }

    async fn list_documents(
        &self,
        request: Request<ListDocumentsRequest>,
    ) -> Result<Response<ListDocumentsResponse>, Status> {
        let req = request.into_inner();
        debug!("Listing documents in collection: {}", req.collection_name);

        // TODO: Implement actual document listing
        let response = ListDocumentsResponse {
            documents: vec![
                DocumentInfo {
                    document_id: uuid::Uuid::new_v4().to_string(),
                    file_path: "/example/document.txt".to_string(),
                    title: "Example Document".to_string(),
                    document_type: crate::proto::DocumentType::Text as i32,
                    file_size: 1024,
                    metadata: std::collections::HashMap::new(),
                    created_at: Some(prost_types::Timestamp {
                        seconds: chrono::Utc::now().timestamp() - 86400,
                        nanos: 0,
                    }),
                    updated_at: Some(prost_types::Timestamp {
                        seconds: chrono::Utc::now().timestamp(),
                        nanos: 0,
                    }),
                },
            ],
            total_count: 1,
            has_more: false,
        };

        Ok(Response::new(response))
    }

    async fn create_collection(
        &self,
        request: Request<CreateCollectionRequest>,
    ) -> Result<Response<CreateCollectionResponse>, Status> {
        let req = request.into_inner();
        info!("Creating collection: {} for project: {}", req.collection_name, req.project_id);

        // TODO: Implement actual collection creation
        let response = CreateCollectionResponse {
            success: true,
            error_message: String::new(),
            collection_id: uuid::Uuid::new_v4().to_string(),
        };

        Ok(Response::new(response))
    }

    async fn delete_collection(
        &self,
        request: Request<DeleteCollectionRequest>,
    ) -> Result<Response<()>, Status> {
        let req = request.into_inner();
        info!("Deleting collection: {} for project: {} (force: {})",
              req.collection_name, req.project_id, req.force);

        // TODO: Implement actual collection deletion

        Ok(Response::new(()))
    }

    async fn list_collections(
        &self,
        request: Request<ListCollectionsRequest>,
    ) -> Result<Response<ListCollectionsResponse>, Status> {
        let req = request.into_inner();
        debug!("Listing collections for project: {}", req.project_id);

        // TODO: Implement actual collection listing
        let response = ListCollectionsResponse {
            collections: vec![
                CollectionInfo {
                    collection_name: "example-collection".to_string(),
                    collection_id: uuid::Uuid::new_v4().to_string(),
                    project_id: req.project_id.clone(),
                    document_count: 10,
                    total_size_bytes: 10240,
                    config: Some(CollectionConfig {
                        vector_size: 384,
                        distance_metric: "Cosine".to_string(),
                        enable_indexing: true,
                        metadata_schema: std::collections::HashMap::new(),
                    }),
                    created_at: Some(prost_types::Timestamp {
                        seconds: chrono::Utc::now().timestamp() - 86400,
                        nanos: 0,
                    }),
                },
            ],
        };

        Ok(Response::new(response))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::*;
    use tempfile::TempDir;
    use tonic::Request;
    use tokio_test;
    use std::collections::HashMap;

    fn create_test_daemon_config() -> DaemonConfig {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let db_path = temp_dir.path().join("test.db");

        DaemonConfig {
            server: ServerConfig {
                host: "127.0.0.1".to_string(),
                port: 0,
                max_connections: 100,
                connection_timeout_secs: 30,
                request_timeout_secs: 60,
                enable_tls: false,
            },
            database: DatabaseConfig {
                sqlite_path: db_path.to_string_lossy().to_string(),
                max_connections: 5,
                connection_timeout_secs: 30,
                enable_wal: true,
            },
            qdrant: QdrantConfig {
                url: "http://localhost:6333".to_string(),
                api_key: None,
                timeout_secs: 30,
                max_retries: 3,
                default_collection: crate::config::CollectionConfig {
                    vector_size: 384,
                    distance_metric: "Cosine".to_string(),
                    enable_indexing: true,
                    replication_factor: 1,
                    shard_number: 1,
                },
            },
            processing: ProcessingConfig {
                max_concurrent_tasks: 2,
                default_chunk_size: 1000,
                default_chunk_overlap: 200,
                max_file_size_bytes: 1024 * 1024,
                supported_extensions: vec!["txt".to_string(), "md".to_string()],
                enable_lsp: false,
                lsp_timeout_secs: 10,
            },
            file_watcher: FileWatcherConfig {
                enabled: false,
                debounce_ms: 500,
                max_watched_dirs: 10,
                ignore_patterns: vec![],
                recursive: true,
            },
            metrics: MetricsConfig {
                enabled: false,
                collection_interval_secs: 60,
                retention_days: 30,
                enable_prometheus: false,
                prometheus_port: 9090,
            },
            logging: LoggingConfig {
                level: "info".to_string(),
                file_path: None,
                json_format: false,
                max_file_size_mb: 100,
                max_files: 5,
            },
        }
    }

    async fn create_test_daemon() -> Arc<WorkspaceDaemon> {
        let config = create_test_daemon_config();
        Arc::new(WorkspaceDaemon::new(config).await.expect("Failed to create daemon"))
    }

    #[tokio::test]
    async fn test_memory_service_impl_new() {
        let daemon = create_test_daemon().await;
        let service = MemoryServiceImpl::new(daemon.clone());

        assert!(Arc::ptr_eq(&service.daemon, &daemon));
    }

    #[tokio::test]
    async fn test_memory_service_impl_debug() {
        let daemon = create_test_daemon().await;
        let service = MemoryServiceImpl::new(daemon);

        let debug_str = format!("{:?}", service);
        assert!(debug_str.contains("MemoryServiceImpl"));
        assert!(debug_str.contains("daemon"));
    }

    #[tokio::test]
    async fn test_add_document_basic() {
        let daemon = create_test_daemon().await;
        let service = MemoryServiceImpl::new(daemon);

        let request = Request::new(AddDocumentRequest {
            file_path: "/test/document.txt".to_string(),
            collection_name: "test_collection".to_string(),
            content: Some(DocumentContent {
                text: "Test document content".to_string(),
                chunks: vec![],
                extracted_metadata: HashMap::new(),
            }),
            metadata: HashMap::new(),
            project_id: "test_project".to_string(),
        });

        let result = service.add_document(request).await;
        assert!(result.is_ok());

        let response = result.unwrap().into_inner();
        assert!(!response.document_id.is_empty());
        assert!(response.success);
        assert!(response.error_message.is_empty());
    }

    #[tokio::test]
    async fn test_add_document_with_metadata() {
        let daemon = create_test_daemon().await;
        let service = MemoryServiceImpl::new(daemon);

        let mut metadata = HashMap::new();
        metadata.insert("author".to_string(), "John Doe".to_string());
        metadata.insert("category".to_string(), "documentation".to_string());
        metadata.insert("version".to_string(), "1.0".to_string());

        let mut extracted_metadata = HashMap::new();
        extracted_metadata.insert("word_count".to_string(), "150".to_string());
        extracted_metadata.insert("language".to_string(), "en".to_string());

        let request = Request::new(AddDocumentRequest {
            file_path: "/test/document_with_metadata.txt".to_string(),
            collection_name: "test_collection".to_string(),
            content: Some(DocumentContent {
                text: "Test document with metadata".to_string(),
                chunks: vec!["chunk1".to_string(), "chunk2".to_string()],
                extracted_metadata,
            }),
            metadata,
            project_id: "test_project".to_string(),
        });

        let result = service.add_document(request).await;
        assert!(result.is_ok());

        let response = result.unwrap().into_inner();
        assert!(!response.document_id.is_empty());
        assert!(response.success);
    }

    #[tokio::test]
    async fn test_add_document_different_file_types() {
        let daemon = create_test_daemon().await;
        let service = MemoryServiceImpl::new(daemon);

        let file_types = [
            ("document.txt", "text/plain"),
            ("readme.md", "text/markdown"),
            ("script.py", "text/x-python"),
            ("code.rs", "text/x-rust"),
            ("data.json", "application/json"),
            ("page.html", "text/html"),
        ];

        for (file_name, _mime_type) in file_types {
            let request = Request::new(AddDocumentRequest {
                file_path: format!("/test/{}", file_name),
                collection_name: "test_collection".to_string(),
                content: Some(DocumentContent {
                    text: format!("Content for {}", file_name),
                    chunks: vec![],
                    extracted_metadata: HashMap::new(),
                }),
                metadata: HashMap::new(),
                project_id: "test_project".to_string(),
            });

            let result = service.add_document(request).await;
            assert!(result.is_ok(), "Failed to add document: {}", file_name);

            let response = result.unwrap().into_inner();
            assert!(response.success);
        }
    }

    #[tokio::test]
    async fn test_update_document_basic() {
        let daemon = create_test_daemon().await;
        let service = MemoryServiceImpl::new(daemon);

        let request = Request::new(UpdateDocumentRequest {
            document_id: "test_doc_123".to_string(),
            collection_name: "test_collection".to_string(),
            content: Some(DocumentContent {
                text: "Updated document content".to_string(),
                chunks: vec![],
                extracted_metadata: HashMap::new(),
            }),
            metadata: HashMap::new(),
        });

        let result = service.update_document(request).await;
        assert!(result.is_ok());

        let response = result.unwrap().into_inner();
        assert!(response.success);
        assert!(response.error_message.is_empty());
        assert!(response.updated_at.is_some());
    }

    #[tokio::test]
    async fn test_update_document_timestamp() {
        let daemon = create_test_daemon().await;
        let service = MemoryServiceImpl::new(daemon);

        let before_update = chrono::Utc::now().timestamp();

        let request = Request::new(UpdateDocumentRequest {
            document_id: "timestamp_test_doc".to_string(),
            collection_name: "test_collection".to_string(),
            content: Some(DocumentContent {
                text: "Timestamp test content".to_string(),
                chunks: vec![],
                extracted_metadata: HashMap::new(),
            }),
            metadata: HashMap::new(),
        });

        let result = service.update_document(request).await;
        assert!(result.is_ok());

        let after_update = chrono::Utc::now().timestamp();
        let response = result.unwrap().into_inner();

        assert!(response.updated_at.is_some());
        let updated_timestamp = response.updated_at.unwrap().seconds;
        assert!(updated_timestamp >= before_update);
        assert!(updated_timestamp <= after_update);
    }

    #[tokio::test]
    async fn test_remove_document() {
        let daemon = create_test_daemon().await;
        let service = MemoryServiceImpl::new(daemon);

        let request = Request::new(RemoveDocumentRequest {
            document_id: "doc_to_remove".to_string(),
            collection_name: "test_collection".to_string(),
        });

        let result = service.remove_document(request).await;
        assert!(result.is_ok());

        let response = result.unwrap().into_inner();
        assert_eq!(response, ());
    }

    #[tokio::test]
    async fn test_get_document() {
        let daemon = create_test_daemon().await;
        let service = MemoryServiceImpl::new(daemon);

        let document_id = "test_document_123";
        let collection_name = "test_collection";

        let request = Request::new(GetDocumentRequest {
            document_id: document_id.to_string(),
            collection_name: collection_name.to_string(),
        });

        let result = service.get_document(request).await;
        assert!(result.is_ok());

        let response = result.unwrap().into_inner();
        assert_eq!(response.document_id, document_id);
        assert!(response.content.is_some());
        assert!(response.created_at.is_some());
        assert!(response.updated_at.is_some());

        let content = response.content.unwrap();
        assert_eq!(content.text, "Example document content");
    }

    #[tokio::test]
    async fn test_get_document_timestamps() {
        let daemon = create_test_daemon().await;
        let service = MemoryServiceImpl::new(daemon);

        let request = Request::new(GetDocumentRequest {
            document_id: "timestamp_doc".to_string(),
            collection_name: "test_collection".to_string(),
        });

        let result = service.get_document(request).await;
        assert!(result.is_ok());

        let response = result.unwrap().into_inner();
        assert!(response.created_at.is_some());
        assert!(response.updated_at.is_some());

        let created_timestamp = response.created_at.unwrap().seconds;
        let updated_timestamp = response.updated_at.unwrap().seconds;

        // Updated should be after created (or equal)
        assert!(updated_timestamp >= created_timestamp);
    }

    #[tokio::test]
    async fn test_list_documents() {
        let daemon = create_test_daemon().await;
        let service = MemoryServiceImpl::new(daemon);

        let request = Request::new(ListDocumentsRequest {
            collection_name: "test_collection".to_string(),
            limit: 10,
            offset: 0,
            filter: HashMap::new(),
        });

        let result = service.list_documents(request).await;
        assert!(result.is_ok());

        let response = result.unwrap().into_inner();
        assert_eq!(response.documents.len(), 1);
        assert_eq!(response.total_count, 1);
        assert!(!response.has_more);

        let document = &response.documents[0];
        assert!(!document.document_id.is_empty());
        assert_eq!(document.file_path, "/example/document.txt");
        assert_eq!(document.title, "Example Document");
        assert_eq!(document.file_size, 1024);
        assert!(document.created_at.is_some());
        assert!(document.updated_at.is_some());
    }

    #[tokio::test]
    async fn test_list_documents_pagination() {
        let daemon = create_test_daemon().await;
        let service = MemoryServiceImpl::new(daemon);

        let pagination_params = [
            (10, 0),   // First page
            (5, 5),    // Second page
            (1, 0),    // Single result
            (20, 10),  // Larger offset
        ];

        for (limit, offset) in pagination_params {
            let request = Request::new(ListDocumentsRequest {
                collection_name: "test_collection".to_string(),
                limit,
                offset,
                filter: HashMap::new(),
            });

            let result = service.list_documents(request).await;
            assert!(result.is_ok(), "Failed for limit={}, offset={}", limit, offset);

            let response = result.unwrap().into_inner();
            assert!(response.total_count >= 0);
        }
    }

    #[tokio::test]
    async fn test_create_collection() {
        let daemon = create_test_daemon().await;
        let service = MemoryServiceImpl::new(daemon);

        let request = Request::new(CreateCollectionRequest {
            collection_name: "new_collection".to_string(),
            project_id: "test_project".to_string(),
            config: Some(crate::proto::CollectionConfig {
                vector_size: 384,
                distance_metric: "Cosine".to_string(),
                enable_indexing: true,
                metadata_schema: HashMap::new(),
            }),
        });

        let result = service.create_collection(request).await;
        assert!(result.is_ok());

        let response = result.unwrap().into_inner();
        assert!(response.success);
        assert!(response.error_message.is_empty());
        assert!(!response.collection_id.is_empty());
    }

    #[tokio::test]
    async fn test_create_collection_different_configs() {
        let daemon = create_test_daemon().await;
        let service = MemoryServiceImpl::new(daemon);

        let configs = [
            (384, "Cosine", true),
            (512, "Euclidean", false),
            (768, "Dot", true),
            (256, "Cosine", false),
        ];

        for (i, (vector_size, distance_metric, enable_indexing)) in configs.iter().enumerate() {
            let request = Request::new(CreateCollectionRequest {
                collection_name: format!("collection_{}", i),
                project_id: "test_project".to_string(),
                config: Some(crate::proto::CollectionConfig {
                    vector_size: *vector_size,
                    distance_metric: distance_metric.to_string(),
                    enable_indexing: *enable_indexing,
                    metadata_schema: HashMap::new(),
                }),
            });

            let result = service.create_collection(request).await;
            assert!(result.is_ok(), "Failed for config: {:?}", (vector_size, distance_metric, enable_indexing));

            let response = result.unwrap().into_inner();
            assert!(response.success);
        }
    }

    #[tokio::test]
    async fn test_delete_collection() {
        let daemon = create_test_daemon().await;
        let service = MemoryServiceImpl::new(daemon);

        let request = Request::new(DeleteCollectionRequest {
            collection_name: "collection_to_delete".to_string(),
            project_id: "test_project".to_string(),
            force: false,
        });

        let result = service.delete_collection(request).await;
        assert!(result.is_ok());

        let response = result.unwrap().into_inner();
        assert_eq!(response, ());
    }

    #[tokio::test]
    async fn test_delete_collection_force() {
        let daemon = create_test_daemon().await;
        let service = MemoryServiceImpl::new(daemon);

        let request = Request::new(DeleteCollectionRequest {
            collection_name: "force_delete_collection".to_string(),
            project_id: "test_project".to_string(),
            force: true,
        });

        let result = service.delete_collection(request).await;
        assert!(result.is_ok());

        let response = result.unwrap().into_inner();
        assert_eq!(response, ());
    }

    #[tokio::test]
    async fn test_list_collections() {
        let daemon = create_test_daemon().await;
        let service = MemoryServiceImpl::new(daemon);

        let project_id = "test_project";
        let request = Request::new(ListCollectionsRequest {
            project_id: project_id.to_string(),
        });

        let result = service.list_collections(request).await;
        assert!(result.is_ok());

        let response = result.unwrap().into_inner();
        assert_eq!(response.collections.len(), 1);

        let collection = &response.collections[0];
        assert_eq!(collection.collection_name, "example-collection");
        assert!(!collection.collection_id.is_empty());
        assert_eq!(collection.project_id, project_id);
        assert_eq!(collection.document_count, 10);
        assert_eq!(collection.total_size_bytes, 10240);
        assert!(collection.config.is_some());
        assert!(collection.created_at.is_some());

        let config = collection.config.as_ref().unwrap();
        assert_eq!(config.vector_size, 384);
        assert_eq!(config.distance_metric, "Cosine");
        assert!(config.enable_indexing);
    }

    #[tokio::test]
    async fn test_list_collections_different_projects() {
        let daemon = create_test_daemon().await;
        let service = MemoryServiceImpl::new(daemon);

        let project_ids = [
            "project_1",
            "project_2",
            "my-awesome-project",
            "test_project_123",
        ];

        for project_id in project_ids {
            let request = Request::new(ListCollectionsRequest {
                project_id: project_id.to_string(),
            });

            let result = service.list_collections(request).await;
            assert!(result.is_ok(), "Failed for project: {}", project_id);

            let response = result.unwrap().into_inner();
            assert_eq!(response.collections.len(), 1);
            assert_eq!(response.collections[0].project_id, project_id);
        }
    }

    #[tokio::test]
    async fn test_concurrent_memory_operations() {
        let daemon = create_test_daemon().await;
        let service = Arc::new(MemoryServiceImpl::new(daemon));

        let mut handles = vec![];

        // Perform multiple concurrent operations
        for i in 0..5 {
            let service_clone = Arc::clone(&service);
            let handle = tokio::spawn(async move {
                // Add document
                let add_request = Request::new(AddDocumentRequest {
                    file_path: format!("/test/concurrent_doc_{}.txt", i),
                    collection_name: "test_collection".to_string(),
                    content: Some(DocumentContent {
                        text: format!("Concurrent document {}", i),
                        chunks: vec![],
                        extracted_metadata: HashMap::new(),
                    }),
                    metadata: HashMap::new(),
                    project_id: "test_project".to_string(),
                });

                let add_result = service_clone.add_document(add_request).await;

                // List documents
                let list_request = Request::new(ListDocumentsRequest {
                    collection_name: "test_collection".to_string(),
                    limit: 10,
                    offset: 0,
                    filter: HashMap::new(),
                });

                let list_result = service_clone.list_documents(list_request).await;

                (add_result, list_result)
            });
            handles.push(handle);
        }

        // Wait for all operations to complete
        let results: Vec<_> = futures_util::future::join_all(handles).await;

        // All operations should complete successfully
        for (i, result) in results.into_iter().enumerate() {
            let (add_result, list_result) = result.unwrap();
            assert!(add_result.is_ok(), "Add operation {} failed", i);
            assert!(list_result.is_ok(), "List operation {} failed", i);
        }
    }

    #[tokio::test]
    async fn test_document_unique_ids() {
        let daemon = create_test_daemon().await;
        let service = MemoryServiceImpl::new(daemon);

        let mut document_ids = std::collections::HashSet::new();

        // Add multiple documents and verify unique IDs
        for i in 0..5 {
            let request = Request::new(AddDocumentRequest {
                file_path: format!("/test/unique_doc_{}.txt", i),
                collection_name: "test_collection".to_string(),
                content: Some(DocumentContent {
                    text: format!("Unique document {}", i),
                    chunks: vec![],
                    extracted_metadata: HashMap::new(),
                }),
                metadata: HashMap::new(),
                project_id: "test_project".to_string(),
            });

            let result = service.add_document(request).await;
            assert!(result.is_ok());

            let response = result.unwrap().into_inner();
            assert!(document_ids.insert(response.document_id.clone()),
                    "Duplicate document ID generated: {}", response.document_id);
        }

        assert_eq!(document_ids.len(), 5);
    }

    #[tokio::test]
    async fn test_collection_unique_ids() {
        let daemon = create_test_daemon().await;
        let service = MemoryServiceImpl::new(daemon);

        let mut collection_ids = std::collections::HashSet::new();

        // Create multiple collections and verify unique IDs
        for i in 0..5 {
            let request = Request::new(CreateCollectionRequest {
                collection_name: format!("unique_collection_{}", i),
                project_id: "test_project".to_string(),
                config: Some(crate::proto::CollectionConfig {
                    vector_size: 384,
                    distance_metric: "Cosine".to_string(),
                    enable_indexing: true,
                    metadata_schema: HashMap::new(),
                }),
            });

            let result = service.create_collection(request).await;
            assert!(result.is_ok());

            let response = result.unwrap().into_inner();
            assert!(collection_ids.insert(response.collection_id.clone()),
                    "Duplicate collection ID generated: {}", response.collection_id);
        }

        assert_eq!(collection_ids.len(), 5);
    }

    #[test]
    fn test_memory_service_impl_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<MemoryServiceImpl>();
        assert_sync::<MemoryServiceImpl>();
    }
}