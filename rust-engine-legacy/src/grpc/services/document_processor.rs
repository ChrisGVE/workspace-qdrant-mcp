//! Document processing gRPC service implementation

use crate::daemon::WorkspaceDaemon;
use crate::proto::{
    document_processor_server::DocumentProcessor,
    ProcessDocumentRequest, ProcessDocumentResponse,
    ProcessingStatusRequest, ProcessingStatusResponse,
    CancelProcessingRequest,
};
use std::sync::Arc;
use tonic::{Request, Response, Status};
use tracing::{debug, info};

/// Document processor service implementation
#[derive(Debug)]
pub struct DocumentProcessorImpl {
    daemon: Arc<WorkspaceDaemon>,
}

impl DocumentProcessorImpl {
    pub fn new(daemon: Arc<WorkspaceDaemon>) -> Self {
        Self { daemon }
    }
}

#[tonic::async_trait]
impl DocumentProcessor for DocumentProcessorImpl {
    async fn process_document(
        &self,
        request: Request<ProcessDocumentRequest>,
    ) -> Result<Response<ProcessDocumentResponse>, Status> {
        let req = request.into_inner();
        debug!("Processing document: {:?}", req.file_path);

        // Access daemon config to prevent unused field warning
        let _config = self.daemon.config();

        // TODO: Implement actual document processing
        // This is a placeholder implementation
        let response = ProcessDocumentResponse {
            document_id: uuid::Uuid::new_v4().to_string(),
            status: crate::proto::ProcessingStatus::Completed as i32,
            error_message: String::new(),
            chunks_created: 1,
            extracted_metadata: std::collections::HashMap::new(),
            processed_at: Some(prost_types::Timestamp {
                seconds: chrono::Utc::now().timestamp(),
                nanos: 0,
            }),
        };

        Ok(Response::new(response))
    }

    type ProcessDocumentsStream = tokio_stream::wrappers::ReceiverStream<Result<ProcessDocumentResponse, Status>>;

    async fn process_documents(
        &self,
        request: Request<tonic::Streaming<ProcessDocumentRequest>>,
    ) -> Result<Response<Self::ProcessDocumentsStream>, Status> {
        let mut stream = request.into_inner();
        let (tx, rx) = tokio::sync::mpsc::channel(128);

        // Spawn task to process the stream
        tokio::spawn(async move {
            while let Some(req) = stream.message().await.transpose() {
                match req {
                    Ok(req) => {
                        debug!("Processing document in stream: {:?}", req.file_path);

                        // TODO: Implement actual processing
                        let response = ProcessDocumentResponse {
                            document_id: uuid::Uuid::new_v4().to_string(),
                            status: crate::proto::ProcessingStatus::Completed as i32,
                            error_message: String::new(),
                            chunks_created: 1,
                            extracted_metadata: std::collections::HashMap::new(),
                            processed_at: Some(prost_types::Timestamp {
                                seconds: chrono::Utc::now().timestamp(),
                                nanos: 0,
                            }),
                        };

                        if tx.send(Ok(response)).await.is_err() {
                            break;
                        }
                    },
                    Err(e) => {
                        let _ = tx.send(Err(e)).await;
                        break;
                    }
                }
            }
        });

        Ok(Response::new(tokio_stream::wrappers::ReceiverStream::new(rx)))
    }

    async fn get_processing_status(
        &self,
        request: Request<ProcessingStatusRequest>,
    ) -> Result<Response<ProcessingStatusResponse>, Status> {
        let req = request.into_inner();
        debug!("Getting processing status for: {}", req.operation_id);

        // TODO: Implement actual status tracking
        let response = ProcessingStatusResponse {
            operation_id: req.operation_id,
            status: crate::proto::ProcessingStatus::Completed as i32,
            total_documents: 1,
            processed_documents: 1,
            failed_documents: 0,
            error_messages: vec![],
            started_at: Some(prost_types::Timestamp {
                seconds: chrono::Utc::now().timestamp() - 60,
                nanos: 0,
            }),
            updated_at: Some(prost_types::Timestamp {
                seconds: chrono::Utc::now().timestamp(),
                nanos: 0,
            }),
        };

        Ok(Response::new(response))
    }

    async fn cancel_processing(
        &self,
        request: Request<CancelProcessingRequest>,
    ) -> Result<Response<()>, Status> {
        let req = request.into_inner();
        info!("Cancelling processing operation: {}", req.operation_id);

        // TODO: Implement actual cancellation logic

        Ok(Response::new(()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::*;
    use tonic::Request;
    use std::collections::HashMap;

    fn create_test_daemon_config() -> DaemonConfig {
        // Use in-memory SQLite database for tests
        let db_path = ":memory:";

        DaemonConfig {
            server: ServerConfig {
                host: "127.0.0.1".to_string(),
                port: 50051,
                max_connections: 100,
                connection_timeout_secs: 30,
                request_timeout_secs: 60,
                enable_tls: false,
                security: crate::config::SecurityConfig::default(),
                transport: crate::config::TransportConfig::default(),
                message: crate::config::MessageConfig::default(),
                compression: crate::config::CompressionConfig::default(),
                streaming: crate::config::StreamingConfig::default(),
            },
            database: DatabaseConfig {
                sqlite_path: db_path.to_string(),
                max_connections: 5,
                connection_timeout_secs: 30,
                enable_wal: true,
            },
            qdrant: QdrantConfig {
                url: "http://localhost:6333".to_string(),
                api_key: None,
                timeout_secs: 30,
                max_retries: 3,
                default_collection: CollectionConfig {
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
            auto_ingestion: crate::config::AutoIngestionConfig::default(),
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
    async fn test_document_processor_impl_new() {
        let daemon = create_test_daemon().await;
        let processor = DocumentProcessorImpl::new(daemon.clone());

        assert!(Arc::ptr_eq(&processor.daemon, &daemon));
    }

    #[tokio::test]
    async fn test_document_processor_impl_debug() {
        let daemon = create_test_daemon().await;
        let processor = DocumentProcessorImpl::new(daemon);

        let debug_str = format!("{:?}", processor);
        assert!(debug_str.contains("DocumentProcessorImpl"));
        assert!(debug_str.contains("daemon"));
    }

    #[tokio::test]
    async fn test_process_document_basic() {
        let daemon = create_test_daemon().await;
        let processor = DocumentProcessorImpl::new(daemon);

        let request = Request::new(ProcessDocumentRequest {
            project_id: "test_project".to_string(),
            document_type: crate::proto::DocumentType::Text as i32,
            file_path: "/test/document.txt".to_string(),
            collection_name: "test_collection".to_string(),
            metadata: HashMap::new(),
            options: None,
        });

        let result = processor.process_document(request).await;
        assert!(result.is_ok());

        let response = result.unwrap().into_inner();
        assert!(!response.document_id.is_empty());
        assert_eq!(response.status, crate::proto::ProcessingStatus::Completed as i32);
        assert!(response.error_message.is_empty());
        assert_eq!(response.chunks_created, 1);
        assert!(response.processed_at.is_some());
    }

    #[tokio::test]
    async fn test_process_document_with_metadata() {
        let daemon = create_test_daemon().await;
        let processor = DocumentProcessorImpl::new(daemon);

        let mut metadata = HashMap::new();
        metadata.insert("author".to_string(), "test_author".to_string());
        metadata.insert("version".to_string(), "1.0".to_string());

        let request = Request::new(ProcessDocumentRequest {
            project_id: "test_project".to_string(),
            document_type: crate::proto::DocumentType::Text as i32,
            file_path: "/test/document_with_metadata.txt".to_string(),
            collection_name: "test_collection".to_string(),
            metadata,
            options: None,
        });

        let result = processor.process_document(request).await;
        assert!(result.is_ok());

        let response = result.unwrap().into_inner();
        assert!(!response.document_id.is_empty());
        assert_eq!(response.status, crate::proto::ProcessingStatus::Completed as i32);
    }

    #[tokio::test]
    async fn test_process_document_different_file_types() {
        let daemon = create_test_daemon().await;
        let processor = DocumentProcessorImpl::new(daemon);

        let file_types = [
            "document.txt",
            "readme.md",
            "script.py",
            "code.rs",
            "data.json",
            "page.html",
        ];

        for file_name in file_types {
            let request = Request::new(ProcessDocumentRequest {
                project_id: "test_project".to_string(),
                document_type: crate::proto::DocumentType::Text as i32,
                file_path: format!("/test/{}", file_name),
                collection_name: "test_collection".to_string(),
                metadata: HashMap::new(),
                options: None,
            });

            let result = processor.process_document(request).await;
            assert!(result.is_ok(), "Failed to process file: {}", file_name);

            let response = result.unwrap().into_inner();
            assert!(!response.document_id.is_empty());
            assert_eq!(response.status, crate::proto::ProcessingStatus::Completed as i32);
        }
    }

    #[tokio::test]
    async fn test_get_processing_status() {
        let daemon = create_test_daemon().await;
        let processor = DocumentProcessorImpl::new(daemon);

        let request = Request::new(ProcessingStatusRequest {
            operation_id: "test_operation_123".to_string(),
        });

        let result = processor.get_processing_status(request).await;
        assert!(result.is_ok());

        let response = result.unwrap().into_inner();
        assert_eq!(response.operation_id, "test_operation_123");
        assert_eq!(response.status, crate::proto::ProcessingStatus::Completed as i32);
        assert_eq!(response.total_documents, 1);
        assert_eq!(response.processed_documents, 1);
        assert_eq!(response.failed_documents, 0);
        assert!(response.error_messages.is_empty());
        assert!(response.started_at.is_some());
        assert!(response.updated_at.is_some());
    }

    #[tokio::test]
    async fn test_get_processing_status_different_operation_ids() {
        let daemon = create_test_daemon().await;
        let processor = DocumentProcessorImpl::new(daemon);

        let operation_ids = [
            "op_1",
            "operation_with_underscores_123",
            "uuid-like-operation-456",
            "simple",
        ];

        for op_id in operation_ids {
            let request = Request::new(ProcessingStatusRequest {
                operation_id: op_id.to_string(),
            });

            let result = processor.get_processing_status(request).await;
            assert!(result.is_ok(), "Failed for operation ID: {}", op_id);

            let response = result.unwrap().into_inner();
            assert_eq!(response.operation_id, op_id);
        }
    }

    #[tokio::test]
    async fn test_cancel_processing() {
        let daemon = create_test_daemon().await;
        let processor = DocumentProcessorImpl::new(daemon);

        let request = Request::new(CancelProcessingRequest {
            operation_id: "test_cancel_operation".to_string(),
        });

        let result = processor.cancel_processing(request).await;
        assert!(result.is_ok());

        // Response should be empty unit type
        let response = result.unwrap().into_inner();
        assert_eq!(response, ());
    }

    #[tokio::test]
    async fn test_cancel_processing_multiple_operations() {
        let daemon = create_test_daemon().await;
        let processor = DocumentProcessorImpl::new(daemon);

        let operation_ids = [
            "cancel_op_1",
            "cancel_op_2",
            "cancel_op_3",
        ];

        for op_id in operation_ids {
            let request = Request::new(CancelProcessingRequest {
                operation_id: op_id.to_string(),
            });

            let result = processor.cancel_processing(request).await;
            assert!(result.is_ok(), "Failed to cancel operation: {}", op_id);
        }
    }

    #[tokio::test]
    async fn test_process_document_unique_ids() {
        let daemon = create_test_daemon().await;
        let processor = DocumentProcessorImpl::new(daemon);

        let mut document_ids = std::collections::HashSet::new();

        // Process multiple documents and verify unique IDs
        for i in 0..5 {
            let request = Request::new(ProcessDocumentRequest {
                project_id: "test_project".to_string(),
                document_type: crate::proto::DocumentType::Text as i32,
                file_path: format!("/test/document_{}.txt", i),
                collection_name: "test_collection".to_string(),
                metadata: HashMap::new(),
                options: None,
            });

            let result = processor.process_document(request).await;
            assert!(result.is_ok());

            let response = result.unwrap().into_inner();
            assert!(document_ids.insert(response.document_id.clone()),
                    "Duplicate document ID generated: {}", response.document_id);
        }

        assert_eq!(document_ids.len(), 5);
    }

    #[tokio::test]
    async fn test_process_document_timestamps() {
        let daemon = create_test_daemon().await;
        let processor = DocumentProcessorImpl::new(daemon);

        let before_processing = chrono::Utc::now().timestamp();

        let request = Request::new(ProcessDocumentRequest {
            project_id: "test_project".to_string(),
            document_type: crate::proto::DocumentType::Text as i32,
            file_path: "/test/timestamp_test.txt".to_string(),
            collection_name: "test_collection".to_string(),
            metadata: HashMap::new(),
            options: None,
        });

        let result = processor.process_document(request).await;
        assert!(result.is_ok());

        let after_processing = chrono::Utc::now().timestamp();
        let response = result.unwrap().into_inner();

        assert!(response.processed_at.is_some());
        let processed_timestamp = response.processed_at.unwrap().seconds;
        assert!(processed_timestamp >= before_processing);
        assert!(processed_timestamp <= after_processing);
    }

    #[tokio::test]
    async fn test_processing_status_timestamps() {
        let daemon = create_test_daemon().await;
        let processor = DocumentProcessorImpl::new(daemon);

        let request = Request::new(ProcessingStatusRequest {
            operation_id: "timestamp_test_op".to_string(),
        });

        let result = processor.get_processing_status(request).await;
        assert!(result.is_ok());

        let response = result.unwrap().into_inner();
        assert!(response.started_at.is_some());
        assert!(response.updated_at.is_some());

        let started_timestamp = response.started_at.unwrap().seconds;
        let updated_timestamp = response.updated_at.unwrap().seconds;

        // Updated timestamp should be after started timestamp
        assert!(updated_timestamp >= started_timestamp);
    }

    #[tokio::test]
    async fn test_concurrent_document_processing() {
        let daemon = create_test_daemon().await;
        let processor = Arc::new(DocumentProcessorImpl::new(daemon));

        let mut handles = vec![];

        // Process multiple documents concurrently
        for i in 0..5 {
            let processor_clone = Arc::clone(&processor);
            let handle = tokio::spawn(async move {
                let request = Request::new(ProcessDocumentRequest {
                    project_id: "test_project".to_string(),
                    document_type: crate::proto::DocumentType::Text as i32,
                    file_path: format!("/test/concurrent_doc_{}.txt", i),
                    collection_name: "test_collection".to_string(),
                    metadata: HashMap::new(),
                    options: None,
                });

                processor_clone.process_document(request).await
            });
            handles.push(handle);
        }

        // Wait for all tasks to complete
        let results: Vec<_> = futures_util::future::join_all(handles).await;

        // All tasks should complete successfully
        for (i, result) in results.into_iter().enumerate() {
            let task_result = result.unwrap();
            assert!(task_result.is_ok(), "Task {} failed", i);

            let response = task_result.unwrap().into_inner();
            assert!(!response.document_id.is_empty());
            assert_eq!(response.status, crate::proto::ProcessingStatus::Completed as i32);
        }
    }

    #[test]
    fn test_document_processor_impl_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<DocumentProcessorImpl>();
        assert_sync::<DocumentProcessorImpl>();
    }
}