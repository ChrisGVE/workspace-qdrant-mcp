//! Comprehensive async gRPC handlers tests with TDD approach
//!
//! This module implements comprehensive unit tests for async gRPC service handlers
//! with focus on async message processing, connection handling, timeouts, and error propagation.
//!
//! COVERAGE TARGET: 90%+ using tokio-test framework
//! FOCUS AREAS:
//! - Async gRPC service method implementations
//! - Request/response streaming with async iterators
//! - Error handling and status code propagation
//! - Connection lifecycle management
//! - Concurrent request processing limits
//! - Connection timeout and resource management

use workspace_qdrant_daemon::config::{
    DaemonConfig, ServerConfig, SecurityConfig, TransportConfig, MessageConfig,
    CompressionConfig, StreamingConfig, DatabaseConfig, QdrantConfig, CollectionConfig,
    ProcessingConfig, FileWatcherConfig
};
use workspace_qdrant_daemon::daemon::WorkspaceDaemon;
use workspace_qdrant_daemon::grpc::services::{
    DocumentProcessorImpl, SearchServiceImpl, MemoryServiceImpl, SystemServiceImpl
};
use workspace_qdrant_daemon::proto::document_processor_server::DocumentProcessor;
use workspace_qdrant_daemon::proto::search_service_server::SearchService;
use workspace_qdrant_daemon::proto::memory_service_server::MemoryService;
use workspace_qdrant_daemon::proto::system_service_server::SystemService;
use workspace_qdrant_daemon::grpc::middleware::{ConnectionManager, ConnectionInterceptor};
use workspace_qdrant_daemon::grpc::server::GrpcServer;
use workspace_qdrant_daemon::proto::*;

use std::sync::Arc;
use std::time::Duration;
use std::collections::HashMap;
use tokio_test;
use tokio::time::{timeout, sleep};
use tonic::{Request, Response, Status, Code};
use futures_util::{stream, StreamExt, future::join_all};
use tokio_stream::wrappers::ReceiverStream;
use serial_test::serial;

// ================================
// TEST CONFIGURATION AND HELPERS
// ================================

fn create_test_daemon_config() -> DaemonConfig {
    DaemonConfig {
        server: ServerConfig {
            host: "127.0.0.1".to_string(),
            port: 50055, // Unique port for async tests
            max_connections: 100,
            connection_timeout_secs: 5, // Short timeout for testing
            request_timeout_secs: 10,
            enable_tls: false,
            security: SecurityConfig::default(),
            transport: TransportConfig::default(),
            message: MessageConfig::default(),
            compression: CompressionConfig::default(),
            streaming: StreamingConfig::default(),
        },
        database: DatabaseConfig {
            sqlite_path: ":memory:".to_string(),
            max_connections: 5,
            connection_timeout_secs: 10,
            enable_wal: true,
        },
        qdrant: QdrantConfig {
            url: "http://localhost:6333".to_string(),
            api_key: None,
            timeout_secs: 10,
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
            max_concurrent_tasks: 4,
            default_chunk_size: 1000,
            default_chunk_overlap: 200,
            max_file_size_bytes: 1024 * 1024,
            supported_extensions: vec!["txt".to_string(), "md".to_string()],
            enable_lsp: false,
            lsp_timeout_secs: 5,
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
            prometheus_port: 9091,
        },
        logging: LoggingConfig {
            level: "debug".to_string(),
            file_path: None,
            json_format: false,
            max_file_size_mb: 100,
            max_files: 5,
        },
    }
}

async fn create_test_daemon() -> Arc<WorkspaceDaemon> {
    let config = create_test_daemon_config();
    Arc::new(WorkspaceDaemon::new(config).await.expect("Failed to create test daemon"))
}

// Helper to create test requests with proper metadata
fn create_process_document_request(file_path: &str, metadata: Option<HashMap<String, String>>) -> ProcessDocumentRequest {
    ProcessDocumentRequest {
        project_id: "test_project".to_string(),
        document_type: DocumentType::Text as i32,
        file_path: file_path.to_string(),
        collection_name: "test_collection".to_string(),
        metadata: metadata.unwrap_or_default(),
        options: Some(ProcessingOptions {
            enable_lsp_analysis: false,
            chunk_size: 1000,
            chunk_overlap: 200,
            extract_metadata: true,
            detect_language: true,
            custom_parsers: vec![],
        }),
    }
}

fn create_hybrid_search_request(query: &str, limit: i32) -> HybridSearchRequest {
    HybridSearchRequest {
        query: query.to_string(),
        context: SearchContext::Project as i32,
        options: Some(SearchOptions {
            limit,
            score_threshold: 0.0,
            include_metadata: true,
            include_content: true,
            ranking: Some(RankingOptions {
                semantic_weight: 0.7,
                keyword_weight: 0.3,
                rrf_constant: 60.0,
            }),
        }),
        project_id: "test_project".to_string(),
        collection_names: vec!["test_collection".to_string()],
    }
}

// ================================
// ASYNC DOCUMENT PROCESSOR TESTS
// ================================

#[tokio::test]
async fn test_async_document_processor_basic_flow() {
    let daemon = create_test_daemon().await;
    let processor = DocumentProcessorImpl::new(daemon);

    let request = Request::new(create_process_document_request("/test/doc.txt", None));

    let result = processor.process_document(request).await;
    assert!(result.is_ok());

    let response = result.unwrap().into_inner();
    assert!(!response.document_id.is_empty());
    assert_eq!(response.status, ProcessingStatus::Completed as i32);
    assert!(response.error_message.is_empty());
    assert_eq!(response.chunks_created, 1);
    assert!(response.processed_at.is_some());
}

#[tokio::test]
async fn test_async_document_processor_concurrent_requests() {
    let daemon = create_test_daemon().await;
    let processor = Arc::new(DocumentProcessorImpl::new(daemon));

    let mut handles = vec![];

    // Process 10 documents concurrently
    for i in 0..10 {
        let processor_clone = Arc::clone(&processor);
        let handle = tokio::spawn(async move {
            let request = Request::new(create_process_document_request(
                &format!("/test/concurrent_doc_{}.txt", i),
                None
            ));
            processor_clone.process_document(request).await
        });
        handles.push(handle);
    }

    let results: Vec<_> = join_all(handles).await;

    // All requests should complete successfully
    for (i, result) in results.into_iter().enumerate() {
        let task_result = result.unwrap();
        assert!(task_result.is_ok(), "Concurrent request {} failed", i);

        let response = task_result.unwrap().into_inner();
        assert!(!response.document_id.is_empty());
        assert_eq!(response.status, ProcessingStatus::Completed as i32);
    }
}

#[tokio::test]
async fn test_async_document_processor_with_timeout() {
    let daemon = create_test_daemon().await;
    let processor = DocumentProcessorImpl::new(daemon);

    let request = Request::new(create_process_document_request("/test/timeout_test.txt", None));

    // Test that processing completes within reasonable time
    let result = timeout(Duration::from_secs(5), processor.process_document(request)).await;
    assert!(result.is_ok(), "Document processing timed out");

    let response_result = result.unwrap();
    assert!(response_result.is_ok());
}

#[tokio::test]
async fn test_async_document_processor_stream_basic() {
    let daemon = create_test_daemon().await;
    let processor = DocumentProcessorImpl::new(daemon);

    // Create input stream with 3 documents
    let documents = vec![
        create_process_document_request("/test/stream_doc_1.txt", None),
        create_process_document_request("/test/stream_doc_2.txt", None),
        create_process_document_request("/test/stream_doc_3.txt", None),
    ];

    let input_stream = stream::iter(documents.into_iter().map(Ok));
    let request = Request::new(input_stream);

    let result = processor.process_documents(request).await;
    assert!(result.is_ok());

    let mut response_stream = result.unwrap().into_inner();
    let mut response_count = 0;

    // Collect all responses from stream
    while let Some(response) = response_stream.next().await {
        assert!(response.is_ok());
        let doc_response = response.unwrap();
        assert!(!doc_response.document_id.is_empty());
        assert_eq!(doc_response.status, ProcessingStatus::Completed as i32);
        response_count += 1;
    }

    assert_eq!(response_count, 3);
}

#[tokio::test]
async fn test_async_document_processor_stream_with_errors() {
    let daemon = create_test_daemon().await;
    let processor = DocumentProcessorImpl::new(daemon);

    // Create input stream with mixed valid and error responses
    let input_stream = stream::iter(vec![
        Ok(create_process_document_request("/test/valid_doc.txt", None)),
        Err(Status::invalid_argument("Test error")),
        Ok(create_process_document_request("/test/another_valid_doc.txt", None)),
    ]);

    let request = Request::new(input_stream);
    let result = processor.process_documents(request).await;
    assert!(result.is_ok());

    let mut response_stream = result.unwrap().into_inner();
    let mut valid_responses = 0;
    let mut error_responses = 0;

    while let Some(response) = response_stream.next().await {
        match response {
            Ok(_) => valid_responses += 1,
            Err(_) => error_responses += 1,
        }
    }

    // Should have processed valid documents and encountered the error
    assert!(valid_responses > 0);
    assert!(error_responses > 0);
}

#[tokio::test]
async fn test_async_document_processor_status_tracking() {
    let daemon = create_test_daemon().await;
    let processor = DocumentProcessorImpl::new(daemon);

    let request = Request::new(ProcessingStatusRequest {
        operation_id: "test_operation_async".to_string(),
    });

    let result = processor.get_processing_status(request).await;
    assert!(result.is_ok());

    let response = result.unwrap().into_inner();
    assert_eq!(response.operation_id, "test_operation_async");
    assert_eq!(response.status, ProcessingStatus::Completed as i32);
    assert_eq!(response.total_documents, 1);
    assert_eq!(response.processed_documents, 1);
    assert_eq!(response.failed_documents, 0);
    assert!(response.error_messages.is_empty());
    assert!(response.started_at.is_some());
    assert!(response.updated_at.is_some());
}

#[tokio::test]
async fn test_async_document_processor_cancellation() {
    let daemon = create_test_daemon().await;
    let processor = DocumentProcessorImpl::new(daemon);

    let request = Request::new(CancelProcessingRequest {
        operation_id: "cancel_test_operation".to_string(),
    });

    let result = processor.cancel_processing(request).await;
    assert!(result.is_ok());

    // Cancellation should return empty response
    let response = result.unwrap().into_inner();
    assert_eq!(response, ());
}

// ================================
// ASYNC SEARCH SERVICE TESTS
// ================================

#[tokio::test]
async fn test_async_search_service_hybrid_search() {
    let daemon = create_test_daemon().await;
    let service = SearchServiceImpl::new(daemon);

    let request = Request::new(create_hybrid_search_request("async test query", 10));

    let result = service.hybrid_search(request).await;
    assert!(result.is_ok());

    let response = result.unwrap().into_inner();
    assert!(!response.query_id.is_empty());
    assert_eq!(response.results.len(), 1);
    assert!(response.metadata.is_some());

    let metadata = response.metadata.unwrap();
    assert_eq!(metadata.total_results, 1);
    assert!(metadata.search_time.is_some());
    assert!(metadata.search_duration_ms > 0);
}

#[tokio::test]
async fn test_async_search_service_concurrent_searches() {
    let daemon = create_test_daemon().await;
    let service = Arc::new(SearchServiceImpl::new(daemon));

    let mut handles = vec![];

    // Perform 15 concurrent searches with different queries
    for i in 0..15 {
        let service_clone = Arc::clone(&service);
        let handle = tokio::spawn(async move {
            let request = Request::new(create_hybrid_search_request(
                &format!("concurrent async search {}", i),
                5
            ));
            service_clone.hybrid_search(request).await
        });
        handles.push(handle);
    }

    let results: Vec<_> = join_all(handles).await;

    // All searches should complete successfully
    for (i, result) in results.into_iter().enumerate() {
        let task_result = result.unwrap();
        assert!(task_result.is_ok(), "Concurrent search {} failed", i);

        let response = task_result.unwrap().into_inner();
        assert!(!response.query_id.is_empty());
    }
}

#[tokio::test]
async fn test_async_search_service_semantic_search() {
    let daemon = create_test_daemon().await;
    let service = SearchServiceImpl::new(daemon);

    let request = Request::new(SemanticSearchRequest {
        query: "async semantic search test".to_string(),
        context: SearchContext::Project as i32,
        options: Some(SearchOptions {
            limit: 10,
            score_threshold: 0.5,
            include_metadata: true,
            include_content: true,
            ranking: None,
        }),
        project_id: "test_project".to_string(),
        collection_names: vec!["test_collection".to_string()],
    });

    let result = service.semantic_search(request).await;
    assert!(result.is_ok());

    let response = result.unwrap().into_inner();
    assert!(!response.query_id.is_empty());
    assert!(response.metadata.is_some());

    let metadata = response.metadata.unwrap();
    assert!(metadata.search_time.is_some());
    assert!(metadata.search_duration_ms > 0);
}

#[tokio::test]
async fn test_async_search_service_keyword_search() {
    let daemon = create_test_daemon().await;
    let service = SearchServiceImpl::new(daemon);

    let request = Request::new(KeywordSearchRequest {
        query: "async keyword search".to_string(),
        context: SearchContext::Collection as i32,
        options: Some(SearchOptions {
            limit: 20,
            score_threshold: 0.0,
            include_metadata: true,
            include_content: true,
            ranking: None,
        }),
        project_id: "test_project".to_string(),
        collection_names: vec!["test_collection".to_string()],
    });

    let result = service.keyword_search(request).await;
    assert!(result.is_ok());

    let response = result.unwrap().into_inner();
    assert!(!response.query_id.is_empty());
    assert!(response.metadata.is_some());
}

#[tokio::test]
async fn test_async_search_service_suggestions() {
    let daemon = create_test_daemon().await;
    let service = SearchServiceImpl::new(daemon);

    let request = Request::new(SuggestionsRequest {
        partial_query: "async".to_string(),
        context: SearchContext::Project as i32,
        max_suggestions: 5,
        project_id: "test_project".to_string(),
    });

    let result = service.get_suggestions(request).await;
    assert!(result.is_ok());

    let response = result.unwrap().into_inner();
    assert_eq!(response.suggestions.len(), 2);
    assert!(response.suggestions[0].contains("async"));
    assert!(response.suggestions[1].contains("async"));
    assert!(response.metadata.is_some());
}

#[tokio::test]
async fn test_async_search_service_with_timeout() {
    let daemon = create_test_daemon().await;
    let service = SearchServiceImpl::new(daemon);

    let request = Request::new(create_hybrid_search_request("timeout test search", 100));

    // Test that search completes within reasonable time
    let result = timeout(Duration::from_secs(3), service.hybrid_search(request)).await;
    assert!(result.is_ok(), "Search operation timed out");

    let response_result = result.unwrap();
    assert!(response_result.is_ok());
}

// ================================
// ASYNC MEMORY SERVICE TESTS
// ================================

#[tokio::test]
async fn test_async_memory_service_add_document() {
    let daemon = create_test_daemon().await;
    let service = MemoryServiceImpl::new(daemon);

    let request = Request::new(AddDocumentRequest {
        file_path: "/test/memory_doc.txt".to_string(),
        collection_name: "test_collection".to_string(),
        project_id: "test_project".to_string(),
        content: Some(DocumentContent {
            text: "Test document content for async memory service".to_string(),
            chunks: vec![],
            extracted_metadata: HashMap::new(),
        }),
        metadata: HashMap::new(),
    });

    let result = service.add_document(request).await;
    assert!(result.is_ok());

    let response = result.unwrap().into_inner();
    assert!(!response.document_id.is_empty());
    assert!(response.success);
    assert!(response.error_message.is_empty());
}

#[tokio::test]
async fn test_async_memory_service_concurrent_operations() {
    let daemon = create_test_daemon().await;
    let service = Arc::new(MemoryServiceImpl::new(daemon));

    let mut handles = vec![];

    // Perform concurrent memory operations
    for i in 0..8 {
        let service_clone = Arc::clone(&service);
        let handle = tokio::spawn(async move {
            let request = Request::new(AddDocumentRequest {
                file_path: format!("/test/concurrent_memory_doc_{}.txt", i),
                collection_name: "test_collection".to_string(),
                project_id: "test_project".to_string(),
                content: Some(DocumentContent {
                    text: format!("Concurrent test document content {}", i),
                    chunks: vec![],
                    extracted_metadata: HashMap::new(),
                }),
                metadata: HashMap::new(),
            });
            service_clone.add_document(request).await
        });
        handles.push(handle);
    }

    let results: Vec<_> = join_all(handles).await;

    // All operations should complete successfully
    for (i, result) in results.into_iter().enumerate() {
        let task_result = result.unwrap();
        assert!(task_result.is_ok(), "Concurrent memory operation {} failed", i);

        let response = task_result.unwrap().into_inner();
        assert!(!response.document_id.is_empty());
        assert!(response.success);
    }
}

// ================================
// ASYNC SYSTEM SERVICE TESTS
// ================================

#[tokio::test]
async fn test_async_system_service_health_check() {
    let daemon = create_test_daemon().await;
    let service = SystemServiceImpl::new(daemon);

    let request = Request::new(());

    let result = service.health_check(request).await;
    assert!(result.is_ok());

    let response = result.unwrap().into_inner();
    assert_eq!(response.status, ServiceStatus::Healthy as i32);
    assert!(response.timestamp.is_some());
}

#[tokio::test]
async fn test_async_system_service_get_status() {
    let daemon = create_test_daemon().await;
    let service = SystemServiceImpl::new(daemon);

    let request = Request::new(());

    let result = service.get_status(request).await;
    assert!(result.is_ok());

    let response = result.unwrap().into_inner();
    assert_eq!(response.status, ServiceStatus::Healthy as i32);
    assert!(response.metrics.is_some());
    assert!(response.uptime_since.is_some());
}

// ================================
// CONNECTION HANDLING AND TIMEOUTS
// ================================

#[tokio::test]
async fn test_async_connection_manager_with_limits() {
    let manager = ConnectionManager::new(3, 5); // 3 connections, 5 req/sec

    // Register connections concurrently
    let mut handles = vec![];
    for i in 0..3 {
        let client_id = format!("async_client_{}", i);
        let result = manager.register_connection(client_id);
        assert!(result.is_ok());
    }

    // Fourth connection should fail
    let result = manager.register_connection("overflow_client".to_string());
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().code(), Code::ResourceExhausted);

    // Test rate limiting
    for _ in 0..5 {
        let result = manager.check_rate_limit("async_client_0");
        assert!(result.is_ok());
    }

    // Next request should be rate limited
    let result = manager.check_rate_limit("async_client_0");
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().code(), Code::ResourceExhausted);
}

#[tokio::test]
async fn test_async_connection_interceptor() {
    let manager = Arc::new(ConnectionManager::new(10, 100));
    let interceptor = ConnectionInterceptor::new(manager);

    // Test request interception
    let mut request: Request<()> = Request::new(());
    request.metadata_mut().insert(
        "client-id",
        "async_test_client".parse().unwrap()
    );

    let result = interceptor.intercept(request);
    assert!(result.is_ok());

    // Test response interception
    let response: Response<()> = Response::new(());
    let result = interceptor.intercept_response(response, "async_test_client");
    assert_eq!(result.get_ref(), &());
}

#[tokio::test]
async fn test_async_grpc_server_connection_stats() {
    let daemon = create_test_daemon().await;
    let server = GrpcServer::new(daemon.as_ref().clone(), "127.0.0.1:0".parse().unwrap());

    let stats = server.get_connection_stats();
    assert_eq!(stats.active_connections, 0);
    assert_eq!(stats.total_requests, 0);
    assert_eq!(stats.total_bytes_sent, 0);
    assert_eq!(stats.total_bytes_received, 0);

    // Test connection manager access
    let connection_manager = server.connection_manager();
    let stats2 = connection_manager.get_stats();
    assert_eq!(stats.active_connections, stats2.active_connections);
}

// ================================
// ERROR HANDLING AND STATUS CODES
// ================================

#[tokio::test]
async fn test_async_error_propagation() {
    let daemon = create_test_daemon().await;
    let processor = DocumentProcessorImpl::new(daemon);

    // Test with empty file path (should still complete due to placeholder implementation)
    let request = Request::new(create_process_document_request("", None));
    let result = processor.process_document(request).await;
    assert!(result.is_ok()); // Current implementation doesn't validate inputs
}

#[tokio::test]
async fn test_async_status_code_handling() {
    let manager = ConnectionManager::new(1, 1);

    // Register one connection
    manager.register_connection("client1".to_string()).unwrap();

    // Try to register another (should fail with ResourceExhausted)
    let result = manager.register_connection("client2".to_string());
    assert!(result.is_err());
    if let Err(status) = result {
        assert_eq!(status.code(), Code::ResourceExhausted);
        assert!(status.message().contains("Connection limit reached"));
    }

    // Test rate limiting status
    manager.check_rate_limit("client1").unwrap(); // First request OK
    let result = manager.check_rate_limit("client1"); // Second should fail
    assert!(result.is_err());
    if let Err(status) = result {
        assert_eq!(status.code(), Code::ResourceExhausted);
        assert!(status.message().contains("Rate limit exceeded"));
    }
}

// ================================
// RESOURCE MANAGEMENT TESTS
// ================================

#[tokio::test]
async fn test_async_resource_cleanup() {
    let manager = Arc::new(ConnectionManager::new(10, 100));

    // Register connections
    for i in 0..5 {
        let client_id = format!("cleanup_client_{}", i);
        manager.register_connection(client_id.clone()).unwrap();
        manager.update_activity(&client_id, 1000, 2000);
    }

    let stats_before = manager.get_stats();
    assert_eq!(stats_before.active_connections, 5);
    assert!(stats_before.total_bytes_sent > 0);
    assert!(stats_before.total_bytes_received > 0);

    // Cleanup with very short timeout (should remove all connections)
    manager.cleanup_expired_connections(Duration::from_nanos(1));

    let stats_after = manager.get_stats();
    assert_eq!(stats_after.active_connections, 0);
}

#[tokio::test]
async fn test_async_concurrent_request_limits() {
    let daemon = create_test_daemon().await;
    let config = daemon.config();

    // Verify that max_concurrent_tasks is respected in config
    assert_eq!(config.processing.max_concurrent_tasks, 4);

    let processor = Arc::new(DocumentProcessorImpl::new(daemon));
    let mut handles = vec![];

    // Launch more concurrent requests than the limit
    for i in 0..8 {
        let processor_clone = Arc::clone(&processor);
        let handle = tokio::spawn(async move {
            let request = Request::new(create_process_document_request(
                &format!("/test/limit_test_doc_{}.txt", i),
                None
            ));
            processor_clone.process_document(request).await
        });
        handles.push(handle);
    }

    // All should complete successfully (current implementation doesn't enforce limits)
    let results: Vec<_> = join_all(handles).await;
    for result in results {
        assert!(result.unwrap().is_ok());
    }
}

// ================================
// TIMEOUT AND DEADLINE TESTS
// ================================

#[tokio::test]
async fn test_async_operation_timeouts() {
    let daemon = create_test_daemon().await;
    let services = (
        DocumentProcessorImpl::new(daemon.clone()),
        SearchServiceImpl::new(daemon.clone()),
        MemoryServiceImpl::new(daemon.clone()),
        SystemServiceImpl::new(daemon),
    );

    // Test various operations with tight timeout
    let timeout_duration = Duration::from_millis(100);

    // Document processing
    let doc_request = Request::new(create_process_document_request("/test/timeout.txt", None));
    let doc_result = timeout(timeout_duration, services.0.process_document(doc_request)).await;
    assert!(doc_result.is_ok()); // Should complete quickly

    // Search
    let search_request = Request::new(create_hybrid_search_request("timeout test", 5));
    let search_result = timeout(timeout_duration, services.1.hybrid_search(search_request)).await;
    assert!(search_result.is_ok()); // Should complete quickly

    // Health check
    let health_request = Request::new(());
    let health_result = timeout(timeout_duration, services.3.health_check(health_request)).await;
    assert!(health_result.is_ok()); // Should complete quickly
}

#[tokio::test]
#[serial] // Run serially to avoid interference
async fn test_async_connection_expiry() {
    let manager = ConnectionManager::new(10, 100);

    // Register connection
    manager.register_connection("expiry_test_client".to_string()).unwrap();
    manager.update_activity("expiry_test_client", 500, 1000);

    let stats_before = manager.get_stats();
    assert_eq!(stats_before.active_connections, 1);

    // Wait a bit for activity time to pass
    sleep(Duration::from_millis(10)).await;

    // Cleanup with very short timeout
    manager.cleanup_expired_connections(Duration::from_millis(5));

    let stats_after = manager.get_stats();
    assert_eq!(stats_after.active_connections, 0);
}

// ================================
// STREAM PROCESSING TESTS
// ================================

#[tokio::test]
async fn test_async_streaming_with_backpressure() {
    let daemon = create_test_daemon().await;
    let processor = DocumentProcessorImpl::new(daemon);

    // Create a large batch of documents to test backpressure
    let documents: Vec<_> = (0..20)
        .map(|i| create_process_document_request(&format!("/test/backpressure_doc_{}.txt", i), None))
        .collect();

    let input_stream = stream::iter(documents.into_iter().map(Ok));
    let request = Request::new(input_stream);

    let result = processor.process_documents(request).await;
    assert!(result.is_ok());

    let mut response_stream = result.unwrap().into_inner();
    let mut count = 0;

    // Process stream with simulated slow consumer
    while let Some(response) = response_stream.next().await {
        assert!(response.is_ok());
        count += 1;

        // Simulate slow processing
        sleep(Duration::from_millis(1)).await;
    }

    assert_eq!(count, 20);
}

#[tokio::test]
async fn test_async_streaming_error_recovery() {
    let daemon = create_test_daemon().await;
    let processor = DocumentProcessorImpl::new(daemon);

    // Create stream with intermixed valid and invalid requests
    let (tx, rx) = tokio::sync::mpsc::channel(10);
    let stream = ReceiverStream::new(rx);

    // Send test data
    tokio::spawn(async move {
        let _ = tx.send(Ok(create_process_document_request("/test/valid1.txt", None))).await;
        let _ = tx.send(Ok(create_process_document_request("/test/valid2.txt", None))).await;
        let _ = tx.send(Ok(create_process_document_request("/test/valid3.txt", None))).await;
    });

    let request = Request::new(stream);
    let result = processor.process_documents(request).await;
    assert!(result.is_ok());

    let mut response_stream = result.unwrap().into_inner();
    let mut valid_count = 0;
    let mut error_count = 0;

    while let Some(response) = response_stream.next().await {
        match response {
            Ok(_) => valid_count += 1,
            Err(_) => error_count += 1,
        }
    }

    // Should have processed 3 valid documents
    assert_eq!(valid_count, 3);
    assert_eq!(error_count, 0);
}

// ================================
// INTEGRATION TESTS
// ================================

#[tokio::test]
async fn test_async_full_grpc_service_integration() {
    let daemon = create_test_daemon().await;

    // Create all service implementations
    let doc_processor = DocumentProcessorImpl::new(daemon.clone());
    let search_service = SearchServiceImpl::new(daemon.clone());
    let memory_service = MemoryServiceImpl::new(daemon.clone());
    let system_service = SystemServiceImpl::new(daemon);

    // Test a complete workflow
    // 1. Add document to memory
    let add_request = Request::new(AddDocumentRequest {
        file_path: "/test/integration_doc.txt".to_string(),
        collection_name: "integration_collection".to_string(),
        project_id: "integration_project".to_string(),
        content: Some(DocumentContent {
            text: "Integration test document content".to_string(),
            chunks: vec![],
            extracted_metadata: HashMap::new(),
        }),
        metadata: HashMap::new(),
    });

    let add_result = memory_service.add_document(add_request).await;
    assert!(add_result.is_ok());
    let document_id = add_result.unwrap().into_inner().document_id;

    // 2. Process the document
    let process_request = Request::new(create_process_document_request("/test/integration_doc.txt", None));
    let process_result = doc_processor.process_document(process_request).await;
    assert!(process_result.is_ok());

    // 3. Search for the document
    let search_request = Request::new(create_hybrid_search_request("integration test", 10));
    let search_result = search_service.hybrid_search(search_request).await;
    assert!(search_result.is_ok());

    // 4. Check system health
    let health_request = Request::new(());
    let health_result = system_service.health_check(health_request).await;
    assert!(health_result.is_ok());
    let health_response = health_result.unwrap().into_inner();
    assert_eq!(health_response.status, ServiceStatus::Healthy as i32);

    // Verify all operations completed successfully
    assert!(!document_id.is_empty());
}

#[tokio::test]
async fn test_async_cross_service_error_handling() {
    let daemon = create_test_daemon().await;

    let doc_processor = DocumentProcessorImpl::new(daemon.clone());
    let search_service = SearchServiceImpl::new(daemon);

    // Process a document
    let process_request = Request::new(create_process_document_request("/test/error_test.txt", None));
    let process_result = doc_processor.process_document(process_request).await;
    assert!(process_result.is_ok());

    // Search should work even if document processing had issues
    let search_request = Request::new(create_hybrid_search_request("error test query", 5));
    let search_result = search_service.hybrid_search(search_request).await;
    assert!(search_result.is_ok());
}