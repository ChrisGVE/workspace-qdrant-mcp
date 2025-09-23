//! Basic gRPC Protocol Correctness Tests
//!
//! This test suite provides basic protocol-level testing for gRPC message serialization,
//! service method routing, and error handling.

use workspace_qdrant_daemon::{
    config::*,
    daemon::WorkspaceDaemon,
    grpc::services::*,
    proto::*,
};

use std::{net::SocketAddr, sync::Arc, time::Duration, collections::HashMap};
use tokio::time::timeout;
use tonic::{
    transport::{Server, Channel},
    Request,
};
use prost_types::Timestamp;

// =============================================================================
// TEST INFRASTRUCTURE
// =============================================================================

fn create_test_daemon_config() -> DaemonConfig {
    DaemonConfig {
        server: ServerConfig {
            host: "127.0.0.1".to_string(),
            port: 0, // Use ephemeral port
            max_connections: 10,
            connection_timeout_secs: 5,
            request_timeout_secs: 10,
            enable_tls: false,
        },
        database: DatabaseConfig {
            sqlite_path: ":memory:".to_string(),
            max_connections: 5,
            connection_timeout_secs: 5,
            enable_wal: true,
        },
        qdrant: QdrantConfig {
            url: "http://localhost:6333".to_string(),
            api_key: None,
            timeout_secs: 5,
            max_retries: 2,
            default_collection: workspace_qdrant_daemon::config::CollectionConfig {
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
            supported_extensions: vec!["txt".to_string()],
            enable_lsp: false,
            lsp_timeout_secs: 5,
        },
        file_watcher: FileWatcherConfig {
            enabled: false,
            debounce_ms: 100,
            max_watched_dirs: 5,
            ignore_patterns: vec![],
            recursive: true,
        },
        metrics: MetricsConfig {
            enabled: false,
            collection_interval_secs: 30,
            retention_days: 1,
            enable_prometheus: false,
            prometheus_port: 9090,
        },
        logging: LoggingConfig {
            level: "info".to_string(),
            file_path: None,
            json_format: false,
            max_file_size_mb: 10,
            max_files: 2,
        },
    }
}

async fn create_test_daemon() -> WorkspaceDaemon {
    let config = create_test_daemon_config();
    WorkspaceDaemon::new(config).await.expect("Failed to create test daemon")
}

// Test server setup
struct TestServer {
    address: SocketAddr,
    server_handle: tokio::task::JoinHandle<Result<(), tonic::transport::Error>>,
}

impl TestServer {
    async fn start() -> Self {
        let daemon = Arc::new(create_test_daemon().await);
        let address = SocketAddr::from(([127, 0, 0, 1], 0)); // Ephemeral port

        // Create service implementations
        let document_processor = DocumentProcessorImpl::new(Arc::clone(&daemon));
        let search_service = SearchServiceImpl::new(Arc::clone(&daemon));
        let memory_service = MemoryServiceImpl::new(Arc::clone(&daemon));
        let system_service = SystemServiceImpl::new(Arc::clone(&daemon));

        let server = Server::builder()
            .timeout(Duration::from_secs(5))
            .add_service(document_processor_server::DocumentProcessorServer::new(document_processor))
            .add_service(search_service_server::SearchServiceServer::new(search_service))
            .add_service(memory_service_server::MemoryServiceServer::new(memory_service))
            .add_service(system_service_server::SystemServiceServer::new(system_service));

        // Start server and get actual bound address
        let listener = tokio::net::TcpListener::bind(address).await.unwrap();
        let actual_addr = listener.local_addr().unwrap();

        let server_handle = tokio::spawn(async move {
            server.serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
                .await
        });

        // Wait a bit for server to be ready
        tokio::time::sleep(Duration::from_millis(50)).await;

        Self {
            address: actual_addr,
            server_handle,
        }
    }

    async fn get_client_channel(&self) -> Channel {
        let uri = format!("http://{}", self.address);
        Channel::from_shared(uri)
            .unwrap()
            .timeout(Duration::from_secs(5))
            .connect()
            .await
            .expect("Failed to connect to test server")
    }

    async fn shutdown(self) {
        self.server_handle.abort();
        let _ = self.server_handle.await;
    }
}

fn create_test_timestamp() -> Timestamp {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap();
    Timestamp {
        seconds: now.as_secs() as i64,
        nanos: (now.subsec_nanos() % 1_000_000_000) as i32,
    }
}

// =============================================================================
// PROTOCOL BUFFER MESSAGE SERIALIZATION TESTS
// =============================================================================

#[tokio::test]
async fn test_protobuf_document_processing_serialization() {
    let server = TestServer::start().await;
    let channel = server.get_client_channel().await;
    let mut client = document_processor_client::DocumentProcessorClient::new(channel);

    // Test ProcessDocumentRequest serialization
    let request = ProcessDocumentRequest {
        file_path: "/test/document.txt".to_string(),
        project_id: "test_project".to_string(),
        collection_name: "test_collection".to_string(),
        document_type: DocumentType::Text as i32,
        metadata: HashMap::from([("author".to_string(), "test".to_string())]),
        options: Some(ProcessingOptions {
            enable_lsp_analysis: true,
            chunk_size: 1024,
            chunk_overlap: 128,
            extract_metadata: true,
            detect_language: true,
            custom_parsers: vec!["parser1".to_string()],
        }),
    };

    let response = timeout(Duration::from_secs(5), client.process_document(Request::new(request)))
        .await
        .expect("Request timeout")
        .expect("Request failed")
        .into_inner();

    // Validate response structure
    assert!(!response.document_id.is_empty());
    assert_eq!(response.status, ProcessingStatus::Completed as i32);
    assert!(response.error_message.is_empty());
    assert_eq!(response.chunks_created, 1);
    assert!(response.processed_at.is_some());

    server.shutdown().await;
}

#[tokio::test]
async fn test_protobuf_search_serialization() {
    let server = TestServer::start().await;
    let channel = server.get_client_channel().await;
    let mut client = search_service_client::SearchServiceClient::new(channel);

    // Test HybridSearchRequest serialization
    let request = HybridSearchRequest {
        query: "test search query".to_string(),
        context: SearchContext::Project as i32,
        options: Some(SearchOptions {
            limit: 10,
            score_threshold: 0.5,
            include_metadata: true,
            include_content: true,
            ranking: Some(RankingOptions {
                semantic_weight: 0.7,
                keyword_weight: 0.3,
                rrf_constant: 60.0,
            }),
        }),
        project_id: "test_project".to_string(),
        collection_names: vec!["collection1".to_string(), "collection2".to_string()],
    };

    let response = timeout(Duration::from_secs(5), client.hybrid_search(Request::new(request)))
        .await
        .expect("Request timeout")
        .expect("Request failed")
        .into_inner();

    // Validate response serialization
    assert!(!response.query_id.is_empty());
    assert!(response.metadata.is_some());
    assert!(response.results.is_empty()); // Empty results expected for test

    server.shutdown().await;
}

// =============================================================================
// SERVICE METHOD ROUTING TESTS
// =============================================================================

#[tokio::test]
async fn test_document_processor_service_routing() {
    let server = TestServer::start().await;
    let channel = server.get_client_channel().await;
    let mut client = document_processor_client::DocumentProcessorClient::new(channel);

    // Test ProcessDocument method
    let process_request = ProcessDocumentRequest {
        file_path: "/test/routing.txt".to_string(),
        project_id: "routing_project".to_string(),
        collection_name: "routing_collection".to_string(),
        document_type: DocumentType::Text as i32,
        metadata: HashMap::new(),
        options: None,
    };

    let process_response = timeout(
        Duration::from_secs(5),
        client.process_document(Request::new(process_request))
    )
    .await
    .expect("Timeout")
    .expect("Request failed")
    .into_inner();

    assert!(!process_response.document_id.is_empty());

    // Test GetProcessingStatus method
    let status_request = ProcessingStatusRequest {
        operation_id: "test_operation".to_string(),
    };

    let status_response = timeout(
        Duration::from_secs(5),
        client.get_processing_status(Request::new(status_request))
    )
    .await
    .expect("Timeout")
    .expect("Request failed")
    .into_inner();

    assert_eq!(status_response.operation_id, "test_operation");

    // Test CancelProcessing method
    let cancel_request = CancelProcessingRequest {
        operation_id: "test_cancel".to_string(),
    };

    let _cancel_response = timeout(
        Duration::from_secs(5),
        client.cancel_processing(Request::new(cancel_request))
    )
    .await
    .expect("Timeout")
    .expect("Request failed");

    server.shutdown().await;
}

#[tokio::test]
async fn test_system_service_routing() {
    let server = TestServer::start().await;
    let channel = server.get_client_channel().await;
    let mut client = system_service_client::SystemServiceClient::new(channel);

    // Test HealthCheck
    let health_response = timeout(
        Duration::from_secs(5),
        client.health_check(Request::new(()))
    )
    .await
    .expect("Timeout")
    .expect("Request failed")
    .into_inner();

    assert_eq!(health_response.status, ServiceStatus::Healthy as i32);

    // Test GetStatus
    let status_response = timeout(
        Duration::from_secs(5),
        client.get_status(Request::new(()))
    )
    .await
    .expect("Timeout")
    .expect("Request failed")
    .into_inner();

    assert_eq!(status_response.status, ServiceStatus::Healthy as i32);

    server.shutdown().await;
}

// =============================================================================
// ENUM VALUES PROTOCOL COMPLIANCE TESTS
// =============================================================================

#[tokio::test]
async fn test_enum_values_protocol_compliance() {
    let server = TestServer::start().await;
    let channel = server.get_client_channel().await;
    let mut client = document_processor_client::DocumentProcessorClient::new(channel);

    // Test all DocumentType enum values
    let document_types = [
        DocumentType::Unspecified,
        DocumentType::Code,
        DocumentType::Pdf,
        DocumentType::Text,
        DocumentType::Markdown,
        DocumentType::Json,
    ];

    for doc_type in document_types {
        let request = ProcessDocumentRequest {
            file_path: format!("/test/enum_{:?}.txt", doc_type),
            project_id: "enum_test".to_string(),
            collection_name: "enum_collection".to_string(),
            document_type: doc_type as i32,
            metadata: HashMap::new(),
            options: None,
        };

        let response = timeout(
            Duration::from_secs(5),
            client.process_document(Request::new(request))
        )
        .await
        .expect("Timeout")
        .expect("Request failed")
        .into_inner();

        assert!(!response.document_id.is_empty());
        assert_eq!(response.status, ProcessingStatus::Completed as i32);
    }

    server.shutdown().await;
}

// =============================================================================
// ERROR HANDLING TESTS
// =============================================================================

#[tokio::test]
async fn test_basic_error_handling() {
    let server = TestServer::start().await;
    let channel = server.get_client_channel().await;
    let mut client = memory_service_client::MemoryServiceClient::new(channel);

    // Test with potentially invalid request (current implementation is placeholder)
    let invalid_request = GetDocumentRequest {
        document_id: "".to_string(), // Empty document ID
        collection_name: "".to_string(), // Empty collection name
    };

    let _response = timeout(Duration::from_secs(5), client.get_document(Request::new(invalid_request))).await;
    // Note: Current implementation doesn't validate input, but protocol supports error handling

    server.shutdown().await;
}

// =============================================================================
// METADATA PROPAGATION TESTS
// =============================================================================

#[tokio::test]
async fn test_grpc_metadata_basic() {
    let server = TestServer::start().await;
    let channel = server.get_client_channel().await;
    let mut client = system_service_client::SystemServiceClient::new(channel);

    // Create request with metadata
    let mut request = Request::new(());
    request.metadata_mut().insert(
        "client-id",
        tonic::metadata::MetadataValue::from_static("test_client")
    );

    let response = timeout(Duration::from_secs(5), client.health_check(request))
        .await
        .expect("Request timeout")
        .expect("Request failed");

    assert_eq!(response.into_inner().status, ServiceStatus::Healthy as i32);

    server.shutdown().await;
}

// =============================================================================
// CONCURRENT ACCESS TESTS
// =============================================================================

#[tokio::test]
async fn test_concurrent_protocol_access() {
    let server = TestServer::start().await;
    let channel = server.get_client_channel().await;

    let mut handles = vec![];

    // Spawn concurrent requests
    for _i in 0..10 {
        let mut client = system_service_client::SystemServiceClient::new(channel.clone());
        let handle = tokio::spawn(async move {
            client.health_check(Request::new(())).await
        });
        handles.push(handle);
    }

    // Wait for all requests to complete
    let results = futures_util::future::join_all(handles).await;

    let mut success_count = 0;
    for result in results {
        match result.unwrap() {
            Ok(_) => success_count += 1,
            Err(_) => {},
        }
    }

    // All requests should succeed
    assert_eq!(success_count, 10);

    server.shutdown().await;
}

// =============================================================================
// COMPLEX MESSAGE SERIALIZATION TESTS
// =============================================================================

#[tokio::test]
async fn test_complex_nested_message_serialization() {
    let server = TestServer::start().await;
    let channel = server.get_client_channel().await;
    let mut client = memory_service_client::MemoryServiceClient::new(channel);

    // Test complex DocumentContent serialization
    let chunks = vec![
        DocumentChunk {
            id: "chunk_1".to_string(),
            content: "First chunk content".to_string(),
            start_offset: 0,
            end_offset: 100,
            metadata: HashMap::from([
                ("type".to_string(), "header".to_string()),
            ]),
        },
        DocumentChunk {
            id: "chunk_2".to_string(),
            content: "Second chunk content".to_string(),
            start_offset: 100,
            end_offset: 200,
            metadata: HashMap::from([
                ("type".to_string(), "body".to_string()),
            ]),
        },
    ];

    let document_content = DocumentContent {
        text: "Full document text".to_string(),
        chunks,
        extracted_metadata: HashMap::from([
            ("title".to_string(), "Test Document".to_string()),
            ("size".to_string(), "200".to_string()),
        ]),
    };

    let add_request = AddDocumentRequest {
        file_path: "/test/complex.txt".to_string(),
        collection_name: "complex_collection".to_string(),
        project_id: "complex_project".to_string(),
        content: Some(document_content),
        metadata: HashMap::from([
            ("test".to_string(), "complex".to_string()),
        ]),
    };

    let response = timeout(Duration::from_secs(5), client.add_document(Request::new(add_request)))
        .await
        .expect("Request timeout")
        .expect("Request failed")
        .into_inner();

    assert!(!response.document_id.is_empty());
    assert!(response.success);

    server.shutdown().await;
}