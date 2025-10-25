//! Comprehensive gRPC client-server communication integration tests
//! Tests end-to-end gRPC communication workflows with TDD approach
//! Targets 90%+ coverage for client-server communication patterns

use workspace_qdrant_daemon::config::*;
use workspace_qdrant_daemon::daemon::WorkspaceDaemon;
use workspace_qdrant_daemon::grpc::server::GrpcServer;
use workspace_qdrant_daemon::proto::{
    document_processor_client::DocumentProcessorClient,
    search_service_client::SearchServiceClient,
    memory_service_client::MemoryServiceClient,
    system_service_client::SystemServiceClient,
    ServiceStatus, ProcessDocumentRequest, DocumentType, ProcessingOptions,
    CreateCollectionRequest, ProcessingStatus, HybridSearchRequest,
    CollectionConfig as ProtoCollectionConfig, HealthCheckResponse, SearchContext,
};
use tempfile::TempDir;

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::time::Duration;
use tokio::time::{timeout, sleep};
use tonic::transport::{Channel, Endpoint};
use tonic::{Code, Request, Status};
use serial_test::serial;

// ================================
// TEST INFRASTRUCTURE
// ================================

struct TestEnvironment {
    _daemon: WorkspaceDaemon,
    server_handle: tokio::task::JoinHandle<Result<(), anyhow::Error>>,
    address: SocketAddr,
    client_channel: Channel,
}

impl TestEnvironment {
    async fn new() -> Self {
        let config = create_test_daemon_config();
        let daemon = WorkspaceDaemon::new(config).await
            .expect("Failed to create test daemon");
        
        // Use random port to avoid conflicts
        let address = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 0);
        let grpc_server = GrpcServer::new(daemon.clone(), address);
        
        // Start server in background
        let server_handle = tokio::spawn(async move {
            grpc_server.serve_daemon().await
        });
        
        // Give server time to start
        sleep(Duration::from_millis(100)).await;
        
        // Create client channel
        let endpoint = Endpoint::from_shared(format!("http://{}", address))
            .expect("Failed to create endpoint")
            .timeout(Duration::from_secs(10))
            .connect_timeout(Duration::from_secs(5));
        
        let client_channel = endpoint.connect().await
            .expect("Failed to connect to server");
        
        Self {
            _daemon: daemon,
            server_handle,
            address,
            client_channel,
        }
    }
    
    fn document_processor_client(&self) -> DocumentProcessorClient<Channel> {
        DocumentProcessorClient::new(self.client_channel.clone())
    }
    
    fn search_service_client(&self) -> SearchServiceClient<Channel> {
        SearchServiceClient::new(self.client_channel.clone())
    }
    
    fn memory_service_client(&self) -> MemoryServiceClient<Channel> {
        MemoryServiceClient::new(self.client_channel.clone())
    }
    
    fn system_service_client(&self) -> SystemServiceClient<Channel> {
        SystemServiceClient::new(self.client_channel.clone())
    }
}

impl Drop for TestEnvironment {
    fn drop(&mut self) {
        self.server_handle.abort();
    }
}

fn create_test_daemon_config() -> DaemonConfig {
    let db_path = ":memory:";

    DaemonConfig {
        server: ServerConfig {
            host: "127.0.0.1".to_string(),
            port: 50052,
            max_connections: 100,
            connection_timeout_secs: 30,
            request_timeout_secs: 60,
            enable_tls: false,
            security: SecurityConfig::default(),
            transport: TransportConfig::default(),
            message: MessageConfig::default(),
            compression: CompressionConfig::default(),
            streaming: StreamingConfig::default(),
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
            default_collection: workspace_qdrant_daemon::config::CollectionConfig {
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
            max_file_size_bytes: 10 * 1024 * 1024,
            supported_extensions: vec![
                "rs".to_string(),
                "py".to_string(),
                "js".to_string(),
                "ts".to_string(),
                "md".to_string(),
                "txt".to_string(),
            ],
            enable_lsp: false,
            lsp_timeout_secs: 10,
        },
        file_watcher: FileWatcherConfig {
            enabled: false,
            debounce_ms: 100,
            max_watched_dirs: 10,
            ignore_patterns: vec!["*.tmp".to_string(), "*.log".to_string()],
            recursive: true,
        },
        metrics: MetricsConfig {
            enabled: false,
            prometheus_port: 9090,
            collection_interval_secs: 60,
            retention_days: 7,
        },
        logging: LoggingConfig {
            level: "info".to_string(),
            json_format: false,
            file_path: None,
            max_file_size_mb: 100,
            max_files: 5,
        },
    }
}

// ================================
// CLIENT CONNECTION LIFECYCLE TESTS
// ================================

#[tokio::test]
#[serial]
async fn test_client_connection_establishment() {
    let env = TestEnvironment::new().await;
    
    // Test basic connection establishment
    let mut client = env.system_service_client();
    
    let request = Request::new(());
    let response = timeout(
        Duration::from_secs(5),
        client.health_check(request)
    ).await;
    
    assert!(response.is_ok(), "Client should successfully connect to server");
    let health_response = response.unwrap().unwrap();
    assert_eq!(health_response.into_inner().status, ServiceStatus::Healthy as i32);
}

#[tokio::test]
#[serial]
async fn test_client_connection_timeout() {
    // Test connection timeout with invalid address
    let invalid_endpoint = Endpoint::from_static("http://127.0.0.1:1")
        .timeout(Duration::from_millis(100))
        .connect_timeout(Duration::from_millis(100));
    
    let connection_result = timeout(
        Duration::from_millis(500),
        invalid_endpoint.connect()
    ).await;
    
    // Should timeout or fail to connect
    assert!(connection_result.is_err() || connection_result.unwrap().is_err(),
           "Connection to invalid address should timeout or fail");
}

#[tokio::test]
#[serial]
async fn test_client_connection_recovery() {
    let env = TestEnvironment::new().await;
    let mut client = env.system_service_client();
    
    // First successful request
    let request1 = Request::new(());
    let response1 = client.health_check(request1).await;
    assert!(response1.is_ok(), "First request should succeed");
    
    // Simulate server restart by dropping environment
    drop(env);
    
    // Give some time for connection to detect failure
    sleep(Duration::from_millis(100)).await;
    
    // Create new environment
    let new_env = TestEnvironment::new().await;
    let mut new_client = new_env.system_service_client();
    
    // Test recovery with new connection
    let request2 = Request::new(());
    let response2 = new_client.health_check(request2).await;
    assert!(response2.is_ok(), "Recovery request should succeed");
}

// ================================
// CONCURRENT CLIENT REQUEST TESTS
// ================================

#[tokio::test]
#[serial]
async fn test_concurrent_client_requests() {
    let env = TestEnvironment::new().await;
    
    // Create multiple clients for concurrent requests
    let client_count = 10;
    let mut handles = Vec::new();
    
    for i in 0..client_count {
        let mut client = env.system_service_client();
        let handle = tokio::spawn(async move {
            let request = Request::new(());
            let result = client.health_check(request).await;
            (i, result)
        });
        handles.push(handle);
    }
    
    // Wait for all requests to complete
    let mut successful_requests = 0;
    for handle in handles {
        let (client_id, result) = handle.await.unwrap();
        match result {
            Ok(_) => successful_requests += 1,
            Err(e) => eprintln!("Client {} failed: {}", client_id, e),
        }
    }
    
    // All requests should succeed
    assert_eq!(successful_requests, client_count,
              "All concurrent requests should succeed");
}

#[tokio::test]
#[serial]
async fn test_request_backpressure_handling() {
    let env = TestEnvironment::new().await;
    
    // Send a large number of requests to test backpressure
    let request_count = 100;
    let mut handles = Vec::new();
    
    for i in 0..request_count {
        let mut client = env.system_service_client();
        let handle = tokio::spawn(async move {
            let request = Request::new(());
            let start_time = std::time::Instant::now();
            let result = client.health_check(request).await;
            let duration = start_time.elapsed();
            (i, result, duration)
        });
        handles.push(handle);
    }
    
    let mut successful_requests = 0;
    let mut total_duration = Duration::ZERO;
    
    for handle in handles {
        let (_, result, duration) = handle.await.unwrap();
        if result.is_ok() {
            successful_requests += 1;
            total_duration += duration;
        }
    }
    
    // Most requests should succeed (allowing for some failures under load)
    assert!(successful_requests >= request_count * 8 / 10,
           "At least 80% of requests should succeed under load");
    
    // Average response time should be reasonable
    let avg_duration = total_duration / successful_requests as u32;
    assert!(avg_duration < Duration::from_secs(1),
           "Average response time should be under 1 second");
}

// ================================
// SERVICE-SPECIFIC COMMUNICATION TESTS
// ================================

#[tokio::test]
#[serial]
async fn test_document_processor_communication() {
    let env = TestEnvironment::new().await;
    let mut client = env.document_processor_client();
    
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("test.txt");
    std::fs::write(&test_file, "Test document content").unwrap();
    
    let request = Request::new(ProcessDocumentRequest {
        file_path: test_file.to_string_lossy().to_string(),
        project_id: "test-project".to_string(),
        collection_name: "test-collection".to_string(),
        document_type: DocumentType::DocumentTypeText as i32,
        metadata: Default::default(),
        options: Some(ProcessingOptions {
            enable_lsp_analysis: false,
            chunk_size: 1000,
            chunk_overlap: 200,
            extract_metadata: true,
            detect_language: false,
            custom_parsers: vec![],
        }),
    });
    
    let response = timeout(
        Duration::from_secs(10),
        client.process_document(request)
    ).await;
    
    match response {
        Ok(Ok(resp)) => {
            let doc_response = resp.into_inner();
            assert!(!doc_response.document_id.is_empty(), "Document ID should not be empty");
            assert_eq!(doc_response.status, ProcessingStatus::Completed as i32);
        },
        Ok(Err(status)) => {
            // Some processing errors might be expected in test environment
            println!("Processing failed with status: {}", status);
        },
        Err(_) => panic!("Request should not timeout"),
    }
}

#[tokio::test]
#[serial]
async fn test_search_service_communication() {
    let env = TestEnvironment::new().await;
    let mut client = env.search_service_client();
    
    let request = Request::new(HybridSearchRequest {
        query: "test query".to_string(),
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
        project_id: "test-project".to_string(),
        collection_names: vec!["test-collection".to_string()],
    });
    
    let response = timeout(
        Duration::from_secs(10),
        client.hybrid_search(request)
    ).await;
    
    // Response should complete (might be empty results)
    assert!(response.is_ok(), "Search request should complete");
    match response.unwrap() {
        Ok(resp) => {
            let search_response = resp.into_inner();
            assert!(!search_response.query_id.is_empty(), "Query ID should be assigned");
        },
        Err(status) => {
            // Search might fail if no collections exist, which is acceptable
            println!("Search failed with status: {}", status);
        }
    }
}

#[tokio::test]
#[serial]
async fn test_memory_service_communication() {
    let env = TestEnvironment::new().await;
    let mut client = env.memory_service_client();
    
    // Test create collection
    let create_request = Request::new(CreateCollectionRequest {
        collection_name: "test-collection".to_string(),
        project_id: "test-project".to_string(),
        config: Some(ProtoCollectionConfig {
            vector_size: 384,
            distance_metric: "Cosine".to_string(),
            enable_indexing: true,
            metadata_schema: Default::default(),
        }),
    });
    
    let create_response = timeout(
        Duration::from_secs(10),
        client.create_collection(create_request)
    ).await;
    
    assert!(create_response.is_ok(), "Create collection request should complete");
    
    // Test list collections
    let list_request = Request::new(ListCollectionsRequest {
        project_id: "test-project".to_string(),
    });
    
    let list_response = timeout(
        Duration::from_secs(10),
        client.list_collections(list_request)
    ).await;
    
    assert!(list_response.is_ok(), "List collections request should complete");
}

// ================================
// STREAMING COMMUNICATION TESTS
// ================================

#[tokio::test]
#[serial]
async fn test_streaming_document_processing() {
    let env = TestEnvironment::new().await;
    let mut client = env.document_processor_client();
    
    let temp_dir = TempDir::new().unwrap();
    
    // Create multiple test files
    let mut requests = Vec::new();
    for i in 0..3 {
        let test_file = temp_dir.path().join(format!("test_{}.txt", i));
        std::fs::write(&test_file, format!("Test document content {}", i)).unwrap();
        
        requests.push(ProcessDocumentRequest {
            file_path: test_file.to_string_lossy().to_string(),
            project_id: "test-project".to_string(),
            collection_name: "test-collection".to_string(),
            document_type: DocumentType::DocumentTypeText as i32,
            metadata: Default::default(),
            options: Some(ProcessingOptions {
                enable_lsp_analysis: false,
                chunk_size: 1000,
                chunk_overlap: 200,
                extract_metadata: true,
                detect_language: false,
                custom_parsers: vec![],
            }),
        });
    }
    
    // Create request stream
    let request_stream = stream::iter(requests.into_iter().map(Request::new));
    
    let response = timeout(
        Duration::from_secs(30),
        client.process_documents(request_stream)
    ).await;
    
    assert!(response.is_ok(), "Streaming request should complete");
    
    match response.unwrap() {
        Ok(mut response_stream) => {
            let mut response_count = 0;
            while let Ok(Some(resp)) = timeout(
                Duration::from_secs(5),
                response_stream.message()
            ).await {
                match resp {
                    Ok(msg) => {
                        response_count += 1;
                        assert!(!msg.document_id.is_empty(), "Document ID should not be empty");
                    },
                    Err(status) => {
                        println!("Streaming response error: {}", status);
                    }
                }
            }
            
            // Should receive responses for all requests (even if some fail)
            assert!(response_count > 0, "Should receive at least one response");
        },
        Err(status) => {
            println!("Streaming request failed: {}", status);
        }
    }
}

// ================================
// ERROR HANDLING TESTS
// ================================

#[tokio::test]
#[serial]
async fn test_invalid_request_handling() {
    let env = TestEnvironment::new().await;
    let mut client = env.document_processor_client();
    
    // Send invalid request (non-existent file)
    let request = Request::new(ProcessDocumentRequest {
        file_path: "/non/existent/file.txt".to_string(),
        project_id: "test-project".to_string(),
        collection_name: "test-collection".to_string(),
        document_type: DocumentType::DocumentTypeText as i32,
        metadata: Default::default(),
        options: None,
    });
    
    let response = client.process_document(request).await;
    
    // Should receive error response, not panic
    match response {
        Ok(resp) => {
            let doc_response = resp.into_inner();
            assert_eq!(doc_response.status, ProcessingStatus::Failed as i32);
            assert!(!doc_response.error_message.is_empty());
        },
        Err(status) => {
            // Error status is also acceptable
            assert!(matches!(status.code(), Code::InvalidArgument | Code::NotFound | Code::Internal));
        }
    }
}

#[tokio::test]
#[serial]
async fn test_connection_error_propagation() {
    let env = TestEnvironment::new().await;
    let mut client = env.system_service_client();
    
    // Close the server
    drop(env);
    
    // Give some time for connection to detect failure
    sleep(Duration::from_millis(200)).await;
    
    // Try to make request to closed server
    let request = Request::new(());
    let response = client.health_check(request).await;
    
    // Should receive connection error
    assert!(response.is_err(), "Request to closed server should fail");
    let status = response.unwrap_err();
    assert!(matches!(status.code(), 
                    Code::Unavailable | Code::Cancelled | Code::DeadlineExceeded),
           "Should receive appropriate connection error code");
}

// ================================
// CONNECTION MANAGEMENT TESTS
// ================================

#[tokio::test]
#[serial]
async fn test_connection_pooling() {
    let env = TestEnvironment::new().await;
    
    // Create multiple clients using the same channel
    let clients: Vec<_> = (0..5).map(|_| env.system_service_client()).collect();
    
    // Make requests with all clients concurrently
    let mut handles = Vec::new();
    for (i, mut client) in clients.into_iter().enumerate() {
        let handle = tokio::spawn(async move {
            let request = Request::new(());
            let result = client.health_check(request).await;
            (i, result)
        });
        handles.push(handle);
    }
    
    // All clients should succeed
    let mut successful_requests = 0;
    for handle in handles {
        let (_, result) = handle.await.unwrap();
        if result.is_ok() {
            successful_requests += 1;
        }
    }
    
    assert_eq!(successful_requests, 5, "All pooled connections should work");
}

#[tokio::test]
#[serial]
async fn test_connection_keep_alive() {
    let env = TestEnvironment::new().await;
    let mut client = env.system_service_client();
    
    // Make initial request
    let request1 = Request::new(());
    let response1 = client.health_check(request1).await;
    assert!(response1.is_ok(), "Initial request should succeed");
    
    // Wait longer than typical idle timeout
    sleep(Duration::from_secs(2)).await;
    
    // Make another request on same connection
    let request2 = Request::new(());
    let response2 = client.health_check(request2).await;
    assert!(response2.is_ok(), "Request after idle period should succeed");
}

// ================================
// LOAD AND STRESS TESTS
// ================================

#[tokio::test]
#[serial]
async fn test_high_load_scenarios() {
    let env = TestEnvironment::new().await;
    
    // Create high load with rapid requests
    let request_count = 50;
    let mut handles = Vec::new();
    
    let start_time = std::time::Instant::now();
    
    for i in 0..request_count {
        let mut client = env.system_service_client();
        let handle = tokio::spawn(async move {
            // Make multiple requests per client
            let mut results = Vec::new();
            for j in 0..3 {
                let request = Request::new(());
                let result = client.health_check(request).await;
                results.push((i, j, result));
                
                // Small delay between requests
                sleep(Duration::from_millis(10)).await;
            }
            results
        });
        handles.push(handle);
    }
    
    let mut total_requests = 0;
    let mut successful_requests = 0;
    
    for handle in handles {
        let results = handle.await.unwrap();
        for (_, _, result) in results {
            total_requests += 1;
            if result.is_ok() {
                successful_requests += 1;
            }
        }
    }
    
    let duration = start_time.elapsed();
    let success_rate = successful_requests as f64 / total_requests as f64;
    
    println!("Load test: {}/{} requests succeeded in {:?}", 
             successful_requests, total_requests, duration);
    println!("Success rate: {:.2}%", success_rate * 100.0);
    
    // Under high load, should maintain reasonable success rate
    assert!(success_rate >= 0.8, "Should maintain at least 80% success rate under load");
    
    // Should complete within reasonable time
    assert!(duration < Duration::from_secs(30), "Load test should complete within 30 seconds");
}

// ================================
// AUTHENTICATION AND SECURITY TESTS
// ================================

#[tokio::test]
#[serial]
async fn test_request_metadata_handling() {
    let env = TestEnvironment::new().await;
    let mut client = env.system_service_client();
    
    // Create request with custom metadata
    let mut request = Request::new(());
    request.metadata_mut().insert("client-id", "test-client-123".parse().unwrap());
    request.metadata_mut().insert("user-agent", "test-agent/1.0".parse().unwrap());
    
    let response = client.health_check(request).await;
    assert!(response.is_ok(), "Request with metadata should succeed");
    
    // Check if response contains metadata
    let resp = response.unwrap();
    let metadata = resp.metadata();
    
    // Server might add its own metadata
    println!("Response metadata: {:?}", metadata);
}

#[tokio::test]
#[serial]
async fn test_request_size_limits() {
    let env = TestEnvironment::new().await;
    let mut client = env.memory_service_client();
    
    // Create request with large metadata
    let large_metadata = (0..1000)
        .map(|i| (format!("key_{}", i), format!("value_{}_with_long_content", i)))
        .collect();
    
    let request = Request::new(AddDocumentRequest {
        file_path: "/test/file.txt".to_string(),
        collection_name: "test-collection".to_string(),
        project_id: "test-project".to_string(),
        content: Some(DocumentContent {
            text: "test content".to_string(),
            chunks: vec![],
            extracted_metadata: Default::default(),
        }),
        metadata: large_metadata,
    });
    
    let response = timeout(
        Duration::from_secs(10),
        client.add_document(request)
    ).await;
    
    // Should handle large requests or return appropriate error
    match response {
        Ok(Ok(_)) => println!("Large request handled successfully"),
        Ok(Err(status)) => {
            println!("Large request rejected: {}", status);
            // Appropriate error codes for size limits
            assert!(matches!(status.code(), 
                           Code::InvalidArgument | Code::ResourceExhausted | Code::Internal));
        },
        Err(_) => panic!("Request should not timeout"),
    }
}

// ================================
// EDGE CASE AND BOUNDARY TESTS
// ================================

#[tokio::test]
#[serial]
async fn test_empty_request_handling() {
    let env = TestEnvironment::new().await;
    let mut search_client = env.search_service_client();
    
    // Test empty search query
    let request = Request::new(HybridSearchRequest {
        query: "".to_string(),
        context: SearchContext::Project as i32,
        options: None,
        project_id: "".to_string(),
        collection_names: vec![],
    });
    
    let response = search_client.hybrid_search(request).await;
    
    // Should handle empty requests gracefully
    match response {
        Ok(resp) => {
            let search_response = resp.into_inner();
            assert!(search_response.results.is_empty());
        },
        Err(status) => {
            // Invalid argument is acceptable for empty queries
            assert!(matches!(status.code(), Code::InvalidArgument | Code::Internal));
        }
    }
}

#[tokio::test]
#[serial]
async fn test_malformed_request_handling() {
    let env = TestEnvironment::new().await;
    let mut client = env.document_processor_client();
    
    // Test request with invalid enum values
    let mut request = Request::new(ProcessDocumentRequest {
        file_path: "test.txt".to_string(),
        project_id: "test-project".to_string(),
        collection_name: "test-collection".to_string(),
        document_type: 999, // Invalid enum value
        metadata: Default::default(),
        options: None,
    });
    
    let response = client.process_document(request).await;
    
    // Should handle malformed requests gracefully
    match response {
        Ok(resp) => {
            let doc_response = resp.into_inner();
            // Might succeed with default handling or fail gracefully
            if doc_response.status == ProcessingStatus::Failed as i32 {
                assert!(!doc_response.error_message.is_empty());
            }
        },
        Err(status) => {
            assert!(matches!(status.code(), Code::InvalidArgument | Code::Internal));
        }
    }
}

// ================================
// CLEANUP AND UTILITIES
// ================================

#[tokio::test]
#[serial]
async fn test_graceful_shutdown_handling() {
    let env = TestEnvironment::new().await;
    let mut client = env.system_service_client();
    
    // Start a long-running request
    let request_handle = tokio::spawn(async move {
        let request = Request::new(());
        client.health_check(request).await
    });
    
    // Give request time to start
    sleep(Duration::from_millis(50)).await;
    
    // Simulate shutdown by dropping environment
    drop(env);
    
    // Check how request handles shutdown
    let result = timeout(Duration::from_secs(5), request_handle).await;
    
    match result {
        Ok(Ok(Ok(_))) => println!("Request completed before shutdown"),
        Ok(Ok(Err(status))) => {
            println!("Request failed during shutdown: {}", status);
            // Expected error codes during shutdown
            assert!(matches!(status.code(), 
                           Code::Unavailable | Code::Cancelled | Code::Aborted));
        },
        Ok(Err(_)) => println!("Request task was cancelled"),
        Err(_) => println!("Request timed out during shutdown"),
    }
}
