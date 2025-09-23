//! Comprehensive gRPC streaming integration tests with TDD approach
//! Tests cover unary, server streaming, client streaming, bidirectional streaming
//! with proper stream lifecycle, backpressure, flow control, and error handling

use workspace_qdrant_daemon::config::*;
use workspace_qdrant_daemon::daemon::WorkspaceDaemon;
use workspace_qdrant_daemon::grpc::server::GrpcServer;
use workspace_qdrant_daemon::proto::{
    document_processor_client,
    system_service_client,
    ProcessDocumentRequest, ProcessDocumentResponse,
    ProcessingStatusRequest, ProcessingStatusResponse,
    DocumentType, ProcessingOptions, ProcessingStatus,
    ServiceStatus, CollectionConfig as ProtoCollectionConfig,
};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_stream::{StreamExt, wrappers::ReceiverStream};
use tonic::transport::Channel;
use tonic::Request;
use tracing::{debug, info};

// ================================
// TEST CONFIGURATION AND SETUP
// ================================

fn create_test_daemon_config() -> DaemonConfig {
    DaemonConfig {
        server: ServerConfig {
            host: "127.0.0.1".to_string(),
            port: 0, // Use random port
            max_connections: 100,
            connection_timeout_secs: 30,
            request_timeout_secs: 60,
            enable_tls: false,
        },
        database: DatabaseConfig {
            sqlite_path: ":memory:".to_string(),
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
            max_file_size_bytes: 1024 * 1024,
            supported_extensions: vec!["txt".to_string(), "md".to_string(), "rs".to_string()],
            enable_lsp: false,
            lsp_timeout_secs: 10,
        },
        file_watcher: FileWatcherConfig {
            enabled: true,
            debounce_ms: 100, // Fast for testing
            max_watched_dirs: 10,
            ignore_patterns: vec![".git".to_string(), "target".to_string()],
            recursive: true,
        },
        metrics: MetricsConfig {
            enabled: true,
            collection_interval_secs: 1, // Fast for testing
            retention_days: 1,
            enable_prometheus: false,
            prometheus_port: 0,
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

async fn create_test_daemon() -> WorkspaceDaemon {
    let config = create_test_daemon_config();
    WorkspaceDaemon::new(config).await.expect("Failed to create test daemon")
}

/// Set up a test gRPC server and return the address and client channel
async fn setup_test_server() -> (SocketAddr, Channel) {
    let daemon = create_test_daemon().await;
    let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0);

    let server = GrpcServer::new(daemon, addr);
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    let bound_addr = listener.local_addr().unwrap();

    // Start server in background
    tokio::spawn(async move {
        // We need to create the server manually since build_server is private
        let daemon_arc = std::sync::Arc::new(daemon);
        let document_processor = workspace_qdrant_daemon::grpc::services::DocumentProcessorImpl::new(daemon_arc.clone());
        let system_service = workspace_qdrant_daemon::grpc::services::SystemServiceImpl::new(daemon_arc.clone());

        let grpc_server = tonic::transport::Server::builder()
            .add_service(workspace_qdrant_daemon::proto::document_processor_server::DocumentProcessorServer::new(document_processor))
            .add_service(workspace_qdrant_daemon::proto::system_service_server::SystemServiceServer::new(system_service));

        grpc_server.serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
            .await
            .unwrap();
    });

    // Wait for server to start
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Create client channel
    let channel = Channel::from_shared(format!("http://{}", bound_addr))
        .unwrap()
        .connect()
        .await
        .unwrap();

    (bound_addr, channel)
}

// ================================
// BIDIRECTIONAL STREAMING TESTS
// ================================

#[tokio::test]
async fn test_process_documents_bidirectional_streaming_basic() {
    let (_addr, channel) = setup_test_server().await;
    let client = document_processor_client::DocumentProcessorClient::new(channel);

    // Create input stream
    let (tx, rx) = mpsc::channel(10);
    let input_stream = ReceiverStream::new(rx);

    // Start streaming request
    let response = client.process_documents(Request::new(input_stream)).await;
    assert!(response.is_ok(), "Failed to start streaming: {:?}", response.err());

    let mut response_stream = response.unwrap().into_inner();

    // Send multiple documents
    let documents = vec![
        ProcessDocumentRequest {
            file_path: "/test/doc1.txt".to_string(),
            project_id: "test-project".to_string(),
            collection_name: "test-collection".to_string(),
            document_type: DocumentType::Text as i32,
            metadata: std::collections::HashMap::new(),
            options: Some(ProcessingOptions {
                enable_lsp_analysis: false,
                chunk_size: 500,
                chunk_overlap: 100,
                extract_metadata: true,
                detect_language: true,
                custom_parsers: vec![],
            }),
        },
        ProcessDocumentRequest {
            file_path: "/test/doc2.md".to_string(),
            project_id: "test-project".to_string(),
            collection_name: "test-collection".to_string(),
            document_type: DocumentType::Markdown as i32,
            metadata: {
                let mut meta = std::collections::HashMap::new();
                meta.insert("author".to_string(), "test".to_string());
                meta
            },
            options: Some(ProcessingOptions {
                enable_lsp_analysis: false,
                chunk_size: 1000,
                chunk_overlap: 200,
                extract_metadata: true,
                detect_language: true,
                custom_parsers: vec![],
            }),
        },
    ];

    // Send documents and verify responses
    for (i, doc) in documents.into_iter().enumerate() {
        tx.send(doc).await.unwrap();

        // Receive response
        let response = response_stream.next().await;
        assert!(response.is_some(), "No response received for document {}", i);

        let response = response.unwrap();
        assert!(response.is_ok(), "Error response for document {}: {:?}", i, response.err());

        let response = response.unwrap();
        assert!(!response.document_id.is_empty(), "Empty document ID");
        assert_eq!(response.status, ProcessingStatus::Completed as i32);
        assert!(response.chunks_created > 0, "No chunks created");
    }

    // Close input stream
    drop(tx);

    // Verify stream ends
    let final_response = response_stream.next().await;
    assert!(final_response.is_none(), "Stream should end after input closes");
}

#[tokio::test]
async fn test_process_documents_streaming_with_errors() {
    let (_addr, channel) = setup_test_server().await;
    let client = document_processor_client::DocumentProcessorClient::new(channel);

    let (tx, rx) = mpsc::channel(10);
    let input_stream = ReceiverStream::new(rx);

    let response = client.process_documents(Request::new(input_stream)).await;
    assert!(response.is_ok());

    let mut response_stream = response.unwrap().into_inner();

    // Send valid document
    let valid_doc = ProcessDocumentRequest {
        file_path: "/test/valid.txt".to_string(),
        project_id: "test-project".to_string(),
        collection_name: "test-collection".to_string(),
        document_type: DocumentType::Text as i32,
        metadata: std::collections::HashMap::new(),
        options: Some(ProcessingOptions::default()),
    };

    tx.send(valid_doc).await.unwrap();

    // Receive response for valid document
    let response = response_stream.next().await;
    assert!(response.is_some());
    assert!(response.unwrap().is_ok());

    // Send invalid document (empty path)
    let invalid_doc = ProcessDocumentRequest {
        file_path: "".to_string(),
        project_id: "test-project".to_string(),
        collection_name: "test-collection".to_string(),
        document_type: DocumentType::Text as i32,
        metadata: std::collections::HashMap::new(),
        options: Some(ProcessingOptions::default()),
    };

    tx.send(invalid_doc).await.unwrap();

    // Should receive error response
    let response = response_stream.next().await;
    assert!(response.is_some());

    // Error handling in stream - should get response with error status or Status error
    match response.unwrap() {
        Ok(resp) => {
            // Implementation might return error in response message
            assert_eq!(resp.status, ProcessingStatus::Failed as i32);
            assert!(!resp.error_message.is_empty());
        }
        Err(status) => {
            // Or return gRPC status error
            assert_eq!(status.code(), tonic::Code::InvalidArgument);
        }
    }

    drop(tx);
}

#[tokio::test]
async fn test_process_documents_streaming_backpressure() {
    let (_addr, channel) = setup_test_server().await;
    let client = document_processor_client::DocumentProcessorClient::new(channel);

    let (tx, rx) = mpsc::channel(2); // Small buffer to test backpressure
    let input_stream = ReceiverStream::new(rx);

    let response = client.process_documents(Request::new(input_stream)).await;
    assert!(response.is_ok());

    let mut response_stream = response.unwrap().into_inner();

    // Send documents rapidly to test backpressure
    let send_handle = tokio::spawn(async move {
        for i in 0..10 {
            let doc = ProcessDocumentRequest {
                file_path: format!("/test/doc{}.txt", i),
                project_id: "test-project".to_string(),
                collection_name: "test-collection".to_string(),
                document_type: DocumentType::Text as i32,
                metadata: std::collections::HashMap::new(),
                options: Some(ProcessingOptions::default()),
            };

            // This should block when buffer is full (backpressure)
            if tx.send(doc).await.is_err() {
                break;
            }
        }
        drop(tx);
    });

    // Slowly consume responses to test backpressure
    let mut count = 0;
    while let Some(response) = response_stream.next().await {
        assert!(response.is_ok());
        count += 1;

        // Add small delay to simulate slow consumer
        tokio::time::sleep(Duration::from_millis(50)).await;

        if count >= 10 {
            break;
        }
    }

    send_handle.await.unwrap();
    assert!(count > 0, "Should have received at least one response");
}

#[tokio::test]
async fn test_process_documents_streaming_cancellation() {
    let (_addr, channel) = setup_test_server().await;
    let client = document_processor_client::DocumentProcessorClient::new(channel);

    let (tx, rx) = mpsc::channel(10);
    let input_stream = ReceiverStream::new(rx);

    let response = client.process_documents(Request::new(input_stream)).await;
    assert!(response.is_ok());

    let mut response_stream = response.unwrap().into_inner();

    // Send a few documents
    for i in 0..3 {
        let doc = ProcessDocumentRequest {
            file_path: format!("/test/doc{}.txt", i),
            project_id: "test-project".to_string(),
            collection_name: "test-collection".to_string(),
            document_type: DocumentType::Text as i32,
            metadata: std::collections::HashMap::new(),
            options: Some(ProcessingOptions::default()),
        };
        tx.send(doc).await.unwrap();
    }

    // Receive some responses
    let mut count = 0;
    while let Some(response) = response_stream.next().await {
        assert!(response.is_ok());
        count += 1;

        // Cancel after receiving 2 responses
        if count >= 2 {
            break;
        }
    }

    // Drop the response stream to cancel
    drop(response_stream);
    drop(tx);

    assert_eq!(count, 2, "Should have received exactly 2 responses before cancellation");
}

// ================================
// SERVER STREAMING TESTS
// ================================

/// Test unary operations in DocumentProcessor service
#[tokio::test]
async fn test_document_processor_unary_operations() {
    // This test checks existing unary operations before streaming

    let (_addr, channel) = setup_test_server().await;
    let client = document_processor_client::DocumentProcessorClient::new(channel);

    // Test getting processing status (unary) - this should work
    let status_request = ProcessingStatusRequest {
        operation_id: "test-operation".to_string(),
    };

    let response = client.get_processing_status(Request::new(status_request)).await;
    assert!(response.is_ok(), "Processing status request failed: {:?}", response.err());

    let response = response.unwrap().into_inner();
    assert_eq!(response.operation_id, "test-operation");

    // Test single document processing (unary)
    let process_request = ProcessDocumentRequest {
        file_path: "/test/single.txt".to_string(),
        project_id: "test-project".to_string(),
        collection_name: "test-collection".to_string(),
        document_type: DocumentType::Text as i32,
        metadata: std::collections::HashMap::new(),
        options: Some(ProcessingOptions::default()),
    };

    let process_response = client.process_document(Request::new(process_request)).await;
    assert!(process_response.is_ok(), "Single document processing failed: {:?}", process_response.err());

    let process_response = process_response.unwrap().into_inner();
    assert!(!process_response.document_id.is_empty(), "Document ID should not be empty");
    assert_eq!(process_response.status, ProcessingStatus::Completed as i32);
}

/// Test basic system service unary operations
#[tokio::test]
async fn test_system_service_unary_operations() {
    let (_addr, channel) = setup_test_server().await;
    let mut client = system_service_client::SystemServiceClient::new(channel);

    // Test basic unary health check first
    let health_response = client.health_check(Request::new(())).await;
    assert!(health_response.is_ok(), "Health check failed: {:?}", health_response.err());

    let health = health_response.unwrap().into_inner();
    assert_eq!(health.status, ServiceStatus::Healthy as i32);

    // Test system status
    let status_response = client.get_status(Request::new(())).await;
    assert!(status_response.is_ok(), "Status check failed: {:?}", status_response.err());

    let status = status_response.unwrap().into_inner();
    assert_eq!(status.status, ServiceStatus::Healthy as i32);

    // TODO: Implement StreamSystemMetrics server streaming
    // This will require adding the method to SystemServiceImpl
}

/// Test that demonstrates where server streaming would be implemented
#[tokio::test]
async fn test_server_streaming_placeholder() {
    // This test would be for the ingestion service StartWatching method
    // which provides server streaming of file watch updates

    let (_addr, channel) = setup_test_server().await;

    // TODO: Implement IngestService with StartWatching streaming
    // For now, verify we can at least connect to the server
    let mut client = system_service_client::SystemServiceClient::new(channel);
    let health_response = client.health_check(Request::new(())).await;
    assert!(health_response.is_ok(), "Basic connectivity test failed");

    // Server streaming implementation would go here
    // Examples:
    // - ProcessFolder progress streaming
    // - File watching events streaming
    // - System metrics streaming
    // - Processing status streaming
}

// ================================
// CLIENT STREAMING TESTS
// ================================

/// Test bulk document upload via client streaming - needs implementation
#[tokio::test]
async fn test_bulk_document_upload_client_streaming() {
    // This would test a hypothetical bulk upload method that takes
    // a client stream of documents and returns a single response

    let (_addr, channel) = setup_test_server().await;

    // For now, test the existing ProcessDocuments which is bidirectional
    // but can be used in a client streaming manner
    let client = document_processor_client::DocumentProcessorClient::new(channel);

    let (tx, rx) = mpsc::channel(100);
    let input_stream = ReceiverStream::new(rx);

    let response = client.process_documents(Request::new(input_stream)).await;
    assert!(response.is_ok());

    let mut response_stream = response.unwrap().into_inner();

    // Send many documents (client streaming pattern)
    let documents: Vec<_> = (0..20).map(|i| {
        ProcessDocumentRequest {
            file_path: format!("/batch/doc{}.txt", i),
            project_id: "batch-project".to_string(),
            collection_name: "batch-collection".to_string(),
            document_type: DocumentType::Text as i32,
            metadata: {
                let mut meta = std::collections::HashMap::new();
                meta.insert("batch_id".to_string(), "test-batch".to_string());
                meta.insert("index".to_string(), i.to_string());
                meta
            },
            options: Some(ProcessingOptions {
                enable_lsp_analysis: false,
                chunk_size: 500,
                chunk_overlap: 50,
                extract_metadata: true,
                detect_language: false,
                custom_parsers: vec![],
            }),
        }
    }).collect();

    // Send all documents
    for doc in documents {
        tx.send(doc).await.unwrap();
    }
    drop(tx); // Signal end of input

    // Collect all responses
    let mut responses = Vec::new();
    while let Some(response) = response_stream.next().await {
        assert!(response.is_ok());
        responses.push(response.unwrap());
    }

    assert_eq!(responses.len(), 20, "Should receive response for each document");

    // Verify all documents were processed successfully
    for (i, response) in responses.iter().enumerate() {
        assert!(!response.document_id.is_empty(), "Document {} should have ID", i);
        assert_eq!(response.status, ProcessingStatus::Completed as i32, "Document {} should be completed", i);
        assert!(response.chunks_created > 0, "Document {} should have chunks", i);
    }
}

// ================================
// STREAM LIFECYCLE TESTS
// ================================

#[tokio::test]
async fn test_stream_initialization_and_negotiation() {
    let (_addr, channel) = setup_test_server().await;
    let client = document_processor_client::DocumentProcessorClient::new(channel);

    // Test stream can be created multiple times
    for i in 0..3 {
        let (tx, rx) = mpsc::channel(10);
        let input_stream = ReceiverStream::new(rx);

        let response = client.process_documents(Request::new(input_stream)).await;
        assert!(response.is_ok(), "Stream {} initialization failed", i);

        let mut response_stream = response.unwrap().into_inner();

        // Send one document to verify stream works
        let doc = ProcessDocumentRequest {
            file_path: format!("/test/stream{}/doc.txt", i),
            project_id: "test-project".to_string(),
            collection_name: "test-collection".to_string(),
            document_type: DocumentType::Text as i32,
            metadata: std::collections::HashMap::new(),
            options: Some(ProcessingOptions::default()),
        };

        tx.send(doc).await.unwrap();
        drop(tx);

        // Verify response
        let response = response_stream.next().await;
        assert!(response.is_some(), "Stream {} should receive response", i);
        assert!(response.unwrap().is_ok(), "Stream {} should receive valid response", i);

        // Verify stream ends properly
        let end_response = response_stream.next().await;
        assert!(end_response.is_none(), "Stream {} should end cleanly", i);
    }
}

#[tokio::test]
async fn test_stream_flow_control_mechanisms() {
    let (_addr, channel) = setup_test_server().await;
    let client = document_processor_client::DocumentProcessorClient::new(channel);

    // Test with different buffer sizes to verify flow control
    let buffer_sizes = [1, 5, 10, 50];

    for buffer_size in buffer_sizes {
        let (tx, rx) = mpsc::channel(buffer_size);
        let input_stream = ReceiverStream::new(rx);

        let response = client.process_documents(Request::new(input_stream)).await;
        assert!(response.is_ok());

        let mut response_stream = response.unwrap().into_inner();

        // Send documents up to buffer size + 1 to test flow control
        let send_handle = tokio::spawn(async move {
            for i in 0..=(buffer_size + 1) {
                let doc = ProcessDocumentRequest {
                    file_path: format!("/flow/doc{}.txt", i),
                    project_id: "flow-project".to_string(),
                    collection_name: "flow-collection".to_string(),
                    document_type: DocumentType::Text as i32,
                    metadata: std::collections::HashMap::new(),
                    options: Some(ProcessingOptions::default()),
                };

                if tx.send(doc).await.is_err() {
                    break;
                }

                // Small delay to allow flow control to work
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            drop(tx);
        });

        // Consume responses with controlled timing
        let mut count = 0;
        while let Some(response) = response_stream.next().await {
            assert!(response.is_ok());
            count += 1;

            // Introduce backpressure by slow consumption
            tokio::time::sleep(Duration::from_millis(20)).await;
        }

        send_handle.await.unwrap();
        assert!(count > 0, "Should have received responses with buffer size {}", buffer_size);
    }
}

#[tokio::test]
async fn test_stream_completion_patterns() {
    let (_addr, channel) = setup_test_server().await;
    let client = document_processor_client::DocumentProcessorClient::new(channel);

    // Test 1: Normal completion by closing input
    {
        let (tx, rx) = mpsc::channel(10);
        let input_stream = ReceiverStream::new(rx);

        let response = client.process_documents(Request::new(input_stream)).await;
        assert!(response.is_ok());

        let mut response_stream = response.unwrap().into_inner();

        // Send documents then close input
        for i in 0..3 {
            let doc = ProcessDocumentRequest {
                file_path: format!("/completion/doc{}.txt", i),
                project_id: "completion-project".to_string(),
                collection_name: "completion-collection".to_string(),
                document_type: DocumentType::Text as i32,
                metadata: std::collections::HashMap::new(),
                options: Some(ProcessingOptions::default()),
            };
            tx.send(doc).await.unwrap();
        }
        drop(tx); // Close input stream

        // Collect all responses
        let mut responses = Vec::new();
        while let Some(response) = response_stream.next().await {
            responses.push(response);
        }

        assert_eq!(responses.len(), 3, "Should receive all responses before completion");
        for response in responses {
            assert!(response.is_ok());
        }
    }

    // Test 2: Early termination by dropping response stream
    {
        let (tx, rx) = mpsc::channel(10);
        let input_stream = ReceiverStream::new(rx);

        let response = client.process_documents(Request::new(input_stream)).await;
        assert!(response.is_ok());

        let mut response_stream = response.unwrap().into_inner();

        // Send one document
        let doc = ProcessDocumentRequest {
            file_path: "/early/doc.txt".to_string(),
            project_id: "early-project".to_string(),
            collection_name: "early-collection".to_string(),
            document_type: DocumentType::Text as i32,
            metadata: std::collections::HashMap::new(),
            options: Some(ProcessingOptions::default()),
        };
        tx.send(doc).await.unwrap();

        // Get one response then drop stream (early termination)
        let response = response_stream.next().await;
        assert!(response.is_some());
        assert!(response.unwrap().is_ok());

        drop(response_stream); // Early termination
        drop(tx);

        // Should complete without hanging
    }
}

// ================================
// ERROR HANDLING IN STREAMS
// ================================

#[tokio::test]
async fn test_streaming_error_propagation() {
    let (_addr, channel) = setup_test_server().await;
    let client = document_processor_client::DocumentProcessorClient::new(channel);

    let (tx, rx) = mpsc::channel(10);
    let input_stream = ReceiverStream::new(rx);

    let response = client.process_documents(Request::new(input_stream)).await;
    assert!(response.is_ok());

    let mut response_stream = response.unwrap().into_inner();

    // Send various invalid requests to test error handling
    let invalid_requests = vec![
        ProcessDocumentRequest {
            file_path: "".to_string(), // Empty path
            project_id: "test-project".to_string(),
            collection_name: "test-collection".to_string(),
            document_type: DocumentType::Unspecified as i32,
            metadata: std::collections::HashMap::new(),
            options: None,
        },
        ProcessDocumentRequest {
            file_path: "/test/doc.txt".to_string(),
            project_id: "".to_string(), // Empty project ID
            collection_name: "test-collection".to_string(),
            document_type: DocumentType::Text as i32,
            metadata: std::collections::HashMap::new(),
            options: None,
        },
        ProcessDocumentRequest {
            file_path: "/test/doc.txt".to_string(),
            project_id: "test-project".to_string(),
            collection_name: "".to_string(), // Empty collection name
            document_type: DocumentType::Text as i32,
            metadata: std::collections::HashMap::new(),
            options: None,
        },
    ];

    for (i, invalid_req) in invalid_requests.into_iter().enumerate() {
        tx.send(invalid_req).await.unwrap();

        let response = response_stream.next().await;
        assert!(response.is_some(), "Should receive response for invalid request {}", i);

        match response.unwrap() {
            Ok(resp) => {
                // Error might be returned in response with Failed status
                if resp.status == ProcessingStatus::Failed as i32 {
                    assert!(!resp.error_message.is_empty(), "Error message should not be empty for request {}", i);
                }
            }
            Err(status) => {
                // Or as gRPC status error
                assert!(
                    status.code() == tonic::Code::InvalidArgument ||
                    status.code() == tonic::Code::FailedPrecondition,
                    "Unexpected error code for request {}: {:?}", i, status.code()
                );
            }
        }
    }

    drop(tx);
}

#[tokio::test]
async fn test_streaming_partial_failure_scenarios() {
    let (_addr, channel) = setup_test_server().await;
    let client = document_processor_client::DocumentProcessorClient::new(channel);

    let (tx, rx) = mpsc::channel(10);
    let input_stream = ReceiverStream::new(rx);

    let response = client.process_documents(Request::new(input_stream)).await;
    assert!(response.is_ok());

    let mut response_stream = response.unwrap().into_inner();

    // Mix of valid and invalid documents
    let requests = vec![
        // Valid document
        ProcessDocumentRequest {
            file_path: "/test/valid1.txt".to_string(),
            project_id: "test-project".to_string(),
            collection_name: "test-collection".to_string(),
            document_type: DocumentType::Text as i32,
            metadata: std::collections::HashMap::new(),
            options: Some(ProcessingOptions::default()),
        },
        // Invalid document
        ProcessDocumentRequest {
            file_path: "".to_string(),
            project_id: "test-project".to_string(),
            collection_name: "test-collection".to_string(),
            document_type: DocumentType::Text as i32,
            metadata: std::collections::HashMap::new(),
            options: Some(ProcessingOptions::default()),
        },
        // Valid document
        ProcessDocumentRequest {
            file_path: "/test/valid2.txt".to_string(),
            project_id: "test-project".to_string(),
            collection_name: "test-collection".to_string(),
            document_type: DocumentType::Text as i32,
            metadata: std::collections::HashMap::new(),
            options: Some(ProcessingOptions::default()),
        },
    ];

    for req in requests {
        tx.send(req).await.unwrap();
    }
    drop(tx);

    // Collect all responses
    let mut responses = Vec::new();
    while let Some(response) = response_stream.next().await {
        responses.push(response);
    }

    assert_eq!(responses.len(), 3, "Should receive response for each request");

    // Verify mixed success/failure pattern
    let mut success_count = 0;
    let mut failure_count = 0;

    for (i, response) in responses.into_iter().enumerate() {
        match response {
            Ok(resp) => {
                if resp.status == ProcessingStatus::Completed as i32 {
                    success_count += 1;
                } else if resp.status == ProcessingStatus::Failed as i32 {
                    failure_count += 1;
                    assert!(!resp.error_message.is_empty(), "Failed response {} should have error message", i);
                }
            }
            Err(_) => {
                failure_count += 1;
            }
        }
    }

    assert!(success_count >= 2, "Should have at least 2 successful responses");
    assert!(failure_count >= 1, "Should have at least 1 failed response");
}

// ================================
// COVERAGE VALIDATION TESTS
// ================================

#[tokio::test]
async fn test_comprehensive_streaming_coverage() {
    // This test ensures we're exercising all major streaming code paths
    let (_addr, channel) = setup_test_server().await;
    let client = document_processor_client::DocumentProcessorClient::new(channel);

    // Test multiple concurrent streams
    let mut handles = Vec::new();

    for stream_id in 0..3 {
        let mut client_clone = client.clone();
        let handle = tokio::spawn(async move {
            let (tx, rx) = mpsc::channel(5);
            let input_stream = ReceiverStream::new(rx);

            let response = client_clone.process_documents(Request::new(input_stream)).await;
            assert!(response.is_ok(), "Stream {} failed to start", stream_id);

            let mut response_stream = response.unwrap().into_inner();

            // Send documents
            for doc_id in 0..5 {
                let doc = ProcessDocumentRequest {
                    file_path: format!("/concurrent/stream{}/doc{}.txt", stream_id, doc_id),
                    project_id: format!("project-{}", stream_id),
                    collection_name: format!("collection-{}", stream_id),
                    document_type: DocumentType::Text as i32,
                    metadata: {
                        let mut meta = std::collections::HashMap::new();
                        meta.insert("stream_id".to_string(), stream_id.to_string());
                        meta.insert("doc_id".to_string(), doc_id.to_string());
                        meta
                    },
                    options: Some(ProcessingOptions::default()),
                };

                tx.send(doc).await.unwrap();
            }
            drop(tx);

            // Collect responses
            let mut count = 0;
            while let Some(response) = response_stream.next().await {
                assert!(response.is_ok(), "Stream {} received error", stream_id);
                count += 1;
            }

            assert_eq!(count, 5, "Stream {} should receive 5 responses", stream_id);
            stream_id
        });

        handles.push(handle);
    }

    // Wait for all concurrent streams to complete
    for handle in handles {
        let stream_id = handle.await.unwrap();
        debug!("Completed concurrent stream {}", stream_id);
    }
}

// ================================
// PERFORMANCE AND STRESS TESTS
// ================================

#[tokio::test]
async fn test_streaming_performance_characteristics() {
    let (_addr, channel) = setup_test_server().await;
    let client = document_processor_client::DocumentProcessorClient::new(channel);

    let start_time = std::time::Instant::now();

    let (tx, rx) = mpsc::channel(100);
    let input_stream = ReceiverStream::new(rx);

    let response = client.process_documents(Request::new(input_stream)).await;
    assert!(response.is_ok());

    let mut response_stream = response.unwrap().into_inner();

    // Send 100 documents to test throughput
    let send_handle = tokio::spawn(async move {
        for i in 0..100 {
            let doc = ProcessDocumentRequest {
                file_path: format!("/perf/doc{}.txt", i),
                project_id: "perf-project".to_string(),
                collection_name: "perf-collection".to_string(),
                document_type: DocumentType::Text as i32,
                metadata: std::collections::HashMap::new(),
                options: Some(ProcessingOptions {
                    enable_lsp_analysis: false,
                    chunk_size: 100, // Small chunks for speed
                    chunk_overlap: 20,
                    extract_metadata: false,
                    detect_language: false,
                    custom_parsers: vec![],
                }),
            };

            tx.send(doc).await.unwrap();
        }
        drop(tx);
    });

    // Process responses and measure time
    let mut count = 0;
    while let Some(response) = response_stream.next().await {
        assert!(response.is_ok());
        count += 1;
    }

    send_handle.await.unwrap();

    let duration = start_time.elapsed();
    assert_eq!(count, 100, "Should process all 100 documents");

    // Performance check - should process at least 10 docs/second
    let throughput = count as f64 / duration.as_secs_f64();
    assert!(throughput >= 10.0, "Throughput too low: {:.2} docs/sec", throughput);

    info!("Streaming performance: {:.2} docs/sec", throughput);
}

// ================================
// INTEGRATION HELPER TESTS
// ================================

#[test]
fn test_stream_traits_and_types() {
    // Verify streaming types implement required traits
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}
    fn assert_stream<T: Stream>(_: T) {}

    // Test that our streaming types have proper trait bounds
    assert_send::<ReceiverStream<ProcessDocumentRequest>>();
    assert_sync::<ProcessDocumentResponse>();

    // Verify we can create streams
    let (tx, rx) = mpsc::channel::<ProcessDocumentRequest>(10);
    let stream = ReceiverStream::new(rx);
    assert_stream(stream);

    drop(tx);
}

#[tokio::test]
async fn test_streaming_with_different_chunk_sizes() {
    let (_addr, channel) = setup_test_server().await;
    let client = document_processor_client::DocumentProcessorClient::new(channel);

    // Test different chunk sizes to verify processing options
    let chunk_sizes = [100, 500, 1000, 2000];

    for &chunk_size in &chunk_sizes {
        let (tx, rx) = mpsc::channel(10);
        let input_stream = ReceiverStream::new(rx);

        let response = client.process_documents(Request::new(input_stream)).await;
        assert!(response.is_ok());

        let mut response_stream = response.unwrap().into_inner();

        let doc = ProcessDocumentRequest {
            file_path: format!("/chunks/doc_{}.txt", chunk_size),
            project_id: "chunk-project".to_string(),
            collection_name: "chunk-collection".to_string(),
            document_type: DocumentType::Text as i32,
            metadata: std::collections::HashMap::new(),
            options: Some(ProcessingOptions {
                enable_lsp_analysis: false,
                chunk_size,
                chunk_overlap: chunk_size / 10, // 10% overlap
                extract_metadata: true,
                detect_language: true,
                custom_parsers: vec![],
            }),
        };

        tx.send(doc).await.unwrap();
        drop(tx);

        let response = response_stream.next().await;
        assert!(response.is_some(), "No response for chunk size {}", chunk_size);

        let response = response.unwrap();
        assert!(response.is_ok(), "Error for chunk size {}", chunk_size);

        let response = response.unwrap();
        assert!(response.chunks_created > 0, "No chunks for size {}", chunk_size);

        debug!("Chunk size {} created {} chunks", chunk_size, response.chunks_created);
    }
}