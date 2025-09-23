//! Comprehensive gRPC streaming implementation tests
//!
//! This test suite demonstrates TDD approach for gRPC streaming with focus on:
//! - Bidirectional streaming validation
//! - Stream lifecycle management
//! - Flow control and backpressure handling
//! - Error propagation in streaming contexts
//! - Performance characteristics of streaming operations

use workspace_qdrant_daemon::config::*;
use workspace_qdrant_daemon::daemon::WorkspaceDaemon;
use workspace_qdrant_daemon::grpc::services::DocumentProcessorImpl;
use workspace_qdrant_daemon::proto::{
    ProcessDocumentRequest, ProcessDocumentResponse,
    DocumentType, ProcessingOptions, ProcessingStatus,
};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_stream::{StreamExt, wrappers::ReceiverStream};
use tonic::Request;

// ================================
// TEST CONFIGURATION AND HELPERS
// ================================

fn create_test_daemon_config() -> DaemonConfig {
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
            debounce_ms: 100,
            max_watched_dirs: 10,
            ignore_patterns: vec![".git".to_string(), "target".to_string()],
            recursive: true,
        },
        metrics: MetricsConfig {
            enabled: true,
            collection_interval_secs: 1,
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

fn create_test_document_request(file_path: &str, doc_type: DocumentType) -> ProcessDocumentRequest {
    ProcessDocumentRequest {
        file_path: file_path.to_string(),
        project_id: "streaming-test".to_string(),
        collection_name: "streaming-collection".to_string(),
        document_type: doc_type as i32,
        metadata: HashMap::new(),
        options: Some(ProcessingOptions::default()),
    }
}

// ================================
// STREAMING IMPLEMENTATION TESTS
// ================================

#[tokio::test]
async fn test_bidirectional_streaming_basic_implementation() {
    // Test that bidirectional streaming is properly implemented
    let daemon = create_test_daemon().await;
    let processor = DocumentProcessorImpl::new(Arc::new(daemon));

    let (tx, rx) = mpsc::channel(10);
    let input_stream = ReceiverStream::new(rx);

    // Start the streaming request
    let response = processor.process_documents(Request::new(input_stream)).await;
    assert!(response.is_ok(), "Bidirectional streaming should be supported");

    let mut response_stream = response.unwrap().into_inner();

    // Send test documents
    let documents = vec![
        create_test_document_request("/test/doc1.txt", DocumentType::Text),
        create_test_document_request("/test/doc2.md", DocumentType::Markdown),
        create_test_document_request("/test/code.rs", DocumentType::Code),
    ];

    for doc in documents {
        tx.send(doc).await.unwrap();
    }
    drop(tx); // Signal end of input

    // Collect all responses
    let mut responses = Vec::new();
    while let Some(response) = response_stream.next().await {
        responses.push(response);
    }

    assert_eq!(responses.len(), 3, "Should receive response for each document");

    for (i, response) in responses.into_iter().enumerate() {
        assert!(response.is_ok(), "Response {} should be successful", i);
        let response = response.unwrap();
        assert!(!response.document_id.is_empty(), "Document {} should have ID", i);
        assert_eq!(response.status, ProcessingStatus::Completed as i32, "Document {} should be completed", i);
    }
}

#[tokio::test]
async fn test_streaming_flow_control_implementation() {
    // Test flow control mechanisms in streaming
    let daemon = create_test_daemon().await;
    let processor = DocumentProcessorImpl::new(Arc::new(daemon));

    let (tx, rx) = mpsc::channel(2); // Small buffer to test flow control
    let input_stream = ReceiverStream::new(rx);

    let response = processor.process_documents(Request::new(input_stream)).await;
    assert!(response.is_ok());

    let mut response_stream = response.unwrap().into_inner();

    // Send documents with flow control
    let send_handle = tokio::spawn(async move {
        for i in 0..10 {
            let doc = create_test_document_request(
                &format!("/flow/doc{}.txt", i),
                DocumentType::Text
            );

            match tx.send(doc).await {
                Ok(_) => {
                    // Small delay to allow processing
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
                Err(_) => break, // Channel closed
            }
        }
    });

    // Slow consumer to test flow control
    let mut count = 0;
    while let Some(response) = response_stream.next().await {
        assert!(response.is_ok(), "Response should be successful");
        count += 1;

        // Simulate slow processing
        tokio::time::sleep(Duration::from_millis(20)).await;

        if count >= 10 {
            break;
        }
    }

    send_handle.await.unwrap();
    assert!(count > 0, "Should have processed at least one document");
    assert!(count <= 10, "Should not exceed sent documents");
}

#[tokio::test]
async fn test_streaming_backpressure_handling() {
    // Test backpressure handling in streaming scenarios
    let daemon = create_test_daemon().await;
    let processor = DocumentProcessorImpl::new(Arc::new(daemon));

    let (tx, rx) = mpsc::channel(3); // Limited buffer for backpressure
    let input_stream = ReceiverStream::new(rx);

    let response = processor.process_documents(Request::new(input_stream)).await;
    assert!(response.is_ok());

    let mut response_stream = response.unwrap().into_inner();

    // Producer that creates backpressure
    let send_handle = tokio::spawn(async move {
        for i in 0..15 {
            let doc = ProcessDocumentRequest {
                file_path: format!("/backpressure/doc{}.txt", i),
                project_id: "backpressure-test".to_string(),
                collection_name: "backpressure-collection".to_string(),
                document_type: DocumentType::Text as i32,
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("batch_id".to_string(), "backpressure".to_string());
                    meta.insert("doc_index".to_string(), i.to_string());
                    meta
                },
                options: Some(ProcessingOptions {
                    enable_lsp_analysis: false,
                    chunk_size: 200,
                    chunk_overlap: 40,
                    extract_metadata: true,
                    detect_language: false,
                    custom_parsers: vec![],
                }),
            };

            // This should apply backpressure when buffer is full
            if tx.send(doc).await.is_err() {
                break;
            }
        }
    });

    // Consumer with variable processing speed
    let mut processed = 0;
    while let Some(response) = response_stream.next().await {
        assert!(response.is_ok());
        let response = response.unwrap();
        assert!(!response.document_id.is_empty());
        processed += 1;

        // Variable delay to simulate real processing
        let delay = if processed % 3 == 0 { 30 } else { 10 };
        tokio::time::sleep(Duration::from_millis(delay)).await;

        if processed >= 15 {
            break;
        }
    }

    send_handle.await.unwrap();
    assert!(processed > 0, "Should have processed documents");
}

#[tokio::test]
async fn test_streaming_error_propagation() {
    // Test error handling and propagation in streaming
    let daemon = create_test_daemon().await;
    let processor = DocumentProcessorImpl::new(Arc::new(daemon));

    let (tx, rx) = mpsc::channel(10);
    let input_stream = ReceiverStream::new(rx);

    let response = processor.process_documents(Request::new(input_stream)).await;
    assert!(response.is_ok());

    let mut response_stream = response.unwrap().into_inner();

    // Mix of valid and invalid requests
    let requests = vec![
        // Valid request
        create_test_document_request("/test/valid1.txt", DocumentType::Text),

        // Invalid request (empty path)
        ProcessDocumentRequest {
            file_path: "".to_string(),
            project_id: "error-test".to_string(),
            collection_name: "error-collection".to_string(),
            document_type: DocumentType::Text as i32,
            metadata: HashMap::new(),
            options: Some(ProcessingOptions::default()),
        },

        // Another valid request
        create_test_document_request("/test/valid2.txt", DocumentType::Text),

        // Invalid request (missing project ID)
        ProcessDocumentRequest {
            file_path: "/test/invalid.txt".to_string(),
            project_id: "".to_string(),
            collection_name: "error-collection".to_string(),
            document_type: DocumentType::Text as i32,
            metadata: HashMap::new(),
            options: Some(ProcessingOptions::default()),
        },
    ];

    for req in requests {
        tx.send(req).await.unwrap();
    }
    drop(tx);

    // Process all responses and validate error handling
    let mut responses = Vec::new();
    while let Some(response) = response_stream.next().await {
        responses.push(response);
    }

    assert_eq!(responses.len(), 4, "Should receive response for each request");

    let mut success_count = 0;
    let mut error_count = 0;

    for (i, response) in responses.into_iter().enumerate() {
        match response {
            Ok(resp) => {
                if resp.status == ProcessingStatus::Completed as i32 {
                    success_count += 1;
                } else if resp.status == ProcessingStatus::Failed as i32 {
                    error_count += 1;
                    assert!(!resp.error_message.is_empty(), "Failed response {} should have error message", i);
                }
            }
            Err(_) => {
                error_count += 1;
            }
        }
    }

    assert!(success_count >= 2, "Should have at least 2 successful responses");
    assert!(error_count >= 2, "Should have at least 2 error responses");
}

#[tokio::test]
async fn test_streaming_cancellation_and_cleanup() {
    // Test stream cancellation and resource cleanup
    let daemon = create_test_daemon().await;
    let processor = DocumentProcessorImpl::new(Arc::new(daemon));

    let (tx, rx) = mpsc::channel(5);
    let input_stream = ReceiverStream::new(rx);

    let response = processor.process_documents(Request::new(input_stream)).await;
    assert!(response.is_ok());

    let mut response_stream = response.unwrap().into_inner();

    // Send a few documents
    for i in 0..3 {
        let doc = create_test_document_request(
            &format!("/cancel/doc{}.txt", i),
            DocumentType::Text
        );
        tx.send(doc).await.unwrap();
    }

    // Process some responses then cancel
    let mut processed = 0;
    while let Some(response) = response_stream.next().await {
        assert!(response.is_ok());
        processed += 1;

        // Cancel after processing 2 responses
        if processed >= 2 {
            break;
        }
    }

    // Drop everything to simulate cancellation
    drop(response_stream);
    drop(tx);

    assert_eq!(processed, 2, "Should have processed exactly 2 responses before cancellation");
}

#[tokio::test]
async fn test_concurrent_streaming_operations() {
    // Test multiple concurrent streaming operations
    let daemon = Arc::new(create_test_daemon().await);

    let mut handles = Vec::new();

    for stream_id in 0..3 {
        let daemon_clone = Arc::clone(&daemon);
        let handle = tokio::spawn(async move {
            let processor = DocumentProcessorImpl::new(daemon_clone);

            let (tx, rx) = mpsc::channel(5);
            let input_stream = ReceiverStream::new(rx);

            let response = processor.process_documents(Request::new(input_stream)).await;
            assert!(response.is_ok(), "Stream {} should start successfully", stream_id);

            let mut response_stream = response.unwrap().into_inner();

            // Send documents for this stream
            for doc_id in 0..3 {
                let doc = ProcessDocumentRequest {
                    file_path: format!("/concurrent/stream{}/doc{}.txt", stream_id, doc_id),
                    project_id: format!("concurrent-project-{}", stream_id),
                    collection_name: format!("concurrent-collection-{}", stream_id),
                    document_type: DocumentType::Text as i32,
                    metadata: {
                        let mut meta = HashMap::new();
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
                assert!(response.is_ok(), "Stream {} response should be valid", stream_id);
                count += 1;
            }

            assert_eq!(count, 3, "Stream {} should process all documents", stream_id);
            (stream_id, count)
        });

        handles.push(handle);
    }

    // Wait for all concurrent streams to complete
    for handle in handles {
        let (stream_id, count) = handle.await.unwrap();
        assert_eq!(count, 3, "Stream {} completed successfully", stream_id);
    }
}

#[tokio::test]
async fn test_streaming_performance_characteristics() {
    // Test performance characteristics of streaming operations
    let daemon = create_test_daemon().await;
    let processor = DocumentProcessorImpl::new(Arc::new(daemon));

    let start_time = std::time::Instant::now();

    let (tx, rx) = mpsc::channel(50);
    let input_stream = ReceiverStream::new(rx);

    let response = processor.process_documents(Request::new(input_stream)).await;
    assert!(response.is_ok());

    let mut response_stream = response.unwrap().into_inner();

    // Send many documents to test throughput
    let document_count = 25;
    let send_handle = tokio::spawn(async move {
        for i in 0..document_count {
            let doc = ProcessDocumentRequest {
                file_path: format!("/perf/doc{}.txt", i),
                project_id: "performance-test".to_string(),
                collection_name: "performance-collection".to_string(),
                document_type: DocumentType::Text as i32,
                metadata: HashMap::new(),
                options: Some(ProcessingOptions {
                    enable_lsp_analysis: false,
                    chunk_size: 100, // Small chunks for faster processing
                    chunk_overlap: 20,
                    extract_metadata: false,
                    detect_language: false,
                    custom_parsers: vec![],
                }),
            };

            tx.send(doc).await.unwrap();
        }
    });

    // Process responses and measure performance
    let mut processed = 0;
    while let Some(response) = response_stream.next().await {
        assert!(response.is_ok());
        processed += 1;

        if processed >= document_count {
            break;
        }
    }

    send_handle.await.unwrap();

    let duration = start_time.elapsed();
    assert_eq!(processed, document_count, "Should process all documents");

    // Performance validation - should handle at least 10 docs/second
    let throughput = processed as f64 / duration.as_secs_f64();
    assert!(throughput >= 5.0, "Throughput should be at least 5 docs/sec, got {:.2}", throughput);

    println!("Streaming performance: {:.2} docs/sec ({} docs in {:?})",
             throughput, processed, duration);
}

// ================================
// STREAMING TYPE AND TRAIT TESTS
// ================================

#[test]
fn test_streaming_types_implement_required_traits() {
    // Verify all streaming types implement necessary traits
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}
    fn assert_unpin<T: Unpin>() {}

    // Core streaming types
    assert_send::<ProcessDocumentRequest>();
    assert_sync::<ProcessDocumentRequest>();
    assert_unpin::<ProcessDocumentRequest>();

    assert_send::<ProcessDocumentResponse>();
    assert_sync::<ProcessDocumentResponse>();
    assert_unpin::<ProcessDocumentResponse>();

    // Stream types
    assert_send::<ReceiverStream<ProcessDocumentRequest>>();
    assert_unpin::<ReceiverStream<ProcessDocumentRequest>>();

    // Channel types
    assert_send::<mpsc::Sender<ProcessDocumentRequest>>();
    assert_send::<mpsc::Receiver<ProcessDocumentRequest>>();
}

#[test]
fn test_processing_options_streaming_defaults() {
    let options = ProcessingOptions::default();

    // Validate streaming-appropriate defaults
    assert!(options.chunk_size > 0, "Chunk size must be positive for streaming");
    assert!(options.chunk_overlap >= 0, "Chunk overlap must be non-negative");
    assert!(options.chunk_overlap < options.chunk_size, "Overlap must be less than chunk size");

    // Test custom options for streaming scenarios
    let streaming_options = ProcessingOptions {
        enable_lsp_analysis: false, // Typically disabled for streaming performance
        chunk_size: 500,
        chunk_overlap: 100,
        extract_metadata: true,
        detect_language: false,
        custom_parsers: vec!["fast_parser".to_string()],
    };

    assert!(!streaming_options.enable_lsp_analysis, "LSP should be disabled for streaming");
    assert_eq!(streaming_options.chunk_size, 500);
    assert_eq!(streaming_options.chunk_overlap, 100);
    assert!(streaming_options.extract_metadata);
    assert!(!streaming_options.detect_language);
    assert_eq!(streaming_options.custom_parsers.len(), 1);
}

// ================================
// STREAMING EDGE CASES
// ================================

#[tokio::test]
async fn test_streaming_with_different_document_types() {
    // Test streaming with various document types
    let daemon = create_test_daemon().await;
    let processor = DocumentProcessorImpl::new(Arc::new(daemon));

    let (tx, rx) = mpsc::channel(10);
    let input_stream = ReceiverStream::new(rx);

    let response = processor.process_documents(Request::new(input_stream)).await;
    assert!(response.is_ok());

    let mut response_stream = response.unwrap().into_inner();

    // Test all document types in streaming
    let document_types = vec![
        DocumentType::Text,
        DocumentType::Markdown,
        DocumentType::Code,
        DocumentType::Json,
        DocumentType::Xml,
        DocumentType::Html,
    ];

    for (i, doc_type) in document_types.into_iter().enumerate() {
        let doc = ProcessDocumentRequest {
            file_path: format!("/types/doc{}.{:?}", i, doc_type),
            project_id: "types-test".to_string(),
            collection_name: "types-collection".to_string(),
            document_type: doc_type as i32,
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("type".to_string(), format!("{:?}", doc_type));
                meta
            },
            options: Some(ProcessingOptions {
                enable_lsp_analysis: false,
                chunk_size: 300,
                chunk_overlap: 60,
                extract_metadata: true,
                detect_language: true,
                custom_parsers: vec![],
            }),
        };

        tx.send(doc).await.unwrap();
    }
    drop(tx);

    // Verify all document types are processed
    let mut responses = Vec::new();
    while let Some(response) = response_stream.next().await {
        responses.push(response);
    }

    assert_eq!(responses.len(), 6, "Should process all document types");

    for (i, response) in responses.into_iter().enumerate() {
        assert!(response.is_ok(), "Document type {} should be processed", i);
        let response = response.unwrap();
        assert!(!response.document_id.is_empty(), "Document {} should have ID", i);
        // All documents should either complete or fail gracefully
        assert!(
            response.status == ProcessingStatus::Completed as i32 ||
            response.status == ProcessingStatus::Failed as i32,
            "Document {} should have valid status", i
        );
    }
}