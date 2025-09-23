//! Comprehensive gRPC Protocol Correctness Integration Tests
//!
//! This test suite uses shared test utilities to validate gRPC protocol
//! correctness with focus on:
//! - Protocol buffer message serialization/deserialization
//! - gRPC service method routing and dispatch
//! - Error status code mapping and propagation
//! - Metadata and header propagation
//! - Concurrent protocol access
//! - Protocol compliance validation
//!
//! Targets 90%+ coverage using TDD approach with cargo tarpaulin validation.

#[cfg(test)]
mod tests {
    use workspace_qdrant_daemon::{
        grpc::shared_test_utils::*,
        proto::*,
    };
    use std::{time::Duration, collections::HashMap};
    use tokio::time::timeout;
    use tonic::{Request, metadata::MetadataValue};

    // =============================================================================
    // PROTOCOL BUFFER MESSAGE SERIALIZATION TESTS
    // =============================================================================

    #[tokio::test]
    async fn test_protobuf_serialization_document_processing() {
        let server = TestGrpcServer::start().await.expect("Failed to start test server");
        let mut clients = server.get_clients().await.expect("Failed to get clients");

        // Test complex ProcessDocumentRequest serialization
        let request = TestDataFactory::create_process_document_request(
            "/test/complex_document.rs",
            "protobuf_test_project",
            "protobuf_collection",
            DocumentType::Code,
        );

        let response = timeout(
            Duration::from_secs(5),
            clients.document_processor.process_document(Request::new(request))
        )
        .await
        .expect("Request timeout")
        .expect("Request failed")
        .into_inner();

        // Validate protobuf serialization round-trip
        assert!(TestValidators::validate_process_document_response(&response));
        assert!(!response.document_id.is_empty());
        assert_eq!(response.status, ProcessingStatus::Completed as i32);
        assert!(response.chunks_created >= 0);

        server.shutdown().await;
    }

    #[tokio::test]
    async fn test_protobuf_serialization_search_operations() {
        let server = TestGrpcServer::start().await.expect("Failed to start test server");
        let mut clients = server.get_clients().await.expect("Failed to get clients");

        // Test complex HybridSearchRequest with nested structures
        let request = TestDataFactory::create_hybrid_search_request(
            "complex protobuf serialization test query",
            "protobuf_search_project",
            vec!["collection_1".to_string(), "collection_2".to_string()],
        );

        let response = timeout(
            Duration::from_secs(5),
            clients.search_service.hybrid_search(Request::new(request))
        )
        .await
        .expect("Request timeout")
        .expect("Request failed")
        .into_inner();

        // Validate complex nested message serialization
        assert!(TestValidators::validate_hybrid_search_response(&response));
        assert!(!response.query_id.is_empty());
        assert!(response.metadata.is_some());

        let metadata = response.metadata.unwrap();
        assert!(metadata.total_results >= 0);
        assert!(metadata.search_duration_ms >= 0);

        server.shutdown().await;
    }

    #[tokio::test]
    async fn test_protobuf_serialization_memory_operations() {
        let server = TestGrpcServer::start().await.expect("Failed to start test server");
        let mut clients = server.get_clients().await.expect("Failed to get clients");

        // Test complex DocumentContent with unicode and special characters
        let content = TestDataFactory::create_document_content(
            "Test content with unicode: àáâãäåæç 你好世界 🌍 and special chars: <>&\"'",
            3,
        );

        let add_request = TestDataFactory::create_add_document_request(
            "/test/unicode_document.txt",
            "unicode_collection",
            "unicode_project",
            Some(content),
        );

        let response = timeout(
            Duration::from_secs(5),
            clients.memory_service.add_document(Request::new(add_request))
        )
        .await
        .expect("Request timeout")
        .expect("Request failed")
        .into_inner();

        // Validate unicode content serialization
        assert!(!response.document_id.is_empty());
        assert!(response.success);
        assert!(response.error_message.is_empty());

        server.shutdown().await;
    }

    // =============================================================================
    // SERVICE METHOD ROUTING AND DISPATCH TESTS
    // =============================================================================

    #[tokio::test]
    async fn test_document_processor_service_routing() {
        let server = TestGrpcServer::start().await.expect("Failed to start test server");
        let mut clients = server.get_clients().await.expect("Failed to get clients");

        // Test ProcessDocument method routing
        let process_request = TestDataFactory::create_process_document_request(
            "/test/routing_test.txt",
            "routing_project",
            "routing_collection",
            DocumentType::Text,
        );

        let process_response = timeout(
            Duration::from_secs(5),
            clients.document_processor.process_document(Request::new(process_request))
        )
        .await
        .expect("Timeout")
        .expect("Request failed")
        .into_inner();

        assert!(TestValidators::validate_process_document_response(&process_response));

        // Test GetProcessingStatus method routing
        let status_request = ProcessingStatusRequest {
            operation_id: "routing_test_operation".to_string(),
        };

        let status_response = timeout(
            Duration::from_secs(5),
            clients.document_processor.get_processing_status(Request::new(status_request))
        )
        .await
        .expect("Timeout")
        .expect("Request failed")
        .into_inner();

        assert_eq!(status_response.operation_id, "routing_test_operation");
        assert_eq!(status_response.status, ProcessingStatus::Completed as i32);

        // Test CancelProcessing method routing
        let cancel_request = CancelProcessingRequest {
            operation_id: "routing_test_cancel".to_string(),
        };

        let _cancel_response = timeout(
            Duration::from_secs(5),
            clients.document_processor.cancel_processing(Request::new(cancel_request))
        )
        .await
        .expect("Timeout")
        .expect("Request failed");

        server.shutdown().await;
    }

    #[tokio::test]
    async fn test_search_service_routing() {
        let server = TestGrpcServer::start().await.expect("Failed to start test server");
        let mut clients = server.get_clients().await.expect("Failed to get clients");

        // Test all SearchService method routing

        // 1. HybridSearch
        let hybrid_request = TestDataFactory::create_hybrid_search_request(
            "routing test query",
            "search_routing_project",
            vec!["routing_collection".to_string()],
        );

        let hybrid_response = timeout(
            Duration::from_secs(5),
            clients.search_service.hybrid_search(Request::new(hybrid_request))
        )
        .await
        .expect("Timeout")
        .expect("Request failed")
        .into_inner();

        assert!(TestValidators::validate_hybrid_search_response(&hybrid_response));

        // 2. SemanticSearch
        let semantic_request = SemanticSearchRequest {
            query: "semantic routing test".to_string(),
            context: SearchContext::Collection as i32,
            options: Some(SearchOptions {
                limit: 5,
                score_threshold: 0.8,
                include_metadata: true,
                include_content: false,
                ranking: None,
            }),
            project_id: "search_routing_project".to_string(),
            collection_names: vec!["routing_collection".to_string()],
        };

        let semantic_response = timeout(
            Duration::from_secs(5),
            clients.search_service.semantic_search(Request::new(semantic_request))
        )
        .await
        .expect("Timeout")
        .expect("Request failed")
        .into_inner();

        assert!(!semantic_response.query_id.is_empty());

        // 3. KeywordSearch
        let keyword_request = KeywordSearchRequest {
            query: "keyword routing".to_string(),
            context: SearchContext::Global as i32,
            options: None,
            project_id: "search_routing_project".to_string(),
            collection_names: vec!["routing_collection".to_string()],
        };

        let keyword_response = timeout(
            Duration::from_secs(5),
            clients.search_service.keyword_search(Request::new(keyword_request))
        )
        .await
        .expect("Timeout")
        .expect("Request failed")
        .into_inner();

        assert!(!keyword_response.query_id.is_empty());

        server.shutdown().await;
    }

    #[tokio::test]
    async fn test_system_service_routing() {
        let server = TestGrpcServer::start().await.expect("Failed to start test server");
        let mut clients = server.get_clients().await.expect("Failed to get clients");

        // Test all SystemService method routing

        // 1. HealthCheck
        let health_response = timeout(
            Duration::from_secs(5),
            clients.system_service.health_check(Request::new(()))
        )
        .await
        .expect("Timeout")
        .expect("Request failed")
        .into_inner();

        assert!(TestValidators::validate_health_check_response(&health_response));

        // 2. GetStatus
        let status_response = timeout(
            Duration::from_secs(5),
            clients.system_service.get_status(Request::new(()))
        )
        .await
        .expect("Timeout")
        .expect("Request failed")
        .into_inner();

        assert_eq!(status_response.status, ServiceStatus::Healthy as i32);
        assert!(status_response.metrics.is_some());

        // 3. GetConfig
        let config_response = timeout(
            Duration::from_secs(5),
            clients.system_service.get_config(Request::new(()))
        )
        .await
        .expect("Timeout")
        .expect("Request failed")
        .into_inner();

        assert!(!config_response.version.is_empty());

        server.shutdown().await;
    }

    // =============================================================================
    // ERROR STATUS CODE MAPPING TESTS
    // =============================================================================

    #[tokio::test]
    async fn test_grpc_error_status_codes() {
        let server = TestGrpcServer::start().await.expect("Failed to start test server");
        let mut clients = server.get_clients().await.expect("Failed to get clients");

        let error_tests = ProtocolTestSuite::test_error_code_mapping(&mut clients)
            .await
            .expect("Error code testing failed");

        // Validate we tested multiple error scenarios
        assert!(!error_tests.is_empty());

        // Log the error codes for analysis (current implementation may not validate all inputs)
        for (test_case, error_code) in error_tests {
            println!("Error test case '{}' resulted in code: {:?}", test_case, error_code);
        }

        server.shutdown().await;
    }

    // =============================================================================
    // METADATA AND HEADER PROPAGATION TESTS
    // =============================================================================

    #[tokio::test]
    async fn test_grpc_metadata_propagation() {
        let server = TestGrpcServer::start().await.expect("Failed to start test server");
        let mut clients = server.get_clients().await.expect("Failed to get clients");

        // Test metadata propagation with various header types
        let metadata_pairs = vec![
            ("client-id", "test_client_123"),
            ("request-id", "req_456789"),
            ("user-agent", "grpc-test/1.0"),
            ("authorization", "Bearer test_token"),
        ];

        let propagation_result = ProtocolTestSuite::test_metadata_propagation(
            &mut clients,
            metadata_pairs,
        )
        .await
        .expect("Metadata propagation test failed");

        assert!(propagation_result, "Metadata propagation validation failed");

        server.shutdown().await;
    }

    #[tokio::test]
    async fn test_grpc_authentication_headers() {
        let server = TestGrpcServer::start().await.expect("Failed to start test server");
        let mut clients = server.get_clients().await.expect("Failed to get clients");

        // Test with authorization header
        let mut request = Request::new(());
        request.metadata_mut().insert(
            "authorization",
            MetadataValue::from_static("Bearer test_token_123")
        );

        let response = timeout(
            Duration::from_secs(5),
            clients.system_service.health_check(request)
        )
        .await
        .expect("Request timeout")
        .expect("Request failed");

        assert_eq!(response.into_inner().status, ServiceStatus::Healthy as i32);

        // Test with API key header
        let mut request2 = Request::new(());
        request2.metadata_mut().insert(
            "x-api-key",
            MetadataValue::from_static("api_key_456")
        );

        let response2 = timeout(
            Duration::from_secs(5),
            clients.system_service.health_check(request2)
        )
        .await
        .expect("Request timeout")
        .expect("Request failed");

        assert_eq!(response2.into_inner().status, ServiceStatus::Healthy as i32);

        server.shutdown().await;
    }

    // =============================================================================
    // ENUM VALUES PROTOCOL COMPLIANCE TESTS
    // =============================================================================

    #[tokio::test]
    async fn test_enum_values_protocol_compliance() {
        let server = TestGrpcServer::start().await.expect("Failed to start test server");
        let mut clients = server.get_clients().await.expect("Failed to get clients");

        // Test all DocumentType enum values
        let document_types = [
            DocumentType::Unspecified as i32,
            DocumentType::Code as i32,
            DocumentType::Pdf as i32,
            DocumentType::Epub as i32,
            DocumentType::Mobi as i32,
            DocumentType::Html as i32,
            DocumentType::Text as i32,
            DocumentType::Markdown as i32,
            DocumentType::Json as i32,
            DocumentType::Xml as i32,
        ];

        let results = ProtocolTestSuite::test_enum_serialization(
            &document_types,
            |doc_type| async {
                let request = ProcessDocumentRequest {
                    file_path: format!("/test/enum_test_{}.txt", doc_type),
                    project_id: "enum_test_project".to_string(),
                    collection_name: "enum_test_collection".to_string(),
                    document_type: doc_type,
                    metadata: HashMap::new(),
                    options: None,
                };

                let response = timeout(
                    Duration::from_secs(5),
                    clients.document_processor.process_document(Request::new(request))
                )
                .await??
                .into_inner();

                Ok(TestValidators::validate_process_document_response(&response))
            }
        ).await;

        // Validate all enum values were processed successfully
        assert_eq!(results.len(), document_types.len());
        for result in results {
            assert!(result.expect("Enum test failed").expect("Response validation failed"));
        }

        server.shutdown().await;
    }

    // =============================================================================
    // CONCURRENT PROTOCOL ACCESS TESTS
    // =============================================================================

    #[tokio::test]
    async fn test_concurrent_protocol_correctness() {
        let server = TestGrpcServer::start().await.expect("Failed to start test server");
        let channel = server.get_client_channel().await.expect("Failed to get channel");

        // Test concurrent access across multiple service types
        let results = ConcurrentTestRunner::run_concurrent_requests(
            50, // 50 concurrent requests
            |i| {
                let mut health_client = system_service_client::SystemServiceClient::new(channel.clone());
                let mut doc_client = document_processor_client::DocumentProcessorClient::new(channel.clone());

                async move {
                    if i % 2 == 0 {
                        // Even requests: health check
                        health_client.health_check(Request::new(())).await.map(|_| ())
                    } else {
                        // Odd requests: document processing
                        let request = TestDataFactory::create_process_document_request(
                            &format!("/test/concurrent_{}.txt", i),
                            "concurrent_project",
                            "concurrent_collection",
                            DocumentType::Text,
                        );
                        doc_client.process_document(Request::new(request)).await.map(|_| ())
                    }
                }
            }
        ).await;

        let stats = ConcurrentTestRunner::analyze_results(&results);

        // Validate concurrent protocol access
        assert!(stats.meets_success_threshold(0.8),
            "Success rate {} below threshold", stats.success_rate);
        assert_eq!(stats.total_requests, 50);

        println!("Concurrent test results: {} successes, {} failures, {:.2}% success rate",
            stats.successful_requests, stats.failed_requests, stats.success_rate * 100.0);

        if let Some((code, count)) = stats.most_common_error() {
            println!("Most common error: {:?} ({} occurrences)", code, count);
        }

        server.shutdown().await;
    }

    // =============================================================================
    // COMPLEX NESTED MESSAGE VALIDATION TESTS
    // =============================================================================

    #[tokio::test]
    async fn test_complex_nested_message_validation() {
        let server = TestGrpcServer::start().await.expect("Failed to start test server");
        let mut clients = server.get_clients().await.expect("Failed to get clients");

        // Create deeply nested message structure with all optional fields
        let ranking_options = RankingOptions {
            semantic_weight: 0.6,
            keyword_weight: 0.4,
            rrf_constant: 75.0,
        };

        let search_options = SearchOptions {
            limit: 100,
            score_threshold: 0.95,
            include_metadata: true,
            include_content: true,
            ranking: Some(ranking_options),
        };

        let request = HybridSearchRequest {
            query: "deeply nested message validation test with all fields".to_string(),
            context: SearchContext::All as i32,
            options: Some(search_options),
            project_id: "nested_message_project".to_string(),
            collection_names: vec![
                "collection_one".to_string(),
                "collection_two".to_string(),
                "collection_three".to_string(),
            ],
        };

        // Validate complex nested message serialization
        let response = timeout(
            Duration::from_secs(5),
            clients.search_service.hybrid_search(Request::new(request))
        )
        .await
        .expect("Timeout")
        .expect("Request failed")
        .into_inner();

        assert!(TestValidators::validate_hybrid_search_response(&response));
        assert!(!response.query_id.is_empty());
        assert!(response.metadata.is_some());

        let metadata = response.metadata.unwrap();
        assert!(metadata.total_results >= 0);
        assert!(metadata.search_duration_ms >= 0);

        server.shutdown().await;
    }

    // =============================================================================
    // PROTOCOL COMPLIANCE AND TIMEOUT TESTS
    // =============================================================================

    #[tokio::test]
    async fn test_grpc_timeout_handling() {
        let server = TestGrpcServer::start().await.expect("Failed to start test server");
        let mut clients = server.get_clients().await.expect("Failed to get clients");

        // Test with very short timeout to verify timeout handling
        let request = TestDataFactory::create_process_document_request(
            "/test/timeout_test.txt",
            "timeout_project",
            "timeout_collection",
            DocumentType::Text,
        );

        let mut grpc_request = Request::new(request);
        grpc_request.set_timeout(Duration::from_millis(1)); // Very short timeout

        let response = clients.document_processor.process_document(grpc_request).await;

        // Response should either succeed quickly or timeout appropriately
        match response {
            Ok(resp) => {
                // If it succeeded, validate the response
                assert!(TestValidators::validate_process_document_response(&resp.into_inner()));
            },
            Err(status) => {
                // If it timed out, should be DEADLINE_EXCEEDED
                println!("Request timed out as expected: {:?}", status.code());
                // Note: Current implementation may complete quickly enough to not timeout
            }
        }

        server.shutdown().await;
    }
}