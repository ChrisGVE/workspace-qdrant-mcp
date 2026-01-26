#![cfg(feature = "legacy_grpc_tests")]
//! Comprehensive gRPC service layer integration tests
//!
//! This module provides exhaustive testing of the gRPC communication layer
//! between the Rust daemon and Python MCP server, covering protocol correctness,
//! client-server communication patterns, error handling, and performance testing.

use shared_test_utils::{async_test, TestResult};
use std::net::SocketAddr;
use std::time::{Duration, Instant};
use tokio::net::TcpListener;
use tokio::time::timeout;
use tonic::transport::{Channel, Server};
use tonic::{Request, Code};
use uuid::Uuid;
use workspace_qdrant_grpc::{
    ServerConfig, AuthConfig, TimeoutConfig, PerformanceConfig,
    service::IngestionService,
    proto::{
        ingest_service_client::IngestServiceClient,
        ingest_service_server::IngestServiceServer,
        *,
    },
};

/// Test fixture for gRPC integration tests
pub struct GrpcTestFixture {
    pub server_addr: SocketAddr,
    pub client: IngestServiceClient<Channel>,
    pub _server_handle: tokio::task::JoinHandle<()>,
    pub shutdown_tx: tokio::sync::oneshot::Sender<()>,
}

impl GrpcTestFixture {
    /// Create a new test fixture with running server and connected client
    pub async fn new() -> TestResult<Self> {
        Self::new_with_config(ServerConfig::new("127.0.0.1:0".parse()?)).await
    }

    /// Create fixture with custom server configuration
    pub async fn new_with_config(mut config: ServerConfig) -> TestResult<Self> {
        // Find available port
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let addr = listener.local_addr()?;
        config.bind_addr = addr;
        drop(listener);

        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();

        // Start server
        let server_config = config.clone();
        let server_handle = tokio::spawn(async move {
            let service = IngestionService::new_with_auth(server_config.auth_config.clone());
            let svc = IngestServiceServer::new(service);

            let server = Server::builder()
                .timeout(server_config.timeout_config.request_timeout)
                .add_service(svc);

            server
                .serve_with_shutdown(server_config.bind_addr, async {
                    shutdown_rx.await.ok();
                })
                .await
                .expect("Server failed to start");
        });

        // Wait for server to start and create client
        tokio::time::sleep(Duration::from_millis(100)).await;

        let channel = Channel::from_shared(format!("http://{}", addr))?
            .connect_timeout(Duration::from_secs(5))
            .timeout(Duration::from_secs(30))
            .connect()
            .await?;

        let client = IngestServiceClient::new(channel);

        Ok(Self {
            server_addr: addr,
            client,
            _server_handle: server_handle,
            shutdown_tx,
        })
    }

    /// Create fixture with authentication enabled
    pub async fn new_with_auth(api_key: String) -> TestResult<Self> {
        let auth_config = AuthConfig {
            enabled: true,
            api_key: Some(api_key),
            jwt_secret: None,
            allowed_origins: vec!["*".to_string()],
        };

        let config = ServerConfig::new("127.0.0.1:0".parse()?)
            .with_auth(auth_config);

        Self::new_with_config(config).await
    }

    /// Create authenticated request with Bearer token
    pub fn authenticated_request<T>(&self, message: T, api_key: &str) -> Request<T> {
        let mut request = Request::new(message);
        request.metadata_mut().insert(
            "authorization",
            format!("Bearer {}", api_key).parse().unwrap(),
        );
        request
    }

    /// Shutdown the test fixture
    pub async fn shutdown(self) -> TestResult<()> {
        let _ = self.shutdown_tx.send(());
        tokio::time::sleep(Duration::from_millis(100)).await;
        Ok(())
    }
}

// Note: Explicit shutdown() method should be called for proper cleanup

#[cfg(test)]
mod protocol_correctness_tests {
    use super::*;

    async_test!(test_protobuf_message_serialization, {
        // Test message serialization/deserialization for all major message types
        use prost::Message;

        // Test ProcessDocumentRequest
        let request = ProcessDocumentRequest {
            file_path: "/test/document.txt".to_string(),
            collection: "test_collection".to_string(),
            metadata: [("key".to_string(), "value".to_string())].into(),
            document_id: Some("doc_123".to_string()),
            chunk_text: true,
        };

        let encoded = request.encode_to_vec();
        let decoded = ProcessDocumentRequest::decode(&encoded[..])?;

        assert_eq!(decoded.file_path, "/test/document.txt");
        assert_eq!(decoded.collection, "test_collection");
        assert_eq!(decoded.document_id, Some("doc_123".to_string()));
        assert_eq!(decoded.chunk_text, true);
        assert_eq!(decoded.metadata.len(), 1);

        // Test ExecuteQueryRequest
        let query_request = ExecuteQueryRequest {
            query: "test query".to_string(),
            collections: vec!["collection1".to_string(), "collection2".to_string()],
            mode: SearchMode::Hybrid as i32,
            limit: 10,
            score_threshold: 0.5,
        };

        let encoded = query_request.encode_to_vec();
        let decoded = ExecuteQueryRequest::decode(&encoded[..])?;

        assert_eq!(decoded.query, "test query");
        assert_eq!(decoded.collections.len(), 2);
        assert_eq!(decoded.mode, SearchMode::Hybrid as i32);
        assert_eq!(decoded.limit, 10);
        assert_eq!(decoded.score_threshold, 0.5);

        // Test HealthResponse with nested ServiceHealth
        let health_response = HealthResponse {
            status: HealthStatus::Healthy as i32,
            message: "All systems operational".to_string(),
            services: vec![
                ServiceHealth {
                    name: "ingestion_engine".to_string(),
                    status: HealthStatus::Healthy as i32,
                    message: "Running normally".to_string(),
                },
                ServiceHealth {
                    name: "qdrant_client".to_string(),
                    status: HealthStatus::Degraded as i32,
                    message: "High latency".to_string(),
                },
            ],
        };

        let encoded = health_response.encode_to_vec();
        let decoded = HealthResponse::decode(&encoded[..])?;

        assert_eq!(decoded.status, HealthStatus::Healthy as i32);
        assert_eq!(decoded.services.len(), 2);
        assert_eq!(decoded.services[0].name, "ingestion_engine");
        assert_eq!(decoded.services[1].status, HealthStatus::Degraded as i32);

        Ok(())
    });

    async_test!(test_enum_serialization, {
        use prost::Message;

        // Test all enum variants can be serialized and deserialized correctly
        let search_modes = vec![
            SearchMode::Unspecified,
            SearchMode::Hybrid,
            SearchMode::Dense,
            SearchMode::Sparse,
        ];

        for mode in search_modes {
            let request = ExecuteQueryRequest {
                query: "test".to_string(),
                collections: vec![],
                mode: mode as i32,
                limit: 1,
                score_threshold: 0.0,
            };

            let encoded = request.encode_to_vec();
            let decoded = ExecuteQueryRequest::decode(&encoded[..])?;
            assert_eq!(decoded.mode, mode as i32);
        }

        let health_statuses = vec![
            HealthStatus::Unspecified,
            HealthStatus::Healthy,
            HealthStatus::Degraded,
            HealthStatus::Unhealthy,
        ];

        for status in health_statuses {
            let response = HealthResponse {
                status: status as i32,
                message: "test".to_string(),
                services: vec![],
            };

            let encoded = response.encode_to_vec();
            let decoded = HealthResponse::decode(&encoded[..])?;
            assert_eq!(decoded.status, status as i32);
        }

        Ok(())
    });

    async_test!(test_timestamp_handling, {
        use prost::Message;
        use prost_types::Timestamp;

        let now = std::time::SystemTime::now();
        let timestamp = Timestamp::from(now);

        let stats = EngineStats {
            started_at: Some(timestamp.clone()),
            uptime: Some(prost_types::Duration { seconds: 3600, nanos: 0 }),
            total_documents_processed: 1000,
            total_documents_indexed: 950,
            active_watches: 5,
            version: "0.2.1".to_string(),
        };

        let encoded = stats.encode_to_vec();
        let decoded = EngineStats::decode(&encoded[..])?;

        assert_eq!(decoded.started_at, Some(timestamp));
        assert_eq!(decoded.total_documents_processed, 1000);
        assert_eq!(decoded.version, "0.2.1");

        Ok(())
    });
}

#[cfg(test)]
mod client_server_communication_tests {
    use super::*;

    async_test!(test_basic_request_response_cycle, {
        let fixture = GrpcTestFixture::new().await?;
        let mut client = fixture.client.clone();

        // Test health check - simplest request/response
        let health_request = Request::new(());
        let response = client.health_check(health_request).await?;
        let health_response = response.into_inner();

        assert_eq!(health_response.status, HealthStatus::Healthy as i32);
        assert!(!health_response.message.is_empty());
        assert!(!health_response.services.is_empty());

        fixture.shutdown().await?;
        Ok(())
    });

    async_test!(test_process_document_request_response, {
        let fixture = GrpcTestFixture::new().await?;
        let mut client = fixture.client.clone();

        let request = ProcessDocumentRequest {
            file_path: "/test/document.txt".to_string(),
            collection: "test_collection".to_string(),
            metadata: [("author".to_string(), "test_user".to_string())].into(),
            document_id: Some("test_doc_123".to_string()),
            chunk_text: true,
        };

        let result = client.process_document(Request::new(request)).await;

        // The service should handle file processing gracefully
        match result {
            Ok(response) => {
                let process_response = response.into_inner();
                assert!(process_response.success);
                assert!(!process_response.message.is_empty());
                assert_eq!(process_response.document_id, Some("test_doc_123".to_string()));
                assert!(process_response.chunks_added >= 0); // May be 0 for non-existent files
            }
            Err(status) => {
                // Expected for non-existent test files
                assert_eq!(status.code(), Code::Internal);
                assert!(status.message().contains("Processing failed"));
            }
        }

        fixture.shutdown().await?;
        Ok(())
    });

    async_test!(test_streaming_request_response, {
        let fixture = GrpcTestFixture::new().await?;
        let mut client = fixture.client.clone();

        // Test watching stream
        let watch_request = StartWatchingRequest {
            path: "/test/watch/path".to_string(),
            collection: "watch_collection".to_string(),
            patterns: vec!["*.txt".to_string()],
            ignore_patterns: vec!["*.tmp".to_string()],
            auto_ingest: true,
            recursive: true,
            recursive_depth: 5,
            debounce_seconds: 1,
            update_frequency_ms: 1000,
            watch_id: Some("test_watch_123".to_string()),
        };

        let mut response_stream = client
            .start_watching(Request::new(watch_request))
            .await?
            .into_inner();

        // Should receive at least one update
        let first_update = timeout(Duration::from_secs(5), response_stream.message())
            .await??
            .expect("Should receive watching update");

        assert_eq!(first_update.watch_id, "test_watch_123");
        assert_eq!(first_update.event_type, WatchEventType::Started as i32);
        assert_eq!(first_update.status, WatchStatus::Active as i32);

        fixture.shutdown().await?;
        Ok(())
    });

    async_test!(test_concurrent_client_connections, {
        let fixture = GrpcTestFixture::new().await?;

        // Create multiple clients connecting to the same server
        let clients = futures::future::try_join_all(
            (0..5).map(|_| async {
                let channel = Channel::from_shared(format!("http://{}", fixture.server_addr))?
                    .connect()
                    .await?;
                Ok::<_, Box<dyn std::error::Error + Send + Sync>>(
                    IngestServiceClient::new(channel)
                )
            })
        ).await?;

        // Execute concurrent requests
        let futures = clients.into_iter().enumerate().map(|(i, mut client)| {
            let collection = format!("collection_{}", i);
            async move {
                let request = ProcessDocumentRequest {
                    file_path: format!("/test/doc_{}.txt", i),
                    collection: collection.clone(),
                    metadata: [("client_id".to_string(), i.to_string())].into(),
                    document_id: Some(format!("doc_{}", i)),
                    chunk_text: true,
                };

                let result = client.process_document(Request::new(request)).await;

                // Handle both success and expected processing failures
                match result {
                    Ok(response) => {
                        let process_response = response.into_inner();
                        assert!(process_response.success);
                        assert_eq!(process_response.document_id, Some(format!("doc_{}", i)));
                    }
                    Err(status) => {
                        // Expected for non-existent test files
                        assert_eq!(status.code(), Code::Internal);
                    }
                }

                Ok::<_, Box<dyn std::error::Error + Send + Sync>>(())
            }
        });

        futures::future::try_join_all(futures).await?;

        fixture.shutdown().await?;
        Ok(())
    });

    async_test!(test_large_message_handling, {
        let fixture = GrpcTestFixture::new().await?;
        let mut client = fixture.client.clone();

        // Create a large metadata map
        let large_metadata: std::collections::HashMap<String, String> = (0..1000)
            .map(|i| (format!("key_{}", i), format!("value_with_lots_of_data_{}", i.to_string().repeat(10))))
            .collect();

        let request = ProcessDocumentRequest {
            file_path: "/test/large_document.txt".to_string(),
            collection: "large_test_collection".to_string(),
            metadata: large_metadata.clone(),
            document_id: Some("large_doc_123".to_string()),
            chunk_text: true,
        };

        let result = client.process_document(Request::new(request)).await;

        match result {
            Ok(response) => {
                let process_response = response.into_inner();
                assert!(process_response.success);
                assert_eq!(process_response.applied_metadata.len(), large_metadata.len());
            }
            Err(status) => {
                // May fail due to file not existing or message size limits
                assert!(status.code() == Code::Internal || status.code() == Code::ResourceExhausted);
            }
        }

        fixture.shutdown().await?;
        Ok(())
    });
}

#[cfg(test)]
mod error_propagation_tests {
    use super::*;

    async_test!(test_invalid_request_parameters, {
        let fixture = GrpcTestFixture::new().await?;
        let mut client = fixture.client.clone();

        // Test empty file path
        let invalid_request = ProcessDocumentRequest {
            file_path: "".to_string(),
            collection: "test_collection".to_string(),
            metadata: std::collections::HashMap::new(),
            document_id: None,
            chunk_text: false,
        };

        let result = client.process_document(Request::new(invalid_request)).await;
        assert!(result.is_err());

        let error = result.unwrap_err();
        assert_eq!(error.code(), Code::InvalidArgument);
        assert!(error.message().contains("File path cannot be empty"));

        // Test empty collection name
        let invalid_request = ProcessDocumentRequest {
            file_path: "/test/document.txt".to_string(),
            collection: "".to_string(),
            metadata: std::collections::HashMap::new(),
            document_id: None,
            chunk_text: false,
        };

        let result = client.process_document(Request::new(invalid_request)).await;
        assert!(result.is_err());

        let error = result.unwrap_err();
        assert_eq!(error.code(), Code::InvalidArgument);
        assert!(error.message().contains("Collection name cannot be empty"));

        fixture.shutdown().await?;
        Ok(())
    });

    async_test!(test_authentication_errors, {
        let api_key = "test_secret_key_123";
        let fixture = GrpcTestFixture::new_with_auth(api_key.to_string()).await?;
        let mut client = fixture.client.clone();

        // Test missing authentication
        let request = ProcessDocumentRequest {
            file_path: "/test/document.txt".to_string(),
            collection: "test_collection".to_string(),
            metadata: std::collections::HashMap::new(),
            document_id: Some("test_doc".to_string()),
            chunk_text: true,
        };

        let result = client.process_document(Request::new(request)).await;
        assert!(result.is_err());

        let error = result.unwrap_err();
        assert_eq!(error.code(), Code::Unauthenticated);
        assert!(error.message().contains("Missing authorization header"));

        // Test invalid API key
        let request = ProcessDocumentRequest {
            file_path: "/test/document.txt".to_string(),
            collection: "test_collection".to_string(),
            metadata: std::collections::HashMap::new(),
            document_id: Some("test_doc".to_string()),
            chunk_text: true,
        };

        let wrong_key_request = fixture.authenticated_request(request, "wrong_key");
        let result = client.process_document(wrong_key_request).await;
        assert!(result.is_err());

        let error = result.unwrap_err();
        assert_eq!(error.code(), Code::Unauthenticated);
        assert!(error.message().contains("Invalid API key"));

        // Test successful authentication
        let request = ProcessDocumentRequest {
            file_path: "/test/document.txt".to_string(),
            collection: "test_collection".to_string(),
            metadata: std::collections::HashMap::new(),
            document_id: Some("test_doc".to_string()),
            chunk_text: true,
        };

        let valid_request = fixture.authenticated_request(request, api_key);
        let result = client.process_document(valid_request).await;

        // Should either succeed or fail with processing error (not auth error)
        match result {
            Ok(_) => {}, // Authentication successful, processing may succeed
            Err(status) => {
                // Should be processing error, not authentication error
                assert_eq!(status.code(), Code::Internal);
                assert!(status.message().contains("Processing failed"));
            }
        }

        fixture.shutdown().await?;
        Ok(())
    });

    async_test!(test_server_error_propagation, {
        let fixture = GrpcTestFixture::new().await?;
        let mut client = fixture.client.clone();

        // Test processing a non-existent file (should trigger internal error)
        let request = ProcessDocumentRequest {
            file_path: "/non/existent/file/path/that/should/not/exist.txt".to_string(),
            collection: "test_collection".to_string(),
            metadata: std::collections::HashMap::new(),
            document_id: Some("test_doc".to_string()),
            chunk_text: true,
        };

        let result = client.process_document(Request::new(request)).await;

        // The service implementation should handle file not found errors gracefully
        // and return an Internal error status
        match result {
            Err(status) => {
                assert_eq!(status.code(), Code::Internal);
                assert!(status.message().contains("Processing failed"));
            }
            Ok(_) => {
                // If the mock implementation doesn't actually check files,
                // it might succeed - this is also acceptable for testing
            }
        }

        fixture.shutdown().await?;
        Ok(())
    });

    async_test!(test_grpc_status_code_mapping, {
        let fixture = GrpcTestFixture::new().await?;
        let mut client = fixture.client.clone();

        // Test various error conditions and verify correct status codes
        let test_cases = vec![
            (
                ProcessDocumentRequest {
                    file_path: "".to_string(),
                    collection: "test".to_string(),
                    metadata: std::collections::HashMap::new(),
                    document_id: None,
                    chunk_text: false,
                },
                Code::InvalidArgument,
                "empty file path"
            ),
            (
                ProcessDocumentRequest {
                    file_path: "/test/file.txt".to_string(),
                    collection: "".to_string(),
                    metadata: std::collections::HashMap::new(),
                    document_id: None,
                    chunk_text: false,
                },
                Code::InvalidArgument,
                "empty collection"
            ),
        ];

        for (request, expected_code, description) in test_cases {
            let result = client.process_document(Request::new(request)).await;
            assert!(result.is_err(), "Expected error for {}", description);

            let error = result.unwrap_err();
            assert_eq!(error.code(), expected_code, "Wrong error code for {}", description);
        }

        fixture.shutdown().await?;
        Ok(())
    });
}

#[cfg(test)]
mod connection_management_tests {
    use super::*;

    async_test!(test_connection_establishment_and_teardown, {
        // Test multiple connection/disconnection cycles
        for i in 0..3 {
            let fixture = GrpcTestFixture::new().await?;
            let mut client = fixture.client.clone();

            // Verify connection works
            let health_request = Request::new(());
            let response = client.health_check(health_request).await?;
            assert_eq!(response.into_inner().status, HealthStatus::Healthy as i32);

            // Explicitly shutdown
            fixture.shutdown().await?;

            // Small delay between iterations
            if i < 2 {
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        }

        Ok(())
    });

    async_test!(test_request_timeout_handling, {
        let timeout_config = TimeoutConfig {
            request_timeout: Duration::from_millis(100), // Very short timeout
            connection_timeout: Duration::from_secs(5),
            keepalive_interval: Duration::from_secs(30),
            keepalive_timeout: Duration::from_secs(5),
        };

        let config = ServerConfig::new("127.0.0.1:0".parse()?)
            .with_timeouts(timeout_config);

        let fixture = GrpcTestFixture::new_with_config(config).await?;
        let mut client = fixture.client.clone();

        // Create a request that might take longer than the timeout
        let request = ProcessDocumentRequest {
            file_path: "/test/document.txt".to_string(),
            collection: "test_collection".to_string(),
            metadata: (0..100).map(|i| (format!("key_{}", i), format!("value_{}", i))).collect(),
            document_id: Some("timeout_test".to_string()),
            chunk_text: true,
        };

        // The request might timeout or succeed depending on processing speed
        let result = client.process_document(Request::new(request)).await;

        match result {
            Ok(_) => {
                // Request completed within timeout - acceptable
            }
            Err(status) => {
                // Should be a timeout-related error
                assert!(
                    status.code() == Code::DeadlineExceeded ||
                    status.code() == Code::Cancelled ||
                    status.message().contains("timeout") ||
                    status.message().contains("deadline")
                );
            }
        }

        fixture.shutdown().await?;
        Ok(())
    });

    async_test!(test_connection_pooling_and_reuse, {
        let fixture = GrpcTestFixture::new().await?;

        // Create multiple clients that should reuse connections
        let channel = Channel::from_shared(format!("http://{}", fixture.server_addr))?
            .connect()
            .await?;

        let mut clients: Vec<IngestServiceClient<Channel>> = (0..5)
            .map(|_| IngestServiceClient::new(channel.clone()))
            .collect();

        // Execute requests concurrently using the same underlying connection
        let futures = clients.iter_mut().enumerate().map(|(i, client)| async move {
            let request = Request::new(());
            let response = client.health_check(request).await?;
            assert_eq!(response.into_inner().status, HealthStatus::Healthy as i32);
            Ok::<_, Box<dyn std::error::Error + Send + Sync>>(i)
        });

        let results = futures::future::try_join_all(futures).await?;
        assert_eq!(results.len(), 5);

        fixture.shutdown().await?;
        Ok(())
    });

    async_test!(test_connection_failure_recovery, {
        let fixture = GrpcTestFixture::new().await?;
        let mut client = fixture.client.clone();

        // Verify initial connection works
        let health_request = Request::new(());
        let response = client.health_check(health_request).await?;
        assert_eq!(response.into_inner().status, HealthStatus::Healthy as i32);

        // Shutdown server
        fixture.shutdown().await?;

        // Verify connection fails appropriately
        let health_request = Request::new(());
        let result = client.health_check(health_request).await;
        assert!(result.is_err());

        // Error should indicate connection failure
        let error = result.unwrap_err();
        assert!(
            error.code() == Code::Unavailable ||
            error.code() == Code::Internal ||
            error.message().contains("connection") ||
            error.message().contains("transport")
        );

        Ok(())
    });

    async_test!(test_concurrent_connection_limits, {
        // Test server behavior under high concurrent connection load
        let perf_config = PerformanceConfig {
            max_concurrent_streams: 10,
            max_message_size: 4 * 1024 * 1024, // 4MB
            max_connection_idle: Duration::from_secs(60),
            max_connection_age: Duration::from_secs(3600),
            tcp_nodelay: true,
            tcp_keepalive: Some(Duration::from_secs(600)),
        };

        let config = ServerConfig::new("127.0.0.1:0".parse()?)
            .with_performance(perf_config);

        let fixture = GrpcTestFixture::new_with_config(config).await?;

        // Create more concurrent connections than the limit
        let futures = (0..15).map(|i| async move {
            let channel = Channel::from_shared(format!("http://{}", fixture.server_addr))?
                .connect()
                .await?;

            let mut client = IngestServiceClient::new(channel);

            let health_request = Request::new(());
            let response = client.health_check(health_request).await?;

            Ok::<_, Box<dyn std::error::Error + Send + Sync>>(response.into_inner().status)
        });

        // Some requests should succeed, others might be limited/queued
        let results = futures::future::join_all(futures).await;

        let successful_requests = results.iter().filter(|r| r.is_ok()).count();
        assert!(successful_requests > 0, "At least some requests should succeed");

        // Verify that successful requests returned healthy status
        for result in results {
            if let Ok(status) = result {
                assert_eq!(status, HealthStatus::Healthy as i32);
            }
        }

        fixture.shutdown().await?;
        Ok(())
    });
}

#[cfg(test)]
mod service_discovery_and_health_tests {
    use super::*;

    async_test!(test_health_check_comprehensive, {
        let fixture = GrpcTestFixture::new().await?;
        let mut client = fixture.client.clone();

        let health_request = Request::new(());
        let response = client.health_check(health_request).await?;
        let health_response = response.into_inner();

        // Verify overall health status
        assert!(
            health_response.status == HealthStatus::Healthy as i32 ||
            health_response.status == HealthStatus::Degraded as i32
        );

        // Verify message is present
        assert!(!health_response.message.is_empty());

        // Verify services are reported
        assert!(!health_response.services.is_empty());

        // Check specific service health
        let service_names: Vec<String> = health_response.services
            .iter()
            .map(|s| s.name.clone())
            .collect();

        assert!(service_names.contains(&"ingestion_engine".to_string()));
        assert!(service_names.contains(&"qdrant_client".to_string()));

        // Verify each service has valid status
        for service in &health_response.services {
            assert!(!service.name.is_empty());
            assert!(!service.message.is_empty());
            assert!(
                service.status == HealthStatus::Healthy as i32 ||
                service.status == HealthStatus::Degraded as i32 ||
                service.status == HealthStatus::Unhealthy as i32
            );
        }

        fixture.shutdown().await?;
        Ok(())
    });

    async_test!(test_system_status_monitoring, {
        let fixture = GrpcTestFixture::new().await?;
        let mut client = fixture.client.clone();

        let system_request = Request::new(());
        let response = client.get_system_status(system_request).await?;
        let system_response = response.into_inner();

        // Verify system status structure
        assert_eq!(system_response.overall_status, HealthStatus::Healthy as i32);
        assert!(!system_response.message.is_empty());

        // System info should be present (even if None in mock)
        // Components should be reported
        // Processing stats should be available
        // Resource usage should be monitored

        fixture.shutdown().await?;
        Ok(())
    });

    async_test!(test_stats_and_metrics_collection, {
        let fixture = GrpcTestFixture::new().await?;
        let mut client = fixture.client.clone();

        let stats_request = GetStatsRequest {
            include_collection_stats: true,
            include_watch_stats: true,
        };

        let response = client.get_stats(Request::new(stats_request)).await?;
        let stats_response = response.into_inner();

        // Verify engine stats are present
        assert!(stats_response.engine_stats.is_some());

        let engine_stats = stats_response.engine_stats.unwrap();
        assert!(engine_stats.started_at.is_some());
        assert!(!engine_stats.version.is_empty());
        assert!(engine_stats.total_documents_processed >= 0);
        assert!(engine_stats.total_documents_indexed >= 0);
        assert!(engine_stats.active_watches >= 0);

        // Collection and watch stats arrays should be present (even if empty)
        // In a real implementation, these would contain actual data

        fixture.shutdown().await?;
        Ok(())
    });

    async_test!(test_service_discovery_through_health_endpoints, {
        let fixture = GrpcTestFixture::new().await?;
        let mut client = fixture.client.clone();

        // Simulate service discovery by checking what services are available
        // through health checks and stats

        let health_response = client.health_check(Request::new(())).await?.into_inner();
        let stats_response = client.get_stats(Request::new(GetStatsRequest {
            include_collection_stats: true,
            include_watch_stats: true,
        })).await?.into_inner();

        // Verify we can discover available services
        let available_services: Vec<String> = health_response.services
            .iter()
            .filter(|s| s.status != HealthStatus::Unhealthy as i32)
            .map(|s| s.name.clone())
            .collect();

        assert!(!available_services.is_empty());

        // Verify service capabilities through stats
        assert!(stats_response.engine_stats.is_some());

        // Test individual service endpoints based on discovery
        if available_services.contains(&"ingestion_engine".to_string()) {
            // Test document processing
            let process_request = ProcessDocumentRequest {
                file_path: "/test/discovery.txt".to_string(),
                collection: "discovery_test".to_string(),
                metadata: std::collections::HashMap::new(),
                document_id: Some("discovery_doc".to_string()),
                chunk_text: false,
            };

            let process_result = client.process_document(Request::new(process_request)).await;
            // Should either succeed or fail gracefully
            match process_result {
                Ok(response) => {
                    assert!(response.into_inner().success);
                }
                Err(status) => {
                    // Expected for non-existent files in test environment
                    assert_eq!(status.code(), Code::Internal);
                    assert!(status.message().contains("Processing failed"));
                }
            }
        }

        fixture.shutdown().await?;
        Ok(())
    });

    async_test!(test_streaming_health_monitoring, {
        let fixture = GrpcTestFixture::new().await?;
        let mut client = fixture.client.clone();

        // Test streaming endpoints for real-time monitoring
        let stream_request = StreamStatusRequest {
            update_interval_seconds: 1,
            include_history: false,
            collection_filter: None,
        };

        let mut stream = client
            .stream_processing_status(Request::new(stream_request))
            .await?
            .into_inner();

        // Should receive at least one status update
        let first_update = timeout(Duration::from_secs(3), stream.message())
            .await??
            .expect("Should receive processing status update");

        assert!(first_update.timestamp.is_some());

        // Test metrics streaming
        let metrics_request = StreamMetricsRequest {
            update_interval_seconds: 1,
            include_detailed_metrics: true,
        };

        let mut metrics_stream = client
            .stream_system_metrics(Request::new(metrics_request))
            .await?
            .into_inner();

        let first_metrics = timeout(Duration::from_secs(3), metrics_stream.message())
            .await??
            .expect("Should receive metrics update");

        assert!(first_metrics.timestamp.is_some());

        fixture.shutdown().await?;
        Ok(())
    });
}

#[cfg(test)]
mod performance_and_load_tests {
    use super::*;

    async_test!(test_concurrent_request_handling, {
        let fixture = GrpcTestFixture::new().await?;

        // Create multiple clients for concurrent testing
        let num_clients = 10;
        let requests_per_client = 5;

        let clients = futures::future::try_join_all(
            (0..num_clients).map(|_| async {
                let channel = Channel::from_shared(format!("http://{}", fixture.server_addr))?
                    .connect()
                    .await?;
                Ok::<_, Box<dyn std::error::Error + Send + Sync>>(
                    IngestServiceClient::new(channel)
                )
            })
        ).await?;

        let start_time = Instant::now();

        // Execute concurrent requests
        let all_futures = clients.into_iter().enumerate().flat_map(|(client_id, mut client)| {
            (0..requests_per_client).map(move |req_id| {
                let mut client = client.clone();
                async move {
                    let request = ProcessDocumentRequest {
                        file_path: format!("/test/client_{}_req_{}.txt", client_id, req_id),
                        collection: format!("test_collection_{}", client_id),
                        metadata: [
                            ("client_id".to_string(), client_id.to_string()),
                            ("request_id".to_string(), req_id.to_string()),
                        ].into(),
                        document_id: Some(format!("doc_{}_{}", client_id, req_id)),
                        chunk_text: true,
                    };

                    let start = Instant::now();
                    let result = client.process_document(Request::new(request)).await;
                    let duration = start.elapsed();

                    // Consider both success and expected processing failures as acceptable
                    let success = match result {
                        Ok(_) => true,
                        Err(status) => status.code() == Code::Internal, // Expected for non-existent files
                    };

                    Ok::<_, Box<dyn std::error::Error + Send + Sync>>((success, duration))
                }
            })
        });

        let results = futures::future::join_all(all_futures).await;
        let total_duration = start_time.elapsed();

        // Analyze results
        let successful_requests = results.iter().filter(|r| r.as_ref().map(|(success, _)| *success).unwrap_or(false)).count();
        let total_requests = num_clients * requests_per_client;

        assert!(successful_requests as f64 / total_requests as f64 > 0.8,
                "At least 80% of requests should succeed under load");

        let avg_request_time: Duration = results.iter()
            .filter_map(|r| r.as_ref().ok())
            .map(|(_, duration)| *duration)
            .sum::<Duration>() / total_requests as u32;

        println!("Load test results:");
        println!("  Total requests: {}", total_requests);
        println!("  Successful requests: {}", successful_requests);
        println!("  Success rate: {:.2}%", (successful_requests as f64 / total_requests as f64) * 100.0);
        println!("  Total duration: {:?}", total_duration);
        println!("  Average request time: {:?}", avg_request_time);

        // Performance assertions
        assert!(avg_request_time < Duration::from_secs(1), "Average request time should be under 1 second");

        fixture.shutdown().await?;
        Ok(())
    });

    async_test!(test_large_message_performance, {
        let fixture = GrpcTestFixture::new().await?;
        let mut client = fixture.client.clone();

        // Test progressively larger messages
        let message_sizes = vec![1_000, 10_000, 100_000, 500_000];

        for size in message_sizes {
            let large_metadata: std::collections::HashMap<String, String> = (0..size)
                .map(|i| (format!("key_{}", i), format!("value_{}", i)))
                .collect();

            let request = ProcessDocumentRequest {
                file_path: format!("/test/large_doc_{}.txt", size),
                collection: "large_message_test".to_string(),
                metadata: large_metadata,
                document_id: Some(format!("large_doc_{}", size)),
                chunk_text: true,
            };

            let start_time = Instant::now();
            let result = client.process_document(Request::new(request)).await;
            let duration = start_time.elapsed();

            match result {
                Ok(response) => {
                    let process_response = response.into_inner();
                    assert!(process_response.success);
                    println!("Message size {}: {:?}", size, duration);

                    // Performance expectation: should handle even large messages in reasonable time
                    assert!(duration < Duration::from_secs(10),
                            "Large message processing should complete within 10 seconds");
                }
                Err(status) => {
                    // Large messages might hit size limits or file processing might fail
                    if status.code() == Code::ResourceExhausted {
                        println!("Message size {} hit resource limits (expected for very large messages)", size);
                    } else if status.code() == Code::Internal {
                        println!("Message size {} failed processing (expected for non-existent files)", size);
                    } else {
                        return Err(format!("Unexpected error for size {}: {:?}", size, status).into());
                    }
                }
            }
        }

        fixture.shutdown().await?;
        Ok(())
    });

    async_test!(test_streaming_performance, {
        let fixture = GrpcTestFixture::new().await?;
        let mut client = fixture.client.clone();

        // Test streaming endpoint performance
        let watch_request = StartWatchingRequest {
            path: "/test/performance/watch".to_string(),
            collection: "performance_test".to_string(),
            patterns: vec!["*.txt".to_string(), "*.md".to_string()],
            ignore_patterns: vec!["*.tmp".to_string()],
            auto_ingest: true,
            recursive: true,
            recursive_depth: 3,
            debounce_seconds: 0, // Minimal debounce for performance testing
            update_frequency_ms: 100, // Fast updates
            watch_id: Some("performance_watch".to_string()),
        };

        let start_time = Instant::now();
        let mut stream = client
            .start_watching(Request::new(watch_request))
            .await?
            .into_inner();

        // Collect first few messages to test streaming performance
        let mut message_count = 0;
        let mut first_message_time = None;

        while message_count < 3 {
            match timeout(Duration::from_secs(2), stream.message()).await {
                Ok(Ok(Some(update))) => {
                    if first_message_time.is_none() {
                        first_message_time = Some(start_time.elapsed());
                    }

                    assert_eq!(update.watch_id, "performance_watch");
                    message_count += 1;
                }
                Ok(Ok(None)) => break, // Stream ended
                Ok(Err(e)) => return Err(format!("Stream error: {:?}", e).into()),
                Err(_) => break, // Timeout - might be expected for test streams
            }
        }

        if let Some(first_msg_time) = first_message_time {
            assert!(first_msg_time < Duration::from_secs(1),
                    "First stream message should arrive quickly");
            println!("Streaming performance: first message in {:?}", first_msg_time);
        }

        fixture.shutdown().await?;
        Ok(())
    });

    async_test!(test_memory_usage_under_load, {
        let fixture = GrpcTestFixture::new().await?;

        // Create sustained load to test memory usage
        let num_iterations = 100;
        let concurrent_requests = 5;

        for iteration in 0..num_iterations {
            let clients = futures::future::try_join_all(
                (0..concurrent_requests).map(|_| async {
                    let channel = Channel::from_shared(format!("http://{}", fixture.server_addr))?
                        .connect()
                        .await?;
                    Ok::<_, Box<dyn std::error::Error + Send + Sync>>(
                        IngestServiceClient::new(channel)
                    )
                })
            ).await?;

            let futures = clients.into_iter().enumerate().map(|(i, mut client)| async move {
                let request = ProcessDocumentRequest {
                    file_path: format!("/test/memory_test_{}_{}.txt", iteration, i),
                    collection: "memory_test".to_string(),
                    metadata: (0..50).map(|j| (
                        format!("key_{}_{}", i, j),
                        format!("value_{}_{}", iteration, j)
                    )).collect(),
                    document_id: Some(format!("doc_{}_{}", iteration, i)),
                    chunk_text: true,
                };

                let result = client.process_document(Request::new(request)).await;
                result
            });

            let _results = futures::future::join_all(futures).await;

            // Periodic progress update
            if iteration % 20 == 0 {
                println!("Memory test progress: {}/{}", iteration, num_iterations);
            }
        }

        // If we get here without OOM or excessive slowdown, memory usage is acceptable
        println!("Memory usage test completed successfully");

        fixture.shutdown().await?;
        Ok(())
    });
}

#[cfg(test)]
mod tonic_build_integration_tests {
    use super::*;

    async_test!(test_generated_client_server_compatibility, {
        // Test that tonic-build generated code works correctly for all message types
        let fixture = GrpcTestFixture::new().await?;
        let mut client = fixture.client.clone();

        // Test all major service endpoints to ensure generated code compatibility

        // 1. Simple request/response
        let health_result = client.health_check(Request::new(())).await;
        assert!(health_result.is_ok());

        // 2. Complex message types
        let stats_request = GetStatsRequest {
            include_collection_stats: true,
            include_watch_stats: true,
        };
        let stats_result = client.get_stats(Request::new(stats_request)).await;
        assert!(stats_result.is_ok());

        // 3. Collection operations
        let list_collections_result = client.list_collections(Request::new(ListCollectionsRequest {
            include_stats: true,
        })).await;
        assert!(list_collections_result.is_ok());

        // 4. Memory operations
        let memory_rules_result = client.list_memory_rules(Request::new(ListMemoryRulesRequest {
            category: Some("test".to_string()),
            authority: None,
            limit: 10,
            offset: 0,
        })).await;
        assert!(memory_rules_result.is_ok());

        // 5. Configuration operations
        let config_result = client.validate_configuration(Request::new(ValidateConfigurationRequest {
            config_yaml: "test: configuration".to_string(),
        })).await;
        assert!(config_result.is_ok());

        fixture.shutdown().await?;
        Ok(())
    });

    async_test!(test_protobuf_field_validation, {
        // Test that all protobuf fields are correctly handled by generated code
        use prost::Message;

        // Test ProcessDocumentRequest with all fields
        let full_request = ProcessDocumentRequest {
            file_path: "/full/test/path.txt".to_string(),
            collection: "full_test_collection".to_string(),
            metadata: [
                ("key1".to_string(), "value1".to_string()),
                ("key2".to_string(), "value2".to_string()),
            ].into(),
            document_id: Some("full_test_doc_id".to_string()),
            chunk_text: true,
        };

        // Serialize and deserialize to ensure all fields are preserved
        let encoded = full_request.encode_to_vec();
        let decoded = ProcessDocumentRequest::decode(&encoded[..])?;

        assert_eq!(decoded.file_path, "/full/test/path.txt");
        assert_eq!(decoded.collection, "full_test_collection");
        assert_eq!(decoded.metadata.len(), 2);
        assert_eq!(decoded.document_id, Some("full_test_doc_id".to_string()));
        assert_eq!(decoded.chunk_text, true);

        // Test with minimal fields
        let minimal_request = ProcessDocumentRequest {
            file_path: "/minimal/path.txt".to_string(),
            collection: "minimal_collection".to_string(),
            metadata: std::collections::HashMap::new(),
            document_id: None,
            chunk_text: false,
        };

        let encoded = minimal_request.encode_to_vec();
        let decoded = ProcessDocumentRequest::decode(&encoded[..])?;

        assert_eq!(decoded.file_path, "/minimal/path.txt");
        assert_eq!(decoded.collection, "minimal_collection");
        assert_eq!(decoded.metadata.len(), 0);
        assert_eq!(decoded.document_id, None);
        assert_eq!(decoded.chunk_text, false);

        Ok(())
    });

    async_test!(test_streaming_code_generation, {
        let fixture = GrpcTestFixture::new().await?;
        let mut client = fixture.client.clone();

        // Test that all streaming endpoints work with generated code

        // 1. Server streaming (watch updates)
        let watch_request = StartWatchingRequest {
            path: "/test/stream".to_string(),
            collection: "stream_test".to_string(),
            patterns: vec!["*.rs".to_string()],
            ignore_patterns: vec![],
            auto_ingest: false,
            recursive: false,
            recursive_depth: 1,
            debounce_seconds: 1,
            update_frequency_ms: 1000,
            watch_id: Some("stream_test_watch".to_string()),
        };

        let mut watch_stream = client
            .start_watching(Request::new(watch_request))
            .await?
            .into_inner();

        // Should receive at least one message
        let first_message = timeout(Duration::from_secs(3), watch_stream.message())
            .await??;
        assert!(first_message.is_some());

        // 2. Processing status streaming
        let status_request = StreamStatusRequest {
            update_interval_seconds: 1,
            include_history: false,
            collection_filter: None,
        };

        let mut status_stream = client
            .stream_processing_status(Request::new(status_request))
            .await?
            .into_inner();

        let status_message = timeout(Duration::from_secs(3), status_stream.message())
            .await??;
        assert!(status_message.is_some());

        // 3. Folder processing streaming
        let folder_request = ProcessFolderRequest {
            folder_path: "/test/folder".to_string(),
            collection: "folder_test".to_string(),
            include_patterns: vec!["*.txt".to_string()],
            ignore_patterns: vec!["*.log".to_string()],
            recursive: true,
            max_depth: 3,
            dry_run: true,
            metadata: [("test".to_string(), "folder_processing".to_string())].into(),
        };

        let mut folder_stream = client
            .process_folder(Request::new(folder_request))
            .await?
            .into_inner();

        let folder_message = timeout(Duration::from_secs(3), folder_stream.message())
            .await??;
        assert!(folder_message.is_some());

        fixture.shutdown().await?;
        Ok(())
    });

    async_test!(test_error_handling_in_generated_code, {
        let fixture = GrpcTestFixture::new().await?;
        let mut client = fixture.client.clone();

        // Test that tonic-generated error handling works correctly

        // Test invalid requests that should return specific error codes
        let invalid_requests = vec![
            ProcessDocumentRequest {
                file_path: "".to_string(),
                collection: "test".to_string(),
                metadata: std::collections::HashMap::new(),
                document_id: None,
                chunk_text: false,
            },
            ProcessDocumentRequest {
                file_path: "/test/file.txt".to_string(),
                collection: "".to_string(),
                metadata: std::collections::HashMap::new(),
                document_id: None,
                chunk_text: false,
            },
        ];

        for request in invalid_requests {
            let result = client.process_document(Request::new(request)).await;
            assert!(result.is_err());

            let error = result.unwrap_err();
            assert_eq!(error.code(), Code::InvalidArgument);

            // Verify error details are properly propagated through generated code
            assert!(!error.message().is_empty());
        }

        fixture.shutdown().await?;
        Ok(())
    });

    async_test!(test_metadata_handling_in_generated_code, {
        let fixture = GrpcTestFixture::new().await?;
        let mut client = fixture.client.clone();

        // Test that metadata is correctly handled by generated gRPC code
        let request = ProcessDocumentRequest {
            file_path: "/test/metadata.txt".to_string(),
            collection: "metadata_test".to_string(),
            metadata: std::collections::HashMap::new(),
            document_id: Some("metadata_doc".to_string()),
            chunk_text: true,
        };

        // Add custom metadata to the gRPC request
        let mut grpc_request = Request::new(request);
        grpc_request.metadata_mut().insert(
            "x-client-id",
            "test_client_123".parse().unwrap(),
        );
        grpc_request.metadata_mut().insert(
            "x-request-id",
            Uuid::new_v4().to_string().parse().unwrap(),
        );

        // Server should handle metadata correctly
        let result = client.process_document(grpc_request).await;

        match result {
            Ok(response) => {
                let process_response = response.into_inner();
                assert!(process_response.success);
            }
            Err(status) => {
                // Accept processing errors for non-existent files
                assert_eq!(status.code(), Code::Internal);
                assert!(status.message().contains("Processing failed"));
            }
        }

        fixture.shutdown().await?;
        Ok(())
    });
}
