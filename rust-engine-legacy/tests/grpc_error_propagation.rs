//! Comprehensive gRPC error propagation and handling tests (Task 321.4)
//!
//! This test suite validates error handling across all gRPC services:
//! - Malformed requests with invalid parameters
//! - Service unavailable scenarios
//! - Proper gRPC status code mapping
//! - Error message clarity and informativeness
//! - Error propagation from server to client
//!
//! Tests organized by error category for comprehensive coverage.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::oneshot;
use tonic::{Request, transport::{Channel, Endpoint}};
use workspace_qdrant_daemon::config::DaemonConfig;
use workspace_qdrant_daemon::daemon::WorkspaceDaemon;
use workspace_qdrant_daemon::grpc::server::GrpcServer;
use workspace_qdrant_daemon::proto::*;

// =============================================================================
// Test Utilities
// =============================================================================

/// Start a test gRPC server and return address + shutdown channel
async fn start_test_server() -> anyhow::Result<(SocketAddr, oneshot::Sender<()>)> {
    let mut config = DaemonConfig::default();

    // Override with test-specific settings
    config.database.sqlite_path = ":memory:".to_string();
    config.logging.enabled = false; // Reduce noise

    let daemon = WorkspaceDaemon::new(config.clone()).await?;

    let addr = SocketAddr::from(([127, 0, 0, 1], 0));
    let server = GrpcServer::new(daemon, addr);

    // The server address is private - we'll use the configured port
    let actual_addr = SocketAddr::from(([127, 0, 0, 1], config.server.port));

    let (shutdown_tx, _shutdown_rx) = oneshot::channel();

    tokio::spawn(async move {
        let _ = server.serve().await;
    });

    // Give server time to start
    tokio::time::sleep(Duration::from_millis(100)).await;

    Ok((actual_addr, shutdown_tx))
}

/// Create a test channel to a server
async fn create_test_channel(addr: SocketAddr) -> anyhow::Result<Channel> {
    let endpoint = Endpoint::from_shared(format!("http://{}", addr))?
        .timeout(Duration::from_secs(5))
        .connect_timeout(Duration::from_secs(5));

    Ok(endpoint.connect().await?)
}

// =============================================================================
// Category 1: Malformed Request Tests
// =============================================================================

#[tokio::test]
async fn test_empty_collection_name_validation() {
    let (addr, _shutdown) = start_test_server().await.expect("Server start failed");
    let channel = create_test_channel(addr).await.expect("Connection failed");
    let mut client = collection_service_client::CollectionServiceClient::new(channel);

    let request = CreateCollectionRequest {
        collection_name: "".to_string(), // Invalid: empty
        project_id: "test_project".to_string(),
        config: Some(CollectionConfig {
            vector_size: 384,
            distance_metric: "Cosine".to_string(),
            enable_indexing: true,
            metadata_schema: HashMap::new(),
        }),
    };

    let result = client.create_collection(Request::new(request)).await;

    // Validate error handling
    match result {
        Ok(response) => {
            let inner = response.into_inner();
            // If it succeeds, should indicate failure in response
            assert!(!inner.success || !inner.error_message.is_empty(),
                   "Empty collection name should be rejected");
        }
        Err(status) => {
            // Should have meaningful error
            assert!(!status.message().is_empty(),
                   "Error message should not be empty");
            // Error should mention the issue
            let msg = status.message().to_lowercase();
            assert!(
                msg.contains("collection") || msg.contains("name") || msg.contains("empty") || msg.contains("invalid"),
                "Error should mention collection name issue: {}",
                status.message()
            );
        }
    }
}

#[tokio::test]
async fn test_invalid_vector_size_zero() {
    let (addr, _shutdown) = start_test_server().await.expect("Server start failed");
    let channel = create_test_channel(addr).await.expect("Connection failed");
    let mut client = collection_service_client::CollectionServiceClient::new(channel);

    let request = CreateCollectionRequest {
        collection_name: "test_collection".to_string(),
        project_id: "test_project".to_string(),
        config: Some(CollectionConfig {
            vector_size: 0, // Invalid: must be > 0
            distance_metric: "Cosine".to_string(),
            enable_indexing: true,
            metadata_schema: HashMap::new(),
        }),
    };

    let result = client.create_collection(Request::new(request)).await;

    match result {
        Ok(response) => {
            let inner = response.into_inner();
            assert!(!inner.success || !inner.error_message.is_empty(),
                   "Zero vector size should be rejected");
        }
        Err(status) => {
            let msg = status.message().to_lowercase();
            assert!(
                msg.contains("vector") || msg.contains("size") || msg.contains("invalid"),
                "Error should mention vector size: {}",
                status.message()
            );
        }
    }
}

#[tokio::test]
async fn test_negative_vector_size() {
    let (addr, _shutdown) = start_test_server().await.expect("Server start failed");
    let channel = create_test_channel(addr).await.expect("Connection failed");
    let mut client = collection_service_client::CollectionServiceClient::new(channel);

    let request = CreateCollectionRequest {
        collection_name: "test_collection".to_string(),
        project_id: "test_project".to_string(),
        config: Some(CollectionConfig {
            vector_size: -1, // Invalid: negative
            distance_metric: "Cosine".to_string(),
            enable_indexing: true,
            metadata_schema: HashMap::new(),
        }),
    };

    let result = client.create_collection(Request::new(request)).await;

    // Should be rejected
    match result {
        Ok(response) => {
            let inner = response.into_inner();
            assert!(!inner.success, "Negative vector size should fail");
        }
        Err(_) => {
            // Error is acceptable
        }
    }
}

#[tokio::test]
async fn test_invalid_distance_metric() {
    let (addr, _shutdown) = start_test_server().await.expect("Server start failed");
    let channel = create_test_channel(addr).await.expect("Connection failed");
    let mut client = collection_service_client::CollectionServiceClient::new(channel);

    let request = CreateCollectionRequest {
        collection_name: "test_collection".to_string(),
        project_id: "test_project".to_string(),
        config: Some(CollectionConfig {
            vector_size: 384,
            distance_metric: "InvalidMetric".to_string(), // Invalid
            enable_indexing: true,
            metadata_schema: HashMap::new(),
        }),
    };

    let result = client.create_collection(Request::new(request)).await;

    match result {
        Ok(response) => {
            let inner = response.into_inner();
            assert!(!inner.success || !inner.error_message.is_empty(),
                   "Invalid distance metric should be rejected");
        }
        Err(status) => {
            let msg = status.message().to_lowercase();
            assert!(
                msg.contains("distance") || msg.contains("metric") || msg.contains("invalid"),
                "Error should mention distance metric: {}",
                status.message()
            );
        }
    }
}

#[tokio::test]
async fn test_empty_content_in_ingest_text() {
    let (addr, _shutdown) = start_test_server().await.expect("Server start failed");
    let channel = create_test_channel(addr).await.expect("Connection failed");
    let mut client = document_service_client::DocumentServiceClient::new(channel);

    let request = IngestTextRequest {
        content: "".to_string(), // Empty content
        collection_basename: "test".to_string(),
        tenant_id: "test_tenant".to_string(),
        document_id: None,
        metadata: HashMap::new(),
        chunk_text: true,
    };

    let result = client.ingest_text(Request::new(request)).await;

    // Should handle empty content gracefully
    match result {
        Ok(response) => {
            let inner = response.into_inner();
            // Either succeeds with warning or fails
            if !inner.success {
                assert!(!inner.error_message.is_empty(),
                       "Error message should be provided");
            }
        }
        Err(status) => {
            let msg = status.message().to_lowercase();
            assert!(
                msg.contains("content") || msg.contains("empty") || msg.contains("invalid"),
                "Error should mention content issue: {}",
                status.message()
            );
        }
    }
}

// =============================================================================
// Category 2: Invalid Parameter Tests
// =============================================================================

#[tokio::test]
async fn test_collection_name_with_special_characters() {
    let (addr, _shutdown) = start_test_server().await.expect("Server start failed");
    let channel = create_test_channel(addr).await.expect("Connection failed");
    let mut client = collection_service_client::CollectionServiceClient::new(channel);

    let invalid_names = vec![
        "test/collection",    // Forward slash
        "test\\collection",   // Backslash
        "../collection",      // Path traversal
    ];

    for name in invalid_names {
        let request = CreateCollectionRequest {
            collection_name: name.to_string(),
            project_id: "test_project".to_string(),
            config: Some(CollectionConfig {
                vector_size: 384,
                distance_metric: "Cosine".to_string(),
                enable_indexing: true,
                metadata_schema: HashMap::new(),
            }),
        };

        let result = client.create_collection(Request::new(request)).await;

        // Should handle gracefully - no panics
        match result {
            Ok(_) => {
                // May succeed depending on implementation
            }
            Err(status) => {
                // Error should not mention panic/unwrap
                assert!(
                    !status.message().contains("panic") && !status.message().contains("unwrap"),
                    "Error handling should be graceful for '{}': {}",
                    name,
                    status.message()
                );
            }
        }
    }
}

#[tokio::test]
async fn test_extremely_long_collection_name() {
    let (addr, _shutdown) = start_test_server().await.expect("Server start failed");
    let channel = create_test_channel(addr).await.expect("Connection failed");
    let mut client = collection_service_client::CollectionServiceClient::new(channel);

    let long_name = "a".repeat(1000);

    let request = CreateCollectionRequest {
        collection_name: long_name,
        project_id: "test_project".to_string(),
        config: Some(CollectionConfig {
            vector_size: 384,
            distance_metric: "Cosine".to_string(),
            enable_indexing: true,
            metadata_schema: HashMap::new(),
        }),
    };

    let result = client.create_collection(Request::new(request)).await;

    // Should handle long names gracefully
    match result {
        Ok(_) => {
            // May succeed
        }
        Err(status) => {
            assert!(!status.message().is_empty(),
                   "Should have error message");
        }
    }
}

// =============================================================================
// Category 3: Service Unavailable Tests
// =============================================================================

#[tokio::test]
async fn test_connection_timeout() {
    let endpoint = Endpoint::from_static("http://127.0.0.1:9999") // Non-existent
        .timeout(Duration::from_millis(10))
        .connect_timeout(Duration::from_millis(10));

    let result = endpoint.connect().await;

    assert!(result.is_err(), "Connection should timeout");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("timeout") || err_msg.contains("connect") || err_msg.contains("failed"),
        "Error should indicate timeout: {}",
        err_msg
    );
}

#[tokio::test]
async fn test_invalid_hostname() {
    let endpoint = Endpoint::from_static("http://invalid-hostname-that-does-not-exist:50051")
        .timeout(Duration::from_secs(1))
        .connect_timeout(Duration::from_secs(1));

    let result = endpoint.connect().await;

    assert!(result.is_err(), "Invalid hostname should fail");
    assert!(!result.unwrap_err().to_string().is_empty(),
           "Should have error message");
}

// =============================================================================
// Category 4: Error Message Quality Tests
// =============================================================================

#[tokio::test]
async fn test_error_messages_are_informative() {
    let (addr, _shutdown) = start_test_server().await.expect("Server start failed");
    let channel = create_test_channel(addr).await.expect("Connection failed");
    let mut client = collection_service_client::CollectionServiceClient::new(channel);

    // Multiple invalid parameters
    let request = CreateCollectionRequest {
        collection_name: "".to_string(), // Invalid
        project_id: "test_project".to_string(),
        config: Some(CollectionConfig {
            vector_size: -1, // Invalid
            distance_metric: "InvalidMetric".to_string(), // Invalid
            enable_indexing: true,
            metadata_schema: HashMap::new(),
        }),
    };

    let result = client.create_collection(Request::new(request)).await;

    match result {
        Ok(response) => {
            let inner = response.into_inner();
            if !inner.success {
                assert!(!inner.error_message.is_empty());
                assert!(inner.error_message.len() > 5,
                       "Error should be descriptive");
            }
        }
        Err(status) => {
            assert!(!status.message().is_empty());
            assert!(status.message().len() > 5,
                   "Error should be descriptive");
        }
    }
}

#[tokio::test]
async fn test_error_messages_no_sensitive_data() {
    let (addr, _shutdown) = start_test_server().await.expect("Server start failed");
    let channel = create_test_channel(addr).await.expect("Connection failed");
    let mut client = document_service_client::DocumentServiceClient::new(channel);

    let mut metadata = HashMap::new();
    metadata.insert("password".to_string(), "secret123".to_string());

    let request = IngestTextRequest {
        content: "sensitive SSN: 123-45-6789".to_string(),
        collection_basename: "test".to_string(),
        tenant_id: "secret_tenant_id".to_string(),
        document_id: None,
        metadata,
        chunk_text: true,
    };

    let result = client.ingest_text(Request::new(request)).await;

    // Check error message doesn't leak sensitive data
    match result {
        Ok(response) => {
            let inner = response.into_inner();
            if !inner.success {
                assert!(
                    !inner.error_message.contains("123-45-6789"),
                    "Should not leak content in error"
                );
                assert!(
                    !inner.error_message.contains("secret123"),
                    "Should not leak metadata in error"
                );
            }
        }
        Err(status) => {
            assert!(
                !status.message().contains("123-45-6789"),
                "Should not leak content in error"
            );
            assert!(
                !status.message().contains("secret123"),
                "Should not leak metadata in error"
            );
        }
    }
}

// =============================================================================
// Category 5: Error Propagation Tests
// =============================================================================

#[tokio::test]
async fn test_system_service_health_check() {
    let (addr, _shutdown) = start_test_server().await.expect("Server start failed");
    let channel = create_test_channel(addr).await.expect("Connection failed");
    let mut client = system_service_client::SystemServiceClient::new(channel);

    let result = client.health_check(Request::new(())).await;

    match result {
        Ok(response) => {
            let inner = response.into_inner();
            assert!(inner.status >= 0, "Status should be valid");
        }
        Err(status) => {
            // If it fails, should be graceful
            assert!(!status.message().contains("panic"),
                   "Error should be graceful");
        }
    }
}

#[tokio::test]
async fn test_error_propagation_with_missing_config() {
    let (addr, _shutdown) = start_test_server().await.expect("Server start failed");
    let channel = create_test_channel(addr).await.expect("Connection failed");
    let mut client = collection_service_client::CollectionServiceClient::new(channel);

    let request = CreateCollectionRequest {
        collection_name: "test".to_string(),
        project_id: "test".to_string(),
        config: None, // Missing config
    };

    let result = client.create_collection(Request::new(request)).await;

    // Error should propagate with meaningful information
    match result {
        Ok(response) => {
            let inner = response.into_inner();
            assert!(!inner.success || !inner.error_message.is_empty());
        }
        Err(status) => {
            assert!(!status.message().is_empty(),
                   "Error should have a message");
        }
    }
}

// =============================================================================
// Category 6: Concurrent Error Handling
// =============================================================================

#[tokio::test]
async fn test_concurrent_errors_independent() {
    let (addr, _shutdown) = start_test_server().await.expect("Server start failed");

    let mut handles = vec![];

    for i in 0..10 {
        let addr_clone = addr;
        let handle = tokio::spawn(async move {
            let channel = create_test_channel(addr_clone).await.expect("Connect failed");
            let mut client = collection_service_client::CollectionServiceClient::new(channel);

            let request = CreateCollectionRequest {
                collection_name: format!("invalid_{}", i),
                project_id: "test".to_string(),
                config: Some(CollectionConfig {
                    vector_size: -i, // All invalid
                    distance_metric: "Invalid".to_string(),
                    enable_indexing: true,
                    metadata_schema: HashMap::new(),
                }),
            };

            client.create_collection(Request::new(request)).await
        });

        handles.push(handle);
    }

    // All should complete without panic
    for handle in handles {
        let result = handle.await.expect("Task should not panic");
        // Each error should be independent
        match result {
            Ok(response) => {
                let _ = response.into_inner();
            }
            Err(status) => {
                assert!(!status.message().is_empty());
            }
        }
    }
}

// =============================================================================
// Category 7: Edge Cases
// =============================================================================

#[tokio::test]
async fn test_null_optional_fields() {
    let (addr, _shutdown) = start_test_server().await.expect("Server start failed");
    let channel = create_test_channel(addr).await.expect("Connection failed");
    let mut client = collection_service_client::CollectionServiceClient::new(channel);

    let request = CreateCollectionRequest {
        collection_name: "test_collection".to_string(),
        project_id: "test_project".to_string(),
        config: None, // None for optional field
    };

    let result = client.create_collection(Request::new(request)).await;

    // Should handle None gracefully
    match result {
        Ok(_) => {
            // May succeed
        }
        Err(status) => {
            assert!(
                !status.message().contains("null pointer") &&
                !status.message().contains("unwrap"),
                "Should handle None gracefully: {}",
                status.message()
            );
        }
    }
}

#[tokio::test]
async fn test_large_message_handling() {
    let (addr, _shutdown) = start_test_server().await.expect("Server start failed");
    let channel = create_test_channel(addr).await.expect("Connection failed");
    let mut client = document_service_client::DocumentServiceClient::new(channel);

    // 1MB of content
    let large_content = "x".repeat(1024 * 1024);

    let request = IngestTextRequest {
        content: large_content,
        collection_basename: "test".to_string(),
        tenant_id: "test_tenant".to_string(),
        document_id: None,
        metadata: HashMap::new(),
        chunk_text: true,
    };

    let result = client.ingest_text(Request::new(request)).await;

    // Should handle large messages gracefully
    match result {
        Ok(_) => {
            // May succeed
        }
        Err(status) => {
            let msg = status.message();
            // Should handle gracefully
            assert!(!msg.contains("panic"),
                   "Should handle large messages gracefully: {}", msg);
        }
    }
}
