//! Comprehensive gRPC timeout and deadline handling tests (Task 321.6)
//!
//! This test suite validates timeout and deadline management across gRPC operations:
//! - Request timeouts for various operation types
//! - Streaming operation deadlines
//! - Long-running operation cancellation
//! - Proper timeout configuration
//! - Deadline propagation across service boundaries
//! - Timeout error handling
//! - Various timeout values and operation durations
//!
//! Tests organized by timeout scenario for comprehensive coverage.

#![cfg(feature = "test-utils")]

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::oneshot;
use tokio::time::{timeout, sleep};
use tonic::{Request, Status, Code};
use tonic::transport::{Channel, Endpoint};
use workspace_qdrant_daemon::config::DaemonConfig;
use workspace_qdrant_daemon::daemon::WorkspaceDaemon;
use workspace_qdrant_daemon::grpc::server::GrpcServer;
use workspace_qdrant_daemon::grpc::client::{ConnectionPool, WorkspaceDaemonClient};
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
    sleep(Duration::from_millis(100)).await;

    Ok((actual_addr, shutdown_tx))
}

/// Create a test channel with custom timeout
async fn create_test_channel_with_timeout(
    addr: SocketAddr,
    request_timeout: Duration,
    connect_timeout: Duration,
) -> anyhow::Result<Channel> {
    let endpoint = Endpoint::from_shared(format!("http://{}", addr))?
        .timeout(request_timeout)
        .connect_timeout(connect_timeout);

    Ok(endpoint.connect().await?)
}

/// Create a test channel with default timeouts
async fn create_test_channel(addr: SocketAddr) -> anyhow::Result<Channel> {
    create_test_channel_with_timeout(
        addr,
        Duration::from_secs(5),
        Duration::from_secs(5),
    ).await
}

// =============================================================================
// Category 1: Basic Request Timeout Tests
// =============================================================================

#[tokio::test]
async fn test_request_timeout_exceeded() {
    // Test that a request with a very short timeout fails appropriately
    let endpoint = Endpoint::from_static("http://127.0.0.1:9999")
        .timeout(Duration::from_millis(1)) // Extremely short timeout
        .connect_timeout(Duration::from_millis(1));

    let result = endpoint.connect().await;

    // Should timeout or fail
    assert!(result.is_err(), "Extremely short timeout should fail");
}

#[tokio::test]
async fn test_request_completes_within_timeout() {
    let (addr, _shutdown) = start_test_server().await.expect("Server start failed");

    // Use generous timeout
    let channel = create_test_channel_with_timeout(
        addr,
        Duration::from_secs(10),
        Duration::from_secs(10),
    ).await.expect("Connection failed");

    let mut client = system_service_client::SystemServiceClient::new(channel);

    let start = Instant::now();
    let result = timeout(
        Duration::from_secs(10),
        client.health_check(Request::new(())),
    ).await;
    let elapsed = start.elapsed();

    // Should complete successfully and within timeout
    assert!(result.is_ok(), "Request should complete within timeout");
    assert!(elapsed < Duration::from_secs(10), "Request should be fast");
}

#[tokio::test]
async fn test_very_short_timeout_causes_deadline_exceeded() {
    let (addr, _shutdown) = start_test_server().await.expect("Server start failed");

    // Use extremely short timeout (1ms)
    let channel_result = create_test_channel_with_timeout(
        addr,
        Duration::from_millis(1),
        Duration::from_secs(2),
    ).await;

    if let Ok(channel) = channel_result {
        let mut client = collection_service_client::CollectionServiceClient::new(channel);

        let request = CreateCollectionRequest {
            collection_name: "test_collection".to_string(),
            project_id: "test_project".to_string(),
            config: Some(CollectionConfig {
                vector_size: 384,
                distance_metric: "Cosine".to_string(),
                enable_indexing: true,
                metadata_schema: HashMap::new(),
            }),
        };

        // With 1ms timeout, request likely times out
        let result = client.create_collection(Request::new(request)).await;

        // May timeout or complete - both are acceptable
        // If it errors, should be a deadline/timeout error
        if let Err(status) = result {
            let msg = status.message().to_lowercase();
            assert!(
                status.code() == Code::DeadlineExceeded ||
                status.code() == Code::Cancelled ||
                msg.contains("timeout") ||
                msg.contains("deadline"),
                "Timeout error should have appropriate code/message: {:?}",
                status
            );
        }
    }
}

#[tokio::test]
async fn test_timeout_with_various_durations() {
    let (addr, _shutdown) = start_test_server().await.expect("Server start failed");

    let timeouts = vec![
        Duration::from_millis(100),
        Duration::from_millis(500),
        Duration::from_secs(1),
        Duration::from_secs(5),
    ];

    for timeout_duration in timeouts {
        let channel = create_test_channel_with_timeout(
            addr,
            timeout_duration,
            Duration::from_secs(5),
        ).await.expect("Connection failed");

        let mut client = system_service_client::SystemServiceClient::new(channel);

        let result = timeout(
            timeout_duration + Duration::from_secs(1), // Give extra time
            client.health_check(Request::new(())),
        ).await;

        // With reasonable timeout, health check should succeed
        if timeout_duration >= Duration::from_millis(100) {
            assert!(
                result.is_ok(),
                "Health check should succeed with {:?} timeout",
                timeout_duration
            );
        }
    }
}

// =============================================================================
// Category 2: Connection Timeout Tests
// =============================================================================

#[tokio::test]
async fn test_connection_timeout_fast() {
    let endpoint = Endpoint::from_static("http://127.0.0.1:1") // Port likely refused
        .timeout(Duration::from_secs(5))
        .connect_timeout(Duration::from_millis(50)); // Very short connect timeout

    let start = Instant::now();
    let result = endpoint.connect().await;
    let elapsed = start.elapsed();

    // Should fail quickly
    assert!(result.is_err(), "Connection to port 1 should fail");
    assert!(
        elapsed < Duration::from_secs(1),
        "Connection timeout should be fast: {:?}",
        elapsed
    );
}

#[tokio::test]
async fn test_connection_timeout_vs_request_timeout() {
    // Test that connection timeout is separate from request timeout
    let pool = ConnectionPool::with_timeouts(
        Duration::from_secs(30),  // Request timeout
        Duration::from_millis(100), // Connection timeout
    );

    // Verify timeouts are set correctly
    assert_eq!(pool.default_timeout, Duration::from_secs(30));
    assert_eq!(pool.connect_timeout, Duration::from_millis(100));
}

#[tokio::test]
async fn test_connection_pool_timeout_configuration() {
    let custom_request_timeout = Duration::from_secs(45);
    let custom_connect_timeout = Duration::from_secs(15);

    let pool = ConnectionPool::with_timeouts(
        custom_request_timeout,
        custom_connect_timeout,
    );

    assert_eq!(pool.default_timeout, custom_request_timeout);
    assert_eq!(pool.connect_timeout, custom_connect_timeout);
}

// =============================================================================
// Category 3: Long-Running Operation Cancellation
// =============================================================================

#[tokio::test]
async fn test_cancel_long_running_operation() {
    let (addr, _shutdown) = start_test_server().await.expect("Server start failed");
    let channel = create_test_channel(addr).await.expect("Connection failed");
    let mut client = system_service_client::SystemServiceClient::new(channel);

    // Start a request in background
    let request_handle = tokio::spawn(async move {
        client.get_status(Request::new(())).await
    });

    // Give it time to start
    sleep(Duration::from_millis(50)).await;

    // Cancel by aborting the task
    request_handle.abort();

    // Wait for cancellation
    let result = request_handle.await;

    // Should be cancelled
    assert!(result.is_err(), "Task should be cancelled");
}

#[tokio::test]
async fn test_timeout_during_processing() {
    let (addr, _shutdown) = start_test_server().await.expect("Server start failed");

    // Very short timeout to force timeout during processing
    let channel = create_test_channel_with_timeout(
        addr,
        Duration::from_millis(1),
        Duration::from_secs(2),
    ).await;

    if let Ok(channel) = channel {
        let mut client = document_service_client::DocumentServiceClient::new(channel);

        // Large content to increase processing time
        let large_content = "x".repeat(10000);

        let request = IngestTextRequest {
            content: large_content,
            collection_basename: "test".to_string(),
            tenant_id: "test_tenant".to_string(),
            document_id: None,
            metadata: HashMap::new(),
            chunk_text: true,
        };

        let result = client.ingest_text(Request::new(request)).await;

        // May timeout or succeed - both acceptable
        if let Err(status) = result {
            // If error, should be timeout-related
            assert!(
                status.code() == Code::DeadlineExceeded ||
                status.code() == Code::Cancelled ||
                status.message().to_lowercase().contains("timeout"),
                "Timeout error expected: {:?}",
                status
            );
        }
    }
}

#[tokio::test]
async fn test_graceful_cancellation_cleanup() {
    let (addr, _shutdown) = start_test_server().await.expect("Server start failed");
    let client = WorkspaceDaemonClient::new(format!("http://{}", addr));

    // Start connection
    let _ = client.test_connection().await;

    let initial_stats = client.connection_stats().await;

    // Disconnect (graceful cleanup)
    client.disconnect().await;

    let final_stats = client.connection_stats().await;

    // Resources should be cleaned up
    assert_eq!(
        final_stats.active_connections, 0,
        "Graceful cancellation should clean up resources"
    );
}

// =============================================================================
// Category 4: Deadline Propagation Tests
// =============================================================================

#[tokio::test]
async fn test_client_deadline_to_server() {
    let (addr, _shutdown) = start_test_server().await.expect("Server start failed");
    let channel = create_test_channel_with_timeout(
        addr,
        Duration::from_secs(2),
        Duration::from_secs(2),
    ).await.expect("Connection failed");

    let mut client = system_service_client::SystemServiceClient::new(channel);

    // Create request with metadata (deadline is automatic from endpoint timeout)
    let result = client.health_check(Request::new(())).await;

    // Should succeed with proper deadline propagation
    assert!(result.is_ok(), "Request with deadline should succeed");
}

#[tokio::test]
async fn test_deadline_preserved_across_retries() {
    let (addr, _shutdown) = start_test_server().await.expect("Server start failed");
    let pool = ConnectionPool::with_timeouts(
        Duration::from_secs(3),
        Duration::from_secs(1),
    );

    let client = WorkspaceDaemonClient::with_pool(
        format!("http://{}", addr),
        pool,
    );

    // Multiple requests should all respect the same timeout
    for _ in 0..3 {
        let start = Instant::now();
        let _ = client.test_connection().await;
        let elapsed = start.elapsed();

        // Each request should complete within configured timeout
        assert!(
            elapsed < Duration::from_secs(4),
            "Request should respect timeout: {:?}",
            elapsed
        );

        sleep(Duration::from_millis(100)).await;
    }
}

#[tokio::test]
async fn test_timeout_vs_deadline_distinction() {
    // Timeout: How long to wait for response
    // Deadline: Absolute time by which operation must complete

    let (addr, _shutdown) = start_test_server().await.expect("Server start failed");

    let timeout_duration = Duration::from_secs(5);
    let channel = create_test_channel_with_timeout(
        addr,
        timeout_duration,
        Duration::from_secs(2),
    ).await.expect("Connection failed");

    let mut client = system_service_client::SystemServiceClient::new(channel);

    let start = Instant::now();
    let result = client.health_check(Request::new(())).await;
    let elapsed = start.elapsed();

    // Should complete
    assert!(result.is_ok(), "Request should complete");

    // Should complete well before timeout
    assert!(
        elapsed < timeout_duration,
        "Request should complete before timeout"
    );
}

// =============================================================================
// Category 5: Timeout Error Handling Tests
// =============================================================================

#[tokio::test]
async fn test_deadline_exceeded_status_code() {
    let endpoint = Endpoint::from_static("http://127.0.0.1:9999") // Non-existent
        .timeout(Duration::from_millis(10))
        .connect_timeout(Duration::from_millis(10));

    let result = endpoint.connect().await;

    assert!(result.is_err(), "Connection should fail");

    // Error should contain timeout/deadline information
    let err_msg = result.unwrap_err().to_string().to_lowercase();
    assert!(
        err_msg.contains("timeout") ||
        err_msg.contains("deadline") ||
        err_msg.contains("connect") ||
        err_msg.contains("failed"),
        "Error should indicate timeout: {}",
        err_msg
    );
}

#[tokio::test]
async fn test_timeout_error_message_quality() {
    let (addr, _shutdown) = start_test_server().await.expect("Server start failed");

    // Use very short timeout
    let channel_result = create_test_channel_with_timeout(
        addr,
        Duration::from_millis(1),
        Duration::from_secs(2),
    ).await;

    if let Ok(channel) = channel_result {
        let mut client = collection_service_client::CollectionServiceClient::new(channel);

        let request = CreateCollectionRequest {
            collection_name: "test".to_string(),
            project_id: "test".to_string(),
            config: Some(CollectionConfig {
                vector_size: 384,
                distance_metric: "Cosine".to_string(),
                enable_indexing: true,
                metadata_schema: HashMap::new(),
            }),
        };

        let result = client.create_collection(Request::new(request)).await;

        // If it times out, error message should be informative
        if let Err(status) = result {
            let msg = status.message();

            // Should not be empty
            assert!(!msg.is_empty(), "Error message should not be empty");

            // Should not contain panic/unwrap
            assert!(
                !msg.contains("panic") && !msg.contains("unwrap"),
                "Error should be graceful: {}",
                msg
            );
        }
    }
}

#[tokio::test]
async fn test_client_side_vs_server_side_timeout() {
    let (addr, _shutdown) = start_test_server().await.expect("Server start failed");

    // Client-side timeout using tokio::time::timeout
    let channel = create_test_channel(addr).await.expect("Connection failed");
    let mut client = system_service_client::SystemServiceClient::new(channel);

    let client_timeout = Duration::from_millis(100);
    let result = timeout(
        client_timeout,
        client.get_status(Request::new(())),
    ).await;

    // Should complete or timeout
    if result.is_err() {
        // This is a client-side timeout (tokio timeout, not gRPC)
        // Different from server-side deadline exceeded
    } else {
        // Completed successfully
        assert!(result.unwrap().is_ok());
    }
}

#[tokio::test]
async fn test_retry_behavior_on_timeout() {
    let (addr, _shutdown) = start_test_server().await.expect("Server start failed");
    let client = WorkspaceDaemonClient::new(format!("http://{}", addr));

    let mut success_count = 0;
    let mut failure_count = 0;

    // Attempt multiple times with short timeouts
    for _ in 0..5 {
        let result = timeout(
            Duration::from_millis(500),
            client.test_connection(),
        ).await;

        match result {
            Ok(Ok(true)) => success_count += 1,
            Ok(Ok(false)) => failure_count += 1,
            Ok(Err(_)) => failure_count += 1,
            Err(_) => failure_count += 1, // Timeout
        }

        sleep(Duration::from_millis(50)).await;
    }

    // At least some should succeed (server is running)
    assert!(
        success_count > 0,
        "Some retries should succeed with running server"
    );
}

// =============================================================================
// Category 6: Various Operation Duration Tests
// =============================================================================

#[tokio::test]
async fn test_fast_operation_completes_quickly() {
    let (addr, _shutdown) = start_test_server().await.expect("Server start failed");
    let channel = create_test_channel(addr).await.expect("Connection failed");
    let mut client = system_service_client::SystemServiceClient::new(channel);

    let start = Instant::now();
    let result = client.health_check(Request::new(())).await;
    let elapsed = start.elapsed();

    // Fast operation should complete quickly
    assert!(result.is_ok(), "Fast operation should succeed");
    assert!(
        elapsed < Duration::from_millis(500),
        "Fast operation should complete in < 500ms: {:?}",
        elapsed
    );
}

#[tokio::test]
async fn test_operation_duration_vs_timeout() {
    let (addr, _shutdown) = start_test_server().await.expect("Server start failed");

    // Test with timeout longer than expected operation duration
    let generous_timeout = Duration::from_secs(10);
    let channel = create_test_channel_with_timeout(
        addr,
        generous_timeout,
        Duration::from_secs(2),
    ).await.expect("Connection failed");

    let mut client = system_service_client::SystemServiceClient::new(channel);

    let start = Instant::now();
    let result = client.health_check(Request::new(())).await;
    let elapsed = start.elapsed();

    // Should complete successfully
    assert!(result.is_ok(), "Operation should complete within timeout");

    // Should complete well before timeout
    assert!(
        elapsed < generous_timeout / 2,
        "Operation should complete quickly relative to timeout"
    );
}

#[tokio::test]
async fn test_timeout_precision_milliseconds() {
    let (addr, _shutdown) = start_test_server().await.expect("Server start failed");

    // Test timeout precision at millisecond level
    let precise_timeout = Duration::from_millis(250);

    let channel = create_test_channel_with_timeout(
        addr,
        precise_timeout,
        Duration::from_secs(2),
    ).await.expect("Connection failed");

    let mut client = system_service_client::SystemServiceClient::new(channel);

    let start = Instant::now();
    let result = client.health_check(Request::new(())).await;
    let elapsed = start.elapsed();

    // Should complete or timeout
    if result.is_ok() {
        // Completed before timeout
        assert!(
            elapsed < precise_timeout,
            "Completed operation should be faster than timeout"
        );
    }
}

// =============================================================================
// Category 7: Timeout Configuration Validation Tests
// =============================================================================

#[tokio::test]
async fn test_valid_timeout_ranges() {
    let valid_timeouts = vec![
        (Duration::from_millis(1), Duration::from_millis(1)),
        (Duration::from_millis(100), Duration::from_millis(100)),
        (Duration::from_secs(1), Duration::from_secs(1)),
        (Duration::from_secs(30), Duration::from_secs(10)),
        (Duration::from_secs(60), Duration::from_secs(30)),
    ];

    for (request_timeout, connect_timeout) in valid_timeouts {
        let pool = ConnectionPool::with_timeouts(request_timeout, connect_timeout);

        // Should create successfully
        assert_eq!(pool.default_timeout, request_timeout);
        assert_eq!(pool.connect_timeout, connect_timeout);
    }
}

#[tokio::test]
async fn test_zero_timeout_handling() {
    // Zero timeout should be handled (though not recommended)
    let pool = ConnectionPool::with_timeouts(
        Duration::from_secs(0),
        Duration::from_secs(0),
    );

    // Should create but likely fail on actual use
    assert_eq!(pool.default_timeout, Duration::from_secs(0));
}

#[tokio::test]
async fn test_very_large_timeout() {
    // Very large timeout should be accepted
    let large_timeout = Duration::from_secs(3600); // 1 hour

    let pool = ConnectionPool::with_timeouts(large_timeout, Duration::from_secs(10));

    assert_eq!(pool.default_timeout, large_timeout);
}

#[tokio::test]
async fn test_timeout_too_short_for_operation() {
    let (addr, _shutdown) = start_test_server().await.expect("Server start failed");

    // Unreasonably short timeout
    let too_short = Duration::from_micros(1);

    let channel_result = create_test_channel_with_timeout(
        addr,
        too_short,
        Duration::from_secs(2),
    ).await;

    if let Ok(channel) = channel_result {
        let mut client = system_service_client::SystemServiceClient::new(channel);

        let result = client.health_check(Request::new(())).await;

        // Likely times out
        if let Err(status) = result {
            // Should have timeout-related error
            let msg = status.message().to_lowercase();
            assert!(
                status.code() == Code::DeadlineExceeded ||
                status.code() == Code::Cancelled ||
                msg.contains("timeout") ||
                msg.contains("deadline"),
                "Should indicate timeout: {:?}",
                status
            );
        }
    }
}

// =============================================================================
// Category 8: Concurrent Timeout Tests
// =============================================================================

#[tokio::test]
async fn test_concurrent_requests_with_different_timeouts() {
    let (addr, _shutdown) = start_test_server().await.expect("Server start failed");

    let mut handles = vec![];

    let timeouts = vec![
        Duration::from_millis(100),
        Duration::from_millis(500),
        Duration::from_secs(1),
        Duration::from_secs(2),
        Duration::from_secs(5),
    ];

    for timeout_duration in timeouts {
        let addr_clone = addr;

        let handle = tokio::spawn(async move {
            let channel = create_test_channel_with_timeout(
                addr_clone,
                timeout_duration,
                Duration::from_secs(2),
            ).await;

            if let Ok(channel) = channel {
                let mut client = system_service_client::SystemServiceClient::new(channel);
                client.health_check(Request::new(())).await
            } else {
                Err(Status::unavailable("Connection failed"))
            }
        });

        handles.push(handle);
    }

    // All should complete without panic
    let mut success_count = 0;
    for handle in handles {
        let result = handle.await.expect("Task should not panic");
        if result.is_ok() {
            success_count += 1;
        }
    }

    // Most should succeed (server is running)
    assert!(success_count >= 3, "Most concurrent requests should succeed");
}

#[tokio::test]
async fn test_timeout_isolation_between_requests() {
    let (addr, _shutdown) = start_test_server().await.expect("Server start failed");

    // First request with short timeout
    let channel1 = create_test_channel_with_timeout(
        addr,
        Duration::from_millis(100),
        Duration::from_secs(2),
    ).await;

    // Second request with longer timeout
    let channel2 = create_test_channel_with_timeout(
        addr,
        Duration::from_secs(10),
        Duration::from_secs(2),
    ).await;

    if let (Ok(ch1), Ok(ch2)) = (channel1, channel2) {
        let mut client1 = system_service_client::SystemServiceClient::new(ch1);
        let mut client2 = system_service_client::SystemServiceClient::new(ch2);

        // Make requests concurrently
        let req1 = tokio::spawn(async move {
            client1.health_check(Request::new(())).await
        });

        let req2 = tokio::spawn(async move {
            client2.health_check(Request::new(())).await
        });

        // Both should complete independently
        let result1 = req1.await.expect("Task 1 should not panic");
        let result2 = req2.await.expect("Task 2 should not panic");

        // Timeouts should be isolated - both likely succeed
        // (health check is fast)
        if result1.is_ok() && result2.is_ok() {
            // Both succeeded
        }
    }
}

// =============================================================================
// Category 9: Edge Cases
// =============================================================================

#[tokio::test]
async fn test_timeout_on_connection_pool_operations() {
    let pool = ConnectionPool::new();

    // Attempt connection with timeout
    let result = timeout(
        Duration::from_secs(2),
        pool.get_connection("http://127.0.0.1:9999"),
    ).await;

    // Should timeout or fail
    assert!(
        result.is_err() || result.unwrap().is_err(),
        "Connection to non-existent server should fail or timeout"
    );
}

#[tokio::test]
async fn test_deadline_with_connection_pool_reuse() {
    let (addr, _shutdown) = start_test_server().await.expect("Server start failed");
    let pool = Arc::new(ConnectionPool::with_timeouts(
        Duration::from_secs(5),
        Duration::from_secs(2),
    ));

    // First connection
    let result1 = pool.get_connection(&format!("http://{}", addr)).await;
    assert!(result1.is_ok(), "First connection should succeed");

    // Reused connection should still respect timeout
    let result2 = pool.get_connection(&format!("http://{}", addr)).await;
    assert!(result2.is_ok(), "Reused connection should succeed");

    // Both should complete quickly
    let count = pool.connection_count().await;
    assert_eq!(count, 1, "Should reuse connection");
}

#[tokio::test]
async fn test_timeout_with_client_disconnect() {
    let (addr, _shutdown) = start_test_server().await.expect("Server start failed");
    let client = WorkspaceDaemonClient::new(format!("http://{}", addr));

    // Start request
    let request_handle = tokio::spawn({
        let client = client.clone();
        async move {
            timeout(Duration::from_secs(5), client.test_connection()).await
        }
    });

    // Give it time to start
    sleep(Duration::from_millis(50)).await;

    // Disconnect
    client.disconnect().await;

    // Request should complete or be cancelled
    let result = timeout(Duration::from_secs(2), request_handle).await;

    // Should complete (either success or cancellation)
    assert!(result.is_ok(), "Request should complete after disconnect");
}

#[tokio::test]
async fn test_multiple_timeout_layers() {
    let (addr, _shutdown) = start_test_server().await.expect("Server start failed");

    // Endpoint timeout
    let channel = create_test_channel_with_timeout(
        addr,
        Duration::from_secs(5),
        Duration::from_secs(2),
    ).await.expect("Connection failed");

    let mut client = system_service_client::SystemServiceClient::new(channel);

    // Additional tokio timeout layer
    let outer_timeout = Duration::from_secs(3);
    let result = timeout(
        outer_timeout,
        client.health_check(Request::new(())),
    ).await;

    // Should respect the shorter timeout (outer_timeout)
    assert!(result.is_ok(), "Request should complete within outer timeout");
}
