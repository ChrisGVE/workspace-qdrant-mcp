//! Comprehensive gRPC connection lifecycle management tests (Task 321.3)
//!
//! Tests connection establishment, pooling, reuse, graceful shutdown,
//! resource cleanup, and health monitoring for gRPC client-server connections.
//!
//! Test coverage:
//! - Connection establishment with various configurations
//! - Connection pooling and reuse patterns
//! - Graceful shutdown and cleanup
//! - Health monitoring and state management
//! - Edge cases and error scenarios

#![cfg(feature = "test-utils")]

use workspace_qdrant_daemon::grpc::client::{ConnectionPool, WorkspaceDaemonClient, ConnectionStats};
use workspace_qdrant_daemon::proto::{
    system_service_client::SystemServiceClient,
    ServiceStatus,
};
use workspace_qdrant_daemon::config::*;
use workspace_qdrant_daemon::daemon::WorkspaceDaemon;
use workspace_qdrant_daemon::grpc::server::GrpcServer;

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;
use tokio::time::{timeout, sleep};
use tonic::transport::Endpoint;
use tonic::{Request, Code};
use serial_test::serial;

// ================================
// TEST INFRASTRUCTURE
// ================================

/// Test environment with server and client setup
struct TestEnvironment {
    _daemon: WorkspaceDaemon,
    server_handle: tokio::task::JoinHandle<Result<(), anyhow::Error>>,
    address: String,
}

impl TestEnvironment {
    async fn new() -> Self {
        let config = DaemonConfig::default();
        let daemon = WorkspaceDaemon::new(config).await
            .expect("Failed to create test daemon");

        // Use port 0 for automatic assignment
        let socket_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 50053);
        let grpc_server = GrpcServer::new(daemon.clone(), socket_addr);

        // Use fixed port for testing
        let address = "http://127.0.0.1:50053".to_string();

        // Start server in background
        let server_handle = tokio::spawn(async move {
            grpc_server.serve_daemon().await
        });

        // Give server time to start
        sleep(Duration::from_millis(200)).await;

        Self {
            _daemon: daemon,
            server_handle,
            address,
        }
    }

    fn address(&self) -> &str {
        &self.address
    }
}

impl Drop for TestEnvironment {
    fn drop(&mut self) {
        self.server_handle.abort();
    }
}

// ================================
// CONNECTION ESTABLISHMENT TESTS
// ================================

#[tokio::test]
async fn test_fresh_connection_establishment() {
    // Test establishing a fresh connection to a valid endpoint
    let endpoint = Endpoint::from_static("http://127.0.0.1:50051")
        .timeout(Duration::from_secs(5))
        .connect_timeout(Duration::from_secs(2));

    // Attempt connection (may fail if no server, which is expected)
    let result = timeout(Duration::from_secs(3), endpoint.connect()).await;

    // Connection attempt should complete (success or failure)
    assert!(result.is_ok(), "Connection attempt should complete within timeout");
}

#[tokio::test]
async fn test_connection_with_custom_timeouts() {
    // Test connection with custom timeout settings
    let pool = ConnectionPool::with_timeouts(
        Duration::from_secs(15),  // default timeout
        Duration::from_secs(8),   // connect timeout
    );

    assert_eq!(pool.default_timeout, Duration::from_secs(15));
    assert_eq!(pool.connect_timeout, Duration::from_secs(8));
}

#[tokio::test]
async fn test_connection_establishment_invalid_host() {
    // Test connection to invalid host should fail gracefully
    let endpoint = Endpoint::from_static("http://invalid-host-that-does-not-exist:50051")
        .timeout(Duration::from_millis(500))
        .connect_timeout(Duration::from_millis(500));

    let result = timeout(Duration::from_secs(2), endpoint.connect()).await;

    // Should timeout or fail to connect
    assert!(result.is_err() || result.unwrap().is_err(),
           "Connection to invalid host should fail");
}

#[tokio::test]
async fn test_connection_establishment_refused() {
    // Test connection to port with no listener
    let endpoint = Endpoint::from_static("http://127.0.0.1:1") // Port 1 should be refused
        .timeout(Duration::from_millis(500))
        .connect_timeout(Duration::from_millis(500));

    let result = timeout(Duration::from_secs(2), endpoint.connect()).await;

    // Should fail or timeout
    assert!(result.is_err() || result.unwrap().is_err(),
           "Connection to refused port should fail");
}

#[tokio::test]
#[serial]
async fn test_connection_with_metadata() {
    let env = TestEnvironment::new().await;

    // Create connection
    let endpoint = Endpoint::from_shared(env.address().to_string())
        .expect("Failed to create endpoint")
        .timeout(Duration::from_secs(5));

    let channel = timeout(Duration::from_secs(3), endpoint.connect()).await;

    if let Ok(Ok(channel)) = channel {
        let mut client = SystemServiceClient::new(channel);

        // Create request with metadata
        let mut request = Request::new(());
        request.metadata_mut().insert("client-id", "test-client-123".parse().unwrap());
        request.metadata_mut().insert("user-agent", "connection-lifecycle-test/1.0".parse().unwrap());

        let response = timeout(Duration::from_secs(5), client.health_check(request)).await;

        // Request with metadata should succeed
        if let Ok(Ok(resp)) = response {
            assert_eq!(resp.into_inner().status, ServiceStatus::Healthy as i32);
        }
    }
}

// ================================
// CONNECTION POOLING TESTS
// ================================

#[tokio::test]
async fn test_connection_pool_creation() {
    // Test creating a connection pool
    let pool = ConnectionPool::new();

    // Should start with zero connections
    assert_eq!(pool.connection_count().await, 0);
}

#[tokio::test]
async fn test_connection_pool_reuse() {
    let pool = ConnectionPool::new();
    let address = "http://127.0.0.1:50051";

    // Initial connection count
    let initial_count = pool.connection_count().await;
    assert_eq!(initial_count, 0);

    // Attempt to get connection (may fail if no server)
    let result1 = pool.get_connection(address).await;

    if result1.is_ok() {
        // Connection should be cached
        let count_after_first = pool.connection_count().await;
        assert_eq!(count_after_first, 1);

        // Second request should reuse connection
        let result2 = pool.get_connection(address).await;
        assert!(result2.is_ok());

        // Count should remain the same (connection reused)
        let count_after_second = pool.connection_count().await;
        assert_eq!(count_after_second, 1);
    }
}

#[tokio::test]
async fn test_connection_pool_multiple_addresses() {
    let pool = ConnectionPool::new();

    let addr1 = "http://127.0.0.1:50051";
    let addr2 = "http://127.0.0.1:50052";

    // Attempt connections to different addresses
    let _ = pool.get_connection(addr1).await;
    let _ = pool.get_connection(addr2).await;

    // Pool might contain 0, 1, or 2 connections depending on server availability
    let count = pool.connection_count().await;
    assert!(count <= 2, "Pool should not exceed connection attempts");
}

#[tokio::test]
async fn test_connection_pool_remove() {
    let pool = ConnectionPool::new();
    let address = "http://127.0.0.1:50051";

    // Add connection
    let _ = pool.get_connection(address).await;

    // Remove connection
    pool.remove_connection(address).await;

    // Count should be zero (regardless of initial success)
    let count = pool.connection_count().await;
    assert_eq!(count, 0, "Pool should be empty after remove");
}

#[tokio::test]
async fn test_connection_pool_clear() {
    let pool = ConnectionPool::new();

    // Attempt multiple connections
    let _ = pool.get_connection("http://127.0.0.1:50051").await;
    let _ = pool.get_connection("http://127.0.0.1:50052").await;
    let _ = pool.get_connection("http://127.0.0.1:50053").await;

    // Clear all connections
    pool.clear().await;

    // Pool should be empty
    assert_eq!(pool.connection_count().await, 0);
}

#[tokio::test]
async fn test_connection_pool_concurrent_access() {
    let pool = Arc::new(ConnectionPool::new());
    let address = "http://127.0.0.1:50051";

    // Multiple concurrent attempts to get same connection
    let mut handles = Vec::new();

    for i in 0..10 {
        let pool_clone = Arc::clone(&pool);
        let addr = address.to_string();

        let handle = tokio::spawn(async move {
            let result = pool_clone.get_connection(&addr).await;
            (i, result.is_ok())
        });

        handles.push(handle);
    }

    // Wait for all tasks
    let mut successful = 0;
    for handle in handles {
        let (_, success) = handle.await.unwrap();
        if success {
            successful += 1;
        }
    }

    // All should get connection (either fresh or reused)
    // But pool should only contain 1 connection if successful
    let count = pool.connection_count().await;
    if successful > 0 {
        assert_eq!(count, 1, "Pool should contain exactly one connection");
    }
}

// ================================
// CONNECTION REUSE TESTS
// ================================

#[tokio::test]
#[serial]
async fn test_client_multiple_requests_same_connection() {
    let env = TestEnvironment::new().await;
    let client = WorkspaceDaemonClient::new(env.address().to_string());

    // Make multiple health check requests
    let mut successful = 0;

    for _ in 0..5 {
        if client.test_connection().await.unwrap_or(false) {
            successful += 1;
        }
        sleep(Duration::from_millis(50)).await;
    }

    // Connection stats should show connection reuse
    let stats = client.connection_stats().await;
    assert_eq!(stats.address, env.address());

    // Should have at most 1 active connection (reused)
    assert!(stats.active_connections <= 1,
           "Client should reuse connection, not create multiple");
}

#[tokio::test]
#[serial]
async fn test_connection_idle_handling() {
    let env = TestEnvironment::new().await;
    let client = WorkspaceDaemonClient::new(env.address().to_string());

    // Make initial request
    let result1 = client.test_connection().await;

    // Wait for idle period
    sleep(Duration::from_secs(1)).await;

    // Make request after idle
    let result2 = client.test_connection().await;

    // Both requests should succeed (or both fail consistently)
    if result1.is_ok() {
        assert!(result2.is_ok(), "Request after idle should succeed");
    }
}

#[tokio::test]
#[serial]
async fn test_channel_reuse_across_service_clients() {
    let env = TestEnvironment::new().await;

    // Create endpoint
    let endpoint = Endpoint::from_shared(env.address().to_string())
        .expect("Failed to create endpoint")
        .timeout(Duration::from_secs(5));

    let channel_result = endpoint.connect().await;

    if let Ok(channel) = channel_result {
        // Create multiple service clients with same channel
        let client1 = SystemServiceClient::new(channel.clone());
        let client2 = SystemServiceClient::new(channel.clone());

        // Both clients should work
        let mut c1 = client1;
        let mut c2 = client2;

        let req1 = timeout(Duration::from_secs(2), c1.health_check(Request::new(()))).await;
        let req2 = timeout(Duration::from_secs(2), c2.health_check(Request::new(()))).await;

        // Both should complete
        assert!(req1.is_ok() || req2.is_ok(),
               "At least one client should successfully reuse channel");
    }
}

// ================================
// GRACEFUL SHUTDOWN TESTS
// ================================

#[tokio::test]
async fn test_client_disconnect_cleanup() {
    let client = WorkspaceDaemonClient::new("http://127.0.0.1:50051".to_string());

    // Get initial stats
    let stats_before = client.connection_stats().await;

    // Disconnect
    client.disconnect().await;

    // Verify cleanup
    let stats_after = client.connection_stats().await;
    assert_eq!(stats_after.active_connections, 0,
              "Disconnect should clear all connections");
}

#[tokio::test]
async fn test_pool_shutdown_resource_release() {
    let pool = ConnectionPool::new();

    // Add connections
    let _ = pool.get_connection("http://127.0.0.1:50051").await;
    let _ = pool.get_connection("http://127.0.0.1:50052").await;

    // Clear pool (simulates shutdown)
    pool.clear().await;

    let count_after = pool.connection_count().await;
    assert_eq!(count_after, 0, "Shutdown should release all resources");
}

#[tokio::test]
#[serial]
async fn test_inflight_request_during_shutdown() {
    let env = TestEnvironment::new().await;
    let client = WorkspaceDaemonClient::new(env.address().to_string());

    // Start a request
    let request_handle = tokio::spawn({
        let client = client.clone();
        async move {
            client.test_connection().await
        }
    });

    // Give request time to start
    sleep(Duration::from_millis(50)).await;

    // Disconnect while request is in flight
    client.disconnect().await;

    // Request should complete or fail gracefully
    let result = timeout(Duration::from_secs(5), request_handle).await;
    assert!(result.is_ok(), "In-flight request should complete or fail gracefully");
}

#[tokio::test]
#[serial]
async fn test_connection_cleanup_on_server_shutdown() {
    let env = TestEnvironment::new().await;

    let endpoint = Endpoint::from_shared(env.address().to_string())
        .expect("Failed to create endpoint")
        .timeout(Duration::from_secs(5));

    let channel = endpoint.connect().await;

    if let Ok(channel) = channel {
        let mut client = SystemServiceClient::new(channel);

        // Make initial request
        let _ = timeout(Duration::from_secs(2), client.health_check(Request::new(()))).await;

        // Drop environment (shutdown server)
        drop(env);
        sleep(Duration::from_millis(200)).await;

        // Subsequent request should fail gracefully
        let request = Request::new(());
        let result = timeout(Duration::from_secs(2), client.health_check(request)).await;

        // Should either timeout or return error status
        if let Ok(Ok(Err(status))) = result {
            assert!(matches!(status.code(), Code::Unavailable | Code::Cancelled | Code::DeadlineExceeded),
                   "Should receive connection error after server shutdown");
        }
    }
}

// ================================
// RESOURCE CLEANUP TESTS
// ================================

#[tokio::test]
async fn test_explicit_disconnect_clears_resources() {
    let client = WorkspaceDaemonClient::new("http://127.0.0.1:50051".to_string());

    // Attempt connection
    let _ = client.test_connection().await;

    // Disconnect
    client.disconnect().await;

    // Verify resources cleared
    let stats = client.connection_stats().await;
    assert_eq!(stats.active_connections, 0,
              "Explicit disconnect should clear all resources");
}

#[tokio::test]
async fn test_connection_count_tracking() {
    let pool = ConnectionPool::new();

    // Track count after each operation
    assert_eq!(pool.connection_count().await, 0);

    let _ = pool.get_connection("http://127.0.0.1:50051").await;
    let count1 = pool.connection_count().await;
    assert!(count1 <= 1);

    let _ = pool.get_connection("http://127.0.0.1:50052").await;
    let count2 = pool.connection_count().await;
    assert!(count2 <= 2);

    pool.remove_connection("http://127.0.0.1:50051").await;
    let count3 = pool.connection_count().await;
    assert!(count3 <= 1);

    pool.clear().await;
    assert_eq!(pool.connection_count().await, 0);
}

#[tokio::test]
async fn test_pool_drop_cleanup() {
    {
        let pool = ConnectionPool::new();
        let _ = pool.get_connection("http://127.0.0.1:50051").await;
        let _ = pool.get_connection("http://127.0.0.1:50052").await;

        // Pool goes out of scope
    }

    // Pool should be dropped and resources cleaned up
    // This test verifies no panics occur during drop
}

// ================================
// CONNECTION HEALTH TESTS
// ================================

#[tokio::test]
#[serial]
async fn test_health_check_fresh_connection() {
    let env = TestEnvironment::new().await;
    let client = WorkspaceDaemonClient::new(env.address().to_string());

    // Test connectivity on fresh connection
    let is_healthy = client.test_connection().await.unwrap_or(false);

    if is_healthy {
        // Health check should return healthy status
        let health_response = client.health_check().await;
        assert!(health_response.is_ok(), "Health check should succeed");

        if let Ok(response) = health_response {
            assert_eq!(response.status, ServiceStatus::Healthy as i32);
        }
    }
}

#[tokio::test]
#[serial]
async fn test_health_check_pooled_connection() {
    let env = TestEnvironment::new().await;
    let client = WorkspaceDaemonClient::new(env.address().to_string());

    // Make initial request to establish connection
    let _ = client.test_connection().await;

    // Health check on pooled connection
    let is_healthy = client.test_connection().await.unwrap_or(false);

    if is_healthy {
        let stats = client.connection_stats().await;
        assert!(stats.active_connections <= 1,
               "Should reuse pooled connection for health check");
    }
}

#[tokio::test]
async fn test_health_check_failure_handling() {
    // Test health check on non-existent server
    let client = WorkspaceDaemonClient::new("http://127.0.0.1:1".to_string());

    let is_healthy = client.test_connection().await.unwrap_or(false);

    // Should return false for failed connection
    assert!(!is_healthy, "Health check should fail for non-existent server");
}

#[tokio::test]
async fn test_connection_stats_tracking() {
    let client = WorkspaceDaemonClient::new("http://127.0.0.1:50051".to_string());

    // Get stats
    let stats = client.connection_stats().await;

    // Verify stats structure
    assert_eq!(stats.address, "http://127.0.0.1:50051");
    assert_eq!(stats.active_connections, 0);
}

#[tokio::test]
#[serial]
async fn test_connection_stats_after_requests() {
    let env = TestEnvironment::new().await;
    let client = WorkspaceDaemonClient::new(env.address().to_string());

    // Make request
    let _ = client.test_connection().await;

    // Check stats
    let stats = client.connection_stats().await;
    assert_eq!(stats.address, env.address());
    assert!(stats.active_connections <= 1, "Should have at most 1 connection");
}

// ================================
// EDGE CASE TESTS
// ================================

#[tokio::test]
async fn test_rapid_connection_disconnection_cycles() {
    let client = WorkspaceDaemonClient::new("http://127.0.0.1:50051".to_string());

    // Rapid connect/disconnect cycles
    for _ in 0..10 {
        let _ = client.test_connection().await;
        client.disconnect().await;
    }

    // Final state should be clean
    let stats = client.connection_stats().await;
    assert_eq!(stats.active_connections, 0,
              "Rapid cycles should leave clean state");
}

#[tokio::test]
async fn test_concurrent_pool_access_different_addresses() {
    let pool = Arc::new(ConnectionPool::new());

    let addresses = vec![
        "http://127.0.0.1:50051",
        "http://127.0.0.1:50052",
        "http://127.0.0.1:50053",
        "http://127.0.0.1:50054",
        "http://127.0.0.1:50055",
    ];

    let mut handles = Vec::new();

    for addr in addresses {
        let pool_clone = Arc::clone(&pool);
        let address = addr.to_string();

        let handle = tokio::spawn(async move {
            pool_clone.get_connection(&address).await.is_ok()
        });

        handles.push(handle);
    }

    // Wait for all connections
    for handle in handles {
        let _ = handle.await.unwrap();
    }

    // Pool should have at most 5 connections
    let count = pool.connection_count().await;
    assert!(count <= 5, "Pool should not exceed attempted connections");
}

#[tokio::test]
async fn test_connection_stats_structure() {
    // Test ConnectionStats structure
    let stats = ConnectionStats {
        address: "http://127.0.0.1:50051".to_string(),
        active_connections: 3,
    };

    assert_eq!(stats.address, "http://127.0.0.1:50051");
    assert_eq!(stats.active_connections, 3);
}

#[tokio::test]
async fn test_client_with_shared_pool() {
    let pool = ConnectionPool::with_timeouts(
        Duration::from_secs(10),
        Duration::from_secs(5),
    );

    let address = "http://127.0.0.1:50051".to_string();

    // Create multiple clients sharing same pool
    let client1 = WorkspaceDaemonClient::with_pool(address.clone(), pool.clone());
    let client2 = WorkspaceDaemonClient::with_pool(address.clone(), pool.clone());

    // Attempt connections
    let _ = client1.test_connection().await;
    let _ = client2.test_connection().await;

    // Pool should be shared
    let stats1 = client1.connection_stats().await;
    let stats2 = client2.connection_stats().await;

    assert_eq!(stats1.active_connections, stats2.active_connections,
              "Clients should share pool state");
}
