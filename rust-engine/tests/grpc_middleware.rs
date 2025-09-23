//! Comprehensive unit tests for grpc/middleware.rs - targeting 90%+ coverage
//!
//! This test suite provides comprehensive coverage for:
//! - ConnectionManager with all edge cases and error conditions
//! - ConnectionInterceptor request/response handling
//! - Retry mechanisms with backoff and timeout scenarios
//! - Rate limiting edge cases and cleanup
//! - Connection pool configuration
//! - Concurrent operations and thread safety

use workspace_qdrant_daemon::grpc::middleware::*;
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicU64, Ordering};
use tonic::{Request, Response, metadata::MetadataValue};

// =============================================================================
// CONNECTION MANAGER COMPREHENSIVE TESTS
// =============================================================================

#[test]
fn test_connection_manager_creation_variants() {
    // Test with different parameter combinations
    let manager1 = ConnectionManager::new(0, 0);
    assert_eq!(manager1.get_stats().max_connections, 0);

    let manager2 = ConnectionManager::new(u64::MAX, u32::MAX);
    assert_eq!(manager2.get_stats().max_connections, u64::MAX);

    let manager3 = ConnectionManager::new(1, 1);
    assert_eq!(manager3.get_stats().max_connections, 1);
}

#[test]
fn test_connection_manager_register_edge_cases() {
    let manager = ConnectionManager::new(1, 10);

    // Test empty client ID
    let result = manager.register_connection("".to_string());
    assert!(result.is_ok());
    assert_eq!(manager.get_stats().active_connections, 1);

    // Test duplicate registration (should succeed creating new connection)
    let result2 = manager.register_connection("".to_string());
    assert!(result2.is_err()); // Should fail due to max_connections = 1

    manager.unregister_connection("");

    // Test very long client ID
    let long_id = "a".repeat(1000);
    let result3 = manager.register_connection(long_id.clone());
    assert!(result3.is_ok());

    manager.unregister_connection(&long_id);

    // Test special characters in client ID
    let special_id = "client-with/special\\chars:123";
    let result4 = manager.register_connection(special_id.to_string());
    assert!(result4.is_ok());

    manager.unregister_connection(special_id);
}

#[test]
fn test_connection_manager_unregister_edge_cases() {
    let manager = ConnectionManager::new(5, 10);

    // Unregister non-existent connection multiple times
    manager.unregister_connection("non_existent_1");
    manager.unregister_connection("non_existent_1"); // Should not panic
    manager.unregister_connection("different_non_existent");

    // Register and unregister same client multiple times
    manager.register_connection("test_client".to_string()).unwrap();
    manager.unregister_connection("test_client");
    manager.unregister_connection("test_client"); // Should not panic

    assert_eq!(manager.get_stats().active_connections, 0);
}

#[test]
fn test_connection_manager_rate_limiting_edge_cases() {
    // Test with zero rate limit
    let manager = ConnectionManager::new(10, 0);

    let result = manager.check_rate_limit("client");
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().code(), tonic::Code::ResourceExhausted);

    // Test with high rate limit
    let manager2 = ConnectionManager::new(10, 1000);
    for i in 0..999 {
        let result = manager2.check_rate_limit("high_rate_client");
        assert!(result.is_ok(), "Failed on request {}", i);
    }

    // Should fail on 1000th request (at limit)
    let result = manager2.check_rate_limit("high_rate_client");
    assert!(result.is_err());
}

#[test]
fn test_connection_manager_rate_limiting_cleanup() {
    let manager = ConnectionManager::new(10, 2);

    // Add some requests
    manager.check_rate_limit("cleanup_client").unwrap();
    manager.check_rate_limit("cleanup_client").unwrap();

    // Next should fail
    assert!(manager.check_rate_limit("cleanup_client").is_err());

    // Wait a bit and then add more requests to trigger cleanup path
    std::thread::sleep(Duration::from_millis(10));

    // Add requests for different clients to trigger the cleanup logic
    for i in 0..10 {
        let client_id = format!("temp_client_{}", i);
        let _ = manager.check_rate_limit(&client_id);
    }
}

#[test]
fn test_connection_manager_rate_limiting_time_window() {
    let manager = ConnectionManager::new(10, 2);

    // Use up rate limit
    assert!(manager.check_rate_limit("time_test").is_ok());
    assert!(manager.check_rate_limit("time_test").is_ok());
    assert!(manager.check_rate_limit("time_test").is_err());

    // Wait for time window to pass (slightly more than 1 second)
    std::thread::sleep(Duration::from_millis(1100));

    // Should be able to make requests again
    assert!(manager.check_rate_limit("time_test").is_ok());
    assert!(manager.check_rate_limit("time_test").is_ok());
    assert!(manager.check_rate_limit("time_test").is_err());
}

#[test]
fn test_connection_manager_update_activity_edge_cases() {
    let manager = ConnectionManager::new(5, 10);

    // Update activity for non-existent connection
    manager.update_activity("non_existent", 100, 200);

    // Register connection and update with zero values
    manager.register_connection("zero_activity".to_string()).unwrap();
    manager.update_activity("zero_activity", 0, 0);

    let stats = manager.get_stats();
    assert_eq!(stats.total_requests, 1); // From register + update_activity call

    // Update with large values
    manager.update_activity("zero_activity", u64::MAX, u64::MAX);

    let stats = manager.get_stats();
    assert_eq!(stats.total_requests, 2);
    assert_eq!(stats.total_bytes_sent, u64::MAX);
    assert_eq!(stats.total_bytes_received, u64::MAX);
}

#[test]
fn test_connection_manager_cleanup_expired_edge_cases() {
    let manager = ConnectionManager::new(10, 10);

    // Cleanup with no connections
    manager.cleanup_expired_connections(Duration::from_secs(0));
    manager.cleanup_expired_connections(Duration::from_secs(3600));

    // Register multiple connections
    for i in 0..5 {
        manager.register_connection(format!("client_{}", i)).unwrap();
    }

    assert_eq!(manager.get_stats().active_connections, 5);

    // Update activity for some connections
    manager.update_activity("client_0", 100, 100);
    manager.update_activity("client_2", 100, 100);
    manager.update_activity("client_4", 100, 100);

    // Cleanup with zero timeout (should remove all)
    manager.cleanup_expired_connections(Duration::from_secs(0));
    assert_eq!(manager.get_stats().active_connections, 0);
}

#[test]
fn test_connection_manager_get_stats_accuracy() {
    let manager = ConnectionManager::new(10, 10);

    // Test with no connections
    let stats = manager.get_stats();
    assert_eq!(stats.active_connections, 0);
    assert_eq!(stats.total_requests, 0);
    assert_eq!(stats.total_bytes_sent, 0);
    assert_eq!(stats.total_bytes_received, 0);
    assert_eq!(stats.max_connections, 10);

    // Add connections with different activity levels
    manager.register_connection("stats_client_1".to_string()).unwrap();
    manager.register_connection("stats_client_2".to_string()).unwrap();

    manager.update_activity("stats_client_1", 1000, 500);
    manager.update_activity("stats_client_1", 2000, 1000);
    manager.update_activity("stats_client_2", 500, 250);

    let stats = manager.get_stats();
    assert_eq!(stats.active_connections, 2);
    assert_eq!(stats.total_requests, 5); // 2 registrations + 3 activity updates
    assert_eq!(stats.total_bytes_sent, 3500);
    assert_eq!(stats.total_bytes_received, 1750);
}

// =============================================================================
// CONNECTION INTERCEPTOR COMPREHENSIVE TESTS
// =============================================================================

#[test]
fn test_connection_interceptor_intercept_metadata_parsing() {
    let manager = Arc::new(ConnectionManager::new(10, 10));
    let interceptor = ConnectionInterceptor::new(manager);

    // Test with static metadata (simpler case)
    let mut request: Request<()> = Request::new(());
    request.metadata_mut().insert(
        "client-id",
        MetadataValue::from_static("test_client")
    );

    // Should handle gracefully
    let result = interceptor.intercept(request);
    assert!(result.is_ok());

    // Test with empty metadata value
    let mut request2: Request<()> = Request::new(());
    request2.metadata_mut().insert("client-id", MetadataValue::from_static(""));

    let result2 = interceptor.intercept(request2);
    assert!(result2.is_ok());

    // Test with whitespace-only client ID
    let mut request3: Request<()> = Request::new(());
    request3.metadata_mut().insert("client-id", MetadataValue::from_static("   "));

    let result3 = interceptor.intercept(request3);
    assert!(result3.is_ok());
}

#[test]
fn test_connection_interceptor_rate_limiting_integration() {
    let manager = Arc::new(ConnectionManager::new(10, 1)); // Very restrictive rate limit
    let interceptor = ConnectionInterceptor::new(manager);

    // First request should succeed
    let mut request1: Request<()> = Request::new(());
    request1.metadata_mut().insert("client-id", MetadataValue::from_static("rate_limited_client"));

    let result1 = interceptor.intercept(request1);
    assert!(result1.is_ok());

    // Second request should fail due to rate limiting
    let mut request2: Request<()> = Request::new(());
    request2.metadata_mut().insert("client-id", MetadataValue::from_static("rate_limited_client"));

    let result2 = interceptor.intercept(request2);
    assert!(result2.is_err());
    assert_eq!(result2.unwrap_err().code(), tonic::Code::ResourceExhausted);
}

#[test]
fn test_connection_interceptor_response_handling() {
    let manager = Arc::new(ConnectionManager::new(10, 10));
    let interceptor = ConnectionInterceptor::new(manager.clone());

    // Register a client first
    manager.register_connection("response_client".to_string()).unwrap();

    // Test response interception with various response types
    let response1: Response<String> = Response::new("test_response".to_string());
    let result1 = interceptor.intercept_response(response1, "response_client");
    assert_eq!(result1.get_ref(), "test_response");

    // Test with unit response
    let response2: Response<()> = Response::new(());
    let result2 = interceptor.intercept_response(response2, "response_client");
    assert_eq!(result2.get_ref(), &());

    // Test with non-existent client (should not panic)
    let response3: Response<i32> = Response::new(42);
    let result3 = interceptor.intercept_response(response3, "non_existent_client");
    assert_eq!(result3.get_ref(), &42);
}

// =============================================================================
// RETRY MECHANISM COMPREHENSIVE TESTS
// =============================================================================

#[tokio::test]
async fn test_with_retry_zero_retries() {
    let config = RetryConfig {
        max_retries: 0,
        initial_delay: Duration::from_millis(1),
        max_delay: Duration::from_secs(1),
        backoff_multiplier: 2.0,
    };

    // Should fail immediately with zero retries
    let result = with_retry(
        || Box::pin(async { Err::<i32, &'static str>("immediate failure") }),
        &config,
    ).await;

    assert_eq!(result.unwrap_err(), "immediate failure");
}

#[tokio::test]
async fn test_with_retry_backoff_progression() {
    let config = RetryConfig {
        max_retries: 4,
        initial_delay: Duration::from_millis(1),
        max_delay: Duration::from_millis(10),
        backoff_multiplier: 2.0,
    };

    let start_time = Instant::now();
    let attempt_count = Arc::new(AtomicU64::new(0));
    let attempt_clone = Arc::clone(&attempt_count);

    let result = with_retry(
        move || {
            attempt_clone.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { Err::<i32, &'static str>("always fail") })
        },
        &config,
    ).await;

    let elapsed = start_time.elapsed();

    // Should have made max_retries attempts
    assert_eq!(attempt_count.load(Ordering::SeqCst), 4);
    assert!(result.is_err());

    // Should have taken some time due to backoff delays
    // Total expected delay: 1ms + 2ms + 4ms = 7ms (capped at max_delay)
    assert!(elapsed >= Duration::from_millis(5));
}

#[tokio::test]
async fn test_with_retry_max_delay_capping() {
    let config = RetryConfig {
        max_retries: 3,
        initial_delay: Duration::from_millis(10),
        max_delay: Duration::from_millis(15), // Cap below expected exponential growth
        backoff_multiplier: 10.0, // High multiplier
    };

    let start_time = Instant::now();

    let result = with_retry(
        || Box::pin(async { Err::<i32, &'static str>("capped delay test") }),
        &config,
    ).await;

    let elapsed = start_time.elapsed();

    // With multiplier 10.0, delays would be: 10ms, 100ms, 1000ms
    // But capped at 15ms each, so total should be around 30ms
    assert!(elapsed < Duration::from_millis(100)); // Much less than uncapped
    assert!(elapsed >= Duration::from_millis(25)); // But still some delay
    assert!(result.is_err());
}

#[tokio::test]
async fn test_with_retry_success_after_retries() {
    let config = RetryConfig {
        max_retries: 5,
        initial_delay: Duration::from_millis(1),
        max_delay: Duration::from_secs(1),
        backoff_multiplier: 1.5,
    };

    let attempt_count = Arc::new(AtomicU64::new(0));
    let attempt_clone = Arc::clone(&attempt_count);

    let result = with_retry(
        move || {
            let current_attempt = attempt_clone.fetch_add(1, Ordering::SeqCst) + 1;
            Box::pin(async move {
                if current_attempt < 4 {
                    Err("not yet")
                } else {
                    Ok::<i32, &'static str>(42)
                }
            })
        },
        &config,
    ).await;

    assert_eq!(result.unwrap(), 42);
    assert_eq!(attempt_count.load(Ordering::SeqCst), 4);
}

#[tokio::test]
async fn test_with_retry_different_error_types() {
    let config = RetryConfig::default();

    // Test with String error
    let result1 = with_retry(
        || Box::pin(async { Err::<i32, String>("string error".to_string()) }),
        &config,
    ).await;
    assert!(result1.is_err());

    // Test with custom error type
    #[derive(Debug, PartialEq)]
    struct CustomError(&'static str);

    let result2 = with_retry(
        || Box::pin(async { Err::<i32, CustomError>(CustomError("custom error")) }),
        &config,
    ).await;
    assert_eq!(result2.unwrap_err(), CustomError("custom error"));
}

// =============================================================================
// CONFIGURATION TESTS
// =============================================================================

#[test]
fn test_pool_config_comprehensive() {
    // Test default values
    let default_config = PoolConfig::default();
    assert_eq!(default_config.max_size, 10);
    assert_eq!(default_config.min_idle, Some(2));
    assert_eq!(default_config.max_lifetime, Some(Duration::from_secs(3600)));
    assert_eq!(default_config.idle_timeout, Some(Duration::from_secs(600)));
    assert_eq!(default_config.connection_timeout, Duration::from_secs(30));

    // Test custom configuration with edge values
    let custom_config = PoolConfig {
        max_size: 0,
        min_idle: None,
        max_lifetime: None,
        idle_timeout: None,
        connection_timeout: Duration::from_millis(1),
    };

    assert_eq!(custom_config.max_size, 0);
    assert_eq!(custom_config.min_idle, None);
    assert_eq!(custom_config.max_lifetime, None);
    assert_eq!(custom_config.idle_timeout, None);
    assert_eq!(custom_config.connection_timeout, Duration::from_millis(1));

    // Test extreme values
    let extreme_config = PoolConfig {
        max_size: usize::MAX,
        min_idle: Some(usize::MAX),
        max_lifetime: Some(Duration::from_secs(u64::MAX)),
        idle_timeout: Some(Duration::from_nanos(1)),
        connection_timeout: Duration::from_secs(0),
    };

    assert_eq!(extreme_config.max_size, usize::MAX);
    assert_eq!(extreme_config.min_idle, Some(usize::MAX));
}

#[test]
fn test_retry_config_comprehensive() {
    // Test default values
    let default_config = RetryConfig::default();
    assert_eq!(default_config.max_retries, 3);
    assert_eq!(default_config.initial_delay, Duration::from_millis(100));
    assert_eq!(default_config.max_delay, Duration::from_secs(30));
    assert_eq!(default_config.backoff_multiplier, 2.0);

    // Test custom configuration with edge values
    let custom_config = RetryConfig {
        max_retries: 0,
        initial_delay: Duration::from_nanos(1),
        max_delay: Duration::from_nanos(1),
        backoff_multiplier: 0.0,
    };

    assert_eq!(custom_config.max_retries, 0);
    assert_eq!(custom_config.initial_delay, Duration::from_nanos(1));
    assert_eq!(custom_config.max_delay, Duration::from_nanos(1));
    assert_eq!(custom_config.backoff_multiplier, 0.0);

    // Test extreme values
    let extreme_config = RetryConfig {
        max_retries: u32::MAX,
        initial_delay: Duration::from_secs(u64::MAX),
        max_delay: Duration::from_secs(u64::MAX),
        backoff_multiplier: f64::MAX,
    };

    assert_eq!(extreme_config.max_retries, u32::MAX);
    assert_eq!(extreme_config.backoff_multiplier, f64::MAX);
}

// =============================================================================
// CONCURRENCY AND THREAD SAFETY TESTS
// =============================================================================

#[tokio::test]
async fn test_connection_manager_concurrent_operations() {
    let manager = Arc::new(ConnectionManager::new(50, 100));
    let mut handles = vec![];

    // Spawn tasks for concurrent registration/unregistration
    for i in 0..20 {
        let manager_clone = Arc::clone(&manager);
        let handle = tokio::spawn(async move {
            let client_id = format!("concurrent_client_{}", i);

            // Register
            manager_clone.register_connection(client_id.clone()).unwrap();

            // Update activity multiple times
            for j in 0..10 {
                manager_clone.update_activity(&client_id, j * 10, j * 5);
                tokio::time::sleep(Duration::from_micros(1)).await;
            }

            // Check rate limits
            for _ in 0..5 {
                let _ = manager_clone.check_rate_limit(&client_id);
            }

            // Unregister
            manager_clone.unregister_connection(&client_id);
        });
        handles.push(handle);
    }

    // Wait for all tasks to complete
    for handle in handles {
        handle.await.unwrap();
    }

    // All connections should be cleaned up
    assert_eq!(manager.get_stats().active_connections, 0);
}

#[tokio::test]
async fn test_connection_manager_concurrent_rate_limiting() {
    let manager = Arc::new(ConnectionManager::new(100, 5)); // 5 requests per second
    let mut handles = vec![];

    // Spawn multiple tasks hitting rate limits simultaneously
    for i in 0..10 {
        let manager_clone = Arc::clone(&manager);
        let handle = tokio::spawn(async move {
            let client_id = format!("rate_client_{}", i);

            // Each client should be able to make 5 requests before hitting limit
            let mut success_count = 0;
            let mut error_count = 0;

            for _ in 0..10 {
                match manager_clone.check_rate_limit(&client_id) {
                    Ok(()) => success_count += 1,
                    Err(_) => error_count += 1,
                }
            }

            (success_count, error_count)
        });
        handles.push(handle);
    }

    // Collect results
    let mut total_success = 0;
    let mut total_errors = 0;

    for handle in handles {
        let (success, errors) = handle.await.unwrap();
        total_success += success;
        total_errors += errors;
    }

    // Each client should have been rate limited
    assert!(total_success > 0);
    assert!(total_errors > 0);
    assert_eq!(total_success + total_errors, 100); // 10 clients * 10 requests each
}

#[tokio::test]
async fn test_connection_interceptor_concurrent_requests() {
    let manager = Arc::new(ConnectionManager::new(100, 50));
    let interceptor = ConnectionInterceptor::new(manager.clone());
    let mut handles = vec![];

    // Register some clients
    for i in 0..10 {
        manager.register_connection(format!("interceptor_client_{}", i)).unwrap();
    }

    // Spawn concurrent request interceptions
    for i in 0..20 {
        let interceptor_clone = interceptor.clone();
        let handle = tokio::spawn(async move {
            let mut request: Request<String> = Request::new(format!("request_{}", i));
            let _client_id = format!("interceptor_client_{}", i % 10);
            request.metadata_mut().insert(
                "client-id",
                MetadataValue::from_static("interceptor_client_0")
            );

            let result = interceptor_clone.intercept(request);
            result.is_ok()
        });
        handles.push(handle);
    }

    // Wait for all interceptions to complete
    let mut success_count = 0;
    for handle in handles {
        if handle.await.unwrap() {
            success_count += 1;
        }
    }

    // Some should succeed (before rate limits hit)
    assert!(success_count > 0);
}

// =============================================================================
// DEBUG AND TRAIT IMPLEMENTATION TESTS
// =============================================================================

#[test]
fn test_debug_implementations() {
    let manager = ConnectionManager::new(10, 5);
    let debug_str = format!("{:?}", manager);
    assert!(debug_str.contains("ConnectionManager"));

    let stats = manager.get_stats();
    let _stats_debug = format!("{:?}", stats);
    assert!(debug_str.contains("ConnectionStats"));

    let config = PoolConfig::default();
    let config_debug = format!("{:?}", config);
    assert!(config_debug.contains("PoolConfig"));

    let retry_config = RetryConfig::default();
    let retry_debug = format!("{:?}", retry_config);
    assert!(retry_debug.contains("RetryConfig"));

    let interceptor = ConnectionInterceptor::new(Arc::new(manager));
    let interceptor_debug = format!("{:?}", interceptor);
    assert!(interceptor_debug.contains("ConnectionInterceptor"));
}

#[test]
fn test_clone_implementations() {
    let stats = ConnectionStats {
        active_connections: 5,
        max_connections: 10,
        total_requests: 100,
        total_bytes_sent: 1000,
        total_bytes_received: 2000,
    };
    let cloned_stats = stats.clone();
    assert_eq!(stats.active_connections, cloned_stats.active_connections);

    let config = PoolConfig::default();
    let cloned_config = config.clone();
    assert_eq!(config.max_size, cloned_config.max_size);

    let retry_config = RetryConfig::default();
    let cloned_retry = retry_config.clone();
    assert_eq!(retry_config.max_retries, cloned_retry.max_retries);

    let manager = Arc::new(ConnectionManager::new(10, 5));
    let interceptor = ConnectionInterceptor::new(manager);
    let cloned_interceptor = interceptor.clone();

    // Should be able to use both interceptors
    let _ = format!("{:?}", interceptor);
    let _ = format!("{:?}", cloned_interceptor);
}

#[test]
fn test_send_sync_traits() {
    fn assert_send_sync<T: Send + Sync>() {}

    assert_send_sync::<ConnectionManager>();
    assert_send_sync::<ConnectionStats>();
    assert_send_sync::<PoolConfig>();
    assert_send_sync::<RetryConfig>();
    assert_send_sync::<ConnectionInterceptor>();
}

// =============================================================================
// ERROR HANDLING AND EDGE CASE TESTS
// =============================================================================

#[test]
fn test_connection_info_clone_atomics() {
    let info = ConnectionInfo {
        client_id: "test_atomic_clone".to_string(),
        connected_at: Instant::now(),
        last_activity: Instant::now(),
        request_count: AtomicU64::new(42),
        bytes_sent: AtomicU64::new(1000),
        bytes_received: AtomicU64::new(2000),
    };

    // Modify original atomics
    info.request_count.store(100, Ordering::SeqCst);
    info.bytes_sent.store(5000, Ordering::SeqCst);
    info.bytes_received.store(10000, Ordering::SeqCst);

    // Clone should capture the current values
    let cloned = info.clone();

    assert_eq!(cloned.client_id, info.client_id);
    assert_eq!(cloned.request_count.load(Ordering::SeqCst), 100);
    assert_eq!(cloned.bytes_sent.load(Ordering::SeqCst), 5000);
    assert_eq!(cloned.bytes_received.load(Ordering::SeqCst), 10000);

    // Modifying original should not affect clone
    info.request_count.store(999, Ordering::SeqCst);
    assert_eq!(cloned.request_count.load(Ordering::SeqCst), 100);
}

#[test]
fn test_connection_manager_stats_edge_cases() {
    let manager = ConnectionManager::new(3, 10);

    // Register connections with various activity levels
    manager.register_connection("stats_test_1".to_string()).unwrap();
    manager.register_connection("stats_test_2".to_string()).unwrap();

    // Create overflow conditions for bytes counting
    manager.update_activity("stats_test_1", u64::MAX / 2, u64::MAX / 2);
    manager.update_activity("stats_test_2", u64::MAX / 2, u64::MAX / 2);

    let stats = manager.get_stats();

    // Should handle large numbers correctly
    assert_eq!(stats.active_connections, 2);
    assert!(stats.total_bytes_sent >= u64::MAX / 2);
    assert!(stats.total_bytes_received >= u64::MAX / 2);
}

#[tokio::test]
async fn test_with_retry_unreachable_path() {
    // This test ensures that the unreachable!() path in with_retry is never hit
    // by testing the loop termination logic

    let config = RetryConfig {
        max_retries: 1,
        initial_delay: Duration::from_millis(1),
        max_delay: Duration::from_secs(1),
        backoff_multiplier: 2.0,
    };

    let result = with_retry(
        || Box::pin(async { Err::<i32, &'static str>("test error") }),
        &config,
    ).await;

    // Should return error, not hit unreachable!()
    assert!(result.is_err());
}

#[test]
fn test_rate_limiter_cleanup_retention() {
    let manager = ConnectionManager::new(10, 10);

    // Add requests for multiple clients
    for i in 0..5 {
        let client_id = format!("retention_client_{}", i);
        let _ = manager.check_rate_limit(&client_id);
    }

    // Force cleanup by waiting and making more requests to trigger the cleanup interval
    std::thread::sleep(Duration::from_millis(10));

    // Adding requests for new clients should potentially trigger cleanup
    for i in 5..10 {
        let client_id = format!("retention_client_{}", i);
        let _ = manager.check_rate_limit(&client_id);
    }

    // The internal state should be maintained correctly
    let stats = manager.get_stats();
    assert_eq!(stats.active_connections, 0); // No connections registered yet
}