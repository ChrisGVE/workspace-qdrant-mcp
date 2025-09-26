//! Simple middleware test coverage
//! Testing the actual APIs that exist in the middleware module

use workspace_qdrant_daemon::grpc::middleware::{
    ConnectionManager, ConnectionInterceptor, ConnectionStats, PoolConfig, RetryConfig
};
use std::sync::Arc;
use std::time::Duration;

// ================================
// CONNECTION MANAGER TESTS
// ================================

#[test]
fn test_connection_manager_creation() {
    let manager = ConnectionManager::new(100, 50);
    let stats = manager.get_stats();

    assert_eq!(stats.active_connections, 0);
    assert_eq!(stats.total_requests, 0);
    assert_eq!(stats.max_connections, 100);
}

#[test]
fn test_connection_manager_debug_format() {
    let manager = ConnectionManager::new(10, 5);
    let debug_str = format!("{:?}", manager);
    assert!(debug_str.contains("ConnectionManager"));
}

#[test]
fn test_connection_stats_fields() {
    let manager = ConnectionManager::new(50, 20);
    let stats = manager.get_stats();

    // Test that all fields are accessible
    assert_eq!(stats.active_connections, 0);
    assert_eq!(stats.max_connections, 50);
    assert_eq!(stats.total_requests, 0);
    assert_eq!(stats.total_bytes_sent, 0);
    assert_eq!(stats.total_bytes_received, 0);
}

#[test]
fn test_connection_stats_clone() {
    let manager = ConnectionManager::new(30, 15);
    let original = manager.get_stats();
    let cloned = original.clone();

    assert_eq!(original.active_connections, cloned.active_connections);
    assert_eq!(original.max_connections, cloned.max_connections);
    assert_eq!(original.total_requests, cloned.total_requests);
    assert_eq!(original.total_bytes_sent, cloned.total_bytes_sent);
    assert_eq!(original.total_bytes_received, cloned.total_bytes_received);
}

#[test]
fn test_connection_stats_debug_format() {
    let manager = ConnectionManager::new(25, 10);
    let stats = manager.get_stats();

    let debug_str = format!("{:?}", stats);
    assert!(debug_str.contains("ConnectionStats"));
    assert!(debug_str.contains("25")); // max_connections
}

#[test]
fn test_connection_manager_register_connection() {
    let manager = ConnectionManager::new(2, 10);

    // First connection should succeed
    let result1 = manager.register_connection("client1".to_string());
    assert!(result1.is_ok());

    let stats = manager.get_stats();
    assert_eq!(stats.active_connections, 1);

    // Second connection should succeed
    let result2 = manager.register_connection("client2".to_string());
    assert!(result2.is_ok());

    let stats = manager.get_stats();
    assert_eq!(stats.active_connections, 2);

    // Third connection should fail (exceeds limit)
    let result3 = manager.register_connection("client3".to_string());
    assert!(result3.is_err());
}

#[test]
fn test_connection_manager_unregister_connection() {
    let manager = ConnectionManager::new(5, 10);

    // Register a connection
    manager.register_connection("test_client".to_string()).unwrap();
    let stats = manager.get_stats();
    assert_eq!(stats.active_connections, 1);

    // Unregister the connection
    manager.unregister_connection("test_client");
    let stats = manager.get_stats();
    assert_eq!(stats.active_connections, 0);

    // Unregistering non-existent connection should not panic
    manager.unregister_connection("non_existent");
}

#[test]
fn test_connection_manager_rate_limiting() {
    let manager = ConnectionManager::new(10, 2); // 2 requests per second

    // Register a client first
    manager.register_connection("rate_test_client".to_string()).unwrap();

    // First request should be allowed
    let result1 = manager.check_rate_limit("rate_test_client");
    assert!(result1.is_ok());

    // Second request should be allowed
    let result2 = manager.check_rate_limit("rate_test_client");
    assert!(result2.is_ok());

    // Third request should be rate limited
    let result3 = manager.check_rate_limit("rate_test_client");
    assert!(result3.is_err());
}

#[test]
fn test_connection_manager_update_activity() {
    let manager = ConnectionManager::new(5, 10);

    // Register a connection
    manager.register_connection("activity_client".to_string()).unwrap();

    // Update activity (should not panic)
    manager.update_activity("activity_client", 100, 50);

    // Check that the connection still exists
    let stats = manager.get_stats();
    assert_eq!(stats.active_connections, 1);

    // Update activity for non-existent client (should not panic)
    manager.update_activity("non_existent", 10, 5);
}

#[test]
fn test_connection_manager_cleanup_expired() {
    let manager = ConnectionManager::new(5, 10);

    // Register a connection
    manager.register_connection("cleanup_client".to_string()).unwrap();
    let stats = manager.get_stats();
    assert_eq!(stats.active_connections, 1);

    // Cleanup with very short timeout (should not remove fresh connections)
    manager.cleanup_expired_connections(Duration::from_millis(1));
    let stats = manager.get_stats();
    assert_eq!(stats.active_connections, 1);

    // Cleanup with very long timeout (should not remove anything)
    manager.cleanup_expired_connections(Duration::from_secs(3600));
    let stats = manager.get_stats();
    assert_eq!(stats.active_connections, 1);
}

// ================================
// CONNECTION INTERCEPTOR TESTS
// ================================

#[test]
fn test_connection_interceptor_creation() {
    let manager = Arc::new(ConnectionManager::new(10, 5));
    let interceptor = ConnectionInterceptor::new(manager);

    // Test that we can create it
    let debug_str = format!("{:?}", interceptor);
    assert!(debug_str.contains("ConnectionInterceptor"));
}

#[test]
fn test_connection_interceptor_clone() {
    let manager = Arc::new(ConnectionManager::new(10, 5));
    let interceptor1 = ConnectionInterceptor::new(manager);
    let interceptor2 = interceptor1.clone();

    // Both should be valid
    let debug1 = format!("{:?}", interceptor1);
    let debug2 = format!("{:?}", interceptor2);

    assert!(debug1.contains("ConnectionInterceptor"));
    assert!(debug2.contains("ConnectionInterceptor"));
}

// ================================
// POOL CONFIG TESTS
// ================================

#[test]
fn test_pool_config_default() {
    let config = PoolConfig::default();

    assert_eq!(config.max_size, 10);
    assert_eq!(config.min_idle, Some(2));
    assert!(config.max_lifetime.is_some());
    assert!(config.idle_timeout.is_some());
    assert!(config.connection_timeout > Duration::from_secs(0));
}

#[test]
fn test_pool_config_debug() {
    let config = PoolConfig::default();
    let debug_str = format!("{:?}", config);

    assert!(debug_str.contains("PoolConfig"));
    assert!(debug_str.contains("10")); // max_size
}

#[test]
fn test_pool_config_clone() {
    let original = PoolConfig::default();
    let cloned = original.clone();

    assert_eq!(original.max_size, cloned.max_size);
    assert_eq!(original.min_idle, cloned.min_idle);
    assert_eq!(original.max_lifetime, cloned.max_lifetime);
    assert_eq!(original.idle_timeout, cloned.idle_timeout);
    assert_eq!(original.connection_timeout, cloned.connection_timeout);
}

#[test]
fn test_pool_config_custom() {
    let config = PoolConfig {
        max_size: 20,
        min_idle: Some(5),
        max_lifetime: Some(Duration::from_secs(7200)),
        idle_timeout: Some(Duration::from_secs(1200)),
        connection_timeout: Duration::from_secs(60),
    };

    assert_eq!(config.max_size, 20);
    assert_eq!(config.min_idle, Some(5));
    assert_eq!(config.max_lifetime, Some(Duration::from_secs(7200)));
    assert_eq!(config.idle_timeout, Some(Duration::from_secs(1200)));
    assert_eq!(config.connection_timeout, Duration::from_secs(60));
}

// ================================
// INTEGRATION TESTS
// ================================

#[test]
fn test_connection_manager_and_interceptor_integration() {
    let manager = Arc::new(ConnectionManager::new(10, 5));
    let manager_clone = Arc::clone(&manager);
    let _interceptor = ConnectionInterceptor::new(manager_clone);

    // Manager should still be functional
    let result = manager.register_connection("integration_client".to_string());
    assert!(result.is_ok());

    let stats = manager.get_stats();
    assert_eq!(stats.active_connections, 1);
}

#[test]
fn test_multi_client_registration() {
    let manager = ConnectionManager::new(5, 10);
    let client_names = vec!["client1", "client2", "client3"];

    for client in &client_names {
        let result = manager.register_connection(client.to_string());
        assert!(result.is_ok(), "Failed to register client: {}", client);
    }

    let stats = manager.get_stats();
    assert_eq!(stats.active_connections, 3);

    // Unregister all clients
    for client in &client_names {
        manager.unregister_connection(client);
    }

    let stats = manager.get_stats();
    assert_eq!(stats.active_connections, 0);
}

#[test]
fn test_connection_manager_full_lifecycle() {
    let manager = ConnectionManager::new(3, 5);

    // Register up to limit
    for i in 1..=3 {
        let client_id = format!("client{}", i);
        let result = manager.register_connection(client_id);
        assert!(result.is_ok());
    }

    let stats = manager.get_stats();
    assert_eq!(stats.active_connections, 3);

    // Try to register beyond limit
    let result = manager.register_connection("overflow_client".to_string());
    assert!(result.is_err());

    // Update activity for existing clients
    manager.update_activity("client1", 100, 50);
    manager.update_activity("client2", 200, 100);

    // Unregister one client
    manager.unregister_connection("client1");
    let stats = manager.get_stats();
    assert_eq!(stats.active_connections, 2);

    // Now we should be able to register a new client
    let result = manager.register_connection("new_client".to_string());
    assert!(result.is_ok());

    let stats = manager.get_stats();
    assert_eq!(stats.active_connections, 3);
}

// ================================
// SEND/SYNC TRAIT TESTS
// ================================

#[test]
fn test_middleware_components_send_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    assert_send::<ConnectionManager>();
    assert_sync::<ConnectionManager>();
    assert_send::<ConnectionStats>();
    assert_sync::<ConnectionStats>();
    assert_send::<PoolConfig>();
    assert_sync::<PoolConfig>();
    assert_send::<ConnectionInterceptor>();
    assert_sync::<ConnectionInterceptor>();
}

// ================================
// ERROR CONDITION TESTS
// ================================

#[test]
fn test_connection_manager_with_zero_max_connections() {
    let manager = ConnectionManager::new(0, 10);

    // Should immediately fail with zero max connections
    let result = manager.register_connection("test_client".to_string());
    assert!(result.is_err());

    let stats = manager.get_stats();
    assert_eq!(stats.active_connections, 0);
    assert_eq!(stats.max_connections, 0);
}

#[test]
fn test_connection_manager_rate_limiting_with_zero_rate() {
    let manager = ConnectionManager::new(5, 0); // Zero requests per second

    // Register a client first
    manager.register_connection("zero_rate_client".to_string()).unwrap();

    // Any request should be rate limited with zero rate
    let result = manager.check_rate_limit("zero_rate_client");
    assert!(result.is_err());
}

#[test]
fn test_connection_manager_multiple_managers() {
    let manager1 = ConnectionManager::new(2, 5);
    let manager2 = ConnectionManager::new(3, 10);

    // Register clients in different managers
    assert!(manager1.register_connection("client1".to_string()).is_ok());
    assert!(manager2.register_connection("client2".to_string()).is_ok());

    // Each manager should track separately
    assert_eq!(manager1.get_stats().active_connections, 1);
    assert_eq!(manager2.get_stats().active_connections, 1);

    // Fill up manager1
    assert!(manager1.register_connection("client3".to_string()).is_ok());
    assert!(manager1.register_connection("client4".to_string()).is_err()); // Should fail

    // manager2 should still accept connections
    assert!(manager2.register_connection("client5".to_string()).is_ok());
}

#[test]
fn test_connection_interceptor_with_different_managers() {
    let manager1 = Arc::new(ConnectionManager::new(5, 10));
    let manager2 = Arc::new(ConnectionManager::new(10, 20));

    let interceptor1 = ConnectionInterceptor::new(manager1);
    let interceptor2 = ConnectionInterceptor::new(manager2);

    // Both should be valid and independent
    let debug1 = format!("{:?}", interceptor1);
    let debug2 = format!("{:?}", interceptor2);

    assert!(debug1.contains("ConnectionInterceptor"));
    assert!(debug2.contains("ConnectionInterceptor"));
}