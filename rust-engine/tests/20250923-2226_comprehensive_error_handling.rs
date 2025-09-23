//! Comprehensive unit tests for error handling, retry logic, and circuit breaker functionality
//!
//! This test suite validates all the error handling infrastructure for task 256.3

use std::time::Duration;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use tokio::time::sleep;
use workspace_qdrant_daemon::error::{DaemonError, DaemonResult};
use workspace_qdrant_daemon::grpc::{
    RetryStrategy, RetryConfig, RetryPredicate, DefaultRetryPredicate,
    CircuitBreaker, CircuitBreakerConfig, CircuitState
};

#[tokio::test]
async fn test_network_error_types() {
    // Test all network-related error variants
    let errors = vec![
        DaemonError::NetworkConnection { message: "Connection refused".to_string() },
        DaemonError::NetworkTimeout { timeout_ms: 5000 },
        DaemonError::NetworkUnavailable { message: "Service down".to_string() },
        DaemonError::DnsResolution { hostname: "example.com".to_string() },
        DaemonError::TlsHandshake { message: "TLS error".to_string() },
    ];

    for error in errors {
        // Test that network errors can be cloned
        let cloned = error.clone();
        assert_eq!(error.to_string(), cloned.to_string());

        // Test that network errors convert to appropriate gRPC status
        let status: tonic::Status = error.into();
        assert!(!status.message().is_empty());
    }
}

#[tokio::test]
async fn test_retry_error_types() {
    // Test retry-related error variants
    let errors = vec![
        DaemonError::RetryLimitExceeded { attempts: 5 },
        DaemonError::RetryConfigInvalid { message: "Invalid config".to_string() },
        DaemonError::BackoffCalculation { message: "Math error".to_string() },
    ];

    for error in errors {
        let cloned = error.clone();
        assert_eq!(error.to_string(), cloned.to_string());
    }
}

#[tokio::test]
async fn test_circuit_breaker_error_types() {
    // Test circuit breaker error variants
    let errors = vec![
        DaemonError::CircuitBreakerOpen { service: "test-svc".to_string() },
        DaemonError::CircuitBreakerHalfOpen { service: "test-svc".to_string() },
        DaemonError::CircuitBreakerConfig { message: "Bad config".to_string() },
    ];

    for error in errors {
        let cloned = error.clone();
        assert_eq!(error.to_string(), cloned.to_string());
    }
}

#[tokio::test]
async fn test_service_health_error_types() {
    // Test service health and load balancing errors
    let errors = vec![
        DaemonError::ServiceHealthCheck {
            service: "test-svc".to_string(),
            reason: "Health check failed".to_string()
        },
        DaemonError::ServiceDegraded {
            service: "test-svc".to_string(),
            details: "High latency".to_string()
        },
        DaemonError::NoHealthyInstances { service: "test-svc".to_string() },
        DaemonError::LoadBalancerConfig { message: "Invalid LB config".to_string() },
    ];

    for error in errors {
        let cloned = error.clone();
        assert_eq!(error.to_string(), cloned.to_string());
    }
}

#[tokio::test]
async fn test_retry_config_comprehensive() {
    // Test all retry configuration methods
    let config = RetryConfig::new()
        .max_attempts(5)
        .initial_delay(Duration::from_millis(50))
        .max_delay(Duration::from_secs(10))
        .backoff_multiplier(1.5)
        .jitter_factor(0.2)
        .enable_jitter(true);

    assert!(config.validate().is_ok());

    // Test invalid configurations
    let invalid_configs = vec![
        RetryConfig::new().max_attempts(0),
        RetryConfig::new().initial_delay(Duration::ZERO),
        RetryConfig::new().initial_delay(Duration::from_secs(10)).max_delay(Duration::from_secs(5)),
        RetryConfig::new().backoff_multiplier(0.5),
        RetryConfig::new().jitter_factor(1.5),
    ];

    for config in invalid_configs {
        assert!(config.validate().is_err());
    }
}

#[tokio::test]
async fn test_default_retry_predicate_comprehensive() {
    let predicate = DefaultRetryPredicate;

    // Should retry these errors
    let retryable_errors = vec![
        DaemonError::NetworkConnection { message: "test".to_string() },
        DaemonError::NetworkTimeout { timeout_ms: 1000 },
        DaemonError::NetworkUnavailable { message: "test".to_string() },
        DaemonError::DnsResolution { hostname: "test".to_string() },
        DaemonError::TlsHandshake { message: "test".to_string() },
        DaemonError::ServiceHealthCheck { service: "test".to_string(), reason: "test".to_string() },
        DaemonError::NoHealthyInstances { service: "test".to_string() },
        DaemonError::CircuitBreakerHalfOpen { service: "test".to_string() },
    ];

    for error in retryable_errors {
        assert!(predicate.should_retry(&error, 0), "Error should be retryable: {}", error);
    }

    // Should NOT retry these errors
    let non_retryable_errors = vec![
        DaemonError::InvalidInput { message: "test".to_string() },
        DaemonError::NotFound { resource: "test".to_string() },
        DaemonError::Configuration { message: "test".to_string() },
        DaemonError::CircuitBreakerOpen { service: "test".to_string() },
    ];

    for error in non_retryable_errors {
        assert!(!predicate.should_retry(&error, 0), "Error should NOT be retryable: {}", error);
    }
}

#[tokio::test]
async fn test_retry_strategy_edge_cases() {
    let config = RetryConfig::new()
        .max_attempts(2)
        .initial_delay(Duration::from_millis(10))
        .enable_jitter(false);

    let strategy = RetryStrategy::with_config(config).unwrap();

    // Test immediate success
    let result = strategy.execute(|| async { Ok::<i32, DaemonError>(42) }).await;
    assert_eq!(result.unwrap(), 42);

    // Test with error that becomes non-retryable after first attempt
    #[derive(Debug)]
    struct TestPredicate;
    impl RetryPredicate for TestPredicate {
        fn should_retry(&self, _error: &DaemonError, attempt: u32) -> bool {
            attempt == 0 // Only retry on first attempt
        }
    }

    let strategy_with_custom = RetryStrategy::new().with_predicate(TestPredicate);
    let counter = Arc::new(AtomicU32::new(0));

    let result = strategy_with_custom.execute(|| {
        let c = counter.clone();
        async move {
            let count = c.fetch_add(1, Ordering::SeqCst);
            Err::<i32, DaemonError>(DaemonError::NetworkConnection {
                message: format!("Attempt {}", count + 1)
            })
        }
    }).await;

    assert!(result.is_err());
    assert_eq!(counter.load(Ordering::SeqCst), 2); // Should try twice
}

#[tokio::test]
async fn test_circuit_breaker_config_comprehensive() {
    // Test all circuit breaker configuration methods
    let config = CircuitBreakerConfig::new()
        .failure_threshold(3)
        .success_threshold(2)
        .recovery_timeout(Duration::from_secs(30))
        .failure_window_size(10)
        .minimum_requests(3)
        .request_timeout(Duration::from_secs(5));

    assert!(config.validate().is_ok());

    // Test invalid configurations
    let invalid_configs = vec![
        CircuitBreakerConfig::new().failure_threshold(0),
        CircuitBreakerConfig::new().success_threshold(0),
        CircuitBreakerConfig::new().failure_window_size(0),
        CircuitBreakerConfig::new().minimum_requests(0),
        CircuitBreakerConfig::new().recovery_timeout(Duration::ZERO),
        CircuitBreakerConfig::new().request_timeout(Duration::ZERO),
    ];

    for config in invalid_configs {
        assert!(config.validate().is_err());
    }
}

#[tokio::test]
async fn test_circuit_breaker_full_lifecycle() {
    let config = CircuitBreakerConfig::new()
        .failure_threshold(2)
        .success_threshold(2)
        .recovery_timeout(Duration::from_millis(50))
        .minimum_requests(1)
        .request_timeout(Duration::from_millis(100));

    let cb = CircuitBreaker::new("test-service".to_string(), config).unwrap();

    // 1. Start in closed state
    assert_eq!(cb.state().await, CircuitState::Closed);

    // 2. Success should keep it closed
    let result = cb.execute(|| async { Ok::<i32, DaemonError>(42) }).await;
    assert_eq!(result.unwrap(), 42);
    assert_eq!(cb.state().await, CircuitState::Closed);

    // 3. Generate failures to open circuit
    for _ in 0..3 {
        let _ = cb.execute(|| async {
            Err::<i32, DaemonError>(DaemonError::NetworkConnection {
                message: "Service failure".to_string()
            })
        }).await;
    }

    // Should be open now
    assert_eq!(cb.state().await, CircuitState::Open);

    // 4. Requests should be immediately rejected
    let result = cb.execute(|| async { Ok::<i32, DaemonError>(42) }).await;
    assert!(matches!(result, Err(DaemonError::CircuitBreakerOpen { .. })));

    // 5. Wait for recovery timeout
    sleep(Duration::from_millis(100)).await;

    // 6. Next request should transition to half-open
    let result = cb.execute(|| async { Ok::<i32, DaemonError>(42) }).await;
    assert_eq!(result.unwrap(), 42);

    // Should be in half-open or closed state
    let state = cb.state().await;
    assert!(state == CircuitState::HalfOpen || state == CircuitState::Closed);
}

#[tokio::test]
async fn test_circuit_breaker_timeout_handling() {
    let config = CircuitBreakerConfig::new()
        .failure_threshold(1)
        .request_timeout(Duration::from_millis(50))
        .minimum_requests(1);

    let cb = CircuitBreaker::new("timeout-test".to_string(), config).unwrap();

    // Test operation that times out
    let result = cb.execute(|| async {
        sleep(Duration::from_millis(100)).await; // Longer than timeout
        Ok::<i32, DaemonError>(42)
    }).await;

    assert!(matches!(result, Err(DaemonError::NetworkTimeout { .. })));

    // Circuit should be open after timeout
    assert_eq!(cb.state().await, CircuitState::Open);
}

#[tokio::test]
async fn test_circuit_breaker_statistics() {
    let cb = CircuitBreaker::new_default("stats-test".to_string()).unwrap();

    // Execute some operations
    let _ = cb.execute(|| async { Ok::<i32, DaemonError>(1) }).await;
    let _ = cb.execute(|| async { Ok::<i32, DaemonError>(2) }).await;
    let _ = cb.execute(|| async {
        Err::<i32, DaemonError>(DaemonError::NetworkConnection {
            message: "failure".to_string()
        })
    }).await;

    let stats = cb.stats().await;
    assert_eq!(stats.total_requests, 3);
    assert_eq!(stats.total_failures, 1);
    assert!(stats.failure_rate > 0.0);
    assert!(stats.time_since_state_change >= Duration::ZERO);
}

#[tokio::test]
async fn test_circuit_breaker_concurrent_access() {
    let cb = Arc::new(CircuitBreaker::new_default("concurrent-test".to_string()).unwrap());
    let mut handles = vec![];

    // Spawn multiple concurrent operations
    for i in 0..10 {
        let cb_clone = cb.clone();
        let handle = tokio::spawn(async move {
            cb_clone.execute(|| async {
                Ok::<i32, DaemonError>(i)
            }).await
        });
        handles.push(handle);
    }

    // Wait for all operations to complete
    let mut success_count = 0;
    for handle in handles {
        if handle.await.unwrap().is_ok() {
            success_count += 1;
        }
    }

    assert_eq!(success_count, 10);

    let stats = cb.stats().await;
    assert_eq!(stats.total_requests, 10);
    assert_eq!(stats.total_failures, 0);
}

#[tokio::test]
async fn test_circuit_breaker_manual_control() {
    let cb = CircuitBreaker::new_default("manual-test".to_string()).unwrap();

    // Test force open
    cb.force_open().await;
    assert_eq!(cb.state().await, CircuitState::Open);

    // Test force close
    cb.force_close().await;
    assert_eq!(cb.state().await, CircuitState::Closed);
}

#[tokio::test]
async fn test_complex_error_scenarios() {
    // Test mixed error types with retry strategy
    let strategy = RetryStrategy::new();
    let counter = Arc::new(AtomicU32::new(0));

    let result = strategy.execute(|| {
        let c = counter.clone();
        async move {
            let count = c.fetch_add(1, Ordering::SeqCst);
            match count {
                0 => Err(DaemonError::NetworkConnection { message: "first failure".to_string() }),
                1 => Err(DaemonError::NetworkTimeout { timeout_ms: 1000 }),
                2 => Ok::<String, DaemonError>("success".to_string()),
                _ => unreachable!(),
            }
        }
    }).await;

    assert_eq!(result.unwrap(), "success");
    assert_eq!(counter.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn test_error_propagation_through_layers() {
    // Test error propagation through retry -> circuit breaker -> actual error
    let cb = CircuitBreaker::new_default("propagation-test".to_string()).unwrap();
    let strategy = RetryStrategy::new();

    // Simulate layered error handling
    let result = strategy.execute(|| {
        let cb_clone = cb.clone();
        async move {
            cb_clone.execute(|| async {
                Err::<i32, DaemonError>(DaemonError::InvalidInput {
                    message: "This should not be retried".to_string()
                })
            }).await
        }
    }).await;

    // Should get the original non-retryable error, not retry limit exceeded
    assert!(matches!(result, Err(DaemonError::InvalidInput { .. })));
}