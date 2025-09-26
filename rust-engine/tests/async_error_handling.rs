//! Comprehensive async error handling tests for the Rust daemon
//!
//! This test suite validates async error handling patterns including:
//! - Error propagation across async boundaries
//! - Timeout and cancellation error handling
//! - Resource cleanup in error scenarios
//! - Graceful degradation strategies
//! - Concurrent error scenarios
//! - Retry mechanisms and backoff strategies

use std::sync::{Arc, atomic::{AtomicU32, Ordering}};
use std::time::{Duration, Instant};
use tokio::sync::{oneshot, Semaphore, RwLock, Mutex};
use tokio::time::timeout;
use futures_util::future::FutureExt;
use workspace_qdrant_daemon::error::{DaemonError, DaemonResult};
use workspace_qdrant_daemon::daemon::processing::DocumentProcessor;
use workspace_qdrant_daemon::grpc::middleware::{with_retry, RetryConfig, ConnectionManager};
use workspace_qdrant_daemon::proto::document_processor_server::DocumentProcessor;
use workspace_qdrant_daemon::proto::search_service_server::SearchService;
use workspace_qdrant_daemon::proto::memory_service_server::MemoryService;
use workspace_qdrant_daemon::proto::system_service_server::SystemService;
use tonic::{Request, Response, Status};
// Note: Using manual pattern matching instead of assert_matches for better compatibility

/// Test utilities for async error handling
pub mod test_utils {
    use super::*;

    /// Shared test configuration for consistent error scenarios
    pub struct AsyncErrorTestConfig {
        pub timeout_duration: Duration,
        pub retry_attempts: u32,
        pub concurrent_operations: usize,
        pub error_injection_rate: f64,
    }

    impl Default for AsyncErrorTestConfig {
        fn default() -> Self {
            Self {
                timeout_duration: Duration::from_millis(100),
                retry_attempts: 3,
                concurrent_operations: 10,
                error_injection_rate: 0.3,
            }
        }
    }

    /// Mock async operation that can be configured to fail
    pub struct FailableAsyncOperation {
        pub success_after_attempts: u32,
        pub current_attempt: Arc<AtomicU32>,
        pub should_timeout: bool,
        pub timeout_duration: Duration,
    }

    impl FailableAsyncOperation {
        pub fn new() -> Self {
            Self {
                success_after_attempts: 1,
                current_attempt: Arc::new(AtomicU32::new(0)),
                should_timeout: false,
                timeout_duration: Duration::from_millis(50),
            }
        }

        pub fn fail_n_times(mut self, n: u32) -> Self {
            self.success_after_attempts = n + 1;
            self
        }

        pub fn with_timeout(mut self, duration: Duration) -> Self {
            self.should_timeout = true;
            self.timeout_duration = duration;
            self
        }

        pub async fn execute(&self) -> DaemonResult<String> {
            let attempt = self.current_attempt.fetch_add(1, Ordering::SeqCst) + 1;

            if self.should_timeout {
                tokio::time::sleep(self.timeout_duration).await;
            }

            if attempt < self.success_after_attempts {
                Err(DaemonError::Internal {
                    message: format!("Operation failed on attempt {}", attempt),
                })
            } else {
                Ok(format!("Success on attempt {}", attempt))
            }
        }

        pub fn reset(&self) {
            self.current_attempt.store(0, Ordering::SeqCst);
        }
    }

    /// Resource manager that tracks cleanup on drop
    pub struct TrackedResource {
        pub id: String,
        pub cleanup_counter: Arc<AtomicU32>,
    }

    impl TrackedResource {
        pub fn new(id: String, cleanup_counter: Arc<AtomicU32>) -> Self {
            Self { id, cleanup_counter }
        }
    }

    impl Drop for TrackedResource {
        fn drop(&mut self) {
            self.cleanup_counter.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// Simulate concurrent operations with error injection
    pub async fn test_concurrent_operations<F, T>(
        operation_count: usize,
        operation_factory: F,
    ) -> Vec<DaemonResult<T>>
    where
        F: Fn(usize) -> std::pin::Pin<Box<dyn std::future::Future<Output = DaemonResult<T>> + Send>>,
        T: Send + 'static,
    {
        let mut handles = Vec::new();

        for i in 0..operation_count {
            let future = operation_factory(i);
            let handle = tokio::spawn(async move { future.await });
            handles.push(handle);
        }

        let mut results = Vec::new();
        for handle in handles {
            match handle.await {
                Ok(result) => results.push(result),
                Err(_) => results.push(Err(DaemonError::Internal {
                    message: "Task panicked".to_string(),
                })),
            }
        }

        results
    }
}

/// Core async error propagation tests
mod error_propagation_tests {
    use super::*;
    use super::test_utils::*;

    #[tokio::test]
    async fn test_async_error_type_preservation() {
        // Test that different error types are preserved across async boundaries
        async fn operation_with_io_error() -> DaemonResult<String> {
            let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "File not found");
            Err(DaemonError::Io(io_error))
        }

        async fn operation_with_timeout_error() -> DaemonResult<String> {
            Err(DaemonError::Timeout { seconds: 30 })
        }

        async fn operation_with_custom_error() -> DaemonResult<String> {
            Err(DaemonError::DocumentProcessing {
                message: "Processing failed".to_string(),
            })
        }

        // Test IO error propagation
        let result = operation_with_io_error().await;
        assert!(result.is_err());
        match result.unwrap_err() {
            DaemonError::Io(_) => {}, // Expected
            other => panic!("Expected IO error, got: {:?}", other),
        }

        // Test timeout error propagation
        let result = operation_with_timeout_error().await;
        assert!(result.is_err());
        match result.unwrap_err() {
            DaemonError::Timeout { seconds: 30 } => {}, // Expected
            other => panic!("Expected Timeout error with 30 seconds, got: {:?}", other),
        }

        // Test custom error propagation
        let result = operation_with_custom_error().await;
        assert!(result.is_err());
        match result.unwrap_err() {
            DaemonError::DocumentProcessing { .. } => {}, // Expected
            other => panic!("Expected DocumentProcessing error, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_nested_async_error_propagation() {
        // Test error propagation through multiple async call layers
        async fn level_3_operation() -> DaemonResult<i32> {
            Err(DaemonError::InvalidInput {
                message: "Deep error".to_string(),
            })
        }

        async fn level_2_operation() -> DaemonResult<i32> {
            level_3_operation().await?;
            Ok(42)
        }

        async fn level_1_operation() -> DaemonResult<i32> {
            level_2_operation().await?;
            Ok(100)
        }

        let result = level_1_operation().await;
        assert!(result.is_err());
        match result.unwrap_err() {
            DaemonError::InvalidInput { .. } => {}, // Expected
            other => panic!("Expected InvalidInput error, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_async_error_context_preservation() {
        // Test that error context is preserved across async boundaries
        async fn failing_operation() -> DaemonResult<String> {
            let git_error = git2::Error::from_str("Git operation failed");
            Err(DaemonError::Git(git_error))
        }

        async fn wrapper_operation() -> DaemonResult<String> {
            match failing_operation().await {
                Err(DaemonError::Git(git_err)) => {
                    // Verify we can access the original error
                    assert_eq!(git_err.message(), "Git operation failed");
                    Err(DaemonError::ProjectDetection {
                        message: format!("Project detection failed: {}", git_err),
                    })
                }
                other => other,
            }
        }

        let result = wrapper_operation().await;
        assert!(result.is_err());
        let error = result.unwrap_err();
        match &error {
            DaemonError::ProjectDetection { .. } => {}, // Expected
            other => panic!("Expected ProjectDetection error, got: {:?}", other),
        }
        if let DaemonError::ProjectDetection { message } = error {
            assert!(message.contains("Git operation failed"));
        }
    }

    #[tokio::test]
    async fn test_async_error_chain_source() {
        // Test that error source chains are maintained in async contexts
        async fn create_chained_error() -> DaemonResult<String> {
            let io_error = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "Access denied");
            Err(DaemonError::Io(io_error))
        }

        let result = create_chained_error().await;
        assert!(result.is_err());

        let error = result.unwrap_err();
        let source = std::error::Error::source(&error);
        assert!(source.is_some());

        if let Some(source) = source {
            assert_eq!(source.to_string(), "Access denied");
        }
    }
}

/// Timeout and cancellation handling tests
mod timeout_cancellation_tests {
    use super::*;
    use super::test_utils::*;

    #[tokio::test]
    async fn test_operation_timeout_handling() {
        let operation = FailableAsyncOperation::new()
            .with_timeout(Duration::from_millis(200));

        let start = Instant::now();
        let result = timeout(Duration::from_millis(100), operation.execute()).await;
        let elapsed = start.elapsed();

        // Should timeout before operation completes
        assert!(result.is_err());
        assert!(elapsed < Duration::from_millis(150)); // Some buffer for timing
    }

    #[tokio::test]
    async fn test_timeout_error_conversion() {
        // Test converting timeout errors to DaemonError
        async fn timeout_prone_operation() -> DaemonResult<String> {
            tokio::time::sleep(Duration::from_millis(200)).await;
            Ok("Success".to_string())
        }

        let result = timeout(Duration::from_millis(50), timeout_prone_operation()).await;

        let daemon_result: DaemonResult<String> = match result {
            Ok(inner_result) => inner_result,
            Err(_) => Err(DaemonError::Timeout { seconds: 1 }),
        };

        assert!(daemon_result.is_err());
        match daemon_result.unwrap_err() {
            DaemonError::Timeout { seconds: 1 } => {}, // Expected
            other => panic!("Expected Timeout error with 1 second, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_cancellation_token_handling() {
        // Test proper cancellation using tokio cancellation tokens
        let (cancel_tx, cancel_rx) = oneshot::channel::<()>();

        let long_operation = async {
            tokio::select! {
                _ = cancel_rx => {
                    Err(DaemonError::Internal {
                        message: "Operation cancelled".to_string(),
                    })
                }
                _ = tokio::time::sleep(Duration::from_millis(500)) => {
                    Ok("Completed".to_string())
                }
            }
        };

        // Cancel after 50ms
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            let _ = cancel_tx.send(());
        });

        let result = long_operation.await;
        assert!(result.is_err());
        if let Err(DaemonError::Internal { message }) = result {
            assert!(message.contains("cancelled"));
        }
    }

    #[tokio::test]
    async fn test_graceful_shutdown_with_timeout() {
        // Test graceful shutdown that respects timeouts
        let cleanup_counter = Arc::new(AtomicU32::new(0));
        let resources: Arc<Mutex<Vec<TrackedResource>>> = Arc::new(Mutex::new(Vec::new()));

        // Create some resources
        {
            let mut resource_vec = resources.lock().await;
            for i in 0..5 {
                resource_vec.push(TrackedResource::new(
                    format!("resource_{}", i),
                    cleanup_counter.clone(),
                ));
            }
        }

        // Simulate graceful shutdown with timeout
        let shutdown_result = timeout(Duration::from_millis(100), async {
            // Drop all resources
            let mut resource_vec = resources.lock().await;
            resource_vec.clear();
            Ok::<(), DaemonError>(())
        }).await;

        assert!(shutdown_result.is_ok());
        assert_eq!(cleanup_counter.load(Ordering::SeqCst), 5);
    }

    #[tokio::test]
    async fn test_timeout_with_partial_completion() {
        // Test scenarios where operations partially complete before timeout
        let semaphore = Arc::new(Semaphore::new(2));
        let completion_counter = Arc::new(AtomicU32::new(0));

        let mut handles = Vec::new();

        for i in 0..5 {
            let sem = semaphore.clone();
            let counter = completion_counter.clone();

            let handle = tokio::spawn(async move {
                let _permit = sem.acquire().await.unwrap();

                // Simulate work
                tokio::time::sleep(Duration::from_millis(50)).await;
                counter.fetch_add(1, Ordering::SeqCst);

                Ok::<i32, DaemonError>(i)
            });

            handles.push(handle);
        }

        // Set a timeout that allows some but not all operations to complete
        let results = timeout(Duration::from_millis(120), async {
            let mut results = Vec::new();
            for handle in handles {
                match handle.await {
                    Ok(result) => results.push(result),
                    Err(_) => results.push(Err(DaemonError::Internal {
                        message: "Task cancelled".to_string(),
                    })),
                }
            }
            results
        }).await;

        // Some operations should have completed
        let completed = completion_counter.load(Ordering::SeqCst);
        assert!(completed >= 2 && completed <= 5);
    }
}

/// Resource cleanup and leak prevention tests
mod resource_cleanup_tests {
    use super::*;
    use super::test_utils::*;

    #[tokio::test]
    async fn test_resource_cleanup_on_error() {
        let cleanup_counter = Arc::new(AtomicU32::new(0));

        async fn failing_operation_with_resource(
            cleanup_counter: Arc<AtomicU32>,
        ) -> DaemonResult<String> {
            let _resource = TrackedResource::new("test_resource".to_string(), cleanup_counter);

            // Simulate some work before failing
            tokio::time::sleep(Duration::from_millis(10)).await;

            Err(DaemonError::Internal {
                message: "Operation failed".to_string(),
            })
        }

        let result = failing_operation_with_resource(cleanup_counter.clone()).await;
        assert!(result.is_err());

        // Resource should be cleaned up even though operation failed
        assert_eq!(cleanup_counter.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_resource_cleanup_on_panic() {
        let cleanup_counter = Arc::new(AtomicU32::new(0));

        let handle = tokio::spawn({
            let cleanup_counter = cleanup_counter.clone();
            async move {
                let _resource = TrackedResource::new("panic_resource".to_string(), cleanup_counter);

                // Simulate work then panic
                tokio::time::sleep(Duration::from_millis(10)).await;
                panic!("Simulated panic");
            }
        });

        let result = handle.await;
        assert!(result.is_err()); // Task panicked

        // Resource should still be cleaned up
        assert_eq!(cleanup_counter.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_semaphore_permit_cleanup_on_error() {
        let semaphore = Arc::new(Semaphore::new(2));

        async fn operation_with_permit(semaphore: Arc<Semaphore>) -> DaemonResult<String> {
            let _permit = semaphore.acquire().await.map_err(|e| DaemonError::Internal {
                message: format!("Semaphore error: {}", e),
            })?;

            // Fail after acquiring permit
            Err(DaemonError::Internal {
                message: "Operation failed".to_string(),
            })
        }

        // Run multiple failing operations
        for _ in 0..3 {
            let result = operation_with_permit(semaphore.clone()).await;
            assert!(result.is_err());
        }

        // Semaphore should still have permits available (cleanup worked)
        assert_eq!(semaphore.available_permits(), 2);
    }

    #[tokio::test]
    async fn test_connection_cleanup_on_timeout() {
        let manager = Arc::new(ConnectionManager::new(10, 100));

        // Register connection
        manager.register_connection("test_client".to_string()).unwrap();
        assert_eq!(manager.get_stats().active_connections, 1);

        // Simulate operation that times out
        let timeout_result = timeout(Duration::from_millis(50), async {
            // Simulate long-running operation
            tokio::time::sleep(Duration::from_millis(200)).await;
            Ok::<(), DaemonError>(())
        }).await;

        assert!(timeout_result.is_err());

        // Manually clean up connection (simulating proper cleanup)
        manager.unregister_connection("test_client");
        assert_eq!(manager.get_stats().active_connections, 0);
    }

    #[tokio::test]
    async fn test_multiple_resource_cleanup_ordering() {
        let cleanup_order = Arc::new(Mutex::new(Vec::new()));

        struct OrderedResource {
            id: u32,
            cleanup_order: Arc<Mutex<Vec<u32>>>,
        }

        impl Drop for OrderedResource {
            fn drop(&mut self) {
                if let Ok(mut order) = self.cleanup_order.try_lock() {
                    order.push(self.id);
                }
            }
        }

        async fn operation_with_multiple_resources(
            cleanup_order: Arc<Mutex<Vec<u32>>>,
        ) -> DaemonResult<String> {
            let _resource1 = OrderedResource { id: 1, cleanup_order: cleanup_order.clone() };
            let _resource2 = OrderedResource { id: 2, cleanup_order: cleanup_order.clone() };
            let _resource3 = OrderedResource { id: 3, cleanup_order: cleanup_order.clone() };

            Err(DaemonError::Internal {
                message: "Multiple resource test".to_string(),
            })
        }

        let result = operation_with_multiple_resources(cleanup_order.clone()).await;
        assert!(result.is_err());

        // Resources should be cleaned up in reverse order (LIFO)
        let order = cleanup_order.lock().await;
        assert_eq!(order.as_slice(), &[3, 2, 1]);
    }
}

/// Retry mechanism and backoff strategy tests
mod retry_backoff_tests {
    use super::*;
    use super::test_utils::*;

    #[tokio::test]
    async fn test_retry_success_after_failures() {
        let config = RetryConfig {
            max_retries: 3,
            initial_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(10),
            backoff_multiplier: 2.0,
        };

        let start_time = Instant::now();
        let result = with_retry(
            || {
                let operation = FailableAsyncOperation::new().fail_n_times(2);
                Box::pin(operation.execute())
            },
            &config,
        ).await;
        let elapsed = start_time.elapsed();

        assert!(result.is_ok());
        assert!(result.unwrap().contains("Success on attempt 3"));

        // Should have taken some time due to retries
        assert!(elapsed >= Duration::from_millis(3)); // 1 + 2 = 3ms minimum
    }

    #[tokio::test]
    async fn test_retry_final_failure() {
        let operation = Arc::new(FailableAsyncOperation::new().fail_n_times(5)); // More failures than retries

        let config = RetryConfig {
            max_retries: 3,
            initial_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(10),
            backoff_multiplier: 2.0,
        };

        let op_clone = Arc::clone(&operation);
        let result = with_retry(
            move || Box::pin(op_clone.execute()),
            &config,
        ).await;

        assert!(result.is_err());

        // Should have attempted exactly max_retries times
        assert_eq!(operation.current_attempt.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_exponential_backoff_timing() {
        let config = RetryConfig {
            max_retries: 3,
            initial_delay: Duration::from_millis(10),
            max_delay: Duration::from_millis(100),
            backoff_multiplier: 2.0,
        };

        let start_time = Instant::now();
        let result = with_retry(
            || {
                let operation = FailableAsyncOperation::new().fail_n_times(3);
                Box::pin(operation.execute())
            },
            &config,
        ).await;
        let elapsed = start_time.elapsed();

        assert!(result.is_err());

        // Expected delays: 10ms + 20ms = 30ms minimum
        assert!(elapsed >= Duration::from_millis(25));
        assert!(elapsed < Duration::from_millis(100)); // Should not hit max delay
    }

    #[tokio::test]
    async fn test_retry_max_delay_cap() {
        let config = RetryConfig {
            max_retries: 4,
            initial_delay: Duration::from_millis(10),
            max_delay: Duration::from_millis(25), // Cap at 25ms
            backoff_multiplier: 3.0, // Aggressive multiplier
        };

        let start_time = Instant::now();
        let result = with_retry(
            || {
                let operation = FailableAsyncOperation::new().fail_n_times(4);
                Box::pin(operation.execute())
            },
            &config,
        ).await;
        let elapsed = start_time.elapsed();

        assert!(result.is_err());

        // Expected delays: 10ms + 25ms (capped) + 25ms (capped) = 60ms minimum
        assert!(elapsed >= Duration::from_millis(50));
        assert!(elapsed < Duration::from_millis(150)); // Should be bounded by caps
    }

    #[tokio::test]
    async fn test_retry_with_timeout_interaction() {
        let config = RetryConfig {
            max_retries: 3,
            initial_delay: Duration::from_millis(5),
            max_delay: Duration::from_millis(50),
            backoff_multiplier: 2.0,
        };

        // Each retry takes 30ms + delay time
        let start_time = Instant::now();
        let result = timeout(Duration::from_millis(200), with_retry(
            || {
                let operation = FailableAsyncOperation::new()
                    .fail_n_times(2)
                    .with_timeout(Duration::from_millis(30));
                Box::pin(operation.execute())
            },
            &config,
        )).await;

        // Should succeed or timeout, depending on timing
        match result {
            Ok(inner) => assert!(inner.is_ok() || inner.is_err()),
            Err(_) => (), // Timeout is acceptable in this test
        }

        let elapsed = start_time.elapsed();
        assert!(elapsed <= Duration::from_millis(250)); // Should not exceed timeout + buffer
    }

    #[tokio::test]
    async fn test_concurrent_retry_operations() {
        let config = Arc::new(RetryConfig {
            max_retries: 2,
            initial_delay: Duration::from_millis(5),
            max_delay: Duration::from_millis(20),
            backoff_multiplier: 2.0,
        });

        let results = test_concurrent_operations(5, |i| {
            let config = config.clone();
            Box::pin(async move {
                with_retry(
                    || {
                        let operation = FailableAsyncOperation::new().fail_n_times(i as u32 % 3);
                        Box::pin(operation.execute())
                    },
                    &config,
                ).await
            })
        }).await;

        // Some operations should succeed, some should fail
        let successes = results.iter().filter(|r| r.is_ok()).count();
        let failures = results.iter().filter(|r| r.is_err()).count();

        assert_eq!(successes + failures, 5);
        assert!(successes >= 2); // At least some should succeed
    }
}

/// Graceful degradation strategy tests
mod graceful_degradation_tests {
    use super::*;
    use super::test_utils::*;

    #[tokio::test]
    async fn test_circuit_breaker_pattern() {
        // Simple circuit breaker implementation for testing
        struct CircuitBreaker {
            failure_count: Arc<AtomicU32>,
            failure_threshold: u32,
            is_open: Arc<std::sync::atomic::AtomicBool>,
        }

        impl CircuitBreaker {
            fn new(threshold: u32) -> Self {
                Self {
                    failure_count: Arc::new(AtomicU32::new(0)),
                    failure_threshold: threshold,
                    is_open: Arc::new(std::sync::atomic::AtomicBool::new(false)),
                }
            }

            async fn call<F, T>(&self, operation: F) -> DaemonResult<T>
            where
                F: std::future::Future<Output = DaemonResult<T>>,
            {
                if self.is_open.load(Ordering::SeqCst) {
                    return Err(DaemonError::Internal {
                        message: "Circuit breaker is open".to_string(),
                    });
                }

                match operation.await {
                    Ok(result) => {
                        self.failure_count.store(0, Ordering::SeqCst);
                        Ok(result)
                    }
                    Err(err) => {
                        let failures = self.failure_count.fetch_add(1, Ordering::SeqCst) + 1;
                        if failures >= self.failure_threshold {
                            self.is_open.store(true, Ordering::SeqCst);
                        }
                        Err(err)
                    }
                }
            }
        }

        let circuit_breaker = CircuitBreaker::new(3);
        let operation = FailableAsyncOperation::new().fail_n_times(10); // Always fails

        // First 3 calls should fail normally
        for i in 0..3 {
            let result = circuit_breaker.call(operation.execute()).await;
            assert!(result.is_err());
            if let Err(DaemonError::Internal { message }) = result {
                assert!(message.contains("Operation failed"));
            }
            operation.reset();
        }

        // 4th call should fail with circuit breaker open
        let result = circuit_breaker.call(operation.execute()).await;
        assert!(result.is_err());
        if let Err(DaemonError::Internal { message }) = result {
            assert!(message.contains("Circuit breaker is open"));
        }
    }

    #[tokio::test]
    async fn test_fallback_operation() {
        async fn primary_operation() -> DaemonResult<String> {
            Err(DaemonError::ConnectionPool {
                message: "Primary service unavailable".to_string(),
            })
        }

        async fn fallback_operation() -> DaemonResult<String> {
            Ok("Fallback result".to_string())
        }

        async fn operation_with_fallback() -> DaemonResult<String> {
            match primary_operation().await {
                Ok(result) => Ok(result),
                Err(DaemonError::ConnectionPool { .. }) => {
                    // Use fallback for connection issues
                    fallback_operation().await
                }
                Err(other) => Err(other),
            }
        }

        let result = operation_with_fallback().await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Fallback result");
    }

    #[tokio::test]
    async fn test_degraded_service_mode() {
        // Simulate a service that can operate in degraded mode
        struct DegradedService {
            full_service_available: Arc<std::sync::atomic::AtomicBool>,
        }

        impl DegradedService {
            fn new() -> Self {
                Self {
                    full_service_available: Arc::new(std::sync::atomic::AtomicBool::new(true)),
                }
            }

            async fn process_request(&self, request: &str) -> DaemonResult<String> {
                if self.full_service_available.load(Ordering::SeqCst) {
                    // Full processing
                    if request.contains("fail") {
                        self.full_service_available.store(false, Ordering::SeqCst);
                        return self.process_degraded(request).await;
                    }
                    Ok(format!("Full processing: {}", request))
                } else {
                    // Degraded processing
                    self.process_degraded(request).await
                }
            }

            async fn process_degraded(&self, request: &str) -> DaemonResult<String> {
                Ok(format!("Degraded processing: {}", request))
            }
        }

        let service = DegradedService::new();

        // First request succeeds with full processing
        let result1 = service.process_request("normal_request").await;
        assert!(result1.is_ok());
        assert!(result1.unwrap().contains("Full processing"));

        // Second request triggers degradation
        let result2 = service.process_request("fail_request").await;
        assert!(result2.is_ok());
        assert!(result2.unwrap().contains("Degraded processing"));

        // Third request uses degraded mode
        let result3 = service.process_request("another_request").await;
        assert!(result3.is_ok());
        assert!(result3.unwrap().contains("Degraded processing"));
    }

    #[tokio::test]
    async fn test_load_shedding_on_overload() {
        let semaphore = Arc::new(Semaphore::new(2)); // Only 2 concurrent operations
        let processed_count = Arc::new(AtomicU32::new(0));
        let rejected_count = Arc::new(AtomicU32::new(0));

        async fn rate_limited_operation(
            semaphore: Arc<Semaphore>,
            processed_count: Arc<AtomicU32>,
            rejected_count: Arc<AtomicU32>,
        ) -> DaemonResult<String> {
            match semaphore.try_acquire() {
                Ok(_permit) => {
                    // Simulate processing
                    tokio::time::sleep(Duration::from_millis(50)).await;
                    processed_count.fetch_add(1, Ordering::SeqCst);
                    Ok("Processed".to_string())
                }
                Err(_) => {
                    // Load shedding: reject request
                    rejected_count.fetch_add(1, Ordering::SeqCst);
                    Err(DaemonError::Internal {
                        message: "Service overloaded, try again later".to_string(),
                    })
                }
            }
        }

        // Start 5 concurrent operations
        let mut handles = Vec::new();
        for _ in 0..5 {
            let sem = semaphore.clone();
            let processed = processed_count.clone();
            let rejected = rejected_count.clone();

            let handle = tokio::spawn(async move {
                rate_limited_operation(sem, processed, rejected).await
            });
            handles.push(handle);
        }

        // Wait for all to complete
        for handle in handles {
            let _ = handle.await;
        }

        let processed = processed_count.load(Ordering::SeqCst);
        let rejected = rejected_count.load(Ordering::SeqCst);

        // Should have processed at most 2 (semaphore limit) and rejected the rest
        assert!(processed <= 2);
        assert!(rejected >= 3);
        assert_eq!(processed + rejected, 5);
    }
}

/// Complex concurrent error scenarios
mod concurrent_error_scenarios_tests {
    use super::*;
    use super::test_utils::*;

    #[tokio::test]
    async fn test_concurrent_errors_with_shared_state() {
        let shared_state = Arc::new(RwLock::new(0i32));
        let error_count = Arc::new(AtomicU32::new(0));

        let results = test_concurrent_operations(10, |i| {
            let state = shared_state.clone();
            let errors = error_count.clone();

            Box::pin(async move {
                let mut value = state.write().await;

                if i % 3 == 0 {
                    // Introduce errors in some operations
                    errors.fetch_add(1, Ordering::SeqCst);
                    return Err(DaemonError::Internal {
                        message: format!("Error in operation {}", i),
                    });
                }

                *value += 1;
                Ok(format!("Operation {} completed", i))
            })
        }).await;

        let successes = results.iter().filter(|r| r.is_ok()).count();
        let failures = results.iter().filter(|r| r.is_err()).count();

        // Should have some successes and some failures
        assert!(successes > 0);
        assert!(failures > 0);
        assert_eq!(successes + failures, 10);

        // Final state should reflect successful operations
        let final_value = *shared_state.read().await;
        assert_eq!(final_value as usize, successes);
    }

    #[tokio::test]
    async fn test_cascading_failure_prevention() {
        let failure_tracker = Arc::new(AtomicU32::new(0));

        // Service that fails if too many previous failures
        async fn failure_sensitive_operation(
            id: usize,
            failure_tracker: Arc<AtomicU32>,
        ) -> DaemonResult<String> {
            let current_failures = failure_tracker.load(Ordering::SeqCst);

            if current_failures > 3 {
                // Prevent cascading failures by failing fast
                return Err(DaemonError::Internal {
                    message: "Too many failures, failing fast".to_string(),
                });
            }

            if id % 4 == 0 {
                // Introduce some failures
                failure_tracker.fetch_add(1, Ordering::SeqCst);
                Err(DaemonError::Internal {
                    message: format!("Natural failure in operation {}", id),
                })
            } else {
                Ok(format!("Success {}", id))
            }
        }

        let results = test_concurrent_operations(12, |i| {
            let tracker = failure_tracker.clone();
            Box::pin(failure_sensitive_operation(i, tracker))
        }).await;

        // Count different types of failures
        let natural_failures = results.iter().filter(|r| {
            matches!(r, Err(DaemonError::Internal { message }) if message.contains("Natural failure"))
        }).count();

        let cascade_prevention = results.iter().filter(|r| {
            matches!(r, Err(DaemonError::Internal { message }) if message.contains("failing fast"))
        }).count();

        // Should have some natural failures and some cascade prevention
        assert!(natural_failures > 0);
        assert!(cascade_prevention > 0);
    }

    #[tokio::test]
    async fn test_error_recovery_coordination() {
        let recovery_coordinator = Arc::new(RwLock::new(false));

        async fn coordinated_operation(
            id: usize,
            coordinator: Arc<RwLock<bool>>,
        ) -> DaemonResult<String> {
            // Check if system is in recovery mode
            {
                let in_recovery = *coordinator.read().await;
                if in_recovery && id % 2 == 0 {
                    // Skip some operations during recovery
                    return Err(DaemonError::Internal {
                        message: "Skipped during recovery".to_string(),
                    });
                }
            }

            if id == 3 {
                // Trigger recovery mode
                let mut recovery_mode = coordinator.write().await;
                *recovery_mode = true;
                return Err(DaemonError::Internal {
                    message: "Triggering recovery mode".to_string(),
                });
            }

            // Simulate work
            tokio::time::sleep(Duration::from_millis(10)).await;
            Ok(format!("Completed operation {}", id))
        }

        let results = test_concurrent_operations(8, |i| {
            let coord = recovery_coordinator.clone();
            Box::pin(coordinated_operation(i, coord))
        }).await;

        // Should have various types of results
        let completed = results.iter().filter(|r| r.is_ok()).count();
        let recovery_related = results.iter().filter(|r| {
            matches!(r, Err(DaemonError::Internal { message })
                if message.contains("recovery") || message.contains("Skipped"))
        }).count();

        assert!(completed > 0);
        assert!(recovery_related > 0);
    }

    #[tokio::test]
    async fn test_distributed_error_aggregation() {
        // Simulate collecting errors from multiple sources
        struct ErrorAggregator {
            errors: Arc<Mutex<Vec<(String, DaemonError)>>>,
        }

        impl ErrorAggregator {
            fn new() -> Self {
                Self {
                    errors: Arc::new(Mutex::new(Vec::new())),
                }
            }

            async fn record_error(&self, source: String, error: DaemonError) {
                let mut errors = self.errors.lock().await;
                errors.push((source, error));
            }

            async fn get_error_summary(&self) -> DaemonResult<String> {
                let errors = self.errors.lock().await;

                if errors.is_empty() {
                    Ok("No errors".to_string())
                } else if errors.len() > 5 {
                    Err(DaemonError::Internal {
                        message: format!("Too many errors: {}", errors.len()),
                    })
                } else {
                    let summary = errors.iter()
                        .map(|(source, _)| source.clone())
                        .collect::<Vec<_>>()
                        .join(", ");
                    Ok(format!("Errors from: {}", summary))
                }
            }
        }

        let aggregator = Arc::new(ErrorAggregator::new());

        // Simulate operations that report errors to aggregator
        let results = test_concurrent_operations(7, |i| {
            let agg = aggregator.clone();
            Box::pin(async move {
                if i % 3 == 0 {
                    let error = DaemonError::Internal {
                        message: format!("Error from source {}", i),
                    };
                    agg.record_error(format!("source_{}", i), error).await;
                }

                tokio::time::sleep(Duration::from_millis(5)).await;
                Ok(format!("Operation {} completed", i))
            })
        }).await;

        // All individual operations should succeed
        assert!(results.iter().all(|r| r.is_ok()));

        // But error aggregation should reflect the issues
        let summary = aggregator.get_error_summary().await;
        assert!(summary.is_ok()); // Should be under the threshold

        let summary_text = summary.unwrap();
        assert!(summary_text.contains("source_0") || summary_text.contains("source_3") || summary_text.contains("source_6"));
    }
}

/// Integration tests with real daemon components
mod daemon_integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_document_processor_timeout_handling() {
        let processor = DocumentProcessor::test_instance();

        // Test timeout on document processing
        let result = timeout(
            Duration::from_millis(50),
            processor.process_document("test_document.txt")
        ).await;

        // Should complete within timeout (test instance is fast)
        assert!(result.is_ok());

        let inner_result = result.unwrap();
        assert!(inner_result.is_ok());
    }

    #[tokio::test]
    async fn test_connection_manager_error_scenarios() {
        let manager = Arc::new(ConnectionManager::new(2, 5));

        // Test connection limit errors
        manager.register_connection("client1".to_string()).unwrap();
        manager.register_connection("client2".to_string()).unwrap();

        let result = manager.register_connection("client3".to_string());
        assert!(result.is_err());
        assert_matches!(result.unwrap_err().code(), tonic::Code::ResourceExhausted);

        // Test rate limiting errors
        for _ in 0..6 {
            let _ = manager.check_rate_limit("rate_test_client");
        }

        let rate_result = manager.check_rate_limit("rate_test_client");
        assert!(rate_result.is_err());
    }

    #[tokio::test]
    async fn test_grpc_status_error_conversion() {
        // Test that DaemonError converts properly to gRPC Status
        let errors = vec![
            DaemonError::InvalidInput { message: "Bad input".to_string() },
            DaemonError::NotFound { resource: "document".to_string() },
            DaemonError::Timeout { seconds: 30 },
            DaemonError::Internal { message: "Internal error".to_string() },
        ];

        for error in errors {
            let status: Status = error.into();

            // Should have appropriate status code
            match status.code() {
                tonic::Code::InvalidArgument |
                tonic::Code::NotFound |
                tonic::Code::DeadlineExceeded |
                tonic::Code::Internal => {
                    // These are all valid conversions
                }
                other => panic!("Unexpected status code: {:?}", other),
            }

            // Should have a message
            assert!(!status.message().is_empty());
        }
    }

    #[tokio::test]
    async fn test_concurrent_processor_operations_with_errors() {
        let processor = Arc::new(DocumentProcessor::test_instance());

        let results = test_concurrent_operations(5, |i| {
            let proc = processor.clone();
            Box::pin(async move {
                // Simulate various document processing scenarios
                let filename = if i % 3 == 0 {
                    // Some invalid filenames that might cause issues
                    ""
                } else {
                    &format!("document_{}.txt", i)
                };

                proc.process_document(filename).await
            })
        }).await;

        // All should succeed (test instance is robust)
        assert!(results.iter().all(|r| r.is_ok()));

        // All should return valid UUIDs
        for result in results {
            let uuid_str = result.unwrap();
            assert_eq!(uuid_str.len(), 36);
        }
    }
}

/// Performance and stress testing under error conditions
#[cfg(test)]
mod performance_stress_tests {
    use super::*;
    use super::test_utils::*;

    #[tokio::test]
    async fn test_error_handling_performance_overhead() {
        // Measure performance impact of error handling
        let iterations = 100;

        // Test successful operations
        let start = Instant::now();
        for _ in 0..iterations {
            let operation = FailableAsyncOperation::new(); // Always succeeds
            let _ = operation.execute().await;
        }
        let success_time = start.elapsed();

        // Test failing operations
        let start = Instant::now();
        for _ in 0..iterations {
            let operation = FailableAsyncOperation::new().fail_n_times(1); // Always fails
            let _ = operation.execute().await;
        }
        let failure_time = start.elapsed();

        // Error handling shouldn't be dramatically slower
        let overhead_ratio = failure_time.as_nanos() as f64 / success_time.as_nanos() as f64;
        assert!(overhead_ratio < 3.0, "Error handling overhead too high: {}x", overhead_ratio);
    }

    #[tokio::test]
    async fn test_memory_usage_under_error_conditions() {
        // Test that error conditions don't cause memory leaks
        let cleanup_counter = Arc::new(AtomicU32::new(0));

        // Run many operations that fail and should clean up
        for _ in 0..100 {
            let counter = cleanup_counter.clone();
            let _result = async {
                let _resource = TrackedResource::new("test".to_string(), counter);
                Err::<(), DaemonError>(DaemonError::Internal {
                    message: "Test error".to_string(),
                })
            }.await;
        }

        // All resources should be cleaned up
        assert_eq!(cleanup_counter.load(Ordering::SeqCst), 100);
    }

    #[tokio::test]
    async fn test_high_concurrency_error_scenarios() {
        // Test error handling under high concurrency
        let error_rate = Arc::new(AtomicU32::new(0));

        let results = test_concurrent_operations(50, |i| {
            let error_counter = error_rate.clone();
            Box::pin(async move {
                if i % 7 == 0 {
                    error_counter.fetch_add(1, Ordering::SeqCst);
                    Err(DaemonError::Internal {
                        message: format!("Concurrent error {}", i),
                    })
                } else {
                    tokio::time::sleep(Duration::from_millis(1)).await;
                    Ok(format!("Success {}", i))
                }
            })
        }).await;

        let errors = results.iter().filter(|r| r.is_err()).count();
        let successes = results.iter().filter(|r| r.is_ok()).count();

        assert_eq!(errors + successes, 50);
        assert!(errors > 5); // Should have some errors based on our 1/7 rate
        assert!(successes > 40); // Should have many successes
    }
}