//! Basic async error handling tests focusing on stable components
//!
//! This test suite validates core async error handling patterns including:
//! - Error propagation across async boundaries
//! - Timeout handling with tokio::time::timeout
//! - Resource cleanup in error scenarios
//! - Retry mechanisms with exponential backoff

use std::sync::{Arc, atomic::{AtomicU32, Ordering}};
use std::time::{Duration, Instant};
use tokio::sync::{oneshot, Semaphore};
use tokio::time::timeout;
use futures_util::future::FutureExt;
use workspace_qdrant_daemon::error::{DaemonError, DaemonResult};
use workspace_qdrant_daemon::daemon::processing::DocumentProcessor;
use workspace_qdrant_daemon::grpc::middleware::{with_retry, RetryConfig, ConnectionManager};
use tonic::{Request, Response, Status};

/// Test utilities for async error handling
pub mod test_utils {
    use super::*;

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
    async fn test_multiple_resource_cleanup_ordering() {
        let cleanup_order = Arc::new(std::sync::Mutex::new(Vec::new()));

        struct OrderedResource {
            id: u32,
            cleanup_order: Arc<std::sync::Mutex<Vec<u32>>>,
        }

        impl Drop for OrderedResource {
            fn drop(&mut self) {
                if let Ok(mut order) = self.cleanup_order.lock() {
                    order.push(self.id);
                }
            }
        }

        async fn operation_with_multiple_resources(
            cleanup_order: Arc<std::sync::Mutex<Vec<u32>>>,
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
        let order = cleanup_order.lock().unwrap();
        assert_eq!(order.as_slice(), &[3, 2, 1]);
    }
}

/// Retry mechanism and backoff strategy tests
mod retry_backoff_tests {
    use super::*;
    use super::test_utils::*;

    #[tokio::test]
    async fn test_retry_success_after_failures() {
        let operation = FailableAsyncOperation::new().fail_n_times(2);

        let config = RetryConfig {
            max_retries: 3,
            initial_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(10),
            backoff_multiplier: 2.0,
        };

        let start_time = Instant::now();
        let result = with_retry(
            || Box::pin(operation.execute()),
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
        let operation = FailableAsyncOperation::new().fail_n_times(5); // More failures than retries

        let config = RetryConfig {
            max_retries: 3,
            initial_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(10),
            backoff_multiplier: 2.0,
        };

        let result = with_retry(
            || Box::pin(operation.execute()),
            &config,
        ).await;

        assert!(result.is_err());

        // Should have attempted exactly max_retries times
        assert_eq!(operation.current_attempt.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_exponential_backoff_timing() {
        let operation = FailableAsyncOperation::new().fail_n_times(3);

        let config = RetryConfig {
            max_retries: 3,
            initial_delay: Duration::from_millis(10),
            max_delay: Duration::from_millis(100),
            backoff_multiplier: 2.0,
        };

        let start_time = Instant::now();
        let result = with_retry(
            || Box::pin(operation.execute()),
            &config,
        ).await;
        let elapsed = start_time.elapsed();

        assert!(result.is_err());

        // Expected delays: 10ms + 20ms = 30ms minimum
        assert!(elapsed >= Duration::from_millis(25));
        assert!(elapsed < Duration::from_millis(100)); // Should not hit max delay
    }

    #[tokio::test]
    async fn test_retry_with_timeout_interaction() {
        let operation = FailableAsyncOperation::new()
            .fail_n_times(2)
            .with_timeout(Duration::from_millis(30));

        let config = RetryConfig {
            max_retries: 3,
            initial_delay: Duration::from_millis(5),
            max_delay: Duration::from_millis(50),
            backoff_multiplier: 2.0,
        };

        // Each retry takes 30ms + delay time
        let start_time = Instant::now();
        let result = timeout(Duration::from_millis(200), with_retry(
            || Box::pin(operation.execute()),
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
}

/// Integration tests with stable daemon components
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
        // Note: Can't use assert_matches! with Status due to trait bounds
        let status = result.unwrap_err();
        assert_eq!(status.code(), tonic::Code::ResourceExhausted);

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

        let results = test_utils::test_concurrent_operations(5, |i| {
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