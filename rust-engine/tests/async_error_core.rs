//! Core async error handling tests focusing on error types and basic patterns
//!
//! This test suite validates essential async error handling without complex dependencies

use std::sync::{Arc, atomic::{AtomicU32, Ordering}};
use std::time::{Duration, Instant};
use tokio::sync::{oneshot};
use tokio::time::timeout;
use workspace_qdrant_daemon::error::{DaemonError, DaemonResult};
use tonic::Status;

/// Test utilities for async error handling
mod test_utils {
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

    /// Simulate concurrent operations
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

/// Error conversion and status tests
mod error_conversion_tests {
    use super::*;

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
    async fn test_async_error_conversion_in_concurrent_context() {
        let results = test_utils::test_concurrent_operations(5, |i| {
            Box::pin(async move {
                if i % 2 == 0 {
                    Err(DaemonError::InvalidInput {
                        message: format!("Invalid input for operation {}", i),
                    })
                } else {
                    Ok(format!("Success {}", i))
                }
            })
        }).await;

        let errors: Vec<_> = results.iter().filter(|r| r.is_err()).collect();
        let successes: Vec<_> = results.iter().filter(|r| r.is_ok()).collect();

        assert!(errors.len() >= 2); // Should have some errors
        assert!(successes.len() >= 2); // Should have some successes

        // Test that errors can be converted to Status
        for error_result in errors {
            if let Err(error) = error_result {
                let status: Status = error.clone().into();
                assert!(!status.message().is_empty());
            }
        }
    }

    #[tokio::test]
    async fn test_error_debug_display_formatting() {
        let errors = vec![
            DaemonError::DocumentProcessing { message: "Test error".to_string() },
            DaemonError::Search { message: "Search failed".to_string() },
            DaemonError::Memory { message: "Memory allocation failed".to_string() },
            DaemonError::System { message: "System call failed".to_string() },
            DaemonError::ProjectDetection { message: "No git repository found".to_string() },
            DaemonError::ConnectionPool { message: "Pool exhausted".to_string() },
            DaemonError::Timeout { seconds: 30 },
            DaemonError::NotFound { resource: "document".to_string() },
            DaemonError::InvalidInput { message: "Invalid format".to_string() },
            DaemonError::Internal { message: "Unexpected state".to_string() },
        ];

        for error in errors {
            // Test Display formatting
            let display_str = format!("{}", error);
            assert!(!display_str.is_empty());

            // Test Debug formatting
            let debug_str = format!("{:?}", error);
            assert!(!debug_str.is_empty());
            assert!(debug_str.contains("DaemonError"));

            // Test async error propagation doesn't lose formatting
            let async_error = async move { Err::<(), DaemonError>(error) }.await;
            assert!(async_error.is_err());

            if let Err(propagated_error) = async_error {
                let propagated_display = format!("{}", propagated_error);
                assert!(!propagated_display.is_empty());
            }
        }
    }
}

/// Performance testing under error conditions
mod performance_tests {
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

        let results = test_utils::test_concurrent_operations(50, |i| {
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