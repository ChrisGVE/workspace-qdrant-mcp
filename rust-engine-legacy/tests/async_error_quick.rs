//! Quick async error handling validation tests
//! Fast execution tests to validate core async error patterns

use std::time::Duration;
use tokio::time::timeout;
use workspace_qdrant_daemon::error::{DaemonError, DaemonResult};
use tonic::Status;

#[tokio::test]
async fn test_basic_async_error_propagation() {
    async fn failing_operation() -> DaemonResult<String> {
        Err(DaemonError::Internal {
            message: "Test error".to_string(),
        })
    }

    async fn calling_operation() -> DaemonResult<String> {
        failing_operation().await?;
        Ok("Should not reach here".to_string())
    }

    let result = calling_operation().await;
    assert!(result.is_err());

    match result.unwrap_err() {
        DaemonError::Internal { message } => {
            assert_eq!(message, "Test error");
        }
        other => panic!("Expected Internal error, got: {:?}", other),
    }
}

#[tokio::test]
async fn test_timeout_error_handling() {
    async fn slow_operation() -> DaemonResult<String> {
        tokio::time::sleep(Duration::from_millis(100)).await;
        Ok("Completed".to_string())
    }

    let result = timeout(Duration::from_millis(50), slow_operation()).await;
    assert!(result.is_err()); // Should timeout

    // Convert timeout to DaemonError
    let daemon_result: DaemonResult<String> = match result {
        Ok(inner) => inner,
        Err(_) => Err(DaemonError::Timeout { seconds: 1 }),
    };

    assert!(daemon_result.is_err());
    match daemon_result.unwrap_err() {
        DaemonError::Timeout { seconds } => assert_eq!(seconds, 1),
        other => panic!("Expected Timeout error, got: {:?}", other),
    }
}

#[tokio::test]
async fn test_resource_cleanup_on_error() {
    use std::sync::{Arc, atomic::{AtomicU32, Ordering}};

    struct TestResource {
        counter: Arc<AtomicU32>,
    }

    impl Drop for TestResource {
        fn drop(&mut self) {
            self.counter.fetch_add(1, Ordering::SeqCst);
        }
    }

    let counter = Arc::new(AtomicU32::new(0));

    async fn operation_with_resource(counter: Arc<AtomicU32>) -> DaemonResult<String> {
        let _resource = TestResource { counter };

        // Fail after creating resource
        Err(DaemonError::Internal {
            message: "Operation failed".to_string(),
        })
    }

    let result = operation_with_resource(counter.clone()).await;
    assert!(result.is_err());

    // Resource should be cleaned up
    assert_eq!(counter.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn test_error_conversion_to_status() {
    let errors = vec![
        DaemonError::InvalidInput { message: "Bad input".to_string() },
        DaemonError::NotFound { resource: "document".to_string() },
        DaemonError::Timeout { seconds: 30 },
        DaemonError::Internal { message: "Internal error".to_string() },
    ];

    for error in errors {
        let status: Status = error.into();
        assert!(!status.message().is_empty());

        // Verify status codes are appropriate
        match status.code() {
            tonic::Code::InvalidArgument |
            tonic::Code::NotFound |
            tonic::Code::DeadlineExceeded |
            tonic::Code::Internal => {
                // Valid conversion
            }
            other => panic!("Unexpected status code: {:?}", other),
        }
    }
}

#[tokio::test]
async fn test_concurrent_error_handling() {
    use std::sync::{Arc, atomic::{AtomicU32, Ordering}};

    let error_count = Arc::new(AtomicU32::new(0));
    let success_count = Arc::new(AtomicU32::new(0));

    let mut handles = Vec::new();

    for i in 0..10 {
        let error_counter = error_count.clone();
        let success_counter = success_count.clone();

        let handle = tokio::spawn(async move {
            if i % 3 == 0 {
                error_counter.fetch_add(1, Ordering::SeqCst);
                Err::<String, DaemonError>(DaemonError::Internal {
                    message: format!("Error {}", i),
                })
            } else {
                success_counter.fetch_add(1, Ordering::SeqCst);
                Ok(format!("Success {}", i))
            }
        });

        handles.push(handle);
    }

    let mut results = Vec::new();
    for handle in handles {
        results.push(handle.await.unwrap());
    }

    let errors = results.iter().filter(|r| r.is_err()).count();
    let successes = results.iter().filter(|r| r.is_ok()).count();

    assert_eq!(errors + successes, 10);
    assert!(errors > 0);
    assert!(successes > 0);

    // Verify atomic counters match
    assert_eq!(error_count.load(Ordering::SeqCst) as usize, errors);
    assert_eq!(success_count.load(Ordering::SeqCst) as usize, successes);
}