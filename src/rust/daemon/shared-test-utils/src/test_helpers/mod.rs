//! Common test helper functions and utilities

pub mod assertions;
pub mod async_ops;
pub mod benchmarks;
pub mod generators;
pub mod tracing;

// Re-export all public items so callers can use the same paths as before
pub use async_ops::{
    assert_future_pending, assert_future_ready, assert_future_ready_err, assert_future_ready_ok,
    measure_time, retry_with_backoff, run_concurrent, test_async_timing,
    test_concurrent_operations, wait_for_async_condition, wait_for_condition, with_custom_timeout,
    with_timeout,
};

pub use assertions::{assert_approx_eq, assert_vectors_approx_eq};

pub use benchmarks::{MemoryTracker, PerformanceBenchmark};

pub use generators::{
    env_test_flag, generate_large_text, generate_test_collection, generate_test_documents,
    generate_test_id, generate_test_vectors, has_test_resource, is_ci, read_test_resource,
    read_test_resource_bytes, require_ci, skip_in_ci, test_data_dir, test_resource,
};

pub use tracing::init_test_tracing;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestResult;
    use std::time::Duration;

    #[tokio::test]
    async fn test_with_timeout_success() -> TestResult {
        let result = with_timeout(async {
            tokio::time::sleep(Duration::from_millis(10)).await;
            Ok::<i32, Box<dyn std::error::Error + Send + Sync>>(42)
        })
        .await?;

        assert_eq!(result, 42);
        Ok(())
    }

    #[tokio::test]
    async fn test_measure_time() -> TestResult {
        let (result, duration) = measure_time(async {
            tokio::time::sleep(Duration::from_millis(100)).await;
            Ok::<i32, Box<dyn std::error::Error + Send + Sync>>(42)
        })
        .await?;

        assert_eq!(result, 42);
        assert!(duration >= Duration::from_millis(90)); // Allow some variance
        Ok(())
    }

    #[tokio::test]
    async fn test_wait_for_condition() -> TestResult {
        use std::sync::{Arc, Mutex};

        let counter = Arc::new(Mutex::new(0));
        let counter_clone = counter.clone();

        // Spawn a task that increments the counter after a delay
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            *counter_clone.lock().unwrap() = 5;
        });

        wait_for_condition(
            || *counter.lock().unwrap() == 5,
            Duration::from_millis(200),
            Duration::from_millis(10),
        )
        .await?;

        Ok(())
    }

    #[test]
    fn test_generate_test_id() {
        let id1 = generate_test_id();
        let id2 = generate_test_id();

        assert!(id1.starts_with("test_"));
        assert!(id2.starts_with("test_"));
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_generate_large_text() {
        let text = generate_large_text(1); // 1KB
        assert!(text.len() >= 1024);
        assert!(text.len() <= 1024 + 200); // Allow some variance due to rounding
    }

    #[test]
    fn test_generate_test_vectors() {
        let vectors = generate_test_vectors(5, 10);
        assert_eq!(vectors.len(), 5);
        assert!(vectors.iter().all(|v| v.len() == 10));
    }

    #[test]
    fn test_assert_approx_eq() -> TestResult {
        assert_approx_eq(1.0, 1.001, 0.01)?;

        let result = assert_approx_eq(1.0, 1.1, 0.01);
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_assert_vectors_approx_eq() -> TestResult {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.001, 2.001, 3.001];

        assert_vectors_approx_eq(&a, &b, 0.01)?;

        let c = vec![1.0, 2.0, 3.1];
        let result = assert_vectors_approx_eq(&a, &c, 0.01);
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_performance_benchmark() {
        let benchmark = PerformanceBenchmark::start("test_operation");
        std::thread::sleep(Duration::from_millis(10));
        let duration = benchmark.end();

        assert!(duration >= Duration::from_millis(5));
    }

    #[test]
    fn test_memory_tracker_returns_rss() {
        let tracker = MemoryTracker::start("rss_test");
        // On macOS and Linux, get_memory_usage should return Some with non-zero RSS
        let diff = tracker.end();
        if cfg!(any(target_os = "macos", target_os = "linux")) {
            assert!(
                diff.is_some(),
                "MemoryTracker should return RSS on macOS/Linux"
            );
        }
    }

    #[test]
    fn test_memory_tracker_reports_allocation() {
        let tracker = MemoryTracker::start("alloc_test");
        // Allocate a non-trivial amount of memory to detect a change
        let _data: Vec<u8> = vec![0u8; 4 * 1024 * 1024]; // 4MB
        let diff = tracker.end();
        if cfg!(any(target_os = "macos", target_os = "linux")) {
            let diff_val = diff.expect("should return a value");
            // The diff should be positive since we allocated 4MB
            assert!(
                diff_val > 0,
                "Expected positive memory diff after 4MB alloc, got {}",
                diff_val
            );
        }
    }
}
