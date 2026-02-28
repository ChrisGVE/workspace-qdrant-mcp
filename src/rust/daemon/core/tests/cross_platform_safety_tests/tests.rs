//! Unit tests for the cross-platform safety test suite
//!
//! Tests for suite creation, memory tracking, file system behavior,
//! network behavior, thread safety, FFI performance, memory safety,
//! performance regression detection, and platform-specific behavior.

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use serial_test::serial;

    use crate::cross_platform_safety_tests::suite::CrossPlatformTestSuite;
    use crate::cross_platform_safety_tests::types::MemoryTracker;

    #[tokio::test]
    async fn test_cross_platform_suite_creation() {
        let suite = CrossPlatformTestSuite::new().unwrap();
        assert_eq!(suite.config.test_platforms.len(), 6);
    }

    #[tokio::test]
    async fn test_memory_tracker() {
        let mut tracker = MemoryTracker::new();
        tracker.start_tracking();

        tracker.track_allocation(0x1000, 1024);
        tracker.track_allocation(0x2000, 2048);
        tracker.track_deallocation(0x1000);

        let leaks = tracker.get_leaks();
        assert_eq!(leaks.len(), 1);
        assert_eq!(leaks[0].1.size, 2048);
    }

    #[tokio::test]
    #[serial]
    async fn test_file_system_behavior() {
        let suite = CrossPlatformTestSuite::new().unwrap();
        let results = suite.test_file_system_behavior().await.unwrap();

        // Basic assertions - these should pass on most platforms
        assert!(results.unicode_support);
        assert!(results.file_watching_accuracy > 0.0);
    }

    #[tokio::test]
    async fn test_network_behavior() {
        let suite = CrossPlatformTestSuite::new().unwrap();
        let results = suite.test_network_behavior().await.unwrap();

        assert!(results.tcp_socket_behavior);
        assert!(results.udp_socket_behavior);
    }

    #[tokio::test]
    async fn test_thread_safety() {
        let suite = CrossPlatformTestSuite::new().unwrap();
        let results = suite.test_thread_safety().await.unwrap();

        assert!(results.concurrent_access_tests);
        assert!(results.async_safety);
        assert!(results.shared_state_integrity);
    }

    #[tokio::test]
    async fn test_ffi_performance_benchmark() {
        let suite = CrossPlatformTestSuite::new().unwrap();
        let results = suite.benchmark_ffi_performance().await.unwrap();

        // Performance benchmarks should complete successfully
        assert!(results.data_transfer_overhead > Duration::ZERO);
        assert!(results.serialization_overhead > Duration::ZERO);
    }

    #[tokio::test]
    async fn test_memory_safety_validation() {
        let suite = CrossPlatformTestSuite::new().unwrap();
        let results = suite.run_memory_safety_tests().await.unwrap();

        assert!(results.allocations_tracked > 0);
        assert!(!results.unsafe_block_validation.is_empty());
    }

    #[tokio::test]
    async fn test_performance_regression_detection() {
        let suite = CrossPlatformTestSuite::new().unwrap();
        let results = suite.run_performance_regression_tests().await.unwrap();

        assert!(results.baseline.throughput_ops_per_sec > 0.0);
        assert!(results.current.throughput_ops_per_sec > 0.0);
    }

    #[tokio::test]
    async fn test_platform_specific_behavior() {
        let suite = CrossPlatformTestSuite::new().unwrap();
        let platform = std::env::consts::OS.to_string();
        let results = suite
            .test_platform_specific_behavior(&platform)
            .await
            .unwrap();

        assert!(results.process_spawning);
        assert!(results.threading_model);
    }
}
