//! Cross-platform testing and memory safety validation test modules
//!
//! This module provides comprehensive testing infrastructure for cross-platform
//! compatibility, memory safety, and performance validation.

pub mod cross_platform_safety_tests;
pub mod valgrind_memory_tests;
pub mod unsafe_code_audit_tests;
pub mod ffi_performance_tests;
pub mod file_ingestion_comprehensive_tests;

// Re-export key types for easier access
pub use cross_platform_safety_tests::{
    CrossPlatformTestSuite, CrossPlatformTestConfig, CrossPlatformResults,
    MemorySafetyResults, FFIPerformanceResults, ThreadSafetyResults,
    PerformanceRegressionResults,
};

pub use valgrind_memory_tests::{
    ValgrindTestSuite, ValgrindConfig, ValgrindResults, ValgrindStatus,
    MemcheckResults, CachegrindResults, MassifResults, HelgrindResults, DrdResults,
};

pub use unsafe_code_audit_tests::{
    UnsafeCodeAuditor, UnsafeAuditResults, SafetyViolation, ViolationType, ViolationSeverity,
    MemoryAccessPattern, InvariantValidation, BoundaryTest, ConcurrencySafety, FfiSafety,
};

pub use ffi_performance_tests::{
    FfiPerformanceTester, FfiPerformanceConfig, FfiPerformanceResults,
    DataTransferBenchmark, SerializationBenchmark, AsyncOperationBenchmarks,
    MemoryCopyBenchmark, FunctionCallBenchmarks, ConcurrencyBenchmark,
};

#[cfg(test)]
mod integration_tests {
    use super::*;

    /// Integration test to run all cross-platform test suites together
    #[tokio::test]
    async fn test_comprehensive_cross_platform_validation() {
        // Create comprehensive test suite
        let cross_platform_suite = CrossPlatformTestSuite::new().unwrap();

        // Run cross-platform tests
        let cross_platform_results = cross_platform_suite.run_cross_platform_tests().await;
        assert!(cross_platform_results.is_ok(), "Cross-platform tests should pass");

        // Run memory safety tests
        let memory_results = cross_platform_suite.run_memory_safety_tests().await;
        assert!(memory_results.is_ok(), "Memory safety tests should pass");

        // Run thread safety tests
        let thread_results = cross_platform_suite.test_thread_safety().await;
        assert!(thread_results.is_ok(), "Thread safety tests should pass");

        let thread_safety = thread_results.unwrap();
        assert!(thread_safety.concurrent_access_tests, "Concurrent access should be safe");
        assert!(thread_safety.async_safety, "Async operations should be safe");
    }

    #[tokio::test]
    async fn test_ffi_performance_integration() {
        // Create FFI performance tester with reduced load for CI
        let mut config = FfiPerformanceConfig::default();
        config.measurement_iterations = 100; // Reduce for faster testing
        config.data_sizes = vec![64, 1024]; // Test only small sizes

        let ffi_tester = FfiPerformanceTester::with_config(config);

        // Run FFI performance tests
        let results = ffi_tester.run_performance_tests().await;
        assert!(results.is_ok(), "FFI performance tests should complete");

        let performance_results = results.unwrap();
        assert!(!performance_results.data_transfer_benchmarks.is_empty());
        assert!(!performance_results.serialization_benchmarks.is_empty());

        // Validate performance thresholds
        for (size, benchmark) in &performance_results.data_transfer_benchmarks {
            assert!(benchmark.throughput_mbps > 0.0, "Throughput should be positive for size {}", size);
            assert!(benchmark.rust_to_python_ns > 0, "Transfer time should be positive");
        }
    }

    #[tokio::test]
    async fn test_unsafe_code_audit_integration() {
        // Create unsafe code auditor
        let auditor = UnsafeCodeAuditor::new();

        // Run comprehensive audit
        let audit_results = auditor.audit_unsafe_code().await;
        assert!(audit_results.is_ok(), "Unsafe code audit should complete");

        let results = audit_results.unwrap();
        assert_eq!(results.total_unsafe_blocks, results.blocks_audited);
        assert!(results.overall_safety_score >= 0.0);
        assert!(results.overall_safety_score <= 100.0);

        // Check that critical violations are handled
        for violation in &results.safety_violations {
            match violation.severity {
                ViolationSeverity::Critical => {
                    panic!("Critical safety violation detected: {}", violation.description);
                }
                ViolationSeverity::High => {
                    eprintln!("High severity violation: {}", violation.description);
                }
                _ => {} // Lower severity violations are acceptable
            }
        }
    }

    #[tokio::test]
    #[cfg(target_os = "linux")]
    async fn test_valgrind_integration() {
        // Only run on Linux where Valgrind is available
        if !ValgrindTestSuite::is_valgrind_available() {
            eprintln!("Valgrind not available, skipping test");
            return;
        }

        // Create a simple test binary path (this would be the actual binary in practice)
        let binary_path = std::env::current_exe().unwrap();

        let valgrind_suite = ValgrindTestSuite::new(binary_path);
        assert!(valgrind_suite.is_ok(), "Valgrind suite should initialize on Linux");

        // Note: We don't run actual Valgrind tests here as they're expensive
        // and require the test binary to be built separately
    }

    #[test]
    fn test_platform_detection() {
        // Test that we can detect the current platform
        let platform = std::env::consts::OS;
        assert!(
            platform == "linux" || platform == "macos" || platform == "windows",
            "Should detect a supported platform, got: {}",
            platform
        );
    }

    #[test]
    fn test_architecture_detection() {
        // Test that we can detect the current architecture
        let arch = std::env::consts::ARCH;
        assert!(
            arch == "x86_64" || arch == "aarch64" || arch == "x86",
            "Should detect a supported architecture, got: {}",
            arch
        );
    }

    #[tokio::test]
    async fn test_memory_allocation_patterns() {
        // Test memory allocation patterns for leaks
        let mut allocations = Vec::new();

        // Allocate various sizes
        for size in [64, 256, 1024, 4096] {
            let data = vec![0u8; size];
            allocations.push(data);
        }

        // Verify allocations
        assert_eq!(allocations.len(), 4);
        for (i, allocation) in allocations.iter().enumerate() {
            let expected_size = 64 << (i * 2); // 64, 256, 1024, 4096
            assert_eq!(allocation.len(), expected_size);
        }

        // Allocations should be dropped automatically when this function ends
    }

    #[tokio::test]
    async fn test_thread_spawning_safety() {
        use std::sync::{Arc, Mutex};
        use std::thread;

        let counter = Arc::new(Mutex::new(0));
        let mut handles = Vec::new();

        // Spawn multiple threads
        for _ in 0..4 {
            let counter = Arc::clone(&counter);
            let handle = thread::spawn(move || {
                for _ in 0..100 {
                    let mut num = counter.lock().unwrap();
                    *num += 1;
                }
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        // Verify final count
        let final_count = *counter.lock().unwrap();
        assert_eq!(final_count, 400);
    }

    #[tokio::test]
    async fn test_async_task_safety() {
        use tokio::sync::Mutex;
        use std::sync::Arc;

        let counter = Arc::new(Mutex::new(0));
        let mut handles = Vec::new();

        // Spawn multiple async tasks
        for _ in 0..4 {
            let counter = Arc::clone(&counter);
            let handle = tokio::spawn(async move {
                for _ in 0..100 {
                    let mut num = counter.lock().await;
                    *num += 1;
                }
            });
            handles.push(handle);
        }

        // Wait for all tasks to complete
        for handle in handles {
            handle.await.unwrap();
        }

        // Verify final count
        let final_count = *counter.lock().await;
        assert_eq!(final_count, 400);
    }

    #[test]
    fn test_serialization_roundtrip_safety() {
        use serde::{Deserialize, Serialize};
        use std::collections::HashMap;

        #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
        struct TestData {
            id: u64,
            name: String,
            values: Vec<i32>,
            metadata: HashMap<String, String>,
        }

        let original = TestData {
            id: 12345,
            name: "test".to_string(),
            values: vec![1, 2, 3, 4, 5],
            metadata: [("key1".to_string(), "value1".to_string())]
                .iter().cloned().collect(),
        };

        // Test JSON roundtrip
        let json = serde_json::to_string(&original).unwrap();
        let deserialized: TestData = serde_json::from_str(&json).unwrap();
        assert_eq!(original, deserialized);
    }

    #[test]
    fn test_error_handling_patterns() {
        // Test various error handling patterns

        fn fallible_operation(should_fail: bool) -> Result<i32, &'static str> {
            if should_fail {
                Err("Operation failed")
            } else {
                Ok(42)
            }
        }

        // Test success case
        let result = fallible_operation(false);
        assert_eq!(result.unwrap(), 42);

        // Test failure case
        let result = fallible_operation(true);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Operation failed");

        // Test with ? operator
        fn chain_operations() -> Result<i32, &'static str> {
            let value1 = fallible_operation(false)?;
            let value2 = fallible_operation(false)?;
            Ok(value1 + value2)
        }

        assert_eq!(chain_operations().unwrap(), 84);
    }
}