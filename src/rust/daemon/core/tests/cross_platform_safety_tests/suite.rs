//! Core test suite struct and public async methods
//!
//! Contains `CrossPlatformTestSuite` construction, and the top-level
//! orchestrating methods that delegate to the private implementations
//! in sibling modules.

use std::sync::{Arc, Mutex};

use tempfile::TempDir;
use tracing::info;

use super::types::{
    CrossPlatformResults, CrossPlatformTestConfig, EnvironmentTestResults, FFIPerformanceResults,
    FileSystemTestResults, MemorySafetyResults, MemoryTracker, NetworkTestResults,
    PerformanceRegressionResults, PlatformSpecificResults, ThreadSafetyResults,
};

/// Comprehensive test suite for cross-platform testing and memory safety
pub struct CrossPlatformTestSuite {
    pub(crate) config: CrossPlatformTestConfig,
    pub(crate) test_dir: TempDir,
    pub(crate) memory_tracker: Arc<Mutex<MemoryTracker>>,
}

impl CrossPlatformTestSuite {
    pub fn new() -> std::io::Result<Self> {
        let config = CrossPlatformTestConfig::default();
        let test_dir = TempDir::new()?;
        let memory_tracker = Arc::new(Mutex::new(MemoryTracker::new()));

        Ok(Self {
            config,
            test_dir,
            memory_tracker,
        })
    }

    pub fn with_config(config: CrossPlatformTestConfig) -> std::io::Result<Self> {
        let test_dir = TempDir::new()?;
        let memory_tracker = Arc::new(Mutex::new(MemoryTracker::new()));

        Ok(Self {
            config,
            test_dir,
            memory_tracker,
        })
    }

    /// Run comprehensive cross-platform tests
    pub async fn run_cross_platform_tests(&self) -> anyhow::Result<Vec<CrossPlatformResults>> {
        let mut results = Vec::new();

        for platform in &self.config.test_platforms {
            info!("Running cross-platform tests for: {}", platform);

            let platform_result = self.test_platform_behavior(platform).await?;
            results.push(platform_result);
        }

        Ok(results)
    }

    pub(crate) async fn test_platform_behavior(
        &self,
        platform: &str,
    ) -> anyhow::Result<CrossPlatformResults> {
        let file_system_tests = self.test_file_system_behavior().await?;
        let network_tests = self.test_network_behavior().await?;
        let environment_tests = self.test_environment_behavior().await?;
        let platform_specific_tests = self.test_platform_specific_behavior(platform).await?;

        Ok(CrossPlatformResults {
            platform: platform.to_string(),
            file_system_tests,
            network_tests,
            environment_tests,
            platform_specific_tests,
        })
    }

    pub(crate) async fn test_file_system_behavior(&self) -> anyhow::Result<FileSystemTestResults> {
        let test_path = self.test_dir.path();

        let path_separator_handling = self.test_path_separators(test_path).await?;
        let case_sensitivity = self.test_case_sensitivity(test_path).await?;
        let symbolic_link_support = self.test_symbolic_links(test_path).await?;
        let long_path_support = self.test_long_paths(test_path).await?;
        let unicode_support = self.test_unicode_paths(test_path).await?;
        let file_watching_accuracy = self.test_file_watching_accuracy(test_path).await?;

        Ok(FileSystemTestResults {
            path_separator_handling,
            case_sensitivity,
            symbolic_link_support,
            long_path_support,
            unicode_support,
            file_watching_accuracy,
        })
    }

    pub(crate) async fn test_network_behavior(&self) -> anyhow::Result<NetworkTestResults> {
        let tcp_socket_behavior = self.test_tcp_sockets().await?;
        let udp_socket_behavior = self.test_udp_sockets().await?;
        let ipv6_support = self.test_ipv6_support().await?;
        let dns_resolution = self.test_dns_resolution().await?;
        let tls_compatibility = self.test_tls_compatibility().await?;

        Ok(NetworkTestResults {
            tcp_socket_behavior,
            udp_socket_behavior,
            ipv6_support,
            dns_resolution,
            tls_compatibility,
        })
    }

    pub(crate) async fn test_environment_behavior(&self) -> anyhow::Result<EnvironmentTestResults> {
        let environment_variable_handling = self.test_environment_variables().await?;
        let path_resolution = self.test_path_resolution().await?;
        let working_directory = self.test_working_directory().await?;
        let home_directory_detection = self.test_home_directory().await?;

        Ok(EnvironmentTestResults {
            environment_variable_handling,
            path_resolution,
            working_directory,
            home_directory_detection,
        })
    }

    pub(crate) async fn test_platform_specific_behavior(
        &self,
        platform: &str,
    ) -> anyhow::Result<PlatformSpecificResults> {
        let native_file_watching = self.test_native_file_watching(platform).await?;
        let process_spawning = self.test_process_spawning(platform).await?;
        let signal_handling = self.test_signal_handling(platform).await?;
        let threading_model = self.test_threading_model(platform).await?;

        Ok(PlatformSpecificResults {
            native_file_watching,
            process_spawning,
            signal_handling,
            threading_model,
        })
    }

    /// Run memory safety validation tests
    pub async fn run_memory_safety_tests(&self) -> anyhow::Result<MemorySafetyResults> {
        info!("Running memory safety validation tests");

        let mut tracker = self.memory_tracker.lock().unwrap();
        tracker.start_tracking();
        drop(tracker);

        let allocations_tracked = self.test_allocation_patterns().await?;
        let unsafe_block_validation = self.validate_unsafe_blocks().await?;
        let memory_leaks_detected = self.detect_memory_leaks().await?;
        let peak_memory_usage = self.measure_peak_memory_usage().await?;
        let memory_fragmentation = self.calculate_memory_fragmentation().await?;

        let mut tracker = self.memory_tracker.lock().unwrap();
        tracker.stop_tracking();

        Ok(MemorySafetyResults {
            allocations_tracked,
            memory_leaks_detected,
            unsafe_block_validation,
            peak_memory_usage,
            memory_fragmentation,
        })
    }

    /// Benchmark Rust-Python FFI performance
    pub async fn benchmark_ffi_performance(&self) -> anyhow::Result<FFIPerformanceResults> {
        info!("Benchmarking Rust-Python FFI performance");

        let data_transfer_overhead = self.benchmark_data_transfer().await?;
        let serialization_overhead = self.benchmark_serialization().await?;
        let async_operation_overhead = self.benchmark_async_operations().await?;
        let memory_copy_overhead = self.benchmark_memory_copy().await?;
        let function_call_overhead = self.benchmark_function_calls().await?;

        Ok(FFIPerformanceResults {
            data_transfer_overhead,
            serialization_overhead,
            async_operation_overhead,
            memory_copy_overhead,
            function_call_overhead,
        })
    }

    /// Test thread safety
    pub async fn test_thread_safety(&self) -> anyhow::Result<ThreadSafetyResults> {
        info!("Testing thread safety");

        let concurrent_access_tests = self.test_concurrent_access().await?;
        let data_race_detection = self.test_data_race_detection().await?;
        let deadlock_detection = self.test_deadlock_detection().await?;
        let async_safety = self.test_async_safety().await?;
        let shared_state_integrity = self.test_shared_state_integrity().await?;

        Ok(ThreadSafetyResults {
            concurrent_access_tests,
            data_race_detection,
            deadlock_detection,
            async_safety,
            shared_state_integrity,
        })
    }

    /// Run performance regression tests
    pub async fn run_performance_regression_tests(
        &self,
    ) -> anyhow::Result<PerformanceRegressionResults> {
        info!("Running performance regression tests");

        let baseline = self.establish_performance_baseline().await?;
        let current = self.measure_current_performance().await?;
        let regressions = self
            .detect_performance_regressions(&baseline, &current)
            .await?;

        Ok(PerformanceRegressionResults {
            baseline,
            current,
            regressions,
        })
    }
}
