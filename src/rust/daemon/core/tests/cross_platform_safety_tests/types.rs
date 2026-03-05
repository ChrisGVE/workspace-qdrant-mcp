//! Type definitions for cross-platform testing and memory safety validation
//!
//! Contains all public and internal types used across the cross-platform
//! test suite: configs, result structs, and helper types.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

// Mock implementations for testing

#[derive(Debug, Clone)]
pub struct PlatformWatcherConfig {
    pub enable_optimizations: bool,
}

impl Default for PlatformWatcherConfig {
    fn default() -> Self {
        Self {
            enable_optimizations: true,
        }
    }
}

pub struct PlatformWatcher;

impl PlatformWatcher {
    pub async fn watch(&mut self, _path: &std::path::Path) -> anyhow::Result<()> {
        Ok(())
    }
}

pub struct PlatformWatcherFactory;

impl PlatformWatcherFactory {
    pub fn create_watcher(_config: PlatformWatcherConfig) -> anyhow::Result<PlatformWatcher> {
        Ok(PlatformWatcher)
    }
}

pub struct DocumentStorage;
pub struct DocumentProcessor;

/// Cross-platform test configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossPlatformTestConfig {
    pub test_platforms: Vec<String>,
    pub memory_test_iterations: usize,
    pub thread_safety_threads: usize,
    pub performance_test_duration: Duration,
    pub leak_detection_enabled: bool,
    pub valgrind_enabled: bool,
}

impl Default for CrossPlatformTestConfig {
    fn default() -> Self {
        Self {
            test_platforms: vec![
                "x86_64-apple-darwin".to_string(),
                "aarch64-apple-darwin".to_string(),
                "x86_64-unknown-linux-gnu".to_string(),
                "aarch64-unknown-linux-gnu".to_string(),
                "x86_64-pc-windows-msvc".to_string(),
                "aarch64-pc-windows-msvc".to_string(),
            ],
            memory_test_iterations: 1000,
            thread_safety_threads: 16,
            performance_test_duration: Duration::from_secs(30),
            leak_detection_enabled: true,
            valgrind_enabled: cfg!(target_os = "linux"),
        }
    }
}

/// Memory safety test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySafetyResults {
    pub allocations_tracked: usize,
    pub memory_leaks_detected: usize,
    pub unsafe_block_validation: HashMap<String, UnsafeBlockResult>,
    pub peak_memory_usage: usize,
    pub memory_fragmentation: f64,
}

/// Unsafe block validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnsafeBlockResult {
    pub location: String,
    pub safety_validated: bool,
    pub memory_access_patterns: Vec<String>,
    pub potential_issues: Vec<String>,
}

/// Cross-platform behavior test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossPlatformResults {
    pub platform: String,
    pub file_system_tests: FileSystemTestResults,
    pub network_tests: NetworkTestResults,
    pub environment_tests: EnvironmentTestResults,
    pub platform_specific_tests: PlatformSpecificResults,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSystemTestResults {
    pub path_separator_handling: bool,
    pub case_sensitivity: bool,
    pub symbolic_link_support: bool,
    pub long_path_support: bool,
    pub unicode_support: bool,
    pub file_watching_accuracy: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkTestResults {
    pub tcp_socket_behavior: bool,
    pub udp_socket_behavior: bool,
    pub ipv6_support: bool,
    pub dns_resolution: bool,
    pub tls_compatibility: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentTestResults {
    pub environment_variable_handling: bool,
    pub path_resolution: bool,
    pub working_directory: bool,
    pub home_directory_detection: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformSpecificResults {
    pub native_file_watching: bool,
    pub process_spawning: bool,
    pub signal_handling: bool,
    pub threading_model: bool,
}

/// FFI performance test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FFIPerformanceResults {
    pub data_transfer_overhead: Duration,
    pub serialization_overhead: Duration,
    pub async_operation_overhead: Duration,
    pub memory_copy_overhead: Duration,
    pub function_call_overhead: Duration,
}

/// Thread safety test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadSafetyResults {
    pub concurrent_access_tests: bool,
    pub data_race_detection: bool,
    pub deadlock_detection: bool,
    pub async_safety: bool,
    pub shared_state_integrity: bool,
}

/// Performance regression test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRegressionResults {
    pub baseline: PerformanceMetrics,
    pub current: PerformanceMetrics,
    pub regressions: Vec<PerformanceRegression>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub throughput_ops_per_sec: f64,
    pub latency_p50_ms: f64,
    pub latency_p95_ms: f64,
    pub latency_p99_ms: f64,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRegression {
    pub metric: String,
    pub baseline_value: f64,
    pub current_value: f64,
    pub regression_percentage: f64,
    pub severity: RegressionSeverity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegressionSeverity {
    Minor,    // 0-10% regression
    Moderate, // 10-25% regression
    Major,    // 25-50% regression
    Critical, // >50% regression
}

/// Memory tracking utility for leak detection
#[derive(Debug)]
pub(crate) struct MemoryTracker {
    pub(crate) allocations: HashMap<usize, AllocationInfo>,
    pub(crate) total_allocated: usize,
    pub(crate) peak_usage: usize,
    pub(crate) tracking_enabled: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct AllocationInfo {
    pub(crate) size: usize,
    pub(crate) _timestamp: Instant,
    pub(crate) stack_trace: String,
}

impl MemoryTracker {
    pub(crate) fn new() -> Self {
        Self {
            allocations: HashMap::new(),
            total_allocated: 0,
            peak_usage: 0,
            tracking_enabled: false,
        }
    }

    pub(crate) fn start_tracking(&mut self) {
        self.tracking_enabled = true;
        self.allocations.clear();
        self.total_allocated = 0;
        self.peak_usage = 0;
    }

    pub(crate) fn stop_tracking(&mut self) {
        self.tracking_enabled = false;
    }

    pub(crate) fn track_allocation(&mut self, ptr: usize, size: usize) {
        if !self.tracking_enabled {
            return;
        }

        let info = AllocationInfo {
            size,
            _timestamp: Instant::now(),
            stack_trace: format!("Stack trace for allocation at {:p}", ptr as *const u8),
        };

        self.allocations.insert(ptr, info);
        self.total_allocated += size;

        if self.total_allocated > self.peak_usage {
            self.peak_usage = self.total_allocated;
        }
    }

    pub(crate) fn track_deallocation(&mut self, ptr: usize) {
        if !self.tracking_enabled {
            return;
        }

        if let Some(info) = self.allocations.remove(&ptr) {
            self.total_allocated -= info.size;
        }
    }

    pub(crate) fn get_leaks(&self) -> Vec<(usize, AllocationInfo)> {
        self.allocations
            .iter()
            .map(|(&ptr, info)| (ptr, info.clone()))
            .collect()
    }
}

/// Helper struct for serialization benchmarks
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct TestData {
    pub(crate) id: u64,
    pub(crate) name: String,
    pub(crate) values: Vec<i32>,
}

pub(crate) fn dummy_function(x: i32) -> i32 {
    x * 2 + 1
}

pub(crate) fn classify_regression_severity(percentage: f64) -> RegressionSeverity {
    match percentage {
        p if p < 10.0 => RegressionSeverity::Minor,
        p if p < 25.0 => RegressionSeverity::Moderate,
        p if p < 50.0 => RegressionSeverity::Major,
        _ => RegressionSeverity::Critical,
    }
}
