//! Public type definitions for FFI performance benchmarking.

use std::collections::HashMap;
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// FFI performance test configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FfiPerformanceConfig {
    pub test_duration: Duration,
    pub warmup_iterations: usize,
    pub measurement_iterations: usize,
    pub data_sizes: Vec<usize>,
    pub concurrency_levels: Vec<usize>,
    pub enable_memory_profiling: bool,
    pub enable_cpu_profiling: bool,
}

impl Default for FfiPerformanceConfig {
    fn default() -> Self {
        Self {
            test_duration: Duration::from_secs(10),
            warmup_iterations: 100,
            measurement_iterations: 1000,
            data_sizes: vec![1, 10, 100, 1024, 10240, 102400, 1048576], // 1B to 1MB
            concurrency_levels: vec![1, 2, 4, 8, 16],
            enable_memory_profiling: true,
            enable_cpu_profiling: true,
        }
    }
}

/// FFI performance test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FfiPerformanceResults {
    pub data_transfer_benchmarks: HashMap<usize, DataTransferBenchmark>,
    pub serialization_benchmarks: HashMap<String, SerializationBenchmark>,
    pub async_operation_benchmarks: AsyncOperationBenchmarks,
    pub memory_copy_benchmarks: HashMap<usize, MemoryCopyBenchmark>,
    pub function_call_benchmarks: FunctionCallBenchmarks,
    pub concurrency_benchmarks: HashMap<usize, ConcurrencyBenchmark>,
    pub overall_performance_score: f64,
}

/// Data transfer performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataTransferBenchmark {
    pub data_size: usize,
    pub rust_to_python_ns: u64,
    pub python_to_rust_ns: u64,
    pub roundtrip_ns: u64,
    pub throughput_mbps: f64,
    pub memory_overhead: usize,
    pub cpu_overhead_percent: f64,
}

/// Serialization performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializationBenchmark {
    pub format: String,
    pub serialize_ns: u64,
    pub deserialize_ns: u64,
    pub roundtrip_ns: u64,
    pub size_overhead_percent: f64,
    pub cpu_overhead_percent: f64,
}

/// Async operation performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsyncOperationBenchmarks {
    pub async_function_call_ns: u64,
    pub future_creation_ns: u64,
    pub await_overhead_ns: u64,
    pub task_spawning_ns: u64,
    pub channel_communication_ns: u64,
}

/// Memory copy performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryCopyBenchmark {
    pub size: usize,
    pub copy_ns: u64,
    pub clone_ns: u64,
    pub zero_copy_supported: bool,
    pub alignment_optimized: bool,
    pub throughput_gbps: f64,
}

/// Function call overhead metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCallBenchmarks {
    pub simple_call_ns: u64,
    pub with_parameters_ns: u64,
    pub with_return_value_ns: u64,
    pub with_error_handling_ns: u64,
    pub callback_overhead_ns: u64,
}

/// Concurrency performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcurrencyBenchmark {
    pub thread_count: usize,
    pub total_operations: usize,
    pub total_duration_ms: u64,
    pub operations_per_second: f64,
    pub scalability_efficiency: f64,
    pub contention_detected: bool,
}
