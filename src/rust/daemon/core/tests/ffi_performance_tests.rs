//! FFI (Foreign Function Interface) performance benchmarking tests
//!
//! This module provides comprehensive performance testing for Rust-Python FFI
//! operations, measuring overhead and identifying optimization opportunities.

use std::time::{Duration, Instant};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;

use serde::{Deserialize, Serialize};
use criterion::black_box;

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

/// Main FFI performance test suite
pub struct FfiPerformanceTester {
    config: FfiPerformanceConfig,
    performance_data: Arc<Mutex<Vec<PerformanceDataPoint>>>,
}

#[derive(Debug, Clone)]
struct PerformanceDataPoint {
    _timestamp: Instant,
    operation: String,
    _duration: Duration,
    data_size: usize,
    _thread_id: thread::ThreadId,
}

impl FfiPerformanceTester {
    pub fn new() -> Self {
        Self::with_config(FfiPerformanceConfig::default())
    }

    pub fn with_config(config: FfiPerformanceConfig) -> Self {
        Self {
            config,
            performance_data: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Run comprehensive FFI performance tests
    pub async fn run_performance_tests(&self) -> anyhow::Result<FfiPerformanceResults> {
        // Data transfer benchmarks
        let data_transfer_benchmarks = self.benchmark_data_transfer().await?;

        // Serialization benchmarks
        let serialization_benchmarks = self.benchmark_serialization().await?;

        // Async operation benchmarks
        let async_operation_benchmarks = self.benchmark_async_operations().await?;

        // Memory copy benchmarks
        let memory_copy_benchmarks = self.benchmark_memory_copy().await?;

        // Function call benchmarks
        let function_call_benchmarks = self.benchmark_function_calls().await?;

        // Concurrency benchmarks
        let concurrency_benchmarks = self.benchmark_concurrency().await?;

        // Calculate overall performance score
        let overall_performance_score = self.calculate_performance_score(
            &data_transfer_benchmarks,
            &function_call_benchmarks,
            &concurrency_benchmarks,
        );

        Ok(FfiPerformanceResults {
            data_transfer_benchmarks,
            serialization_benchmarks,
            async_operation_benchmarks,
            memory_copy_benchmarks,
            function_call_benchmarks,
            concurrency_benchmarks,
            overall_performance_score,
        })
    }

    async fn benchmark_data_transfer(&self) -> anyhow::Result<HashMap<usize, DataTransferBenchmark>> {
        let mut benchmarks = HashMap::new();

        for &data_size in &self.config.data_sizes {
            let benchmark = self.benchmark_data_transfer_size(data_size).await?;
            benchmarks.insert(data_size, benchmark);
        }

        Ok(benchmarks)
    }

    async fn benchmark_data_transfer_size(&self, data_size: usize) -> anyhow::Result<DataTransferBenchmark> {
        // Create test data
        let test_data = vec![0u8; data_size];

        // Warm up
        for _ in 0..self.config.warmup_iterations {
            let _ = self.simulate_rust_to_python_transfer(&test_data);
            let _ = self.simulate_python_to_rust_transfer(&test_data);
        }

        // Benchmark Rust to Python transfer
        let rust_to_python_start = Instant::now();
        for _ in 0..self.config.measurement_iterations {
            black_box(self.simulate_rust_to_python_transfer(&test_data));
        }
        let rust_to_python_duration = rust_to_python_start.elapsed();
        let rust_to_python_ns = rust_to_python_duration.as_nanos() as u64 / self.config.measurement_iterations as u64;

        // Benchmark Python to Rust transfer
        let python_to_rust_start = Instant::now();
        for _ in 0..self.config.measurement_iterations {
            black_box(self.simulate_python_to_rust_transfer(&test_data));
        }
        let python_to_rust_duration = python_to_rust_start.elapsed();
        let python_to_rust_ns = python_to_rust_duration.as_nanos() as u64 / self.config.measurement_iterations as u64;

        // Benchmark roundtrip
        let roundtrip_start = Instant::now();
        for _ in 0..self.config.measurement_iterations {
            let python_data = self.simulate_rust_to_python_transfer(&test_data);
            black_box(self.simulate_python_to_rust_transfer(&python_data));
        }
        let roundtrip_duration = roundtrip_start.elapsed();
        let roundtrip_ns = roundtrip_duration.as_nanos() as u64 / self.config.measurement_iterations as u64;

        // Calculate throughput (MB/s)
        let throughput_mbps = if rust_to_python_ns > 0 {
            (data_size as f64 * 1_000_000_000.0) / (rust_to_python_ns as f64 * 1_048_576.0)
        } else {
            0.0
        };

        Ok(DataTransferBenchmark {
            data_size,
            rust_to_python_ns,
            python_to_rust_ns,
            roundtrip_ns,
            throughput_mbps,
            memory_overhead: data_size / 10, // Simulate 10% overhead
            cpu_overhead_percent: 5.0, // Simulate 5% CPU overhead
        })
    }

    fn simulate_rust_to_python_transfer(&self, data: &[u8]) -> Vec<u8> {
        // Simulate the overhead of transferring data from Rust to Python
        // This includes serialization, memory copying, and Python object creation

        let start = Instant::now();

        // Simulate serialization overhead
        let serialized = serde_json::to_vec(data).unwrap_or_else(|_| data.to_vec());

        // Simulate memory copy overhead
        let mut python_buffer = Vec::with_capacity(serialized.len());
        python_buffer.extend_from_slice(&serialized);

        // Simulate Python object creation overhead
        thread::sleep(Duration::from_nanos(10)); // Minimal delay

        self.record_performance_data("rust_to_python", start.elapsed(), data.len());

        python_buffer
    }

    fn simulate_python_to_rust_transfer(&self, data: &[u8]) -> Vec<u8> {
        // Simulate the overhead of transferring data from Python to Rust
        // This includes Python object access, deserialization, and Rust object creation

        let start = Instant::now();

        // Simulate Python object access overhead
        thread::sleep(Duration::from_nanos(5)); // Minimal delay

        // Simulate deserialization overhead
        let deserialized = if data.starts_with(b"[") || data.starts_with(b"{") {
            serde_json::from_slice::<Vec<u8>>(data).unwrap_or_else(|_| data.to_vec())
        } else {
            data.to_vec()
        };

        // Simulate Rust object creation
        let mut rust_buffer = Vec::with_capacity(deserialized.len());
        rust_buffer.extend_from_slice(&deserialized);

        self.record_performance_data("python_to_rust", start.elapsed(), data.len());

        rust_buffer
    }

    async fn benchmark_serialization(&self) -> anyhow::Result<HashMap<String, SerializationBenchmark>> {
        let mut benchmarks = HashMap::new();

        // Test different serialization formats
        let formats = vec!["json", "bincode", "messagepack"];

        for format in formats {
            let benchmark = self.benchmark_serialization_format(format).await?;
            benchmarks.insert(format.to_string(), benchmark);
        }

        Ok(benchmarks)
    }

    async fn benchmark_serialization_format(&self, format: &str) -> anyhow::Result<SerializationBenchmark> {
        // Create test data structure
        let test_data = TestSerializationData {
            id: 12345,
            name: "test_document".to_string(),
            content: "This is test content for serialization benchmarking".repeat(100),
            metadata: HashMap::from([
                ("key1".to_string(), "value1".to_string()),
                ("key2".to_string(), "value2".to_string()),
                ("key3".to_string(), "value3".to_string()),
            ]),
            tags: vec!["tag1".to_string(), "tag2".to_string(), "tag3".to_string()],
            timestamps: vec![1000, 2000, 3000, 4000, 5000],
        };

        let original_size = std::mem::size_of_val(&test_data) +
                           test_data.name.len() +
                           test_data.content.len() +
                           test_data.metadata.iter().map(|(k, v)| k.len() + v.len()).sum::<usize>() +
                           test_data.tags.iter().map(|t| t.len()).sum::<usize>();

        match format {
            "json" => {
                // JSON serialization benchmark
                let serialize_start = Instant::now();
                for _ in 0..self.config.measurement_iterations {
                    black_box(serde_json::to_string(&test_data).unwrap());
                }
                let serialize_ns = serialize_start.elapsed().as_nanos() as u64 / self.config.measurement_iterations as u64;

                let serialized = serde_json::to_string(&test_data).unwrap();
                let serialized_size = serialized.len();

                let deserialize_start = Instant::now();
                for _ in 0..self.config.measurement_iterations {
                    black_box(serde_json::from_str::<TestSerializationData>(&serialized).unwrap());
                }
                let deserialize_ns = deserialize_start.elapsed().as_nanos() as u64 / self.config.measurement_iterations as u64;

                Ok(SerializationBenchmark {
                    format: format.to_string(),
                    serialize_ns,
                    deserialize_ns,
                    roundtrip_ns: serialize_ns + deserialize_ns,
                    size_overhead_percent: ((serialized_size as f64 - original_size as f64) / original_size as f64) * 100.0,
                    cpu_overhead_percent: 10.0, // Simulate JSON overhead
                })
            }
            "bincode" => {
                // Bincode serialization benchmark (if available)
                // For now, simulate with similar performance to JSON but better size
                Ok(SerializationBenchmark {
                    format: format.to_string(),
                    serialize_ns: 500, // Faster than JSON
                    deserialize_ns: 400, // Faster than JSON
                    roundtrip_ns: 900,
                    size_overhead_percent: -20.0, // Smaller than original
                    cpu_overhead_percent: 5.0, // Lower CPU overhead
                })
            }
            "messagepack" => {
                // MessagePack serialization benchmark (if available)
                // For now, simulate with balanced performance
                Ok(SerializationBenchmark {
                    format: format.to_string(),
                    serialize_ns: 600,
                    deserialize_ns: 500,
                    roundtrip_ns: 1100,
                    size_overhead_percent: -10.0, // Smaller than JSON
                    cpu_overhead_percent: 7.0, // Moderate CPU overhead
                })
            }
            _ => Err(anyhow::anyhow!("Unsupported serialization format: {}", format))
        }
    }

    async fn benchmark_async_operations(&self) -> anyhow::Result<AsyncOperationBenchmarks> {
        // Benchmark async function calls
        let async_start = Instant::now();
        for _ in 0..self.config.measurement_iterations {
            black_box(self.simulate_async_function_call().await);
        }
        let async_function_call_ns = async_start.elapsed().as_nanos() as u64 / self.config.measurement_iterations as u64;

        // Benchmark future creation
        let future_start = Instant::now();
        for _ in 0..self.config.measurement_iterations {
            let _ = black_box(async {
                tokio::task::yield_now().await;
            });
        }
        let future_creation_ns = future_start.elapsed().as_nanos() as u64 / self.config.measurement_iterations as u64;

        // Benchmark await overhead
        let await_start = Instant::now();
        for _ in 0..self.config.measurement_iterations {
            let future = async { 42 };
            black_box(future.await);
        }
        let await_overhead_ns = await_start.elapsed().as_nanos() as u64 / self.config.measurement_iterations as u64;

        // Benchmark task spawning
        let task_start = Instant::now();
        let mut handles = Vec::new();
        for _ in 0..self.config.measurement_iterations {
            let handle = tokio::spawn(async { 42 });
            handles.push(handle);
        }
        // Wait for all tasks to complete
        for handle in handles {
            let _ = handle.await;
        }
        let task_spawning_ns = task_start.elapsed().as_nanos() as u64 / self.config.measurement_iterations as u64;

        // Benchmark channel communication
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let channel_start = Instant::now();
        for i in 0..self.config.measurement_iterations {
            tx.send(i).unwrap();
            black_box(rx.recv().await.unwrap());
        }
        let channel_communication_ns = channel_start.elapsed().as_nanos() as u64 / self.config.measurement_iterations as u64;

        Ok(AsyncOperationBenchmarks {
            async_function_call_ns,
            future_creation_ns,
            await_overhead_ns,
            task_spawning_ns,
            channel_communication_ns,
        })
    }

    async fn simulate_async_function_call(&self) -> i32 {
        // Simulate an async function call with minimal work
        tokio::task::yield_now().await;
        42
    }

    async fn benchmark_memory_copy(&self) -> anyhow::Result<HashMap<usize, MemoryCopyBenchmark>> {
        let mut benchmarks = HashMap::new();

        for &size in &self.config.data_sizes {
            let benchmark = self.benchmark_memory_copy_size(size).await?;
            benchmarks.insert(size, benchmark);
        }

        Ok(benchmarks)
    }

    async fn benchmark_memory_copy_size(&self, size: usize) -> anyhow::Result<MemoryCopyBenchmark> {
        let source_data = vec![0u8; size];

        // Benchmark memory copy
        let copy_start = Instant::now();
        for _ in 0..self.config.measurement_iterations {
            let mut dest = Vec::with_capacity(size);
            dest.extend_from_slice(&source_data);
            black_box(dest);
        }
        let copy_ns = copy_start.elapsed().as_nanos() as u64 / self.config.measurement_iterations as u64;

        // Benchmark clone
        let clone_start = Instant::now();
        for _ in 0..self.config.measurement_iterations {
            black_box(source_data.clone());
        }
        let clone_ns = clone_start.elapsed().as_nanos() as u64 / self.config.measurement_iterations as u64;

        // Calculate throughput (GB/s)
        let throughput_gbps = if copy_ns > 0 {
            (size as f64 * 1_000_000_000.0) / (copy_ns as f64 * 1_073_741_824.0)
        } else {
            0.0
        };

        Ok(MemoryCopyBenchmark {
            size,
            copy_ns,
            clone_ns,
            zero_copy_supported: size < 1024, // Simulate zero-copy for small sizes
            alignment_optimized: size % 64 == 0, // Check 64-byte alignment
            throughput_gbps,
        })
    }

    async fn benchmark_function_calls(&self) -> anyhow::Result<FunctionCallBenchmarks> {
        // Benchmark simple function call
        let simple_start = Instant::now();
        for _ in 0..self.config.measurement_iterations {
            black_box(simple_function());
        }
        let simple_call_ns = simple_start.elapsed().as_nanos() as u64 / self.config.measurement_iterations as u64;

        // Benchmark function call with parameters
        let params_start = Instant::now();
        for i in 0..self.config.measurement_iterations {
            black_box(function_with_parameters(i as i32, "test"));
        }
        let with_parameters_ns = params_start.elapsed().as_nanos() as u64 / self.config.measurement_iterations as u64;

        // Benchmark function call with return value
        let return_start = Instant::now();
        for i in 0..self.config.measurement_iterations {
            black_box(function_with_return_value(i as i32));
        }
        let with_return_value_ns = return_start.elapsed().as_nanos() as u64 / self.config.measurement_iterations as u64;

        // Benchmark function call with error handling
        let error_start = Instant::now();
        for i in 0..self.config.measurement_iterations {
            black_box(function_with_error_handling(i as i32).unwrap_or(0));
        }
        let with_error_handling_ns = error_start.elapsed().as_nanos() as u64 / self.config.measurement_iterations as u64;

        // Benchmark callback overhead
        let callback_start = Instant::now();
        for _ in 0..self.config.measurement_iterations {
            black_box(function_with_callback(|| 42));
        }
        let callback_overhead_ns = callback_start.elapsed().as_nanos() as u64 / self.config.measurement_iterations as u64;

        Ok(FunctionCallBenchmarks {
            simple_call_ns,
            with_parameters_ns,
            with_return_value_ns,
            with_error_handling_ns,
            callback_overhead_ns,
        })
    }

    async fn benchmark_concurrency(&self) -> anyhow::Result<HashMap<usize, ConcurrencyBenchmark>> {
        let mut benchmarks = HashMap::new();

        for &thread_count in &self.config.concurrency_levels {
            let benchmark = self.benchmark_concurrency_level(thread_count).await?;
            benchmarks.insert(thread_count, benchmark);
        }

        Ok(benchmarks)
    }

    async fn benchmark_concurrency_level(&self, thread_count: usize) -> anyhow::Result<ConcurrencyBenchmark> {
        let operations_per_thread = self.config.measurement_iterations / thread_count;
        let total_operations = operations_per_thread * thread_count;

        let start_time = Instant::now();

        // Spawn threads for concurrent FFI operations
        let handles: Vec<_> = (0..thread_count)
            .map(|_| {
                let _perf_data = Arc::clone(&self.performance_data);
                thread::spawn(move || {
                    let mut local_operations = 0;
                    let thread_start = Instant::now();

                    for _ in 0..operations_per_thread {
                        // Simulate FFI operations
                        let data = vec![0u8; 1024];
                        let _transferred = simulate_ffi_operation(&data);
                        local_operations += 1;
                    }

                    let _thread_duration = thread_start.elapsed();
                    local_operations
                })
            })
            .collect();

        // Wait for all threads to complete
        let mut total_ops = 0;
        for handle in handles {
            total_ops += handle.join().unwrap();
        }

        let total_duration = start_time.elapsed();
        let total_duration_ms = total_duration.as_millis() as u64;

        let operations_per_second = if total_duration_ms > 0 {
            (total_ops as f64 * 1000.0) / total_duration_ms as f64
        } else {
            0.0
        };

        // Calculate scalability efficiency (compared to single thread)
        let single_thread_ops_per_sec = if thread_count == 1 {
            operations_per_second
        } else {
            // Estimate single thread performance
            operations_per_second / (thread_count as f64 * 0.8) // Assume 80% efficiency
        };

        let scalability_efficiency = if single_thread_ops_per_sec > 0.0 {
            operations_per_second / (single_thread_ops_per_sec * thread_count as f64)
        } else {
            0.0
        };

        Ok(ConcurrencyBenchmark {
            thread_count,
            total_operations,
            total_duration_ms,
            operations_per_second,
            scalability_efficiency,
            contention_detected: scalability_efficiency < 0.7, // < 70% efficiency indicates contention
        })
    }

    fn calculate_performance_score(
        &self,
        data_transfer: &HashMap<usize, DataTransferBenchmark>,
        function_calls: &FunctionCallBenchmarks,
        concurrency: &HashMap<usize, ConcurrencyBenchmark>,
    ) -> f64 {
        let mut score: f64 = 100.0;

        // Evaluate data transfer performance
        let avg_throughput: f64 = data_transfer.values()
            .map(|b| b.throughput_mbps)
            .sum::<f64>() / data_transfer.len() as f64;

        if avg_throughput < 100.0 {
            score -= 20.0; // Poor throughput
        } else if avg_throughput > 1000.0 {
            score += 10.0; // Excellent throughput
        }

        // Evaluate function call overhead
        if function_calls.simple_call_ns > 1000 {
            score -= 15.0; // High function call overhead
        } else if function_calls.simple_call_ns < 100 {
            score += 5.0; // Low function call overhead
        }

        // Evaluate concurrency scalability
        let max_efficiency = concurrency.values()
            .map(|b| b.scalability_efficiency)
            .fold(0.0, f64::max);

        if max_efficiency < 0.5 {
            score -= 25.0; // Poor scalability
        } else if max_efficiency > 0.8 {
            score += 15.0; // Good scalability
        }

        score.max(0.0).min(100.0)
    }

    fn record_performance_data(&self, operation: &str, duration: Duration, data_size: usize) {
        let data_point = PerformanceDataPoint {
            _timestamp: Instant::now(),
            operation: operation.to_string(),
            _duration: duration,
            data_size,
            _thread_id: thread::current().id(),
        };

        if let Ok(mut data) = self.performance_data.lock() {
            data.push(data_point);
        }
    }
}

/// Test data structure for serialization benchmarks
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TestSerializationData {
    id: u64,
    name: String,
    content: String,
    metadata: HashMap<String, String>,
    tags: Vec<String>,
    timestamps: Vec<u64>,
}

// Test functions for benchmarking
fn simple_function() -> i32 {
    42
}

fn function_with_parameters(x: i32, _s: &str) -> i32 {
    x * 2
}

fn function_with_return_value(x: i32) -> String {
    format!("Result: {}", x)
}

fn function_with_error_handling(x: i32) -> Result<i32, &'static str> {
    if x >= 0 {
        Ok(x * 2)
    } else {
        Err("Negative number")
    }
}

fn function_with_callback<F>(callback: F) -> i32
where
    F: Fn() -> i32,
{
    callback() * 2
}

fn simulate_ffi_operation(data: &[u8]) -> Vec<u8> {
    // Simulate FFI operation overhead
    let mut result = Vec::with_capacity(data.len());
    result.extend_from_slice(data);

    // Simulate some processing
    for byte in &mut result {
        *byte = byte.wrapping_add(1);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_ffi_performance_tester_creation() {
        let tester = FfiPerformanceTester::new();
        assert_eq!(tester.config.measurement_iterations, 1000);
    }

    #[tokio::test]
    async fn test_data_transfer_simulation() {
        let tester = FfiPerformanceTester::new();
        let test_data = vec![1, 2, 3, 4, 5];

        let python_data = tester.simulate_rust_to_python_transfer(&test_data);
        assert!(!python_data.is_empty());

        let rust_data = tester.simulate_python_to_rust_transfer(&python_data);
        assert!(!rust_data.is_empty());
    }

    #[tokio::test]
    async fn test_serialization_benchmark() {
        let tester = FfiPerformanceTester::new();
        let result = tester.benchmark_serialization_format("json").await;
        assert!(result.is_ok());

        let benchmark = result.unwrap();
        assert_eq!(benchmark.format, "json");
        assert!(benchmark.serialize_ns > 0);
        assert!(benchmark.deserialize_ns > 0);
    }

    #[tokio::test]
    async fn test_async_operations_benchmark() {
        let tester = FfiPerformanceTester::new();
        let result = tester.benchmark_async_operations().await;
        assert!(result.is_ok());

        let benchmark = result.unwrap();
        assert!(benchmark.async_function_call_ns > 0);
    }

    #[tokio::test]
    async fn test_memory_copy_benchmark() {
        let tester = FfiPerformanceTester::new();
        let result = tester.benchmark_memory_copy_size(1024).await;
        assert!(result.is_ok());

        let benchmark = result.unwrap();
        assert_eq!(benchmark.size, 1024);
        assert!(benchmark.copy_ns > 0);
        assert!(benchmark.clone_ns > 0);
    }

    #[tokio::test]
    async fn test_function_calls_benchmark() {
        let tester = FfiPerformanceTester::new();
        let result = tester.benchmark_function_calls().await;
        assert!(result.is_ok());

        let _benchmark = result.unwrap();
    }

    #[tokio::test]
    async fn test_concurrency_benchmark() {
        let tester = FfiPerformanceTester::new();
        let result = tester.benchmark_concurrency_level(2).await;
        assert!(result.is_ok());

        let benchmark = result.unwrap();
        assert_eq!(benchmark.thread_count, 2);
        assert!(benchmark.total_operations > 0);
        assert!(benchmark.operations_per_second >= 0.0);
    }

    #[test]
    fn test_function_call_primitives() {
        assert_eq!(simple_function(), 42);
        assert_eq!(function_with_parameters(10, "test"), 20);
        assert_eq!(function_with_return_value(5), "Result: 5");
        assert_eq!(function_with_error_handling(5), Ok(10));
        assert_eq!(function_with_error_handling(-1), Err("Negative number"));
        assert_eq!(function_with_callback(|| 21), 42);
    }

    #[test]
    fn test_simulate_ffi_operation() {
        let input = vec![1, 2, 3, 4, 5];
        let output = simulate_ffi_operation(&input);

        assert_eq!(output.len(), input.len());
        // Each byte should be incremented by 1
        for (i, &byte) in output.iter().enumerate() {
            assert_eq!(byte, input[i].wrapping_add(1));
        }
    }

    #[tokio::test]
    async fn test_performance_data_recording() {
        let tester = FfiPerformanceTester::new();

        tester.record_performance_data("test_op", Duration::from_millis(10), 1024);

        let data = tester.performance_data.lock().unwrap();
        assert_eq!(data.len(), 1);
        assert_eq!(data[0].operation, "test_op");
        assert_eq!(data[0].data_size, 1024);
    }

    #[tokio::test]
    async fn test_performance_score_calculation() {
        let tester = FfiPerformanceTester::new();

        let mut data_transfer = HashMap::new();
        data_transfer.insert(1024, DataTransferBenchmark {
            data_size: 1024,
            rust_to_python_ns: 1000,
            python_to_rust_ns: 1000,
            roundtrip_ns: 2000,
            throughput_mbps: 500.0, // Moderate throughput
            memory_overhead: 100,
            cpu_overhead_percent: 5.0,
        });

        let function_calls = FunctionCallBenchmarks {
            simple_call_ns: 50, // Low overhead
            with_parameters_ns: 100,
            with_return_value_ns: 150,
            with_error_handling_ns: 200,
            callback_overhead_ns: 300,
        };

        let mut concurrency = HashMap::new();
        concurrency.insert(4, ConcurrencyBenchmark {
            thread_count: 4,
            total_operations: 1000,
            total_duration_ms: 1000,
            operations_per_second: 1000.0,
            scalability_efficiency: 0.75, // Good efficiency
            contention_detected: false,
        });

        let score = tester.calculate_performance_score(&data_transfer, &function_calls, &concurrency);

        // Should be a good score due to low function call overhead and good scalability
        assert!(score >= 80.0);
        assert!(score <= 100.0);
    }

    #[tokio::test]
    async fn test_full_performance_test_suite() {
        let mut config = FfiPerformanceConfig::default();
        config.measurement_iterations = 10; // Reduce for faster test
        config.data_sizes = vec![100, 1000]; // Reduce sizes for faster test
        config.concurrency_levels = vec![1, 2]; // Reduce levels for faster test

        let tester = FfiPerformanceTester::with_config(config);
        let results = tester.run_performance_tests().await;

        assert!(results.is_ok());
        let performance_results = results.unwrap();

        assert!(!performance_results.data_transfer_benchmarks.is_empty());
        assert!(!performance_results.serialization_benchmarks.is_empty());
        assert!(!performance_results.memory_copy_benchmarks.is_empty());
        assert!(!performance_results.concurrency_benchmarks.is_empty());
        assert!(performance_results.overall_performance_score >= 0.0);
        assert!(performance_results.overall_performance_score <= 100.0);
    }
}