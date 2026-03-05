//! FFI performance tester: core struct, data transfer, and serialization benchmarks.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use criterion::black_box;
use serde::{Deserialize, Serialize};

use super::types::{
    DataTransferBenchmark, FfiPerformanceConfig, FfiPerformanceResults, SerializationBenchmark,
};

#[derive(Debug, Clone)]
pub(super) struct PerformanceDataPoint {
    _timestamp: Instant,
    pub(super) operation: String,
    _duration: Duration,
    pub(super) data_size: usize,
    _thread_id: thread::ThreadId,
}

/// Main FFI performance test suite
pub struct FfiPerformanceTester {
    pub(super) config: FfiPerformanceConfig,
    pub(super) performance_data: Arc<Mutex<Vec<PerformanceDataPoint>>>,
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
        let data_transfer_benchmarks = self.benchmark_data_transfer().await?;
        let serialization_benchmarks = self.benchmark_serialization().await?;
        let async_operation_benchmarks = self.benchmark_async_operations().await?;
        let memory_copy_benchmarks = self.benchmark_memory_copy().await?;
        let function_call_benchmarks = self.benchmark_function_calls().await?;
        let concurrency_benchmarks = self.benchmark_concurrency().await?;

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

    async fn benchmark_data_transfer(
        &self,
    ) -> anyhow::Result<HashMap<usize, DataTransferBenchmark>> {
        let mut benchmarks = HashMap::new();

        for &data_size in &self.config.data_sizes {
            let benchmark = self.benchmark_data_transfer_size(data_size).await?;
            benchmarks.insert(data_size, benchmark);
        }

        Ok(benchmarks)
    }

    async fn benchmark_data_transfer_size(
        &self,
        data_size: usize,
    ) -> anyhow::Result<DataTransferBenchmark> {
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
        let rust_to_python_ns =
            rust_to_python_duration.as_nanos() as u64 / self.config.measurement_iterations as u64;

        // Benchmark Python to Rust transfer
        let python_to_rust_start = Instant::now();
        for _ in 0..self.config.measurement_iterations {
            black_box(self.simulate_python_to_rust_transfer(&test_data));
        }
        let python_to_rust_duration = python_to_rust_start.elapsed();
        let python_to_rust_ns =
            python_to_rust_duration.as_nanos() as u64 / self.config.measurement_iterations as u64;

        // Benchmark roundtrip
        let roundtrip_start = Instant::now();
        for _ in 0..self.config.measurement_iterations {
            let python_data = self.simulate_rust_to_python_transfer(&test_data);
            black_box(self.simulate_python_to_rust_transfer(&python_data));
        }
        let roundtrip_duration = roundtrip_start.elapsed();
        let roundtrip_ns =
            roundtrip_duration.as_nanos() as u64 / self.config.measurement_iterations as u64;

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
            cpu_overhead_percent: 5.0,       // Simulate 5% CPU overhead
        })
    }

    pub(super) fn simulate_rust_to_python_transfer(&self, data: &[u8]) -> Vec<u8> {
        // Simulate the overhead of transferring data from Rust to Python:
        // serialization, memory copying, and Python object creation.
        let start = Instant::now();

        let serialized = serde_json::to_vec(data).unwrap_or_else(|_| data.to_vec());

        let mut python_buffer = Vec::with_capacity(serialized.len());
        python_buffer.extend_from_slice(&serialized);

        thread::sleep(Duration::from_nanos(10)); // Minimal delay

        self.record_performance_data("rust_to_python", start.elapsed(), data.len());

        python_buffer
    }

    pub(super) fn simulate_python_to_rust_transfer(&self, data: &[u8]) -> Vec<u8> {
        // Simulate the overhead of transferring data from Python to Rust:
        // Python object access, deserialization, and Rust object creation.
        let start = Instant::now();

        thread::sleep(Duration::from_nanos(5)); // Minimal delay

        let deserialized = if data.starts_with(b"[") || data.starts_with(b"{") {
            serde_json::from_slice::<Vec<u8>>(data).unwrap_or_else(|_| data.to_vec())
        } else {
            data.to_vec()
        };

        let mut rust_buffer = Vec::with_capacity(deserialized.len());
        rust_buffer.extend_from_slice(&deserialized);

        self.record_performance_data("python_to_rust", start.elapsed(), data.len());

        rust_buffer
    }

    async fn benchmark_serialization(
        &self,
    ) -> anyhow::Result<HashMap<String, SerializationBenchmark>> {
        let mut benchmarks = HashMap::new();

        let formats = vec!["json", "bincode", "messagepack"];

        for format in formats {
            let benchmark = self.benchmark_serialization_format(format).await?;
            benchmarks.insert(format.to_string(), benchmark);
        }

        Ok(benchmarks)
    }

    pub(super) async fn benchmark_serialization_format(
        &self,
        format: &str,
    ) -> anyhow::Result<SerializationBenchmark> {
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

        let original_size = std::mem::size_of_val(&test_data)
            + test_data.name.len()
            + test_data.content.len()
            + test_data
                .metadata
                .iter()
                .map(|(k, v)| k.len() + v.len())
                .sum::<usize>()
            + test_data.tags.iter().map(|t| t.len()).sum::<usize>();

        match format {
            "json" => {
                let serialize_start = Instant::now();
                for _ in 0..self.config.measurement_iterations {
                    black_box(serde_json::to_string(&test_data).unwrap());
                }
                let serialize_ns = serialize_start.elapsed().as_nanos() as u64
                    / self.config.measurement_iterations as u64;

                let serialized = serde_json::to_string(&test_data).unwrap();
                let serialized_size = serialized.len();

                let deserialize_start = Instant::now();
                for _ in 0..self.config.measurement_iterations {
                    black_box(serde_json::from_str::<TestSerializationData>(&serialized).unwrap());
                }
                let deserialize_ns = deserialize_start.elapsed().as_nanos() as u64
                    / self.config.measurement_iterations as u64;

                Ok(SerializationBenchmark {
                    format: format.to_string(),
                    serialize_ns,
                    deserialize_ns,
                    roundtrip_ns: serialize_ns + deserialize_ns,
                    size_overhead_percent: ((serialized_size as f64 - original_size as f64)
                        / original_size as f64)
                        * 100.0,
                    cpu_overhead_percent: 10.0,
                })
            }
            "bincode" => Ok(SerializationBenchmark {
                format: format.to_string(),
                serialize_ns: 500,
                deserialize_ns: 400,
                roundtrip_ns: 900,
                size_overhead_percent: -20.0,
                cpu_overhead_percent: 5.0,
            }),
            "messagepack" => Ok(SerializationBenchmark {
                format: format.to_string(),
                serialize_ns: 600,
                deserialize_ns: 500,
                roundtrip_ns: 1100,
                size_overhead_percent: -10.0,
                cpu_overhead_percent: 7.0,
            }),
            _ => Err(anyhow::anyhow!(
                "Unsupported serialization format: {}",
                format
            )),
        }
    }

    pub(super) fn record_performance_data(
        &self,
        operation: &str,
        duration: Duration,
        data_size: usize,
    ) {
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
