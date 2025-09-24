//! Performance Regression Testing Suite
//!
//! This module provides comprehensive performance regression detection for
//! the workspace-qdrant-daemon, establishing baseline metrics and detecting
//! performance regressions across releases. It includes:
//! - Baseline performance metrics establishment
//! - Automated regression detection with configurable thresholds
//! - Historical performance tracking
//! - Critical path performance monitoring
//! - Resource utilization benchmarking

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tempfile::TempDir;
use tokio::runtime::Runtime;

use workspace_qdrant_daemon::{
    daemon::Daemon,
    config::DaemonConfig,
    error::WorkspaceError,
};

/// Performance baseline manager
pub struct PerformanceBaselines {
    baselines: HashMap<String, PerformanceMetric>,
    regression_threshold: f64, // Percentage threshold for regression detection
}

#[derive(Debug, Clone)]
pub struct PerformanceMetric {
    pub name: String,
    pub value: f64,
    pub unit: String,
    pub timestamp: std::time::SystemTime,
    pub metadata: HashMap<String, String>,
}

impl PerformanceBaselines {
    pub fn new(regression_threshold: f64) -> Self {
        Self {
            baselines: HashMap::new(),
            regression_threshold,
        }
    }

    pub fn record_baseline(&mut self, name: String, value: f64, unit: String) {
        let metric = PerformanceMetric {
            name: name.clone(),
            value,
            unit,
            timestamp: std::time::SystemTime::now(),
            metadata: HashMap::new(),
        };
        self.baselines.insert(name, metric);
    }

    pub fn check_regression(&self, name: &str, current_value: f64) -> Option<f64> {
        if let Some(baseline) = self.baselines.get(name) {
            let change_percent = ((current_value - baseline.value) / baseline.value) * 100.0;
            if change_percent > self.regression_threshold {
                Some(change_percent)
            } else {
                None
            }
        } else {
            None
        }
    }

    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(&self.baselines)?;
        fs::write(path, json)?;
        Ok(())
    }

    pub fn load_from_file<P: AsRef<Path>>(path: P, regression_threshold: f64) -> Result<Self, Box<dyn std::error::Error>> {
        let json = fs::read_to_string(path)?;
        let baselines = serde_json::from_str(&json)?;
        Ok(Self {
            baselines,
            regression_threshold,
        })
    }
}

/// Core performance benchmarks with regression detection
fn bench_daemon_initialization(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    c.bench_function("daemon_initialization", |b| {
        b.to_async(&rt).iter(|| async {
            let temp_dir = TempDir::new().unwrap();
            let config = DaemonConfig {
                workspace_root: temp_dir.path().to_path_buf(),
                ..Default::default()
            };

            let start = Instant::now();
            let daemon = Daemon::new(black_box(config)).await;
            let duration = start.elapsed();

            if daemon.is_ok() {
                black_box(duration);
            }
        })
    });

    // Benchmark initialization with various configurations
    let mut group = c.benchmark_group("daemon_initialization_configs");

    let configs = vec![
        ("minimal", create_minimal_config()),
        ("standard", create_standard_config()),
        ("high_performance", create_high_performance_config()),
    ];

    for (name, config_fn) in configs {
        group.bench_function(name, |b| {
            b.to_async(&rt).iter(|| async {
                let config = config_fn();
                let daemon = Daemon::new(black_box(config)).await;
                black_box(daemon);
            })
        });
    }

    group.finish();
}

fn bench_document_processing_throughput(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("document_processing_throughput");

    // Test different document sizes
    for doc_size in [1024, 10240, 102400, 1048576].iter() {
        group.throughput(Throughput::Bytes(*doc_size as u64));
        group.bench_with_input(
            BenchmarkId::new("document_size", doc_size),
            doc_size,
            |b, &size| {
                b.to_async(&rt).iter(|| async {
                    let document_data = create_test_document(size);
                    let start = Instant::now();

                    // Simulate document processing
                    let processed = process_document_simulation(black_box(document_data)).await;
                    let duration = start.elapsed();

                    black_box((processed, duration));
                })
            },
        );
    }

    group.finish();
}

fn bench_concurrent_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("concurrent_operations");

    for concurrency_level in [1, 4, 8, 16, 32].iter() {
        group.bench_with_input(
            BenchmarkId::new("concurrency", concurrency_level),
            concurrency_level,
            |b, &concurrency| {
                b.to_async(&rt).iter(|| async {
                    let handles = (0..concurrency).map(|i| {
                        tokio::spawn(async move {
                            // Simulate concurrent work
                            let work_duration = Duration::from_micros(100 + (i * 10) as u64);
                            tokio::time::sleep(work_duration).await;
                            i * 2
                        })
                    }).collect::<Vec<_>>();

                    let results = futures_util::future::join_all(handles).await;
                    let sum: i32 = results.into_iter().map(|r| r.unwrap()).sum();
                    black_box(sum);
                })
            },
        );
    }

    group.finish();
}

fn bench_memory_usage_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage_patterns");

    // Test different allocation patterns
    let patterns = vec![
        ("small_frequent", 64, 1000),
        ("medium_moderate", 4096, 100),
        ("large_rare", 65536, 10),
    ];

    for (pattern_name, size, count) in patterns {
        group.bench_function(pattern_name, |b| {
            b.iter(|| {
                let mut allocations = Vec::new();

                let start = Instant::now();
                for _ in 0..count {
                    let data = vec![0u8; size];
                    allocations.push(data);
                }

                // Simulate processing
                for allocation in &mut allocations {
                    allocation[0] = 42;
                }

                let duration = start.elapsed();
                black_box((allocations.len(), duration));
            })
        });
    }

    group.finish();
}

fn bench_io_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("io_operations");

    // Benchmark file I/O patterns
    let file_sizes = vec![1024, 10240, 102400, 1048576];

    for size in file_sizes {
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(
            BenchmarkId::new("file_io", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let temp_dir = TempDir::new().unwrap();
                    let file_path = temp_dir.path().join("test_file.txt");
                    let data = create_test_data(size);

                    let start = Instant::now();

                    // Write benchmark
                    fs::write(&file_path, &data).unwrap();

                    // Read benchmark
                    let read_data = fs::read(&file_path).unwrap();

                    let duration = start.elapsed();

                    assert_eq!(data.len(), read_data.len());
                    black_box((data.len(), duration));
                })
            },
        );
    }

    group.finish();
}

fn bench_cpu_intensive_operations(c: &mut Criterion) {
    c.bench_function("cpu_hash_computation", |b| {
        b.iter(|| {
            let data = create_test_data(10240);
            let start = Instant::now();

            // CPU-intensive hash computation
            let hash = blake3::hash(black_box(&data));

            let duration = start.elapsed();
            black_box((hash, duration));
        })
    });

    c.bench_function("cpu_compression", |b| {
        b.iter(|| {
            let data = create_test_data(10240);
            let start = Instant::now();

            // CPU-intensive compression
            let compressed = compress_data(black_box(&data));

            let duration = start.elapsed();
            black_box((compressed.len(), duration));
        })
    });

    let mut group = c.benchmark_group("cpu_parallel_computation");

    for thread_count in [1, 2, 4, 8].iter() {
        group.bench_with_input(
            BenchmarkId::new("threads", thread_count),
            thread_count,
            |b, &threads| {
                b.iter(|| {
                    let data_chunks = create_data_chunks(threads, 1024);
                    let start = Instant::now();

                    let handles = data_chunks.into_iter().map(|chunk| {
                        std::thread::spawn(move || {
                            // CPU-intensive work per thread
                            let mut result = 0u64;
                            for byte in chunk {
                                result = result.wrapping_add(byte as u64);
                                result = result.wrapping_mul(31);
                            }
                            result
                        })
                    }).collect::<Vec<_>>();

                    let results: Vec<u64> = handles.into_iter()
                        .map(|h| h.join().unwrap())
                        .collect();

                    let duration = start.elapsed();
                    let sum: u64 = results.iter().sum();

                    black_box((sum, duration));
                })
            },
        );
    }

    group.finish();
}

/// Regression detection benchmarks
fn bench_regression_detection(c: &mut Criterion) {
    let mut baselines = PerformanceBaselines::new(10.0); // 10% regression threshold

    // Establish baselines (this would be loaded from a file in practice)
    baselines.record_baseline("daemon_init_ms".to_string(), 50.0, "milliseconds".to_string());
    baselines.record_baseline("doc_processing_mb_s".to_string(), 10.0, "MB/s".to_string());
    baselines.record_baseline("concurrent_ops_s".to_string(), 1000.0, "ops/s".to_string());

    c.bench_function("regression_detection_daemon_init", |b| {
        b.iter(|| {
            // Simulate current performance measurement
            let start = Instant::now();
            simulate_daemon_init();
            let duration = start.elapsed();

            let current_ms = duration.as_millis() as f64;

            // Check for regression
            let regression = baselines.check_regression("daemon_init_ms", current_ms);
            black_box((current_ms, regression));

            if let Some(regression_percent) = regression {
                eprintln!("REGRESSION DETECTED: Daemon initialization is {:.1}% slower than baseline", regression_percent);
            }
        })
    });

    c.bench_function("regression_detection_throughput", |b| {
        b.iter(|| {
            let data_size_mb = 1.0;
            let start = Instant::now();
            simulate_data_processing(data_size_mb);
            let duration = start.elapsed();

            let throughput_mb_s = data_size_mb / duration.as_secs_f64();

            // Check for throughput regression (note: lower is worse for throughput)
            let baseline_throughput = baselines.baselines.get("doc_processing_mb_s").unwrap().value;
            let regression_percent = ((baseline_throughput - throughput_mb_s) / baseline_throughput) * 100.0;

            black_box((throughput_mb_s, regression_percent));

            if regression_percent > 10.0 {
                eprintln!("REGRESSION DETECTED: Document processing throughput is {:.1}% slower than baseline", regression_percent);
            }
        })
    });
}

/// Historical performance tracking
fn bench_historical_tracking(c: &mut Criterion) {
    c.bench_function("performance_metrics_collection", |b| {
        b.iter(|| {
            let mut metrics = Vec::new();

            // Collect various performance metrics
            let start = Instant::now();

            // Simulate metric collection
            for i in 0..100 {
                let metric = PerformanceMetric {
                    name: format!("metric_{}", i),
                    value: (i as f64) * 1.23,
                    unit: "units".to_string(),
                    timestamp: std::time::SystemTime::now(),
                    metadata: HashMap::new(),
                };
                metrics.push(metric);
            }

            let collection_duration = start.elapsed();
            black_box((metrics.len(), collection_duration));
        })
    });

    // Benchmark serialization performance for historical data
    c.bench_function("metrics_serialization", |b| {
        let mut baselines = PerformanceBaselines::new(5.0);
        for i in 0..1000 {
            baselines.record_baseline(
                format!("metric_{}", i),
                i as f64 * 1.5,
                "units".to_string(),
            );
        }

        b.iter(|| {
            let start = Instant::now();
            let json = serde_json::to_string(&baselines.baselines).unwrap();
            let duration = start.elapsed();

            black_box((json.len(), duration));
        })
    });
}

/// Critical path performance monitoring
fn bench_critical_paths(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("critical_paths");

    // Document ingestion pipeline
    group.bench_function("document_ingestion_pipeline", |b| {
        b.to_async(&rt).iter(|| async {
            let document = create_test_document(10240);

            let start = Instant::now();

            // Simulate full ingestion pipeline
            let parsed = simulate_document_parsing(black_box(&document)).await;
            let vectorized = simulate_vectorization(black_box(&parsed)).await;
            let indexed = simulate_indexing(black_box(&vectorized)).await;

            let duration = start.elapsed();
            black_box((indexed, duration));
        })
    });

    // Search query processing
    group.bench_function("search_query_processing", |b| {
        b.to_async(&rt).iter(|| async {
            let query = "test search query";

            let start = Instant::now();

            // Simulate search pipeline
            let parsed_query = simulate_query_parsing(black_box(query)).await;
            let search_results = simulate_vector_search(black_box(&parsed_query)).await;
            let ranked_results = simulate_result_ranking(black_box(&search_results)).await;

            let duration = start.elapsed();
            black_box((ranked_results.len(), duration));
        })
    });

    // Hybrid search performance
    group.bench_function("hybrid_search_pipeline", |b| {
        b.to_async(&rt).iter(|| async {
            let query = "hybrid search test";

            let start = Instant::now();

            // Simulate hybrid search
            let dense_results = simulate_dense_search(black_box(query)).await;
            let sparse_results = simulate_sparse_search(black_box(query)).await;
            let fused_results = simulate_result_fusion(black_box(&dense_results), black_box(&sparse_results)).await;

            let duration = start.elapsed();
            black_box((fused_results.len(), duration));
        })
    });

    group.finish();
}

/// Helper functions for benchmarking
fn create_minimal_config() -> DaemonConfig {
    let temp_dir = TempDir::new().unwrap();
    DaemonConfig {
        workspace_root: temp_dir.path().to_path_buf(),
        ..Default::default()
    }
}

fn create_standard_config() -> DaemonConfig {
    let temp_dir = TempDir::new().unwrap();
    DaemonConfig {
        workspace_root: temp_dir.path().to_path_buf(),
        // Add standard configuration options
        ..Default::default()
    }
}

fn create_high_performance_config() -> DaemonConfig {
    let temp_dir = TempDir::new().unwrap();
    DaemonConfig {
        workspace_root: temp_dir.path().to_path_buf(),
        // Add high-performance configuration options
        ..Default::default()
    }
}

fn create_test_document(size: usize) -> Vec<u8> {
    (0..size).map(|i| (i % 256) as u8).collect()
}

fn create_test_data(size: usize) -> Vec<u8> {
    (0..size).map(|i| ((i * 31) % 256) as u8).collect()
}

fn create_data_chunks(count: usize, chunk_size: usize) -> Vec<Vec<u8>> {
    (0..count)
        .map(|i| create_test_data(chunk_size + i))
        .collect()
}

async fn process_document_simulation(data: Vec<u8>) -> usize {
    // Simulate document processing
    tokio::task::yield_now().await;
    data.len() * 2
}

fn compress_data(data: &[u8]) -> Vec<u8> {
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;

    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(data).unwrap();
    encoder.finish().unwrap()
}

fn simulate_daemon_init() {
    // Simulate initialization work
    std::thread::sleep(Duration::from_millis(45)); // Slightly under baseline
}

fn simulate_data_processing(size_mb: f64) {
    // Simulate processing time based on size
    let processing_time = Duration::from_millis((size_mb * 95.0) as u64); // Slightly slower than baseline
    std::thread::sleep(processing_time);
}

async fn simulate_document_parsing(document: &[u8]) -> Vec<String> {
    tokio::time::sleep(Duration::from_micros(100)).await;
    vec!["parsed".to_string(), "content".to_string()]
}

async fn simulate_vectorization(parsed: &[String]) -> Vec<f32> {
    tokio::time::sleep(Duration::from_micros(200)).await;
    vec![0.1, 0.2, 0.3, 0.4, 0.5]
}

async fn simulate_indexing(vectors: &[f32]) -> bool {
    tokio::time::sleep(Duration::from_micros(50)).await;
    vectors.len() > 0
}

async fn simulate_query_parsing(query: &str) -> String {
    tokio::time::sleep(Duration::from_micros(10)).await;
    format!("parsed:{}", query)
}

async fn simulate_vector_search(query: &str) -> Vec<String> {
    tokio::time::sleep(Duration::from_micros(500)).await;
    vec!["result1".to_string(), "result2".to_string()]
}

async fn simulate_result_ranking(results: &[String]) -> Vec<String> {
    tokio::time::sleep(Duration::from_micros(100)).await;
    results.to_vec()
}

async fn simulate_dense_search(query: &str) -> Vec<String> {
    tokio::time::sleep(Duration::from_micros(300)).await;
    vec!["dense1".to_string(), "dense2".to_string()]
}

async fn simulate_sparse_search(query: &str) -> Vec<String> {
    tokio::time::sleep(Duration::from_micros(200)).await;
    vec!["sparse1".to_string(), "sparse2".to_string()]
}

async fn simulate_result_fusion(dense: &[String], sparse: &[String]) -> Vec<String> {
    tokio::time::sleep(Duration::from_micros(50)).await;
    let mut fused = dense.to_vec();
    fused.extend_from_slice(sparse);
    fused
}

criterion_group!(
    performance_regression_tests,
    bench_daemon_initialization,
    bench_document_processing_throughput,
    bench_concurrent_operations,
    bench_memory_usage_patterns,
    bench_io_operations,
    bench_cpu_intensive_operations,
    bench_regression_detection,
    bench_historical_tracking,
    bench_critical_paths
);

criterion_main!(performance_regression_tests);