//! Stress Testing Benchmarks for File Ingestion
//!
//! Comprehensive stress tests measuring daemon performance under extreme conditions:
//! - High volume (1000+ files)
//! - High rate (100+ files/second)
//! - Multiple watchers
//! - Large files
//! - Memory constraints
//! - Network interruptions
//!
//! Run with: cargo bench --bench stress_benchmarks

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tempfile::{tempdir, NamedTempFile};
use tokio::fs;
use tokio::io::AsyncWriteExt;
use tokio::runtime::Runtime;
use tokio::sync::Semaphore;

// ============================================================================
// Mock Components for Stress Testing
// ============================================================================

/// Mock document processor with configurable processing delay
struct MockDocumentProcessor {
    processing_delay_ms: u64,
    chunk_count: usize,
}

impl MockDocumentProcessor {
    fn new(chunk_count: usize, delay_ms: u64) -> Self {
        Self {
            chunk_count,
            processing_delay_ms: delay_ms,
        }
    }

    async fn process_file(&self, path: &Path) -> Result<ProcessingResult, String> {
        // Simulate processing time
        if self.processing_delay_ms > 0 {
            tokio::time::sleep(Duration::from_millis(self.processing_delay_ms)).await;
        }

        // Read actual file size
        let file_size = tokio::fs::metadata(path)
            .await
            .map(|m| m.len() as usize)
            .unwrap_or(0);

        Ok(ProcessingResult {
            chunks_created: self.chunk_count,
            processing_time_ms: self.processing_delay_ms,
            bytes_processed: file_size,
        })
    }
}

struct ProcessingResult {
    chunks_created: usize,
    processing_time_ms: u64,
    bytes_processed: usize,
}

/// Mock embedding generator with configurable latency
struct MockEmbeddingGenerator {
    vector_size: usize,
    generation_delay_ms: u64,
}

impl MockEmbeddingGenerator {
    fn new(vector_size: usize, delay_ms: u64) -> Self {
        Self {
            vector_size,
            generation_delay_ms: delay_ms,
        }
    }

    async fn generate_embedding(&self, text: &str) -> Vec<f32> {
        if self.generation_delay_ms > 0 {
            tokio::time::sleep(Duration::from_millis(self.generation_delay_ms)).await;
        }

        // Simple deterministic embedding
        let hash = text.len() as f32;
        (0..self.vector_size)
            .map(|i| ((hash + i as f32) * 0.1) % 1.0)
            .collect()
    }
}

/// Mock storage client with network simulation
struct MockStorageClient {
    network_delay_ms: u64,
    fail_rate: f64,
    points_stored: Arc<AtomicUsize>,
}

impl MockStorageClient {
    fn new(delay_ms: u64, fail_rate: f64) -> Self {
        Self {
            network_delay_ms: delay_ms,
            fail_rate,
            points_stored: Arc::new(AtomicUsize::new(0)),
        }
    }

    async fn store_points(&self, points: Vec<Vec<f32>>) -> Result<(), String> {
        // Simulate network delay
        if self.network_delay_ms > 0 {
            tokio::time::sleep(Duration::from_millis(self.network_delay_ms)).await;
        }

        // Simulate failure rate
        if fastrand::f64() < self.fail_rate {
            return Err("Simulated network failure".to_string());
        }

        self.points_stored.fetch_add(points.len(), Ordering::SeqCst);
        Ok(())
    }

    fn get_stored_count(&self) -> usize {
        self.points_stored.load(Ordering::SeqCst)
    }

    fn reset(&self) {
        self.points_stored.store(0, Ordering::SeqCst);
    }
}

// ============================================================================
// Test File Generators
// ============================================================================

/// Create temporary file with specified size in KB
async fn create_test_file(size_kb: usize) -> Result<NamedTempFile, std::io::Error> {
    let temp_file = NamedTempFile::new()?;
    let mut file = fs::File::create(temp_file.path()).await?;

    let pattern = b"Lorem ipsum dolor sit amet, consectetur adipiscing elit. ";
    let target_bytes = size_kb * 1024;
    let mut written = 0;

    while written < target_bytes {
        let to_write = std::cmp::min(pattern.len(), target_bytes - written);
        file.write_all(&pattern[..to_write]).await?;
        written += to_write;
    }

    file.flush().await?;
    drop(file);

    Ok(temp_file)
}

/// Generate code file content with specified number of functions
fn generate_code_content(function_count: usize) -> String {
    let mut content = String::from("// Auto-generated code for stress testing\n\n");

    for i in 0..function_count {
        content.push_str(&format!(
            r#"
/// Documentation for function_{i}
pub fn function_{i}(x: u32, y: u32) -> u32 {{
    let result = (x + y) * {i};
    println!("Processing: {{}}", result);
    result
}}

#[test]
fn test_function_{i}() {{
    assert_eq!(function_{i}(1, 2), {});
}}
"#,
            i * 3
        ));
    }

    content
}

// ============================================================================
// Stress Test Pipeline
// ============================================================================

/// Full ingestion pipeline simulation
async fn ingest_file_pipeline(
    file_path: &Path,
    processor: &MockDocumentProcessor,
    embedding_gen: &MockEmbeddingGenerator,
    storage: &MockStorageClient,
) -> Result<(), String> {
    // Process file
    let result = processor.process_file(file_path).await?;

    // Generate embeddings for chunks
    let mut embeddings = Vec::new();
    for i in 0..result.chunks_created {
        let chunk_text = format!("Chunk {} content from file", i);
        let embedding = embedding_gen.generate_embedding(&chunk_text).await;
        embeddings.push(embedding);
    }

    // Store in vector database
    storage.store_points(embeddings).await?;

    Ok(())
}

// ============================================================================
// Benchmark Functions
// ============================================================================

/// Benchmark 1: High volume ingestion (1000+ files)
fn bench_high_volume_ingestion(c: &mut Criterion) {
    let mut group = c.benchmark_group("high_volume_ingestion");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(60));

    for file_count in [100, 500, 1000, 1500].iter() {
        group.throughput(Throughput::Elements(*file_count as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(file_count),
            file_count,
            |b, &file_count| {
                let rt = Runtime::new().unwrap();

                b.to_async(&rt).iter(|| async {
                    let processor = MockDocumentProcessor::new(5, 1); // 5 chunks, 1ms delay
                    let embedding_gen = MockEmbeddingGenerator::new(384, 0);
                    let storage = MockStorageClient::new(1, 0.0); // 1ms network delay, no failures

                    // Create test files
                    let temp_dir = tempdir().unwrap();
                    let mut tasks = Vec::new();

                    for i in 0..file_count {
                        let file = create_test_file(5).await.unwrap(); // 5KB files
                        let path = file.path().to_path_buf();

                        let proc = &processor;
                        let emb = &embedding_gen;
                        let stor = &storage;

                        tasks.push(tokio::spawn(async move {
                            let _ = ingest_file_pipeline(&path, proc, emb, stor).await;
                            drop(file); // Keep file alive until processing done
                        }));
                    }

                    // Wait for all tasks
                    for task in tasks {
                        let _ = task.await;
                    }

                    black_box(storage.get_stored_count())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark 2: High rate ingestion (100+ files/second)
fn bench_high_rate_ingestion(c: &mut Criterion) {
    let mut group = c.benchmark_group("high_rate_ingestion");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    for target_rate in [50, 100, 150, 200].iter() {
        group.throughput(Throughput::Elements(*target_rate as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_per_sec", target_rate)),
            target_rate,
            |b, &target_rate| {
                let rt = Runtime::new().unwrap();

                b.to_async(&rt).iter(|| async {
                    let processor = MockDocumentProcessor::new(3, 0); // Fast processing
                    let embedding_gen = MockEmbeddingGenerator::new(384, 0);
                    let storage = MockStorageClient::new(0, 0.0); // No delay

                    let duration = Duration::from_secs(2); // 2 second burst
                    let file_count = target_rate * 2;
                    let interval = duration / file_count as u32;

                    let start = tokio::time::Instant::now();
                    let mut tasks = Vec::new();

                    for i in 0..file_count {
                        let target_time = start + (interval * i as u32);
                        tokio::time::sleep_until(target_time).await;

                        let file = create_test_file(2).await.unwrap();
                        let path = file.path().to_path_buf();

                        let proc = &processor;
                        let emb = &embedding_gen;
                        let stor = &storage;

                        tasks.push(tokio::spawn(async move {
                            let _ = ingest_file_pipeline(&path, proc, emb, stor).await;
                            drop(file);
                        }));
                    }

                    for task in tasks {
                        let _ = task.await;
                    }

                    black_box(storage.get_stored_count())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark 3: Multiple concurrent watchers
fn bench_multiple_watchers(c: &mut Criterion) {
    let mut group = c.benchmark_group("multiple_watchers");
    group.sample_size(10);

    for watcher_count in [5, 10, 15, 20].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_watchers", watcher_count)),
            watcher_count,
            |b, &watcher_count| {
                let rt = Runtime::new().unwrap();

                b.to_async(&rt).iter(|| async {
                    let files_per_watcher = 50;
                    let processor = MockDocumentProcessor::new(3, 1);
                    let embedding_gen = MockEmbeddingGenerator::new(384, 0);
                    let storage = MockStorageClient::new(1, 0.0);

                    let mut all_tasks = Vec::new();

                    // Simulate multiple watchers processing in parallel
                    for _watcher_id in 0..watcher_count {
                        for _file_id in 0..files_per_watcher {
                            let file = create_test_file(3).await.unwrap();
                            let path = file.path().to_path_buf();

                            let proc = &processor;
                            let emb = &embedding_gen;
                            let stor = &storage;

                            all_tasks.push(tokio::spawn(async move {
                                let _ = ingest_file_pipeline(&path, proc, emb, stor).await;
                                drop(file);
                            }));
                        }
                    }

                    for task in all_tasks {
                        let _ = task.await;
                    }

                    black_box(storage.get_stored_count())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark 4: Large file processing
fn bench_large_files(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_files");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(60));

    for file_size_mb in [10, 50, 100].iter() {
        group.throughput(Throughput::Bytes((*file_size_mb * 1024 * 1024) as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}MB", file_size_mb)),
            file_size_mb,
            |b, &file_size_mb| {
                let rt = Runtime::new().unwrap();

                b.to_async(&rt).iter(|| async {
                    let chunks_for_size = (file_size_mb / 5).max(10); // ~5MB chunks
                    let processor = MockDocumentProcessor::new(chunks_for_size, 2);
                    let embedding_gen = MockEmbeddingGenerator::new(384, 1);
                    let storage = MockStorageClient::new(2, 0.0);

                    let file = create_test_file(file_size_mb * 1024).await.unwrap();
                    let path = file.path();

                    let result = ingest_file_pipeline(path, &processor, &embedding_gen, &storage)
                        .await;

                    drop(file);
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark 5: Memory constrained processing
fn bench_memory_constraints(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_constraints");
    group.sample_size(10);

    for max_concurrent in [5, 10, 20].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("max_{}_concurrent", max_concurrent)),
            max_concurrent,
            |b, &max_concurrent| {
                let rt = Runtime::new().unwrap();

                b.to_async(&rt).iter(|| async {
                    let file_count = 200;
                    let processor = MockDocumentProcessor::new(5, 1);
                    let embedding_gen = MockEmbeddingGenerator::new(384, 0);
                    let storage = MockStorageClient::new(1, 0.0);
                    let semaphore = Arc::new(Semaphore::new(max_concurrent));

                    let mut tasks = Vec::new();

                    for _ in 0..file_count {
                        let file = create_test_file(10).await.unwrap();
                        let path = file.path().to_path_buf();
                        let sem = semaphore.clone();

                        let proc = &processor;
                        let emb = &embedding_gen;
                        let stor = &storage;

                        tasks.push(tokio::spawn(async move {
                            let _permit = sem.acquire().await.unwrap();
                            let _ = ingest_file_pipeline(&path, proc, emb, stor).await;
                            drop(file);
                        }));
                    }

                    for task in tasks {
                        let _ = task.await;
                    }

                    black_box(storage.get_stored_count())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark 6: Network failure resilience
fn bench_network_failures(c: &mut Criterion) {
    let mut group = c.benchmark_group("network_failures");
    group.sample_size(10);

    for fail_rate in [0.0, 0.1, 0.25, 0.5].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}%_failure", (fail_rate * 100.0) as u32)),
            fail_rate,
            |b, &fail_rate| {
                let rt = Runtime::new().unwrap();

                b.to_async(&rt).iter(|| async {
                    let file_count = 100;
                    let processor = MockDocumentProcessor::new(3, 1);
                    let embedding_gen = MockEmbeddingGenerator::new(384, 0);
                    let storage = MockStorageClient::new(5, fail_rate); // Network issues

                    let mut success_count = 0;
                    let mut tasks = Vec::new();

                    for _ in 0..file_count {
                        let file = create_test_file(5).await.unwrap();
                        let path = file.path().to_path_buf();

                        let proc = &processor;
                        let emb = &embedding_gen;
                        let stor = &storage;

                        tasks.push(tokio::spawn(async move {
                            // Retry logic
                            let mut retries = 3;
                            while retries > 0 {
                                match ingest_file_pipeline(&path, proc, emb, stor).await {
                                    Ok(_) => {
                                        drop(file);
                                        return 1;
                                    }
                                    Err(_) => {
                                        retries -= 1;
                                        tokio::time::sleep(Duration::from_millis(10)).await;
                                    }
                                }
                            }
                            drop(file);
                            0
                        }));
                    }

                    for task in tasks {
                        if let Ok(result) = task.await {
                            success_count += result;
                        }
                    }

                    black_box(success_count)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark 7: Code analysis workload
fn bench_code_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("code_analysis");
    group.sample_size(10);

    for function_count in [10, 50, 100, 200].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_functions", function_count)),
            function_count,
            |b, &function_count| {
                let rt = Runtime::new().unwrap();

                b.to_async(&rt).iter(|| async {
                    let file_count = 50;
                    let processor = MockDocumentProcessor::new(function_count / 5, 2);
                    let embedding_gen = MockEmbeddingGenerator::new(384, 1);
                    let storage = MockStorageClient::new(1, 0.0);

                    let temp_dir = tempdir().unwrap();
                    let mut tasks = Vec::new();

                    for i in 0..file_count {
                        let content = generate_code_content(function_count);
                        let file_path = temp_dir.path().join(format!("code_{}.rs", i));

                        // Write code file
                        let mut file = fs::File::create(&file_path).await.unwrap();
                        file.write_all(content.as_bytes()).await.unwrap();
                        file.flush().await.unwrap();
                        drop(file);

                        let path = file_path.clone();
                        let proc = &processor;
                        let emb = &embedding_gen;
                        let stor = &storage;

                        tasks.push(tokio::spawn(async move {
                            let _ = ingest_file_pipeline(&path, proc, emb, stor).await;
                        }));
                    }

                    for task in tasks {
                        let _ = task.await;
                    }

                    black_box(storage.get_stored_count())
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Criterion Configuration
// ============================================================================

criterion_group!(
    name = stress_benches;
    config = Criterion::default()
        .sample_size(10)
        .measurement_time(Duration::from_secs(30))
        .warm_up_time(Duration::from_secs(5));
    targets =
        bench_high_volume_ingestion,
        bench_high_rate_ingestion,
        bench_multiple_watchers,
        bench_large_files,
        bench_memory_constraints,
        bench_network_failures,
        bench_code_analysis
);

criterion_main!(stress_benches);
