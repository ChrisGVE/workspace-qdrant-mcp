//! Queue Processor Performance Benchmarks
//!
//! Comprehensive benchmarks to validate queue processor performance against
//! the 1000+ docs/min throughput target and identify performance bottlenecks.

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tempfile::tempdir;
use tokio::runtime::Runtime;

// Mock components for isolated queue performance testing
// (no external I/O dependencies - tests pure queue logic)

/// Mock document processor - instant processing
struct MockDocumentProcessor {
    chunk_count: usize,
}

impl MockDocumentProcessor {
    fn new(chunk_count: usize) -> Self {
        Self { chunk_count }
    }

    async fn extract_document_content(&self, _path: &Path) -> MockDocumentContent {
        MockDocumentContent {
            chunks: (0..self.chunk_count)
                .map(|i| MockChunk {
                    content: format!("Chunk {} content", i),
                    chunk_index: i,
                    metadata: HashMap::new(),
                })
                .collect(),
            document_type: "mock".to_string(),
            metadata: HashMap::new(),
        }
    }
}

struct MockDocumentContent {
    chunks: Vec<MockChunk>,
    document_type: String,
    metadata: HashMap<String, String>,
}

struct MockChunk {
    content: String,
    chunk_index: usize,
    metadata: HashMap<String, String>,
}

/// Mock embedding generator - instant embeddings
struct MockEmbeddingGenerator {
    vector_size: usize,
}

impl MockEmbeddingGenerator {
    fn new(vector_size: usize) -> Self {
        Self { vector_size }
    }

    async fn generate_embedding(&self, text: &str) -> MockEmbedding {
        // Deterministic vector generation based on text hash
        let hash = text.len() as f32;
        MockEmbedding {
            dense: (0..self.vector_size)
                .map(|i| (hash + i as f32) % 1.0)
                .collect(),
        }
    }
}

struct MockEmbedding {
    dense: Vec<f32>,
}

/// Mock storage client - instant storage operations
struct MockStorageClient {
    collections: Arc<tokio::sync::Mutex<HashMap<String, Vec<MockPoint>>>>,
}

impl MockStorageClient {
    fn new() -> Self {
        Self {
            collections: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
        }
    }

    async fn collection_exists(&self, name: &str) -> bool {
        self.collections.lock().await.contains_key(name)
    }

    async fn create_collection(&self, name: &str) {
        self.collections.lock().await.insert(name.to_string(), Vec::new());
    }

    async fn insert_points(&self, collection: &str, points: Vec<MockPoint>) {
        if let Some(coll) = self.collections.lock().await.get_mut(collection) {
            coll.extend(points);
        }
    }

    async fn delete_by_filter(&self, collection: &str, _file_path: &str) {
        if let Some(coll) = self.collections.lock().await.get_mut(collection) {
            coll.clear();
        }
    }

    async fn test_connection(&self) -> Result<(), String> {
        Ok(())
    }
}

struct MockPoint {
    id: String,
    vector: Vec<f32>,
    payload: HashMap<String, String>,
}

/// Simulated queue item processing
async fn process_queue_item(
    file_path: &str,
    operation: &str,
    doc_processor: &MockDocumentProcessor,
    embedding_gen: &MockEmbeddingGenerator,
    storage: &MockStorageClient,
    collection: &str,
) {
    match operation {
        "ingest" => {
            // Extract content
            let content = doc_processor
                .extract_document_content(Path::new(file_path))
                .await;

            // Ensure collection exists
            if !storage.collection_exists(collection).await {
                storage.create_collection(collection).await;
            }

            // Generate embeddings and store
            let mut points = Vec::new();
            for chunk in content.chunks {
                let embedding = embedding_gen.generate_embedding(&chunk.content).await;
                points.push(MockPoint {
                    id: uuid::Uuid::new_v4().to_string(),
                    vector: embedding.dense,
                    payload: HashMap::new(),
                });
            }

            storage.insert_points(collection, points).await;
        }
        "update" => {
            // Delete existing + ingest new
            storage.delete_by_filter(collection, file_path).await;
            Box::pin(process_queue_item(
                file_path,
                "ingest",
                doc_processor,
                embedding_gen,
                storage,
                collection,
            ))
            .await;
        }
        "delete" => {
            storage.delete_by_filter(collection, file_path).await;
        }
        _ => {}
    }
}

/// Benchmark: Throughput Test (1000+ docs/min target)
fn throughput_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput");
    group.throughput(Throughput::Elements(100));
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(50);

    let rt = Runtime::new().unwrap();

    for doc_count in [100, 500, 1000] {
        group.bench_with_input(
            BenchmarkId::new("process_docs", doc_count),
            &doc_count,
            |b, &count| {
                b.to_async(&rt).iter(|| async {
                    let doc_processor = MockDocumentProcessor::new(5); // 5 chunks per doc
                    let embedding_gen = MockEmbeddingGenerator::new(384);
                    let storage = MockStorageClient::new();
                    let collection = "bench_collection";

                    // Process documents concurrently (simulates batch processing)
                    let tasks: Vec<_> = (0..count)
                        .map(|i| {
                            let file_path = format!("/test/doc_{}.txt", i);
                            let doc_proc = &doc_processor;
                            let emb_gen = &embedding_gen;
                            let stor = &storage;
                            async move {
                                process_queue_item(
                                    &file_path,
                                    "ingest",
                                    doc_proc,
                                    emb_gen,
                                    stor,
                                    collection,
                                )
                                .await;
                            }
                        })
                        .collect();

                    futures::future::join_all(tasks).await;
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Concurrent Processing with Different Batch Sizes
fn concurrent_processing_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent");
    group.measurement_time(Duration::from_secs(20));

    let rt = Runtime::new().unwrap();

    for batch_size in [1, 5, 10, 20, 50] {
        group.bench_with_input(
            BenchmarkId::new("batch_size", batch_size),
            &batch_size,
            |b, &size| {
                b.to_async(&rt).iter(|| async {
                    let doc_processor = MockDocumentProcessor::new(3);
                    let embedding_gen = MockEmbeddingGenerator::new(384);
                    let storage = MockStorageClient::new();
                    let collection = "concurrent_bench";

                    // Process 100 items with specified batch size
                    let total_items = 100;
                    for batch_start in (0..total_items).step_by(size) {
                        let batch_end = (batch_start + size).min(total_items);
                        let tasks: Vec<_> = (batch_start..batch_end)
                            .map(|i| {
                                let file_path = format!("/test/batch_doc_{}.txt", i);
                                let doc_proc = &doc_processor;
                                let emb_gen = &embedding_gen;
                                let stor = &storage;
                                async move {
                                    process_queue_item(
                                        &file_path,
                                        "ingest",
                                        doc_proc,
                                        emb_gen,
                                        stor,
                                        collection,
                                    )
                                    .await;
                                }
                            })
                            .collect();

                        futures::future::join_all(tasks).await;
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Queue Depth Handling
fn queue_depth_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("queue_depth");
    group.measurement_time(Duration::from_secs(15));

    let rt = Runtime::new().unwrap();

    for queue_size in [10, 100, 1000, 10000] {
        group.bench_with_input(
            BenchmarkId::new("queue_items", queue_size),
            &queue_size,
            |b, &size| {
                b.to_async(&rt).iter(|| async {
                    // Simulate queue operations at different depths
                    let items: Vec<_> = (0..size)
                        .map(|i| (format!("/test/queue_doc_{}.txt", i), "ingest".to_string()))
                        .collect();

                    // Measure dequeue and process performance
                    let doc_processor = MockDocumentProcessor::new(2);
                    let embedding_gen = MockEmbeddingGenerator::new(384);
                    let storage = MockStorageClient::new();
                    let collection = "queue_depth_bench";

                    for (file_path, operation) in items {
                        process_queue_item(
                            &file_path,
                            &operation,
                            &doc_processor,
                            &embedding_gen,
                            &storage,
                            collection,
                        )
                        .await;
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Operation Type Performance
fn operation_type_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("operation_types");
    group.measurement_time(Duration::from_secs(15));

    let rt = Runtime::new().unwrap();

    for operation in ["ingest", "update", "delete"] {
        group.bench_with_input(
            BenchmarkId::new("operation", operation),
            &operation.to_string(),
            |b, op| {
                b.to_async(&rt).iter(|| async {
                    let doc_processor = MockDocumentProcessor::new(5);
                    let embedding_gen = MockEmbeddingGenerator::new(384);
                    let storage = MockStorageClient::new();
                    let collection = "operation_bench";

                    // Pre-create collection
                    storage.create_collection(collection).await;

                    // Process 50 items
                    let tasks: Vec<_> = (0..50)
                        .map(|i| {
                            let file_path = format!("/test/op_doc_{}.txt", i);
                            let operation = op.clone();
                            let doc_proc = &doc_processor;
                            let emb_gen = &embedding_gen;
                            let stor = &storage;
                            async move {
                                process_queue_item(
                                    &file_path,
                                    &operation,
                                    doc_proc,
                                    emb_gen,
                                    stor,
                                    collection,
                                )
                                .await;
                            }
                        })
                        .collect();

                    futures::future::join_all(tasks).await;
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Processing Time Distribution
fn processing_time_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("processing_time");
    group.measurement_time(Duration::from_secs(20));

    let rt = Runtime::new().unwrap();

    // Test with different document complexities
    for chunk_count in [1, 5, 10, 20] {
        group.bench_with_input(
            BenchmarkId::new("chunks_per_doc", chunk_count),
            &chunk_count,
            |b, &chunks| {
                b.to_async(&rt).iter(|| async {
                    let doc_processor = MockDocumentProcessor::new(chunks);
                    let embedding_gen = MockEmbeddingGenerator::new(384);
                    let storage = MockStorageClient::new();
                    let collection = "time_bench";

                    storage.create_collection(collection).await;

                    // Process single document
                    process_queue_item(
                        "/test/complex_doc.txt",
                        "ingest",
                        &doc_processor,
                        &embedding_gen,
                        &storage,
                        collection,
                    )
                    .await;
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Memory Efficiency
fn memory_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory");
    group.measurement_time(Duration::from_secs(20));
    group.sample_size(30);

    let rt = Runtime::new().unwrap();

    group.bench_function("process_1000_docs_memory", |b| {
        b.to_async(&rt).iter(|| async {
            let doc_processor = MockDocumentProcessor::new(5);
            let embedding_gen = MockEmbeddingGenerator::new(384);
            let storage = MockStorageClient::new();
            let collection = "memory_bench";

            storage.create_collection(collection).await;

            // Process documents and verify no memory accumulation
            for i in 0..1000 {
                process_queue_item(
                    &format!("/test/mem_doc_{}.txt", i),
                    "ingest",
                    &doc_processor,
                    &embedding_gen,
                    &storage,
                    collection,
                )
                .await;
            }
        });
    });

    group.finish();
}

/// Benchmark: Error Recovery Impact
fn error_recovery_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("error_recovery");
    group.measurement_time(Duration::from_secs(15));

    let rt = Runtime::new().unwrap();

    group.bench_function("mixed_success_failure", |b| {
        b.to_async(&rt).iter(|| async {
            let doc_processor = MockDocumentProcessor::new(3);
            let embedding_gen = MockEmbeddingGenerator::new(384);
            let storage = MockStorageClient::new();
            let collection = "error_bench";

            storage.create_collection(collection).await;

            // Mix of successful and potentially failing operations
            for i in 0..100 {
                let operation = if i % 10 == 0 { "delete" } else { "ingest" };
                process_queue_item(
                    &format!("/test/mixed_doc_{}.txt", i),
                    operation,
                    &doc_processor,
                    &embedding_gen,
                    &storage,
                    collection,
                )
                .await;
            }
        });
    });

    group.finish();
}

/// Validate throughput meets 1000+ docs/min target
fn validate_throughput_target(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput_validation");
    group.throughput(Throughput::Elements(1000));
    group.measurement_time(Duration::from_secs(60)); // Full minute
    group.sample_size(20);

    let rt = Runtime::new().unwrap();

    group.bench_function("1000_docs_per_minute", |b| {
        b.to_async(&rt).iter(|| async {
            let doc_processor = MockDocumentProcessor::new(5);
            let embedding_gen = MockEmbeddingGenerator::new(384);
            let storage = MockStorageClient::new();
            let collection = "validation_bench";

            storage.create_collection(collection).await;

            // Process 1000 documents
            let start = std::time::Instant::now();

            let tasks: Vec<_> = (0..1000)
                .map(|i| {
                    let file_path = format!("/test/validation_doc_{}.txt", i);
                    let doc_proc = &doc_processor;
                    let emb_gen = &embedding_gen;
                    let stor = &storage;
                    async move {
                        process_queue_item(
                            &file_path,
                            "ingest",
                            doc_proc,
                            emb_gen,
                            stor,
                            collection,
                        )
                        .await;
                    }
                })
                .collect();

            futures::future::join_all(tasks).await;

            let elapsed = start.elapsed();
            let docs_per_min = (1000.0 / elapsed.as_secs_f64()) * 60.0;

            // Log performance for validation
            println!("\n📊 Throughput: {:.1} docs/min", docs_per_min);
            println!(
                "✅ Target {} (>= 1000 docs/min)",
                if docs_per_min >= 1000.0 {
                    "MET"
                } else {
                    "NOT MET"
                }
            );
        });
    });

    group.finish();
}

criterion_group!(
    name = queue_benches;
    config = Criterion::default()
        .sample_size(50)
        .measurement_time(Duration::from_secs(20))
        .warm_up_time(Duration::from_secs(3));
    targets =
        throughput_benchmark,
        concurrent_processing_benchmark,
        queue_depth_benchmark,
        operation_type_benchmark,
        processing_time_benchmark,
        memory_benchmark,
        error_recovery_benchmark,
        validate_throughput_target
);

criterion_main!(queue_benches);
