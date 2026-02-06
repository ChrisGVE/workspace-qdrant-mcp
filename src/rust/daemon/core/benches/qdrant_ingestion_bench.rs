//! Qdrant Ingestion vs Deletion Benchmark (Task 508)
//!
//! Measures Qdrant ingestion (upsert) time vs deletion time to inform
//! whether surgical updates are worth pursuing.
//!
//! Uses the libraries collection with plain overlap chunking (no tree-sitter/LSP).
//! Requires a running Qdrant instance.
//!
//! Run with: cargo bench --bench qdrant_ingestion_bench
//!
//! NOTE: This benchmark requires a live Qdrant instance on localhost:6334.
//! It will create and delete test points in the 'libraries' collection.
//! The benchmark is NOT run as part of normal CI - it's for manual profiling.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

use workspace_qdrant_core::{
    EmbeddingGenerator, EmbeddingConfig,
    StorageClient, StorageConfig, DocumentPoint,
};

/// Generate synthetic text content of approximately the given size in bytes
fn generate_text(size_bytes: usize) -> String {
    let paragraph = "The quick brown fox jumps over the lazy dog. \
        This is a benchmark test paragraph designed to simulate realistic text content \
        that might appear in source code documentation, library readmes, or technical articles. \
        Each paragraph contains multiple sentences with varied vocabulary to produce \
        meaningful embedding vectors rather than repetitive noise. ";

    let mut text = String::with_capacity(size_bytes + paragraph.len());
    let mut counter = 0u64;
    while text.len() < size_bytes {
        // Add variation to prevent identical embeddings
        text.push_str(&format!("[Section {}] ", counter));
        text.push_str(paragraph);
        counter += 1;
    }
    text.truncate(size_bytes);
    text
}

/// Chunk text using simple overlap chunking (no tree-sitter)
fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let total = chars.len();
    let mut start = 0;

    while start < total {
        let end = (start + chunk_size).min(total);
        let chunk: String = chars[start..end].iter().collect();
        chunks.push(chunk);

        if end >= total {
            break;
        }
        start = end.saturating_sub(overlap);
    }
    chunks
}

/// Benchmark result for a single run
#[derive(Debug, Clone)]
struct BenchResult {
    file_size_bytes: usize,
    chunk_count: usize,
    embedding_time_ms: u64,
    ingestion_time_ms: u64,
    deletion_time_ms: u64,
}

#[allow(dead_code)]
impl BenchResult {
    fn ingestion_per_chunk_ms(&self) -> f64 {
        if self.chunk_count == 0 { return 0.0; }
        self.ingestion_time_ms as f64 / self.chunk_count as f64
    }

    fn deletion_per_chunk_ms(&self) -> f64 {
        if self.chunk_count == 0 { return 0.0; }
        self.deletion_time_ms as f64 / self.chunk_count as f64
    }

    fn ratio(&self) -> f64 {
        if self.deletion_time_ms == 0 { return f64::INFINITY; }
        self.ingestion_time_ms as f64 / self.deletion_time_ms as f64
    }
}

/// Run a single benchmark iteration
async fn run_benchmark_iteration(
    embedding_gen: &EmbeddingGenerator,
    storage_client: &StorageClient,
    file_size: usize,
    iteration: usize,
) -> Result<BenchResult, String> {
    let collection = "libraries";
    let bench_tenant = format!("bench_tenant_{}", iteration);
    let bench_file = format!("/bench/test_file_{}_{}bytes.txt", iteration, file_size);

    // Step 1: Generate and chunk text
    let text = generate_text(file_size);
    let chunks = chunk_text(&text, 512, 50);
    let chunk_count = chunks.len();

    // Step 2: Generate embeddings
    let embed_start = Instant::now();
    let mut points: Vec<DocumentPoint> = Vec::with_capacity(chunk_count);

    for (i, chunk) in chunks.iter().enumerate() {
        let result = embedding_gen
            .generate_embedding(chunk, "all-MiniLM-L6-v2")
            .await
            .map_err(|e| format!("Embedding failed: {}", e))?;

        let sparse_map: HashMap<u32, f32> = result.sparse.indices.iter()
            .zip(result.sparse.values.iter())
            .map(|(&idx, &val)| (idx, val))
            .collect();

        let mut payload = HashMap::new();
        payload.insert("file_path".to_string(), serde_json::json!(bench_file));
        payload.insert("tenant_id".to_string(), serde_json::json!(bench_tenant));
        payload.insert("chunk_index".to_string(), serde_json::json!(i));
        payload.insert("content".to_string(), serde_json::json!(chunk));

        points.push(DocumentPoint {
            id: format!("bench_{}_{}", iteration, i),
            dense_vector: result.dense.vector,
            sparse_vector: Some(sparse_map),
            payload,
        });
    }
    let embedding_time_ms = embed_start.elapsed().as_millis() as u64;

    // Step 3: Measure ingestion (upsert)
    let ingest_start = Instant::now();
    storage_client
        .insert_points_batch(collection, points, Some(50))
        .await
        .map_err(|e| format!("Ingestion failed: {}", e))?;

    // Wait for Qdrant to acknowledge all points
    tokio::time::sleep(Duration::from_millis(200)).await;
    let ingestion_time_ms = ingest_start.elapsed().as_millis() as u64;

    // Step 4: Measure deletion
    let delete_start = Instant::now();
    storage_client
        .delete_points_by_filter(collection, &bench_file)
        .await
        .map_err(|e| format!("Deletion failed: {}", e))?;

    // Wait for Qdrant to complete deletion
    tokio::time::sleep(Duration::from_millis(200)).await;
    let deletion_time_ms = delete_start.elapsed().as_millis() as u64;

    Ok(BenchResult {
        file_size_bytes: file_size,
        chunk_count,
        embedding_time_ms,
        ingestion_time_ms,
        deletion_time_ms,
    })
}

fn qdrant_ingestion_benchmark(c: &mut Criterion) {
    // Check if Qdrant is available before running
    let rt = tokio::runtime::Runtime::new().unwrap();

    let storage_config = StorageConfig::daemon_mode();
    let storage_client = Arc::new(StorageClient::with_config(storage_config));

    // Quick connection check
    let connected = rt.block_on(async {
        storage_client.test_connection().await.unwrap_or(false)
    });

    if !connected {
        eprintln!("WARNING: Qdrant not available at localhost:6334. Skipping benchmark.");
        eprintln!("Start Qdrant to run this benchmark.");

        // Create a dummy benchmark that does nothing
        let mut group = c.benchmark_group("qdrant_ingestion_vs_deletion");
        group.bench_function("skipped_no_qdrant", |b| {
            b.iter(|| std::thread::sleep(Duration::from_millis(1)))
        });
        group.finish();
        return;
    }

    // Initialize embedding model
    // The fastembed crate downloads the model on first use.
    // Set FASTEMBED_CACHE_DIR to override the cache directory.
    let embedding_config = EmbeddingConfig::default();
    let embedding_gen = Arc::new(
        EmbeddingGenerator::new(embedding_config)
            .expect("Failed to create embedding generator")
    );

    // Warm up the model (first call downloads if not cached)
    eprintln!("Warming up embedding model...");
    let model_ok = rt.block_on(async {
        embedding_gen.generate_embedding("warmup text for model initialization", "all-MiniLM-L6-v2").await.is_ok()
    });

    if !model_ok {
        eprintln!("WARNING: Embedding model failed to initialize (network may be unavailable).");
        eprintln!("Skipping benchmark. Ensure the model is cached locally.");

        let mut group = c.benchmark_group("qdrant_ingestion_vs_deletion");
        group.bench_function("skipped_no_model", |b| {
            b.iter(|| std::thread::sleep(Duration::from_millis(1)))
        });
        group.finish();
        return;
    }

    let file_sizes: Vec<usize> = vec![
        100 * 1024,   // 100KB
        500 * 1024,   // 500KB
        1024 * 1024,  // 1MB
    ];

    let mut group = c.benchmark_group("qdrant_ingestion_vs_deletion");
    group.sample_size(10); // Fewer samples for network operations
    group.measurement_time(Duration::from_secs(60));
    group.warm_up_time(Duration::from_secs(5));

    for &size in &file_sizes {
        let size_label = match size {
            s if s >= 1024 * 1024 => format!("{}MB", s / (1024 * 1024)),
            s => format!("{}KB", s / 1024),
        };

        let eg = embedding_gen.clone();
        let sc = storage_client.clone();

        group.bench_with_input(
            BenchmarkId::new("full_cycle", &size_label),
            &size,
            |b, &file_size| {
                let mut iteration = 0;
                b.to_async(tokio::runtime::Runtime::new().unwrap()).iter(|| {
                    let eg = eg.clone();
                    let sc = sc.clone();
                    iteration += 1;
                    async move {
                        match run_benchmark_iteration(&eg, &sc, file_size, iteration).await {
                            Ok(result) => {
                                eprintln!(
                                    "  [{} chunks] embed={}ms ingest={}ms delete={}ms ratio={:.1}x",
                                    result.chunk_count,
                                    result.embedding_time_ms,
                                    result.ingestion_time_ms,
                                    result.deletion_time_ms,
                                    result.ratio(),
                                );
                            }
                            Err(e) => eprintln!("  ERROR: {}", e),
                        }
                    }
                });
            },
        );
    }
    group.finish();

    // Run a detailed report with multiple iterations
    eprintln!("\n=== Detailed Benchmark Report ===");
    rt.block_on(async {
        for &size in &file_sizes {
            let size_label = match size {
                s if s >= 1024 * 1024 => format!("{}MB", s / (1024 * 1024)),
                s => format!("{}KB", s / 1024),
            };

            let mut results = Vec::new();
            for i in 0..3 {
                match run_benchmark_iteration(&embedding_gen, &storage_client, size, 1000 + i).await {
                    Ok(r) => results.push(r),
                    Err(e) => eprintln!("Error in iteration {}: {}", i, e),
                }
            }

            if results.is_empty() {
                continue;
            }

            let avg_embed: f64 = results.iter().map(|r| r.embedding_time_ms as f64).sum::<f64>() / results.len() as f64;
            let avg_ingest: f64 = results.iter().map(|r| r.ingestion_time_ms as f64).sum::<f64>() / results.len() as f64;
            let avg_delete: f64 = results.iter().map(|r| r.deletion_time_ms as f64).sum::<f64>() / results.len() as f64;
            let avg_chunks = results[0].chunk_count;
            let avg_ratio = if avg_delete > 0.0 { avg_ingest / avg_delete } else { f64::INFINITY };
            let avg_ingest_per_chunk = avg_ingest / avg_chunks as f64;
            let avg_delete_per_chunk = avg_delete / avg_chunks as f64;

            eprintln!("\n--- {} ({} chunks, {} iterations) ---", size_label, avg_chunks, results.len());
            eprintln!("  Embedding:  {:.0}ms avg", avg_embed);
            eprintln!("  Ingestion:  {:.0}ms avg ({:.2}ms/chunk)", avg_ingest, avg_ingest_per_chunk);
            eprintln!("  Deletion:   {:.0}ms avg ({:.2}ms/chunk)", avg_delete, avg_delete_per_chunk);
            eprintln!("  Ratio:      {:.1}x (ingestion/deletion)", avg_ratio);
        }
    });
}

criterion_group!(benches, qdrant_ingestion_benchmark);
criterion_main!(benches);
