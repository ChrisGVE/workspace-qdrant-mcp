//! Qdrant Ingestion vs Deletion Benchmark (Task 508/516)
//!
//! Measures embedding generation, Qdrant upsert, and deletion timing using
//! real PDF content at multiple sizes (100KB, 500KB, 1MB, full).
//!
//! Benchmark results (2026-02-09, Apple M-series, Qdrant localhost):
//!
//!   Size       Chunks  Embed(ms/chunk)  Ingest(ms/chunk)  Delete(ms/chunk)  Ratio
//!   100KB        220     18.72            0.61              0.01             44.6x
//!   500KB       1071     16.60            0.44              0.01             39.6x
//!   1MB         2181     24.63            0.37              0.01             31.0x
//!   full_pdf    2527     36.28            0.41              0.01             28.3x
//!
//!   Conclusion: delete+re-ingest is the right strategy; deletion is free.
//!   Embedding dominates total time (>98%). Ingestion/deletion ratio: 28-45x.
//!
//! Uses a dedicated `bench_ingestion` collection with plain overlap chunking (no tree-sitter/LSP).
//! The collection is created at the start and deleted at the end (no production data touched).
//! Requires a running Qdrant instance and a PDF file for input.
//!
//! Run with:
//!   cargo bench --bench qdrant_ingestion_bench
//!
//! Set BENCH_PDF_PATH to override the default PDF file:
//!   BENCH_PDF_PATH=/path/to/file.pdf cargo bench --bench qdrant_ingestion_bench
//!
//! NOTE: This benchmark requires a live Qdrant instance on localhost:6334.
//! It will create and delete test points in the 'libraries' collection.
//! The benchmark is NOT run as part of normal CI - it's for manual profiling.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use uuid::Uuid;

use workspace_qdrant_core::{
    EmbeddingGenerator, EmbeddingConfig,
    StorageClient, StorageConfig, DocumentPoint,
};

/// Namespace UUID for benchmark point IDs (deterministic)
const BENCH_UUID_NAMESPACE: Uuid = Uuid::from_bytes([
    0xb6, 0x43, 0x1a, 0x2e, 0x7f, 0x8c, 0x4d, 0x1a,
    0x9e, 0x3b, 0x5c, 0x2d, 0x4e, 0x6f, 0x7a, 0x8b,
]);

/// Default PDF path relative to the project root
const DEFAULT_PDF_RELATIVE: &str =
    "Particles in the Dark Universe -  A Student\u{2019}s Guide to Particle Physics and Cosmology.pdf";

/// Resolve the PDF path: BENCH_PDF_PATH env var, or default relative to project root
fn resolve_pdf_path() -> Option<PathBuf> {
    // Check env var first
    if let Ok(path) = std::env::var("BENCH_PDF_PATH") {
        let p = PathBuf::from(&path);
        if p.exists() {
            return Some(p);
        }
        eprintln!("WARNING: BENCH_PDF_PATH={} does not exist", path);
    }

    // Walk up from the bench file to find the project root (contains CLAUDE.md)
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for _ in 0..5 {
        let candidate = dir.join(DEFAULT_PDF_RELATIVE);
        if candidate.exists() {
            return Some(candidate);
        }
        if !dir.pop() {
            break;
        }
    }

    None
}

/// Extract text from a PDF file
fn extract_pdf_text(path: &PathBuf) -> Result<String, String> {
    pdf_extract::extract_text(path)
        .map_err(|e| format!("PDF extraction failed: {}", e))
}

/// Clean extracted text: collapse whitespace, remove control chars
fn clean_text(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut last_was_whitespace = false;

    for ch in text.chars() {
        if ch.is_control() && ch != '\n' {
            continue;
        }
        if ch.is_whitespace() {
            if !last_was_whitespace {
                result.push(' ');
                last_was_whitespace = true;
            }
        } else {
            result.push(ch);
            last_was_whitespace = false;
        }
    }
    result
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

/// Dedicated benchmark collection name (not a production collection)
const BENCH_COLLECTION: &str = "bench_ingestion";

/// Run a single benchmark iteration: embed + ingest + delete
async fn run_iteration(
    embedding_gen: &EmbeddingGenerator,
    storage_client: &StorageClient,
    chunks: &[String],
    iteration: usize,
) -> Result<(u64, u64, u64), String> {
    let collection = BENCH_COLLECTION;
    let bench_file = format!("/bench/pdf_iter_{}", iteration);

    // Step 1: Generate embeddings
    let embed_start = Instant::now();
    let mut points: Vec<DocumentPoint> = Vec::with_capacity(chunks.len());

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
        payload.insert("tenant_id".to_string(), serde_json::json!("bench_pdf"));
        payload.insert("chunk_index".to_string(), serde_json::json!(i));
        payload.insert("content".to_string(), serde_json::json!(chunk));

        // Generate deterministic UUID v5 from the bench ID string
        let id_str = format!("bench_pdf_{}_{}", iteration, i);
        let point_uuid = Uuid::new_v5(&BENCH_UUID_NAMESPACE, id_str.as_bytes());

        points.push(DocumentPoint {
            id: point_uuid.to_string(),
            dense_vector: result.dense.vector,
            sparse_vector: Some(sparse_map),
            payload,
        });
    }
    let embedding_ms = embed_start.elapsed().as_millis() as u64;

    // Step 2: Measure ingestion (upsert with wait=true for accurate timing)
    let ingest_start = Instant::now();
    storage_client
        .insert_points_batch_with_wait(collection, points, Some(50), true)
        .await
        .map_err(|e| format!("Ingestion failed: {}", e))?;
    let ingestion_ms = ingest_start.elapsed().as_millis() as u64;

    // Step 3: Measure deletion (already uses wait=true internally)
    let delete_start = Instant::now();
    storage_client
        .delete_points_by_filter(collection, &bench_file, "bench_pdf")
        .await
        .map_err(|e| format!("Deletion failed: {}", e))?;
    let deletion_ms = delete_start.elapsed().as_millis() as u64;

    Ok((embedding_ms, ingestion_ms, deletion_ms))
}

fn qdrant_ingestion_benchmark(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    // --- Resolve PDF ---
    let pdf_path = match resolve_pdf_path() {
        Some(p) => {
            eprintln!("Using PDF: {}", p.display());
            p
        }
        None => {
            eprintln!("WARNING: No PDF file found. Set BENCH_PDF_PATH or place the default PDF in the project root.");
            let mut group = c.benchmark_group("qdrant_ingestion_vs_deletion");
            group.bench_function("skipped_no_pdf", |b| {
                b.iter(|| std::thread::sleep(Duration::from_millis(1)))
            });
            group.finish();
            return;
        }
    };

    // --- Extract PDF text ---
    eprintln!("Extracting text from PDF...");
    let raw_text = match extract_pdf_text(&pdf_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("ERROR: {}", e);
            let mut group = c.benchmark_group("qdrant_ingestion_vs_deletion");
            group.bench_function("skipped_pdf_error", |b| {
                b.iter(|| std::thread::sleep(Duration::from_millis(1)))
            });
            group.finish();
            return;
        }
    };

    let full_text = clean_text(&raw_text);
    let full_text_bytes = full_text.len();
    let full_text_chars = full_text.chars().count();
    eprintln!(
        "PDF text extracted: {} bytes, {} chars",
        full_text_bytes, full_text_chars
    );

    // --- Check Qdrant ---
    let storage_config = StorageConfig::daemon_mode();
    let storage_client = Arc::new(StorageClient::with_config(storage_config));

    let connected = rt.block_on(async {
        storage_client.test_connection().await.unwrap_or(false)
    });

    if !connected {
        eprintln!("WARNING: Qdrant not available at localhost:6334. Skipping benchmark.");
        let mut group = c.benchmark_group("qdrant_ingestion_vs_deletion");
        group.bench_function("skipped_no_qdrant", |b| {
            b.iter(|| std::thread::sleep(Duration::from_millis(1)))
        });
        group.finish();
        return;
    }

    // --- Create dedicated benchmark collection ---
    let collection_created = rt.block_on(async {
        // Delete any leftover from a previous run
        let _ = storage_client.delete_collection(BENCH_COLLECTION).await;
        storage_client
            .create_collection(BENCH_COLLECTION, Some(384), None)
            .await
            .is_ok()
    });

    if !collection_created {
        eprintln!("WARNING: Failed to create benchmark collection '{}'. Skipping.", BENCH_COLLECTION);
        let mut group = c.benchmark_group("qdrant_ingestion_vs_deletion");
        group.bench_function("skipped_collection_error", |b| {
            b.iter(|| std::thread::sleep(Duration::from_millis(1)))
        });
        group.finish();
        return;
    }
    eprintln!("Created dedicated benchmark collection: {}", BENCH_COLLECTION);

    // --- Initialize embedding model ---
    let embedding_config = EmbeddingConfig::default();
    let embedding_gen = Arc::new(
        EmbeddingGenerator::new(embedding_config)
            .expect("Failed to create embedding generator")
    );

    eprintln!("Warming up embedding model...");
    let model_ok = rt.block_on(async {
        embedding_gen
            .generate_embedding("warmup text for model initialization", "all-MiniLM-L6-v2")
            .await
            .is_ok()
    });

    if !model_ok {
        eprintln!("WARNING: Embedding model failed to initialize.");
        let mut group = c.benchmark_group("qdrant_ingestion_vs_deletion");
        group.bench_function("skipped_no_model", |b| {
            b.iter(|| std::thread::sleep(Duration::from_millis(1)))
        });
        group.finish();
        return;
    }

    // --- Prepare slices of the PDF text at different sizes ---
    // Use actual content subsets rather than synthetic data
    let slice_sizes: Vec<(String, usize)> = vec![
        ("100KB".to_string(), 100 * 1024),
        ("500KB".to_string(), 500 * 1024),
        ("1MB".to_string(), 1024 * 1024),
        ("full_pdf".to_string(), full_text_bytes),
    ];

    // Pre-chunk each slice size
    let chunk_size = 512;
    let overlap = 50;
    let sliced_chunks: Vec<(String, Vec<String>)> = slice_sizes
        .iter()
        .map(|(label, size)| {
            let text_slice = if *size >= full_text_bytes {
                full_text.clone()
            } else {
                // Take the first N bytes, respecting char boundaries
                let chars: Vec<char> = full_text.chars().collect();
                let mut byte_count = 0;
                let mut char_end = 0;
                for (i, ch) in chars.iter().enumerate() {
                    byte_count += ch.len_utf8();
                    if byte_count >= *size {
                        char_end = i + 1;
                        break;
                    }
                }
                if char_end == 0 {
                    full_text.clone()
                } else {
                    chars[..char_end].iter().collect()
                }
            };
            let chunks = chunk_text(&text_slice, chunk_size, overlap);
            eprintln!("  {} → {} chunks", label, chunks.len());
            (label.clone(), chunks)
        })
        .collect();

    // --- Criterion benchmarks ---
    let mut group = c.benchmark_group("qdrant_ingestion_vs_deletion");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(120));
    group.warm_up_time(Duration::from_secs(5));

    for (label, chunks) in &sliced_chunks {
        let eg = embedding_gen.clone();
        let sc = storage_client.clone();
        let bench_chunks = chunks.clone();

        group.bench_with_input(
            BenchmarkId::new("pdf_cycle", label),
            &bench_chunks,
            |b, chunks| {
                let mut iteration = 0;
                b.to_async(tokio::runtime::Runtime::new().unwrap()).iter(|| {
                    let eg = eg.clone();
                    let sc = sc.clone();
                    let ch = chunks.clone();
                    iteration += 1;
                    async move {
                        match run_iteration(&eg, &sc, &ch, iteration).await {
                            Ok((embed, ingest, delete)) => {
                                let ratio = if delete > 0 {
                                    ingest as f64 / delete as f64
                                } else {
                                    f64::INFINITY
                                };
                                eprintln!(
                                    "  [{} chunks] embed={}ms ingest={}ms delete={}ms ratio={:.1}x",
                                    ch.len(), embed, ingest, delete, ratio,
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

    // --- Detailed report ---
    eprintln!("\n=== Detailed Benchmark Report (PDF content) ===");
    eprintln!("Source: {}", pdf_path.display());
    eprintln!("Total text: {} bytes, {} chars", full_text_bytes, full_text_chars);

    rt.block_on(async {
        for (label, chunks) in &sliced_chunks {
            let iterations = 3;
            let mut embed_times = Vec::new();
            let mut ingest_times = Vec::new();
            let mut delete_times = Vec::new();

            for i in 0..iterations {
                match run_iteration(&embedding_gen, &storage_client, chunks, 2000 + i).await {
                    Ok((e, ing, del)) => {
                        embed_times.push(e);
                        ingest_times.push(ing);
                        delete_times.push(del);
                    }
                    Err(e) => eprintln!("  Error in iteration {}: {}", i, e),
                }
            }

            if embed_times.is_empty() {
                continue;
            }

            let n = embed_times.len() as f64;
            let avg_embed = embed_times.iter().sum::<u64>() as f64 / n;
            let avg_ingest = ingest_times.iter().sum::<u64>() as f64 / n;
            let avg_delete = delete_times.iter().sum::<u64>() as f64 / n;
            let chunk_count = chunks.len();
            let ratio = if avg_delete > 0.0 { avg_ingest / avg_delete } else { f64::INFINITY };

            eprintln!("\n--- {} ({} chunks, {} iterations) ---", label, chunk_count, embed_times.len());
            eprintln!("  Embedding:  {:.0}ms avg ({:.2}ms/chunk)", avg_embed, avg_embed / chunk_count as f64);
            eprintln!("  Ingestion:  {:.0}ms avg ({:.2}ms/chunk)", avg_ingest, avg_ingest / chunk_count as f64);
            eprintln!("  Deletion:   {:.0}ms avg ({:.2}ms/chunk)", avg_delete, avg_delete / chunk_count as f64);
            eprintln!("  Ratio:      {:.1}x (ingestion/deletion)", ratio);
        }
    });

    // --- Cleanup benchmark collection ---
    eprintln!("\nCleaning up benchmark collection: {}", BENCH_COLLECTION);
    rt.block_on(async {
        let _ = storage_client.delete_collection(BENCH_COLLECTION).await;
    });
}

criterion_group!(benches, qdrant_ingestion_benchmark);
criterion_main!(benches);
