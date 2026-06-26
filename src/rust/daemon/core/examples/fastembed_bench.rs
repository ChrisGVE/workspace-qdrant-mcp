//! Standalone fastembed benchmark for keyword pipeline viability.
//!
//! Tests embedding throughput at batch sizes matching the keyword extraction
//! pipeline: 50, 100, 200 short phrases. Measures wall time per batch and
//! per-phrase to determine if local ONNX is viable for the keyword pipeline
//! on this machine.
//!
//! Run:
//!   ORT_LIB_LOCATION=$HOME/.onnxruntime-static/lib \
//!     cargo run --release --manifest-path src/rust/Cargo.toml \
//!     --package workspace-qdrant-core --example fastembed_bench

use std::time::Instant;

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

fn main() {
    println!("=== FastEmbed Keyword Pipeline Benchmark ===\n");

    let init_start = Instant::now();
    let init_options =
        InitOptions::new(EmbeddingModel::AllMiniLML6V2).with_show_download_progress(true);
    let mut model = TextEmbedding::try_new(init_options).expect("Failed to init FastEmbed");
    println!("Model init: {}ms\n", init_start.elapsed().as_millis());

    let phrases = generate_keyword_phrases();

    // Warmup
    let _ = model.embed(vec!["warmup phrase"], None);

    for batch_size in [10, 50, 100, 200] {
        let batch: Vec<&str> = phrases
            .iter()
            .take(batch_size)
            .map(|s| s.as_str())
            .collect();
        let actual = batch.len();

        // Run 3 iterations, take median
        let mut times = Vec::new();
        for _ in 0..3 {
            let start = Instant::now();
            let result = model.embed(batch.clone(), None).expect("embed failed");
            let elapsed = start.elapsed();
            assert_eq!(result.len(), actual);
            times.push(elapsed);
        }
        times.sort();
        let median = times[1];

        println!(
            "Batch {:>3}: {:>6.1}ms total | {:>5.2}ms/phrase | {:>5.0} phrases/sec",
            actual,
            median.as_secs_f64() * 1000.0,
            median.as_secs_f64() * 1000.0 / actual as f64,
            actual as f64 / median.as_secs_f64(),
        );
    }

    // Simulate full keyword pipeline: 200 candidates + 50 tag diversity + 30 baskets
    println!("\n--- Simulated keyword pipeline (280 phrases) ---");
    let full_batch: Vec<&str> = phrases.iter().take(200).map(|s| s.as_str()).collect();
    let tag_batch: Vec<&str> = phrases
        .iter()
        .skip(200)
        .take(50)
        .map(|s| s.as_str())
        .collect();
    let basket_batch: Vec<&str> = phrases
        .iter()
        .skip(250)
        .take(30)
        .map(|s| s.as_str())
        .collect();

    let mut total_times = Vec::new();
    for _ in 0..3 {
        let start = Instant::now();
        let _ = model.embed(full_batch.clone(), None).expect("embed failed");
        let _ = model.embed(tag_batch.clone(), None).expect("embed failed");
        let _ = model
            .embed(basket_batch.clone(), None)
            .expect("embed failed");
        total_times.push(start.elapsed());
    }
    total_times.sort();
    let median = total_times[1];
    println!(
        "Full pipeline: {:>6.1}ms (3 batch calls: 200+50+30)",
        median.as_secs_f64() * 1000.0,
    );
    println!("vs OpenAI estimate: ~1000-1300ms (3-5 HTTP round-trips)\n",);

    // Thread scaling test
    println!("--- Thread scaling (batch=200) ---");
    let batch200: Vec<&str> = phrases.iter().take(200).map(|s| s.as_str()).collect();
    for threads in [1, 2, 4] {
        let opts = InitOptions::new(EmbeddingModel::AllMiniLML6V2).with_num_threads(threads);
        let mut m = TextEmbedding::try_new(opts).expect("init failed");
        let _ = m.embed(vec!["warmup"], None);

        let mut times = Vec::new();
        for _ in 0..3 {
            let start = Instant::now();
            let _ = m.embed(batch200.clone(), None).expect("embed failed");
            times.push(start.elapsed());
        }
        times.sort();
        let median = times[1];
        println!(
            "  {} thread(s): {:>6.1}ms",
            threads,
            median.as_secs_f64() * 1000.0,
        );
    }
}

fn generate_keyword_phrases() -> Vec<String> {
    let base_phrases = vec![
        "async_trait",
        "tokio::sync::RwLock",
        "circuit_breaker",
        "queue_processor",
        "unified_queue",
        "dead_letter_queue",
        "schema_version",
        "migration",
        "foreign_keys",
        "sqlite_pool",
        "graph_store",
        "pagerank",
        "betweenness_centrality",
        "community_detection",
        "narrative_query",
        "embedding_generator",
        "qdrant_upsert",
        "batch_processing",
        "error_classification",
        "rate_limit",
        "retry_budget",
        "resurrection_count",
        "watch_folder",
        "file_watcher",
        "debounced_events",
        "tree_sitter",
        "semantic_chunking",
        "code_intelligence",
        "lsp_server",
        "document_symbols",
        "text_document",
        "keyword_extraction",
        "tf_idf",
        "bm25_scoring",
        "sparse_vector",
        "dense_embedding",
        "hybrid_search",
        "reciprocal_rank_fusion",
        "collection_config",
        "tenant_id",
        "project_id",
        "branch_management",
        "content_hash",
        "deduplication",
        "base_point",
        "grpc_service",
        "tonic_request",
        "status_code",
        "prometheus_metrics",
        "health_check",
        "probe_interval",
        "configuration_yaml",
        "environment_variable",
        "import React from",
        "useState useEffect",
        "pub async fn process",
        "impl GraphStore for",
        "SELECT COUNT FROM",
        "INSERT OR REPLACE INTO",
        "fn main",
        "struct Config",
        "enum ErrorType",
        "trait Provider",
        "type Result",
        "mod tests",
        "cargo build release",
        "npm run build",
        "git commit amend",
        "docker compose up",
        "kubernetes deployment",
        "terraform apply",
        "continuous integration",
        "pull request review",
        "code coverage report",
        "performance benchmark",
        "memory allocation",
        "garbage collection",
        "thread pool executor",
        "async runtime",
        "connection pooling",
        "database migration",
        "api endpoint handler",
        "middleware chain",
        "authentication token",
        "authorization scope",
        "serialization format",
        "deserialization error",
        "logging framework",
        "tracing subscriber",
        "error propagation",
        "result unwrap expect",
        "iterator adapter",
        "closure capture",
        "lifetime annotation",
        "borrow checker",
        "smart pointer",
        "reference counting",
        "channel sender receiver",
        "mutex guard",
        "atomic ordering",
        "memory fence",
        "file system watcher",
        "directory traversal",
        "path normalization",
        "symlink resolution",
        "process spawning",
        "signal handling",
        "network socket",
        "tcp listener",
        "http request response",
        "websocket connection",
        "json parsing",
        "yaml configuration",
        "regex pattern matching",
        "string interpolation",
        "hash map bucket",
        "binary tree node",
        "linked list traversal",
        "stack overflow",
        "queue data structure",
        "priority heap",
        "sorting algorithm",
        "binary search",
        "dynamic programming",
        "graph traversal",
        "depth first search",
        "breadth first search",
        "minimum spanning tree",
        "shortest path",
        "cache invalidation",
        "lazy evaluation",
        "memoization technique",
        "tail recursion",
    ];

    let mut phrases = Vec::with_capacity(300);
    for (i, p) in base_phrases.iter().enumerate() {
        phrases.push(p.to_string());
        if i < 200 {
            phrases.push(format!("{}_extended_variant", p));
        }
    }
    phrases.truncate(300);
    phrases
}
