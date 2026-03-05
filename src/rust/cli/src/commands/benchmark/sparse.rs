//! Sparse vector benchmark -- BM25 vs SPLADE++ evaluation.
//!
//! Samples documents from SQLite tracked_files, generates both BM25 and SPLADE++
//! sparse vectors, and reports comparative metrics with per-sample latency tracking.

use anyhow::{Context, Result};
use std::time::{Duration, Instant};

use super::stats::LatencyStats;
use crate::config::get_database_path;
use crate::output;

/// Aggregated results from a sparse vector generation benchmark run.
struct BenchmarkResult {
    count: usize,
    avg_dims: f64,
    stats: Option<LatencyStats>,
    elapsed: Duration,
}

/// Additional SPLADE-specific metrics (first-call initialization cost).
struct SpladeExtras {
    first_call_ms: f64,
}

/// Sample file paths from the SQLite `tracked_files` table and read their content.
///
/// Returns `(path, snippet)` pairs where snippet is up to 512 chars of non-empty text.
async fn sample_files(collection: &str, sample_size: usize) -> Result<Vec<(String, String)>> {
    let db_path = get_database_path()?;
    let conn = rusqlite::Connection::open(&db_path).context("Failed to open database")?;

    let mut stmt = conn.prepare(
        "SELECT file_path FROM tracked_files tf
         JOIN watch_folders wf ON tf.watch_folder_id = wf.watch_id
         WHERE wf.collection = ?1
         ORDER BY RANDOM()
         LIMIT ?2",
    )?;

    let file_paths: Vec<String> = stmt
        .query_map(rusqlite::params![collection, sample_size as i64], |row| {
            row.get(0)
        })?
        .filter_map(|r| r.ok())
        .collect();

    let mut samples: Vec<(String, String)> = Vec::new();
    for path in file_paths.iter().take(sample_size) {
        if let Ok(content) = tokio::fs::read_to_string(path).await {
            let snippet = content.chars().take(512).collect::<String>();
            if !snippet.trim().is_empty() {
                samples.push((path.clone(), snippet));
            }
        }
    }

    Ok(samples)
}

/// Generate BM25 sparse vectors for all samples, tracking per-sample latency.
fn benchmark_bm25(samples: &[(String, String)]) -> BenchmarkResult {
    let start = Instant::now();
    let mut total_dims = 0usize;
    let mut count = 0usize;
    let mut latencies: Vec<f64> = Vec::new();

    for (_, text) in samples {
        let call_start = Instant::now();
        let tokens = workspace_qdrant_core::embedding::tokenize_for_bm25(text);
        let mut bm25 = workspace_qdrant_core::BM25::new(1.2);
        bm25.add_document(&tokens);
        let sparse = bm25.generate_sparse_vector(&tokens);
        let call_ms = call_start.elapsed().as_secs_f64() * 1000.0;
        latencies.push(call_ms);
        total_dims += sparse.indices.len();
        count += 1;
    }

    let elapsed = start.elapsed();
    let avg_dims = if count > 0 {
        total_dims as f64 / count as f64
    } else {
        0.0
    };
    let stats = LatencyStats::from_latencies(&latencies);

    BenchmarkResult {
        count,
        avg_dims,
        stats,
        elapsed,
    }
}

/// Generate SPLADE++ sparse vectors for all samples, tracking per-sample latency.
///
/// The first call is excluded from latency stats (includes model download/init).
/// Returns `None` if the first SPLADE call fails (model init failure).
async fn benchmark_splade(
    samples: &[(String, String)],
    gen_splade: &workspace_qdrant_core::EmbeddingGenerator,
) -> Option<(BenchmarkResult, SpladeExtras)> {
    let start = Instant::now();
    let mut total_dims = 0usize;
    let mut count = 0usize;
    let mut latencies: Vec<f64> = Vec::new();
    let mut first_call_ms = 0f64;

    for (i, (_, text)) in samples.iter().enumerate() {
        let call_start = Instant::now();
        match gen_splade.generate_splade_sparse_vector(text).await {
            Ok(sparse) => {
                let call_ms = call_start.elapsed().as_secs_f64() * 1000.0;
                if i == 0 {
                    first_call_ms = call_ms;
                } else {
                    latencies.push(call_ms);
                }
                total_dims += sparse.indices.len();
                count += 1;
            }
            Err(e) => {
                if i == 0 {
                    output::error(format!(
                        "SPLADE++ model initialization failed: {}. Benchmark aborted.",
                        e
                    ));
                    return None;
                }
                eprintln!("  Warning: SPLADE++ failed for sample {}: {}", i, e);
            }
        }
    }

    let elapsed = start.elapsed();
    let avg_dims = if count > 0 {
        total_dims as f64 / count as f64
    } else {
        0.0
    };
    let stats = LatencyStats::from_latencies(&latencies);

    Some((
        BenchmarkResult {
            count,
            avg_dims,
            stats,
            elapsed,
        },
        SpladeExtras { first_call_ms },
    ))
}

/// Print the formatted comparison table to stdout.
fn print_comparison(bm25: &BenchmarkResult, splade: &BenchmarkResult, extras: &SpladeExtras) {
    println!();
    println!("Results");
    println!("-------");
    println!("{:<25} {:>12} {:>12}", "", "BM25", "SPLADE++");
    println!("{:<25} {:>12} {:>12}", "---", "----", "--------");
    println!(
        "{:<25} {:>12} {:>12}",
        "Vectors generated", bm25.count, splade.count
    );
    println!(
        "{:<25} {:>12.1} {:>12.1}",
        "Avg non-zero dims", bm25.avg_dims, splade.avg_dims
    );

    if let Some(ref bs) = bm25.stats {
        let ss = splade.stats.as_ref();
        println!(
            "{:<25} {:>10.2}ms {:>10.2}ms",
            "Median latency",
            bs.median,
            ss.map_or(0.0, |s| s.median)
        );
        println!(
            "{:<25} {:>10.2}ms {:>10.2}ms",
            "Mean latency",
            bs.mean,
            ss.map_or(0.0, |s| s.mean)
        );
        println!(
            "{:<25} {:>10.2}ms {:>10.2}ms",
            "Std dev",
            bs.std_dev,
            ss.map_or(0.0, |s| s.std_dev)
        );
        println!(
            "{:<25} {:>10.2}ms {:>10.2}ms",
            "P95 latency",
            bs.p95,
            ss.map_or(0.0, |s| s.p95)
        );
        println!(
            "{:<25} {:>10.2}ms {:>10.2}ms",
            "P99 latency",
            bs.p99,
            ss.map_or(0.0, |s| s.p99)
        );
        println!(
            "{:<25} {:>10.2}ms {:>10.2}ms",
            "Min latency",
            bs.min,
            ss.map_or(0.0, |s| s.min)
        );
        println!(
            "{:<25} {:>10.2}ms {:>10.2}ms",
            "Max latency",
            bs.max,
            ss.map_or(0.0, |s| s.max)
        );
    }

    println!(
        "{:<25} {:>12} {:>10.0}ms",
        "First call (incl init)", "~0", extras.first_call_ms
    );
    println!(
        "{:<25} {:>10.0}ms {:>10.0}ms",
        "Total time",
        bm25.elapsed.as_secs_f64() * 1000.0,
        splade.elapsed.as_secs_f64() * 1000.0
    );
    println!();
    println!("Notes:");
    println!("  - SPLADE++ first call includes model download (~150MB) + initialization");
    println!("  - SPLADE++ uses BERT vocab (30522 tokens); BM25 uses dynamic corpus vocab");
    println!("  - Higher avg non-zero dims = more expressive sparse representation");
}

/// Write the JSON benchmark report to the given path.
async fn write_json_report(
    path: &str,
    collection: &str,
    sample_count: usize,
    bm25: &BenchmarkResult,
    splade: &BenchmarkResult,
    extras: &SpladeExtras,
) -> Result<()> {
    let report = serde_json::json!({
        "collection": collection,
        "sample_size": sample_count,
        "bm25": {
            "vectors_generated": bm25.count,
            "avg_nonzero_dims": bm25.avg_dims,
            "median_ms": bm25.stats.as_ref().map(|s| s.median),
            "mean_ms": bm25.stats.as_ref().map(|s| s.mean),
            "std_dev_ms": bm25.stats.as_ref().map(|s| s.std_dev),
            "p95_ms": bm25.stats.as_ref().map(|s| s.p95),
            "p99_ms": bm25.stats.as_ref().map(|s| s.p99),
            "min_ms": bm25.stats.as_ref().map(|s| s.min),
            "max_ms": bm25.stats.as_ref().map(|s| s.max),
            "total_ms": bm25.elapsed.as_secs_f64() * 1000.0,
        },
        "splade": {
            "vectors_generated": splade.count,
            "avg_nonzero_dims": splade.avg_dims,
            "median_ms": splade.stats.as_ref().map(|s| s.median),
            "mean_ms": splade.stats.as_ref().map(|s| s.mean),
            "std_dev_ms": splade.stats.as_ref().map(|s| s.std_dev),
            "p95_ms": splade.stats.as_ref().map(|s| s.p95),
            "p99_ms": splade.stats.as_ref().map(|s| s.p99),
            "min_ms": splade.stats.as_ref().map(|s| s.min),
            "max_ms": splade.stats.as_ref().map(|s| s.max),
            "first_call_ms": extras.first_call_ms,
            "total_ms": splade.elapsed.as_secs_f64() * 1000.0,
        },
    });

    tokio::fs::write(path, serde_json::to_string_pretty(&report)?)
        .await
        .context("Failed to write JSON report")?;
    println!("Report written to: {}", path);
    Ok(())
}

/// Execute the sparse vector benchmark.
pub async fn execute(
    collection: &str,
    sample_size: usize,
    _query_count: usize,
    output_file: Option<String>,
) -> Result<()> {
    let samples = sample_files(collection, sample_size).await?;

    if samples.is_empty() {
        output::error(
            "No files found in the specified collection. Ensure the collection has indexed files.",
        );
        return Ok(());
    }

    println!("Sparse Vector Benchmark");
    println!("=======================");
    println!("Collection:  {}", collection);
    println!("Files found: {}", samples.len());
    println!();

    let config_bm25 = workspace_qdrant_core::EmbeddingConfig {
        sparse_vector_mode: "bm25".to_string(),
        ..Default::default()
    };
    let config_splade = workspace_qdrant_core::EmbeddingConfig {
        sparse_vector_mode: "splade".to_string(),
        ..Default::default()
    };

    let _gen_bm25 = workspace_qdrant_core::EmbeddingGenerator::new(config_bm25)
        .context("Failed to create BM25 embedding generator")?;
    let gen_splade = workspace_qdrant_core::EmbeddingGenerator::new(config_splade)
        .context("Failed to create SPLADE embedding generator")?;

    println!("Readable samples: {}", samples.len());
    println!();

    println!("Generating BM25 sparse vectors...");
    let bm25 = benchmark_bm25(&samples);

    println!("Generating SPLADE++ sparse vectors (first call downloads model)...");
    let (splade, extras) = match benchmark_splade(&samples, &gen_splade).await {
        Some(result) => result,
        None => return Ok(()),
    };

    print_comparison(&bm25, &splade, &extras);

    if let Some(path) = output_file {
        write_json_report(&path, collection, samples.len(), &bm25, &splade, &extras).await?;
    }

    Ok(())
}
