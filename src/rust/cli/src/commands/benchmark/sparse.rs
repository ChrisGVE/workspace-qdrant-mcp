//! Sparse vector benchmark — BM25 vs SPLADE++ evaluation.
//!
//! Samples documents from SQLite tracked_files, generates both BM25 and SPLADE++
//! sparse vectors, and reports comparative metrics.

use anyhow::{Context, Result};
use std::time::Instant;

use crate::config::get_database_path;
use crate::output;

/// Execute the sparse vector benchmark.
pub async fn execute(
    collection: &str,
    sample_size: usize,
    _query_count: usize,
    output_file: Option<String>,
) -> Result<()> {
    let db_path = get_database_path()?;
    let conn = rusqlite::Connection::open(&db_path)
        .context("Failed to open database")?;

    // Sample file paths from tracked_files
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

    if file_paths.is_empty() {
        output::error("No files found in the specified collection. Ensure the collection has indexed files.");
        return Ok(());
    }

    println!("Sparse Vector Benchmark");
    println!("=======================");
    println!("Collection:  {}", collection);
    println!("Files found: {}", file_paths.len());
    println!();

    // Initialize embedding generators
    let config_bm25 = workspace_qdrant_core::EmbeddingConfig {
        sparse_vector_mode: "bm25".to_string(),
        ..Default::default()
    };
    let config_splade = workspace_qdrant_core::EmbeddingConfig {
        sparse_vector_mode: "splade".to_string(),
        ..Default::default()
    };

    // BM25 uses the standalone BM25 struct directly, not through the generator
    let _gen_bm25 = workspace_qdrant_core::EmbeddingGenerator::new(config_bm25)
        .context("Failed to create BM25 embedding generator")?;
    let gen_splade = workspace_qdrant_core::EmbeddingGenerator::new(config_splade)
        .context("Failed to create SPLADE embedding generator")?;

    // Read sample files and extract text snippets
    let mut samples: Vec<(String, String)> = Vec::new();
    for path in file_paths.iter().take(sample_size) {
        if let Ok(content) = tokio::fs::read_to_string(path).await {
            let snippet = content.chars().take(512).collect::<String>();
            if !snippet.trim().is_empty() {
                samples.push((path.clone(), snippet));
            }
        }
    }

    if samples.is_empty() {
        output::error("No readable files found. Ensure file paths in tracked_files are accessible.");
        return Ok(());
    }

    println!("Readable samples: {}", samples.len());
    println!();

    // Generate BM25 sparse vectors
    println!("Generating BM25 sparse vectors...");
    let bm25_start = Instant::now();
    let mut bm25_total_dims = 0usize;
    let mut bm25_count = 0usize;

    for (_, text) in &samples {
        let tokens = workspace_qdrant_core::embedding::tokenize_for_bm25(text);
        let mut bm25 = workspace_qdrant_core::BM25::new(1.2);
        bm25.add_document(&tokens);
        let sparse = bm25.generate_sparse_vector(&tokens);
        bm25_total_dims += sparse.indices.len();
        bm25_count += 1;
    }
    let bm25_elapsed = bm25_start.elapsed();

    // Generate SPLADE++ sparse vectors
    println!("Generating SPLADE++ sparse vectors (first call downloads model)...");
    let splade_start = Instant::now();
    let mut splade_total_dims = 0usize;
    let mut splade_count = 0usize;
    let mut splade_latencies: Vec<f64> = Vec::new();
    let mut splade_first_call_ms = 0f64;

    for (i, (_, text)) in samples.iter().enumerate() {
        let call_start = Instant::now();
        match gen_splade.generate_splade_sparse_vector(text).await {
            Ok(sparse) => {
                let call_ms = call_start.elapsed().as_secs_f64() * 1000.0;
                if i == 0 {
                    splade_first_call_ms = call_ms;
                } else {
                    splade_latencies.push(call_ms);
                }
                splade_total_dims += sparse.indices.len();
                splade_count += 1;
            }
            Err(e) => {
                if i == 0 {
                    output::error(format!("SPLADE++ model initialization failed: {}. Benchmark aborted.", e));
                    return Ok(());
                }
                eprintln!("  Warning: SPLADE++ failed for sample {}: {}", i, e);
            }
        }
    }
    let splade_elapsed = splade_start.elapsed();

    // Compute stats
    let bm25_avg_dims = if bm25_count > 0 { bm25_total_dims as f64 / bm25_count as f64 } else { 0.0 };
    let splade_avg_dims = if splade_count > 0 { splade_total_dims as f64 / splade_count as f64 } else { 0.0 };
    let bm25_avg_ms = if bm25_count > 0 { bm25_elapsed.as_secs_f64() * 1000.0 / bm25_count as f64 } else { 0.0 };

    splade_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let splade_avg_ms = if !splade_latencies.is_empty() {
        splade_latencies.iter().sum::<f64>() / splade_latencies.len() as f64
    } else {
        0.0
    };
    let splade_p95_ms = if !splade_latencies.is_empty() {
        let p95_idx = (splade_latencies.len() as f64 * 0.95) as usize;
        splade_latencies[p95_idx.min(splade_latencies.len() - 1)]
    } else {
        0.0
    };

    // Print results
    println!();
    println!("Results");
    println!("-------");
    println!("{:<25} {:>12} {:>12}", "", "BM25", "SPLADE++");
    println!("{:<25} {:>12} {:>12}", "---", "----", "--------");
    println!("{:<25} {:>12} {:>12}", "Vectors generated", bm25_count, splade_count);
    println!("{:<25} {:>12.1} {:>12.1}", "Avg non-zero dims", bm25_avg_dims, splade_avg_dims);
    println!("{:<25} {:>10.2}ms {:>10.2}ms", "Avg latency (excl init)", bm25_avg_ms, splade_avg_ms);
    println!("{:<25} {:>12} {:>10.2}ms", "P95 latency", "n/a", splade_p95_ms);
    println!("{:<25} {:>12} {:>10.0}ms", "First call (incl init)", "~0", splade_first_call_ms);
    println!("{:<25} {:>10.0}ms {:>10.0}ms", "Total time",
        bm25_elapsed.as_secs_f64() * 1000.0,
        splade_elapsed.as_secs_f64() * 1000.0,
    );
    println!();
    println!("Notes:");
    println!("  - SPLADE++ first call includes model download (~150MB) + initialization");
    println!("  - SPLADE++ uses BERT vocab (30522 tokens); BM25 uses dynamic corpus vocab");
    println!("  - Higher avg non-zero dims = more expressive sparse representation");

    // Write JSON report if requested
    if let Some(path) = output_file {
        let report = serde_json::json!({
            "collection": collection,
            "sample_size": samples.len(),
            "bm25": {
                "vectors_generated": bm25_count,
                "avg_nonzero_dims": bm25_avg_dims,
                "avg_latency_ms": bm25_avg_ms,
                "total_ms": bm25_elapsed.as_secs_f64() * 1000.0,
            },
            "splade": {
                "vectors_generated": splade_count,
                "avg_nonzero_dims": splade_avg_dims,
                "avg_latency_ms": splade_avg_ms,
                "p95_latency_ms": splade_p95_ms,
                "first_call_ms": splade_first_call_ms,
                "total_ms": splade_elapsed.as_secs_f64() * 1000.0,
            },
        });

        tokio::fs::write(&path, serde_json::to_string_pretty(&report)?)
            .await
            .context("Failed to write JSON report")?;
        println!("Report written to: {}", path);
    }

    Ok(())
}
