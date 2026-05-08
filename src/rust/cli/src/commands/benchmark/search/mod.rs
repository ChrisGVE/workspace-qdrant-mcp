//! Search benchmark — FTS5 search DB vs ripgrep (rg).
//!
//! Runs a set of representative queries through both the FTS5 index
//! (search.db) and ripgrep, comparing latency, result count, and overlap.
//! Reports median, mean, std dev, min, max for each query.

mod engines;
mod report;
mod types;

use anyhow::{Context, Result};
use std::path::PathBuf;

use workspace_qdrant_core::search_db::SearchDbManager;

use super::stats::LatencyStats;
use crate::config::get_database_path;
use crate::output;
use crate::output::style::home_to_tilde;

use engines::{check_rg_available, resolve_project_root, run_fts5_query, run_rg_query};
use report::{print_results, write_json_report};
use types::{BenchQuery, QueryComparison, DEFAULT_QUERIES};

/// Execute the search benchmark.
pub async fn execute(
    tenant_id: Option<String>,
    warmup: usize,
    iterations: usize,
    output_file: Option<String>,
) -> Result<()> {
    let db_path = get_database_path()?;
    let search_db_path = workspace_qdrant_core::search_db::search_db_path_from_state(&db_path);

    if !search_db_path.exists() {
        output::error(
            "search.db not found. Ensure the daemon has indexed files (FTS5 must be enabled).",
        );
        return Ok(());
    }

    let project_root = resolve_project_root(&db_path, tenant_id.as_deref())?;
    if !check_rg_available() {
        output::error("ripgrep (rg) not found in PATH. Install it to run the full benchmark.");
        return Ok(());
    }

    let search_db = SearchDbManager::new(&search_db_path)
        .await
        .context("Failed to open search.db")?;

    let queries = DEFAULT_QUERIES;
    print_benchmark_header(
        queries,
        &project_root,
        tenant_id.as_deref(),
        warmup,
        iterations,
    );

    run_warmup(
        &search_db,
        queries,
        &project_root,
        tenant_id.as_deref(),
        warmup,
    )
    .await;

    let comparisons = run_benchmark_queries(
        &search_db,
        queries,
        &project_root,
        tenant_id.as_deref(),
        iterations,
    )
    .await?;

    search_db.close().await;

    print_results(&comparisons, iterations);

    if let Some(path) = output_file {
        write_json_report(&path, &comparisons, &project_root, iterations).await?;
        println!("Report written to: {}", path);
    }

    Ok(())
}

/// Print the benchmark run header.
fn print_benchmark_header(
    queries: &[BenchQuery],
    project_root: &PathBuf,
    tenant_id: Option<&str>,
    warmup: usize,
    iterations: usize,
) {
    println!("Search Benchmark: FTS5 vs ripgrep");
    println!("=================================");
    println!(
        "Project root: {}",
        home_to_tilde(&project_root.display().to_string())
    );
    if let Some(tid) = tenant_id {
        println!("Tenant ID:    {}", tid);
    }
    println!(
        "Queries:      {} ({} exact, {} regex)",
        queries.len(),
        queries.iter().filter(|q| !q.regex).count(),
        queries.iter().filter(|q| q.regex).count(),
    );
    println!("Warmup:       {} iteration(s)", warmup);
    println!("Iterations:   {}", iterations);
    println!();
}

/// Run warmup passes over all queries to prime caches.
async fn run_warmup(
    search_db: &SearchDbManager,
    queries: &[BenchQuery],
    project_root: &PathBuf,
    tenant_id: Option<&str>,
    warmup: usize,
) {
    if warmup > 0 {
        println!("Warming up...");
        for _ in 0..warmup {
            for q in queries {
                let _ = run_fts5_query(search_db, q, tenant_id).await;
                let _ = run_rg_query(project_root, q);
            }
        }
    }
}

/// Run timed iterations for all queries and collect comparisons.
async fn run_benchmark_queries(
    search_db: &SearchDbManager,
    queries: &[BenchQuery],
    project_root: &PathBuf,
    tenant_id: Option<&str>,
    iterations: usize,
) -> Result<Vec<QueryComparison>> {
    println!("Running benchmark...");
    let mut comparisons: Vec<QueryComparison> = Vec::new();

    for q in queries {
        let mut fts5_latencies = Vec::new();
        let mut rg_latencies = Vec::new();
        let mut last_fts5 = None;
        let mut last_rg = None;

        for _ in 0..iterations {
            let fts5_result = run_fts5_query(search_db, q, tenant_id).await?;
            fts5_latencies.push(fts5_result.latency_ms);
            last_fts5 = Some(fts5_result);

            let rg_result = run_rg_query(project_root, q)?;
            rg_latencies.push(rg_result.latency_ms);
            last_rg = Some(rg_result);
        }

        let fts5 = last_fts5.unwrap();
        let rg = last_rg.unwrap();
        let fts5_stats = LatencyStats::from_latencies(&fts5_latencies).unwrap();
        let rg_stats = LatencyStats::from_latencies(&rg_latencies).unwrap();

        let shared = fts5.file_paths.intersection(&rg.file_paths).count();
        let fts5_only = fts5.file_paths.difference(&rg.file_paths).count();
        let rg_only = rg.file_paths.difference(&fts5.file_paths).count();

        comparisons.push(QueryComparison {
            label: q.label.to_string(),
            pattern: q.pattern.to_string(),
            fts5_stats,
            rg_stats,
            fts5_match_count: fts5.match_count,
            rg_match_count: rg.match_count,
            fts5_files: fts5.file_paths,
            rg_files: rg.file_paths,
            shared_files: shared,
            fts5_only_files: fts5_only,
            rg_only_files: rg_only,
        });
    }

    Ok(comparisons)
}

#[cfg(test)]
mod tests {
    use super::types::DEFAULT_QUERIES;

    #[test]
    fn test_default_queries_not_empty() {
        let queries = DEFAULT_QUERIES;
        assert!(!queries.is_empty());
        let exact = queries.iter().filter(|q| !q.regex).count();
        let regex = queries.iter().filter(|q| q.regex).count();
        assert!(exact >= 8, "should have at least 8 exact queries");
        assert!(regex >= 8, "should have at least 8 regex queries");
    }
}
