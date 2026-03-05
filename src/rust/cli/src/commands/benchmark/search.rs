//! Search benchmark — FTS5 search DB vs ripgrep (rg).
//!
//! Runs a set of representative queries through both the FTS5 index
//! (search.db) and ripgrep, comparing latency, result count, and overlap.
//! Reports median, mean, std dev, min, max for each query.

use anyhow::{Context, Result};
use std::collections::HashSet;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use workspace_qdrant_core::search_db::SearchDbManager;
use workspace_qdrant_core::text_search::{self, SearchOptions};

use super::stats::LatencyStats;
use crate::config::get_database_path;
use crate::output;

/// A single benchmark query with its search parameters.
struct BenchQuery {
    label: &'static str,
    pattern: &'static str,
    regex: bool,
}

/// Results from a single query run on one engine.
struct EngineResult {
    latency_ms: f64,
    match_count: usize,
    file_paths: HashSet<String>,
}

/// Aggregated comparison for a single query.
struct QueryComparison {
    label: String,
    pattern: String,
    fts5_stats: LatencyStats,
    rg_stats: LatencyStats,
    fts5_match_count: usize,
    rg_match_count: usize,
    fts5_files: HashSet<String>,
    rg_files: HashSet<String>,
    shared_files: usize,
    fts5_only_files: usize,
    rg_only_files: usize,
}

/// Default query set covering typical code search patterns.
static DEFAULT_QUERIES: &[BenchQuery] = &[
    BenchQuery {
        label: "exact: common keyword",
        pattern: "async fn",
        regex: false,
    },
    BenchQuery {
        label: "exact: struct name",
        pattern: "ProcessingContext",
        regex: false,
    },
    BenchQuery {
        label: "exact: import path",
        pattern: "use std::collections::HashMap",
        regex: false,
    },
    BenchQuery {
        label: "exact: error handling",
        pattern: "anyhow::Result",
        regex: false,
    },
    BenchQuery {
        label: "exact: rare symbol",
        pattern: "CentralityCache",
        regex: false,
    },
    BenchQuery {
        label: "exact: trait impl",
        pattern: "impl Default for",
        regex: false,
    },
    BenchQuery {
        label: "exact: test annotation",
        pattern: "#[cfg(test)]",
        regex: false,
    },
    BenchQuery {
        label: "exact: multi-word",
        pattern: "queue processor",
        regex: false,
    },
    BenchQuery {
        label: "regex: fn signature",
        pattern: r"pub async fn \w+\(",
        regex: true,
    },
    BenchQuery {
        label: "regex: struct definition",
        pattern: r"pub struct \w+ \{",
        regex: true,
    },
    BenchQuery {
        label: "regex: fn definition",
        pattern: r"fn \w+\(",
        regex: true,
    },
    BenchQuery {
        label: "regex: mutable binding",
        pattern: r"let mut \w+",
        regex: true,
    },
    BenchQuery {
        label: "regex: trait impl",
        pattern: r"impl \w+ for \w+",
        regex: true,
    },
    BenchQuery {
        label: "regex: std imports",
        pattern: r"use (std|tokio|serde)::\w+",
        regex: true,
    },
    BenchQuery {
        label: "regex: derive macros",
        pattern: r"#\[derive\(\w+",
        regex: true,
    },
    BenchQuery {
        label: "regex: public decls",
        pattern: r"pub (fn|struct|enum|trait|type) \w+",
        regex: true,
    },
    BenchQuery {
        label: "regex: async Result",
        pattern: r"async fn \w+.*-> Result",
        regex: true,
    },
    BenchQuery {
        label: "regex: method chains",
        pattern: r"\.(await|unwrap|expect)\b",
        regex: true,
    },
];

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
    println!("Project root: {}", project_root.display());
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

/// Run a query through FTS5 search.db.
async fn run_fts5_query(
    db: &SearchDbManager,
    query: &BenchQuery,
    tenant_id: Option<&str>,
) -> Result<EngineResult> {
    let options = SearchOptions {
        tenant_id: tenant_id.map(|s| s.to_string()),
        max_results: 1000,
        ..Default::default()
    };

    let start = Instant::now();
    let results = if query.regex {
        text_search::search_regex(db, query.pattern, &options).await
    } else {
        text_search::search_exact(db, query.pattern, &options).await
    };
    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

    match results {
        Ok(r) => {
            let file_paths: HashSet<String> =
                r.matches.iter().map(|m| m.file_path.clone()).collect();
            Ok(EngineResult {
                latency_ms,
                match_count: r.matches.len(),
                file_paths,
            })
        }
        Err(e) => {
            eprintln!("  FTS5 error for '{}': {}", query.pattern, e);
            Ok(EngineResult {
                latency_ms,
                match_count: 0,
                file_paths: HashSet::new(),
            })
        }
    }
}

/// Run a query through ripgrep.
fn run_rg_query(project_root: &PathBuf, query: &BenchQuery) -> Result<EngineResult> {
    let start = Instant::now();

    let mut cmd = Command::new("rg");
    cmd.arg("--no-heading")
        .arg("--with-filename")
        .arg("--line-number")
        .arg("--max-count=1000")
        .arg("--color=never");

    if !query.regex {
        cmd.arg("--fixed-strings");
    }

    cmd.arg(query.pattern).arg(project_root);

    let output = cmd.output().context("Failed to execute rg")?;
    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut file_paths = HashSet::new();
    let mut match_count = 0;

    for line in stdout.lines() {
        if line.is_empty() {
            continue;
        }
        match_count += 1;
        if let Some(colon_pos) = line.find(':') {
            file_paths.insert(line[..colon_pos].to_string());
        }
    }

    Ok(EngineResult {
        latency_ms,
        match_count,
        file_paths,
    })
}

/// Print formatted results table with statistics.
fn print_results(comparisons: &[QueryComparison], iterations: usize) {
    println!();
    println!("Results (n={} iterations per query)", iterations);
    println!("{}", "─".repeat(120));
    println!(
        "{:<28} {:>8} {:>8} {:>8} {:>7} {:>7} {:>8} {:>8} {:>8} {:>8}",
        "Query",
        "FTS5 p50",
        "rg p50",
        "Ratio",
        "FTS5 #",
        "rg #",
        "Shared",
        "stdF",
        "stdR",
        "min/max"
    );
    println!("{}", "─".repeat(120));

    let mut total_fts5_ms = 0.0;
    let mut total_rg_ms = 0.0;

    for c in comparisons {
        let ratio = if c.rg_stats.median > 0.0 {
            c.fts5_stats.median / c.rg_stats.median
        } else {
            f64::INFINITY
        };

        total_fts5_ms += c.fts5_stats.median;
        total_rg_ms += c.rg_stats.median;

        let ratio_str = if ratio < 1.0 {
            format!("{:.2}x", ratio)
        } else {
            format!("{:.1}x", ratio)
        };

        let minmax = format!("{:.0}/{:.0}", c.fts5_stats.min, c.fts5_stats.max,);

        println!(
            "{:<28} {:>7.1} {:>7.1} {:>8} {:>7} {:>7} {:>8} {:>7.1} {:>7.1} {:>8}",
            truncate_label(&c.label, 27),
            c.fts5_stats.median,
            c.rg_stats.median,
            ratio_str,
            c.fts5_match_count,
            c.rg_match_count,
            c.shared_files,
            c.fts5_stats.std_dev,
            c.rg_stats.std_dev,
            minmax,
        );
    }

    println!("{}", "─".repeat(120));

    let total_ratio = if total_rg_ms > 0.0 {
        total_fts5_ms / total_rg_ms
    } else {
        0.0
    };
    println!(
        "{:<28} {:>7.1} {:>7.1} {:>7.2}x",
        "TOTAL", total_fts5_ms, total_rg_ms, total_ratio,
    );

    println!();
    println!("Legend:");
    println!("  FTS5 p50 = FTS5 median latency (ms)");
    println!("  rg p50   = ripgrep median latency (ms)");
    println!("  Ratio    = FTS5/rg (< 1.0 = FTS5 faster)");
    println!("  stdF/R   = standard deviation for FTS5/rg");
    println!("  min/max  = FTS5 min/max latency (ms)");
}

/// Write a JSON report to disk with full statistics.
async fn write_json_report(
    path: &str,
    comparisons: &[QueryComparison],
    project_root: &PathBuf,
    iterations: usize,
) -> Result<()> {
    let queries: Vec<serde_json::Value> = comparisons
        .iter()
        .map(|c| {
            let ratio = if c.rg_stats.median > 0.0 {
                c.fts5_stats.median / c.rg_stats.median
            } else {
                0.0
            };
            serde_json::json!({
                "label": c.label,
                "pattern": c.pattern,
                "fts5": {
                    "median_ms": c.fts5_stats.median,
                    "mean_ms": c.fts5_stats.mean,
                    "std_dev_ms": c.fts5_stats.std_dev,
                    "min_ms": c.fts5_stats.min,
                    "max_ms": c.fts5_stats.max,
                    "p95_ms": c.fts5_stats.p95,
                    "p99_ms": c.fts5_stats.p99,
                    "match_count": c.fts5_match_count,
                    "unique_files": c.fts5_files.len(),
                },
                "rg": {
                    "median_ms": c.rg_stats.median,
                    "mean_ms": c.rg_stats.mean,
                    "std_dev_ms": c.rg_stats.std_dev,
                    "min_ms": c.rg_stats.min,
                    "max_ms": c.rg_stats.max,
                    "p95_ms": c.rg_stats.p95,
                    "p99_ms": c.rg_stats.p99,
                    "match_count": c.rg_match_count,
                    "unique_files": c.rg_files.len(),
                },
                "comparison": {
                    "latency_ratio": ratio,
                    "shared_files": c.shared_files,
                    "fts5_only_files": c.fts5_only_files,
                    "rg_only_files": c.rg_only_files,
                },
            })
        })
        .collect();

    let report = serde_json::json!({
        "benchmark": "search",
        "project_root": project_root.display().to_string(),
        "iterations": iterations,
        "queries": queries,
    });

    tokio::fs::write(path, serde_json::to_string_pretty(&report)?)
        .await
        .context("Failed to write JSON report")?;
    Ok(())
}

/// Resolve the project root path from state.db watch_folders.
fn resolve_project_root(db_path: &PathBuf, tenant_id: Option<&str>) -> Result<PathBuf> {
    let conn = rusqlite::Connection::open(db_path).context("Failed to open state.db")?;

    let query = if let Some(tid) = tenant_id {
        let path: String = conn
            .query_row(
                "SELECT path FROM watch_folders WHERE tenant_id = ?1 AND collection = 'projects' LIMIT 1",
                rusqlite::params![tid],
                |row| row.get(0),
            )
            .context(format!("No project found for tenant_id '{}'", tid))?;
        path
    } else {
        let path: String = conn
            .query_row(
                "SELECT path FROM watch_folders WHERE collection = 'projects' ORDER BY is_active DESC LIMIT 1",
                [],
                |row| row.get(0),
            )
            .context("No projects found in watch_folders")?;
        path
    };

    Ok(PathBuf::from(query))
}

/// Check if ripgrep is available in PATH.
fn check_rg_available() -> bool {
    Command::new("rg")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Truncate a label to fit column width.
fn truncate_label(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max - 3])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_short() {
        assert_eq!(truncate_label("hello", 10), "hello");
    }

    #[test]
    fn test_truncate_long() {
        assert_eq!(truncate_label("hello world foo", 10), "hello w...");
    }

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
