//! Result printing and JSON report writing for the search benchmark.

use anyhow::{Context, Result};
use std::path::PathBuf;

use super::types::QueryComparison;

/// Print formatted results table with statistics.
pub(super) fn print_results(comparisons: &[QueryComparison], iterations: usize) {
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

        let minmax = format!("{:.0}/{:.0}", c.fts5_stats.min, c.fts5_stats.max);

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
pub(super) async fn write_json_report(
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

/// Truncate a label to fit column width.
pub(super) fn truncate_label(s: &str, max: usize) -> String {
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
}
