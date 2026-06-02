//! `wqm admin perf` — display pipeline performance statistics.
//!
//! Table and columnar templates per cli-feedback.md.
//! Data layer in `perf_data.rs`.

use std::collections::HashSet;

use anyhow::{Context, Result};
use serde::Serialize;
use tabled::Tabled;
use wqm_common::duration_fmt::fmt_approx_duration;

use crate::output::canvas;
use crate::output::columnar::ColumnarBuilder;
use crate::output::table::ColumnHints;
use crate::output::{self};

use super::perf_data::{
    apply_sort, fmt_thousands, fmt_thousands_f, format_avg_cell, parse_group_by, parse_sort,
    truncate_key, GroupStats, SortSpec,
};
use super::perf_queries::{
    build_tenant_name_map, dimension_label, query_grouped_stats, query_queue_depth,
    query_total_items, query_two_level_stats,
};

/// Row struct for the borderless table output.
#[derive(Tabled, Serialize)]
struct PerfRow {
    #[tabled(rename = "Group")]
    #[serde(rename = "group")]
    group: String,
    #[tabled(rename = "Count")]
    #[serde(rename = "count")]
    count: String,
    #[tabled(rename = "Avg (ms)")]
    #[serde(rename = "avg_ms")]
    avg_ms: String,
    #[tabled(rename = "P50 (ms)")]
    #[serde(rename = "p50_ms")]
    p50_ms: String,
    #[tabled(rename = "P95 (ms)")]
    #[serde(rename = "p95_ms")]
    p95_ms: String,
    #[tabled(rename = "P99 (ms)")]
    #[serde(rename = "p99_ms")]
    p99_ms: String,
}

impl ColumnHints for PerfRow {
    fn content_columns() -> &'static [usize] {
        &[0] // Group is content
    }

    fn numeric_columns() -> &'static [usize] {
        &[1, 2, 3, 4, 5] // Count, Avg, P50, P95, P99
    }
}

/// Execute the perf subcommand.
pub async fn execute(
    window_hours: f64,
    json: bool,
    group_by: Option<String>,
    sort: Option<String>,
    collection: Option<String>,
) -> Result<()> {
    let cutoff = window_to_cutoff(window_hours)?;
    let conn = open_timings_db(&collection)?;
    let coll_filter = collection.as_deref();

    let total_items = query_total_items(&conn, &cutoff, coll_filter);

    if total_items == 0 {
        output::info(format!(
            "No processing timings in the last {} hours.",
            window_hours
        ));
        return Ok(());
    }

    let queue_depth = query_queue_depth(&conn);
    let dimensions = parse_group_by(group_by.as_deref())?;
    let sort_spec = parse_sort(sort.as_deref())?;

    let tenant_names = build_tenant_name_map(&conn);
    let valid_tenants: HashSet<String> = tenant_names.keys().cloned().collect();

    dispatch_output(
        &conn,
        &cutoff,
        &dimensions,
        &sort_spec,
        &tenant_names,
        &valid_tenants,
        coll_filter,
        json,
        total_items,
        queue_depth,
        window_hours,
    )
}

/// Convert a `--window` value (hours, possibly fractional) into a SQLite
/// `datetime('now', ?)` modifier. Uses minute granularity so sub-hour windows
/// (e.g. `0.5`) are preserved instead of being truncated to `-0 hours` by an
/// integer cast, which would silently match zero rows. Rejects non-positive and
/// NaN inputs (the `!(x > 0.0)` form also catches NaN).
fn window_to_cutoff(window_hours: f64) -> Result<String> {
    if !(window_hours > 0.0) {
        anyhow::bail!("--window must be greater than 0 (got {})", window_hours);
    }
    let cutoff_minutes = (window_hours * 60.0).round().max(1.0) as i64;
    Ok(format!("-{} minutes", cutoff_minutes))
}

/// Open the database and validate the timings table exists.
fn open_timings_db(collection: &Option<String>) -> Result<rusqlite::Connection> {
    let db_path = crate::config::get_database_path().map_err(|e| anyhow::anyhow!("{}", e))?;

    if !db_path.exists() {
        anyhow::bail!("Database not found at {}", db_path.display());
    }

    let conn =
        rusqlite::Connection::open_with_flags(&db_path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
            .context("Failed to open state database")?;

    conn.execute_batch("PRAGMA busy_timeout=5000;")
        .context("Failed to set busy_timeout")?;

    let table_exists: bool = conn
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master \
             WHERE type='table' AND name='processing_timings')",
            [],
            |row| row.get(0),
        )
        .unwrap_or(false);

    if !table_exists {
        anyhow::bail!(
            "No processing_timings table found. Daemon may not have recorded any timings yet."
        );
    }

    if let Some(ref c) = collection {
        let valid = ["projects", "libraries", "rules", "scratchpad"];
        if !valid.contains(&c.as_str()) {
            anyhow::bail!("Unknown collection '{}'. Valid: {}", c, valid.join(", "));
        }
    }

    Ok(conn)
}

/// Dispatch to single-level or two-level output.
#[allow(clippy::too_many_arguments)]
fn dispatch_output(
    conn: &rusqlite::Connection,
    cutoff: &str,
    dimensions: &[String],
    sort_spec: &Option<SortSpec>,
    tenant_names: &std::collections::HashMap<String, String>,
    valid_tenants: &HashSet<String>,
    coll_filter: Option<&str>,
    json: bool,
    total_items: i64,
    queue_depth: i64,
    window_hours: f64,
) -> Result<()> {
    if dimensions.len() <= 1 {
        let dim = dimensions.first().map_or("phase", |d| d.as_str());
        let mut stats =
            query_grouped_stats(conn, cutoff, dim, tenant_names, valid_tenants, coll_filter)?;
        apply_sort(&mut stats, sort_spec);
        if json {
            print_json_grouped(&stats, total_items, queue_depth, window_hours);
        } else {
            print_table_grouped(&stats, total_items, queue_depth, window_hours);
        }
    } else {
        let mut stats = query_two_level_stats(
            conn,
            cutoff,
            &dimensions[0],
            &dimensions[1],
            tenant_names,
            valid_tenants,
            coll_filter,
        )?;
        for (_, sub) in &mut stats {
            apply_sort(sub, sort_spec);
        }
        let label1 = dimension_label(&dimensions[0]);
        if json {
            print_json_two_level(&stats, total_items, queue_depth, window_hours);
        } else {
            print_table_two_level(&stats, label1, total_items, queue_depth, window_hours);
        }
    }

    Ok(())
}

// ─── Row conversion ──────────────────────────────────────────────────────────

fn stats_to_row(s: &GroupStats) -> PerfRow {
    PerfRow {
        group: truncate_key(&s.group_key, 30),
        count: fmt_thousands(s.count),
        avg_ms: format_avg_cell(s.avg_ms, s.std_err, s.count),
        p50_ms: fmt_thousands_f(s.p50_ms),
        p95_ms: fmt_thousands_f(s.p95_ms),
        p99_ms: fmt_thousands_f(s.p99_ms),
    }
}

// ─── Table output ────────────────────────────────────────────────────────────

fn print_table_grouped(
    stats: &[GroupStats],
    total_items: i64,
    queue_depth: i64,
    window_hours: f64,
) {
    canvas::print_title(&format!("Pipeline Performance (Last {}h)", window_hours));
    canvas::print_blank();

    let rows: Vec<PerfRow> = stats.iter().map(stats_to_row).collect();
    output::print_table_auto(&rows);

    print_summary(total_items, queue_depth, window_hours);
}

fn print_table_two_level(
    stats: &[(String, Vec<GroupStats>)],
    label1: &str,
    total_items: i64,
    queue_depth: i64,
    window_hours: f64,
) {
    canvas::print_title(&format!("Pipeline Performance (Last {}h)", window_hours));
    canvas::print_blank();

    for (group_name, sub_stats) in stats {
        output::info(format!("{} {}", label1, group_name));
        let rows: Vec<PerfRow> = sub_stats.iter().map(stats_to_row).collect();
        output::print_table_auto(&rows);
        println!();
    }

    print_summary(total_items, queue_depth, window_hours);
}

fn print_summary(total_items: i64, queue_depth: i64, window_hours: f64) {
    let rate = total_items as f64 / window_hours;

    let mut builder = ColumnarBuilder::new()
        .section(Some("Summary"))
        .kv("Items Processed", fmt_thousands(total_items))
        .kv(
            "Processing Rate",
            format!("{} items/hour", fmt_thousands_f(rate)),
        )
        .kv("Queue Depth", fmt_thousands(queue_depth));

    if queue_depth > 0 && rate > 0.0 {
        let drain_secs = (queue_depth as f64 / rate) * 3600.0;
        builder = builder.kv("Est. Drain Time", fmt_approx_duration(drain_secs));
    }

    builder.render();
}

// ─── JSON output ─────────────────────────────────────────────────────────────

fn print_json_grouped(stats: &[GroupStats], total_items: i64, queue_depth: i64, window_hours: f64) {
    let groups: Vec<serde_json::Value> = stats.iter().map(group_to_json).collect();

    let rate = total_items as f64 / window_hours;
    let obj = serde_json::json!({
        "window_hours": window_hours,
        "items_processed": total_items,
        "processing_rate_per_hour": rate,
        "queue_depth": queue_depth,
        "groups": groups,
    });

    println!("{}", serde_json::to_string_pretty(&obj).unwrap());
}

fn print_json_two_level(
    stats: &[(String, Vec<GroupStats>)],
    total_items: i64,
    queue_depth: i64,
    window_hours: f64,
) {
    let groups: Vec<serde_json::Value> = stats
        .iter()
        .map(|(key, subs)| {
            let sub_groups: Vec<serde_json::Value> = subs.iter().map(group_to_json).collect();
            serde_json::json!({
                "group": key,
                "breakdown": sub_groups,
            })
        })
        .collect();

    let rate = total_items as f64 / window_hours;
    let obj = serde_json::json!({
        "window_hours": window_hours,
        "items_processed": total_items,
        "processing_rate_per_hour": rate,
        "queue_depth": queue_depth,
        "groups": groups,
    });

    println!("{}", serde_json::to_string_pretty(&obj).unwrap());
}

fn group_to_json(s: &GroupStats) -> serde_json::Value {
    serde_json::json!({
        "group": s.group_key,
        "count": s.count,
        "avg_ms": s.avg_ms,
        "avg_ms_margin": s.std_err,
        "p50_ms": s.p50_ms,
        "p95_ms": s.p95_ms,
        "p99_ms": s.p99_ms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_window_to_cutoff_fractional_preserved() {
        // 0.5h must not truncate to "-0 hours"; it becomes 30 minutes.
        assert_eq!(window_to_cutoff(0.5).unwrap(), "-30 minutes");
        assert_eq!(window_to_cutoff(24.0).unwrap(), "-1440 minutes");
        assert_eq!(window_to_cutoff(1.0).unwrap(), "-60 minutes");
        // Tiny positive window clamps to at least one minute.
        assert_eq!(window_to_cutoff(0.001).unwrap(), "-1 minutes");
    }

    #[test]
    fn test_window_to_cutoff_rejects_non_positive_and_nan() {
        assert!(window_to_cutoff(0.0).is_err());
        assert!(window_to_cutoff(-1.0).is_err());
        assert!(window_to_cutoff(f64::NAN).is_err());
    }

    #[test]
    fn test_stats_to_row_preserves_formatting() {
        let stats = GroupStats {
            group_key: "embed".to_string(),
            count: 6812,
            avg_ms: 5788.0,
            std_err: 121.0,
            p50_ms: 532.0,
            p95_ms: 22340.0,
            p99_ms: 45895.0,
        };
        let row = stats_to_row(&stats);
        assert_eq!(row.group, "embed");
        assert_eq!(row.count, "6'812");
        assert!(row.avg_ms.contains("5'788"));
        assert!(row.avg_ms.contains("121"));
        assert_eq!(row.p50_ms, "532");
        assert_eq!(row.p95_ms, "22'340");
        assert_eq!(row.p99_ms, "45'895");
    }
}
