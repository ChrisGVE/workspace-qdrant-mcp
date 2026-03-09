//! `wqm admin perf` — display pipeline performance statistics
//!
//! Queries the `processing_timings` SQLite table to compute per-phase
//! aggregates (avg, p50, p95, p99) and throughput over a configurable window.
//! Supports `--group-by` for multi-dimensional breakdowns (project, phase,
//! language, op) with up to 2 grouping levels.

use std::collections::HashMap;

use anyhow::{Context, Result};

use crate::output;

/// Aggregate stats for a group of timing records.
struct GroupStats {
    /// Label for this group (e.g. phase name, project name, language).
    group_key: String,
    count: i64,
    avg_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
}

/// Valid grouping dimensions.
const VALID_DIMENSIONS: &[&str] = &["project", "phase", "language", "op"];

/// Execute the perf subcommand.
pub async fn execute(window_hours: f64, json: bool, group_by: Option<String>) -> Result<()> {
    let db_path = crate::config::get_database_path().map_err(|e| anyhow::anyhow!("{}", e))?;

    if !db_path.exists() {
        anyhow::bail!("Database not found at {}", db_path.display());
    }

    let conn =
        rusqlite::Connection::open_with_flags(&db_path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
            .context("Failed to open state database")?;

    // Check if the processing_timings table exists
    let table_exists: bool = conn
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master \
             WHERE type='table' AND name='processing_timings')",
            [],
            |row| row.get(0),
        )
        .unwrap_or(false);

    if !table_exists {
        output::warning(
            "No processing_timings table found. Daemon may not have recorded any timings yet.",
        );
        return Ok(());
    }

    let cutoff = format!("-{} hours", window_hours as i64);

    // Total items processed in window
    let total_items: i64 = conn
        .query_row(
            "SELECT COUNT(DISTINCT queue_id) FROM processing_timings \
             WHERE created_at > datetime('now', ?1)",
            rusqlite::params![&cutoff],
            |row| row.get(0),
        )
        .unwrap_or(0);

    if total_items == 0 {
        output::info(format!(
            "No processing timings in the last {} hours.",
            window_hours
        ));
        return Ok(());
    }

    // Queue depth (from unified_queue)
    let queue_depth: i64 = conn
        .query_row("SELECT COUNT(*) FROM unified_queue", [], |row| row.get(0))
        .unwrap_or(0);

    // Parse group-by dimensions
    let dimensions = parse_group_by(group_by.as_deref())?;

    // Build tenant→name mapping for human-readable project names
    let tenant_names = build_tenant_name_map(&conn);

    if dimensions.is_empty() {
        // Default view: group by phase
        let stats = query_grouped_stats(&conn, &cutoff, "phase", &tenant_names)?;
        if json {
            print_json_grouped(&stats, None, total_items, queue_depth, window_hours);
        } else {
            print_table_grouped(
                &stats,
                "Phase",
                None,
                total_items,
                queue_depth,
                window_hours,
            );
        }
    } else if dimensions.len() == 1 {
        let stats = query_grouped_stats(&conn, &cutoff, &dimensions[0], &tenant_names)?;
        let label = dimension_label(&dimensions[0]);
        if json {
            print_json_grouped(&stats, None, total_items, queue_depth, window_hours);
        } else {
            print_table_grouped(&stats, label, None, total_items, queue_depth, window_hours);
        }
    } else {
        // Two-level grouping
        let stats = query_two_level_stats(
            &conn,
            &cutoff,
            &dimensions[0],
            &dimensions[1],
            &tenant_names,
        )?;
        let label1 = dimension_label(&dimensions[0]);
        let label2 = dimension_label(&dimensions[1]);
        if json {
            print_json_two_level(&stats, total_items, queue_depth, window_hours);
        } else {
            print_table_two_level(
                &stats,
                label1,
                label2,
                total_items,
                queue_depth,
                window_hours,
            );
        }
    }

    Ok(())
}

/// Parse and validate `--group-by` argument (comma-separated, max 2).
fn parse_group_by(input: Option<&str>) -> Result<Vec<String>> {
    let input = match input {
        Some(s) if !s.is_empty() => s,
        _ => return Ok(vec![]),
    };

    let dims: Vec<String> = input.split(',').map(|s| s.trim().to_lowercase()).collect();

    if dims.len() > 2 {
        anyhow::bail!(
            "--group-by supports at most 2 dimensions (got {})",
            dims.len()
        );
    }

    for d in &dims {
        if !VALID_DIMENSIONS.contains(&d.as_str()) {
            anyhow::bail!(
                "Unknown dimension '{}'. Valid: {}",
                d,
                VALID_DIMENSIONS.join(", ")
            );
        }
    }

    if dims.len() == 2 && dims[0] == dims[1] {
        anyhow::bail!("Cannot group by the same dimension twice");
    }

    Ok(dims)
}

/// Map dimension name to SQL column.
fn dimension_column(dim: &str) -> &'static str {
    match dim {
        "project" => "tenant_id",
        "phase" => "phase",
        "language" => "language",
        "op" => "op",
        _ => "phase",
    }
}

/// Map dimension name to display label.
fn dimension_label(dim: &str) -> &'static str {
    match dim {
        "project" => "Project",
        "phase" => "Phase",
        "language" => "Language",
        "op" => "Operation",
        _ => "Group",
    }
}

/// Build a tenant_id → project_name mapping from watch_folders.
/// Falls back to the tenant_id itself if no name can be derived.
fn build_tenant_name_map(conn: &rusqlite::Connection) -> HashMap<String, String> {
    let mut map = HashMap::new();
    let mut name_count: HashMap<String, usize> = HashMap::new();

    // First pass: collect all tenant→path pairs and count name occurrences
    let mut entries: Vec<(String, String)> = Vec::new();
    if let Ok(mut stmt) = conn.prepare(
        "SELECT tenant_id, path FROM watch_folders \
         WHERE parent_watch_id IS NULL AND collection = 'projects'",
    ) {
        if let Ok(rows) = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        }) {
            for r in rows.flatten() {
                let name =
                    r.1.rsplit('/')
                        .find(|s| !s.is_empty())
                        .unwrap_or(&r.0)
                        .to_string();
                *name_count.entry(name.clone()).or_default() += 1;
                entries.push((r.0, name));
            }
        }
    }

    // Second pass: use name alone when unique, name (ID) when ambiguous
    for (tenant_id, name) in entries {
        let display = if name_count.get(&name).copied().unwrap_or(0) > 1 {
            format!("{} ({})", name, tenant_id)
        } else {
            name
        };
        map.insert(tenant_id, display);
    }

    map
}

/// Resolve a raw group key to a display name (handles tenant_id→project name).
fn resolve_group_key(dim: &str, raw: &str, tenant_names: &HashMap<String, String>) -> String {
    if dim == "project" {
        tenant_names
            .get(raw)
            .cloned()
            .unwrap_or_else(|| raw.to_string())
    } else if raw.is_empty() {
        "(unknown)".to_string()
    } else {
        raw.to_string()
    }
}

/// Query aggregate stats grouped by a single dimension.
fn query_grouped_stats(
    conn: &rusqlite::Connection,
    cutoff: &str,
    dim: &str,
    tenant_names: &HashMap<String, String>,
) -> Result<Vec<GroupStats>> {
    let col = dimension_column(dim);
    let sql = format!(
        "SELECT COALESCE({col}, '') as grp, COUNT(*) \
         FROM processing_timings \
         WHERE created_at > datetime('now', ?1) \
         GROUP BY grp ORDER BY grp"
    );

    let mut stmt = conn.prepare(&sql)?;
    let groups: Vec<(String, i64)> = stmt
        .query_map(rusqlite::params![cutoff], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
        })?
        .filter_map(|r| r.ok())
        .collect();

    let mut results = Vec::new();
    for (raw_key, count) in &groups {
        let durations = fetch_sorted_durations(conn, cutoff, col, raw_key)?;
        let avg = average(&durations);
        results.push(GroupStats {
            group_key: resolve_group_key(dim, raw_key, tenant_names),
            count: *count,
            avg_ms: avg,
            p50_ms: percentile(&durations, 50),
            p95_ms: percentile(&durations, 95),
            p99_ms: percentile(&durations, 99),
        });
    }

    Ok(results)
}

/// Query aggregate stats with two-level grouping.
fn query_two_level_stats(
    conn: &rusqlite::Connection,
    cutoff: &str,
    dim1: &str,
    dim2: &str,
    tenant_names: &HashMap<String, String>,
) -> Result<Vec<(String, Vec<GroupStats>)>> {
    let col1 = dimension_column(dim1);
    let col2 = dimension_column(dim2);

    // Get distinct values for the first dimension
    let sql1 = format!(
        "SELECT DISTINCT COALESCE({col1}, '') FROM processing_timings \
         WHERE created_at > datetime('now', ?1) ORDER BY 1"
    );
    let mut stmt1 = conn.prepare(&sql1)?;
    let level1_keys: Vec<String> = stmt1
        .query_map(rusqlite::params![cutoff], |row| row.get::<_, String>(0))?
        .filter_map(|r| r.ok())
        .collect();

    let mut results = Vec::new();

    for raw_key1 in &level1_keys {
        let display_key = resolve_group_key(dim1, raw_key1, tenant_names);

        // Query sub-groups for this level-1 key
        let sql2 = format!(
            "SELECT COALESCE({col2}, '') as grp2, COUNT(*) \
             FROM processing_timings \
             WHERE created_at > datetime('now', ?1) AND COALESCE({col1}, '') = ?2 \
             GROUP BY grp2 ORDER BY grp2"
        );
        let mut stmt2 = conn.prepare(&sql2)?;
        let sub_groups: Vec<(String, i64)> = stmt2
            .query_map(rusqlite::params![cutoff, raw_key1], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
            })?
            .filter_map(|r| r.ok())
            .collect();

        let mut sub_stats = Vec::new();
        for (raw_key2, count) in &sub_groups {
            let durations =
                fetch_sorted_durations_2d(conn, cutoff, col1, raw_key1, col2, raw_key2)?;
            let avg = average(&durations);
            sub_stats.push(GroupStats {
                group_key: resolve_group_key(dim2, raw_key2, tenant_names),
                count: *count,
                avg_ms: avg,
                p50_ms: percentile(&durations, 50),
                p95_ms: percentile(&durations, 95),
                p99_ms: percentile(&durations, 99),
            });
        }

        results.push((display_key, sub_stats));
    }

    Ok(results)
}

fn fetch_sorted_durations(
    conn: &rusqlite::Connection,
    cutoff: &str,
    col: &str,
    value: &str,
) -> Result<Vec<i64>> {
    let sql = format!(
        "SELECT duration_ms FROM processing_timings \
         WHERE created_at > datetime('now', ?1) AND COALESCE({col}, '') = ?2 \
         ORDER BY duration_ms"
    );
    let mut stmt = conn.prepare(&sql)?;
    let durations: Vec<i64> = stmt
        .query_map(rusqlite::params![cutoff, value], |row| row.get(0))?
        .filter_map(|r| r.ok())
        .collect();
    Ok(durations)
}

fn fetch_sorted_durations_2d(
    conn: &rusqlite::Connection,
    cutoff: &str,
    col1: &str,
    val1: &str,
    col2: &str,
    val2: &str,
) -> Result<Vec<i64>> {
    let sql = format!(
        "SELECT duration_ms FROM processing_timings \
         WHERE created_at > datetime('now', ?1) \
           AND COALESCE({col1}, '') = ?2 \
           AND COALESCE({col2}, '') = ?3 \
         ORDER BY duration_ms"
    );
    let mut stmt = conn.prepare(&sql)?;
    let durations: Vec<i64> = stmt
        .query_map(rusqlite::params![cutoff, val1, val2], |row| row.get(0))?
        .filter_map(|r| r.ok())
        .collect();
    Ok(durations)
}

fn percentile(sorted: &[i64], pct: u8) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((pct as f64 / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    let idx = idx.min(sorted.len() - 1);
    sorted[idx] as f64
}

fn average(values: &[i64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<i64>() as f64 / values.len() as f64
}

// === Table output ===

fn print_table_grouped(
    stats: &[GroupStats],
    label: &str,
    _label2: Option<&str>,
    total_items: i64,
    queue_depth: i64,
    window_hours: f64,
) {
    output::section(format!("Pipeline Performance (last {}h)", window_hours));

    let max_key_len = stats
        .iter()
        .map(|s| s.group_key.len())
        .max()
        .unwrap_or(10)
        .max(label.len())
        .min(30);

    println!(
        "  {:<width$} {:>8} {:>8} {:>8} {:>8} {:>8}",
        label,
        "Count",
        "Avg(ms)",
        "P50(ms)",
        "P95(ms)",
        "P99(ms)",
        width = max_key_len
    );
    println!("  {}", "-".repeat(max_key_len + 48));

    for s in stats {
        let key = if s.group_key.len() > 30 {
            format!("{}...", &s.group_key[..27])
        } else {
            s.group_key.clone()
        };
        println!(
            "  {:<width$} {:>8} {:>8.0} {:>8.0} {:>8.0} {:>8.0}",
            key,
            s.count,
            s.avg_ms,
            s.p50_ms,
            s.p95_ms,
            s.p99_ms,
            width = max_key_len
        );
    }

    print_summary(total_items, queue_depth, window_hours);
}

fn print_table_two_level(
    stats: &[(String, Vec<GroupStats>)],
    label1: &str,
    label2: &str,
    total_items: i64,
    queue_depth: i64,
    window_hours: f64,
) {
    output::section(format!("Pipeline Performance (last {}h)", window_hours));

    let max_key2_len = stats
        .iter()
        .flat_map(|(_, subs)| subs.iter().map(|s| s.group_key.len()))
        .max()
        .unwrap_or(10)
        .max(label2.len())
        .min(24);

    for (group_name, sub_stats) in stats {
        println!();
        println!("  {} {}", label1, group_name);
        println!(
            "    {:<width$} {:>8} {:>8} {:>8} {:>8} {:>8}",
            label2,
            "Count",
            "Avg(ms)",
            "P50(ms)",
            "P95(ms)",
            "P99(ms)",
            width = max_key2_len
        );
        println!("    {}", "-".repeat(max_key2_len + 48));

        for s in sub_stats {
            let key = if s.group_key.len() > 24 {
                format!("{}...", &s.group_key[..21])
            } else {
                s.group_key.clone()
            };
            println!(
                "    {:<width$} {:>8} {:>8.0} {:>8.0} {:>8.0} {:>8.0}",
                key,
                s.count,
                s.avg_ms,
                s.p50_ms,
                s.p95_ms,
                s.p99_ms,
                width = max_key2_len
            );
        }
    }

    print_summary(total_items, queue_depth, window_hours);
}

fn print_summary(total_items: i64, queue_depth: i64, window_hours: f64) {
    println!();
    output::kv("Items processed", &total_items.to_string());
    let rate = total_items as f64 / window_hours;
    output::kv("Processing rate", &format!("{:.0} items/hour", rate));
    output::kv("Queue depth", &queue_depth.to_string());

    if queue_depth > 0 && rate > 0.0 {
        let drain_minutes = (queue_depth as f64 / rate) * 60.0;
        output::kv("Est. drain time", &format!("{:.1} minutes", drain_minutes));
    }
}

// === JSON output ===

fn print_json_grouped(
    stats: &[GroupStats],
    _nested: Option<&str>,
    total_items: i64,
    queue_depth: i64,
    window_hours: f64,
) {
    let groups: Vec<serde_json::Value> = stats
        .iter()
        .map(|s| {
            serde_json::json!({
                "group": s.group_key,
                "count": s.count,
                "avg_ms": s.avg_ms,
                "p50_ms": s.p50_ms,
                "p95_ms": s.p95_ms,
                "p99_ms": s.p99_ms,
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

fn print_json_two_level(
    stats: &[(String, Vec<GroupStats>)],
    total_items: i64,
    queue_depth: i64,
    window_hours: f64,
) {
    let groups: Vec<serde_json::Value> = stats
        .iter()
        .map(|(key, subs)| {
            let sub_groups: Vec<serde_json::Value> = subs
                .iter()
                .map(|s| {
                    serde_json::json!({
                        "group": s.group_key,
                        "count": s.count,
                        "avg_ms": s.avg_ms,
                        "p50_ms": s.p50_ms,
                        "p95_ms": s.p95_ms,
                        "p99_ms": s.p99_ms,
                    })
                })
                .collect();
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
