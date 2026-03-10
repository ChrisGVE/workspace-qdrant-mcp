//! `wqm admin perf` — display pipeline performance statistics
//!
//! Queries the `processing_timings` SQLite table to compute per-phase
//! aggregates (avg ± std_err, p50, p95, p99) and throughput over a
//! configurable window. Supports `--group-by` for multi-dimensional
//! breakdowns (project, phase, language, op) with up to 2 grouping levels.
//! Supports `--sort` for column-based sorting and `--collection` filtering.

use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result};

use crate::output;

/// Aggregate stats for a group of timing records.
struct GroupStats {
    group_key: String,
    count: i64,
    avg_ms: f64,
    std_err: f64,
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
}

/// Parsed sort specification.
struct SortSpec {
    column: String,
    descending: bool,
}

/// Valid grouping dimensions.
const VALID_DIMENSIONS: &[&str] = &["project", "phase", "language", "op"];

/// Execute the perf subcommand.
pub async fn execute(
    window_hours: f64,
    json: bool,
    group_by: Option<String>,
    sort: Option<String>,
    collection: Option<String>,
) -> Result<()> {
    let db_path = crate::config::get_database_path().map_err(|e| anyhow::anyhow!("{}", e))?;

    if !db_path.exists() {
        anyhow::bail!("Database not found at {}", db_path.display());
    }

    let conn =
        rusqlite::Connection::open_with_flags(&db_path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
            .context("Failed to open state database")?;

    conn.execute_batch("PRAGMA busy_timeout=5000;")
        .context("Failed to set busy_timeout")?;

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

    // Validate collection filter
    if let Some(ref c) = collection {
        let valid = ["projects", "libraries", "rules", "scratchpad"];
        if !valid.contains(&c.as_str()) {
            anyhow::bail!(
                "Unknown collection '{}'. Valid: {}",
                c,
                valid.join(", ")
            );
        }
    }

    let cutoff = format!("-{} hours", window_hours as i64);
    let coll_filter = collection.as_deref();

    // Total items processed in window
    let total_items: i64 = {
        let (sql, params) = if let Some(c) = coll_filter {
            (
                "SELECT COUNT(DISTINCT queue_id) FROM processing_timings \
                 WHERE created_at > datetime('now', ?1) AND collection = ?2"
                    .to_string(),
                vec![cutoff.clone(), c.to_string()],
            )
        } else {
            (
                "SELECT COUNT(DISTINCT queue_id) FROM processing_timings \
                 WHERE created_at > datetime('now', ?1)"
                    .to_string(),
                vec![cutoff.clone()],
            )
        };
        query_scalar_params(&conn, &sql, &params).unwrap_or(0)
    };

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

    // Parse group-by dimensions and sort spec
    let dimensions = parse_group_by(group_by.as_deref())?;
    let sort_spec = parse_sort(sort.as_deref())?;

    // Build tenant→name mapping (also gives us the set of valid tenant_ids)
    let tenant_names = build_tenant_name_map(&conn);
    let valid_tenants: HashSet<String> = tenant_names.keys().cloned().collect();

    if dimensions.is_empty() {
        let mut stats = query_grouped_stats(
            &conn,
            &cutoff,
            "phase",
            &tenant_names,
            &valid_tenants,
            coll_filter,
        )?;
        apply_sort(&mut stats, &sort_spec);
        if json {
            print_json_grouped(&stats, total_items, queue_depth, window_hours);
        } else {
            print_table_grouped(&stats, "Phase", total_items, queue_depth, window_hours);
        }
    } else if dimensions.len() == 1 {
        let mut stats = query_grouped_stats(
            &conn,
            &cutoff,
            &dimensions[0],
            &tenant_names,
            &valid_tenants,
            coll_filter,
        )?;
        apply_sort(&mut stats, &sort_spec);
        let label = dimension_label(&dimensions[0]);
        if json {
            print_json_grouped(&stats, total_items, queue_depth, window_hours);
        } else {
            print_table_grouped(&stats, label, total_items, queue_depth, window_hours);
        }
    } else {
        let mut stats = query_two_level_stats(
            &conn,
            &cutoff,
            &dimensions[0],
            &dimensions[1],
            &tenant_names,
            &valid_tenants,
            coll_filter,
        )?;
        for (_, sub) in &mut stats {
            apply_sort(sub, &sort_spec);
        }
        let label1 = dimension_label(&dimensions[0]);
        let label2 = dimension_label(&dimensions[1]);
        if json {
            print_json_two_level(&stats, total_items, queue_depth, window_hours);
        } else {
            print_table_two_level(&stats, label1, label2, total_items, queue_depth, window_hours);
        }
    }

    Ok(())
}

// === Formatting helpers ===

/// Format an integer with apostrophe thousand separators (e.g. 1'234'567).
fn fmt_thousands(n: i64) -> String {
    if n < 0 {
        return format!("-{}", fmt_thousands(-n));
    }
    let s = n.to_string();
    let bytes = s.as_bytes();
    let mut result = String::with_capacity(s.len() + s.len() / 3);
    for (i, &b) in bytes.iter().enumerate() {
        if i > 0 && (bytes.len() - i) % 3 == 0 {
            result.push('\'');
        }
        result.push(b as char);
    }
    result
}

/// Format a float as integer with apostrophe thousand separators.
fn fmt_thousands_f(v: f64) -> String {
    fmt_thousands(v.round() as i64)
}

// === Parsing helpers ===

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

/// Parse and validate `--sort` argument (format: "column:direction").
fn parse_sort(input: Option<&str>) -> Result<Option<SortSpec>> {
    let input = match input {
        Some(s) if !s.is_empty() => s,
        _ => return Ok(None),
    };

    let valid_columns = ["count", "avg_ms", "p50_ms", "p95_ms", "p99_ms"];

    let (col, desc) = if let Some((c, d)) = input.split_once(':') {
        let desc = match d.to_lowercase().as_str() {
            "desc" | "d" => true,
            "asc" | "a" => false,
            _ => anyhow::bail!("Invalid sort direction '{}'. Use asc or desc", d),
        };
        (c.to_lowercase(), desc)
    } else {
        (input.to_lowercase(), true)
    };

    if !valid_columns.contains(&col.as_str()) {
        anyhow::bail!(
            "Unknown sort column '{}'. Valid: {}",
            col,
            valid_columns.join(", ")
        );
    }

    Ok(Some(SortSpec {
        column: col,
        descending: desc,
    }))
}

/// Apply sort specification to grouped stats.
fn apply_sort(stats: &mut [GroupStats], sort_spec: &Option<SortSpec>) {
    let spec = match sort_spec {
        Some(s) => s,
        None => return,
    };

    let key_fn = |s: &GroupStats| -> f64 {
        match spec.column.as_str() {
            "count" => s.count as f64,
            "avg_ms" => s.avg_ms,
            "p50_ms" => s.p50_ms,
            "p95_ms" => s.p95_ms,
            "p99_ms" => s.p99_ms,
            _ => 0.0,
        }
    };

    stats.sort_by(|a, b| {
        let va = key_fn(a);
        let vb = key_fn(b);
        if spec.descending {
            vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal)
        } else {
            va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
        }
    });
}

// === Dimension helpers ===

fn dimension_column(dim: &str) -> &'static str {
    match dim {
        "project" => "tenant_id",
        "phase" => "phase",
        "language" => "language",
        "op" => "op",
        _ => "phase",
    }
}

fn dimension_label(dim: &str) -> &'static str {
    match dim {
        "project" => "Project",
        "phase" => "Phase",
        "language" => "Language",
        "op" => "Operation",
        _ => "Group",
    }
}

// === Tenant name resolution ===

/// Build a tenant_id → project_name mapping from watch_folders.
fn build_tenant_name_map(conn: &rusqlite::Connection) -> HashMap<String, String> {
    let mut map = HashMap::new();
    let mut name_count: HashMap<String, usize> = HashMap::new();

    let mut entries: Vec<(String, String)> = Vec::new();
    if let Ok(mut stmt) = conn.prepare(
        "SELECT tenant_id, path FROM watch_folders \
         WHERE parent_watch_id IS NULL AND collection = 'projects'",
    ) {
        if let Ok(rows) = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        }) {
            for r in rows.flatten() {
                let name = r
                    .1
                    .rsplit('/')
                    .find(|s| !s.is_empty())
                    .unwrap_or(&r.0)
                    .to_string();
                *name_count.entry(name.clone()).or_default() += 1;
                entries.push((r.0, name));
            }
        }
    }

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

/// Returns true if this row should be skipped (unknown language, dead project).
fn should_skip_row(
    dim: &str,
    raw_key: &str,
    valid_tenants: &HashSet<String>,
) -> bool {
    // Skip dead projects
    if dim == "project" && !valid_tenants.contains(raw_key) {
        return true;
    }
    // Skip unknown/empty language
    if dim == "language" && (raw_key.is_empty() || raw_key == "(unknown)") {
        return true;
    }
    false
}

// === Parameterized query helper ===

fn query_scalar_params(conn: &rusqlite::Connection, sql: &str, params: &[String]) -> Option<i64> {
    let mut stmt = conn.prepare(sql).ok()?;
    let result = match params.len() {
        1 => stmt.query_row(rusqlite::params![&params[0]], |r| r.get(0)),
        2 => stmt.query_row(rusqlite::params![&params[0], &params[1]], |r| r.get(0)),
        _ => return None,
    };
    result.ok()
}

/// Build the optional collection filter SQL fragment with the given param index.
fn collection_clause(coll: Option<&str>, param_idx: u32) -> (String, Vec<String>) {
    match coll {
        Some(c) => (
            format!(" AND collection = ?{}", param_idx),
            vec![c.to_string()],
        ),
        None => (String::new(), vec![]),
    }
}

// === Query functions ===

fn query_grouped_stats(
    conn: &rusqlite::Connection,
    cutoff: &str,
    dim: &str,
    tenant_names: &HashMap<String, String>,
    valid_tenants: &HashSet<String>,
    coll_filter: Option<&str>,
) -> Result<Vec<GroupStats>> {
    let col = dimension_column(dim);
    let (coll_clause, coll_params) = collection_clause(coll_filter, 2);

    let sql = format!(
        "SELECT COALESCE({col}, '') as grp, COUNT(*) \
         FROM processing_timings \
         WHERE created_at > datetime('now', ?1){coll_clause} \
         GROUP BY grp ORDER BY grp"
    );

    let mut stmt = conn.prepare(&sql)?;
    let groups: Vec<(String, i64)> = if coll_params.is_empty() {
        stmt.query_map(rusqlite::params![cutoff], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
        })?
        .filter_map(|r| r.ok())
        .collect()
    } else {
        stmt.query_map(rusqlite::params![cutoff, &coll_params[0]], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
        })?
        .filter_map(|r| r.ok())
        .collect()
    };

    let mut results = Vec::new();
    for (raw_key, count) in &groups {
        if should_skip_row(dim, raw_key, valid_tenants) {
            continue;
        }
        let durations = fetch_sorted_durations(conn, cutoff, col, raw_key, coll_filter)?;
        let avg = average(&durations);
        results.push(GroupStats {
            group_key: resolve_group_key(dim, raw_key, tenant_names),
            count: *count,
            avg_ms: avg,
            std_err: std_error(&durations),
            p50_ms: percentile(&durations, 50),
            p95_ms: percentile(&durations, 95),
            p99_ms: percentile(&durations, 99),
        });
    }

    Ok(results)
}

fn query_two_level_stats(
    conn: &rusqlite::Connection,
    cutoff: &str,
    dim1: &str,
    dim2: &str,
    tenant_names: &HashMap<String, String>,
    valid_tenants: &HashSet<String>,
    coll_filter: Option<&str>,
) -> Result<Vec<(String, Vec<GroupStats>)>> {
    let col1 = dimension_column(dim1);
    let col2 = dimension_column(dim2);
    // sql1 has 1 param (?1=cutoff), so collection is ?2
    let (coll_clause1, coll_params) = collection_clause(coll_filter, 2);
    // sql2 has 2 params (?1=cutoff, ?2=key1), so collection is ?3
    let (coll_clause2, _) = collection_clause(coll_filter, 3);

    let sql1 = format!(
        "SELECT DISTINCT COALESCE({col1}, '') FROM processing_timings \
         WHERE created_at > datetime('now', ?1){coll_clause1} ORDER BY 1"
    );
    let mut stmt1 = conn.prepare(&sql1)?;
    let level1_keys: Vec<String> = if coll_params.is_empty() {
        stmt1
            .query_map(rusqlite::params![cutoff], |row| row.get::<_, String>(0))?
            .filter_map(|r| r.ok())
            .collect()
    } else {
        stmt1
            .query_map(rusqlite::params![cutoff, &coll_params[0]], |row| {
                row.get::<_, String>(0)
            })?
            .filter_map(|r| r.ok())
            .collect()
    };

    let mut results = Vec::new();

    for raw_key1 in &level1_keys {
        if should_skip_row(dim1, raw_key1, valid_tenants) {
            continue;
        }

        let display_key = resolve_group_key(dim1, raw_key1, tenant_names);

        let sql2 = format!(
            "SELECT COALESCE({col2}, '') as grp2, COUNT(*) \
             FROM processing_timings \
             WHERE created_at > datetime('now', ?1) AND COALESCE({col1}, '') = ?2{coll_clause2} \
             GROUP BY grp2 ORDER BY grp2"
        );
        let mut stmt2 = conn.prepare(&sql2)?;
        let sub_groups: Vec<(String, i64)> = if coll_params.is_empty() {
            stmt2
                .query_map(rusqlite::params![cutoff, raw_key1], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
                })?
                .filter_map(|r| r.ok())
                .collect()
        } else {
            stmt2
                .query_map(
                    rusqlite::params![cutoff, raw_key1, &coll_params[0]],
                    |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?)),
                )?
                .filter_map(|r| r.ok())
                .collect()
        };

        let mut sub_stats = Vec::new();
        for (raw_key2, count) in &sub_groups {
            if should_skip_row(dim2, raw_key2, valid_tenants) {
                continue;
            }
            let durations = fetch_sorted_durations_2d(
                conn,
                cutoff,
                col1,
                raw_key1,
                col2,
                raw_key2,
                coll_filter,
            )?;
            let avg = average(&durations);
            sub_stats.push(GroupStats {
                group_key: resolve_group_key(dim2, raw_key2, tenant_names),
                count: *count,
                avg_ms: avg,
                std_err: std_error(&durations),
                p50_ms: percentile(&durations, 50),
                p95_ms: percentile(&durations, 95),
                p99_ms: percentile(&durations, 99),
            });
        }

        if !sub_stats.is_empty() {
            results.push((display_key, sub_stats));
        }
    }

    Ok(results)
}

// === Duration fetchers ===

fn fetch_sorted_durations(
    conn: &rusqlite::Connection,
    cutoff: &str,
    col: &str,
    value: &str,
    coll_filter: Option<&str>,
) -> Result<Vec<i64>> {
    // Params: ?1=cutoff, ?2=value, so collection is ?3
    let (coll_clause, coll_params) = collection_clause(coll_filter, 3);
    let sql = format!(
        "SELECT duration_ms FROM processing_timings \
         WHERE created_at > datetime('now', ?1) AND COALESCE({col}, '') = ?2{coll_clause} \
         ORDER BY duration_ms"
    );
    let mut stmt = conn.prepare(&sql)?;
    let durations: Vec<i64> = if coll_params.is_empty() {
        stmt.query_map(rusqlite::params![cutoff, value], |row| row.get(0))?
            .filter_map(|r| r.ok())
            .collect()
    } else {
        stmt.query_map(rusqlite::params![cutoff, value, &coll_params[0]], |row| {
            row.get(0)
        })?
        .filter_map(|r| r.ok())
        .collect()
    };
    Ok(durations)
}

fn fetch_sorted_durations_2d(
    conn: &rusqlite::Connection,
    cutoff: &str,
    col1: &str,
    val1: &str,
    col2: &str,
    val2: &str,
    coll_filter: Option<&str>,
) -> Result<Vec<i64>> {
    // Params: ?1=cutoff, ?2=val1, ?3=val2, so collection is ?4
    let (coll_clause, coll_params) = collection_clause(coll_filter, 4);
    let sql = format!(
        "SELECT duration_ms FROM processing_timings \
         WHERE created_at > datetime('now', ?1) \
           AND COALESCE({col1}, '') = ?2 \
           AND COALESCE({col2}, '') = ?3{coll_clause} \
         ORDER BY duration_ms"
    );
    let mut stmt = conn.prepare(&sql)?;
    let durations: Vec<i64> = if coll_params.is_empty() {
        stmt.query_map(rusqlite::params![cutoff, val1, val2], |row| row.get(0))?
            .filter_map(|r| r.ok())
            .collect()
    } else {
        stmt.query_map(
            rusqlite::params![cutoff, val1, val2, &coll_params[0]],
            |row| row.get(0),
        )?
        .filter_map(|r| r.ok())
        .collect()
    };
    Ok(durations)
}

// === Statistics ===

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

/// Standard error of the mean: std_dev / sqrt(n).
fn std_error(values: &[i64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return 0.0;
    }
    let mean = average(values);
    let variance =
        values.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
    variance.sqrt() / (n as f64).sqrt()
}

/// Split avg ± std_err into (value_str, Option<error_str>) for table alignment.
///
/// Returns the value and, when the error is meaningful, the error as separate
/// strings so the caller can right-align them in independent sub-columns.
fn avg_uncertainty_parts(avg: f64, std_err: f64, count: i64) -> (String, Option<String>) {
    if count < 2 {
        (format!("{}~", fmt_thousands_f(avg)), None)
    } else if std_err < 1.0 {
        (fmt_thousands_f(avg), None)
    } else {
        (fmt_thousands_f(avg), Some(fmt_thousands_f(std_err)))
    }
}

/// Render a pre-aligned avg cell as three logical sub-columns:
///   `<right-aligned value> ± <right-aligned error>`
///
/// When `err_width == 0` (no row in this table has an error term) the cell is
/// just the right-aligned value.  When the current row has no error but other
/// rows do, the ` ± ` separator is replaced by spaces so columns stay aligned.
fn format_avg_cols(
    parts: &(String, Option<String>),
    val_width: usize,
    err_width: usize,
) -> String {
    let (value, error) = parts;
    if err_width == 0 {
        format!("{:>vw$}", value, vw = val_width)
    } else {
        match error {
            Some(err) => format!(
                "{:>vw$} \u{00b1} {:>ew$}",
                value,
                err,
                vw = val_width,
                ew = err_width
            ),
            None => format!(
                "{:>vw$}   {:>ew$}",
                value,
                "",
                vw = val_width,
                ew = err_width
            ),
        }
    }
}

// === Table output ===

fn print_table_grouped(
    stats: &[GroupStats],
    label: &str,
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

    // Split avg into (value, Option<error>) for independent sub-column alignment
    let avg_parts: Vec<(String, Option<String>)> = stats
        .iter()
        .map(|s| avg_uncertainty_parts(s.avg_ms, s.std_err, s.count))
        .collect();
    let val_width = avg_parts
        .iter()
        .map(|(v, _)| v.len())
        .max()
        .unwrap_or(7)
        .max(7);
    let err_width = avg_parts
        .iter()
        .filter_map(|(_, e)| e.as_ref())
        .map(|e| e.len())
        .max()
        .unwrap_or(0);
    let avg_col_width = val_width + if err_width > 0 { 3 + err_width } else { 0 };

    println!(
        "  {:<width$} {:>8} {:>avg_w$} {:>8} {:>8} {:>8}",
        label,
        "Count",
        "Avg(ms)",
        "P50(ms)",
        "P95(ms)",
        "P99(ms)",
        width = max_key_len,
        avg_w = avg_col_width,
    );
    println!("  {}", "-".repeat(max_key_len + avg_col_width + 38));

    for (i, s) in stats.iter().enumerate() {
        let key = truncate_key(&s.group_key, 30);
        println!(
            "  {:<width$} {:>8} {} {:>8} {:>8} {:>8}",
            key,
            fmt_thousands(s.count),
            format_avg_cols(&avg_parts[i], val_width, err_width),
            fmt_thousands_f(s.p50_ms),
            fmt_thousands_f(s.p95_ms),
            fmt_thousands_f(s.p99_ms),
            width = max_key_len,
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
        // Split avg into (value, Option<error>) for independent sub-column alignment
        let avg_parts: Vec<(String, Option<String>)> = sub_stats
            .iter()
            .map(|s| avg_uncertainty_parts(s.avg_ms, s.std_err, s.count))
            .collect();
        let val_width = avg_parts
            .iter()
            .map(|(v, _)| v.len())
            .max()
            .unwrap_or(7)
            .max(7);
        let err_width = avg_parts
            .iter()
            .filter_map(|(_, e)| e.as_ref())
            .map(|e| e.len())
            .max()
            .unwrap_or(0);
        let avg_col_width = val_width + if err_width > 0 { 3 + err_width } else { 0 };

        println!();
        println!("  {} {}", label1, group_name);
        println!(
            "    {:<width$} {:>8} {:>avg_w$} {:>8} {:>8} {:>8}",
            label2,
            "Count",
            "Avg(ms)",
            "P50(ms)",
            "P95(ms)",
            "P99(ms)",
            width = max_key2_len,
            avg_w = avg_col_width,
        );
        println!("    {}", "-".repeat(max_key2_len + avg_col_width + 38));

        for (i, s) in sub_stats.iter().enumerate() {
            let key = truncate_key(&s.group_key, 24);
            println!(
                "    {:<width$} {:>8} {} {:>8} {:>8} {:>8}",
                key,
                fmt_thousands(s.count),
                format_avg_cols(&avg_parts[i], val_width, err_width),
                fmt_thousands_f(s.p50_ms),
                fmt_thousands_f(s.p95_ms),
                fmt_thousands_f(s.p99_ms),
                width = max_key2_len,
            );
        }
    }

    print_summary(total_items, queue_depth, window_hours);
}

fn truncate_key(key: &str, max_len: usize) -> String {
    if key.len() > max_len {
        format!("{}...", &key[..max_len.saturating_sub(3)])
    } else {
        key.to_string()
    }
}

fn print_summary(total_items: i64, queue_depth: i64, window_hours: f64) {
    println!();
    output::kv("Items processed", &fmt_thousands(total_items));
    let rate = total_items as f64 / window_hours;
    output::kv("Processing rate", &format!("{} items/hour", fmt_thousands_f(rate)));
    output::kv("Queue depth", &fmt_thousands(queue_depth));

    if queue_depth > 0 && rate > 0.0 {
        let drain_minutes = (queue_depth as f64 / rate) * 60.0;
        output::kv("Est. drain time", &format!("{:.1} minutes", drain_minutes));
    }
}

// === JSON output ===

fn print_json_grouped(
    stats: &[GroupStats],
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
                "avg_ms_margin": s.std_err,
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
                        "avg_ms_margin": s.std_err,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fmt_thousands_small() {
        assert_eq!(fmt_thousands(0), "0");
        assert_eq!(fmt_thousands(42), "42");
        assert_eq!(fmt_thousands(999), "999");
    }

    #[test]
    fn test_fmt_thousands_large() {
        assert_eq!(fmt_thousands(1000), "1'000");
        assert_eq!(fmt_thousands(12345), "12'345");
        assert_eq!(fmt_thousands(1234567), "1'234'567");
    }

    #[test]
    fn test_fmt_thousands_negative() {
        assert_eq!(fmt_thousands(-1234), "-1'234");
    }

    #[test]
    fn test_should_skip_unknown_language() {
        let valid = HashSet::new();
        assert!(should_skip_row("language", "", &valid));
        assert!(should_skip_row("language", "(unknown)", &valid));
        assert!(!should_skip_row("language", "rust", &valid));
    }

    #[test]
    fn test_should_skip_dead_project() {
        let mut valid = HashSet::new();
        valid.insert("t1".to_string());
        assert!(!should_skip_row("project", "t1", &valid));
        assert!(should_skip_row("project", "t_gone", &valid));
    }

    #[test]
    fn test_avg_parts_low_count() {
        let (val, err) = avg_uncertainty_parts(42.0, 0.0, 1);
        assert!(val.contains("42"), "low count value: {}", val);
        assert!(val.contains("~"), "low count should have ~: {}", val);
        assert!(err.is_none(), "low count should have no error term");
    }

    #[test]
    fn test_avg_parts_low_stderr() {
        let (val, err) = avg_uncertainty_parts(42.0, 0.3, 100);
        assert_eq!(val, "42");
        assert!(err.is_none(), "sub-1 stderr should produce no error term");
    }

    #[test]
    fn test_avg_parts_with_uncertainty() {
        let (val, err) = avg_uncertainty_parts(3168.0, 340.0, 50);
        assert!(val.contains("3'168"), "should have thousands sep: {}", val);
        assert_eq!(err.as_deref(), Some("340"), "error term: {:?}", err);
    }

    #[test]
    fn test_format_avg_cols_alignment() {
        // Both rows have an error term — three sub-columns
        let with_err = (String::from("1'448"), Some(String::from("33")));
        let no_err = (String::from("28"), None);
        let val_w = 5;
        let err_w = 2;
        let s_with = format_avg_cols(&with_err, val_w, err_w);
        let s_none = format_avg_cols(&no_err, val_w, err_w);
        // Both strings must be the same display width (compare char count, not bytes;
        // ± is 2 bytes but 1 display column).
        assert_eq!(
            s_with.chars().count(),
            s_none.chars().count(),
            "cells must have equal display width: {:?} vs {:?}",
            s_with,
            s_none
        );
        assert!(s_with.contains('±'), "error row should contain ±");
        assert!(!s_none.contains('±'), "no-error row must not contain ±");
    }

    #[test]
    fn test_parse_sort_valid() {
        let spec = parse_sort(Some("avg_ms:desc")).unwrap().unwrap();
        assert_eq!(spec.column, "avg_ms");
        assert!(spec.descending);
    }

    #[test]
    fn test_parse_sort_invalid_column() {
        assert!(parse_sort(Some("foo:desc")).is_err());
    }

    #[test]
    fn test_parse_group_by_valid() {
        let dims = parse_group_by(Some("project,phase")).unwrap();
        assert_eq!(dims, vec!["project", "phase"]);
    }

    #[test]
    fn test_parse_group_by_too_many() {
        assert!(parse_group_by(Some("project,phase,language")).is_err());
    }

    #[test]
    fn test_truncate_key() {
        assert_eq!(truncate_key("short", 30), "short");
        let long = "a".repeat(35);
        let truncated = truncate_key(&long, 30);
        assert!(truncated.len() <= 30);
        assert!(truncated.ends_with("..."));
    }
}
