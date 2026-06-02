//! Pipeline performance queries and statistics.
//!
//! SQLite queries for `wqm admin perf` — grouped stats, two-level breakdowns,
//! tenant name resolution, and basic statistical functions.

use std::collections::{HashMap, HashSet};

use anyhow::Result;

use super::perf_data::{should_skip_row, GroupStats};

// ─── Dimension helpers ───────────────────────────────────────────────────────

fn dimension_column(dim: &str) -> &'static str {
    match dim {
        "project" => "tenant_id",
        "phase" => "phase",
        "language" => "language",
        // `operation` is an alias for `op` (parse_group_by normalizes it; map
        // both here defensively).
        "op" | "operation" => "op",
        "collection" => "collection",
        "file_type" => "file_type",
        "embedding_engine" => "embedding_engine",
        _ => "phase",
    }
}

pub fn dimension_label(dim: &str) -> &'static str {
    match dim {
        "project" => "Project",
        "phase" => "Phase",
        "language" => "Language",
        "op" | "operation" => "Operation",
        "collection" => "Collection",
        "file_type" => "File Type",
        "embedding_engine" => "Embedding Engine",
        _ => "Group",
    }
}

// ─── Tenant name resolution ──────────────────────────────────────────────────

/// Build a tenant_id -> project_name mapping from watch_folders.
pub fn build_tenant_name_map(conn: &rusqlite::Connection) -> HashMap<String, String> {
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

// ─── Parameterized query helper ──────────────────────────────────────────────

fn query_scalar_params(conn: &rusqlite::Connection, sql: &str, params: &[String]) -> Option<i64> {
    let mut stmt = conn.prepare(sql).ok()?;
    let result = match params.len() {
        1 => stmt.query_row(rusqlite::params![&params[0]], |r| r.get(0)),
        2 => stmt.query_row(rusqlite::params![&params[0], &params[1]], |r| r.get(0)),
        _ => return None,
    };
    result.ok()
}

fn collection_clause(coll: Option<&str>, param_idx: u32) -> (String, Vec<String>) {
    match coll {
        Some(c) => (
            format!(" AND collection = ?{}", param_idx),
            vec![c.to_string()],
        ),
        None => (String::new(), vec![]),
    }
}

// ─── Query functions ─────────────────────────────────────────────────────────

/// Query total items processed in the time window.
pub fn query_total_items(
    conn: &rusqlite::Connection,
    cutoff: &str,
    coll_filter: Option<&str>,
) -> i64 {
    let (sql, params) = if let Some(c) = coll_filter {
        (
            "SELECT COUNT(DISTINCT queue_id) FROM processing_timings \
             WHERE created_at > datetime('now', ?1) AND collection = ?2"
                .to_string(),
            vec![cutoff.to_string(), c.to_string()],
        )
    } else {
        (
            "SELECT COUNT(DISTINCT queue_id) FROM processing_timings \
             WHERE created_at > datetime('now', ?1)"
                .to_string(),
            vec![cutoff.to_string()],
        )
    };
    query_scalar_params(conn, &sql, &params).unwrap_or(0)
}

/// Query current queue depth.
pub fn query_queue_depth(conn: &rusqlite::Connection) -> i64 {
    conn.query_row("SELECT COUNT(*) FROM unified_queue", [], |row| row.get(0))
        .unwrap_or(0)
}

pub fn query_grouped_stats(
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
         GROUP BY COALESCE({col}, '') ORDER BY grp"
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
    for (raw_key, _count) in &groups {
        if should_skip_row(dim, raw_key, valid_tenants) {
            continue;
        }
        let durations = fetch_sorted_durations(conn, cutoff, col, raw_key, coll_filter)?;
        let count = durations.len() as i64;
        let avg = average(&durations);
        results.push(GroupStats {
            group_key: resolve_group_key(dim, raw_key, tenant_names),
            count,
            avg_ms: avg,
            std_err: std_error(&durations),
            p50_ms: percentile(&durations, 50),
            p95_ms: percentile(&durations, 95),
            p99_ms: percentile(&durations, 99),
        });
    }

    Ok(results)
}

pub fn query_two_level_stats(
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
    let (coll_clause1, coll_params) = collection_clause(coll_filter, 2);
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

        let sub_groups = query_sub_groups(
            conn,
            cutoff,
            col1,
            raw_key1,
            col2,
            &coll_clause2,
            &coll_params,
        )?;

        let mut sub_stats = Vec::new();
        for (raw_key2, _count) in &sub_groups {
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
            let count = durations.len() as i64;
            let avg = average(&durations);
            sub_stats.push(GroupStats {
                group_key: resolve_group_key(dim2, raw_key2, tenant_names),
                count,
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

fn query_sub_groups(
    conn: &rusqlite::Connection,
    cutoff: &str,
    col1: &str,
    raw_key1: &str,
    col2: &str,
    coll_clause2: &str,
    coll_params: &[String],
) -> Result<Vec<(String, i64)>> {
    let sql2 = format!(
        "SELECT COALESCE({col2}, '') as grp2, COUNT(*) \
         FROM processing_timings \
         WHERE created_at > datetime('now', ?1) AND COALESCE({col1}, '') = ?2{coll_clause2} \
         GROUP BY COALESCE({col2}, '') ORDER BY grp2"
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
    Ok(sub_groups)
}

// ─── Duration fetchers ───────────────────────────────────────────────────────

fn fetch_sorted_durations(
    conn: &rusqlite::Connection,
    cutoff: &str,
    col: &str,
    value: &str,
    coll_filter: Option<&str>,
) -> Result<Vec<i64>> {
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

// ─── Statistics ──────────────────────────────────────────────────────────────

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

fn std_error(values: &[i64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return 0.0;
    }
    let mean = average(values);
    let variance = values
        .iter()
        .map(|&v| (v as f64 - mean).powi(2))
        .sum::<f64>()
        / (n as f64 - 1.0);
    variance.sqrt() / (n as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::dimension_column;

    /// Guards against `VALID_DIMENSIONS` (accepted by `parse_group_by`) and
    /// `dimension_column` (the SQL-column mapper) silently diverging: every
    /// accepted dimension except `phase` must map to a non-default column, so a
    /// dimension can never pass validation yet fall through to the `_ => "phase"`
    /// default and silently group by the wrong column.
    #[test]
    fn every_valid_dimension_maps_to_a_column() {
        for dim in super::super::perf_data::VALID_DIMENSIONS {
            if *dim == "phase" {
                continue;
            }
            assert_ne!(
                dimension_column(dim),
                "phase",
                "dimension '{dim}' falls through to the default column in dimension_column"
            );
        }
    }
}
