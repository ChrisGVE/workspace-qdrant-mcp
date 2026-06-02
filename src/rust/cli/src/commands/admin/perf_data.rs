//! Pipeline performance types, formatting, and parsing.
//!
//! Types and helpers for `wqm admin perf`. Query logic in `perf_queries.rs`.

use std::collections::HashSet;

use anyhow::Result;

/// Aggregate stats for a group of timing records.
pub struct GroupStats {
    pub group_key: String,
    pub count: i64,
    pub avg_ms: f64,
    pub std_err: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
}

/// Parsed sort specification.
pub struct SortSpec {
    pub column: String,
    pub descending: bool,
}

// ─── Formatting ──────────────────────────────────────────────────────────────

/// Format an integer with apostrophe thousand separators (e.g. 1'234'567).
pub fn fmt_thousands(n: i64) -> String {
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
pub fn fmt_thousands_f(v: f64) -> String {
    fmt_thousands(v.round() as i64)
}

/// Format avg with uncertainty as a single cell string.
pub fn format_avg_cell(avg: f64, std_err: f64, count: i64) -> String {
    if count < 2 {
        format!("{}~", fmt_thousands_f(avg))
    } else if std_err < 1.0 {
        fmt_thousands_f(avg)
    } else {
        format!(
            "{} \u{00b1} {}",
            fmt_thousands_f(avg),
            fmt_thousands_f(std_err)
        )
    }
}

/// Truncate a key string to max length with ellipsis.
pub fn truncate_key(key: &str, max_len: usize) -> String {
    if key.len() > max_len {
        format!("{}...", &key[..max_len.saturating_sub(3)])
    } else {
        key.to_string()
    }
}

// ─── Parsing ─────────────────────────────────────────────────────────────────

/// Valid grouping dimensions. `operation` is accepted as an alias for `op`
/// (normalized to `op` by [`parse_group_by`] before this list is consulted, so
/// it is intentionally not listed here).
pub(crate) const VALID_DIMENSIONS: &[&str] = &[
    "project",
    "phase",
    "language",
    "op",
    "collection",
    "file_type",
    "embedding_engine",
];

/// Parse and validate `--group-by` argument (comma-separated, max 2).
pub fn parse_group_by(input: Option<&str>) -> Result<Vec<String>> {
    let input = match input {
        Some(s) if !s.is_empty() => s,
        _ => return Ok(vec![]),
    };

    // Normalize the `operation` alias to the canonical `op` so dedup and the
    // dimension→column mapping stay consistent.
    let dims: Vec<String> = input
        .split(',')
        .map(|s| {
            let d = s.trim().to_lowercase();
            if d == "operation" {
                "op".to_string()
            } else {
                d
            }
        })
        .collect();

    if dims.len() > 2 {
        anyhow::bail!(
            "--group-by supports at most 2 dimensions (got {})",
            dims.len()
        );
    }

    for d in &dims {
        if !VALID_DIMENSIONS.contains(&d.as_str()) {
            anyhow::bail!(
                "Unknown dimension '{}'. Valid: {} (operation = alias of op)",
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
pub fn parse_sort(input: Option<&str>) -> Result<Option<SortSpec>> {
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
pub fn apply_sort(stats: &mut [GroupStats], sort_spec: &Option<SortSpec>) {
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

/// Returns true if this row should be skipped (unknown language, dead project).
pub fn should_skip_row(dim: &str, raw_key: &str, valid_tenants: &HashSet<String>) -> bool {
    if dim == "project" && !valid_tenants.contains(raw_key) {
        return true;
    }
    if dim == "language" && (raw_key.is_empty() || raw_key == "(unknown)") {
        return true;
    }
    // Drop rows with no value for the optional per-item dimensions (NULL on
    // older rows, deletes, and preamble updates) rather than render a blank key.
    if matches!(dim, "file_type" | "embedding_engine") && raw_key.is_empty() {
        return true;
    }
    false
}

// ─── Tests ───────────────────────────────────────────────────────────────────

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
    fn test_format_avg_cell_low_count() {
        let cell = format_avg_cell(42.0, 0.0, 1);
        assert!(cell.contains("42"), "low count value: {}", cell);
        assert!(cell.contains("~"), "low count should have ~: {}", cell);
    }

    #[test]
    fn test_format_avg_cell_low_stderr() {
        let cell = format_avg_cell(42.0, 0.3, 100);
        assert_eq!(cell, "42");
    }

    #[test]
    fn test_format_avg_cell_with_uncertainty() {
        let cell = format_avg_cell(3168.0, 340.0, 50);
        assert!(
            cell.contains("3'168"),
            "should have thousands sep: {}",
            cell
        );
        assert!(cell.contains('\u{00b1}'), "should contain +/-: {}", cell);
        assert!(cell.contains("340"), "should contain error: {}", cell);
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
    fn test_parse_group_by_new_dimensions() {
        // E3: collection, file_type, embedding_engine are valid breakdown dims.
        assert_eq!(
            parse_group_by(Some("collection")).unwrap(),
            vec!["collection"]
        );
        assert_eq!(
            parse_group_by(Some("file_type,embedding_engine")).unwrap(),
            vec!["file_type", "embedding_engine"]
        );
    }

    #[test]
    fn test_parse_group_by_operation_alias() {
        // `operation` normalizes to the canonical `op`.
        assert_eq!(parse_group_by(Some("operation")).unwrap(), vec!["op"]);
        // ...and so a operation+op pair collapses to a duplicate and is rejected.
        assert!(parse_group_by(Some("operation,op")).is_err());
    }

    #[test]
    fn test_parse_group_by_unknown_dimension() {
        assert!(parse_group_by(Some("file_size")).is_err());
    }

    #[test]
    fn test_should_skip_blank_optional_dims() {
        let valid = HashSet::new();
        assert!(should_skip_row("file_type", "", &valid));
        assert!(should_skip_row("embedding_engine", "", &valid));
        assert!(!should_skip_row("file_type", "code", &valid));
        assert!(!should_skip_row("embedding_engine", "fastembed", &valid));
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
