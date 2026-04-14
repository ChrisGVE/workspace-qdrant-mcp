//! Metrics history subcommand.
//!
//! Columnar template per cli-feedback.md.

use anyhow::{Context, Result};
use rusqlite::Connection;
use wqm_common::timestamps;

use crate::config::get_database_path_checked;
use crate::output::canvas;
use crate::output::columnar::ColumnarBuilder;
use crate::output::number::{format_float, NumberLocale};

/// Show historical metrics for the given time range.
pub async fn history(range: &str) -> Result<()> {
    let seconds = parse_range_to_seconds(range);
    let conn = connect_history_readonly()?;

    let table_exists: bool = conn
        .query_row(
            "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='table' AND name='metrics_history'",
            [],
            |row| row.get(0),
        )
        .unwrap_or(false);

    if !table_exists {
        canvas::print_title(&format!("Metrics History ({})", range));
        canvas::print_blank();

        ColumnarBuilder::new()
            .kv("Status", "metrics history table not found")
            .kv("Required", "daemon with schema v5+")
            .render();
        return Ok(());
    }

    let cutoff = chrono::Utc::now() - chrono::Duration::seconds(seconds);
    let cutoff_str = timestamps::format_utc(&cutoff);

    let metric_summaries = query_metric_summaries(&conn, &cutoff_str)?;

    canvas::print_title(&format!("Metrics History ({})", range));
    canvas::print_blank();

    if metric_summaries.is_empty() {
        ColumnarBuilder::new()
            .kv("Status", "no metrics in requested time range")
            .kv("Note", "daemon collects metrics every 60 seconds")
            .render();
        return Ok(());
    }

    let locale = NumberLocale::default();
    let mut builder = ColumnarBuilder::new();

    for summary in &metric_summaries {
        builder = builder
            .section(Some(&summary.name))
            .kv("Latest", format_float(summary.latest, 1, &locale))
            .kv("Average", format_float(summary.avg, 1, &locale))
            .kv("Min", format_float(summary.min, 1, &locale))
            .kv("Max", format_float(summary.max, 1, &locale))
            .kv("Samples", summary.count.to_string());
    }

    builder.render();

    canvas::print_footnote(&format!(
        "{} metrics tracked over {}",
        metric_summaries.len(),
        range
    ));

    Ok(())
}

struct MetricSummary {
    name: String,
    avg: f64,
    min: f64,
    max: f64,
    count: i64,
    latest: f64,
}

fn query_metric_summaries(conn: &Connection, cutoff_str: &str) -> Result<Vec<MetricSummary>> {
    let mut name_stmt = conn.prepare(
        "SELECT DISTINCT metric_name FROM metrics_history \
         WHERE timestamp >= ?1 AND aggregation_period = 'raw' \
         ORDER BY metric_name",
    )?;

    let metric_names: Vec<String> = name_stmt
        .query_map([cutoff_str], |row| row.get(0))?
        .filter_map(|r| r.ok())
        .collect();

    let mut summaries = Vec::new();
    for name in &metric_names {
        if let Ok(s) = query_single_metric(conn, name, cutoff_str) {
            summaries.push(s);
        }
    }
    Ok(summaries)
}

fn query_single_metric(
    conn: &Connection,
    name: &str,
    cutoff_str: &str,
) -> Result<MetricSummary, rusqlite::Error> {
    conn.query_row(
        "SELECT AVG(metric_value), MIN(metric_value), MAX(metric_value), \
         COUNT(*), \
         (SELECT metric_value FROM metrics_history \
          WHERE metric_name = ?1 AND timestamp >= ?2 AND aggregation_period = 'raw' \
          ORDER BY timestamp DESC LIMIT 1) \
         FROM metrics_history \
         WHERE metric_name = ?1 AND timestamp >= ?2 AND aggregation_period = 'raw'",
        rusqlite::params![name, cutoff_str],
        |row| {
            Ok(MetricSummary {
                name: name.to_string(),
                avg: row.get(0)?,
                min: row.get(1)?,
                max: row.get(2)?,
                count: row.get(3)?,
                latest: row.get(4)?,
            })
        },
    )
}

/// Parse range string (1h, 24h, 7d, 30d) to seconds.
pub fn parse_range_to_seconds(range: &str) -> i64 {
    let range = range.trim().to_lowercase();
    if let Some(hours) = range.strip_suffix('h') {
        hours.parse::<i64>().unwrap_or(1) * 3600
    } else if let Some(days) = range.strip_suffix('d') {
        days.parse::<i64>().unwrap_or(1) * 86400
    } else if let Some(minutes) = range.strip_suffix('m') {
        minutes.parse::<i64>().unwrap_or(60) * 60
    } else {
        3600 // default 1h
    }
}

fn connect_history_readonly() -> Result<Connection> {
    let db_path = get_database_path_checked().map_err(|e| anyhow::anyhow!("{}", e))?;

    let conn = Connection::open_with_flags(
        &db_path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .context(format!("Failed to open state database at {:?}", db_path))?;

    conn.execute_batch("PRAGMA busy_timeout=5000;")
        .context("Failed to set busy_timeout")?;

    Ok(conn)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_range_hours() {
        assert_eq!(parse_range_to_seconds("1h"), 3600);
        assert_eq!(parse_range_to_seconds("24h"), 86400);
        assert_eq!(parse_range_to_seconds("0h"), 0);
    }

    #[test]
    fn test_parse_range_days() {
        assert_eq!(parse_range_to_seconds("1d"), 86400);
        assert_eq!(parse_range_to_seconds("7d"), 604800);
    }

    #[test]
    fn test_parse_range_minutes() {
        assert_eq!(parse_range_to_seconds("5m"), 300);
        assert_eq!(parse_range_to_seconds("60m"), 3600);
    }

    #[test]
    fn test_parse_range_case_insensitive() {
        assert_eq!(parse_range_to_seconds("1H"), 3600);
        assert_eq!(parse_range_to_seconds("7D"), 604800);
    }

    #[test]
    fn test_parse_range_with_whitespace() {
        assert_eq!(parse_range_to_seconds(" 1h "), 3600);
    }

    #[test]
    fn test_parse_range_invalid_defaults_1h() {
        assert_eq!(parse_range_to_seconds("xyz"), 3600);
        assert_eq!(parse_range_to_seconds(""), 3600);
    }

    #[test]
    fn test_parse_range_invalid_number_with_valid_suffix() {
        assert_eq!(parse_range_to_seconds("abch"), 3600);
    }
}
