//! Metrics history subcommand.

use anyhow::{Context, Result};
use rusqlite::Connection;
use wqm_common::timestamps;

use crate::config::get_database_path_checked;
use crate::output;

/// Show historical metrics for the given time range.
pub async fn history(range: &str) -> Result<()> {
    output::section(format!("Metrics History ({})", range));

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
        output::warning("Metrics history table not found. Daemon needs to run with schema v5+.");
        output::info("Start the daemon to enable metrics collection.");
        return Ok(());
    }

    let cutoff = chrono::Utc::now() - chrono::Duration::seconds(seconds);
    let cutoff_str = timestamps::format_utc(&cutoff);

    let mut stmt = conn.prepare(
        "SELECT DISTINCT metric_name FROM metrics_history \
         WHERE timestamp >= ?1 AND aggregation_period = 'raw' \
         ORDER BY metric_name",
    )?;

    let metric_names: Vec<String> = stmt
        .query_map([&cutoff_str], |row| row.get(0))?
        .filter_map(|r| r.ok())
        .collect();

    if metric_names.is_empty() {
        output::info("No historical metrics found in the requested time range.");
        output::info("The daemon collects metrics every 60 seconds.");
        return Ok(());
    }

    for name in &metric_names {
        print_metric_summary(&conn, name, &cutoff_str);
    }

    output::separator();
    output::info(format!(
        "{} metrics tracked over {}",
        metric_names.len(),
        range
    ));

    Ok(())
}

fn print_metric_summary(conn: &Connection, name: &str, cutoff_str: &str) {
    let stats: Result<(f64, f64, f64, i64, f64), _> = conn.query_row(
        "SELECT AVG(metric_value), MIN(metric_value), MAX(metric_value), \
         COUNT(*), \
         (SELECT metric_value FROM metrics_history \
          WHERE metric_name = ?1 AND timestamp >= ?2 AND aggregation_period = 'raw' \
          ORDER BY timestamp DESC LIMIT 1) \
         FROM metrics_history \
         WHERE metric_name = ?1 AND timestamp >= ?2 AND aggregation_period = 'raw'",
        rusqlite::params![name, cutoff_str],
        |row| {
            Ok((
                row.get(0)?,
                row.get(1)?,
                row.get(2)?,
                row.get(3)?,
                row.get(4)?,
            ))
        },
    );

    match stats {
        Ok((avg, min, max, count, latest)) => {
            output::kv(
                name,
                format!(
                    "latest={:.1}  avg={:.1}  min={:.1}  max={:.1}  ({} samples)",
                    latest, avg, min, max, count
                ),
            );
        }
        Err(_) => {
            output::kv(name, "no data");
        }
    }
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
        // "abch" -> strip_suffix('h') = "abc", parse fails -> unwrap_or(1) = 1 * 3600
        assert_eq!(parse_range_to_seconds("abch"), 3600);
    }
}
