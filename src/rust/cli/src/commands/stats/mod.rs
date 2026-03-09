//! Stats command - search instrumentation analysis
//!
//! Queries search_events and search_behavior view to display
//! tool distribution, behavior rates, performance metrics,
//! top queries, and resolution rates.

mod overview;
mod processing;

use anyhow::{Context, Result};
use clap::{Args, Subcommand, ValueEnum};
use rusqlite::Connection;

use crate::config::get_database_path;
use wqm_common::timestamps;

/// Time period filter for stats queries
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
pub enum StatsPeriod {
    /// Last 24 hours
    Day,
    /// Last 7 days
    #[default]
    Week,
    /// Last 30 days
    Month,
    /// All time
    All,
}

impl StatsPeriod {
    /// Returns the ISO 8601 timestamp for the start of this period
    fn start_timestamp(&self) -> Option<String> {
        use chrono::{Duration, Utc};
        match self {
            StatsPeriod::Day => Some(
                (Utc::now() - Duration::days(1))
                    .format("%Y-%m-%dT%H:%M:%S%.3fZ")
                    .to_string(),
            ),
            StatsPeriod::Week => Some(
                (Utc::now() - Duration::days(7))
                    .format("%Y-%m-%dT%H:%M:%S%.3fZ")
                    .to_string(),
            ),
            StatsPeriod::Month => Some(
                (Utc::now() - Duration::days(30))
                    .format("%Y-%m-%dT%H:%M:%S%.3fZ")
                    .to_string(),
            ),
            StatsPeriod::All => None,
        }
    }

    fn label(&self) -> &'static str {
        match self {
            StatsPeriod::Day => "Last 24 hours",
            StatsPeriod::Week => "Last 7 days",
            StatsPeriod::Month => "Last 30 days",
            StatsPeriod::All => "All time",
        }
    }
}

/// Stats command arguments
#[derive(Args)]
pub struct StatsArgs {
    #[command(subcommand)]
    command: StatsCommand,
}

/// Stats subcommands
#[derive(Subcommand)]
enum StatsCommand {
    /// Show search instrumentation overview
    Overview {
        /// Time period to analyze
        #[arg(short, long, value_enum, default_value_t = StatsPeriod::Week)]
        period: StatsPeriod,
    },

    /// Show processing timing stats (per-phase breakdown, percentiles)
    Processing {
        /// Time period to analyze
        #[arg(short, long, value_enum, default_value_t = StatsPeriod::Week)]
        period: StatsPeriod,

        /// Filter by operation type (add, update, delete, scan)
        #[arg(long)]
        op: Option<String>,

        /// Filter by item type (file, text, folder, tenant)
        #[arg(long)]
        item_type: Option<String>,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Log a search event (used by wrapper scripts)
    LogSearch {
        /// Tool name (rg, grep, etc.)
        #[arg(long)]
        tool: String,

        /// Search query text
        #[arg(long)]
        query: String,

        /// Actor (claude, user)
        #[arg(long, default_value = "claude")]
        actor: String,

        /// Session ID (optional)
        #[arg(long)]
        session_id: Option<String>,
    },
}

/// Execute stats command
pub async fn execute(args: StatsArgs) -> Result<()> {
    match args.command {
        StatsCommand::Overview { period } => overview::run(period).await,
        StatsCommand::Processing {
            period,
            op,
            item_type,
            json,
        } => processing::run(period, op.as_deref(), item_type.as_deref(), json).await,
        StatsCommand::LogSearch {
            tool,
            query,
            actor,
            session_id,
        } => log_search(&tool, &query, &actor, session_id.as_deref()).await,
    }
}

/// Open state database with read-only pragmas
fn open_db() -> Result<Connection> {
    let db_path = get_database_path().map_err(|e| anyhow::anyhow!("{}", e))?;
    if !db_path.exists() {
        anyhow::bail!(
            "Database not found at {}. Run daemon first: wqm service start",
            db_path.display()
        );
    }
    let conn = Connection::open(&db_path).context("Failed to open state database")?;
    conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA busy_timeout=5000;")
        .context("Failed to set SQLite pragmas")?;
    Ok(conn)
}

/// Check if a table exists in the database
fn table_exists(conn: &Connection, name: &str) -> bool {
    conn.query_row(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?",
        [name],
        |_| Ok(true),
    )
    .unwrap_or(false)
}

async fn log_search(tool: &str, query: &str, actor: &str, session_id: Option<&str>) -> Result<()> {
    let conn = open_db()?;

    if !table_exists(&conn, "search_events") {
        anyhow::bail!("search_events table not found. Run daemon to initialize schema.");
    }

    let event_id = uuid::Uuid::new_v4().to_string();
    let now = timestamps::now_utc();

    conn.execute(
        "INSERT INTO search_events (id, session_id, actor, tool, op, query_text, ts, created_at) \
         VALUES (?1, ?2, ?3, ?4, 'search', ?5, ?6, ?6)",
        rusqlite::params![&event_id, session_id, actor, tool, query, &now],
    )
    .context("Failed to insert search event")?;

    println!("{}", event_id);
    Ok(())
}

/// Compute percentile from a sorted slice of values.
pub(super) fn percentile(sorted: &[i64], pct: f64) -> i64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * pct / 100.0).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Format a duration in ms as human-readable.
pub(super) fn format_duration_ms(ms: i64) -> String {
    if ms < 1000 {
        format!("{} ms", ms)
    } else if ms < 60_000 {
        format!("{:.1} s", ms as f64 / 1000.0)
    } else {
        format!("{:.1} m", ms as f64 / 60_000.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_period_start_timestamp_day() {
        let ts = StatsPeriod::Day.start_timestamp();
        assert!(ts.is_some());
        assert!(ts.unwrap().contains("T"));
    }

    #[test]
    fn test_period_start_timestamp_all() {
        assert!(StatsPeriod::All.start_timestamp().is_none());
    }

    #[test]
    fn test_period_week_default() {
        let period = StatsPeriod::default();
        assert!(matches!(period, StatsPeriod::Week));
    }

    #[test]
    fn test_percentile_empty() {
        assert_eq!(percentile(&[], 50.0), 0);
    }

    #[test]
    fn test_percentile_single() {
        assert_eq!(percentile(&[42], 50.0), 42);
        assert_eq!(percentile(&[42], 0.0), 42);
        assert_eq!(percentile(&[42], 100.0), 42);
    }

    #[test]
    fn test_percentile_multiple() {
        let data = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
        assert_eq!(percentile(&data, 0.0), 10);
        assert_eq!(percentile(&data, 50.0), 60);
        assert_eq!(percentile(&data, 100.0), 100);
    }

    #[test]
    fn test_percentile_quartiles() {
        let data: Vec<i64> = (1..=100).collect();
        let q1 = percentile(&data, 25.0);
        let median = percentile(&data, 50.0);
        let q3 = percentile(&data, 75.0);
        assert!(q1 > 0 && q1 < median);
        assert!(median > q1 && median < q3);
        assert!(q3 > median && q3 <= 100);
    }

    #[test]
    fn test_format_duration_ms_milliseconds() {
        assert_eq!(format_duration_ms(0), "0 ms");
        assert_eq!(format_duration_ms(42), "42 ms");
        assert_eq!(format_duration_ms(999), "999 ms");
    }

    #[test]
    fn test_format_duration_ms_seconds() {
        assert_eq!(format_duration_ms(1000), "1.0 s");
        assert_eq!(format_duration_ms(1500), "1.5 s");
        assert_eq!(format_duration_ms(59999), "60.0 s");
    }

    #[test]
    fn test_format_duration_ms_minutes() {
        assert_eq!(format_duration_ms(60000), "1.0 m");
        assert_eq!(format_duration_ms(90000), "1.5 m");
        assert_eq!(format_duration_ms(300000), "5.0 m");
    }
}
