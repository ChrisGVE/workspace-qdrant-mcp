//! Stats command - search instrumentation analysis
//!
//! Queries search_events and search_behavior view to display
//! tool distribution, behavior rates, performance metrics,
//! top queries, and resolution rates.

use anyhow::{Context, Result};
use clap::{Args, Subcommand, ValueEnum};
use rusqlite::Connection;

use wqm_common::timestamps;
use crate::config::get_database_path;
use crate::output;

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
            StatsPeriod::Day => Some((Utc::now() - Duration::days(1)).format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string()),
            StatsPeriod::Week => Some((Utc::now() - Duration::days(7)).format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string()),
            StatsPeriod::Month => Some((Utc::now() - Duration::days(30)).format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string()),
            StatsPeriod::All => None,
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
        StatsCommand::Overview { period } => overview(period).await,
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
    let conn = Connection::open(&db_path)
        .context("Failed to open state database")?;
    conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")
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

async fn overview(period: StatsPeriod) -> Result<()> {
    let period_label = match period {
        StatsPeriod::Day => "Last 24 hours",
        StatsPeriod::Week => "Last 7 days",
        StatsPeriod::Month => "Last 30 days",
        StatsPeriod::All => "All time",
    };
    output::section(format!("Search Instrumentation Stats ({})", period_label));

    let conn = open_db()?;

    if !table_exists(&conn, "search_events") {
        output::info("No search_events table found. Run daemon to initialize schema.");
        return Ok(());
    }

    let ts_filter = period.start_timestamp();

    // 1. Total event count
    let total_count: i64 = if let Some(ref ts) = ts_filter {
        conn.query_row(
            "SELECT COUNT(*) FROM search_events WHERE ts >= ?",
            [ts],
            |row| row.get(0),
        )?
    } else {
        conn.query_row("SELECT COUNT(*) FROM search_events", [], |row| row.get(0))?
    };

    if total_count == 0 {
        output::info("No search events recorded in this period.");
        return Ok(());
    }

    output::kv("Total Events", &total_count.to_string());
    output::separator();

    // 2. Search tool distribution
    output::info("Tool Distribution:");
    {
        let query = if let Some(ref ts) = ts_filter {
            format!(
                "SELECT tool, COUNT(*) as cnt FROM search_events WHERE ts >= '{}' GROUP BY tool ORDER BY cnt DESC",
                ts
            )
        } else {
            "SELECT tool, COUNT(*) as cnt FROM search_events GROUP BY tool ORDER BY cnt DESC".to_string()
        };

        let mut stmt = conn.prepare(&query)?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
        })?;

        for row in rows {
            let (tool, count) = row?;
            let pct = (count as f64 / total_count as f64 * 100.0).round();
            output::kv(&format!("  {}", tool), &format!("{} ({:.0}%)", count, pct));
        }
    }

    output::separator();

    // 3. Behavior rates (from search_behavior view)
    if table_exists(&conn, "search_behavior") {
        output::info("Behavior Classification:");
        let query = if let Some(ref ts) = ts_filter {
            format!(
                "SELECT behavior, COUNT(*) as cnt FROM search_behavior WHERE ts >= '{}' GROUP BY behavior ORDER BY cnt DESC",
                ts
            )
        } else {
            "SELECT behavior, COUNT(*) as cnt FROM search_behavior GROUP BY behavior ORDER BY cnt DESC".to_string()
        };

        let mut stmt = conn.prepare(&query)?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
        })?;

        let mut behavior_total: i64 = 0;
        let mut behaviors: Vec<(String, i64)> = Vec::new();
        for row in rows {
            let (behavior, count) = row?;
            behavior_total += count;
            behaviors.push((behavior, count));
        }

        for (behavior, count) in &behaviors {
            let pct = if behavior_total > 0 {
                (*count as f64 / behavior_total as f64 * 100.0).round()
            } else {
                0.0
            };
            output::kv(&format!("  {}", behavior), &format!("{} ({:.0}%)", count, pct));
        }
        output::separator();
    }

    // 4. Performance (latency stats)
    output::info("Performance (mcp_qdrant):");
    {
        let query = if let Some(ref ts) = ts_filter {
            format!(
                "SELECT \
                    COUNT(*) as cnt, \
                    ROUND(AVG(latency_ms)) as avg_ms, \
                    latency_ms \
                FROM search_events \
                WHERE tool = 'mcp_qdrant' AND latency_ms IS NOT NULL AND ts >= '{}' \
                ORDER BY latency_ms",
                ts
            )
        } else {
            "SELECT \
                COUNT(*) as cnt, \
                ROUND(AVG(latency_ms)) as avg_ms, \
                latency_ms \
            FROM search_events \
            WHERE tool = 'mcp_qdrant' AND latency_ms IS NOT NULL \
            ORDER BY latency_ms".to_string()
        };

        // Get count and average
        let avg_query = if let Some(ref ts) = ts_filter {
            format!(
                "SELECT COUNT(*), ROUND(AVG(latency_ms)) FROM search_events WHERE tool = 'mcp_qdrant' AND latency_ms IS NOT NULL AND ts >= '{}'",
                ts
            )
        } else {
            "SELECT COUNT(*), ROUND(AVG(latency_ms)) FROM search_events WHERE tool = 'mcp_qdrant' AND latency_ms IS NOT NULL".to_string()
        };

        let _ = query; // suppress unused
        match conn.query_row(&avg_query, [], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, Option<f64>>(1)?))
        }) {
            Ok((count, avg)) => {
                if count > 0 {
                    output::kv("  Searches with latency", &count.to_string());
                    if let Some(avg_val) = avg {
                        output::kv("  Average latency", &format!("{:.0} ms", avg_val));
                    }

                    // Percentiles using SQLite window functions
                    let percentile_query = if let Some(ref ts) = ts_filter {
                        format!(
                            "SELECT latency_ms FROM search_events \
                             WHERE tool = 'mcp_qdrant' AND latency_ms IS NOT NULL AND ts >= '{}' \
                             ORDER BY latency_ms",
                            ts
                        )
                    } else {
                        "SELECT latency_ms FROM search_events \
                         WHERE tool = 'mcp_qdrant' AND latency_ms IS NOT NULL \
                         ORDER BY latency_ms".to_string()
                    };

                    let mut pstmt = conn.prepare(&percentile_query)?;
                    let latencies: Vec<i64> = pstmt
                        .query_map([], |row| row.get::<_, i64>(0))?
                        .filter_map(|r| r.ok())
                        .collect();

                    if !latencies.is_empty() {
                        let p50 = latencies[latencies.len() * 50 / 100];
                        let p95 = latencies[latencies.len() * 95 / 100];
                        let p99 = latencies[(latencies.len() * 99 / 100).min(latencies.len() - 1)];
                        output::kv("  P50", &format!("{} ms", p50));
                        output::kv("  P95", &format!("{} ms", p95));
                        output::kv("  P99", &format!("{} ms", p99));
                    }
                } else {
                    output::info("  No latency data recorded yet.");
                }
            }
            Err(_) => {
                output::info("  No latency data available.");
            }
        }
    }

    output::separator();

    // 5. Top queries
    output::info("Top Queries:");
    {
        let query = if let Some(ref ts) = ts_filter {
            format!(
                "SELECT query_text, COUNT(*) as cnt FROM search_events \
                 WHERE query_text IS NOT NULL AND ts >= '{}' \
                 GROUP BY query_text ORDER BY cnt DESC LIMIT 10",
                ts
            )
        } else {
            "SELECT query_text, COUNT(*) as cnt FROM search_events \
             WHERE query_text IS NOT NULL \
             GROUP BY query_text ORDER BY cnt DESC LIMIT 10".to_string()
        };

        let mut stmt = conn.prepare(&query)?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
        })?;

        let mut found = false;
        for row in rows {
            let (query_text, count) = row?;
            let display = if query_text.len() > 60 {
                format!("{}...", &query_text[..57])
            } else {
                query_text
            };
            output::kv(&format!("  {} x", count), &display);
            found = true;
        }
        if !found {
            output::info("  No queries recorded yet.");
        }
    }

    output::separator();

    // 6. Resolution rate
    if table_exists(&conn, "resolution_events") {
        output::info("Resolution Rate:");
        let resolution_query = if let Some(ref ts) = ts_filter {
            format!(
                "SELECT COUNT(*) FROM resolution_events WHERE ts >= '{}'",
                ts
            )
        } else {
            "SELECT COUNT(*) FROM resolution_events".to_string()
        };

        match conn.query_row(&resolution_query, [], |row| row.get::<_, i64>(0)) {
            Ok(resolution_count) => {
                if total_count > 0 {
                    let rate = (resolution_count as f64 / total_count as f64 * 100.0).round();
                    output::kv("  Searches with resolution", &format!("{} ({:.0}%)", resolution_count, rate));
                }
            }
            Err(_) => {
                output::info("  No resolution data available.");
            }
        }
    }

    Ok(())
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

    // Quick output for wrapper scripts (must be fast)
    println!("{}", event_id);
    Ok(())
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
        // Week is the default period
        let period = StatsPeriod::default();
        assert!(matches!(period, StatsPeriod::Week));
    }
}
