//! Search instrumentation overview display.

use anyhow::Result;
use rusqlite::Connection;

use crate::output;
use super::{open_db, table_exists, StatsPeriod};

/// Show search instrumentation overview.
pub(super) async fn run(period: StatsPeriod) -> Result<()> {
    output::section(format!("Search Instrumentation Stats ({})", period.label()));

    let conn = open_db()?;

    if !table_exists(&conn, "search_events") {
        output::info("No search_events table found. Run daemon to initialize schema.");
        return Ok(());
    }

    let ts_filter = period.start_timestamp();

    let total_count = query_total_count(&conn, ts_filter.as_deref())?;
    if total_count == 0 {
        output::info("No search events recorded in this period.");
        return Ok(());
    }

    output::kv("Total Events", &total_count.to_string());
    output::separator();

    display_tool_distribution(&conn, ts_filter.as_deref(), total_count)?;
    display_behavior_rates(&conn, ts_filter.as_deref())?;
    display_performance(&conn, ts_filter.as_deref())?;
    display_top_queries(&conn, ts_filter.as_deref())?;
    display_resolution_rate(&conn, ts_filter.as_deref(), total_count)?;

    Ok(())
}

fn query_total_count(conn: &Connection, ts_filter: Option<&str>) -> Result<i64> {
    let count = if let Some(ts) = ts_filter {
        conn.query_row(
            "SELECT COUNT(*) FROM search_events WHERE ts >= ?",
            [ts],
            |row| row.get(0),
        )?
    } else {
        conn.query_row("SELECT COUNT(*) FROM search_events", [], |row| row.get(0))?
    };
    Ok(count)
}

fn display_tool_distribution(conn: &Connection, ts_filter: Option<&str>, total_count: i64) -> Result<()> {
    output::info("Tool Distribution:");

    let query = if let Some(ts) = ts_filter {
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

    output::separator();
    Ok(())
}

fn display_behavior_rates(conn: &Connection, ts_filter: Option<&str>) -> Result<()> {
    if !table_exists(conn, "search_behavior") {
        return Ok(());
    }

    output::info("Behavior Classification:");
    let query = if let Some(ts) = ts_filter {
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
    Ok(())
}

fn display_performance(conn: &Connection, ts_filter: Option<&str>) -> Result<()> {
    output::info("Performance (mcp_qdrant):");

    let avg_query = if let Some(ts) = ts_filter {
        format!(
            "SELECT COUNT(*), ROUND(AVG(latency_ms)) FROM search_events WHERE tool = 'mcp_qdrant' AND latency_ms IS NOT NULL AND ts >= '{}'",
            ts
        )
    } else {
        "SELECT COUNT(*), ROUND(AVG(latency_ms)) FROM search_events WHERE tool = 'mcp_qdrant' AND latency_ms IS NOT NULL".to_string()
    };

    match conn.query_row(&avg_query, [], |row| {
        Ok((row.get::<_, i64>(0)?, row.get::<_, Option<f64>>(1)?))
    }) {
        Ok((count, avg)) => {
            if count > 0 {
                output::kv("  Searches with latency", &count.to_string());
                if let Some(avg_val) = avg {
                    output::kv("  Average latency", &format!("{:.0} ms", avg_val));
                }
                display_latency_percentiles(conn, ts_filter)?;
            } else {
                output::info("  No latency data recorded yet.");
            }
        }
        Err(_) => {
            output::info("  No latency data available.");
        }
    }

    output::separator();
    Ok(())
}

fn display_latency_percentiles(conn: &Connection, ts_filter: Option<&str>) -> Result<()> {
    let query = if let Some(ts) = ts_filter {
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

    let mut pstmt = conn.prepare(&query)?;
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

    Ok(())
}

fn display_top_queries(conn: &Connection, ts_filter: Option<&str>) -> Result<()> {
    output::info("Top Queries:");

    let query = if let Some(ts) = ts_filter {
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

    output::separator();
    Ok(())
}

fn display_resolution_rate(conn: &Connection, ts_filter: Option<&str>, total_count: i64) -> Result<()> {
    if !table_exists(conn, "resolution_events") {
        return Ok(());
    }

    output::info("Resolution Rate:");
    let query = if let Some(ts) = ts_filter {
        format!(
            "SELECT COUNT(*) FROM resolution_events WHERE ts >= '{}'",
            ts
        )
    } else {
        "SELECT COUNT(*) FROM resolution_events".to_string()
    };

    match conn.query_row(&query, [], |row| row.get::<_, i64>(0)) {
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

    Ok(())
}
