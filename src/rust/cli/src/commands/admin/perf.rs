//! `wqm admin perf` — display pipeline performance statistics
//!
//! Queries the `processing_timings` SQLite table to compute per-phase
//! aggregates (avg, p50, p95, p99) and throughput over a configurable window.

use anyhow::{Context, Result};

use crate::output;

/// Phase-level aggregate stats from a single SQLite query.
struct PhaseStats {
    phase: String,
    count: i64,
    avg_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
}

/// Execute the perf subcommand.
pub async fn execute(window_hours: f64, json: bool) -> Result<()> {
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
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='processing_timings')",
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
            "SELECT COUNT(DISTINCT queue_id) FROM processing_timings WHERE created_at > datetime('now', ?1)",
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

    // Per-phase statistics using window functions for percentiles
    let phase_stats = query_phase_stats(&conn, &cutoff)?;

    // Queue depth (from unified_queue)
    let queue_depth: i64 = conn
        .query_row("SELECT COUNT(*) FROM unified_queue", [], |row| row.get(0))
        .unwrap_or(0);

    if json {
        print_json(&phase_stats, total_items, queue_depth, window_hours);
    } else {
        print_table(&phase_stats, total_items, queue_depth, window_hours);
    }

    Ok(())
}

fn query_phase_stats(conn: &rusqlite::Connection, cutoff: &str) -> Result<Vec<PhaseStats>> {
    // SQLite doesn't have native percentile functions, so we compute them manually
    // by sorting durations per phase and picking the Nth element.
    let mut stmt = conn.prepare(
        "SELECT phase, COUNT(*), AVG(duration_ms), \
         MIN(duration_ms), MAX(duration_ms) \
         FROM processing_timings \
         WHERE created_at > datetime('now', ?1) \
         GROUP BY phase ORDER BY phase",
    )?;

    let phases: Vec<(String, i64)> = stmt
        .query_map(rusqlite::params![cutoff], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
        })?
        .filter_map(|r| r.ok())
        .collect();

    let mut results = Vec::new();

    for (phase, count) in &phases {
        let avg: f64 = conn
            .query_row(
                "SELECT AVG(duration_ms) FROM processing_timings \
                 WHERE phase = ?1 AND created_at > datetime('now', ?2)",
                rusqlite::params![phase, cutoff],
                |row| row.get(0),
            )
            .unwrap_or(0.0);

        // Fetch sorted durations for percentile calculation
        let mut durations_stmt = conn.prepare(
            "SELECT duration_ms FROM processing_timings \
             WHERE phase = ?1 AND created_at > datetime('now', ?2) \
             ORDER BY duration_ms",
        )?;
        let durations: Vec<i64> = durations_stmt
            .query_map(rusqlite::params![phase, cutoff], |row| row.get(0))?
            .filter_map(|r| r.ok())
            .collect();

        let p50 = percentile(&durations, 50);
        let p95 = percentile(&durations, 95);
        let p99 = percentile(&durations, 99);

        results.push(PhaseStats {
            phase: phase.clone(),
            count: *count,
            avg_ms: avg,
            p50_ms: p50,
            p95_ms: p95,
            p99_ms: p99,
        });
    }

    Ok(results)
}

fn percentile(sorted: &[i64], pct: u8) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((pct as f64 / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    let idx = idx.min(sorted.len() - 1);
    sorted[idx] as f64
}

fn print_table(stats: &[PhaseStats], total_items: i64, queue_depth: i64, window_hours: f64) {
    output::section(format!("Pipeline Performance (last {}h)", window_hours));

    // Phase breakdown table
    println!(
        "  {:<10} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "Phase", "Count", "Avg(ms)", "P50(ms)", "P95(ms)", "P99(ms)"
    );
    println!("  {}", "-".repeat(58));

    for s in stats {
        println!(
            "  {:<10} {:>8} {:>8.0} {:>8.0} {:>8.0} {:>8.0}",
            s.phase, s.count, s.avg_ms, s.p50_ms, s.p95_ms, s.p99_ms
        );
    }

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

fn print_json(stats: &[PhaseStats], total_items: i64, queue_depth: i64, window_hours: f64) {
    let phases: Vec<serde_json::Value> = stats
        .iter()
        .map(|s| {
            serde_json::json!({
                "phase": s.phase,
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
        "phases": phases,
    });

    println!("{}", serde_json::to_string_pretty(&obj).unwrap());
}
