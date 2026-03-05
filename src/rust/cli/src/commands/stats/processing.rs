//! Processing timing statistics display.

use anyhow::Result;
use rusqlite::Connection;

use super::{format_duration_ms, open_db, percentile, table_exists, StatsPeriod};
use crate::output;

/// Show processing timing statistics.
pub(super) async fn run(
    period: StatsPeriod,
    op_filter: Option<&str>,
    item_type_filter: Option<&str>,
    json_output: bool,
) -> Result<()> {
    let conn = open_db()?;

    if !table_exists(&conn, "processing_timings") {
        if json_output {
            println!("{{\"error\":\"processing_timings table not found\"}}");
        } else {
            output::info("No processing_timings table found. Upgrade daemon to schema v26+.");
        }
        return Ok(());
    }

    let ts_filter = period.start_timestamp();

    let (conditions, params) = build_filter_params(&ts_filter, op_filter, item_type_filter);
    let where_clause = if conditions.is_empty() {
        String::new()
    } else {
        format!(" WHERE {}", conditions.join(" AND "))
    };

    let total = query_total(&conn, &where_clause, &params)?;
    if total == 0 {
        if json_output {
            println!("{{\"total\":0,\"operations\":[],\"phases\":[]}}");
        } else {
            output::info(format!(
                "No processing timing data for {}.",
                period.label().to_lowercase()
            ));
        }
        return Ok(());
    }

    let op_rows = query_operations(&conn, &where_clause, &params)?;
    let phase_rows = query_phases(&conn, &where_clause, &params)?;
    let phase_percentiles = compute_phase_percentiles(&conn, &phase_rows, &conditions, &params)?;

    if json_output {
        print_json(total, &op_rows, &phase_rows, &phase_percentiles);
    } else {
        print_text(period, total, &op_rows, &phase_rows, &phase_percentiles);
    }

    Ok(())
}

fn build_filter_params(
    ts_filter: &Option<String>,
    op_filter: Option<&str>,
    item_type_filter: Option<&str>,
) -> (Vec<String>, Vec<Box<dyn rusqlite::types::ToSql>>) {
    let mut conditions = Vec::new();
    let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

    if let Some(ref ts) = ts_filter {
        conditions.push("created_at >= ?".to_string());
        params.push(Box::new(ts.clone()));
    }
    if let Some(op) = op_filter {
        conditions.push("op = ?".to_string());
        params.push(Box::new(op.to_string()));
    }
    if let Some(it) = item_type_filter {
        conditions.push("item_type = ?".to_string());
        params.push(Box::new(it.to_string()));
    }

    (conditions, params)
}

fn query_total(
    conn: &Connection,
    where_clause: &str,
    params: &[Box<dyn rusqlite::types::ToSql>],
) -> Result<i64> {
    let query = format!("SELECT COUNT(*) FROM processing_timings{}", where_clause);
    let total = conn.query_row(
        &query,
        rusqlite::params_from_iter(params.iter().map(|p| p.as_ref())),
        |row| row.get(0),
    )?;
    Ok(total)
}

fn query_operations(
    conn: &Connection,
    where_clause: &str,
    params: &[Box<dyn rusqlite::types::ToSql>],
) -> Result<Vec<(String, String, i64, i64)>> {
    let query = format!(
        "SELECT op, item_type, COUNT(*) as cnt, SUM(duration_ms) as total_ms \
         FROM processing_timings{} \
         GROUP BY op, item_type ORDER BY cnt DESC",
        where_clause
    );

    let mut stmt = conn.prepare(&query)?;
    let rows: Vec<(String, String, i64, i64)> = stmt
        .query_map(
            rusqlite::params_from_iter(params.iter().map(|p| p.as_ref())),
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, i64>(3)?,
                ))
            },
        )?
        .filter_map(|r| r.ok())
        .collect();

    Ok(rows)
}

fn query_phases(
    conn: &Connection,
    where_clause: &str,
    params: &[Box<dyn rusqlite::types::ToSql>],
) -> Result<Vec<(String, i64, i64, f64, i64, i64)>> {
    let query = format!(
        "SELECT phase, COUNT(*) as cnt, \
         MIN(duration_ms) as min_ms, \
         ROUND(AVG(duration_ms)) as avg_ms, \
         MAX(duration_ms) as max_ms, \
         SUM(duration_ms) as total_ms \
         FROM processing_timings{} \
         GROUP BY phase ORDER BY total_ms DESC",
        where_clause
    );

    let mut stmt = conn.prepare(&query)?;
    let rows: Vec<(String, i64, i64, f64, i64, i64)> = stmt
        .query_map(
            rusqlite::params_from_iter(params.iter().map(|p| p.as_ref())),
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, f64>(3)?,
                    row.get::<_, i64>(4)?,
                    row.get::<_, i64>(5)?,
                ))
            },
        )?
        .filter_map(|r| r.ok())
        .collect();

    Ok(rows)
}

fn compute_phase_percentiles(
    conn: &Connection,
    phase_rows: &[(String, i64, i64, f64, i64, i64)],
    conditions: &[String],
    params: &[Box<dyn rusqlite::types::ToSql>],
) -> Result<Vec<(String, i64, i64, i64, i64, i64)>> {
    let mut phase_percentiles = Vec::new();

    for (phase, _, _, _, _, _) in phase_rows {
        let extra_where = if conditions.is_empty() {
            String::new()
        } else {
            format!(" AND {}", conditions.join(" AND "))
        };
        let query = format!(
            "SELECT duration_ms FROM processing_timings \
             WHERE phase = ?{} ORDER BY duration_ms",
            extra_where
        );

        let mut pct_params: Vec<Box<dyn rusqlite::types::ToSql>> = vec![Box::new(phase.clone())];
        for p in params {
            pct_params.push(Box::new(
                p.as_ref()
                    .to_sql()
                    .unwrap_or(rusqlite::types::ToSqlOutput::from(rusqlite::types::Null)),
            ));
        }

        let mut pct_stmt = conn.prepare(&query)?;
        let durations: Vec<i64> = pct_stmt
            .query_map(
                rusqlite::params_from_iter(pct_params.iter().map(|p| p.as_ref())),
                |row| row.get::<_, i64>(0),
            )?
            .filter_map(|r| r.ok())
            .collect();

        let min = *durations.first().unwrap_or(&0);
        let q1 = percentile(&durations, 25.0);
        let median = percentile(&durations, 50.0);
        let q3 = percentile(&durations, 75.0);
        let max = *durations.last().unwrap_or(&0);
        phase_percentiles.push((phase.clone(), min, q1, median, q3, max));
    }

    Ok(phase_percentiles)
}

fn print_text(
    period: StatsPeriod,
    total: i64,
    op_rows: &[(String, String, i64, i64)],
    phase_rows: &[(String, i64, i64, f64, i64, i64)],
    phase_percentiles: &[(String, i64, i64, i64, i64, i64)],
) {
    output::section(format!("Processing Timing Stats ({})", period.label()));
    output::kv("Total timing records", total.to_string());
    output::separator();

    output::info("Operations:");
    for (op, item_type, cnt, total_ms) in op_rows {
        output::kv(
            format!("  {} {}", op, item_type),
            format!("{} events, {} total", cnt, format_duration_ms(*total_ms)),
        );
    }
    output::separator();

    output::info("Phase Breakdown (min / Q1 / median / Q3 / max):");
    for ((phase, count, _min_ms, avg_ms, _max_ms, total_ms), (_, min, q1, median, q3, max)) in
        phase_rows.iter().zip(phase_percentiles.iter())
    {
        output::kv(
            format!(
                "  {} ({} records, {} total, ~{} avg)",
                phase,
                count,
                format_duration_ms(*total_ms),
                format_duration_ms(*avg_ms as i64)
            ),
            format!(
                "{} / {} / {} / {} / {}",
                format_duration_ms(*min),
                format_duration_ms(*q1),
                format_duration_ms(*median),
                format_duration_ms(*q3),
                format_duration_ms(*max),
            ),
        );
    }
}

fn print_json(
    total: i64,
    op_rows: &[(String, String, i64, i64)],
    phase_rows: &[(String, i64, i64, f64, i64, i64)],
    phase_percentiles: &[(String, i64, i64, i64, i64, i64)],
) {
    let ops: Vec<String> = op_rows
        .iter()
        .map(|(op, item_type, cnt, total_ms)| {
            format!(
                "{{\"op\":\"{}\",\"item_type\":\"{}\",\"count\":{},\"total_ms\":{}}}",
                op, item_type, cnt, total_ms
            )
        })
        .collect();

    let phases: Vec<String> = phase_rows
        .iter()
        .zip(phase_percentiles.iter())
        .map(|((phase, count, min_ms, avg_ms, max_ms, total_ms), (_, _min, q1, median, q3, _max))| {
            format!(
                "{{\"phase\":\"{}\",\"count\":{},\"min_ms\":{},\"q1_ms\":{},\"median_ms\":{},\"q3_ms\":{},\"max_ms\":{},\"avg_ms\":{:.0},\"total_ms\":{}}}",
                phase, count, min_ms, q1, median, q3, max_ms, avg_ms, total_ms
            )
        })
        .collect();

    println!(
        "{{\"total\":{},\"operations\":[{}],\"phases\":[{}]}}",
        total,
        ops.join(","),
        phases.join(",")
    );
}
