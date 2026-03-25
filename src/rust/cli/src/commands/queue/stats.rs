//! Queue stats subcommand

use std::collections::HashMap;

use anyhow::Result;
use chrono::{DateTime, Utc};
use rusqlite::Connection;

use crate::output;
use crate::output::style::short_id;

use super::db::connect_readonly;
use super::formatters::{format_status, QueueStatsSummary, StatusBreakdown};

pub async fn execute(json: bool, by_type: bool, by_op: bool, by_collection: bool) -> Result<()> {
    let conn = connect_readonly()?;

    let mut summary = build_summary(&conn)?;
    populate_oldest_pending(&conn, &mut summary)?;
    populate_active_counts(&conn, &mut summary)?;

    if json {
        print_json(&conn, &summary, by_type, by_op, by_collection)?;
    } else {
        print_text(&conn, &summary, by_type, by_op, by_collection)?;
    }

    Ok(())
}

fn build_summary(conn: &Connection) -> Result<QueueStatsSummary> {
    let mut summary = QueueStatsSummary {
        total_items: 0,
        by_status: StatusBreakdown::default(),
        oldest_pending_age_seconds: None,
        oldest_pending_id: None,
        active_collections: 0,
        active_projects: 0,
    };

    let mut stmt = conn.prepare("SELECT status, COUNT(*) FROM unified_queue GROUP BY status")?;
    let rows = stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
    })?;

    for row in rows {
        let (status, count) = row?;
        summary.total_items += count;
        match status.as_str() {
            "pending" => summary.by_status.pending = count,
            "in_progress" => summary.by_status.in_progress = count,
            "done" => summary.by_status.done = count,
            "failed" => summary.by_status.failed = count,
            _ => {}
        }
    }

    Ok(summary)
}

fn populate_oldest_pending(conn: &Connection, summary: &mut QueueStatsSummary) -> Result<()> {
    let oldest_result: Result<(String, String), _> = conn.query_row(
        "SELECT queue_id, created_at FROM unified_queue \
         WHERE status = 'pending' ORDER BY created_at ASC LIMIT 1",
        [],
        |row| Ok((row.get(0)?, row.get(1)?)),
    );

    if let Ok((id, created_at)) = oldest_result {
        summary.oldest_pending_id = Some(id);
        if let Ok(dt) = DateTime::parse_from_rfc3339(&created_at) {
            let age = Utc::now().signed_duration_since(dt.with_timezone(&Utc));
            summary.oldest_pending_age_seconds = Some(age.num_seconds() as f64);
        }
    }

    Ok(())
}

fn populate_active_counts(conn: &Connection, summary: &mut QueueStatsSummary) -> Result<()> {
    summary.active_collections = conn
        .query_row(
            "SELECT COUNT(DISTINCT collection) FROM unified_queue \
             WHERE status IN ('pending', 'in_progress')",
            [],
            |row| row.get(0),
        )
        .unwrap_or(0);

    summary.active_projects = conn
        .query_row(
            "SELECT COUNT(DISTINCT tenant_id) FROM unified_queue \
             WHERE status IN ('pending', 'in_progress')",
            [],
            |row| row.get(0),
        )
        .unwrap_or(0);

    Ok(())
}

fn print_json(
    conn: &Connection,
    summary: &QueueStatsSummary,
    by_type: bool,
    by_op: bool,
    by_collection: bool,
) -> Result<()> {
    let mut output_data = serde_json::json!({
        "summary": summary,
    });

    if by_type {
        let breakdown = get_breakdown(conn, "item_type")?;
        output_data["by_item_type"] = serde_json::to_value(breakdown)?;
    }
    if by_op {
        let breakdown = get_breakdown(conn, "op")?;
        output_data["by_operation"] = serde_json::to_value(breakdown)?;
    }
    if by_collection {
        let breakdown = get_breakdown(conn, "collection")?;
        output_data["by_collection"] = serde_json::to_value(breakdown)?;
    }

    output::print_json(&output_data);
    Ok(())
}

fn print_text(
    conn: &Connection,
    summary: &QueueStatsSummary,
    by_type: bool,
    by_op: bool,
    by_collection: bool,
) -> Result<()> {
    output::section("Queue Statistics");

    output::kv("Total Items", summary.total_items.to_string());
    output::separator();

    output::section("By Status");
    output::kv(
        format!("  {}", format_status("pending")),
        summary.by_status.pending.to_string(),
    );
    output::kv(
        format!("  {}", format_status("in_progress")),
        summary.by_status.in_progress.to_string(),
    );
    output::kv(
        format!("  {}", format_status("done")),
        summary.by_status.done.to_string(),
    );
    output::kv(
        format!("  {}", format_status("failed")),
        summary.by_status.failed.to_string(),
    );

    output::separator();
    output::kv("Active Collections", summary.active_collections.to_string());
    output::kv("Active Projects", summary.active_projects.to_string());

    if let Some(age) = summary.oldest_pending_age_seconds {
        output::separator();
        output::kv(
            "Oldest Pending Age",
            wqm_common::duration_fmt::format_duration(age, 0),
        );
        if let Some(ref id) = summary.oldest_pending_id {
            output::kv("Oldest Pending ID", short_id(id));
        }
    }

    if by_type {
        output::separator();
        print_breakdown(conn, "item_type", "By Item Type")?;
    }
    if by_op {
        output::separator();
        print_breakdown(conn, "op", "By Operation")?;
    }
    if by_collection {
        output::separator();
        print_breakdown(conn, "collection", "By Collection")?;
    }

    Ok(())
}

fn get_breakdown(conn: &Connection, column: &str) -> Result<HashMap<String, StatusBreakdown>> {
    let query = format!(
        "SELECT {}, status, COUNT(*) FROM unified_queue GROUP BY {}, status",
        column, column
    );

    let mut stmt = conn.prepare(&query)?;
    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, i64>(2)?,
        ))
    })?;

    let mut result: HashMap<String, StatusBreakdown> = HashMap::new();

    for row in rows {
        let (key, status, count) = row?;
        let entry = result.entry(key).or_default();
        match status.as_str() {
            "pending" => entry.pending = count,
            "in_progress" => entry.in_progress = count,
            "done" => entry.done = count,
            "failed" => entry.failed = count,
            _ => {}
        }
    }

    Ok(result)
}

fn print_breakdown(conn: &Connection, column: &str, title: &str) -> Result<()> {
    let breakdown = get_breakdown(conn, column)?;

    output::section(title);
    for (key, stats) in &breakdown {
        let total = stats.pending + stats.in_progress + stats.done + stats.failed;
        let detail = format!(
            "{} ({}={}, {}={}, {}={}, {}={})",
            total,
            format_status("pending"),
            stats.pending,
            format_status("in_progress"),
            stats.in_progress,
            format_status("done"),
            stats.done,
            format_status("failed"),
            stats.failed,
        );
        output::kv(format!("  {key}"), detail);
    }

    Ok(())
}
