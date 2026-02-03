//! Queue command - unified queue inspector for debugging
//!
//! Read-only queue inspector for debugging and monitoring.
//! Subcommands: list, show, stats
//!
//! Note: Queue cleanup is automatic (daemon handles this).

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use clap::{Args, Subcommand};
use colored::Colorize;
use rusqlite::{Connection, params};
use serde::Serialize;
use tabled::{settings::Style, Table, Tabled};

use crate::config::get_database_path_checked;
use crate::output;

/// Queue command arguments
#[derive(Args)]
pub struct QueueArgs {
    #[command(subcommand)]
    command: QueueCommand,
}

/// Queue subcommands
#[derive(Subcommand)]
enum QueueCommand {
    /// List queue items with optional filters
    List {
        /// Filter by status (pending, in_progress, done, failed)
        #[arg(short, long)]
        status: Option<String>,

        /// Filter by collection name
        #[arg(short, long)]
        collection: Option<String>,

        /// Filter by item type (file, folder, content, project, library)
        #[arg(short = 't', long)]
        item_type: Option<String>,

        /// Maximum number of items to show
        #[arg(short, long, default_value = "50")]
        limit: i64,

        /// Skip first N items (for pagination)
        #[arg(long, default_value = "0")]
        offset: i64,

        /// Order by field (created_at, priority, status)
        #[arg(short = 'o', long, default_value = "created_at")]
        order_by: String,

        /// Descending order
        #[arg(short = 'd', long)]
        desc: bool,

        /// Output as JSON
        #[arg(long)]
        json: bool,

        /// Show more columns
        #[arg(short, long)]
        verbose: bool,
    },

    /// Show detailed information for a specific queue item
    Show {
        /// Queue ID or idempotency key prefix
        queue_id: String,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Show queue statistics
    Stats {
        /// Output as JSON
        #[arg(long)]
        json: bool,

        /// Show breakdown by item type
        #[arg(short = 't', long)]
        by_type: bool,

        /// Show breakdown by operation
        #[arg(short = 'o', long)]
        by_op: bool,

        /// Show breakdown by collection
        #[arg(short = 'c', long)]
        by_collection: bool,
    },
}

/// Queue item for list display
#[derive(Debug, Tabled, Serialize)]
pub struct QueueListItem {
    #[tabled(rename = "ID")]
    pub queue_id: String,
    #[tabled(rename = "Type")]
    pub item_type: String,
    #[tabled(rename = "Op")]
    pub op: String,
    #[tabled(rename = "Collection")]
    pub collection: String,
    #[tabled(rename = "Status")]
    pub status: String,
    #[tabled(rename = "Pri")]
    pub priority: i32,
    #[tabled(rename = "Age")]
    pub age: String,
    #[tabled(rename = "Retry")]
    pub retry_count: i32,
}

/// Queue item for verbose list display
#[derive(Debug, Tabled, Serialize)]
pub struct QueueListItemVerbose {
    #[tabled(rename = "ID")]
    pub queue_id: String,
    #[tabled(rename = "Idempotency Key")]
    pub idempotency_key: String,
    #[tabled(rename = "Type")]
    pub item_type: String,
    #[tabled(rename = "Op")]
    pub op: String,
    #[tabled(rename = "Collection")]
    pub collection: String,
    #[tabled(rename = "Status")]
    pub status: String,
    #[tabled(rename = "Pri")]
    pub priority: i32,
    #[tabled(rename = "Created")]
    pub created_at: String,
    #[tabled(rename = "Retry")]
    pub retry_count: i32,
    #[tabled(rename = "Worker")]
    pub worker_id: String,
}

/// Queue item detail view
#[derive(Debug, Serialize)]
pub struct QueueDetailItem {
    pub queue_id: String,
    pub idempotency_key: String,
    pub item_type: String,
    pub op: String,
    pub tenant_id: String,
    pub collection: String,
    pub priority: i32,
    pub status: String,
    pub branch: String,
    pub payload_json: String,
    pub metadata: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    pub lease_until: Option<String>,
    pub worker_id: Option<String>,
    pub retry_count: i32,
    pub max_retries: i32,
    pub error_message: Option<String>,
    pub last_error_at: Option<String>,
}

/// Queue statistics
#[derive(Debug, Serialize)]
pub struct QueueStatsSummary {
    pub total_items: i64,
    pub by_status: StatusBreakdown,
    pub oldest_pending_age_seconds: Option<f64>,
    pub oldest_pending_id: Option<String>,
    pub active_collections: i64,
    pub active_projects: i64,
}

/// Status breakdown
#[derive(Debug, Serialize, Default)]
pub struct StatusBreakdown {
    pub pending: i64,
    pub in_progress: i64,
    pub done: i64,
    pub failed: i64,
}

/// Execute queue command
pub async fn execute(args: QueueArgs) -> Result<()> {
    match args.command {
        QueueCommand::List {
            status,
            collection,
            item_type,
            limit,
            offset,
            order_by,
            desc,
            json,
            verbose,
        } => list(status, collection, item_type, limit, offset, &order_by, desc, json, verbose).await,
        QueueCommand::Show { queue_id, json } => show(&queue_id, json).await,
        QueueCommand::Stats { json, by_type, by_op, by_collection } => {
            stats(json, by_type, by_op, by_collection).await
        }
    }
}

/// Connect to the state database (read-only for safety)
fn connect_readonly() -> Result<Connection> {
    let db_path = get_database_path_checked()
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    let conn = Connection::open_with_flags(
        &db_path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .context(format!("Failed to open state database at {:?}", db_path))?;

    Ok(conn)
}

/// Format relative time from ISO timestamp
fn format_relative_time(timestamp_str: &str) -> String {
    if let Ok(dt) = DateTime::parse_from_rfc3339(timestamp_str) {
        let now = Utc::now();
        let duration = now.signed_duration_since(dt.with_timezone(&Utc));

        let secs = duration.num_seconds();
        if secs < 0 {
            return "future".to_string();
        }

        if secs < 60 {
            format!("{}s ago", secs)
        } else if secs < 3600 {
            format!("{}m ago", secs / 60)
        } else if secs < 86400 {
            format!("{}h ago", secs / 3600)
        } else {
            format!("{}d ago", secs / 86400)
        }
    } else {
        "unknown".to_string()
    }
}

/// Format status with color
fn format_status(status: &str) -> String {
    match status {
        "pending" => "pending".yellow().to_string(),
        "in_progress" => "in_progress".blue().to_string(),
        "done" => "done".green().to_string(),
        "failed" => "failed".red().to_string(),
        _ => status.to_string(),
    }
}

/// Truncate string for display
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

async fn list(
    status: Option<String>,
    collection: Option<String>,
    item_type: Option<String>,
    limit: i64,
    offset: i64,
    order_by: &str,
    desc: bool,
    json: bool,
    verbose: bool,
) -> Result<()> {
    let conn = connect_readonly()?;

    // Build WHERE clause
    let mut conditions: Vec<String> = Vec::new();
    let mut params_vec: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();

    if let Some(ref s) = status {
        conditions.push("status = ?".to_string());
        params_vec.push(Box::new(s.clone()));
    }
    if let Some(ref c) = collection {
        conditions.push("collection = ?".to_string());
        params_vec.push(Box::new(c.clone()));
    }
    if let Some(ref t) = item_type {
        conditions.push("item_type = ?".to_string());
        params_vec.push(Box::new(t.clone()));
    }

    let where_clause = if conditions.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", conditions.join(" AND "))
    };

    // Validate order_by
    let order_column = match order_by {
        "created_at" | "priority" | "status" | "item_type" | "collection" => order_by,
        _ => "created_at",
    };
    let order_direction = if desc { "DESC" } else { "ASC" };

    let query = format!(
        r#"
        SELECT queue_id, idempotency_key, item_type, op, collection, status,
               priority, created_at, retry_count, worker_id
        FROM unified_queue
        {}
        ORDER BY {} {}
        LIMIT ? OFFSET ?
        "#,
        where_clause, order_column, order_direction
    );

    params_vec.push(Box::new(limit));
    params_vec.push(Box::new(offset));

    let params_slice: Vec<&dyn rusqlite::ToSql> = params_vec.iter().map(|p| p.as_ref()).collect();

    let mut stmt = conn.prepare(&query)?;
    let rows = stmt.query_map(params_slice.as_slice(), |row| {
        Ok((
            row.get::<_, String>(0)?,  // queue_id
            row.get::<_, String>(1)?,  // idempotency_key
            row.get::<_, String>(2)?,  // item_type
            row.get::<_, String>(3)?,  // op
            row.get::<_, String>(4)?,  // collection
            row.get::<_, String>(5)?,  // status
            row.get::<_, i32>(6)?,     // priority
            row.get::<_, String>(7)?,  // created_at
            row.get::<_, i32>(8)?,     // retry_count
            row.get::<_, Option<String>>(9)?,  // worker_id
        ))
    })?;

    let items: Vec<_> = rows.filter_map(|r| r.ok()).collect();

    if items.is_empty() {
        if json {
            println!("[]");
        } else {
            output::info("No queue items found");
        }
        return Ok(());
    }

    if verbose {
        let display_items: Vec<QueueListItemVerbose> = items
            .iter()
            .map(|(queue_id, idempotency_key, item_type, op, collection, status, priority, created_at, retry_count, worker_id)| {
                QueueListItemVerbose {
                    queue_id: truncate(queue_id, 12),
                    idempotency_key: truncate(idempotency_key, 12),
                    item_type: item_type.clone(),
                    op: op.clone(),
                    collection: truncate(collection, 20),
                    status: status.clone(),
                    priority: *priority,
                    created_at: created_at.clone(),
                    retry_count: *retry_count,
                    worker_id: worker_id.clone().unwrap_or_default(),
                }
            })
            .collect();

        if json {
            output::print_json(&display_items);
        } else {
            let table = Table::new(&display_items).with(Style::rounded()).to_string();
            println!("{}", table);
            output::info(format!("Showing {} items", display_items.len()));
        }
    } else {
        let display_items: Vec<QueueListItem> = items
            .iter()
            .map(|(queue_id, _idempotency_key, item_type, op, collection, status, priority, created_at, retry_count, _worker_id)| {
                QueueListItem {
                    queue_id: truncate(queue_id, 12),
                    item_type: item_type.clone(),
                    op: op.clone(),
                    collection: truncate(collection, 20),
                    status: format_status(status),
                    priority: *priority,
                    age: format_relative_time(created_at),
                    retry_count: *retry_count,
                }
            })
            .collect();

        if json {
            output::print_json(&display_items);
        } else {
            let table = Table::new(&display_items).with(Style::rounded()).to_string();
            println!("{}", table);
            output::info(format!("Showing {} items", display_items.len()));
        }
    }

    Ok(())
}

async fn show(queue_id: &str, json: bool) -> Result<()> {
    let conn = connect_readonly()?;

    // Try exact match first, then prefix match
    let query = r#"
        SELECT queue_id, idempotency_key, item_type, op, tenant_id, collection,
               priority, status, branch, payload_json, metadata, created_at,
               updated_at, lease_until, worker_id, retry_count, max_retries,
               error_message, last_error_at
        FROM unified_queue
        WHERE queue_id = ? OR queue_id LIKE ? OR idempotency_key LIKE ?
        LIMIT 1
    "#;

    let prefix = format!("{}%", queue_id);
    let mut stmt = conn.prepare(query)?;
    let result = stmt.query_row(params![queue_id, &prefix, &prefix], |row| {
        Ok(QueueDetailItem {
            queue_id: row.get(0)?,
            idempotency_key: row.get(1)?,
            item_type: row.get(2)?,
            op: row.get(3)?,
            tenant_id: row.get(4)?,
            collection: row.get(5)?,
            priority: row.get(6)?,
            status: row.get(7)?,
            branch: row.get(8)?,
            payload_json: row.get(9)?,
            metadata: row.get(10)?,
            created_at: row.get(11)?,
            updated_at: row.get(12)?,
            lease_until: row.get(13)?,
            worker_id: row.get(14)?,
            retry_count: row.get(15)?,
            max_retries: row.get(16)?,
            error_message: row.get(17)?,
            last_error_at: row.get(18)?,
        })
    });

    match result {
        Ok(item) => {
            if json {
                output::print_json(&item);
            } else {
                output::section("Queue Item Details");
                output::kv("Queue ID", &item.queue_id);
                output::kv("Idempotency Key", &item.idempotency_key);
                output::separator();
                output::kv("Item Type", &item.item_type);
                output::kv("Operation", &item.op);
                output::kv("Tenant ID", &item.tenant_id);
                output::kv("Collection", &item.collection);
                output::kv("Branch", &item.branch);
                output::separator();
                output::kv("Status", &format_status(&item.status));
                output::kv("Priority", &item.priority.to_string());
                output::kv("Retry Count", &format!("{}/{}", item.retry_count, item.max_retries));
                output::separator();
                output::kv("Created At", &item.created_at);
                output::kv("Updated At", &item.updated_at);
                if let Some(ref lease) = item.lease_until {
                    output::kv("Lease Until", lease);
                }
                if let Some(ref worker) = item.worker_id {
                    output::kv("Worker ID", worker);
                }

                if item.error_message.is_some() || item.last_error_at.is_some() {
                    output::separator();
                    if let Some(ref err) = item.error_message {
                        output::kv("Error Message", err);
                    }
                    if let Some(ref err_at) = item.last_error_at {
                        output::kv("Last Error At", err_at);
                    }
                }

                // Pretty-print payload JSON
                output::separator();
                println!("{}", "Payload:".bold());
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&item.payload_json) {
                    if let Ok(pretty) = serde_json::to_string_pretty(&parsed) {
                        for line in pretty.lines() {
                            println!("  {}", line);
                        }
                    } else {
                        println!("  {}", item.payload_json);
                    }
                } else {
                    println!("  {}", item.payload_json);
                }

                if let Some(ref meta) = item.metadata {
                    output::separator();
                    println!("{}", "Metadata:".bold());
                    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(meta) {
                        if let Ok(pretty) = serde_json::to_string_pretty(&parsed) {
                            for line in pretty.lines() {
                                println!("  {}", line);
                            }
                        } else {
                            println!("  {}", meta);
                        }
                    } else {
                        println!("  {}", meta);
                    }
                }
            }
        }
        Err(rusqlite::Error::QueryReturnedNoRows) => {
            output::error(format!("Queue item not found: {}", queue_id));
        }
        Err(e) => {
            return Err(e.into());
        }
    }

    Ok(())
}

async fn stats(json: bool, by_type: bool, by_op: bool, by_collection: bool) -> Result<()> {
    let conn = connect_readonly()?;

    // Get overall stats
    let mut summary = QueueStatsSummary {
        total_items: 0,
        by_status: StatusBreakdown::default(),
        oldest_pending_age_seconds: None,
        oldest_pending_id: None,
        active_collections: 0,
        active_projects: 0,
    };

    // Count by status
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

    // Get oldest pending item
    let oldest_result: Result<(String, String), _> = conn.query_row(
        "SELECT queue_id, created_at FROM unified_queue WHERE status = 'pending' ORDER BY created_at ASC LIMIT 1",
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

    // Count active collections
    let active_collections: i64 = conn.query_row(
        "SELECT COUNT(DISTINCT collection) FROM unified_queue WHERE status IN ('pending', 'in_progress')",
        [],
        |row| row.get(0),
    ).unwrap_or(0);
    summary.active_collections = active_collections;

    // Count active tenants (projects)
    let active_projects: i64 = conn.query_row(
        "SELECT COUNT(DISTINCT tenant_id) FROM unified_queue WHERE status IN ('pending', 'in_progress')",
        [],
        |row| row.get(0),
    ).unwrap_or(0);
    summary.active_projects = active_projects;

    if json {
        let mut output_data = serde_json::json!({
            "summary": summary,
        });

        if by_type {
            let breakdown = get_breakdown(&conn, "item_type")?;
            output_data["by_item_type"] = serde_json::to_value(breakdown)?;
        }
        if by_op {
            let breakdown = get_breakdown(&conn, "op")?;
            output_data["by_operation"] = serde_json::to_value(breakdown)?;
        }
        if by_collection {
            let breakdown = get_breakdown(&conn, "collection")?;
            output_data["by_collection"] = serde_json::to_value(breakdown)?;
        }

        output::print_json(&output_data);
    } else {
        output::section("Queue Statistics");

        output::kv("Total Items", &summary.total_items.to_string());
        output::separator();

        println!("{}", "By Status:".bold());
        println!("  {} {}", format_status("pending"), summary.by_status.pending);
        println!("  {} {}", format_status("in_progress"), summary.by_status.in_progress);
        println!("  {} {}", format_status("done"), summary.by_status.done);
        println!("  {} {}", format_status("failed"), summary.by_status.failed);

        output::separator();
        output::kv("Active Collections", &summary.active_collections.to_string());
        output::kv("Active Projects", &summary.active_projects.to_string());

        if let Some(age) = summary.oldest_pending_age_seconds {
            output::separator();
            output::kv("Oldest Pending Age", &output::format_duration(age as u64));
            if let Some(ref id) = summary.oldest_pending_id {
                output::kv("Oldest Pending ID", &truncate(id, 20));
            }
        }

        if by_type {
            output::separator();
            print_breakdown(&conn, "item_type", "By Item Type")?;
        }
        if by_op {
            output::separator();
            print_breakdown(&conn, "op", "By Operation")?;
        }
        if by_collection {
            output::separator();
            print_breakdown(&conn, "collection", "By Collection")?;
        }
    }

    Ok(())
}

fn get_breakdown(conn: &Connection, column: &str) -> Result<std::collections::HashMap<String, StatusBreakdown>> {
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

    let mut result: std::collections::HashMap<String, StatusBreakdown> = std::collections::HashMap::new();

    for row in rows {
        let (key, status, count) = row?;
        let entry = result.entry(key).or_insert(StatusBreakdown::default());
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

    println!("{}", title.bold());
    for (key, stats) in &breakdown {
        let total = stats.pending + stats.in_progress + stats.done + stats.failed;
        println!(
            "  {}: {} (pending={}, in_progress={}, done={}, failed={})",
            truncate(key, 30),
            total,
            stats.pending,
            stats.in_progress,
            stats.done,
            stats.failed
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_relative_time() {
        // Test with a timestamp 30 seconds ago
        let now = Utc::now();
        let timestamp = (now - chrono::Duration::seconds(30)).to_rfc3339();
        let result = format_relative_time(&timestamp);
        assert!(result.contains("s ago") || result.contains("0m ago"));

        // Test with a timestamp 2 hours ago
        let timestamp = (now - chrono::Duration::hours(2)).to_rfc3339();
        let result = format_relative_time(&timestamp);
        assert!(result.contains("h ago"));

        // Test with invalid timestamp
        let result = format_relative_time("invalid");
        assert_eq!(result, "unknown");
    }

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("short", 10), "short");
        assert_eq!(truncate("this is a long string", 10), "this is...");
        assert_eq!(truncate("abc", 3), "abc");
    }

    #[test]
    fn test_format_status() {
        // Just verify it doesn't panic
        let _ = format_status("pending");
        let _ = format_status("in_progress");
        let _ = format_status("done");
        let _ = format_status("failed");
        let _ = format_status("unknown");
    }
}
