//! Queue command - unified queue management and inspection
//!
//! Queue inspector and management for debugging and monitoring.
//! Subcommands: list, show, stats, retry, clean, remove

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use clap::{Args, Subcommand};
use colored::Colorize;
use rusqlite::{Connection, params};
use serde::Serialize;
use tabled::Tabled;

use crate::config::get_database_path_checked;
use crate::output::{self, ColumnHints};

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

    /// Retry failed queue items
    Retry {
        /// Queue ID to retry (omit for --all)
        queue_id: Option<String>,

        /// Retry all failed items
        #[arg(long)]
        all: bool,
    },

    /// Clean old completed or failed queue items
    Clean {
        /// Remove items older than N days (default: 7)
        #[arg(long, default_value = "7")]
        days: i64,

        /// Only clean items with this status (done, failed)
        #[arg(long)]
        status: Option<String>,

        /// Skip confirmation prompt
        #[arg(short = 'y', long)]
        yes: bool,
    },

    /// Remove a specific queue item by ID
    Remove {
        /// Queue ID or ID prefix
        queue_id: String,
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
    #[tabled(rename = "Age")]
    pub age: String,
    #[tabled(rename = "Retry")]
    pub retry_count: i32,
}

impl ColumnHints for QueueListItem {
    // All categorical
    fn content_columns() -> &'static [usize] { &[] }
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
    #[tabled(rename = "Created")]
    pub created_at: String,
    #[tabled(rename = "Retry")]
    pub retry_count: i32,
    #[tabled(rename = "Worker")]
    pub worker_id: String,
}

impl ColumnHints for QueueListItemVerbose {
    // All categorical
    fn content_columns() -> &'static [usize] { &[] }
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
    pub status: String,
    pub branch: String,
    pub payload_json: String,
    pub metadata: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    pub lease_until: Option<String>,
    pub worker_id: Option<String>,
    pub retry_count: i32,
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
        QueueCommand::Retry { queue_id, all } => retry(queue_id, all).await,
        QueueCommand::Clean { days, status, yes } => clean(days, status, yes).await,
        QueueCommand::Remove { queue_id } => remove(&queue_id).await,
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
        "created_at" | "status" | "item_type" | "collection" => order_by,
        _ => "created_at",
    };
    let order_direction = if desc { "DESC" } else { "ASC" };

    let query = format!(
        r#"
        SELECT queue_id, idempotency_key, item_type, op, collection, status,
               created_at, retry_count, worker_id
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
            row.get::<_, String>(6)?,  // created_at
            row.get::<_, i32>(7)?,     // retry_count
            row.get::<_, Option<String>>(8)?,  // worker_id
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
            .map(|(queue_id, idempotency_key, item_type, op, collection, status, created_at, retry_count, worker_id)| {
                QueueListItemVerbose {
                    queue_id: queue_id.clone(),
                    idempotency_key: idempotency_key.clone(),
                    item_type: item_type.clone(),
                    op: op.clone(),
                    collection: collection.clone(),
                    status: status.clone(),
                    created_at: wqm_common::timestamp_fmt::format_local(created_at),
                    retry_count: *retry_count,
                    worker_id: worker_id.clone().unwrap_or_default(),
                }
            })
            .collect();

        if json {
            output::print_json(&display_items);
        } else {
            output::print_table_auto(&display_items);
            output::info(format!("Showing {} items", display_items.len()));
        }
    } else {
        let display_items: Vec<QueueListItem> = items
            .iter()
            .map(|(queue_id, _idempotency_key, item_type, op, collection, status, created_at, retry_count, _worker_id)| {
                QueueListItem {
                    queue_id: queue_id.clone(),
                    item_type: item_type.clone(),
                    op: op.clone(),
                    collection: collection.clone(),
                    status: format_status(status),
                    age: format_relative_time(created_at),
                    retry_count: *retry_count,
                }
            })
            .collect();

        if json {
            output::print_json(&display_items);
        } else {
            output::print_table_auto(&display_items);
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
               status, branch, payload_json, metadata, created_at,
               updated_at, lease_until, worker_id, retry_count,
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
            status: row.get(6)?,
            branch: row.get(7)?,
            payload_json: row.get(8)?,
            metadata: row.get(9)?,
            created_at: row.get(10)?,
            updated_at: row.get(11)?,
            lease_until: row.get(12)?,
            worker_id: row.get(13)?,
            retry_count: row.get(14)?,
            error_message: row.get(15)?,
            last_error_at: row.get(16)?,
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
                output::kv("Retry Count", &item.retry_count.to_string());
                output::separator();
                output::kv("Created At", &wqm_common::timestamp_fmt::format_local(&item.created_at));
                output::kv("Updated At", &wqm_common::timestamp_fmt::format_local(&item.updated_at));
                if let Some(ref lease) = item.lease_until {
                    output::kv("Lease Until", &wqm_common::timestamp_fmt::format_local(lease));
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
                        output::kv("Last Error At", &wqm_common::timestamp_fmt::format_local(err_at));
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
            output::kv("Oldest Pending Age", &wqm_common::duration_fmt::format_duration(age, 0));
            if let Some(ref id) = summary.oldest_pending_id {
                output::kv("Oldest Pending ID", id);
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
            key,
            total,
            stats.pending,
            stats.in_progress,
            stats.done,
            stats.failed
        );
    }

    Ok(())
}

/// Connect to the state database (read-write for retry/clean)
fn connect_readwrite() -> Result<Connection> {
    let db_path = get_database_path_checked()
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    let conn = Connection::open(&db_path)
        .context(format!("Failed to open state database at {:?}", db_path))?;

    conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")
        .context("Failed to set SQLite pragmas")?;

    Ok(conn)
}

async fn retry(queue_id: Option<String>, all: bool) -> Result<()> {
    if !all && queue_id.is_none() {
        anyhow::bail!("Specify a queue_id or use --all to retry all failed items");
    }

    let conn = connect_readwrite()?;

    if all {
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM unified_queue WHERE status = 'failed'",
            [],
            |row| row.get(0),
        )?;

        if count == 0 {
            output::success("No failed items to retry");
            return Ok(());
        }

        let updated = conn.execute(
            r#"
            UPDATE unified_queue
            SET status = 'pending',
                retry_count = 0,
                error_message = NULL,
                last_error_at = NULL,
                lease_until = NULL,
                worker_id = NULL,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE status = 'failed'
            "#,
            [],
        )?;

        output::success(format!("Reset {} failed items to pending", updated));
    } else {
        let id = queue_id.unwrap();
        let prefix = format!("{}%", id);

        // Find the item first
        let result: Result<(String, String, i32), _> = conn.query_row(
            "SELECT queue_id, status, retry_count FROM unified_queue WHERE queue_id = ?1 OR queue_id LIKE ?2 LIMIT 1",
            params![&id, &prefix],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        );

        match result {
            Ok((found_id, status, retry_count)) => {
                if status != "failed" {
                    output::warning(format!(
                        "Item {} has status '{}', not 'failed'. Use --status filter with list to find failed items.",
                        found_id, status
                    ));
                    return Ok(());
                }

                conn.execute(
                    r#"
                    UPDATE unified_queue
                    SET status = 'pending',
                        retry_count = 0,
                        error_message = NULL,
                        last_error_at = NULL,
                        lease_until = NULL,
                        worker_id = NULL,
                        updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                    WHERE queue_id = ?1
                    "#,
                    params![&found_id],
                )?;

                output::success(format!(
                    "Reset item {} to pending (was retry {})",
                    found_id, retry_count
                ));
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => {
                output::error(format!("Queue item not found: {}", id));
            }
            Err(e) => return Err(e.into()),
        }
    }

    Ok(())
}

async fn clean(days: i64, status_filter: Option<String>, yes: bool) -> Result<()> {
    // Validate status filter
    let valid_statuses = match status_filter.as_deref() {
        Some("done") => vec!["done"],
        Some("failed") => vec!["failed"],
        Some(other) => anyhow::bail!("Invalid status '{}'. Use 'done' or 'failed'.", other),
        None => vec!["done", "failed"],
    };

    let conn = connect_readwrite()?;

    // Count items to be cleaned
    let placeholders = valid_statuses.iter().map(|_| "?").collect::<Vec<_>>().join(",");
    let count_query = format!(
        "SELECT COUNT(*) FROM unified_queue WHERE status IN ({}) AND updated_at < datetime('now', '-{} days')",
        placeholders, days
    );

    let mut stmt = conn.prepare(&count_query)?;
    let params_slice: Vec<&dyn rusqlite::ToSql> = valid_statuses
        .iter()
        .map(|s| s as &dyn rusqlite::ToSql)
        .collect();
    let count: i64 = stmt.query_row(params_slice.as_slice(), |row| row.get(0))?;

    if count == 0 {
        output::success(format!(
            "No {} items older than {} days to clean",
            valid_statuses.join("/"), days
        ));
        return Ok(());
    }

    if !yes {
        output::warning(format!(
            "Will remove {} {} items older than {} days. Use -y to skip this confirmation.",
            count, valid_statuses.join("/"), days
        ));
        return Ok(());
    }

    let delete_query = format!(
        "DELETE FROM unified_queue WHERE status IN ({}) AND updated_at < datetime('now', '-{} days')",
        placeholders, days
    );

    let mut stmt = conn.prepare(&delete_query)?;
    let deleted = stmt.execute(params_slice.as_slice())?;

    output::success(format!("Removed {} old queue items", deleted));

    Ok(())
}

async fn remove(queue_id: &str) -> Result<()> {
    let conn = connect_readwrite()?;

    // Look up item using exact match first, then prefix match (same as show)
    let lookup_query = r#"
        SELECT queue_id, item_type, op, collection, status
        FROM unified_queue
        WHERE queue_id = ? OR queue_id LIKE ? OR idempotency_key LIKE ?
        LIMIT 1
    "#;

    let prefix = format!("{}%", queue_id);
    let mut stmt = conn.prepare(lookup_query)?;
    let result = stmt.query_row(params![queue_id, &prefix, &prefix], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, String>(3)?,
            row.get::<_, String>(4)?,
        ))
    });

    match result {
        Ok((resolved_id, item_type, op, collection, status)) => {
            // Warn if item is currently being processed
            if status == "in_progress" {
                output::warning(format!(
                    "Item {} is currently in_progress — removing may cause the processor to error",
                    &resolved_id[..8.min(resolved_id.len())]
                ));
            }

            // Delete using the resolved exact ID
            conn.execute(
                "DELETE FROM unified_queue WHERE queue_id = ?",
                params![&resolved_id],
            )?;

            output::success(format!(
                "Removed queue item {} ({} {} in {}), was {}",
                &resolved_id[..8.min(resolved_id.len())],
                item_type,
                op,
                collection,
                format_status(&status),
            ));
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
    fn test_format_status() {
        // Just verify it doesn't panic
        let _ = format_status("pending");
        let _ = format_status("in_progress");
        let _ = format_status("done");
        let _ = format_status("failed");
        let _ = format_status("unknown");
    }

    /// Helper: create in-memory database with unified_queue schema
    fn setup_test_db() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE unified_queue (
                queue_id TEXT PRIMARY KEY,
                idempotency_key TEXT UNIQUE NOT NULL,
                item_type TEXT NOT NULL,
                op TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                collection TEXT NOT NULL,
                priority INTEGER DEFAULT 5,
                status TEXT DEFAULT 'pending',
                branch TEXT,
                payload_json TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                retry_count INTEGER DEFAULT 0,
                last_error TEXT,
                leased_by TEXT,
                lease_expires_at TEXT,
                error_message TEXT,
                last_error_at TEXT,
                worker_id TEXT,
                lease_until TEXT
            )",
        )
        .unwrap();
        conn
    }

    /// Helper: insert a test queue item
    fn insert_test_item(conn: &Connection, id: &str, item_type: &str, op: &str, status: &str) {
        conn.execute(
            "INSERT INTO unified_queue (queue_id, idempotency_key, item_type, op, tenant_id, collection, status, payload_json, created_at, updated_at)
             VALUES (?, ?, ?, ?, 'test-tenant', 'projects', ?, '{}', datetime('now'), datetime('now'))",
            params![id, format!("key_{}", id), item_type, op, status],
        )
        .unwrap();
    }

    #[test]
    fn test_remove_exact_match() {
        let conn = setup_test_db();
        insert_test_item(&conn, "abc123def456", "file", "add", "pending");

        // Verify item exists
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM unified_queue WHERE queue_id = 'abc123def456'", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 1);

        // Delete using exact match
        conn.execute("DELETE FROM unified_queue WHERE queue_id = ?", params!["abc123def456"]).unwrap();

        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM unified_queue", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_remove_prefix_match() {
        let conn = setup_test_db();
        insert_test_item(&conn, "abc123def456", "file", "add", "done");

        // Find by prefix
        let prefix = "abc123%";
        let resolved_id: String = conn
            .query_row(
                "SELECT queue_id FROM unified_queue WHERE queue_id = ? OR queue_id LIKE ?",
                params!["abc123", prefix],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(resolved_id, "abc123def456");

        // Delete the resolved item
        let deleted = conn
            .execute("DELETE FROM unified_queue WHERE queue_id = ?", params![&resolved_id])
            .unwrap();
        assert_eq!(deleted, 1);
    }

    #[test]
    fn test_remove_not_found() {
        let conn = setup_test_db();
        insert_test_item(&conn, "abc123def456", "file", "add", "pending");

        let prefix = "xyz%";
        let result = conn.query_row(
            "SELECT queue_id FROM unified_queue WHERE queue_id = ? OR queue_id LIKE ?",
            params!["xyz", prefix],
            |r| r.get::<_, String>(0),
        );
        assert!(matches!(result, Err(rusqlite::Error::QueryReturnedNoRows)));
    }

    #[test]
    fn test_remove_no_cascade_to_other_tables() {
        let conn = setup_test_db();
        // Create tracked_files table to verify no cascade
        conn.execute_batch(
            "CREATE TABLE tracked_files (
                file_id INTEGER PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                file_path TEXT NOT NULL
            )",
        )
        .unwrap();
        conn.execute(
            "INSERT INTO tracked_files (tenant_id, file_path) VALUES ('test-tenant', '/foo/bar.rs')",
            [],
        )
        .unwrap();

        insert_test_item(&conn, "abc123def456", "file", "add", "pending");
        conn.execute("DELETE FROM unified_queue WHERE queue_id = ?", params!["abc123def456"]).unwrap();

        // tracked_files should be unaffected
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM tracked_files", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 1);
    }
}
