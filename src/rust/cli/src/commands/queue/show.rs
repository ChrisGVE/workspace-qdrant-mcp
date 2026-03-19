//! Queue show subcommand

use std::collections::HashMap;

use anyhow::Result;
use colored::Colorize;
use rusqlite::params;

use crate::output;
use crate::output::style::{home_to_tilde, short_id};

use super::db::connect_readonly;
use super::formatters::{extract_subject, format_relative_time, format_status, QueueDetailItem};

pub async fn execute(queue_id: &str, json: bool) -> Result<()> {
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
                let tenant_names = build_tenant_name_map(&conn);
                print_detail(&item, &tenant_names);
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

/// Build a `tenant_id` to project name mapping from `watch_folders`.
///
/// When multiple projects share the same directory name, the tenant_id is
/// appended in parentheses to disambiguate.  Falls back gracefully if the
/// `watch_folders` table does not exist.
fn build_tenant_name_map(conn: &rusqlite::Connection) -> HashMap<String, String> {
    let mut map = HashMap::new();
    let mut name_count: HashMap<String, usize> = HashMap::new();

    let mut entries: Vec<(String, String)> = Vec::new();
    if let Ok(mut stmt) = conn.prepare(
        "SELECT tenant_id, path FROM watch_folders \
         WHERE parent_watch_id IS NULL AND collection = 'projects'",
    ) {
        if let Ok(rows) = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        }) {
            for r in rows.flatten() {
                let name =
                    r.1.rsplit('/')
                        .find(|s| !s.is_empty())
                        .unwrap_or(&r.0)
                        .to_string();
                *name_count.entry(name.clone()).or_default() += 1;
                entries.push((r.0, name));
            }
        }
    }

    for (tenant_id, name) in entries {
        let display = if name_count.get(&name).copied().unwrap_or(0) > 1 {
            format!("{} ({})", name, short_id(&tenant_id))
        } else {
            name
        };
        map.insert(tenant_id, display);
    }

    map
}

/// Resolve a tenant_id to a human-readable project name, falling back to
/// a shortened tenant_id when no mapping exists.
fn resolve_project_name(tenant_id: &str, tenant_names: &HashMap<String, String>) -> String {
    tenant_names
        .get(tenant_id)
        .cloned()
        .unwrap_or_else(|| short_id(tenant_id))
}

/// Look up the full project path for a tenant from `watch_folders`.
fn resolve_project_path(conn: &rusqlite::Connection, tenant_id: &str) -> Option<String> {
    conn.prepare(
        "SELECT path FROM watch_folders \
         WHERE tenant_id = ? AND parent_watch_id IS NULL AND collection = 'projects' \
         LIMIT 1",
    )
    .ok()
    .and_then(|mut stmt| {
        stmt.query_row(params![tenant_id], |row| row.get::<_, String>(0))
            .ok()
    })
}

/// Format a timestamp as "relative (absolute local)" for human-readable display.
fn format_timestamp_rich(utc_str: &str) -> String {
    let local = wqm_common::timestamp_fmt::format_local(utc_str);
    let relative = format_relative_time(utc_str);
    format!("{} ({})", local, relative)
}

fn print_detail(item: &QueueDetailItem, tenant_names: &HashMap<String, String>) {
    let conn = connect_readonly().ok();
    let project_name = resolve_project_name(&item.tenant_id, tenant_names);
    let subject = extract_subject(&item.item_type, &item.payload_json);

    // ── Identity ────────────────────────────────────────────────────
    output::section("Queue Item Details");
    output::kv("Queue ID", &item.queue_id);
    output::kv("Idempotency Key", &item.idempotency_key);

    // ── Operation ───────────────────────────────────────────────────
    output::separator();
    output::kv("Item Type", &item.item_type);
    output::kv("Operation", &item.op);
    output::kv("Collection", &item.collection);
    output::kv("Branch", &item.branch);

    // ── Project ─────────────────────────────────────────────────────
    output::separator();
    output::kv("Project", &project_name);
    output::kv("Tenant ID", output::dim_style(&item.tenant_id));
    if let Some(ref conn) = conn {
        if let Some(path) = resolve_project_path(conn, &item.tenant_id) {
            output::kv("Project Path", home_to_tilde(&path));
        }
    }
    if !subject.is_empty() {
        output::kv("Subject", &subject);
    }

    // ── Status ──────────────────────────────────────────────────────
    output::separator();
    output::kv("Status", format_status(&item.status));
    output::kv("Retry Count", item.retry_count.to_string());

    // ── Timestamps ──────────────────────────────────────────────────
    output::separator();
    output::kv("Created At", format_timestamp_rich(&item.created_at));
    output::kv("Updated At", format_timestamp_rich(&item.updated_at));
    if let Some(ref lease) = item.lease_until {
        output::kv("Lease Until", format_timestamp_rich(lease));
    }
    if let Some(ref worker) = item.worker_id {
        output::kv("Worker ID", worker);
    }

    // ── Errors ──────────────────────────────────────────────────────
    print_errors(item);

    // ── Payload ─────────────────────────────────────────────────────
    print_payload(item);

    // ── Metadata ────────────────────────────────────────────────────
    print_metadata(item);
}

fn print_errors(item: &QueueDetailItem) {
    if item.error_message.is_none() && item.last_error_at.is_none() {
        return;
    }

    output::section("Error Details");
    if let Some(ref err) = item.error_message {
        // Show full error message, not truncated
        if err.lines().count() > 1 {
            println!("{}", "Error Message:".bold());
            for line in err.lines() {
                println!("  {}", line.red());
            }
        } else {
            output::kv("Error Message", err.red());
        }
    }
    if let Some(ref err_at) = item.last_error_at {
        output::kv("Last Error At", format_timestamp_rich(err_at));
    }
}

fn print_payload(item: &QueueDetailItem) {
    output::section("Payload");
    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&item.payload_json) {
        if let Some(obj) = parsed.as_object() {
            for (key, value) in obj {
                let display_value = match value {
                    serde_json::Value::String(s) => {
                        // Show paths with ~ substitution
                        if key.ends_with("_path") || key == "path" {
                            home_to_tilde(s)
                        } else {
                            s.clone()
                        }
                    }
                    serde_json::Value::Null => "null".dimmed().to_string(),
                    other => other.to_string(),
                };
                output::kv(format!("  {}", key), display_value);
            }
        } else if let Ok(pretty) = serde_json::to_string_pretty(&parsed) {
            for line in pretty.lines() {
                println!("  {}", line);
            }
        }
    } else {
        println!("  {}", item.payload_json);
    }
}

fn print_metadata(item: &QueueDetailItem) {
    let Some(ref meta) = item.metadata else {
        return;
    };

    output::section("Metadata");
    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(meta) {
        if let Some(obj) = parsed.as_object() {
            for (key, value) in obj {
                let display_value = match value {
                    serde_json::Value::String(s) => s.clone(),
                    serde_json::Value::Null => "null".dimmed().to_string(),
                    other => other.to_string(),
                };
                output::kv(format!("  {}", key), display_value);
            }
        } else if let Ok(pretty) = serde_json::to_string_pretty(&parsed) {
            for line in pretty.lines() {
                println!("  {}", line);
            }
        }
    } else {
        println!("  {}", meta);
    }
}
