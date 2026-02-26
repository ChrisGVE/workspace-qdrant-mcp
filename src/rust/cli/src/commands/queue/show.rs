//! Queue show subcommand

use anyhow::Result;
use colored::Colorize;
use rusqlite::params;

use crate::output;

use super::db::connect_readonly;
use super::formatters::{QueueDetailItem, format_status};

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
                print_detail(&item);
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

fn print_detail(item: &QueueDetailItem) {
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
    output::kv(
        "Created At",
        &wqm_common::timestamp_fmt::format_local(&item.created_at),
    );
    output::kv(
        "Updated At",
        &wqm_common::timestamp_fmt::format_local(&item.updated_at),
    );
    if let Some(ref lease) = item.lease_until {
        output::kv(
            "Lease Until",
            &wqm_common::timestamp_fmt::format_local(lease),
        );
    }
    if let Some(ref worker) = item.worker_id {
        output::kv("Worker ID", worker);
    }

    print_errors(item);
    print_payload(item);
    print_metadata(item);
}

fn print_errors(item: &QueueDetailItem) {
    if item.error_message.is_some() || item.last_error_at.is_some() {
        output::separator();
        if let Some(ref err) = item.error_message {
            output::kv("Error Message", err);
        }
        if let Some(ref err_at) = item.last_error_at {
            output::kv(
                "Last Error At",
                &wqm_common::timestamp_fmt::format_local(err_at),
            );
        }
    }
}

fn print_payload(item: &QueueDetailItem) {
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
}

fn print_metadata(item: &QueueDetailItem) {
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
