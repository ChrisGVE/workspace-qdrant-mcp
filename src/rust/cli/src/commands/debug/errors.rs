//! Error inspection subcommands for debug
//!
//! Provides `errors` (daemon health + error log) and `queue_errors` (failed queue items).

use anyhow::{Context, Result};
use std::process::Command;

use crate::grpc::client::DaemonClient;
use crate::output::{self, ServiceStatus};

/// Show recent errors from all sources.
pub async fn errors(count: usize, component: Option<String>) -> Result<()> {
    output::section("Recent Errors");

    if let Some(comp) = &component {
        output::kv("Component filter", comp);
    }
    output::kv("Max errors", count.to_string());
    output::separator();

    // Try to get errors from daemon health endpoint
    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            output::status_line("Daemon", ServiceStatus::Healthy);
            show_health_errors(&mut client, count, &component).await;
        }
        Err(_) => {
            output::status_line("Daemon", ServiceStatus::Unhealthy);
            output::warning("Cannot check errors without daemon connection");
        }
    }

    // Also check error log file
    output::separator();
    output::info("Checking error log file...");
    show_error_log(count)?;

    Ok(())
}

/// Query daemon health for component errors.
async fn show_health_errors(client: &mut DaemonClient, count: usize, component: &Option<String>) {
    match client.system().health(()).await {
        Ok(response) => {
            let health = response.into_inner();
            let mut error_count = 0;

            for comp_health in &health.components {
                // Filter by component if specified
                if let Some(filter) = component {
                    if !comp_health
                        .component_name
                        .to_lowercase()
                        .contains(&filter.to_lowercase())
                    {
                        continue;
                    }
                }

                // Only show components with issues
                if comp_health.status != 0 && !comp_health.message.is_empty() {
                    error_count += 1;
                    if error_count > count {
                        break;
                    }

                    let status = ServiceStatus::from_proto(comp_health.status);
                    output::status_line(&comp_health.component_name, status);
                    output::kv("  Message", &comp_health.message);
                }
            }

            if error_count == 0 {
                output::success("No errors found in daemon components");
            }
        }
        Err(e) => {
            output::error(format!("Failed to get health: {}", e));
        }
    }
}

/// Check the error log file at the well-known path.
fn show_error_log(count: usize) -> Result<()> {
    let err_log_path = "/tmp/memexd.err.log";
    let path = std::path::Path::new(err_log_path);

    if path.exists() {
        let output_result = Command::new("tail")
            .args(["-n", &count.to_string(), err_log_path])
            .output()?;

        if output_result.status.success() {
            let content = String::from_utf8_lossy(&output_result.stdout);
            if !content.trim().is_empty() {
                output::separator();
                output::info(format!("Last {} lines from {}:", count, err_log_path));
                println!("{}", content);
            } else {
                output::success("Error log is empty");
            }
        }
    } else {
        output::info("No error log file found at /tmp/memexd.err.log");
    }

    Ok(())
}

/// Show queue processing errors from SQLite.
pub async fn queue_errors(count: usize, operation: Option<String>) -> Result<()> {
    output::section("Queue Processing Errors");

    if let Some(op) = &operation {
        output::kv("Operation filter", op);
    }
    output::kv("Max errors", count.to_string());
    output::separator();

    // Query unified queue for failed items
    let db_path = dirs::home_dir()
        .map(|h| h.join(".workspace-qdrant/state.db"))
        .ok_or_else(|| anyhow::anyhow!("Cannot determine home directory"))?;

    if !db_path.exists() {
        output::warning("Database not found - daemon may not have been started");
        return Ok(());
    }

    let conn = rusqlite::Connection::open(&db_path)?;
    conn.execute_batch("PRAGMA busy_timeout=5000;")
        .context("Failed to set busy_timeout")?;

    // Build query with optional operation filter
    let mut query = String::from(
        "SELECT queue_id, item_type, op, tenant_id, last_error, retry_count, updated_at
         FROM unified_queue
         WHERE status = 'failed'",
    );

    if let Some(op) = &operation {
        query.push_str(&format!(" AND op = '{}'", op));
    }

    query.push_str(&format!(" ORDER BY updated_at DESC LIMIT {}", count));

    let mut stmt = conn.prepare(&query)?;
    let mut rows = stmt.query([])?;

    let mut error_count = 0;
    while let Some(row) = rows.next()? {
        error_count += 1;

        let queue_id: String = row.get(0)?;
        let item_type: String = row.get(1)?;
        let op: String = row.get(2)?;
        let tenant_id: String = row.get(3)?;
        let last_error: Option<String> = row.get(4)?;
        let retry_count: i32 = row.get(5)?;
        let updated_at: String = row.get(6)?;

        output::separator();
        output::kv("Queue ID", &queue_id[..8.min(queue_id.len())]);
        output::kv("Type", &item_type);
        output::kv("Operation", &op);
        output::kv("Tenant", &tenant_id);
        output::kv("Retries", retry_count.to_string());
        output::kv("Updated", &updated_at);
        if let Some(err) = last_error {
            output::kv("Error", &err);
        }
    }

    if error_count == 0 {
        output::success("No failed queue items found");
    } else {
        output::separator();
        output::info(format!("Found {} failed queue items", error_count));
        output::info("Use 'wqm queue show <id>' for details");
    }

    Ok(())
}
