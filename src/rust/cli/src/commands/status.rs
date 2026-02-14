//! Status command - consolidated monitoring
//!
//! Phase 1 HIGH priority command for consolidated status monitoring.
//! Replaces old: observability, queue, messages, errors commands.
//! Subcommands: default, history, queue, watch, performance, live,
//!              messages (list/clear), errors, health

use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use rusqlite::Connection;
use serde::Serialize;
use wqm_common::timestamps;

use crate::config::get_database_path_checked;
use crate::grpc::client::DaemonClient;
use crate::output::{self, ServiceStatus};

/// Status command arguments
#[derive(Args)]
pub struct StatusArgs {
    #[command(subcommand)]
    command: Option<StatusCommand>,

    /// Show queue status
    #[arg(long)]
    queue: bool,

    /// Show watch status
    #[arg(long)]
    watch: bool,

    /// Show performance metrics
    #[arg(long)]
    performance: bool,

    /// Output as JSON
    #[arg(long)]
    json: bool,
}

/// Status subcommands
#[derive(Subcommand)]
enum StatusCommand {
    /// Show historical metrics
    History {
        /// Time range (1h, 24h, 7d)
        #[arg(short, long, default_value = "1h")]
        range: String,
    },

    /// Show ingestion queue details
    Queue {
        /// Show detailed queue items
        #[arg(short, long)]
        verbose: bool,
    },

    /// Show file watcher status
    Watch,

    /// Show performance metrics
    Performance,

    /// Live updating dashboard
    Live {
        /// Refresh interval in seconds
        #[arg(short, long, default_value = "2")]
        interval: u64,
    },

    /// Message management
    Messages {
        #[command(subcommand)]
        action: Option<MessageAction>,
    },

    /// Show recent errors
    Errors {
        /// Number of errors to show
        #[arg(short = 'n', long, default_value = "10")]
        limit: usize,
    },

    /// Show system health
    Health {
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },
}

/// Message subcommands
#[derive(Subcommand)]
enum MessageAction {
    /// List all messages
    List,
    /// Clear all messages
    Clear,
}

/// Execute status command
pub async fn execute(args: StatusArgs) -> Result<()> {
    let json = args.json;

    // Handle flags for default status
    if args.queue || args.watch || args.performance {
        return default_status(args.queue, args.watch, args.performance, json).await;
    }

    // Handle subcommands
    match args.command {
        None => default_status(false, false, false, json).await,
        Some(StatusCommand::History { range }) => history(&range).await,
        Some(StatusCommand::Queue { verbose }) => queue(verbose).await,
        Some(StatusCommand::Watch) => watch().await,
        Some(StatusCommand::Performance) => performance().await,
        Some(StatusCommand::Live { interval }) => live(interval).await,
        Some(StatusCommand::Messages { action }) => messages(action).await,
        Some(StatusCommand::Errors { limit }) => errors(limit).await,
        Some(StatusCommand::Health { json: sub_json }) => health(json || sub_json).await,
    }
}

/// JSON-serializable system status
#[derive(Serialize)]
struct SystemStatusJson {
    connected: bool,
    status: String,
    collections: i32,
    documents: i32,
    active_projects: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pending_operations: Option<i32>,
}

/// JSON-serializable health status
#[derive(Serialize)]
struct HealthStatusJson {
    connected: bool,
    health: String,
    components: Vec<HealthComponentJson>,
}

/// JSON-serializable health component
#[derive(Serialize)]
struct HealthComponentJson {
    name: String,
    status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<String>,
}

fn status_label(s: ServiceStatus) -> &'static str {
    match s {
        ServiceStatus::Healthy => "healthy",
        ServiceStatus::Degraded => "degraded",
        ServiceStatus::Unhealthy => "unhealthy",
        ServiceStatus::Unknown => "unknown",
    }
}

async fn default_status(show_queue: bool, show_watch: bool, show_performance: bool, json: bool) -> Result<()> {
    if !json {
        output::section("System Status");
    }

    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            if !json {
                output::status_line("Daemon", ServiceStatus::Healthy);
            }

            // Get comprehensive status
            match client.system().get_status(()).await {
                Ok(response) => {
                    let status = response.into_inner();
                    let overall = ServiceStatus::from_proto(status.status);

                    if json {
                        let pending = status.metrics.as_ref().map(|m| m.pending_operations);
                        let json_out = SystemStatusJson {
                            connected: true,
                            status: status_label(overall).to_string(),
                            collections: status.total_collections,
                            documents: status.total_documents,
                            active_projects: status.active_projects,
                            pending_operations: pending,
                        };
                        output::print_json(&json_out);
                        return Ok(());
                    }

                    output::status_line("Overall", overall);

                    output::separator();
                    output::kv("Collections", &status.total_collections.to_string());
                    output::kv("Documents", &status.total_documents.to_string());
                    output::kv("Active Projects", &status.active_projects.len().to_string());

                    if let Some(metrics) = &status.metrics {
                        output::kv("Pending Operations", &metrics.pending_operations.to_string());
                    }
                }
                Err(e) => {
                    if json {
                        let json_out = SystemStatusJson {
                            connected: true,
                            status: "unknown".to_string(),
                            collections: 0,
                            documents: 0,
                            active_projects: Vec::new(),
                            pending_operations: None,
                        };
                        output::print_json(&json_out);
                        return Ok(());
                    }
                    output::warning(format!("Could not get status: {}", e));
                }
            }
        }
        Err(_) => {
            if json {
                let json_out = SystemStatusJson {
                    connected: false,
                    status: "unhealthy".to_string(),
                    collections: 0,
                    documents: 0,
                    active_projects: Vec::new(),
                    pending_operations: None,
                };
                output::print_json(&json_out);
                return Ok(());
            }
            output::status_line("Daemon", ServiceStatus::Unhealthy);
            output::error("Daemon not running. Start with: wqm service start");
        }
    }

    if show_queue {
        output::separator();
        queue(false).await?;
    }

    if show_watch {
        output::separator();
        watch().await?;
    }

    if show_performance {
        output::separator();
        performance().await?;
    }

    Ok(())
}

async fn history(range: &str) -> Result<()> {
    output::section(format!("Metrics History ({})", range));

    let seconds = parse_range_to_seconds(range);
    let conn = connect_history_readonly()?;

    // Check if metrics_history table exists
    let table_exists: bool = conn.query_row(
        "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='table' AND name='metrics_history'",
        [],
        |row| row.get(0),
    ).unwrap_or(false);

    if !table_exists {
        output::warning("Metrics history table not found. Daemon needs to run with schema v5+.");
        output::info("Start the daemon to enable metrics collection.");
        return Ok(());
    }

    let cutoff = chrono::Utc::now() - chrono::Duration::seconds(seconds);
    let cutoff_str = timestamps::format_utc(&cutoff);

    // Get available metrics in the time range
    let mut stmt = conn.prepare(
        "SELECT DISTINCT metric_name FROM metrics_history \
         WHERE timestamp >= ?1 AND aggregation_period = 'raw' \
         ORDER BY metric_name"
    )?;

    let metric_names: Vec<String> = stmt.query_map([&cutoff_str], |row| {
        row.get(0)
    })?.filter_map(|r| r.ok()).collect();

    if metric_names.is_empty() {
        output::info("No historical metrics found in the requested time range.");
        output::info("The daemon collects metrics every 60 seconds.");
        return Ok(());
    }

    // Show summary for each metric
    for name in &metric_names {
        let stats: Result<(f64, f64, f64, i64, f64), _> = conn.query_row(
            "SELECT AVG(metric_value), MIN(metric_value), MAX(metric_value), \
             COUNT(*), \
             (SELECT metric_value FROM metrics_history \
              WHERE metric_name = ?1 AND timestamp >= ?2 AND aggregation_period = 'raw' \
              ORDER BY timestamp DESC LIMIT 1) \
             FROM metrics_history \
             WHERE metric_name = ?1 AND timestamp >= ?2 AND aggregation_period = 'raw'",
            rusqlite::params![name, &cutoff_str],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?)),
        );

        match stats {
            Ok((avg, min, max, count, latest)) => {
                output::kv(
                    name,
                    &format!(
                        "latest={:.1}  avg={:.1}  min={:.1}  max={:.1}  ({} samples)",
                        latest, avg, min, max, count
                    ),
                );
            }
            Err(_) => {
                output::kv(name, "no data");
            }
        }
    }

    output::separator();
    output::info(format!("{} metrics tracked over {}", metric_names.len(), range));

    Ok(())
}

/// Parse range string (1h, 24h, 7d, 30d) to seconds
fn parse_range_to_seconds(range: &str) -> i64 {
    let range = range.trim().to_lowercase();
    if let Some(hours) = range.strip_suffix('h') {
        hours.parse::<i64>().unwrap_or(1) * 3600
    } else if let Some(days) = range.strip_suffix('d') {
        days.parse::<i64>().unwrap_or(1) * 86400
    } else if let Some(minutes) = range.strip_suffix('m') {
        minutes.parse::<i64>().unwrap_or(60) * 60
    } else {
        3600 // default 1h
    }
}

fn connect_history_readonly() -> Result<Connection> {
    let db_path = get_database_path_checked()
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    let conn = Connection::open_with_flags(
        &db_path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .context(format!("Failed to open state database at {:?}", db_path))?;

    Ok(conn)
}

async fn queue(verbose: bool) -> Result<()> {
    output::section("Ingestion Queue");

    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            match client.system().get_metrics(()).await {
                Ok(response) => {
                    let metrics_resp = response.into_inner();

                    // Extract queue-related metrics
                    let mut pending = 0.0;
                    let mut processed = 0.0;
                    let mut failed = 0.0;

                    for metric in &metrics_resp.metrics {
                        match metric.name.as_str() {
                            "queue_pending" => pending = metric.value,
                            "queue_processed" => processed = metric.value,
                            "queue_failed" => failed = metric.value,
                            _ => {}
                        }
                    }

                    output::kv("Pending", &(pending as i64).to_string());
                    output::kv("Processed", &(processed as i64).to_string());
                    output::kv("Failed", &(failed as i64).to_string());

                    if verbose {
                        output::separator();
                        output::info("Queue Details:");
                        output::info("  (Queue items stored in SQLite unified_queue table)");
                        output::info("  Use: sqlite3 ~/.workspace-qdrant/state.db 'SELECT * FROM unified_queue'");
                    }
                }
                Err(e) => {
                    output::error(format!("Failed to get queue status: {}", e));
                }
            }
        }
        Err(_) => {
            output::error("Cannot connect to daemon");
        }
    }

    Ok(())
}

async fn watch() -> Result<()> {
    output::section("Watch Status");

    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            match client.system().get_status(()).await {
                Ok(response) => {
                    let status = response.into_inner();

                    if status.active_projects.is_empty() {
                        output::info("No active projects being watched");
                    } else {
                        output::info("Active Projects:");
                        for project in &status.active_projects {
                            println!("  • {}", project);
                        }
                    }
                }
                Err(e) => {
                    output::error(format!("Failed to get watch status: {}", e));
                }
            }

            // Note: Detailed watch folder configuration is in SQLite
            output::separator();
            output::info("Watch folders configured in SQLite:");
            output::info("  Use: sqlite3 ~/.local/share/workspace-qdrant/state.db 'SELECT watch_id, path, enabled FROM watch_folders'");
        }
        Err(_) => {
            output::error("Cannot connect to daemon");
        }
    }

    Ok(())
}

async fn performance() -> Result<()> {
    output::section("Performance Metrics");

    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            match client.system().get_status(()).await {
                Ok(response) => {
                    let status = response.into_inner();

                    if let Some(metrics) = status.metrics {
                        output::kv("CPU Usage", &format!("{:.1}%", metrics.cpu_usage_percent));
                        output::kv("Memory Used", &format_bytes(metrics.memory_usage_bytes));
                        output::kv("Memory Total", &format_bytes(metrics.memory_total_bytes));
                        output::kv("Disk Used", &format_bytes(metrics.disk_usage_bytes));
                        output::kv("Disk Total", &format_bytes(metrics.disk_total_bytes));
                        output::separator();
                        output::kv("Active Connections", &metrics.active_connections.to_string());
                        output::kv("Pending Operations", &metrics.pending_operations.to_string());
                    } else {
                        output::warning("Metrics not available from daemon");
                    }
                }
                Err(e) => {
                    output::error(format!("Failed to get performance metrics: {}", e));
                }
            }
        }
        Err(_) => {
            output::error("Cannot connect to daemon");
        }
    }

    Ok(())
}

fn format_bytes(bytes: i64) -> String {
    const KB: i64 = 1024;
    const MB: i64 = KB * 1024;
    const GB: i64 = MB * 1024;

    if bytes < KB {
        format!("{} B", bytes)
    } else if bytes < MB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else if bytes < GB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    }
}

async fn live(interval: u64) -> Result<()> {
    output::info(format!(
        "Live dashboard (refresh every {}s, Ctrl+C to exit)",
        interval
    ));
    output::separator();

    loop {
        // Clear screen and move cursor to top
        print!("\x1B[2J\x1B[H");

        output::section("Live Dashboard");

        match DaemonClient::connect_default().await {
            Ok(mut client) => {
                match client.system().get_status(()).await {
                    Ok(response) => {
                        let status = response.into_inner();
                        let overall = ServiceStatus::from_proto(status.status);

                        output::status_line("Daemon", ServiceStatus::Healthy);
                        output::status_line("Overall", overall);
                        output::separator();

                        output::kv("Collections", &status.total_collections.to_string());
                        output::kv("Documents", &status.total_documents.to_string());
                        output::kv("Active Projects", &status.active_projects.len().to_string());

                        if let Some(metrics) = status.metrics {
                            output::separator();
                            output::kv("CPU", &format!("{:.1}%", metrics.cpu_usage_percent));
                            output::kv("Memory", &format_bytes(metrics.memory_usage_bytes));
                            output::kv("Pending Ops", &metrics.pending_operations.to_string());
                            output::kv("Connections", &metrics.active_connections.to_string());
                        }
                    }
                    Err(_) => {
                        output::warning("Could not fetch status");
                    }
                }
            }
            Err(_) => {
                output::status_line("Daemon", ServiceStatus::Unhealthy);
                output::error("Not connected");
            }
        }

        output::separator();
        output::info(format!("Refreshing every {}s (Ctrl+C to exit)...", interval));

        tokio::time::sleep(tokio::time::Duration::from_secs(interval)).await;
    }
}

async fn messages(action: Option<MessageAction>) -> Result<()> {
    match action {
        None | Some(MessageAction::List) => {
            output::section("System Messages");
            // Messages would come from metrics or a dedicated message service
            // For now, show info about where to find logs
            output::info("System messages available in daemon logs:");
            output::info("  macOS: /tmp/memexd.out.log, /tmp/memexd.err.log");
            output::info("  Linux: journalctl --user -u memexd");
            output::separator();
            output::info("Use 'wqm service logs' to view recent messages");
        }
        Some(MessageAction::Clear) => {
            output::info("Message clearing not supported - logs are managed by the system");
        }
    }
    Ok(())
}

async fn errors(limit: usize) -> Result<()> {
    output::section(format!("Recent Errors (last {})", limit));

    // Errors would come from metrics or a dedicated error tracking
    output::info("Error tracking available via daemon logs:");
    output::info(&format!("  Use: wqm service logs -n {}", limit));
    output::info("  Or: grep -i error /tmp/memexd.err.log | tail -n {}", );

    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            match client.system().get_metrics(()).await {
                Ok(response) => {
                    let metrics_resp = response.into_inner();

                    for metric in &metrics_resp.metrics {
                        if metric.name.contains("error") || metric.name.contains("failed") {
                            output::kv(&metric.name, &format!("{:.0}", metric.value));
                        }
                    }
                }
                Err(_) => {}
            }
        }
        Err(_) => {
            output::warning("Cannot connect to daemon for error metrics");
        }
    }

    Ok(())
}

async fn health(json: bool) -> Result<()> {
    if !json {
        output::section("System Health");
    }

    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            if !json {
                output::status_line("Daemon Connection", ServiceStatus::Healthy);
            }

            match client.system().health(()).await {
                Ok(response) => {
                    let health = response.into_inner();
                    let overall = ServiceStatus::from_proto(health.status);

                    if json {
                        let components: Vec<HealthComponentJson> = health
                            .components
                            .iter()
                            .map(|c| HealthComponentJson {
                                name: c.component_name.clone(),
                                status: status_label(ServiceStatus::from_proto(c.status))
                                    .to_string(),
                                message: if c.message.is_empty() {
                                    None
                                } else {
                                    Some(c.message.clone())
                                },
                            })
                            .collect();
                        let json_out = HealthStatusJson {
                            connected: true,
                            health: status_label(overall).to_string(),
                            components,
                        };
                        output::print_json(&json_out);
                    } else {
                        output::status_line("Overall Health", overall);

                        if !health.components.is_empty() {
                            output::separator();
                            for comp in health.components {
                                let comp_status = ServiceStatus::from_proto(comp.status);
                                output::status_line(&comp.component_name, comp_status);
                                if !comp.message.is_empty() {
                                    output::kv("  Message", &comp.message);
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    if json {
                        let json_out = HealthStatusJson {
                            connected: true,
                            health: "unknown".to_string(),
                            components: Vec::new(),
                        };
                        output::print_json(&json_out);
                    } else {
                        output::status_line("Health Check", ServiceStatus::Unknown);
                        output::warning(format!("Could not get health: {}", e));
                    }
                }
            }
        }
        Err(_) => {
            if json {
                let json_out = HealthStatusJson {
                    connected: false,
                    health: "unhealthy".to_string(),
                    components: Vec::new(),
                };
                output::print_json(&json_out);
            } else {
                output::status_line("Daemon Connection", ServiceStatus::Unhealthy);
                output::error("Daemon not running");
                output::info("Start with: wqm service start");
            }
        }
    }

    Ok(())
}
