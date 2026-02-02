//! Status command - consolidated monitoring
//!
//! Phase 1 HIGH priority command for consolidated status monitoring.
//! Replaces old: observability, queue, messages, errors commands.
//! Subcommands: default, history, queue, watch, performance, live,
//!              messages (list/clear), errors, health

use anyhow::Result;
use clap::{Args, Subcommand};

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
    Health,
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
    // Handle flags for default status
    if args.queue || args.watch || args.performance {
        return default_status(args.queue, args.watch, args.performance).await;
    }

    // Handle subcommands
    match args.command {
        None => default_status(false, false, false).await,
        Some(StatusCommand::History { range }) => history(&range).await,
        Some(StatusCommand::Queue { verbose }) => queue(verbose).await,
        Some(StatusCommand::Watch) => watch().await,
        Some(StatusCommand::Performance) => performance().await,
        Some(StatusCommand::Live { interval }) => live(interval).await,
        Some(StatusCommand::Messages { action }) => messages(action).await,
        Some(StatusCommand::Errors { limit }) => errors(limit).await,
        Some(StatusCommand::Health) => health().await,
    }
}

async fn default_status(show_queue: bool, show_watch: bool, show_performance: bool) -> Result<()> {
    output::section("System Status");

    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            output::status_line("Daemon", ServiceStatus::Healthy);

            // Get comprehensive status
            match client.system().get_status(()).await {
                Ok(response) => {
                    let status = response.into_inner();
                    let overall = ServiceStatus::from_proto(status.status);
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
                    output::warning(format!("Could not get status: {}", e));
                }
            }
        }
        Err(_) => {
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

    // Note: GetMetrics returns current point-in-time metrics, not historical
    // Historical metrics would require additional storage/implementation
    output::info("Historical metrics require time-series storage.");
    output::info("Showing current metrics snapshot instead:");

    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            match client.system().get_metrics(()).await {
                Ok(response) => {
                    let metrics_resp = response.into_inner();
                    for metric in &metrics_resp.metrics {
                        output::kv(&metric.name, &format!("{:.2}", metric.value));
                    }
                }
                Err(e) => {
                    output::error(format!("Failed to get metrics: {}", e));
                }
            }
        }
        Err(_) => {
            output::error("Cannot connect to daemon");
        }
    }

    Ok(())
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
                        output::info("  (Queue items stored in SQLite ingestion_queue table)");
                        output::info("  Use: sqlite3 ~/.local/share/workspace-qdrant/state.db 'SELECT * FROM ingestion_queue'");
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

async fn health() -> Result<()> {
    output::section("System Health");

    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            output::status_line("Daemon Connection", ServiceStatus::Healthy);

            match client.system().health(()).await {
                Ok(response) => {
                    let health = response.into_inner();
                    let status = ServiceStatus::from_proto(health.status);
                    output::status_line("Overall Health", status);

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
                Err(e) => {
                    output::status_line("Health Check", ServiceStatus::Unknown);
                    output::warning(format!("Could not get health: {}", e));
                }
            }
        }
        Err(_) => {
            output::status_line("Daemon Connection", ServiceStatus::Unhealthy);
            output::error("Daemon not running");
            output::info("Start with: wqm service start");
        }
    }

    Ok(())
}
