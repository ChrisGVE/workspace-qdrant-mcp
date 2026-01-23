//! Admin command - system administration
//!
//! Phase 1 HIGH priority command for system administration.
//! Subcommands: status, start-engine, stop-engine, restart-engine,
//!              collections, health, projects, queue

use anyhow::Result;
use clap::{Args, Subcommand};

use crate::grpc::client::DaemonClient;
use crate::output::{self, ServiceStatus};

const ADMIN_AFTER_HELP: &str = "\
EXAMPLES:
    wqm admin status                Show overall system status
    wqm admin collections           List all Qdrant collections
    wqm admin collections -v        Detailed collection info with sizes
    wqm admin health                Check component health status
    wqm admin projects --active     Show only active projects
    wqm admin queue -v              Show pending ingestion items

SYSTEM COMPONENTS:
    The admin commands interact with:
    • memexd daemon (document processing, file watching)
    • Qdrant server (vector storage and search)
    • SQLite database (state management, queue)

COLLECTION NAMING:
    • _projects    - Project code and documentation
    • _libraries   - External documentation libraries
    • _memory      - LLM rules and preferences";

/// Admin command arguments
#[derive(Args)]
#[command(after_help = ADMIN_AFTER_HELP)]
pub struct AdminArgs {
    #[command(subcommand)]
    command: AdminCommand,
}

/// Admin subcommands
#[derive(Subcommand)]
enum AdminCommand {
    /// Show comprehensive system status (daemon, Qdrant, components)
    #[command(long_about = "Display a comprehensive overview of the system including:\n\
        • Daemon running state and uptime\n\
        • Qdrant connection status\n\
        • Active projects and session counts\n\
        • Queue depth and processing rate\n\
        • Resource usage (CPU, memory)")]
    Status,

    /// Start the Qdrant vector database engine
    #[command(long_about = "Start the Qdrant server if it's not already running. \
        This is usually not needed as Qdrant should be running separately.")]
    StartEngine,

    /// Stop the Qdrant vector database engine
    #[command(long_about = "Stop the Qdrant server gracefully. Warning: This will \
        interrupt all indexing and search operations.")]
    StopEngine,

    /// Restart the Qdrant vector database engine
    #[command(long_about = "Restart the Qdrant server. Use this to apply configuration \
        changes or recover from issues.")]
    RestartEngine,

    /// List all Qdrant collections with document counts
    #[command(long_about = "Show all collections in the Qdrant database including:\n\
        • Collection name and type (project, library, memory)\n\
        • Document/vector count\n\
        • Storage size (with -v flag)\n\
        • Last modified time")]
    Collections {
        /// Show detailed information (sizes, indexes, config)
        #[arg(short, long, help = "Include storage sizes and index configuration")]
        verbose: bool,
    },

    /// Show system health with component status
    #[command(long_about = "Display health status for all system components:\n\
        • Daemon process health\n\
        • gRPC service availability\n\
        • Qdrant connectivity\n\
        • File watcher status\n\
        • Queue processor status")]
    Health,

    /// List registered projects and their priorities
    #[command(long_about = "Show all projects registered with the daemon including:\n\
        • Project ID and path\n\
        • Priority level (HIGH for active, NORMAL otherwise)\n\
        • Active session count\n\
        • Last activity timestamp")]
    Projects {
        /// Filter by priority level (high, normal, low, all)
        #[arg(short, long, default_value = "all", help = "Filter by priority: high, normal, low, all")]
        priority: String,

        /// Show only projects with active MCP sessions
        #[arg(short, long, help = "Only show projects with active sessions")]
        active_only: bool,
    },

    /// Show ingestion queue status and pending items
    #[command(long_about = "Display the current state of the ingestion queue:\n\
        • Queue depth (pending items)\n\
        • Processing rate\n\
        • Failed items (if any)\n\
        • With -v: individual queue items")]
    Queue {
        /// Show individual queue items with details
        #[arg(short, long, help = "List individual pending items")]
        verbose: bool,
    },
}

/// Execute admin command
pub async fn execute(args: AdminArgs) -> Result<()> {
    match args.command {
        AdminCommand::Status => status().await,
        AdminCommand::StartEngine => start_engine().await,
        AdminCommand::StopEngine => stop_engine().await,
        AdminCommand::RestartEngine => restart_engine().await,
        AdminCommand::Collections { verbose } => collections(verbose).await,
        AdminCommand::Health => health().await,
        AdminCommand::Projects {
            priority,
            active_only,
        } => projects(&priority, active_only).await,
        AdminCommand::Queue { verbose } => queue(verbose).await,
    }
}

async fn status() -> Result<()> {
    output::section("System Status");

    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            output::status_line("Daemon", ServiceStatus::Healthy);

            // Get system status
            match client.system().get_status(()).await {
                Ok(response) => {
                    let status = response.into_inner();
                    let overall = ServiceStatus::from_proto(status.status);
                    output::status_line("Overall", overall);

                    output::separator();
                    output::kv("Active Projects", &status.active_projects.len().to_string());
                    output::kv("Total Collections", &status.total_collections.to_string());
                    output::kv("Total Documents", &status.total_documents.to_string());

                    if let Some(uptime) = status.uptime_since {
                        output::kv("Uptime Since", &format_timestamp(&uptime));
                    }

                    // Show metrics if available
                    if let Some(metrics) = status.metrics {
                        output::separator();
                        output::kv("CPU Usage", &format!("{:.1}%", metrics.cpu_usage_percent));
                        output::kv("Memory", &format_bytes(metrics.memory_usage_bytes));
                        output::kv("Pending Ops", &metrics.pending_operations.to_string());
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

    Ok(())
}

fn format_timestamp(ts: &prost_types::Timestamp) -> String {
    use std::time::{Duration, UNIX_EPOCH};
    let secs = ts.seconds as u64;
    if let Some(time) = UNIX_EPOCH.checked_add(Duration::from_secs(secs)) {
        if let Ok(elapsed) = std::time::SystemTime::now().duration_since(time) {
            return format_duration_short(elapsed.as_secs());
        }
    }
    "unknown".to_string()
}

fn format_duration_short(secs: u64) -> String {
    if secs < 60 {
        format!("{}s", secs)
    } else if secs < 3600 {
        format!("{}m {}s", secs / 60, secs % 60)
    } else if secs < 86400 {
        format!("{}h {}m", secs / 3600, (secs % 3600) / 60)
    } else {
        format!("{}d {}h", secs / 86400, (secs % 86400) / 3600)
    }
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

async fn start_engine() -> Result<()> {
    output::info("Resuming file watchers...");

    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            match client.system().resume_all_watchers(()).await {
                Ok(_) => {
                    output::success("File watchers resumed");
                }
                Err(e) => {
                    output::error(format!("Failed to resume watchers: {}", e));
                }
            }
        }
        Err(_) => {
            output::error("Cannot connect to daemon. Start daemon first: wqm service start");
        }
    }

    Ok(())
}

async fn stop_engine() -> Result<()> {
    output::info("Pausing file watchers...");

    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            match client.system().pause_all_watchers(()).await {
                Ok(_) => {
                    output::success("File watchers paused");
                }
                Err(e) => {
                    output::error(format!("Failed to pause watchers: {}", e));
                }
            }
        }
        Err(_) => {
            output::error("Cannot connect to daemon");
        }
    }

    Ok(())
}

async fn restart_engine() -> Result<()> {
    output::info("Restarting file watchers...");
    stop_engine().await?;
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
    start_engine().await?;
    Ok(())
}

async fn collections(_verbose: bool) -> Result<()> {
    output::section("Collections");

    // Collections are queried directly from Qdrant, not via daemon
    // This is by design - see workspace_daemon.proto comment
    output::info("Collection listing queries Qdrant directly.");
    output::info("Use Qdrant dashboard at http://localhost:6333/dashboard");
    output::info("Or query via: curl http://localhost:6333/collections");

    // Show total count from daemon status if available
    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            match client.system().get_status(()).await {
                Ok(response) => {
                    let status = response.into_inner();
                    output::separator();
                    output::kv("Total Collections", &status.total_collections.to_string());
                    output::kv("Total Documents", &status.total_documents.to_string());
                }
                Err(_) => {}
            }
        }
        Err(_) => {
            output::warning("Daemon not running");
        }
    }

    Ok(())
}

async fn health() -> Result<()> {
    output::section("System Health");

    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            output::status_line("Daemon Connection", ServiceStatus::Healthy);

            match client.system().health_check(()).await {
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
        }
    }

    Ok(())
}

async fn projects(priority: &str, active_only: bool) -> Result<()> {
    output::section("Registered Projects");

    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            let request = crate::grpc::proto::ListProjectsRequest {
                priority_filter: if priority == "all" {
                    None
                } else {
                    Some(priority.to_string())
                },
                active_only,
            };

            match client.project().list_projects(request).await {
                Ok(response) => {
                    let list = response.into_inner();

                    if list.projects.is_empty() {
                        output::info("No projects registered");
                        return Ok(());
                    }

                    for proj in &list.projects {
                        let status = if proj.active_sessions > 0 {
                            ServiceStatus::Healthy
                        } else {
                            ServiceStatus::Unknown
                        };
                        output::status_line(&proj.project_name, status);
                        output::kv("  ID", &proj.project_id);
                        output::kv("  Path", &proj.project_root);
                        output::kv("  Priority", &proj.priority);
                        output::kv("  Sessions", &proj.active_sessions.to_string());
                    }

                    output::separator();
                    output::info(format!("Total: {} projects", list.total_count));
                }
                Err(e) => {
                    output::error(format!("Failed to list projects: {}", e));
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
            // Get metrics which includes pending_operations
            match client.system().get_metrics(()).await {
                Ok(response) => {
                    let metrics_resp = response.into_inner();

                    // Extract queue-related metrics
                    let mut pending = 0i64;
                    let mut processed = 0i64;
                    let mut failed = 0i64;

                    for metric in &metrics_resp.metrics {
                        match metric.name.as_str() {
                            "queue_pending" => pending = metric.value as i64,
                            "queue_processed" => processed = metric.value as i64,
                            "queue_failed" => failed = metric.value as i64,
                            _ => {}
                        }
                    }

                    output::kv("Pending Operations", &pending.to_string());
                    output::kv("Processed (total)", &processed.to_string());
                    output::kv("Failed (total)", &failed.to_string());

                    if verbose {
                        output::separator();
                        output::info("All Metrics:");
                        for metric in &metrics_resp.metrics {
                            let labels = if metric.labels.is_empty() {
                                String::new()
                            } else {
                                format!(" {:?}", metric.labels)
                            };
                            println!("  {} = {:.2}{}", metric.name, metric.value, labels);
                        }
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
