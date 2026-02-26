//! Live dashboard subcommand.

use anyhow::Result;

use crate::grpc::client::DaemonClient;
use crate::output::{self, ServiceStatus};

use super::types::format_bytes;

/// Run a live-updating dashboard that refreshes at the given interval.
pub async fn live(interval: u64) -> Result<()> {
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
                        output::kv(
                            "Active Projects",
                            &status.active_projects.len().to_string(),
                        );

                        if let Some(metrics) = status.metrics {
                            output::separator();
                            output::kv("CPU", &format!("{:.1}%", metrics.cpu_usage_percent));
                            output::kv("Memory", &format_bytes(metrics.memory_usage_bytes));
                            output::kv(
                                "Pending Ops",
                                &metrics.pending_operations.to_string(),
                            );
                            output::kv(
                                "Connections",
                                &metrics.active_connections.to_string(),
                            );
                        }

                        if let Some(ref mode) = status.resource_mode {
                            output::separator();
                            let idle_str = status
                                .idle_seconds
                                .map(|s| format!("{:.0}s idle", s))
                                .unwrap_or_default();
                            let emb_str = status
                                .current_max_embeddings
                                .map(|e| format!(", {} emb", e))
                                .unwrap_or_default();
                            output::kv(
                                "Resources",
                                &format!("{} ({}{})", mode, idle_str, emb_str),
                            );
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
        output::info(format!(
            "Refreshing every {}s (Ctrl+C to exit)...",
            interval
        ));

        tokio::time::sleep(tokio::time::Duration::from_secs(interval)).await;
    }
}
