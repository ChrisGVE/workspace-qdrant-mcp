//! Performance metrics subcommand.

use anyhow::Result;

use crate::grpc::client::DaemonClient;
use crate::output;

use super::types::format_bytes;

/// Show performance metrics from the daemon.
pub async fn performance() -> Result<()> {
    output::section("Performance Metrics");

    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            match client.system().get_status(()).await {
                Ok(response) => {
                    let status = response.into_inner();

                    if let Some(metrics) = status.metrics {
                        output::kv(
                            "CPU Usage",
                            &format!("{:.1}%", metrics.cpu_usage_percent),
                        );
                        output::kv("Memory Used", &format_bytes(metrics.memory_usage_bytes));
                        output::kv("Memory Total", &format_bytes(metrics.memory_total_bytes));
                        output::kv("Disk Used", &format_bytes(metrics.disk_usage_bytes));
                        output::kv("Disk Total", &format_bytes(metrics.disk_total_bytes));
                        output::separator();
                        output::kv(
                            "Active Connections",
                            &metrics.active_connections.to_string(),
                        );
                        output::kv(
                            "Pending Operations",
                            &metrics.pending_operations.to_string(),
                        );
                    } else {
                        output::warning("Metrics not available from daemon");
                    }
                }
                Err(e) => {
                    output::error(format!(
                        "Failed to get performance metrics: {}",
                        e
                    ));
                }
            }
        }
        Err(_) => {
            output::error("Cannot connect to daemon");
        }
    }

    Ok(())
}
