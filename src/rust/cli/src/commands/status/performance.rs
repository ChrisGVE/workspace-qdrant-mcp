//! Performance metrics subcommand.
//!
//! Shows system resource metrics (CPU, memory, disk) from the daemon.
//! For pipeline timing statistics, use `wqm admin perf`.

use anyhow::Result;

use crate::output;

use super::types::format_bytes;

/// Show system resource metrics from the daemon.
///
/// Prints a hint directing users to `wqm admin perf` for pipeline
/// timing statistics (per-phase breakdown, percentiles, throughput).
pub async fn performance() -> Result<()> {
    output::section("System Resource Metrics");

    match crate::grpc::connect_default().await {
        Ok(mut client) => match client.system().get_status(()).await {
            Ok(response) => {
                let status = response.into_inner();

                if let Some(metrics) = status.metrics {
                    output::kv("CPU Usage", format!("{:.1}%", metrics.cpu_usage_percent));
                    output::kv("Memory Used", format_bytes(metrics.memory_usage_bytes));
                    output::kv("Memory Total", format_bytes(metrics.memory_total_bytes));
                    output::kv("Disk Used", format_bytes(metrics.disk_usage_bytes));
                    output::kv("Disk Total", format_bytes(metrics.disk_total_bytes));
                    output::separator();
                    output::kv("Active Connections", metrics.active_connections.to_string());
                    output::kv("Pending Operations", metrics.pending_operations.to_string());
                } else {
                    output::warning("Metrics not available from daemon");
                }
            }
            Err(e) => {
                output::error(format!("Failed to get resource metrics: {}", e));
            }
        },
        Err(_) => {
            output::error("Cannot connect to daemon");
        }
    }

    println!();
    output::info("For pipeline timing stats, use: wqm admin perf");

    Ok(())
}
