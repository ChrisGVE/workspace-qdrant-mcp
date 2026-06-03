//! Ingestion status subcommand handler

use anyhow::Result;

use crate::output::style::home_to_tilde;
use crate::output::{self, ServiceStatus};

pub async fn ingest_status(verbose: bool) -> Result<()> {
    output::section("Ingestion Status");

    match crate::grpc::connect_default().await {
        Ok(mut client) => {
            output::status_line("Daemon", ServiceStatus::Healthy);

            match client.system().get_metrics(()).await {
                Ok(response) => {
                    let metrics = response.into_inner();

                    let mut pending = 0.0;
                    let mut processed = 0.0;
                    let mut failed = 0.0;

                    for metric in &metrics.metrics {
                        match metric.name.as_str() {
                            "queue_pending" => pending = metric.value,
                            "queue_processed" => processed = metric.value,
                            "queue_failed" => failed = metric.value,
                            _ => {}
                        }
                    }

                    output::separator();
                    output::kv("Pending", (pending as i64).to_string());
                    output::kv("Processed", (processed as i64).to_string());
                    output::kv("Failed", (failed as i64).to_string());

                    if verbose {
                        output::separator();
                        output::info("Queue details stored in SQLite unified_queue table:");
                        let db_path = wqm_common::paths::get_database_path()
                            .unwrap_or_else(|_| std::path::PathBuf::from("<unknown>"));
                        output::info(format!(
                            "  sqlite3 {} 'SELECT * FROM unified_queue LIMIT 20'",
                            home_to_tilde(&db_path.display().to_string())
                        ));
                    }
                }
                Err(e) => {
                    output::error(format!("Failed to get queue status: {}", e));
                }
            }
        }
        Err(_) => {
            output::status_line("Daemon", ServiceStatus::Unhealthy);
            output::error("Daemon not running");
        }
    }

    Ok(())
}
