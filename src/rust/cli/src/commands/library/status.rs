//! Library status subcommand

use anyhow::{Context, Result};

use super::helpers::open_db;
use crate::grpc::client::DaemonClient;
use crate::output::{self, ServiceStatus};

/// Show watch status for all libraries
pub async fn execute() -> Result<()> {
    output::section("Library Watch Status");

    check_daemon_health().await;
    output::separator();
    show_library_status()
}

/// Check and display daemon health status
async fn check_daemon_health() {
    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            output::status_line("Daemon", ServiceStatus::Healthy);

            match client.system().health(()).await {
                Ok(response) => {
                    let health = response.into_inner();

                    for comp in &health.components {
                        if comp.component_name.contains("watcher")
                            || comp.component_name.contains("library")
                        {
                            let comp_status = ServiceStatus::from_proto(comp.status);
                            output::status_line(&comp.component_name, comp_status);
                            if !comp.message.is_empty() {
                                output::kv("  Message", &comp.message);
                            }
                        }
                    }
                }
                Err(_) => {
                    output::warning("Could not get health details");
                }
            }
        }
        Err(_) => {
            output::status_line("Daemon", ServiceStatus::Unhealthy);
            output::warning("Daemon not running - start it with: wqm service start");
        }
    }
}

/// Show library watch status from SQLite
fn show_library_status() -> Result<()> {
    let conn = match open_db() {
        Ok(c) => c,
        Err(_) => {
            output::info("No database found. Run daemon first to initialize.");
            return Ok(());
        }
    };

    let mut stmt = conn
        .prepare(
            "SELECT tenant_id, path, library_mode, enabled \
         FROM watch_folders WHERE collection = 'libraries' ORDER BY tenant_id",
        )
        .context("Failed to query watch_folders")?;

    let libraries: Vec<(String, String, Option<String>, bool)> = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, Option<String>>(2)?,
                row.get::<_, i32>(3)? != 0,
            ))
        })
        .context("Failed to read library rows")?
        .collect::<Result<Vec<_>, _>>()
        .context("Failed to parse library rows")?;

    if libraries.is_empty() {
        output::info("No libraries configured.");
        return Ok(());
    }

    output::info(format!("{} library/libraries configured:", libraries.len()));

    let watching = libraries.iter().filter(|(_, _, _, e)| *e).count();
    let paused = libraries.len() - watching;
    output::kv("  Watching", watching.to_string());
    output::kv("  Paused", paused.to_string());
    output::separator();

    for (tag, path, mode, enabled) in &libraries {
        let status_icon = if *enabled { "watching" } else { "paused" };
        output::info(format!(
            "{}: {} [{}] ({})",
            tag,
            path,
            status_icon,
            mode.as_deref().unwrap_or("incremental")
        ));
    }

    Ok(())
}
