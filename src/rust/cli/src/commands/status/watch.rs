//! Watch status subcommand.

use anyhow::Result;

use crate::grpc::client::DaemonClient;
use crate::output;

/// Show file watcher status.
pub async fn watch() -> Result<()> {
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
                            println!("  \u{2022} {}", project);
                        }
                    }
                }
                Err(e) => {
                    output::error(format!("Failed to get watch status: {}", e));
                }
            }

            output::separator();
            output::info("Watch folders configured in SQLite:");
            output::info(
                "  Use: sqlite3 ~/.local/share/workspace-qdrant/state.db \
                 'SELECT watch_id, path, enabled FROM watch_folders'",
            );
        }
        Err(_) => {
            output::error("Cannot connect to daemon");
        }
    }

    Ok(())
}
