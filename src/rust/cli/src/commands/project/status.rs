//! Show project status

use std::path::PathBuf;

use anyhow::{Context, Result};

use crate::grpc::client::DaemonClient;
use crate::grpc::proto::GetProjectStatusRequest;
use crate::output::{self, ServiceStatus};

use super::resolver::calculate_project_id;

pub(super) async fn project_status(path: Option<PathBuf>) -> Result<()> {
    let project_path = path.unwrap_or_else(|| std::env::current_dir().unwrap_or_default());
    let abs_path = project_path
        .canonicalize()
        .context("Could not resolve path")?;

    output::section(format!("Project Status: {}", abs_path.display()));

    // Generate project ID using the same algorithm as the daemon
    let project_id = calculate_project_id(&abs_path);

    output::kv("Path", &abs_path.display().to_string());
    output::kv("Project ID", &project_id);
    output::separator();

    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            let request = GetProjectStatusRequest {
                project_id: project_id.clone(),
            };

            match client.project().get_project_status(request).await {
                Ok(response) => {
                    let status = response.into_inner();

                    if status.found {
                        output::status_line("Registered", ServiceStatus::Healthy);
                        output::kv("Name", &status.project_name);
                        output::kv("Priority", &status.priority);
                        output::kv("Active", if status.is_active { "Yes" } else { "No" });
                        if let Some(remote) = status.git_remote {
                            output::kv("Git Remote", &remote);
                        }
                    } else {
                        output::status_line("Registered", ServiceStatus::Unknown);
                        output::info("Project not registered with daemon");
                        output::info("Register with: wqm project register");
                    }
                }
                Err(e) => {
                    output::warning(format!("Could not get status: {}", e));
                }
            }
        }
        Err(_) => {
            output::error("Daemon not running");
        }
    }

    Ok(())
}
