//! Show project status

use anyhow::Result;

use crate::grpc::client::DaemonClient;
use crate::grpc::proto::GetProjectStatusRequest;
use crate::output::style::home_to_tilde;
use crate::output::{self, ServiceStatus};

use super::resolver::resolve_project_id_or_cwd;

pub(super) async fn project_status(project: Option<&str>) -> Result<()> {
    let project_id = resolve_project_id_or_cwd(project)?;

    output::section("Project Status");

    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            let request = GetProjectStatusRequest {
                project_id: project_id.clone(),
            };

            match client.project().get_project_status(request).await {
                Ok(response) => {
                    let status = response.into_inner();

                    if status.found {
                        output::kv("Name", &status.project_name);
                        output::kv("Project ID", &status.project_id);
                        output::kv("Path", home_to_tilde(&status.project_root));
                        output::kv("Active", if status.is_active { "Yes" } else { "No" });
                        if let Some(remote) = status.git_remote {
                            output::kv("Git Remote", &remote);
                        }
                        if status.is_worktree {
                            output::kv("Worktree", "Yes");
                            if let Some(main_path) = status.main_worktree_path {
                                output::kv("Main Working Tree", home_to_tilde(&main_path));
                            }
                        }
                        output::separator();
                        output::status_line("Registered", ServiceStatus::Healthy);
                    } else {
                        output::kv("Project ID", &project_id);
                        output::separator();
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
            output::kv("Project ID", &project_id);
            output::separator();
            output::error("Daemon not running");
        }
    }

    Ok(())
}
