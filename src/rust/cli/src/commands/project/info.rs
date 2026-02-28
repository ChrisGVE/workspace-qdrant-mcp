//! Show detailed project info

use anyhow::Result;

use crate::grpc::client::DaemonClient;
use crate::grpc::proto::GetProjectStatusRequest;
use crate::output;

use super::resolver::resolve_project_id_or_cwd;

pub(super) async fn project_info(project: Option<&str>) -> Result<()> {
    let project_id = resolve_project_id_or_cwd(project)?;

    output::section(format!("Project Info: {}", project_id));

    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            let request = GetProjectStatusRequest {
                project_id: project_id.clone(),
            };

            match client.project().get_project_status(request).await {
                Ok(response) => {
                    let status = response.into_inner();

                    if status.found {
                        output::kv("Project ID", &status.project_id);
                        output::kv("Name", &status.project_name);
                        output::kv("Path", &status.project_root);
                        output::kv("Priority", &status.priority);
                        output::kv("Active", if status.is_active { "Yes" } else { "No" });
                        if let Some(remote) = status.git_remote {
                            output::kv("Git Remote", &remote);
                        }
                    } else {
                        output::warning("Project not found");
                        output::info("Use 'wqm project list' to see registered projects");
                    }
                }
                Err(e) => {
                    output::error(format!("Failed to get project info: {}", e));
                }
            }
        }
        Err(_) => {
            output::error("Daemon not running");
        }
    }

    Ok(())
}
