//! Activate and deactivate projects

use anyhow::Result;

use crate::grpc::client::DaemonClient;
use crate::grpc::proto::{DeprioritizeProjectRequest, RegisterProjectRequest};
use crate::output;

use super::resolver::resolve_project_id_or_cwd;

pub(super) async fn activate_project(project: Option<&str>) -> Result<()> {
    let project_id = resolve_project_id_or_cwd(project)?;

    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            // Use RegisterProject with priority="high" and register_if_new=false
            let request = RegisterProjectRequest {
                path: String::new(), // Not needed for existing projects
                project_id: project_id.clone(),
                name: None,
                git_remote: None,
                register_if_new: false,
                priority: Some("high".to_string()),
            };

            match client.project().register_project(request).await {
                Ok(response) => {
                    let result = response.into_inner();
                    if result.priority == "none" {
                        output::error(format!("Project not found: {}", project_id));
                        output::info("Register first with: wqm project register");
                    } else {
                        output::success(format!(
                            "Project {} activated (priority: {})",
                            project_id, result.priority
                        ));
                    }
                }
                Err(e) => {
                    output::error(format!("Failed to activate: {}", e.message()));
                }
            }
        }
        Err(_) => {
            output::error("Daemon not running. Start with: wqm service start");
        }
    }

    Ok(())
}

pub(super) async fn deactivate_project(project: Option<&str>) -> Result<()> {
    let project_id = resolve_project_id_or_cwd(project)?;

    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            let request = DeprioritizeProjectRequest {
                project_id: project_id.clone(),
            };

            match client.project().deprioritize_project(request).await {
                Ok(response) => {
                    let result = response.into_inner();
                    output::success(format!(
                        "Project {} deactivated (priority: {})",
                        project_id, result.new_priority
                    ));
                }
                Err(e) => {
                    let msg = e.message();
                    if msg.contains("not found") {
                        output::error(format!("Project not found: {}", project_id));
                    } else {
                        output::error(format!("Failed to deactivate: {}", msg));
                    }
                }
            }
        }
        Err(_) => {
            output::error("Daemon not running. Start with: wqm service start");
        }
    }

    Ok(())
}
