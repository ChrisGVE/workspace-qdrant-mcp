//! Delete a project and its data

use anyhow::Result;

use crate::grpc::client::DaemonClient;
use crate::grpc::proto::DeleteProjectRequest;
use crate::output;

use super::resolver::resolve_project_id_or_cwd;

pub(super) async fn delete_project(
    project: Option<&str>,
    yes: bool,
    delete_qdrant_data: bool,
) -> Result<()> {
    let project_id = resolve_project_id_or_cwd(project)?;

    output::section("Delete Project");
    output::kv("Project ID", &project_id);
    output::kv(
        "Delete Qdrant data",
        if delete_qdrant_data { "Yes" } else { "No" },
    );
    output::separator();

    if !yes && !output::confirm("Delete this project? This cannot be undone.") {
        output::info("Aborted");
        return Ok(());
    }

    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            let request = DeleteProjectRequest {
                project_id: project_id.clone(),
                delete_qdrant_data,
            };

            match client.project().delete_project(request).await {
                Ok(response) => {
                    let result = response.into_inner();
                    output::success(&result.message);
                }
                Err(e) => {
                    let msg = e.message();
                    if msg.contains("not found") {
                        output::error(format!("Project not found: {}", project_id));
                    } else {
                        output::error(format!("Failed to delete project: {}", msg));
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
