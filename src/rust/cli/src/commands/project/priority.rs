//! Set project priority level

use anyhow::Result;

use crate::grpc::client::DaemonClient;
use crate::grpc::proto::SetProjectPriorityRequest;
use crate::output;

use super::resolver::resolve_project_id_or_cwd;

pub(super) async fn set_priority(project: Option<&str>, level: &str) -> Result<()> {
    let project_id = resolve_project_id_or_cwd(project)?;

    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            let request = SetProjectPriorityRequest {
                project_id: project_id.clone(),
                priority: level.to_string(),
            };

            match client.project().set_project_priority(request).await {
                Ok(response) => {
                    let result = response.into_inner();
                    output::success(format!(
                        "Priority: {} -> {} ({} queue items updated)",
                        result.previous_priority, result.new_priority, result.queue_items_updated
                    ));
                }
                Err(e) => {
                    let msg = e.message();
                    if msg.contains("not found") {
                        output::error(format!("Project not found: {}", project_id));
                    } else {
                        output::error(format!("Failed to set priority: {}", msg));
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
