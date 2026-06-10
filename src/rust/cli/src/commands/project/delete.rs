//! Delete a project and its data

use anyhow::Result;

use crate::grpc::proto::DeleteProjectRequest;
use crate::output;

use super::resolver::resolve_project_id_or_cwd;

/// Check if a directory is a git worktree (`.git` is a file, not a directory).
fn is_git_worktree(path: &std::path::Path) -> bool {
    let git_path = path.join(".git");
    git_path.is_file()
}

pub(super) async fn delete_project(
    project: Option<&str>,
    yes: bool,
    delete_qdrant_data: bool,
) -> Result<()> {
    let project_id = resolve_project_id_or_cwd(project)?;

    output::section("Delete Project");

    // Warn if running from a worktree when no explicit project was given
    if project.is_none() {
        if let Ok(cwd) = std::env::current_dir() {
            if is_git_worktree(&cwd) {
                output::warning(format!(
                    "You are in a worktree at {}",
                    crate::output::style::home_to_tilde(&cwd.display().to_string()),
                ));
                output::warning("This will delete the main project and all associated data.");
            }
        }
    }

    output::kv("Project ID", &project_id);
    output::kv(
        "Delete Qdrant data",
        if delete_qdrant_data { "Yes" } else { "No" },
    );
    output::separator();

    if !yes {
        output::warning("This deletes the project's tracking data and cannot be undone.");
        if !output::typed_confirm(&project_id) {
            output::info("Aborted");
            return Ok(());
        }
    }

    match crate::grpc::connect_default().await {
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
