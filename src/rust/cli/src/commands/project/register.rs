//! Register a project for tracking

use std::path::PathBuf;

use anyhow::{Context, Result};
use wqm_common::paths::CanonicalPath;

use crate::grpc::client::DaemonClient;
use crate::grpc::proto::RegisterProjectRequest;
use crate::output;

use super::resolver::calculate_project_id;

async fn call_daemon_register(
    abs_path: std::path::PathBuf,
    project_id: String,
    project_name: String,
    git_remote: Option<String>,
) {
    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            let request = RegisterProjectRequest {
                path: abs_path.display().to_string(),
                project_id: project_id.clone(),
                name: Some(project_name),
                git_remote,
                register_if_new: true,
                priority: None,
            };

            match client.project().register_project(request).await {
                Ok(response) => {
                    let result = response.into_inner();
                    if result.created {
                        output::success("Project registered successfully");
                    } else {
                        output::info("Project already registered");
                    }
                    output::kv("Active", if result.is_active { "Yes" } else { "No" });
                    if result.is_worktree {
                        output::info(format!(
                            "Note: This path is a git worktree. \
                             It is indexed under project {}",
                            result.project_id
                        ));
                        if let Some(watch_path) = &result.watch_path {
                            output::kv("Watch Path", watch_path);
                        }
                    }
                }
                Err(e) => {
                    output::error(format!("Failed to register project: {}", e));
                }
            }
        }
        Err(_) => {
            output::error("Daemon not running. Start with: wqm service start");
        }
    }
}

/// Build a [`CanonicalPath`] from a CLI path argument, absolutizing
/// relative inputs syntactically against CWD. No fs canonicalize.
fn canonical_from_cli_path(path: &std::path::Path) -> Result<CanonicalPath> {
    let s = path.to_str().context("Path contains invalid UTF-8")?;
    if let Ok(cp) = CanonicalPath::from_user_input(s) {
        return Ok(cp);
    }
    let cwd = std::env::current_dir().context("Could not determine current directory")?;
    let joined = cwd.join(path);
    let joined_str = joined
        .to_str()
        .context("Path contains invalid UTF-8 after CWD join")?;
    CanonicalPath::from_user_input(joined_str)
        .map_err(|e| anyhow::anyhow!("Could not resolve path: {e}"))
}

pub(super) async fn register_project(
    path: Option<PathBuf>,
    name: Option<String>,
    yes: bool,
) -> Result<()> {
    let project_path = path.unwrap_or_else(|| std::env::current_dir().unwrap_or_default());
    let abs_canonical = canonical_from_cli_path(&project_path)?;
    let abs_path = PathBuf::from(abs_canonical.as_str());

    let project_name = name.unwrap_or_else(|| {
        abs_path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string())
    });

    // Generate project ID using the same algorithm as the daemon
    let project_id = calculate_project_id(&abs_path);

    // Detect git remote (same function used by project ID calculation)
    let git_remote = wqm_common::project_id::detect_git_remote(&abs_path);

    // Check if the path is already part of a registered project
    if let Ok(db_path) = crate::config::get_database_path_checked() {
        if let Some((existing_id, existing_path)) =
            wqm_common::project_id::resolve_path_to_project(&db_path, &abs_path)
        {
            output::section("Register Project");
            output::info(format!(
                "This directory is already part of project '{}'",
                existing_id
            ));
            output::kv("Existing Project ID", &existing_id);
            output::kv(
                "Project Path",
                crate::output::style::home_to_tilde(&existing_path),
            );
            return Ok(());
        }
    }

    // Display summary
    output::section("Register Project");
    output::kv("Path", abs_canonical.as_str());
    output::kv("Name", &project_name);
    output::kv("Project ID", &project_id);
    if let Some(remote) = &git_remote {
        output::kv("Git Remote", remote);
    }
    output::separator();

    // Confirm unless --yes
    if !yes && !output::confirm("Register this project?") {
        output::info("Aborted");
        return Ok(());
    }

    call_daemon_register(abs_path, project_id, project_name, git_remote).await;

    Ok(())
}
