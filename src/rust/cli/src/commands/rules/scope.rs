//! Scope management subcommand
//!
//! Lists available rule scopes and shows scope-specific information
//! including active projects from the daemon.

use anyhow::Result;

use crate::grpc::client::DaemonClient;
use crate::output::{self, ServiceStatus};

/// Manage rule scopes: list available scopes or show a specific one.
pub async fn manage_scopes(
    list: bool,
    show: Option<String>,
    verbose: bool,
) -> Result<()> {
    output::section("Rule Scopes");

    if list || show.is_none() {
        print_scope_types();
        print_active_scopes(verbose).await;
    }

    if let Some(scope_name) = show {
        print_scope_detail(&scope_name);
    }

    Ok(())
}

/// Display the scope type hierarchy.
fn print_scope_types() {
    output::info("Available scope types:");
    output::separator();

    output::kv("global", "Rules apply to all projects");
    output::kv("project:<id>", "Rules apply to a specific project");
    output::kv("branch:<name>", "Rules apply to a specific branch");
    output::separator();

    output::info("Scope hierarchy (highest to lowest priority):");
    output::info("  1. branch:* - Branch-specific rules");
    output::info("  2. project:* - Project-specific rules");
    output::info("  3. global - Global rules");
    output::separator();
}

/// Query the daemon for active project scopes.
async fn print_active_scopes(_verbose: bool) {
    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            output::status_line("Daemon", ServiceStatus::Healthy);
            output::separator();

            output::info("Active project scopes:");
            match client.system().get_status(()).await {
                Ok(response) => {
                    let status = response.into_inner();
                    if status.active_projects.is_empty() {
                        output::info("  (no active projects)");
                    } else {
                        for project_id in &status.active_projects {
                            output::kv("  project", project_id);
                        }
                    }
                }
                Err(e) => {
                    output::warning(format!("Could not get projects: {}", e));
                }
            }
        }
        Err(_) => {
            output::status_line("Daemon", ServiceStatus::Unhealthy);
            output::warning("Cannot list active scopes without daemon");
        }
    }
}

/// Show details and MCP command for a specific scope.
fn print_scope_detail(scope_name: &str) {
    output::separator();
    output::kv("Showing scope", scope_name);
    output::separator();

    output::info("Rules in this scope:");
    output::info("  (Query daemon for scope-specific rules)");
    output::separator();

    output::info("MCP command to list rules for this scope:");
    if scope_name == "global" {
        output::info("  mcp__workspace_qdrant__rules(action=\"list\", scope=\"global\")");
    } else if scope_name.starts_with("project:") {
        let project = scope_name
            .strip_prefix("project:")
            .unwrap_or(scope_name);
        output::info(&format!(
            "  mcp__workspace_qdrant__rules(action=\"list\", project=\"{}\")",
            project
        ));
    } else {
        output::info(&format!(
            "  mcp__workspace_qdrant__rules(action=\"list\", scope=\"{}\")",
            scope_name
        ));
    }
}
