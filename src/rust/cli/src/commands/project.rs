//! Project command - project and branch management
//!
//! Subcommands: list, status, register, info, delete, priority, activate, deactivate, branch

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Args, Subcommand};

use crate::grpc::client::DaemonClient;
use crate::grpc::proto::{
    DeleteProjectRequest, DeprioritizeProjectRequest, GetProjectStatusRequest,
    ListProjectsRequest, RegisterProjectRequest, SetProjectPriorityRequest,
};
use crate::output::{self, ServiceStatus};

/// Project command arguments
#[derive(Args)]
pub struct ProjectArgs {
    #[command(subcommand)]
    command: ProjectCommand,
}

/// Project subcommands
#[derive(Subcommand)]
enum ProjectCommand {
    /// List all registered projects
    List {
        /// Show only active projects
        #[arg(short, long)]
        active: bool,

        /// Filter by priority (high, normal, low)
        #[arg(short, long)]
        priority: Option<String>,
    },

    /// Show project status
    Status {
        /// Project path (current directory if omitted)
        path: Option<PathBuf>,
    },

    /// Register a project for tracking
    Register {
        /// Project path (current directory if omitted)
        path: Option<PathBuf>,

        /// Human-readable project name
        #[arg(short, long)]
        name: Option<String>,

        /// Skip confirmation prompt
        #[arg(short = 'y', long)]
        yes: bool,
    },

    /// Show detailed project info
    Info {
        /// Project ID or path
        project: String,
    },

    /// Delete a project and its data
    Delete {
        /// Project ID or path
        project: String,

        /// Skip confirmation prompt
        #[arg(short = 'y', long)]
        yes: bool,

        /// Keep Qdrant vector data (only remove from SQLite)
        #[arg(long)]
        keep_data: bool,
    },

    /// Set project priority level
    Priority {
        /// Project ID or path
        project: String,

        /// Priority level
        #[arg(value_parser = ["high", "normal"])]
        level: String,
    },

    /// Activate a project (set to high priority with LSP)
    Activate {
        /// Project ID or path
        project: String,
    },

    /// Deactivate a project (set to normal priority)
    Deactivate {
        /// Project ID or path
        project: String,
    },

    /// Branch management
    Branch {
        #[command(subcommand)]
        action: BranchAction,
    },
}

/// Branch subcommands
#[derive(Subcommand)]
enum BranchAction {
    /// List branches for current project
    List,

    /// Show current branch info
    Info,

    /// Switch active branch for indexing
    Switch {
        /// Branch name
        branch: String,
    },
}

/// Execute project command
pub async fn execute(args: ProjectArgs) -> Result<()> {
    match args.command {
        ProjectCommand::List { active, priority } => list_projects(active, priority).await,
        ProjectCommand::Status { path } => project_status(path).await,
        ProjectCommand::Register { path, name, yes } => register_project(path, name, yes).await,
        ProjectCommand::Info { project } => project_info(&project).await,
        ProjectCommand::Delete { project, yes, keep_data } => {
            delete_project(&project, yes, !keep_data).await
        }
        ProjectCommand::Priority { project, level } => set_priority(&project, &level).await,
        ProjectCommand::Activate { project } => activate_project(&project).await,
        ProjectCommand::Deactivate { project } => deactivate_project(&project).await,
        ProjectCommand::Branch { action } => branch_command(action).await,
    }
}

/// Resolve a project argument to a project_id.
///
/// If the argument looks like a path (contains `/` or `.`), resolve it to an
/// absolute path and compute the project_id. Otherwise use it as a direct ID.
fn resolve_project_id(project: &str) -> String {
    if project.contains('/') || project.contains('.') || project == "~" {
        let path = PathBuf::from(project);
        match path.canonicalize() {
            Ok(abs_path) => calculate_project_id(&abs_path),
            Err(_) => project.to_string(),
        }
    } else {
        project.to_string()
    }
}

async fn list_projects(active_only: bool, priority: Option<String>) -> Result<()> {
    output::section("Registered Projects");

    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            let request = ListProjectsRequest {
                priority_filter: priority,
                active_only,
            };

            match client.project().list_projects(request).await {
                Ok(response) => {
                    let list = response.into_inner();

                    if list.projects.is_empty() {
                        output::info("No projects registered");
                        output::info("Register a project with: wqm project register [path]");
                        return Ok(());
                    }

                    for proj in &list.projects {
                        let status = if proj.is_active {
                            ServiceStatus::Healthy
                        } else {
                            ServiceStatus::Unknown
                        };
                        output::status_line(&proj.project_name, status);
                        output::kv("  ID", &proj.project_id);
                        output::kv("  Path", &proj.project_root);
                        output::kv("  Priority", &proj.priority);
                        output::kv("  Active", if proj.is_active { "Yes" } else { "No" });
                    }

                    output::separator();
                    output::info(format!("Total: {} projects", list.total_count));
                }
                Err(e) => {
                    output::error(format!("Failed to list projects: {}", e));
                }
            }
        }
        Err(_) => {
            output::error("Daemon not running. Start with: wqm service start");
        }
    }

    Ok(())
}

async fn project_status(path: Option<PathBuf>) -> Result<()> {
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

/// Calculate the canonical project ID using the same algorithm as the daemon.
fn calculate_project_id(abs_path: &std::path::Path) -> String {
    use wqm_common::project_id::{ProjectIdCalculator, detect_git_remote};

    let git_remote = detect_git_remote(abs_path);
    let calculator = ProjectIdCalculator::new();
    calculator.calculate(abs_path, git_remote.as_deref(), None)
}

async fn register_project(path: Option<PathBuf>, name: Option<String>, yes: bool) -> Result<()> {
    let project_path = path.unwrap_or_else(|| std::env::current_dir().unwrap_or_default());
    let abs_path = project_path
        .canonicalize()
        .context("Could not resolve path")?;

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

    // Display summary
    output::section("Register Project");
    output::kv("Path", &abs_path.display().to_string());
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

    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            let request = RegisterProjectRequest {
                path: abs_path.display().to_string(),
                project_id: project_id.clone(),
                name: Some(project_name),
                git_remote,
                register_if_new: true,
                priority: None, // CLI registers at NORMAL priority
            };

            match client.project().register_project(request).await {
                Ok(response) => {
                    let result = response.into_inner();
                    if result.created {
                        output::success("Project registered successfully");
                    } else {
                        output::info("Project already registered");
                    }
                    output::kv("Priority", &result.priority);
                    output::kv("Active", if result.is_active { "Yes" } else { "No" });
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

    Ok(())
}

async fn delete_project(project: &str, yes: bool, delete_qdrant_data: bool) -> Result<()> {
    let project_id = resolve_project_id(project);

    output::section("Delete Project");
    output::kv("Project ID", &project_id);
    output::kv("Delete Qdrant data", if delete_qdrant_data { "Yes" } else { "No" });
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

async fn set_priority(project: &str, level: &str) -> Result<()> {
    let project_id = resolve_project_id(project);

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

async fn activate_project(project: &str) -> Result<()> {
    let project_id = resolve_project_id(project);

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

async fn deactivate_project(project: &str) -> Result<()> {
    let project_id = resolve_project_id(project);

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

async fn project_info(project: &str) -> Result<()> {
    output::section(format!("Project Info: {}", project));

    let project_id = resolve_project_id(project);

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

async fn branch_command(action: BranchAction) -> Result<()> {
    match action {
        BranchAction::List => branch_list().await,
        BranchAction::Info => branch_info().await,
        BranchAction::Switch { branch } => branch_switch(&branch).await,
    }
}

async fn branch_list() -> Result<()> {
    output::section("Git Branches");

    let output_result = std::process::Command::new("git")
        .args(["branch", "-a"])
        .output();

    match output_result {
        Ok(out) if out.status.success() => {
            let branches = String::from_utf8_lossy(&out.stdout);
            for line in branches.lines() {
                if line.contains("* ") {
                    output::info(&format!("{} (current)", line));
                } else {
                    println!("{}", line);
                }
            }
        }
        Ok(_) => {
            output::warning("Not a git repository");
        }
        Err(e) => {
            output::error(format!("Failed to list branches: {}", e));
        }
    }

    Ok(())
}

async fn branch_info() -> Result<()> {
    output::section("Current Branch");

    let branch = std::process::Command::new("git")
        .args(["branch", "--show-current"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
            } else {
                None
            }
        });

    match branch {
        Some(b) => {
            output::kv("Branch", &b);

            // Get last commit info
            if let Ok(out) = std::process::Command::new("git")
                .args(["log", "-1", "--format=%h %s"])
                .output()
            {
                if out.status.success() {
                    let commit = String::from_utf8_lossy(&out.stdout);
                    output::kv("Last Commit", commit.trim());
                }
            }
        }
        None => {
            output::warning("Not a git repository or no branch checked out");
        }
    }

    Ok(())
}

async fn branch_switch(branch: &str) -> Result<()> {
    output::section(format!("Switch Branch: {}", branch));

    output::info("Branch switching affects which content gets indexed.");
    output::info(&format!("Documents will be tagged with branch='{}'", branch));
    output::separator();

    // Git checkout
    let status = std::process::Command::new("git")
        .args(["checkout", branch])
        .status();

    match status {
        Ok(s) if s.success() => {
            output::success(format!("Switched to branch '{}'", branch));

            // Signal daemon to re-index
            if let Ok(mut client) = DaemonClient::connect_default().await {
                let request = crate::grpc::proto::RefreshSignalRequest {
                    queue_type: crate::grpc::proto::QueueType::WatchedProjects as i32,
                    lsp_languages: vec![],
                    grammar_languages: vec![],
                };

                if client.system().send_refresh_signal(request).await.is_ok() {
                    output::info("Daemon notified to update index for new branch");
                }
            }
        }
        Ok(_) => {
            output::error("Failed to switch branch");
        }
        Err(e) => {
            output::error(format!("Git error: {}", e));
        }
    }

    Ok(())
}
