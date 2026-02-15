//! Project command - project and branch management
//!
//! Subcommands: list, status, register, info, delete, priority, activate, deactivate, check, branch

use std::path::PathBuf;

use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use colored::Colorize;
use serde::Serialize;
use walkdir::WalkDir;

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

    /// Show detailed project info (auto-detects from CWD if project omitted)
    Info {
        /// Project ID or path (auto-detected from current directory if omitted)
        project: Option<String>,
    },

    /// Delete a project and its data (auto-detects from CWD if project omitted)
    Delete {
        /// Project ID or path (auto-detected from current directory if omitted)
        project: Option<String>,

        /// Skip confirmation prompt
        #[arg(short = 'y', long)]
        yes: bool,

        /// Keep Qdrant vector data (only remove from SQLite)
        #[arg(long)]
        keep_data: bool,
    },

    /// Set project priority level (auto-detects from CWD if project omitted)
    Priority {
        /// Project ID or path (auto-detected from current directory if omitted)
        project: Option<String>,

        /// Priority level
        #[arg(value_parser = ["high", "normal"])]
        level: String,
    },

    /// Activate a project (auto-detects from CWD if project omitted)
    Activate {
        /// Project ID or path (auto-detected from current directory if omitted)
        project: Option<String>,
    },

    /// Deactivate a project (auto-detects from CWD if project omitted)
    Deactivate {
        /// Project ID or path (auto-detected from current directory if omitted)
        project: Option<String>,
    },

    /// Check ingestion status: compare tracked files against filesystem
    Check {
        /// Project ID or path (auto-detected from current directory if omitted)
        project: Option<String>,

        /// Show per-file status
        #[arg(short, long)]
        verbose: bool,

        /// Output as JSON
        #[arg(long)]
        json: bool,
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
        ProjectCommand::Info { project } => project_info(project.as_deref()).await,
        ProjectCommand::Delete { project, yes, keep_data } => {
            delete_project(project.as_deref(), yes, !keep_data).await
        }
        ProjectCommand::Priority { project, level } => set_priority(project.as_deref(), &level).await,
        ProjectCommand::Activate { project } => activate_project(project.as_deref()).await,
        ProjectCommand::Deactivate { project } => deactivate_project(project.as_deref()).await,
        ProjectCommand::Check { project, verbose, json } => {
            check_project(project.as_deref(), verbose, json).await
        }
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

/// Resolve an optional project argument to a project_id.
///
/// If `project` is Some, delegates to `resolve_project_id`. If None,
/// auto-detects the project from the current working directory by looking up
/// registered projects in SQLite.
fn resolve_project_id_or_cwd(project: Option<&str>) -> Result<String> {
    let (id, auto_detected) = resolve_project_id_or_cwd_quiet(project)?;
    if auto_detected {
        output::info(format!("Auto-detected project: {}", id));
    }
    Ok(id)
}

fn resolve_project_id_or_cwd_quiet(project: Option<&str>) -> Result<(String, bool)> {
    if let Some(p) = project {
        return Ok((resolve_project_id(p), false));
    }

    // Auto-detect from CWD
    let cwd = std::env::current_dir()
        .context("Could not determine current directory")?;

    let db_path = crate::config::get_database_path_checked()
        .map_err(|e| anyhow::anyhow!("Database not found: {}", e))?;

    match wqm_common::project_id::resolve_path_to_project(&db_path, &cwd) {
        Some((tenant_id, _path)) => {
            Ok((tenant_id, true))
        }
        None => {
            anyhow::bail!(
                "Could not detect project from current directory.\n\
                 Run from within a registered project directory, or pass a project ID explicitly."
            );
        }
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

async fn delete_project(project: Option<&str>, yes: bool, delete_qdrant_data: bool) -> Result<()> {
    let project_id = resolve_project_id_or_cwd(project)?;

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

async fn set_priority(project: Option<&str>, level: &str) -> Result<()> {
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

async fn activate_project(project: Option<&str>) -> Result<()> {
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

async fn deactivate_project(project: Option<&str>) -> Result<()> {
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

async fn project_info(project: Option<&str>) -> Result<()> {
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

/// File check status
#[derive(Debug, Clone, Serialize)]
struct FileCheckEntry {
    path: String,
    status: &'static str,
}

/// Summary of check results
#[derive(Debug, Serialize)]
struct CheckSummary {
    project_id: String,
    project_root: String,
    up_to_date: u64,
    to_add: u64,
    to_update: u64,
    to_delete: u64,
    total_tracked: u64,
    total_on_disk: u64,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    files: Vec<FileCheckEntry>,
}

async fn check_project(project: Option<&str>, verbose: bool, json: bool) -> Result<()> {
    let (project_id, auto_detected) = resolve_project_id_or_cwd_quiet(project)?;
    if auto_detected && !json {
        output::info(format!("Auto-detected project: {}", project_id));
    }

    let db_path = crate::config::get_database_path_checked()
        .map_err(|e| anyhow::anyhow!("Database not found: {}", e))?;

    let conn = rusqlite::Connection::open_with_flags(
        &db_path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .context("Failed to open state database")?;

    // Find the watch_folder for this project
    let watch_row: Result<(String, String), _> = conn.query_row(
        "SELECT watch_id, path FROM watch_folders WHERE tenant_id = ?1 AND collection = 'projects' LIMIT 1",
        rusqlite::params![&project_id],
        |row| Ok((row.get(0)?, row.get(1)?)),
    );

    let (watch_id, project_root) = match watch_row {
        Ok(r) => r,
        Err(rusqlite::Error::QueryReturnedNoRows) => {
            if json {
                output::print_json(&serde_json::json!({"error": "Project not found", "project_id": project_id}));
            } else {
                output::error(format!("Project not found: {}", project_id));
            }
            return Ok(());
        }
        Err(e) => return Err(e.into()),
    };

    // Get all tracked files for this watch_folder
    let mut stmt = conn.prepare(
        "SELECT file_path, file_hash FROM tracked_files WHERE watch_folder_id = ?1"
    )?;
    let tracked_rows: Vec<(String, String)> = stmt
        .query_map(rusqlite::params![&watch_id], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?
        .filter_map(|r| r.ok())
        .collect();

    let tracked_files: HashMap<String, String> = tracked_rows.into_iter().collect();

    // Walk the filesystem from project root
    let project_root_path = std::path::Path::new(&project_root);
    if !project_root_path.exists() {
        if json {
            output::print_json(&serde_json::json!({
                "error": "Project root does not exist",
                "project_root": project_root,
            }));
        } else {
            output::error(format!("Project root does not exist: {}", project_root));
        }
        return Ok(());
    }

    // Build allowed extensions checker
    let allowed = workspace_qdrant_core::allowed_extensions::AllowedExtensions::default();

    let mut disk_files: HashSet<String> = HashSet::new();

    for entry in WalkDir::new(&project_root)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| {
            if e.file_type().is_dir() {
                let name = e.file_name().to_string_lossy();
                !workspace_qdrant_core::patterns::exclusion::should_exclude_directory(&name)
            } else {
                true
            }
        })
    {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };

        if !entry.file_type().is_file() {
            continue;
        }

        let abs_path = entry.path();
        let rel_path = match abs_path.strip_prefix(project_root_path) {
            Ok(r) => r.to_string_lossy().to_string(),
            Err(_) => continue,
        };

        // Apply exclusion and extension filters
        if workspace_qdrant_core::patterns::exclusion::should_exclude_file(&rel_path) {
            continue;
        }
        if !allowed.is_allowed(&rel_path, "projects") {
            continue;
        }

        disk_files.insert(rel_path);
    }

    // Compare tracked vs disk
    let mut up_to_date: u64 = 0;
    let mut to_update: Vec<String> = Vec::new();
    let mut to_delete: Vec<String> = Vec::new();
    let mut to_add: Vec<String> = Vec::new();

    // Check tracked files against disk
    for (rel_path, tracked_hash) in &tracked_files {
        if disk_files.contains(rel_path) {
            // File exists on disk — check if hash changed
            let abs = project_root_path.join(rel_path);
            match wqm_common::hashing::compute_file_hash(&abs) {
                Ok(disk_hash) => {
                    if disk_hash == *tracked_hash {
                        up_to_date += 1;
                    } else {
                        to_update.push(rel_path.clone());
                    }
                }
                Err(_) => {
                    // Can't read file — treat as needs update
                    to_update.push(rel_path.clone());
                }
            }
        } else {
            // Tracked but not on disk — needs delete
            to_delete.push(rel_path.clone());
        }
    }

    // Check disk files not in tracked
    for rel_path in &disk_files {
        if !tracked_files.contains_key(rel_path) {
            to_add.push(rel_path.clone());
        }
    }

    // Sort for stable output
    to_add.sort();
    to_update.sort();
    to_delete.sort();

    // Build summary
    let mut files = Vec::new();
    if verbose || json {
        for p in &to_add {
            files.push(FileCheckEntry { path: p.clone(), status: "add" });
        }
        for p in &to_update {
            files.push(FileCheckEntry { path: p.clone(), status: "update" });
        }
        for p in &to_delete {
            files.push(FileCheckEntry { path: p.clone(), status: "delete" });
        }
    }

    let summary = CheckSummary {
        project_id: project_id.clone(),
        project_root: project_root.clone(),
        up_to_date,
        to_add: to_add.len() as u64,
        to_update: to_update.len() as u64,
        to_delete: to_delete.len() as u64,
        total_tracked: tracked_files.len() as u64,
        total_on_disk: disk_files.len() as u64,
        files,
    };

    if json {
        output::print_json(&summary);
    } else {
        output::section(format!("Project Check: {}", project_id));
        output::kv("Path", &project_root);
        output::separator();

        output::kv("Tracked files", &summary.total_tracked.to_string());
        output::kv("Files on disk", &summary.total_on_disk.to_string());
        output::separator();

        if summary.up_to_date > 0 {
            println!("  {} {}", "=".green(), format!("{} up to date", summary.up_to_date));
        }
        if summary.to_add > 0 {
            println!("  {} {}", "+".yellow(), format!("{} to add", summary.to_add));
        }
        if summary.to_update > 0 {
            println!("  {} {}", "~".blue(), format!("{} to update", summary.to_update));
        }
        if summary.to_delete > 0 {
            println!("  {} {}", "-".red(), format!("{} to delete", summary.to_delete));
        }

        if summary.to_add == 0 && summary.to_update == 0 && summary.to_delete == 0 {
            output::success("All tracked files are up to date");
        }

        if verbose {
            if !to_add.is_empty() {
                output::separator();
                println!("{}", "Files to add:".bold());
                for p in &to_add {
                    println!("  {} {}", "+".yellow(), p);
                }
            }
            if !to_update.is_empty() {
                output::separator();
                println!("{}", "Files to update:".bold());
                for p in &to_update {
                    println!("  {} {}", "~".blue(), p);
                }
            }
            if !to_delete.is_empty() {
                output::separator();
                println!("{}", "Files to delete:".bold());
                for p in &to_delete {
                    println!("  {} {}", "-".red(), p);
                }
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    // --- resolve_project_id tests ---

    #[test]
    fn test_resolve_project_id_plain_id_passthrough() {
        // A string without '/' or '.' is treated as a direct project ID
        let result = resolve_project_id("my-project-abc123");
        assert_eq!(result, "my-project-abc123");
    }

    #[test]
    fn test_resolve_project_id_with_dot_triggers_path_resolution() {
        // A string containing '.' is treated as a path; if canonicalize fails,
        // it falls back to the original string
        let result = resolve_project_id("nonexistent.path");
        // canonicalize will fail → falls back to original
        assert_eq!(result, "nonexistent.path");
    }

    #[test]
    fn test_resolve_project_id_with_slash_triggers_path_resolution() {
        // A string containing '/' is treated as a path
        let result = resolve_project_id("/nonexistent/path");
        // canonicalize will fail → falls back to original
        assert_eq!(result, "/nonexistent/path");
    }

    #[test]
    fn test_resolve_project_id_tilde_triggers_path_resolution() {
        // "~" alone is treated as a path
        let result = resolve_project_id("~");
        // "~" won't canonicalize through std::path → falls back
        assert_eq!(result, "~");
    }

    #[test]
    fn test_resolve_project_id_real_path() {
        // Use a path that actually exists: /tmp
        let result = resolve_project_id("/tmp");
        // /tmp on macOS is a symlink to /private/tmp, so canonicalize resolves it
        // The result should be a valid project_id (SHA256 hash)
        assert!(!result.is_empty());
        assert_ne!(result, "/tmp"); // canonicalize + calculate_project_id transforms it
    }

    #[test]
    fn test_resolve_project_id_current_dir() {
        // "." contains a dot, so it triggers path resolution
        let result = resolve_project_id(".");
        // Should resolve CWD and compute a project ID
        assert!(!result.is_empty());
        assert_ne!(result, ".");
    }

    // --- resolve_project_id_or_cwd_quiet tests ---

    #[test]
    fn test_resolve_project_id_or_cwd_quiet_with_explicit_id() {
        let (id, auto) = resolve_project_id_or_cwd_quiet(Some("explicit-id")).unwrap();
        assert_eq!(id, "explicit-id");
        assert!(!auto);
    }

    #[test]
    fn test_resolve_project_id_or_cwd_quiet_with_explicit_path() {
        let (id, auto) = resolve_project_id_or_cwd_quiet(Some("/tmp")).unwrap();
        // Path triggers calculate_project_id
        assert!(!id.is_empty());
        assert!(!auto);
    }

    // --- CheckSummary serialization tests ---

    #[test]
    fn test_check_summary_json_serialization() {
        let summary = CheckSummary {
            project_id: "test-project".to_string(),
            project_root: "/home/user/project".to_string(),
            up_to_date: 50,
            to_add: 3,
            to_update: 2,
            to_delete: 1,
            total_tracked: 53,
            total_on_disk: 55,
            files: vec![
                FileCheckEntry { path: "src/new.rs".to_string(), status: "add" },
                FileCheckEntry { path: "src/changed.rs".to_string(), status: "update" },
                FileCheckEntry { path: "src/deleted.rs".to_string(), status: "delete" },
            ],
        };
        let serialized = serde_json::to_string(&summary).unwrap();
        assert!(serialized.contains("\"project_id\":\"test-project\""));
        assert!(serialized.contains("\"up_to_date\":50"));
        assert!(serialized.contains("\"to_add\":3"));
        assert!(serialized.contains("\"to_update\":2"));
        assert!(serialized.contains("\"to_delete\":1"));
        assert!(serialized.contains("\"total_tracked\":53"));
        assert!(serialized.contains("\"total_on_disk\":55"));

        // Verify file entries
        let value: serde_json::Value = serde_json::from_str(&serialized).unwrap();
        let files = value["files"].as_array().unwrap();
        assert_eq!(files.len(), 3);
        assert_eq!(files[0]["status"], "add");
        assert_eq!(files[1]["status"], "update");
        assert_eq!(files[2]["status"], "delete");
    }

    #[test]
    fn test_check_summary_empty_files_omitted() {
        let summary = CheckSummary {
            project_id: "test".to_string(),
            project_root: "/tmp".to_string(),
            up_to_date: 10,
            to_add: 0,
            to_update: 0,
            to_delete: 0,
            total_tracked: 10,
            total_on_disk: 10,
            files: Vec::new(),
        };
        let serialized = serde_json::to_string(&summary).unwrap();
        // files field is skipped when empty due to skip_serializing_if
        assert!(!serialized.contains("\"files\""));
    }

    #[test]
    fn test_check_summary_roundtrip() {
        let summary = CheckSummary {
            project_id: "proj-123".to_string(),
            project_root: "/path/to/project".to_string(),
            up_to_date: 100,
            to_add: 5,
            to_update: 3,
            to_delete: 1,
            total_tracked: 104,
            total_on_disk: 108,
            files: Vec::new(),
        };
        let json_str = serde_json::to_string(&summary).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert_eq!(value["up_to_date"], 100);
        assert_eq!(value["to_add"], 5);
        assert_eq!(value["total_tracked"], 104);
        assert_eq!(value["total_on_disk"], 108);
    }
}
