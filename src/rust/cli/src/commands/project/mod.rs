//! Project command - project and branch management
//!
//! Subcommands: list, status, register, info, delete, activate, deactivate, check, branch

use std::path::PathBuf;

use anyhow::Result;
use clap::{Args, Subcommand};

mod activate;
mod branch;
mod check;
mod delete;
mod info;
mod list;
mod register;
pub(crate) mod resolver;
mod status;
#[cfg(test)]
mod tests;

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
        ProjectCommand::List { active, priority } => list::list_projects(active, priority).await,
        ProjectCommand::Status { path } => status::project_status(path).await,
        ProjectCommand::Register { path, name, yes } => {
            register::register_project(path, name, yes).await
        }
        ProjectCommand::Info { project } => info::project_info(project.as_deref()).await,
        ProjectCommand::Delete {
            project,
            yes,
            keep_data,
        } => delete::delete_project(project.as_deref(), yes, !keep_data).await,
        ProjectCommand::Activate { project } => {
            activate::activate_project(project.as_deref()).await
        }
        ProjectCommand::Deactivate { project } => {
            activate::deactivate_project(project.as_deref()).await
        }
        ProjectCommand::Check {
            project,
            verbose,
            json,
        } => check::check_project(project.as_deref(), verbose, json).await,
        ProjectCommand::Branch { action } => match action {
            BranchAction::List => branch::branch_list().await,
            BranchAction::Info => branch::branch_info().await,
            BranchAction::Switch { branch } => branch::branch_switch(&branch).await,
        },
    }
}
