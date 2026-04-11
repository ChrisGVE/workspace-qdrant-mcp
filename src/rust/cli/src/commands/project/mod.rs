//! Project command - project and branch management
//!
//! Subcommands: list, status, register, info, delete, activate, deactivate, check, branch, watch

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
mod search;
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
    #[command(
        long_about = "Show all projects registered with the daemon. Displays project name, path, \
            tenant ID, active status, and document count. Use --active to filter to only \
            projects currently being watched.",
        after_long_help = "Examples:\n  \
            wqm project list                            List all projects\n  \
            wqm project list --active                   List only active projects"
    )]
    List {
        /// Show only active projects
        #[arg(short, long)]
        active: bool,
    },

    /// Show project status
    #[command(
        long_about = "Display the indexing status of a project, including document counts, \
            last sync time, and active branch. Defaults to the current working directory \
            if no path is specified.",
        after_long_help = "Examples:\n  \
            wqm project status                          Status for current directory\n  \
            wqm project status /path/to/project         Status for a specific project"
    )]
    Status {
        /// Project path (current directory if omitted)
        path: Option<PathBuf>,
    },

    /// Register a project for tracking
    #[command(
        long_about = "Register a directory as a project for file watching and indexing. The daemon \
            will begin tracking file changes and building the search index. The project must \
            be a Git repository. Use --name to set a human-readable label.",
        after_long_help = "Examples:\n  \
            wqm project register                        Register current directory\n  \
            wqm project register .                      Register current directory (explicit)\n  \
            wqm project register /path/to/repo          Register a specific path\n  \
            wqm project register . --name my-project    Register with a custom name\n  \
            wqm project register . -y                   Skip confirmation prompt"
    )]
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
    #[command(
        long_about = "Display detailed information about a project, including its tenant ID, \
            root path, registered branches, collection sizes, and configuration. \
            Auto-detects the project from the current directory if no argument is given.",
        after_long_help = "Examples:\n  \
            wqm project info                            Info for current project\n  \
            wqm project info proj_abc123                Info by project ID"
    )]
    Info {
        /// Project ID or path (auto-detected from current directory if omitted)
        project: Option<String>,
    },

    /// Delete a project and its data (auto-detects from CWD if project omitted)
    #[command(
        long_about = "Remove a project from tracking and optionally delete all associated vector \
            data from Qdrant. By default, both SQLite metadata and Qdrant vectors are removed. \
            Use --keep-data to preserve the Qdrant vectors.",
        after_long_help = "Examples:\n  \
            wqm project delete                          Delete current project (with prompt)\n  \
            wqm project delete -y                       Delete without confirmation\n  \
            wqm project delete --keep-data              Remove tracking, keep vectors\n  \
            wqm project delete proj_abc123              Delete by project ID"
    )]
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

    /// Activate a project (internal — projects activate automatically)
    #[command(hide = true)]
    Activate {
        /// Project ID or path (auto-detected from current directory if omitted)
        project: Option<String>,
    },

    /// Deactivate a project (internal — handled by daemon)
    #[command(hide = true)]
    Deactivate {
        /// Project ID or path (auto-detected from current directory if omitted)
        project: Option<String>,
    },

    /// Check ingestion status: compare tracked files against filesystem
    #[command(
        long_about = "Compare the daemon's tracked file index against the actual filesystem to \
            find missing, stale, or extra files. Useful for diagnosing indexing gaps after \
            bulk file operations or repository changes.",
        after_long_help = "Examples:\n  \
            wqm project check                           Check current project\n  \
            wqm project check --verbose                 Show per-file status\n  \
            wqm project check --json                    Output as JSON\n  \
            wqm project check proj_abc123               Check by project ID"
    )]
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

    /// Search project content (text or regex)
    #[command(
        long_about = "Full-text search across all indexed files in the current project. Supports \
            plain text and regex patterns. Results show matching lines with optional context. \
            Filter by file path globs to narrow results.",
        after_long_help = "Examples:\n  \
            wqm project search 'TODO'                   Search for text\n  \
            wqm project search 'fn\\s+main' --regex     Regex search\n  \
            wqm project search 'error' --path-glob '**/*.rs'  Filter by file type\n  \
            wqm project search 'fixme' -C 3             Show 3 lines of context\n  \
            wqm project search 'Bug' --case-sensitive   Case-sensitive search"
    )]
    Search {
        /// Search query (text string or regex pattern with --regex)
        query: String,

        /// Treat query as a regex pattern
        #[arg(long)]
        regex: bool,

        /// Case-sensitive search
        #[arg(long)]
        case_sensitive: bool,

        /// Filter by file path glob (e.g., "**/*.rs")
        #[arg(long)]
        path_glob: Option<String>,

        /// Maximum results
        #[arg(short = 'n', long, default_value = "20")]
        limit: usize,

        /// Lines of context around matches
        #[arg(short = 'C', long, default_value = "0")]
        context_lines: u32,
    },

    /// Watch folder management (list, show)
    #[command(
        long_about = "Manage file watch folders for the current project. List active watch \
            directories and their configuration.",
        after_long_help = "Examples:\n  \
            wqm project watch list                      List watch folders\n  \
            wqm project watch pause                     Pause file watching\n  \
            wqm project watch resume                    Resume file watching"
    )]
    Watch(super::watch::WatchArgs),

    /// Branch management
    #[command(
        long_about = "View and manage indexed branches for the current project. Lists branches \
            with their document counts and indexing status.",
        after_long_help = "Examples:\n  \
            wqm project branch list                     List indexed branches"
    )]
    Branch {
        #[command(subcommand)]
        action: BranchAction,
    },
}

/// Branch subcommands
#[derive(Subcommand)]
enum BranchAction {
    /// List indexed branches with document counts
    List,

    /// Show current branch info (use git branch instead)
    #[command(hide = true)]
    Info,

    /// Switch active branch for indexing (use git checkout instead)
    #[command(hide = true)]
    Switch {
        /// Branch name
        branch: String,
    },
}

/// Execute project command
pub async fn execute(args: ProjectArgs) -> Result<()> {
    match args.command {
        ProjectCommand::List { active } => list::list_projects(active, None).await,
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
        ProjectCommand::Search {
            query,
            regex,
            case_sensitive,
            path_glob,
            limit,
            context_lines,
        } => {
            search::search_project(
                &query,
                regex,
                case_sensitive,
                path_glob,
                limit,
                context_lines,
            )
            .await
        }
        ProjectCommand::Watch(args) => super::watch::execute(args).await,
        ProjectCommand::Branch { action } => match action {
            BranchAction::List => branch::branch_list().await,
            BranchAction::Info => branch::branch_info().await,
            BranchAction::Switch { branch } => branch::branch_switch(&branch).await,
        },
    }
}
