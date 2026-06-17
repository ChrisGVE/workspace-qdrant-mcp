//! Project command - project management
//!
//! Subcommands: list, status, register, delete, search, branches, groups

use std::path::PathBuf;

use anyhow::Result;
use clap::{Args, Subcommand};

mod activate;
mod branches;
mod delete;
mod groups;
mod list;
mod recover;
mod register;
pub(crate) mod resolver;
mod search;
pub(super) mod status;
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
            status, and document count.\n\n\
            Projects are sorted by activity (active first) then by name. Orphaned projects \
            (present in Qdrant but not tracked by the daemon) are included with a warning indicator.",
        after_long_help = "Examples:\n  \
            wqm project list                            List all projects\n  \
            wqm project list --active                   List only active projects"
    )]
    List {
        /// Show only active projects
        #[arg(short, long)]
        active: bool,

        /// Show additional columns: ID, Languages, Chunks
        #[arg(short, long)]
        verbose: bool,
    },

    /// Show project status
    #[command(
        long_about = "Display comprehensive status for a project: identity, git info, content \
            statistics, and database sync state. Defaults to the current working directory \
            if no project is specified. Accepts a project name, ID, or path.\n\n\
            Works from worktrees — detects the parent project automatically.",
        after_long_help = "Examples:\n  \
            wqm project status                          Status for current directory\n  \
            wqm project status /path/to/project         Status for a specific path\n  \
            wqm project status my-project               Status by project name\n  \
            wqm project status 4ed81466dec7             Status by project ID"
    )]
    Status {
        /// Project name, ID, or path (auto-detected from CWD if omitted)
        project: Option<String>,
    },

    /// Register a project for tracking
    #[command(
        long_about = "Register a directory as a project for file watching and indexing. The daemon \
            will begin tracking file changes and building the search index. The project must \
            be a Git repository. Use --name to set a human-readable label.\n\n\
            If the directory is already part of a registered project, the command will inform \
            you and stop without prompting.",
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

    /// Delete a project and its data
    #[command(
        long_about = "Remove a project from tracking and optionally delete all associated vector \
            data from Qdrant. By default, both SQLite metadata and Qdrant vectors are removed. \
            Use --keep-data to preserve the Qdrant vectors.\n\n\
            Works from worktrees — detects the parent project automatically.",
        after_long_help = "Examples:\n  \
            wqm project delete                          Delete current project (with prompt)\n  \
            wqm project delete -y                       Delete without confirmation\n  \
            wqm project delete --keep-data              Remove tracking, keep vectors\n  \
            wqm project delete proj_abc123              Delete by project ID"
    )]
    Delete {
        /// Project name, ID, or path (auto-detected from CWD if omitted)
        project: Option<String>,

        /// Skip confirmation prompt
        #[arg(short = 'y', long)]
        yes: bool,

        /// Keep Qdrant vector data (only remove from SQLite)
        #[arg(long)]
        keep_data: bool,
    },

    /// Reconcile (recover) a drifted project registration
    #[command(
        long_about = "Reconcile a project whose registration has drifted from reality: a moved \
            directory (--new-path) or a changed git remote (--rescan-remote). Re-points the \
            stored location AND rewrites every stored file path old->new, in both SQLite and \
            Qdrant. A tenancy flip (local <-> remote) migrates all tenant-keyed data so nothing \
            is orphaned or duplicated.\n\n\
            With no flags it auto-detects the current path and remote and reconciles any drift. \
            Re-running on an already-correct registration is a no-op.",
        after_long_help = "Examples:\n  \
            wqm project recover --dry-run               Report what would change for current dir\n  \
            wqm project recover /old/path --new-path /new/path   Re-point a moved project\n  \
            wqm project recover myproj --rescan-remote  Reconcile a local<->remote tenancy flip"
    )]
    Recover {
        /// Project name, ID, or path (auto-detected from CWD if omitted)
        project: Option<String>,

        /// New filesystem path the project moved to
        #[arg(long)]
        new_path: Option<PathBuf>,

        /// Recompute tenancy from the current git remote (local <-> remote)
        #[arg(long)]
        rescan_remote: bool,

        /// Report old->new and counts without writing
        #[arg(long)]
        dry_run: bool,
    },

    /// List branches known to the index for this project
    #[command(
        long_about = "Show all git branches that have indexed files in this project. Each branch \
            is listed with its indexed file count. The currently checked-out branch is marked \
            in the Current column. Branch data comes from the `tracked_files.branches` JSON \
            arrays in the local state database — no daemon connection required.\n\n\
            Use `--branch \"*\"` on `wqm search project` to search across all branches.",
        after_long_help = "Examples:\n  \
            wqm project branches                        List branches for current project\n  \
            wqm project branches my-project             List branches for a named project\n  \
            wqm project branches /path/to/repo          List branches for a specific path"
    )]
    Branches {
        /// Project name, ID, or path (auto-detected from CWD if omitted)
        project: Option<String>,
    },

    /// Show group memberships for this project
    #[command(
        long_about = "Show project groups the current project belongs to. Groups represent \
            relationships between projects such as shared dependencies, workspace co-location, \
            Git organization, or tag affinity. Each group lists all member projects and the \
            confidence score assigned by the grouping strategy.\n\n\
            Groups are computed automatically by the daemon scheduler. Defaults to the \
            current working directory if no project is specified.",
        after_long_help = "Examples:\n  \
            wqm project groups                          Groups for current project\n  \
            wqm project groups my-project               Groups for a named project\n  \
            wqm project groups --strategy workspace     Filter by strategy\n  \
            wqm project groups --json                   Output as JSON\n  \
            wqm project groups --script                 Machine-readable output"
    )]
    Groups {
        /// Project name, ID, or path (auto-detected from CWD if omitted)
        project: Option<String>,

        /// Filter by grouping strategy (dependency, workspace, git_org, affinity, tag_affinity)
        #[arg(long)]
        strategy: Option<String>,

        /// Output as JSON
        #[arg(long)]
        json: bool,

        /// Output as tab-separated script-friendly format
        #[arg(long)]
        script: bool,

        /// Omit headers in script output
        #[arg(long)]
        no_headers: bool,
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

    // ── Hidden commands (internal) ──────────────────────────────────────
    /// Activate a project (internal)
    #[command(hide = true)]
    Activate { project: Option<String> },

    /// Deactivate a project (internal)
    #[command(hide = true)]
    Deactivate { project: Option<String> },
}

/// Execute project command
pub async fn execute(args: ProjectArgs) -> Result<()> {
    match args.command {
        ProjectCommand::List { active, verbose } => {
            list::list_projects(active, verbose, None).await
        }
        ProjectCommand::Branches { project } => branches::list_branches(project.as_deref()).await,
        ProjectCommand::Groups {
            project,
            strategy,
            json,
            script,
            no_headers,
        } => {
            groups::project_groups(
                project.as_deref(),
                strategy.as_deref(),
                json,
                script,
                no_headers,
            )
            .await
        }
        ProjectCommand::Status { project } => status::project_status(project.as_deref()).await,
        ProjectCommand::Register { path, name, yes } => {
            register::register_project(path, name, yes).await
        }
        ProjectCommand::Delete {
            project,
            yes,
            keep_data,
        } => delete::delete_project(project.as_deref(), yes, !keep_data).await,
        ProjectCommand::Recover {
            project,
            new_path,
            rescan_remote,
            dry_run,
        } => recover::recover_project(project.as_deref(), new_path, rescan_remote, dry_run).await,
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
        ProjectCommand::Activate { project } => {
            activate::activate_project(project.as_deref()).await
        }
        ProjectCommand::Deactivate { project } => {
            activate::deactivate_project(project.as_deref()).await
        }
    }
}
