//! Admin command - system administration operations
//!
//! Provides administrative operations that affect the daemon's persistent state.
//! Subcommands: rename-tenant, idle-history, prune-logs, cleanup-orphans, recover-state

use anyhow::Result;
use clap::{Args, Subcommand};

use wqm_common::constants::{
    COLLECTION_LIBRARIES, COLLECTION_PROJECTS, COLLECTION_RULES, COLLECTION_SCRATCHPAD,
};

mod cleanup_orphans;
mod idle_history;
mod perf;
mod prune_logs;
mod rename_tenant;

/// Canonical collection names (validated against wqm-common constants)
pub(super) const VALID_COLLECTIONS: &[&str] =
    &[COLLECTION_PROJECTS, COLLECTION_LIBRARIES, COLLECTION_RULES];

/// All 4 canonical collections for orphan scanning
pub(super) const ALL_COLLECTIONS: &[&str] = &[
    COLLECTION_PROJECTS,
    COLLECTION_LIBRARIES,
    COLLECTION_RULES,
    COLLECTION_SCRATCHPAD,
];

/// Admin command arguments
#[derive(Args)]
pub struct AdminArgs {
    #[command(subcommand)]
    command: AdminCommand,
}

/// Admin subcommands
#[derive(Subcommand)]
enum AdminCommand {
    /// Rename a tenant_id across SQLite tables (watch_folders, unified_queue, tracked_files)
    RenameTenant {
        /// Current tenant_id to rename
        old_id: String,

        /// New tenant_id value
        new_id: String,

        /// Skip confirmation prompt
        #[arg(short, long)]
        yes: bool,
    },
    /// Show idle/active state transition history and flip-flop analysis
    IdleHistory {
        /// Hours of history to analyze (default: 24)
        #[arg(short = 'H', long, default_value = "24")]
        hours: f64,

        /// Script-friendly space-separated output (no ANSI, one row per line)
        #[arg(long)]
        script: bool,

        /// Omit the header row (requires --script)
        #[arg(long, requires = "script")]
        no_headers: bool,
    },
    /// Prune old log files from the canonical log directory
    PruneLogs {
        /// Only list files that would be deleted, without deleting
        #[arg(long)]
        dry_run: bool,

        /// Retention period in hours (default: 36)
        #[arg(long, default_value = "36")]
        retention_hours: u64,
    },
    /// Detect and optionally delete orphaned tenants across all collections
    ///
    /// Orphans are tenant_ids (or library_names) that exist in Qdrant but have
    /// no corresponding watch_folder or tracked_files entry in SQLite.
    CleanupOrphans {
        /// Actually delete orphaned points (default: dry-run report only)
        #[arg(long)]
        delete: bool,

        /// Limit to a specific collection (default: all 4 canonical collections)
        #[arg(long)]
        collection: Option<String>,
    },
    /// Rebuild state.db from Qdrant collections
    ///
    /// Scrolls all Qdrant collections and reconstructs watch_folders, tracked_files,
    /// qdrant_chunks, and rules_mirror. Existing state.db is backed up first.
    RecoverState {
        /// Actually perform recovery (default: dry-run description)
        #[arg(long)]
        confirm: bool,
    },
    /// Display pipeline performance statistics (per-phase timing breakdown)
    Perf {
        /// Time window in hours (default: 24)
        #[arg(short = 'w', long, default_value = "24")]
        window: f64,

        /// Output in JSON format
        #[arg(long)]
        json: bool,
    },
}

/// Execute admin command
pub async fn execute(args: AdminArgs) -> Result<()> {
    match args.command {
        AdminCommand::RenameTenant {
            old_id,
            new_id,
            yes,
        } => rename_tenant::execute(old_id, new_id, yes).await,
        AdminCommand::IdleHistory {
            hours,
            script,
            no_headers,
        } => idle_history::execute(hours, script, no_headers),
        AdminCommand::PruneLogs {
            dry_run,
            retention_hours,
        } => prune_logs::execute(dry_run, retention_hours),
        AdminCommand::CleanupOrphans { delete, collection } => {
            cleanup_orphans::execute(delete, collection).await
        }
        AdminCommand::RecoverState { confirm } => super::recover_state::execute(confirm).await,
        AdminCommand::Perf { window, json } => perf::execute(window, json).await,
    }
}
