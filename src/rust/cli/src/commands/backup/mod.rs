//! Backup command - Qdrant snapshot management
//!
//! Creates and manages Qdrant snapshots for data backup.
//! Subcommands: create, list, delete
//!
//! For restoration, use the separate 'restore' command.

mod create;
mod delete;
mod list;
mod types;

use std::path::PathBuf;

use anyhow::Result;
use clap::{Args, Subcommand};

use create::create_backup;
use delete::delete_backup;
use list::list_backups;

/// Backup command arguments
#[derive(Args)]
pub struct BackupArgs {
    #[command(subcommand)]
    command: BackupCommand,
}

/// Backup subcommands
#[derive(Subcommand)]
enum BackupCommand {
    /// Create a new snapshot
    Create {
        /// Collection name (or 'all' for full backup)
        #[arg(short, long, default_value = "all")]
        collection: String,

        /// Output directory to download snapshot to
        #[arg(short, long, value_parser = crate::path_arg::parse_path)]
        output: Option<PathBuf>,

        /// Add description/label to the snapshot
        #[arg(short, long)]
        description: Option<String>,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// List existing snapshots
    List {
        /// Collection name (optional, shows all if omitted)
        collection: Option<String>,

        /// Show detailed information
        #[arg(short, long)]
        verbose: bool,

        /// Output as JSON
        #[arg(long)]
        json: bool,

        /// Script-friendly space-separated output (no ANSI, one row per line)
        #[arg(long, conflicts_with = "json")]
        script: bool,

        /// Omit the header row (requires --script)
        #[arg(long, requires = "script")]
        no_headers: bool,
    },

    /// Delete a snapshot
    Delete {
        /// Snapshot name
        snapshot: String,

        /// Collection the snapshot belongs to (use 'all' for full snapshots)
        #[arg(short, long)]
        collection: String,

        /// Force deletion without confirmation
        #[arg(short, long)]
        force: bool,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },
}

/// Execute backup command
pub async fn execute(args: BackupArgs) -> Result<()> {
    match args.command {
        BackupCommand::Create {
            collection,
            output,
            description,
            json,
        } => create_backup(&collection, output, description, json).await,
        BackupCommand::List {
            collection,
            verbose,
            json,
            script,
            no_headers,
        } => list_backups(collection, verbose, json, script, no_headers).await,
        BackupCommand::Delete {
            snapshot,
            collection,
            force,
            json,
        } => delete_backup(&snapshot, &collection, force, json).await,
    }
}
