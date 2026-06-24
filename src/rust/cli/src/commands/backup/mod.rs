//! Backup command - Qdrant snapshot management + truth-inclusive full backup
//!
//! Subcommands (existing): create, list, delete  -- Qdrant snapshot management.
//! Flag (new, F20):        --full <dest>          -- truth-inclusive bundle.
//!
//! For restoration, use the separate 'restore' command.
//! For full restore, use: wqm restore --full <archive>

pub(crate) mod compressor;
mod create;
mod delete;
pub(crate) mod diskspace;
pub(crate) mod full;
mod list;
pub(crate) mod manifest;
pub(crate) mod stores;
pub(crate) mod types;

use std::path::PathBuf;

use anyhow::{bail, Result};
use clap::{Args, Subcommand};

use create::create_backup;
use delete::delete_backup;
use list::list_backups;

/// Backup command arguments
#[derive(Args)]
pub struct BackupArgs {
    /// Produce a truth-inclusive full backup archive (SQLite stores + Qdrant
    /// snapshot + manifest).  Specify the destination file path as the
    /// argument.  Cannot be combined with a subcommand.
    #[arg(long, value_name = "DEST")]
    full: Option<PathBuf>,

    #[command(subcommand)]
    command: Option<BackupCommand>,
}

/// Backup subcommands (Qdrant snapshot management)
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
    // --full and a subcommand are mutually exclusive.
    if args.full.is_some() && args.command.is_some() {
        bail!(
            "--full cannot be combined with a subcommand. \
             Use either `wqm backup --full <dest>` or `wqm backup <subcommand>`."
        );
    }

    if let Some(dest) = args.full {
        return full::backup_full(&dest).await;
    }

    match args.command {
        Some(BackupCommand::Create {
            collection,
            output,
            description,
            json,
        }) => create_backup(&collection, output, description, json).await,
        Some(BackupCommand::List {
            collection,
            verbose,
            json,
            script,
            no_headers,
        }) => list_backups(collection, verbose, json, script, no_headers).await,
        Some(BackupCommand::Delete {
            snapshot,
            collection,
            force,
            json,
        }) => delete_backup(&snapshot, &collection, force, json).await,
        None => {
            bail!(
                "no subcommand specified. \
                 Use `wqm backup --full <dest>` for a truth-inclusive backup, \
                 or `wqm backup <create|list|delete>` for Qdrant snapshot management."
            );
        }
    }
}
