//! Restore command - Qdrant snapshot restoration + truth-inclusive full restore
//!
//! Subcommands (existing): snapshot, from-backup, list, verify
//! Flag (new, F20):        --full <archive>  -- truth-inclusive restore.
//!
//! `restore --full` refuses to run while the daemon is live (AC-F20.4).

pub(crate) mod client;
mod from_backup;
pub(crate) mod full;
mod list;
mod snapshot;
mod verify;

use std::path::PathBuf;

use anyhow::{bail, Result};
use clap::{Args, Subcommand};

/// Restore command arguments
#[derive(Args)]
pub struct RestoreArgs {
    /// Restore from a truth-inclusive full backup archive produced by
    /// `wqm backup --full`.  Specify the archive file path as the argument.
    /// The daemon must be stopped before running this command (AC-F20.4).
    /// Cannot be combined with a subcommand.
    #[arg(long, value_name = "ARCHIVE")]
    full: Option<PathBuf>,

    /// Force restore without confirmation prompt (use with --full).
    #[arg(long)]
    force: bool,

    #[command(subcommand)]
    command: Option<RestoreCommand>,
}

/// Restore subcommands (Qdrant snapshot restoration)
#[derive(Subcommand)]
enum RestoreCommand {
    /// Restore from a Qdrant snapshot
    Snapshot {
        /// Snapshot name or path
        snapshot: String,

        /// Collection to restore to
        #[arg(short, long)]
        collection: String,

        /// Force restore even if collection exists
        #[arg(short, long)]
        force: bool,
    },

    /// Restore from a local snapshot file (upload to Qdrant)
    FromBackup {
        /// Path to snapshot file
        path: std::path::PathBuf,

        /// Target collection
        #[arg(short, long)]
        collection: String,

        /// Force restore even if collection exists
        #[arg(short, long)]
        force: bool,
    },

    /// List available snapshots for restoration
    List {
        /// Show snapshots for a specific collection
        #[arg(short, long)]
        collection: Option<String>,
    },

    /// Verify a snapshot without restoring
    Verify {
        /// Snapshot name
        snapshot: String,

        /// Collection the snapshot belongs to (use 'all' for full snapshots)
        #[arg(short, long)]
        collection: String,
    },
}

/// Execute restore command
pub async fn execute(args: RestoreArgs) -> Result<()> {
    // --full and a subcommand are mutually exclusive.
    if args.full.is_some() && args.command.is_some() {
        bail!(
            "--full cannot be combined with a subcommand. \
             Use either `wqm restore --full <archive>` or `wqm restore <subcommand>`."
        );
    }

    if let Some(archive) = args.full {
        return full::restore_full(&archive, args.force).await;
    }

    match args.command {
        Some(RestoreCommand::Snapshot {
            snapshot,
            collection,
            force,
        }) => snapshot::restore_snapshot(&snapshot, &collection, force).await,
        Some(RestoreCommand::FromBackup {
            path,
            collection,
            force,
        }) => from_backup::restore_from_backup(&path, &collection, force).await,
        Some(RestoreCommand::List { collection }) => list::list_snapshots(collection).await,
        Some(RestoreCommand::Verify {
            snapshot,
            collection,
        }) => verify::verify_snapshot(&snapshot, &collection).await,
        None => {
            bail!(
                "no subcommand specified. \
                 Use `wqm restore --full <archive>` for a truth-inclusive restore, \
                 or `wqm restore <snapshot|from-backup|list|verify>` for Qdrant operations."
            );
        }
    }
}
