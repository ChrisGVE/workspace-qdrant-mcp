//! Restore command - Qdrant snapshot restoration
//!
//! Restores data from Qdrant snapshots.
//! Subcommands: snapshot, from-backup, list, verify

mod client;
mod from_backup;
mod list;
mod snapshot;
mod verify;

use anyhow::Result;
use clap::{Args, Subcommand};

/// Restore command arguments
#[derive(Args)]
pub struct RestoreArgs {
    #[command(subcommand)]
    command: RestoreCommand,
}

/// Restore subcommands
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
    match args.command {
        RestoreCommand::Snapshot {
            snapshot,
            collection,
            force,
        } => snapshot::restore_snapshot(&snapshot, &collection, force).await,
        RestoreCommand::FromBackup {
            path,
            collection,
            force,
        } => from_backup::restore_from_backup(&path, &collection, force).await,
        RestoreCommand::List { collection } => list::list_snapshots(collection).await,
        RestoreCommand::Verify {
            snapshot,
            collection,
        } => verify::verify_snapshot(&snapshot, &collection).await,
    }
}
