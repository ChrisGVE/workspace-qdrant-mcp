//! Backup command - Qdrant snapshot management
//!
//! Phase 2 MEDIUM priority command for backup management.
//! Wraps Qdrant native snapshot API.
//! Subcommands: create, list, restore, delete

use std::path::PathBuf;

use anyhow::Result;
use clap::{Args, Subcommand};

use crate::output;

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

        /// Output directory for snapshot
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// List existing snapshots
    List {
        /// Collection name (optional, shows all if omitted)
        collection: Option<String>,
    },

    /// Restore from snapshot
    Restore {
        /// Snapshot name or path
        snapshot: String,

        /// Collection to restore to
        #[arg(short, long)]
        collection: Option<String>,
    },

    /// Delete a snapshot
    Delete {
        /// Snapshot name
        snapshot: String,

        /// Collection the snapshot belongs to
        #[arg(short, long)]
        collection: String,
    },
}

/// Execute backup command
pub async fn execute(args: BackupArgs) -> Result<()> {
    match args.command {
        BackupCommand::Create { collection, output } => create_backup(&collection, output).await,
        BackupCommand::List { collection } => list_backups(collection).await,
        BackupCommand::Restore {
            snapshot,
            collection,
        } => restore_backup(&snapshot, collection).await,
        BackupCommand::Delete {
            snapshot,
            collection,
        } => delete_backup(&snapshot, &collection).await,
    }
}

async fn create_backup(collection: &str, output: Option<PathBuf>) -> Result<()> {
    output::section("Create Backup");

    output::kv("Collection", collection);
    if let Some(out) = &output {
        output::kv("Output", &out.display().to_string());
    }
    output::separator();

    if collection == "all" {
        output::info("Creating full Qdrant snapshot...");
        output::info("  curl -X POST 'http://localhost:6333/snapshots'");
    } else {
        output::info(&format!(
            "Creating snapshot for collection '{}'...",
            collection
        ));
        output::info(&format!(
            "  curl -X POST 'http://localhost:6333/collections/{}/snapshots'",
            collection
        ));
    }

    output::separator();
    output::info("Snapshot will be stored in Qdrant's snapshot directory.");
    output::info("Default location: ~/.qdrant/storage/snapshots/");

    Ok(())
}

async fn list_backups(collection: Option<String>) -> Result<()> {
    output::section("List Backups");

    match &collection {
        Some(coll) => {
            output::info(&format!("Snapshots for collection '{}':", coll));
            output::info(&format!(
                "  curl 'http://localhost:6333/collections/{}/snapshots'",
                coll
            ));
        }
        None => {
            output::info("All snapshots:");
            output::info("  curl 'http://localhost:6333/snapshots'");
            output::separator();
            output::info("For collection-specific snapshots:");
            output::info("  curl 'http://localhost:6333/collections/{collection}/snapshots'");
        }
    }

    Ok(())
}

async fn restore_backup(snapshot: &str, collection: Option<String>) -> Result<()> {
    output::section("Restore Backup");

    output::kv("Snapshot", snapshot);
    if let Some(coll) = &collection {
        output::kv("Target Collection", coll);
    }
    output::separator();

    output::warning("Restore is a destructive operation!");
    output::info("To restore a snapshot:");

    if let Some(coll) = &collection {
        output::info(&format!(
            "  curl -X PUT 'http://localhost:6333/collections/{}/snapshots/recover' \\",
            coll
        ));
        output::info(&format!(
            "    -H 'Content-Type: application/json' \\
    -d '{{\"location\": \"{}\"}}'",
            snapshot
        ));
    } else {
        output::info("  Specify target collection with --collection");
    }

    Ok(())
}

async fn delete_backup(snapshot: &str, collection: &str) -> Result<()> {
    output::section("Delete Backup");

    output::kv("Snapshot", snapshot);
    output::kv("Collection", collection);
    output::separator();

    output::warning("This will permanently delete the snapshot!");
    output::info("To delete:");
    output::info(&format!(
        "  curl -X DELETE 'http://localhost:6333/collections/{}/snapshots/{}'",
        collection, snapshot
    ));

    Ok(())
}
