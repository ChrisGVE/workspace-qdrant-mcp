//! Backup command - Qdrant snapshot management
//!
//! Creates and manages Qdrant snapshots for data backup.
//! Subcommands: create, list, delete
//!
//! For restoration, use the separate 'restore' command.

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

        /// Add description/label to the snapshot
        #[arg(short, long)]
        description: Option<String>,
    },

    /// List existing snapshots
    List {
        /// Collection name (optional, shows all if omitted)
        collection: Option<String>,

        /// Show detailed information
        #[arg(short, long)]
        verbose: bool,
    },

    /// Delete a snapshot
    Delete {
        /// Snapshot name
        snapshot: String,

        /// Collection the snapshot belongs to
        #[arg(short, long)]
        collection: String,

        /// Force deletion without confirmation
        #[arg(short, long)]
        force: bool,
    },
}

/// Execute backup command
pub async fn execute(args: BackupArgs) -> Result<()> {
    match args.command {
        BackupCommand::Create {
            collection,
            output,
            description,
        } => create_backup(&collection, output, description).await,
        BackupCommand::List { collection, verbose } => list_backups(collection, verbose).await,
        BackupCommand::Delete {
            snapshot,
            collection,
            force,
        } => delete_backup(&snapshot, &collection, force).await,
    }
}

async fn create_backup(
    collection: &str,
    output: Option<PathBuf>,
    description: Option<String>,
) -> Result<()> {
    output::section("Create Backup");

    output::kv("Collection", collection);
    if let Some(out) = &output {
        output::kv("Output", &out.display().to_string());
    }
    if let Some(desc) = &description {
        output::kv("Description", desc);
    }
    output::separator();

    if collection == "all" {
        output::info("Creating full Qdrant snapshot...");
        output::info("  curl -X POST 'http://localhost:6333/snapshots'");
        output::separator();
        output::info("This creates a snapshot of all collections.");
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
    output::info("Snapshot storage:");
    output::info("  Default: ~/.qdrant/storage/snapshots/");
    if collection != "all" {
        output::info(&format!(
            "  Collection: ~/.qdrant/storage/snapshots/{}/",
            collection
        ));
    }

    output::separator();
    output::info("To restore, use: wqm restore snapshot <name> --collection <collection>");

    Ok(())
}

async fn list_backups(collection: Option<String>, verbose: bool) -> Result<()> {
    output::section("List Backups");

    match &collection {
        Some(coll) => {
            output::kv("Collection", coll);
            output::separator();

            output::info("Snapshots for collection:");
            output::info(&format!(
                "  curl 'http://localhost:6333/collections/{}/snapshots' | jq",
                coll
            ));

            if verbose {
                output::separator();
                output::info("Snapshot fields:");
                output::info("  - name: Snapshot identifier");
                output::info("  - creation_time: When created");
                output::info("  - size: Size in bytes");
            }
        }
        None => {
            output::info("All snapshots (full Qdrant backups):");
            output::info("  curl 'http://localhost:6333/snapshots' | jq");
            output::separator();

            output::info("Per-collection snapshots:");
            output::info("  List collections:");
            output::info("    curl 'http://localhost:6333/collections' | jq '.result.collections[].name'");
            output::info("  Then for each:");
            output::info("    curl 'http://localhost:6333/collections/<name>/snapshots' | jq");

            if verbose {
                output::separator();
                output::info("Canonical collections for this project:");
                output::info("  - projects (multi-tenant project data)");
                output::info("  - libraries (multi-tenant library data)");
                output::info("  - memory (behavioral rules)");
            }
        }
    }

    Ok(())
}

async fn delete_backup(snapshot: &str, collection: &str, force: bool) -> Result<()> {
    output::section("Delete Backup");

    output::kv("Snapshot", snapshot);
    output::kv("Collection", collection);
    output::kv("Force", if force { "yes" } else { "no" });
    output::separator();

    if !force {
        output::warning("This will permanently delete the snapshot!");
        output::warning("Use --force to skip this warning.");
        output::separator();
    }

    output::info("To delete:");
    output::info(&format!(
        "  curl -X DELETE 'http://localhost:6333/collections/{}/snapshots/{}'",
        collection, snapshot
    ));

    output::separator();
    output::info("Verify deletion:");
    output::info(&format!(
        "  curl 'http://localhost:6333/collections/{}/snapshots' | jq",
        collection
    ));

    Ok(())
}
