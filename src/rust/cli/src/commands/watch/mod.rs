//! Watch command - watch folder management
//!
//! Phase 1 HIGH priority command for managing file watch configurations.
//! Subcommands: list, enable, disable, show, archive, unarchive, pause,
//! resume
//!
//! Task 25: Top-level wqm watch command per spec.

mod archive;
mod enable_disable;
mod helpers;
mod list;
mod pause_resume;
mod resolver;
mod show;
mod types;

pub use types::{WatchDetailItem, WatchListItem, WatchListItemVerbose};

use anyhow::Result;
use clap::{Args, Subcommand};

/// Watch command arguments
#[derive(Args)]
pub struct WatchArgs {
    #[command(subcommand)]
    command: WatchCommand,
}

/// Watch subcommands
#[derive(Subcommand)]
enum WatchCommand {
    /// List all watch configurations
    List {
        /// Show only enabled watches
        #[arg(long)]
        enabled: bool,

        /// Show only disabled watches
        #[arg(long, conflicts_with = "enabled")]
        disabled: bool,

        /// Filter by collection name
        #[arg(short, long)]
        collection: Option<String>,

        /// Output as JSON
        #[arg(long)]
        json: bool,

        /// Show more columns
        #[arg(short, long)]
        verbose: bool,

        /// Include archived watch folders in the list
        #[arg(long)]
        show_archived: bool,
    },

    /// Enable a watch configuration
    Enable {
        /// Watch ID to enable
        watch_id: String,
    },

    /// Disable a watch configuration
    Disable {
        /// Watch ID to disable
        watch_id: String,
    },

    /// Show detailed information for a specific watch
    Show {
        /// Watch ID or path prefix
        watch_id: String,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Archive a watch folder (stops watching/ingesting, data remains
    /// searchable)
    Archive {
        /// Watch ID or path to the watch folder to archive
        watch_id: String,
    },

    /// Unarchive a watch folder (resumes watching/ingesting)
    Unarchive {
        /// Watch ID or path to the watch folder to unarchive
        watch_id: String,
    },

    /// Pause all enabled watchers (stops file event processing)
    Pause,

    /// Resume all paused watchers (restarts file event processing)
    Resume,
}

/// Execute watch command
pub async fn execute(args: WatchArgs) -> Result<()> {
    match args.command {
        WatchCommand::List {
            enabled,
            disabled,
            collection,
            json,
            verbose,
            show_archived,
        } => list::list(enabled, disabled, collection, json, verbose, show_archived).await,
        WatchCommand::Enable { watch_id } => enable_disable::enable(&watch_id).await,
        WatchCommand::Disable { watch_id } => enable_disable::disable(&watch_id).await,
        WatchCommand::Show { watch_id, json } => show::show(&watch_id, json).await,
        WatchCommand::Archive { watch_id } => archive::archive(&watch_id).await,
        WatchCommand::Unarchive { watch_id } => archive::unarchive(&watch_id).await,
        WatchCommand::Pause => pause_resume::pause().await,
        WatchCommand::Resume => pause_resume::resume().await,
    }
}
