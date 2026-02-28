//! Queue command - unified queue management and inspection
//!
//! Queue inspector and management for debugging and monitoring.
//! Subcommands: list, show, stats, retry, clean, remove

mod clean;
mod db;
pub mod formatters;
mod list;
mod remove;
mod retry;
mod show;
mod stats;
#[cfg(test)]
mod tests;

use anyhow::Result;
use clap::{Args, Subcommand};

// Re-export public types so external imports remain unchanged.
// These were public in the original single-file module.
#[allow(unused_imports)]
pub use formatters::{
    QueueDetailItem, QueueListItem, QueueListItemVerbose, QueueStatsSummary,
    StatusBreakdown,
};

/// Queue command arguments
#[derive(Args)]
pub struct QueueArgs {
    #[command(subcommand)]
    command: QueueCommand,
}

/// Queue subcommands
#[derive(Subcommand)]
enum QueueCommand {
    /// List queue items with optional filters
    List {
        /// Filter by status (pending, in_progress, done, failed)
        #[arg(short, long)]
        status: Option<String>,

        /// Filter by collection name
        #[arg(short, long)]
        collection: Option<String>,

        /// Filter by item type (file, folder, content, project, library)
        #[arg(short = 't', long)]
        item_type: Option<String>,

        /// Maximum number of items to show
        #[arg(short, long, default_value = "50")]
        limit: i64,

        /// Skip first N items (for pagination)
        #[arg(long, default_value = "0")]
        offset: i64,

        /// Order by field (created_at, priority, status)
        #[arg(short = 'o', long, default_value = "created_at")]
        order_by: String,

        /// Descending order
        #[arg(short = 'd', long)]
        desc: bool,

        /// Output as JSON
        #[arg(long)]
        json: bool,

        /// Show more columns
        #[arg(short, long)]
        verbose: bool,
    },

    /// Show detailed information for a specific queue item
    Show {
        /// Queue ID or idempotency key prefix
        queue_id: String,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Show queue statistics
    Stats {
        /// Output as JSON
        #[arg(long)]
        json: bool,

        /// Show breakdown by item type
        #[arg(short = 't', long)]
        by_type: bool,

        /// Show breakdown by operation
        #[arg(short = 'o', long)]
        by_op: bool,

        /// Show breakdown by collection
        #[arg(short = 'c', long)]
        by_collection: bool,
    },

    /// Retry failed queue items
    Retry {
        /// Queue ID to retry (omit for --all)
        queue_id: Option<String>,

        /// Retry all failed items
        #[arg(long)]
        all: bool,
    },

    /// Clean old completed or failed queue items
    Clean {
        /// Remove items older than N days (default: 7)
        #[arg(long, default_value = "7")]
        days: i64,

        /// Only clean items with this status (done, failed)
        #[arg(long)]
        status: Option<String>,

        /// Skip confirmation prompt
        #[arg(short = 'y', long)]
        yes: bool,
    },

    /// Remove a specific queue item by ID
    Remove {
        /// Queue ID or ID prefix
        queue_id: String,
    },
}

/// Execute queue command
pub async fn execute(args: QueueArgs) -> Result<()> {
    match args.command {
        QueueCommand::List {
            status,
            collection,
            item_type,
            limit,
            offset,
            order_by,
            desc,
            json,
            verbose,
        } => {
            list::execute(
                status, collection, item_type, limit, offset, &order_by, desc,
                json, verbose,
            )
            .await
        }
        QueueCommand::Show { queue_id, json } => show::execute(&queue_id, json).await,
        QueueCommand::Stats {
            json,
            by_type,
            by_op,
            by_collection,
        } => stats::execute(json, by_type, by_op, by_collection).await,
        QueueCommand::Retry { queue_id, all } => retry::execute(queue_id, all).await,
        QueueCommand::Clean { days, status, yes } => {
            clean::execute(days, status, yes).await
        }
        QueueCommand::Remove { queue_id } => remove::execute(&queue_id).await,
    }
}
