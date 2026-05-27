//! Queue command - unified queue management and inspection
//!
//! Queue inspector and management for debugging and monitoring.
//! Subcommands: list, show, stats, retry, clean, remove, cancel

mod cancel;
mod clean;
mod dlq;
mod drop;
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
    QueueDetailItem, QueueListItem, QueueListItemVerbose, QueueStatsSummary, StatusBreakdown,
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

        /// Maximum number of items to show (default: 50)
        #[arg(short, long, default_value = "50", conflicts_with = "all")]
        limit: i64,

        /// Skip first N items (for pagination)
        #[arg(long, default_value = "0")]
        offset: i64,

        /// Show all items (override default page size)
        #[arg(short, long)]
        all: bool,

        /// Order by field (created_at, priority, status)
        #[arg(short = 'o', long, default_value = "created_at")]
        order_by: String,

        /// Descending order
        #[arg(short = 'd', long)]
        desc: bool,

        /// Output as JSON
        #[arg(long)]
        json: bool,

        /// Script-friendly space-separated output (no ANSI, one row per line)
        #[arg(long, conflicts_with = "json")]
        script: bool,

        /// Omit the header row (requires --script)
        #[arg(long, requires = "script")]
        no_headers: bool,

        /// Show the ID column
        #[arg(long)]
        id: bool,

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

        /// Retry only transient failed items (clears resurrection count)
        #[arg(long)]
        all_transient: bool,
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

    /// Drop failed items from the queue
    Drop {
        /// Queue ID to drop (omit for bulk operations)
        queue_id: Option<String>,

        /// Drop all permanently failed items ([permanent_*] prefix)
        #[arg(long)]
        all_permanent: bool,

        /// Drop all stale items (files no longer on disk)
        #[arg(long)]
        all_stale: bool,

        /// Skip confirmation prompt
        #[arg(short = 'y', long)]
        yes: bool,
    },

    /// Cancel all pending (or failed) queue items for a project
    ///
    /// Resolves the project by tenant ID, exact path, or name substring.
    /// In-progress items are never cancelled. Use --dry-run to preview.
    Cancel {
        /// Project name, path, or tenant ID
        project: String,

        /// Statuses to cancel (default: pending); comma-separated, e.g. "pending,failed"
        #[arg(long, default_value = "pending", value_delimiter = ',')]
        status: Vec<String>,

        /// Preview count without deleting
        #[arg(long)]
        dry_run: bool,

        /// Skip confirmation prompt
        #[arg(short = 'y', long)]
        yes: bool,
    },

    /// Dead letter queue management
    Dlq {
        #[command(subcommand)]
        command: dlq::DlqCommand,
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
            all,
            order_by,
            desc,
            json,
            script,
            no_headers,
            id,
            verbose,
        } => {
            let effective_limit = if all { i64::MAX } else { limit };
            list::execute(
                status,
                collection,
                item_type,
                effective_limit,
                offset,
                &order_by,
                desc,
                json,
                script,
                no_headers,
                verbose,
                id || verbose,
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
        QueueCommand::Retry {
            queue_id,
            all,
            all_transient,
        } => retry::execute(queue_id, all, all_transient).await,
        QueueCommand::Clean { days, status, yes } => clean::execute(days, status, yes).await,
        QueueCommand::Remove { queue_id } => remove::execute(&queue_id).await,
        QueueCommand::Drop {
            queue_id,
            all_permanent,
            all_stale,
            yes,
        } => drop::execute(queue_id, all_permanent, all_stale, yes).await,
        QueueCommand::Cancel {
            project,
            status,
            dry_run,
            yes,
        } => {
            let status_refs: Vec<&str> = status.iter().map(String::as_str).collect();
            cancel::execute(&project, &status_refs, dry_run, yes).await
        }
        QueueCommand::Dlq { command } => dlq::execute(command).await,
    }
}
