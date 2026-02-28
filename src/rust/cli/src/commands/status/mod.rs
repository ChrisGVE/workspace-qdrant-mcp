//! Status command - consolidated monitoring
//!
//! Phase 1 HIGH priority command for consolidated status monitoring.
//! Replaces old: observability, queue, messages, errors commands.
//! Subcommands: default, history, queue, watch, performance, live,
//!              messages (list/clear), errors, health

mod health;
mod history;
mod live;
mod messages;
mod overview;
mod performance;
mod queue;
mod types;
mod watch;

use anyhow::Result;
use clap::{Args, Subcommand};

use self::messages::MessageAction;

/// Status command arguments
#[derive(Args)]
pub struct StatusArgs {
    #[command(subcommand)]
    command: Option<StatusCommand>,

    /// Show queue status
    #[arg(long)]
    queue: bool,

    /// Show watch status
    #[arg(long)]
    watch: bool,

    /// Show performance metrics
    #[arg(long)]
    performance: bool,

    /// Output as JSON
    #[arg(long)]
    json: bool,
}

/// Status subcommands
#[derive(Subcommand)]
enum StatusCommand {
    /// Show historical metrics
    History {
        /// Time range (1h, 24h, 7d)
        #[arg(short, long, default_value = "1h")]
        range: String,
    },

    /// Show ingestion queue details
    Queue {
        /// Show detailed queue items
        #[arg(short, long)]
        verbose: bool,
    },

    /// Show file watcher status
    Watch,

    /// Show performance metrics
    Performance,

    /// Live updating dashboard
    Live {
        /// Refresh interval in seconds
        #[arg(short, long, default_value = "2")]
        interval: u64,
    },

    /// Message management
    Messages {
        #[command(subcommand)]
        action: Option<MessageAction>,
    },

    /// Show recent errors
    Errors {
        /// Number of errors to show
        #[arg(short = 'n', long, default_value = "10")]
        limit: usize,
    },

    /// Show system health
    Health {
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },
}

/// Execute status command
pub async fn execute(args: StatusArgs) -> Result<()> {
    let json = args.json;

    // Handle flags for default status
    if args.queue || args.watch || args.performance {
        return overview::default_status(args.queue, args.watch, args.performance, json).await;
    }

    // Handle subcommands
    match args.command {
        None => overview::default_status(false, false, false, json).await,
        Some(StatusCommand::History { range }) => history::history(&range).await,
        Some(StatusCommand::Queue { verbose }) => queue::queue(verbose).await,
        Some(StatusCommand::Watch) => watch::watch().await,
        Some(StatusCommand::Performance) => performance::performance().await,
        Some(StatusCommand::Live { interval }) => live::live(interval).await,
        Some(StatusCommand::Messages { action }) => messages::messages(action).await,
        Some(StatusCommand::Errors { limit }) => messages::errors(limit).await,
        Some(StatusCommand::Health { json: sub_json }) => health::health(json || sub_json).await,
    }
}
