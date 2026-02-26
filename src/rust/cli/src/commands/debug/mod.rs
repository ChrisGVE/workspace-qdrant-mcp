//! Debug command - consolidated diagnostics
//!
//! Provides diagnostic tools for troubleshooting the daemon and services.
//! Subcommands: logs, errors, queue-errors, language

mod errors;
mod language;
pub mod log_parsing;
mod logs;
#[cfg(test)]
mod tests;

use anyhow::Result;
use clap::{Args, Subcommand, ValueEnum};

/// Debug command arguments
#[derive(Args)]
pub struct DebugArgs {
    #[command(subcommand)]
    command: DebugCommand,
}

/// Log component filter
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
pub enum LogComponent {
    /// Show logs from all components (merged and sorted by timestamp)
    #[default]
    All,
    /// Show daemon logs only
    Daemon,
    /// Show MCP server logs only
    McpServer,
}

/// Debug subcommands
#[derive(Subcommand)]
enum DebugCommand {
    /// View daemon and MCP server logs (merged from canonical paths)
    Logs {
        /// Number of lines to show (default: 50)
        #[arg(short = 'n', long, default_value = "50")]
        lines: usize,

        /// Follow log output (like tail -f)
        #[arg(short, long)]
        follow: bool,

        /// Filter by component
        #[arg(short, long, value_enum, default_value = "all")]
        component: LogComponent,

        /// Filter by MCP session ID
        #[arg(short, long)]
        session: Option<String>,

        /// Output in JSON format (raw log entries)
        #[arg(long)]
        json: bool,

        /// Show only ERROR and WARN level entries
        #[arg(short, long)]
        errors_only: bool,

        /// Show entries since time (e.g. '1h', '30m', '2d', or ISO 8601 timestamp)
        #[arg(long)]
        since: Option<String>,
    },

    /// Show recent errors from all sources
    Errors {
        /// Number of errors to show (default: 20)
        #[arg(short = 'n', long, default_value = "20")]
        count: usize,

        /// Filter by component (daemon, queue, lsp, grammar)
        #[arg(short, long)]
        component: Option<String>,
    },

    /// Show queue processing errors
    QueueErrors {
        /// Number of errors to show (default: 20)
        #[arg(short = 'n', long, default_value = "20")]
        count: usize,

        /// Filter by operation type (ingest, update, delete)
        #[arg(short, long)]
        operation: Option<String>,
    },

    /// Diagnose language support issues
    Language {
        /// Language to diagnose
        language: String,

        /// Run verbose diagnostics
        #[arg(short, long)]
        verbose: bool,
    },
}

/// Execute debug command
pub async fn execute(args: DebugArgs) -> Result<()> {
    match args.command {
        DebugCommand::Logs {
            lines,
            follow,
            component,
            session,
            json,
            errors_only,
            since,
        } => logs::logs(lines, follow, component, session, json, errors_only, since).await,
        DebugCommand::Errors { count, component } => errors::errors(count, component).await,
        DebugCommand::QueueErrors { count, operation } => {
            errors::queue_errors(count, operation).await
        }
        DebugCommand::Language { language, verbose } => {
            language::diagnose_language(&language, verbose).await
        }
    }
}
