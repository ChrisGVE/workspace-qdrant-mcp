//! Service command - daemon lifecycle management
//!
//! Manages the memexd daemon runtime.
//! Subcommands: start, stop, restart, status, install, uninstall, logs

mod install;
mod logs;
pub mod platform;
mod restart;
mod start;
mod status;
mod stop;
mod uninstall;

use anyhow::Result;
use clap::{Args, Subcommand};

use crate::output;

/// Service command arguments
#[derive(Args)]
pub struct ServiceArgs {
    #[command(subcommand)]
    command: ServiceCommand,
}

/// Service subcommands
#[derive(Subcommand)]
enum ServiceCommand {
    /// Start the daemon
    Start,

    /// Stop the daemon
    Stop,

    /// Restart the daemon
    Restart,

    /// Show daemon status
    Status {
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Install the daemon as a system service
    Install {
        /// Path to memexd binary (auto-detected if not specified)
        #[arg(long)]
        binary: Option<String>,
    },

    /// Uninstall the daemon system service
    Uninstall {
        /// Also remove data directory (~/.workspace-qdrant/)
        #[arg(long)]
        remove_data: bool,
    },

    /// View daemon logs (convenience shortcut for `wqm debug logs`)
    Logs {
        /// Number of lines to show (default: 50)
        #[arg(short = 'n', long, default_value = "50")]
        lines: usize,

        /// Follow log output (like tail -f)
        #[arg(short, long)]
        follow: bool,

        /// Show only ERROR and WARN level entries
        #[arg(short, long)]
        errors_only: bool,
    },
}

/// Execute service command
pub async fn execute(args: ServiceArgs, cli_cmd: Option<&mut clap::Command>) -> Result<()> {
    match args.command {
        ServiceCommand::Start => start::execute().await,
        ServiceCommand::Stop => stop::execute().await,
        ServiceCommand::Restart => restart::execute().await,
        ServiceCommand::Status { json } => status::execute(json).await,
        ServiceCommand::Install { binary } => {
            install::execute(binary).await?;
            // Install man pages alongside the service (non-fatal)
            if let Some(cmd) = cli_cmd {
                if let Err(e) = super::man::install_man_pages(cmd) {
                    output::warning(format!("Man page installation failed: {}", e));
                }
            }
            Ok(())
        }
        ServiceCommand::Uninstall { remove_data } => uninstall::execute(remove_data).await,
        ServiceCommand::Logs {
            lines,
            follow,
            errors_only,
        } => logs::execute(lines, follow, errors_only).await,
    }
}
