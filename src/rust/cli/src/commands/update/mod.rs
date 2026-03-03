//! Update command - daemon update management
//!
//! Phase 1 HIGH priority command for updating the daemon.
//! Subcommands: check, install, --force, --version
//!
//! Module layout:
//!   github    - GitHub API types and release fetching
//!   platform  - Target triple, binary names, install location
//!   installer - Download, checksum, binary replacement, daemon lifecycle
//!   handlers  - check / check_and_install / install handler functions

use anyhow::Result;
use clap::{Args, Subcommand};

mod github;
mod handlers;
mod installer;
mod platform;

/// GitHub repository for releases
const GITHUB_REPO: &str = "ChrisGVE/workspace-qdrant-mcp";

/// Current daemon version (embedded at compile time)
const CURRENT_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Update command arguments
#[derive(Args)]
pub struct UpdateArgs {
    #[command(subcommand)]
    command: Option<UpdateCommand>,

    /// Force reinstall even if already at latest version
    #[arg(short, long)]
    force: bool,

    /// Install a specific version
    #[arg(short = 'V', long)]
    version: Option<String>,

    /// Update channel (stable, beta, rc, alpha)
    #[arg(short, long, default_value = "stable")]
    channel: String,
}

/// Update subcommands
#[derive(Subcommand)]
enum UpdateCommand {
    /// Check for updates without installing
    Check {
        /// Update channel (stable, beta, rc, alpha)
        #[arg(short, long, default_value = "stable")]
        channel: String,
    },

    /// Install the latest version (or specified version)
    Install {
        /// Force reinstall even if already at latest version
        #[arg(short, long)]
        force: bool,

        /// Install a specific version
        #[arg(short = 'V', long)]
        version: Option<String>,

        /// Update channel (stable, beta, rc, alpha)
        #[arg(short, long, default_value = "stable")]
        channel: String,
    },
}

/// Execute update command
pub async fn execute(args: UpdateArgs) -> Result<()> {
    match args.command {
        Some(UpdateCommand::Check { channel }) => handlers::check(&channel).await,
        Some(UpdateCommand::Install {
            force,
            version,
            channel,
        }) => handlers::install(force, version, &channel).await,
        None => {
            // Default: check and install if update available
            if args.version.is_some() || args.force {
                handlers::install(args.force, args.version, &args.channel).await
            } else {
                handlers::check_and_install(&args.channel).await
            }
        }
    }
}
