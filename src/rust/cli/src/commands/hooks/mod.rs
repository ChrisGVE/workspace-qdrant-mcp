//! Hooks command - Claude Code integration hook management
//!
//! Installs/uninstalls a SessionStart hook into Claude Code's settings.json
//! that injects workspace-qdrant rules into context at session start,
//! `/clear`, and compaction.
//!
//! Subcommands: install, uninstall, status

use anyhow::Result;
use clap::{Args, Subcommand};

mod install;
mod matchers;
mod settings;
mod status;
mod uninstall;

use install::install_hooks;
use status::status_hooks;
use uninstall::uninstall_hooks;

/// Hooks command arguments
#[derive(Args)]
pub struct HooksArgs {
    #[command(subcommand)]
    command: HooksCommand,
}

/// Hooks subcommands
#[derive(Subcommand)]
enum HooksCommand {
    /// Install Claude Code SessionStart hook for rule injection
    Install,

    /// Remove wqm hooks from Claude Code settings
    Uninstall,

    /// Check if hooks are installed
    Status,
}

/// Execute the hooks command
pub async fn execute(args: HooksArgs) -> Result<()> {
    match args.command {
        HooksCommand::Install => install_hooks().await,
        HooksCommand::Uninstall => uninstall_hooks().await,
        HooksCommand::Status => status_hooks().await,
    }
}
