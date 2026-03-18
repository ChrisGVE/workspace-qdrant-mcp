//! Init command - setup and installation tools
//!
//! Groups setup concerns: shell completions, man page generation, and
//! Claude Code hooks management.
//!
//! Usage:
//!   wqm init completions zsh
//!   wqm init man install
//!   wqm init hooks install

use anyhow::Result;
use clap::{Args, Command, Subcommand};

pub mod completions;

/// Init command arguments
#[derive(Args)]
#[command(about = "Setup tools: shell completions, man pages, hooks")]
pub struct InitArgs {
    #[command(subcommand)]
    command: InitCommand,
}

/// Init subcommands
#[derive(Subcommand)]
pub enum InitCommand {
    /// Generate shell completion scripts (bash, zsh, fish, powershell)
    #[command(alias = "completion")]
    Completions {
        #[command(subcommand)]
        command: completions::CompletionsCommand,
    },

    /// Man page generation and installation
    Man(super::man::ManArgs),

    /// Claude Code hooks management (install, uninstall, status)
    Hooks(super::hooks::HooksArgs),
}

/// Execute init command
pub async fn execute(args: InitArgs, cmd: &mut Command) -> Result<()> {
    match args.command {
        InitCommand::Completions { command } => completions::execute(command, cmd).await,
        InitCommand::Man(man_args) => super::man::execute(man_args, cmd).await,
        InitCommand::Hooks(hooks_args) => super::hooks::execute(hooks_args).await,
    }
}
