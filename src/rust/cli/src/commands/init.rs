//! Init command - shell completion generation
//!
//! Generates shell completion scripts for bash, zsh, fish, and PowerShell.
//! Usage: eval "$(wqm init zsh)" or wqm init bash > completions.bash

use anyhow::Result;
use clap::{Args, Command, Subcommand};
use clap_complete::{generate, Shell};

/// Init command arguments
#[derive(Args)]
pub struct InitArgs {
    #[command(subcommand)]
    command: InitCommand,
}

/// Init subcommands
#[derive(Subcommand)]
enum InitCommand {
    /// Generate bash completion script
    Bash,

    /// Generate zsh completion script
    Zsh,

    /// Generate fish completion script
    Fish,

    /// Generate PowerShell completion script
    Powershell,
}

/// Execute init command — generates completion script to stdout
pub async fn execute(args: InitArgs, cmd: &mut Command) -> Result<()> {
    let shell = match args.command {
        InitCommand::Bash => Shell::Bash,
        InitCommand::Zsh => Shell::Zsh,
        InitCommand::Fish => Shell::Fish,
        InitCommand::Powershell => Shell::PowerShell,
    };

    let name = cmd.get_name().to_string();
    generate(shell, cmd, name, &mut std::io::stdout());
    Ok(())
}
