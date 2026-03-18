//! Completions subcommand - shell completion generation
//!
//! Generates shell completion scripts for bash, zsh, fish, and PowerShell.
//! Usage: eval "$(wqm init completions zsh)" or wqm init completions bash > completions.bash

use anyhow::Result;
use clap::{Command, Subcommand};
use clap_complete::{generate, Shell};

/// Completions subcommands (one per shell)
#[derive(Subcommand)]
pub enum CompletionsCommand {
    /// Generate bash completion script
    Bash,

    /// Generate zsh completion script
    Zsh,

    /// Generate fish completion script
    Fish,

    /// Generate PowerShell completion script
    Powershell,
}

/// Execute completions subcommand - generates completion script to stdout
pub async fn execute(command: CompletionsCommand, cmd: &mut Command) -> Result<()> {
    let shell = match command {
        CompletionsCommand::Bash => Shell::Bash,
        CompletionsCommand::Zsh => Shell::Zsh,
        CompletionsCommand::Fish => Shell::Fish,
        CompletionsCommand::Powershell => Shell::PowerShell,
    };

    let name = cmd.get_name().to_string();
    generate(shell, cmd, name, &mut std::io::stdout());
    Ok(())
}
