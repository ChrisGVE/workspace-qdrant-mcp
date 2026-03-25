//! Init command - setup and installation tools
//!
//! Groups setup concerns: shell completions and Claude Code hooks management.
//!
//! Usage:
//!   wqm init zsh
//!   wqm init bash
//!   wqm init hooks install

use anyhow::Result;
use clap::{Args, Command, Subcommand};
use clap_complete::{generate, Shell};

/// Init command arguments
#[derive(Args)]
#[command(about = "Setup: shell completions, hooks")]
pub struct InitArgs {
    #[command(subcommand)]
    command: InitCommand,
}

/// Init subcommands
#[derive(Subcommand)]
pub enum InitCommand {
    /// Generate bash completion script
    Bash,

    /// Generate zsh completion script
    Zsh,

    /// Generate fish completion script
    Fish,

    /// Generate PowerShell completion script
    Powershell,

    /// Claude Code hooks management (install, uninstall, status)
    Hooks(super::hooks::HooksArgs),
}

/// Execute init command
pub async fn execute(args: InitArgs, cmd: &mut Command) -> Result<()> {
    match args.command {
        InitCommand::Bash => generate_completions(Shell::Bash, cmd),
        InitCommand::Zsh => generate_completions(Shell::Zsh, cmd),
        InitCommand::Fish => generate_completions(Shell::Fish, cmd),
        InitCommand::Powershell => generate_completions(Shell::PowerShell, cmd),
        InitCommand::Hooks(hooks_args) => super::hooks::execute(hooks_args).await,
    }
}

fn generate_completions(shell: Shell, cmd: &mut Command) -> Result<()> {
    let name = cmd.get_name().to_string();
    generate(shell, cmd, name, &mut std::io::stdout());
    Ok(())
}
