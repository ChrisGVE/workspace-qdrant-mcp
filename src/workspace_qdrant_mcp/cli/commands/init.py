"""Shell completion initialization command.

This module provides the `wqm init` command to generate shell completion scripts
for bash, zsh, and fish shells, following standard CLI patterns.

Usage:
    wqm init bash               # Generate bash completion
    wqm init zsh                # Generate zsh completion  
    wqm init fish               # Generate fish completion
    wqm init help               # Show detailed setup instructions
"""

import sys
from enum import Enum

import typer
from typer.completion import get_completion_script

from ..utils import CLIError, create_command_app, handle_cli_error, success_message


# Available shells for completion
class Shell(str, Enum):
    """Supported shells for completion."""
    BASH = "bash"
    ZSH = "zsh"
    FISH = "fish"

# Initialize the init app using shared utilities  
init_app = create_command_app(
    name="init",
    help_text="Generate shell completion scripts for wqm command.\n\nSupported shells: bash, zsh, fish\n\nExample: eval \"$(wqm init bash)\"",
    no_args_is_help=False  # Let callback handle this
)

@init_app.callback(invoke_without_command=True)
def init_callback(ctx: typer.Context) -> None:
    """Handle init command with improved usage information."""
    if not ctx.invoked_subcommand:
        # Show custom help with better formatting
        print("Usage: wqm init SHELL")
        print("")
        print("Generate shell completion scripts for wqm command")
        print("")
        print("Available shells:")
        print("  bash    Generate bash completion script")
        print("  zsh     Generate zsh completion script") 
        print("  fish    Generate fish completion script")
        print("  help    Show detailed setup instructions")
        print("")
        print("Examples:")
        print("  eval \"$(wqm init bash)\"    # Enable bash completion")
        print("  eval \"$(wqm init zsh)\"     # Enable zsh completion")
        print("  wqm init fish | source     # Enable fish completion")
        print("  wqm init help             # Show detailed instructions")
        raise typer.Exit()


@init_app.command("bash")
def bash_completion(
    prog_name: str = typer.Option(
        "wqm",
        "--prog-name",
        help="Program name for completion (default: wqm)"
    ),
) -> None:
    """Generate bash completion script.
    
    Usage: eval "$(wqm init bash)"
    """
    generate_completion_script(Shell.BASH, prog_name)

@init_app.command("zsh")
def zsh_completion(
    prog_name: str = typer.Option(
        "wqm",
        "--prog-name", 
        help="Program name for completion (default: wqm)"
    ),
) -> None:
    """Generate zsh completion script.
    
    Usage: eval "$(wqm init zsh)"
    """
    generate_completion_script(Shell.ZSH, prog_name)

@init_app.command("fish")
def fish_completion(
    prog_name: str = typer.Option(
        "wqm",
        "--prog-name",
        help="Program name for completion (default: wqm)"
    ),
) -> None:
    """Generate fish completion script.
    
    Usage: wqm init fish | source
    """
    generate_completion_script(Shell.FISH, prog_name)

@init_app.command("help")
def detailed_help() -> None:
    """Show detailed shell completion setup instructions."""
    help_text = """
Shell Completion Setup for wqm
==============================

Quick Setup (temporary for current shell session):

  Bash:
    eval "$(wqm init bash)"
  
  Zsh:
    eval "$(wqm init zsh)"
  
  Fish:
    wqm init fish | source

Permanent installation:

  Bash - Add to ~/.bashrc or ~/.bash_profile:
    echo 'eval "$(wqm init bash)"' >> ~/.bashrc
    source ~/.bashrc
  
  Zsh - Add to ~/.zshrc:
    echo 'eval "$(wqm init zsh)"' >> ~/.zshrc
    source ~/.zshrc
  
  Fish - Add to ~/.config/fish/config.fish:
    echo 'wqm init fish | source' >> ~/.config/fish/config.fish
    source ~/.config/fish/config.fish

Verification:
  After setup, type 'wqm ' and press TAB to see available commands.
  
TROUBLESHOOTING:
  - Make sure wqm is in your PATH
  - Restart your shell after permanent installation
  - For zsh, ensure compinit is loaded before the eval statement
  - For fish, use 'wqm init fish | source' instead of eval

Custom program name:
  wqm init bash --prog-name my-wqm    # For custom command names
"""
    print(help_text.strip())
    success_message("Setup instructions displayed above")

def generate_completion_script(shell: Shell, prog_name: str) -> None:
    """Generate and output completion script for the specified shell."""
    try:
        # Generate completion script for the specified shell
        complete_var = f"_{prog_name.upper()}_COMPLETE"
        script = get_completion_script(
            prog_name=prog_name, 
            complete_var=complete_var, 
            shell=shell.value
        )
        
        # Output the script directly (no formatting for shell evaluation)
        print(script, end="")
        
    except Exception as e:
        handle_cli_error(CLIError(f"Error generating completion script: {e}"))

# Alias for backward compatibility
cli = init_app