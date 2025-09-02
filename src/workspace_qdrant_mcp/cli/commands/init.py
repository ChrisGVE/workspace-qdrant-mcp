"""Shell completion initialization command.

This module provides the `wqm init` command to generate shell completion scripts
for bash, zsh, and fish shells, following standard CLI patterns.

Usage:
    wqm init --help             # Show supported shells
    eval "$(wqm init bash)"     # Enable bash completion
    eval "$(wqm init zsh)"      # Enable zsh completion  
    eval "$(wqm init fish)"     # Enable fish completion
"""

import sys
from enum import Enum

import typer
from typer.completion import get_completion_script

# Available shells for completion
class Shell(str, Enum):
    """Supported shells for completion."""
    BASH = "bash"
    ZSH = "zsh"
    FISH = "fish"

# Initialize the init app - make it simple with direct commands
init_app = typer.Typer(
    name="init",
    help="Initialize shell completion for wqm. Supports bash, zsh, and fish shells.",
    no_args_is_help=True,
    rich_markup_mode=None  # Disable Rich formatting completely
)

@init_app.command("bash")
def bash_completion(
    prog_name: str = typer.Option(
        "wqm",
        "--prog-name",
        help="Program name for completion (default: wqm)"
    ),
) -> None:
    """Generate bash completion script for shell evaluation."""
    generate_completion_script(Shell.BASH, prog_name)

@init_app.command("zsh")
def zsh_completion(
    prog_name: str = typer.Option(
        "wqm",
        "--prog-name", 
        help="Program name for completion (default: wqm)"
    ),
) -> None:
    """Generate zsh completion script for shell evaluation."""
    generate_completion_script(Shell.ZSH, prog_name)

@init_app.command("fish")
def fish_completion(
    prog_name: str = typer.Option(
        "wqm",
        "--prog-name",
        help="Program name for completion (default: wqm)"
    ),
) -> None:
    """Generate fish completion script for shell evaluation."""
    generate_completion_script(Shell.FISH, prog_name)

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
        # Use stderr for errors so they don't interfere with script output
        print(f"Error generating completion script: {e}", file=sys.stderr)
        raise typer.Exit(1)

# Alias for backward compatibility (if needed)
cli = init_app