"""Shell completion initialization command.

This module provides the `wqm init` command to generate shell completion scripts
for bash, zsh, and fish shells, following standard CLI patterns.

Usage:
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
    help="Initialize shell completion for wqm",
    no_args_is_help=True,
)

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
    
    Usage: eval "$(wqm init fish)"
    """
    generate_completion_script(Shell.FISH, prog_name)

@init_app.command("help")
def show_help() -> None:
    """Show detailed help for shell completion setup."""
    help_text = """
Shell Completion Setup for wqm

The 'wqm init' command generates shell completion scripts that can be 
evaluated to enable tab completion for wqm commands and options.

BASIC USAGE:
    eval "$(wqm init SHELL)"

SUPPORTED SHELLS:
    bash    - Bash shell completion
    zsh     - Zsh shell completion  
    fish    - Fish shell completion

EXAMPLES:

  Temporary activation (current session only):
    eval "$(wqm init bash)"      # For Bash
    eval "$(wqm init zsh)"       # For Zsh  
    eval "$(wqm init fish)"      # For Fish

  Permanent installation:
  
    Bash:
      wqm init bash > ~/.bash_completion.d/wqm
      # Or add to ~/.bashrc:
      echo 'eval "$(wqm init bash)"' >> ~/.bashrc

    Zsh:
      wqm init zsh > ~/.zsh/completions/_wqm
      # Or add to ~/.zshrc:
      echo 'eval "$(wqm init zsh)"' >> ~/.zshrc
      
    Fish:
      wqm init fish > ~/.config/fish/completions/wqm.fish

WHAT YOU GET:
  - Tab completion for all wqm commands
  - Tab completion for command options  
  - Context-aware argument completion
  - Help text display during completion

TROUBLESHOOTING:
  - Make sure wqm is in your PATH
  - Restart your shell after permanent installation
  - For fish, ensure the completions directory exists
  - Use 'wqm init SHELL' to regenerate if needed
"""
    print(help_text.strip())

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