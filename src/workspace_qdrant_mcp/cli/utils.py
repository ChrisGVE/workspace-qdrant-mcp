"""Shared CLI utilities for consistent user experience.

This module provides common patterns, error handling, and formatting
for all wqm CLI commands to ensure consistency across the interface.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import typer


class CLIError(Exception):
    """Standard CLI error with exit code and message."""

    def __init__(self, message: str, exit_code: int = 1):
        super().__init__(message)
        self.message = message
        self.exit_code = exit_code


def handle_cli_error(error: Exception, debug: bool = False) -> None:
    """Standardized CLI error handling and formatting.

    Args:
        error: Exception to handle
        debug: Whether to show debug information
    """
    if isinstance(error, CLIError):
        print(f"Error: {error.message}", file=sys.stderr)
        raise typer.Exit(error.exit_code)
    elif isinstance(error, KeyboardInterrupt):
        print("\nOperation cancelled by user", file=sys.stderr)
        raise typer.Exit(1)
    else:
        if debug:
            print(f"Error: {error}", file=sys.stderr)
            print(f"Exception type: {type(error).__name__}", file=sys.stderr)
            import traceback

            traceback.print_exc(file=sys.stderr)
        else:
            print(f"Error: {error}", file=sys.stderr)
        raise typer.Exit(1)


def handle_async(coro, debug: bool = False):
    """Standard async command wrapper with error handling.

    Args:
        coro: Async coroutine to execute
        debug: Enable debug mode for error reporting

    Returns:
        Result of the coroutine
    """
    try:
        return asyncio.run(coro)
    except Exception as e:
        handle_cli_error(e, debug=debug)


def format_table(
    headers: List[str], rows: List[List[Any]], title: Optional[str] = None
) -> str:
    """Format data as a plain text table with aligned columns.

    Args:
        headers: Table column headers
        rows: Table data rows
        title: Optional table title

    Returns:
        Formatted table string
    """
    if not rows:
        return "No data to display"

    # Calculate column widths
    col_widths = [len(header) for header in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(str(cell)))

    # Build table
    lines = []
    if title:
        lines.append(f"\n{title}")
        lines.append("=" * len(title))

    # Header row
    header_line = "  ".join(
        header.ljust(col_widths[i]) for i, header in enumerate(headers)
    )
    lines.append(header_line)
    lines.append("-" * len(header_line))

    # Data rows
    for row in rows:
        row_line = "  ".join(
            str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)
        )
        lines.append(row_line)

    return "\n".join(lines)


def confirm(prompt: str, default: bool = True) -> bool:
    """Get yes/no confirmation from user.

    Args:
        prompt: Confirmation prompt text
        default: Default value if user just presses enter

    Returns:
        True if confirmed, False if not
    """
    suffix = " [Y/n]: " if default else " [y/N]: "
    while True:
        try:
            response = input(prompt + suffix).strip().lower()
            if not response:
                return default
            if response in ("y", "yes"):
                return True
            if response in ("n", "no"):
                return False
            print("Please enter 'y' or 'n'")
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            raise typer.Exit(1)


def prompt_input(prompt: str, default: Optional[str] = None) -> str:
    """Get user input with optional default value.

    Args:
        prompt: Input prompt text
        default: Optional default value

    Returns:
        User input string
    """
    try:
        if default:
            full_prompt = f"{prompt} [{default}]: "
        else:
            full_prompt = f"{prompt}: "

        response = input(full_prompt).strip()
        return response if response else (default or "")
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        raise typer.Exit(1)


def validate_file_path(path: str, must_exist: bool = True) -> Path:
    """Validate and return a Path object.

    Args:
        path: Path string to validate
        must_exist: Whether the file must exist

    Returns:
        Validated Path object

    Raises:
        CLIError: If validation fails
    """
    try:
        file_path = Path(path).expanduser().resolve()

        if must_exist and not file_path.exists():
            raise CLIError(f"File does not exist: {path}")

        if must_exist and not file_path.is_file():
            raise CLIError(f"Path is not a file: {path}")

        return file_path
    except Exception as e:
        if isinstance(e, CLIError):
            raise
        raise CLIError(f"Invalid file path: {path}")


def validate_directory_path(
    path: str, must_exist: bool = True, create: bool = False
) -> Path:
    """Validate and return a directory Path object.

    Args:
        path: Directory path string to validate
        must_exist: Whether the directory must exist
        create: Whether to create the directory if it doesn't exist

    Returns:
        Validated Path object

    Raises:
        CLIError: If validation fails
    """
    try:
        dir_path = Path(path).expanduser().resolve()

        if not dir_path.exists():
            if create:
                dir_path.mkdir(parents=True, exist_ok=True)
            elif must_exist:
                raise CLIError(f"Directory does not exist: {path}")
        elif not dir_path.is_dir():
            raise CLIError(f"Path is not a directory: {path}")

        return dir_path
    except Exception as e:
        if isinstance(e, CLIError):
            raise
        raise CLIError(f"Invalid directory path: {path}")


# Common option factory functions for consistency
def verbose_option() -> typer.Option:
    """Standard verbose option."""
    return typer.Option(False, "--verbose", "-v", help="Enable verbose output")


def debug_option() -> typer.Option:
    """Standard debug option."""
    return typer.Option(
        False, "--debug", help="Enable debug mode with detailed error information"
    )


def json_output_option() -> typer.Option:
    """Standard JSON output option."""
    return typer.Option(False, "--json", help="Output results as JSON")


def force_option() -> typer.Option:
    """Standard force option."""
    return typer.Option(False, "--force", "-f", help="Skip confirmation prompts")


def dry_run_option() -> typer.Option:
    """Standard dry run option."""
    return typer.Option(
        False, "--dry-run", help="Show what would be done without making changes"
    )


def config_path_option() -> typer.Option:
    """Standard config path option."""
    return typer.Option(None, "--config", "-c", help="Custom configuration file path")


# Standard typer app factory with consistent settings
def create_command_app(
    name: str, help_text: str, no_args_is_help: bool = True
) -> typer.Typer:
    """Create a standardized typer app for commands.

    Args:
        name: Command name
        help_text: Help text for the command
        no_args_is_help: Whether to show help when no args provided

    Returns:
        Configured Typer app
    """
    return typer.Typer(
        name=name,
        help=help_text,
        no_args_is_help=no_args_is_help,
        rich_markup_mode=None,  # Disable Rich formatting for consistency
        add_completion=False,  # Use global completion
    )


# Success/info message formatting
def success_message(message: str) -> None:
    """Print a success message to stdout."""
    print(f"✓ {message}")


def info_message(message: str) -> None:
    """Print an info message to stdout."""
    print(f"ℹ {message}")


def warning_message(message: str) -> None:
    """Print a warning message to stderr."""
    print(f"⚠ {message}", file=sys.stderr)


def error_message(message: str) -> None:
    """Print an error message to stderr."""
    print(f"✗ {message}", file=sys.stderr)


# Alias for backward compatibility with service commands
handle_async_command = handle_async
