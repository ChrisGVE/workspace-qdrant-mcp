"""Shared CLI utilities for consistent user experience.

This module provides common patterns, error handling, and formatting
for all wqm CLI commands to ensure consistency across the interface.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import platform
import typer
import urllib3
from qdrant_client import QdrantClient


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


# Success/info message formatting (using Rich for consistency)
def success_message(message: str) -> None:
    """Print a success message to stdout."""
    from .formatting import simple_success
    simple_success(message)


def info_message(message: str) -> None:
    """Print an info message to stdout."""
    from .formatting import simple_info
    simple_info(message)


def warning_message(message: str) -> None:
    """Print a warning message to stderr."""
    from .formatting import simple_warning
    simple_warning(message)


def show_service_restart_notification(reason: str = "configuration changes") -> None:
    """Show service restart notification with platform-appropriate commands."""
    platform_name = platform.system().lower()
    
    # Platform-appropriate terminology
    if platform_name in ["darwin", "linux"]:
        service_term = "service"
    elif platform_name == "windows":
        service_term = "service"
    else:
        service_term = "service"
    
    restart_cmd = "wqm service restart"
    
    # Enhanced warning with clear restart instruction
    print("")
    warning_message(
        f"{reason.capitalize()} require {service_term} restart to take effect."
    )
    print(f"   Run: {restart_cmd}")
    print("")


def get_platform_service_commands() -> dict:
    """Get platform-appropriate service management commands."""
    platform_name = platform.system().lower()
    
    commands = {
        "restart": "wqm service restart",
        "start": "wqm service start", 
        "stop": "wqm service stop",
        "status": "wqm service status",
        "logs": "wqm service logs"
    }
    
    # Add platform-specific system service commands if needed
    if platform_name in ["darwin", "linux", "windows"]:
        commands["system_restart"] = "wqm service restart --system"
        commands["system_status"] = "wqm service status --system"
    
    return commands


def show_service_restart_help() -> None:
    """Show comprehensive service restart help."""
    platform_name = platform.system().lower()
    commands = get_platform_service_commands()
    
    print("Service Management Commands:")
    print("=" * 50)
    print(f"  Restart service: {commands['restart']}")
    print(f"  Start service:   {commands['start']}")
    print(f"  Stop service:    {commands['stop']}")
    print(f"  Check status:    {commands['status']}")
    print(f"  View logs:       {commands['logs']}")
    
    if platform_name in ["darwin", "linux", "windows"]:
        print("\nSystem Service Commands (requires elevated privileges):")
        print("-" * 50)
        print(f"  Restart system service: {commands['system_restart']}")
        print(f"  Check system status:    {commands['system_status']}")


def error_message(message: str) -> None:
    """Print an error message to stderr."""
    from .formatting import simple_error
    simple_error(message)


# Configuration change tracking
def requires_service_restart(config_key: str) -> bool:
    """Check if a configuration key requires service restart."""
    restart_required_keys = {
        "qdrant.url",
        "qdrant.api_key", 
        "qdrant.timeout",
        "qdrant.prefer_grpc",
        "embedding.model",
        "embedding.enable_sparse_vectors", 
        "embedding.chunk_size",
        "embedding.chunk_overlap",
        "embedding.batch_size",
        # Note: workspace.collection_prefix and workspace.max_collections removed
        # as part of multi-tenant architecture migration
        "auto_ingestion.enabled",
        "auto_ingestion.auto_create_watches",
        "auto_ingestion.include_common_files",
        "auto_ingestion.include_source_files",
        "auto_ingestion.target_collection_suffix",
        "auto_ingestion.max_files_per_batch",
        "auto_ingestion.batch_delay_seconds",
        "auto_ingestion.max_file_size_mb",
        "auto_ingestion.debounce_seconds",
        "host",
        "port",
        "debug"
    }
    
    return any(config_key.startswith(pattern.split('.')[0]) and 
              (len(pattern.split('.')) == 1 or config_key == pattern) 
              for pattern in restart_required_keys)


def get_configured_client(config=None) -> QdrantClient:
    """Create a properly configured QdrantClient with SSL handling.
    
    This function centralizes the client creation pattern used across CLI commands,
    handling SSL configuration for localhost connections and providing consistent
    error handling for connection failures.
    
    Args:
        config: Optional Config object. If None, creates a new one.
        
    Returns:
        Configured QdrantClient instance
        
    Raises:
        CLIError: If client creation or connection fails
    """
    try:
        # Import here to avoid circular imports
        from common.core.config import get_config_manager
        from common.core.ssl_config import get_ssl_manager
        import warnings
        
        if config is None:
            config = get_config_manager()
            
        ssl_manager = get_ssl_manager()
        
        # Create client with comprehensive SSL warning suppression for all URLs
        with warnings.catch_warnings():
            # Suppress all SSL-related warnings
            warnings.filterwarnings("ignore", message=".*Api key is used with an insecure connection.*", category=UserWarning)
            warnings.filterwarnings("ignore", message=".*insecure connection.*", category=urllib3.exceptions.InsecureRequestWarning)
            warnings.filterwarnings("ignore", message=".*unverified HTTPS request.*", category=urllib3.exceptions.InsecureRequestWarning)
            warnings.filterwarnings("ignore", message=".*SSL.*", category=UserWarning)
            
            from common.core.ssl_config import suppress_qdrant_ssl_warnings
            with suppress_qdrant_ssl_warnings():
                # Build client parameters from config
                url = config.get("qdrant.url", "http://localhost:6333")
                api_key = config.get("qdrant.api_key")
                timeout_val = config.get("qdrant.timeout", 30)
                # Convert timeout string to number if needed
                if isinstance(timeout_val, str) and timeout_val.endswith("s"):
                    timeout = int(timeout_val[:-1])
                else:
                    timeout = timeout_val if isinstance(timeout_val, (int, float)) else 30
                
                client_params = {"url": url, "timeout": timeout}
                if api_key:
                    client_params["api_key"] = api_key
                
                client = QdrantClient(**client_params)
            
        # Test connection to ensure client is working
        try:
            with warnings.catch_warnings():
                # Apply same comprehensive warning suppression for connection test
                warnings.filterwarnings("ignore", message=".*Api key is used with an insecure connection.*", category=UserWarning)
                warnings.filterwarnings("ignore", message=".*insecure connection.*", category=urllib3.exceptions.InsecureRequestWarning)
                warnings.filterwarnings("ignore", message=".*unverified HTTPS request.*", category=urllib3.exceptions.InsecureRequestWarning)
                warnings.filterwarnings("ignore", message=".*SSL.*", category=UserWarning)
                
                if ssl_manager.is_localhost_url(url):
                    with ssl_manager.for_localhost():
                        client.get_collections()
                else:
                    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                    client.get_collections()
        except Exception as e:
            raise CLIError(
                f"Failed to connect to Qdrant server at {url}: {str(e)}"
            )
            
        return client
        
    except CLIError:
        raise
    except Exception as e:
        raise CLIError(f"Failed to create Qdrant client: {str(e)}")


