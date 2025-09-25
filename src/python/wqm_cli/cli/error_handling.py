"""Advanced Error Handling and Recovery for wqm CLI.

This module provides sophisticated error handling, recovery mechanisms, and
user-friendly error messages with actionable suggestions.

Task 251: Advanced CLI features with comprehensive error handling.
"""

import os
import sys
import traceback
from typing import Dict, List, Optional, Union, Any, Callable, Type
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import time
import subprocess

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.prompt import Confirm, Prompt
from loguru import logger

console = Console()


class ErrorSeverity(str, Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(str, Enum):
    """Error categories for better organization."""
    CONFIGURATION = "configuration"
    CONNECTION = "connection"
    AUTHENTICATION = "authentication"
    FILE_SYSTEM = "filesystem"
    PERMISSION = "permission"
    VALIDATION = "validation"
    SERVICE = "service"
    NETWORK = "network"
    USER_INPUT = "user_input"
    SYSTEM = "system"


@dataclass
class ErrorContext:
    """Context information for errors."""
    command: str
    subcommand: Optional[str] = None
    arguments: List[str] = None
    flags: Dict[str, Any] = None
    working_dir: str = None
    config_file: Optional[str] = None
    environment: Dict[str, str] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.arguments is None:
            self.arguments = []
        if self.flags is None:
            self.flags = {}
        if self.working_dir is None:
            self.working_dir = os.getcwd()
        if self.environment is None:
            self.environment = dict(os.environ)


@dataclass
class RecoveryAction:
    """A suggested recovery action."""
    action: str
    description: str
    command: Optional[str] = None
    auto_applicable: bool = False
    requires_confirmation: bool = True


@dataclass
class WqmError:
    """Structured error information."""
    title: str
    message: str
    category: ErrorCategory
    severity: ErrorSeverity
    context: ErrorContext
    original_exception: Optional[Exception] = None
    recovery_actions: List[RecoveryAction] = None
    related_commands: List[str] = None
    documentation_links: List[str] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.recovery_actions is None:
            self.recovery_actions = []
        if self.related_commands is None:
            self.related_commands = []
        if self.documentation_links is None:
            self.documentation_links = []


class ErrorHandler:
    """Advanced error handling and recovery system."""

    def __init__(self):
        """Initialize error handler."""
        self.error_patterns = self._initialize_error_patterns()
        self.recovery_strategies = self._initialize_recovery_strategies()
        self.last_errors: List[WqmError] = []
        self.error_history_limit = 10

    def _initialize_error_patterns(self) -> Dict[str, Callable]:
        """Initialize error pattern matchers."""
        return {
            "connection_refused": self._handle_connection_refused,
            "file_not_found": self._handle_file_not_found,
            "permission_denied": self._handle_permission_denied,
            "config_invalid": self._handle_config_invalid,
            "service_not_running": self._handle_service_not_running,
            "authentication_failed": self._handle_authentication_failed,
            "disk_space": self._handle_disk_space,
            "network_timeout": self._handle_network_timeout,
            "invalid_command": self._handle_invalid_command,
            "malformed_input": self._handle_malformed_input,
        }

    def _initialize_recovery_strategies(self) -> Dict[ErrorCategory, List[RecoveryAction]]:
        """Initialize recovery strategies by category."""
        return {
            ErrorCategory.CONNECTION: [
                RecoveryAction(
                    "check_service",
                    "Check if Qdrant service is running",
                    "wqm service status",
                    auto_applicable=True,
                    requires_confirmation=False
                ),
                RecoveryAction(
                    "start_service",
                    "Start the Qdrant service",
                    "wqm service start",
                    requires_confirmation=True
                ),
                RecoveryAction(
                    "check_config",
                    "Verify Qdrant connection configuration",
                    "wqm config get qdrant",
                    auto_applicable=True,
                    requires_confirmation=False
                ),
            ],
            ErrorCategory.CONFIGURATION: [
                RecoveryAction(
                    "validate_config",
                    "Validate configuration file",
                    "wqm config validate",
                    auto_applicable=True,
                    requires_confirmation=False
                ),
                RecoveryAction(
                    "reset_config",
                    "Reset configuration to defaults",
                    "wqm config init-unified --force",
                    requires_confirmation=True
                ),
                RecoveryAction(
                    "edit_config",
                    "Edit configuration manually",
                    "wqm config edit"
                ),
            ],
            ErrorCategory.FILE_SYSTEM: [
                RecoveryAction(
                    "check_permissions",
                    "Check file/directory permissions",
                    auto_applicable=True,
                    requires_confirmation=False
                ),
                RecoveryAction(
                    "create_directory",
                    "Create missing directories",
                    auto_applicable=True
                ),
            ],
            ErrorCategory.SERVICE: [
                RecoveryAction(
                    "restart_service",
                    "Restart the service",
                    "wqm service restart",
                    requires_confirmation=True
                ),
                RecoveryAction(
                    "check_logs",
                    "Check service logs for details",
                    "wqm service logs",
                    auto_applicable=True,
                    requires_confirmation=False
                ),
            ],
        }

    def handle_exception(
        self,
        exception: Exception,
        context: ErrorContext,
        show_traceback: bool = False
    ) -> WqmError:
        """Handle an exception with context."""
        # Classify the error
        error = self._classify_error(exception, context)

        # Add to history
        self.last_errors.append(error)
        if len(self.last_errors) > self.error_history_limit:
            self.last_errors.pop(0)

        # Display error
        self._display_error(error, show_traceback)

        # Suggest recovery actions
        self._suggest_recovery(error)

        return error

    def _classify_error(self, exception: Exception, context: ErrorContext) -> WqmError:
        """Classify an exception into structured error information."""
        error_type = type(exception).__name__
        error_message = str(exception).lower()

        # Pattern matching by exception type first
        if error_type in ["ConnectionRefusedError", "ConnectionError", "ConnectionResetError"]:
            return self._handle_connection_refused(exception, context)
        elif error_type in ["FileNotFoundError", "NotADirectoryError"]:
            return self._handle_file_not_found(exception, context)
        elif error_type in ["PermissionError"]:
            return self._handle_permission_denied(exception, context)
        elif error_type in ["TimeoutError"]:
            return self._handle_network_timeout(exception, context)
        elif error_type in ["OSError"] and ("no space left" in error_message or "disk full" in error_message):
            return self._handle_disk_space(exception, context)
        elif error_type == "ValidationError":
            return self._handle_config_invalid(exception, context)
        # Pattern matching by message content for generic exceptions
        elif "connection refused" in error_message or "connection failed" in error_message:
            return self._handle_connection_refused(exception, context)
        elif "no such file or directory" in error_message or "file not found" in error_message:
            return self._handle_file_not_found(exception, context)
        elif "permission denied" in error_message:
            return self._handle_permission_denied(exception, context)
        elif "authentication" in error_message or "unauthorized" in error_message:
            return self._handle_authentication_failed(exception, context)
        elif "timeout" in error_message:
            return self._handle_network_timeout(exception, context)
        elif "no space left" in error_message or "disk full" in error_message:
            return self._handle_disk_space(exception, context)
        elif error_type == "CommandNotFoundError" or "command not found" in error_message:
            return self._handle_invalid_command(exception, context)
        else:
            return self._handle_generic_error(exception, context)

    def _handle_connection_refused(self, exception: Exception, context: ErrorContext) -> WqmError:
        """Handle connection refused errors."""
        return WqmError(
            title="Connection Failed",
            message="Cannot connect to Qdrant server. The service may not be running or configuration may be incorrect.",
            category=ErrorCategory.CONNECTION,
            severity=ErrorSeverity.HIGH,
            context=context,
            original_exception=exception,
            recovery_actions=[
                RecoveryAction(
                    "check_service_status",
                    "Check if Qdrant service is running",
                    "wqm admin status",
                    auto_applicable=True,
                    requires_confirmation=False
                ),
                RecoveryAction(
                    "verify_config",
                    "Verify Qdrant URL configuration",
                    "wqm config get qdrant.url",
                    auto_applicable=True,
                    requires_confirmation=False
                ),
                RecoveryAction(
                    "start_qdrant",
                    "Start local Qdrant service",
                    "docker run -p 6333:6333 qdrant/qdrant",
                    requires_confirmation=True
                ),
            ],
            related_commands=["admin", "config", "service"],
            documentation_links=[
                "https://qdrant.tech/documentation/quick-start/",
                "wqm help admin"
            ]
        )

    def _handle_file_not_found(self, exception: Exception, context: ErrorContext) -> WqmError:
        """Handle file not found errors."""
        return WqmError(
            title="File Not Found",
            message=f"Required file or directory could not be found: {exception}",
            category=ErrorCategory.FILE_SYSTEM,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            original_exception=exception,
            recovery_actions=[
                RecoveryAction(
                    "check_path",
                    "Verify the file path is correct",
                    auto_applicable=True,
                    requires_confirmation=False
                ),
                RecoveryAction(
                    "create_file",
                    "Create the missing file",
                    requires_confirmation=True
                ),
                RecoveryAction(
                    "check_working_dir",
                    "Verify you're in the correct directory",
                    "pwd",
                    auto_applicable=True,
                    requires_confirmation=False
                ),
            ],
            related_commands=["config", "ingest"],
        )

    def _handle_permission_denied(self, exception: Exception, context: ErrorContext) -> WqmError:
        """Handle permission denied errors."""
        return WqmError(
            title="Permission Denied",
            message="Insufficient permissions to access the requested resource.",
            category=ErrorCategory.PERMISSION,
            severity=ErrorSeverity.HIGH,
            context=context,
            original_exception=exception,
            recovery_actions=[
                RecoveryAction(
                    "check_permissions",
                    "Check file/directory permissions",
                    f"ls -la {context.working_dir}",
                    auto_applicable=True,
                    requires_confirmation=False
                ),
                RecoveryAction(
                    "fix_permissions",
                    "Fix permissions (be careful!)",
                    requires_confirmation=True
                ),
                RecoveryAction(
                    "run_as_admin",
                    "Try running with elevated privileges",
                    requires_confirmation=True
                ),
            ],
            related_commands=["admin", "service"],
        )

    def _handle_config_invalid(self, exception: Exception, context: ErrorContext) -> WqmError:
        """Handle invalid configuration errors."""
        return WqmError(
            title="Configuration Error",
            message="Configuration file contains invalid settings or syntax errors.",
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            context=context,
            original_exception=exception,
            recovery_actions=[
                RecoveryAction(
                    "validate_config",
                    "Run configuration validation",
                    "wqm config validate",
                    auto_applicable=True,
                    requires_confirmation=False
                ),
                RecoveryAction(
                    "show_config",
                    "Show current configuration",
                    "wqm config show",
                    auto_applicable=True,
                    requires_confirmation=False
                ),
                RecoveryAction(
                    "edit_config",
                    "Edit configuration file",
                    "wqm config edit"
                ),
                RecoveryAction(
                    "reset_config",
                    "Reset to default configuration",
                    "wqm config init-unified --force",
                    requires_confirmation=True
                ),
            ],
            related_commands=["config"],
            documentation_links=["wqm help config"]
        )

    def _handle_service_not_running(self, exception: Exception, context: ErrorContext) -> WqmError:
        """Handle service not running errors."""
        return WqmError(
            title="Service Not Running",
            message="Required service is not running or accessible.",
            category=ErrorCategory.SERVICE,
            severity=ErrorSeverity.HIGH,
            context=context,
            original_exception=exception,
            recovery_actions=[
                RecoveryAction(
                    "check_status",
                    "Check service status",
                    "wqm service status",
                    auto_applicable=True,
                    requires_confirmation=False
                ),
                RecoveryAction(
                    "start_service",
                    "Start the service",
                    "wqm service start",
                    requires_confirmation=True
                ),
                RecoveryAction(
                    "check_logs",
                    "Check service logs",
                    "wqm service logs",
                    auto_applicable=True,
                    requires_confirmation=False
                ),
            ],
            related_commands=["service", "admin"]
        )

    def _handle_authentication_failed(self, exception: Exception, context: ErrorContext) -> WqmError:
        """Handle authentication failures."""
        return WqmError(
            title="Authentication Failed",
            message="Authentication credentials are invalid or missing.",
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.HIGH,
            context=context,
            original_exception=exception,
            recovery_actions=[
                RecoveryAction(
                    "check_api_key",
                    "Verify API key configuration",
                    "wqm config get qdrant.api_key",
                    auto_applicable=True,
                    requires_confirmation=False
                ),
                RecoveryAction(
                    "set_api_key",
                    "Set or update API key",
                    "wqm config set qdrant.api_key YOUR_KEY",
                    requires_confirmation=True
                ),
            ],
            related_commands=["config"]
        )

    def _handle_network_timeout(self, exception: Exception, context: ErrorContext) -> WqmError:
        """Handle network timeout errors."""
        return WqmError(
            title="Network Timeout",
            message="Network operation timed out. The server may be slow or unreachable.",
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            original_exception=exception,
            recovery_actions=[
                RecoveryAction(
                    "retry",
                    "Retry the operation",
                    auto_applicable=True
                ),
                RecoveryAction(
                    "increase_timeout",
                    "Increase timeout setting",
                    "wqm config set qdrant.timeout 60",
                    requires_confirmation=True
                ),
                RecoveryAction(
                    "check_connection",
                    "Test network connectivity",
                    auto_applicable=True,
                    requires_confirmation=False
                ),
            ],
            related_commands=["config", "admin"]
        )

    def _handle_disk_space(self, exception: Exception, context: ErrorContext) -> WqmError:
        """Handle disk space errors."""
        return WqmError(
            title="Insufficient Disk Space",
            message="Not enough disk space available for the operation.",
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.CRITICAL,
            context=context,
            original_exception=exception,
            recovery_actions=[
                RecoveryAction(
                    "check_space",
                    "Check available disk space",
                    "df -h",
                    auto_applicable=True,
                    requires_confirmation=False
                ),
                RecoveryAction(
                    "cleanup",
                    "Clean up temporary files",
                    requires_confirmation=True
                ),
            ],
            related_commands=["admin"]
        )

    def _handle_invalid_command(self, exception: Exception, context: ErrorContext) -> WqmError:
        """Handle invalid command errors."""
        return WqmError(
            title="Invalid Command",
            message=f"Command '{context.command}' is not recognized.",
            category=ErrorCategory.USER_INPUT,
            severity=ErrorSeverity.LOW,
            context=context,
            original_exception=exception,
            recovery_actions=[
                RecoveryAction(
                    "show_help",
                    "Show available commands",
                    "wqm help discover",
                    auto_applicable=True,
                    requires_confirmation=False
                ),
                RecoveryAction(
                    "suggest_similar",
                    "Find similar commands",
                    f"wqm help suggest {context.command}",
                    auto_applicable=True,
                    requires_confirmation=False
                ),
            ],
            related_commands=["help"]
        )

    def _handle_malformed_input(self, exception: Exception, context: ErrorContext) -> WqmError:
        """Handle malformed input errors."""
        return WqmError(
            title="Invalid Input",
            message="The provided input is malformed or invalid.",
            category=ErrorCategory.USER_INPUT,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            original_exception=exception,
            recovery_actions=[
                RecoveryAction(
                    "show_usage",
                    "Show command usage",
                    f"wqm {context.command} --help",
                    auto_applicable=True,
                    requires_confirmation=False
                ),
                RecoveryAction(
                    "show_examples",
                    "Show usage examples",
                    f"wqm help {context.command} --examples",
                    auto_applicable=True,
                    requires_confirmation=False
                ),
            ],
            related_commands=["help"]
        )

    def _handle_generic_error(self, exception: Exception, context: ErrorContext) -> WqmError:
        """Handle generic/unknown errors."""
        return WqmError(
            title="Unexpected Error",
            message=f"An unexpected error occurred: {exception}",
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            original_exception=exception,
            recovery_actions=[
                RecoveryAction(
                    "retry",
                    "Retry the operation",
                    auto_applicable=True
                ),
                RecoveryAction(
                    "check_logs",
                    "Check application logs",
                    auto_applicable=True,
                    requires_confirmation=False
                ),
                RecoveryAction(
                    "report_issue",
                    "Report this issue to support",
                    requires_confirmation=True
                ),
            ],
            related_commands=["admin", "observability"]
        )

    def _display_error(self, error: WqmError, show_traceback: bool = False) -> None:
        """Display error information in a user-friendly format."""
        # Color coding by severity
        severity_colors = {
            ErrorSeverity.LOW: "yellow",
            ErrorSeverity.MEDIUM: "orange1",
            ErrorSeverity.HIGH: "red",
            ErrorSeverity.CRITICAL: "bright_red"
        }

        color = severity_colors.get(error.severity, "white")

        # Create error panel
        title = f"[{color}]❌ {error.title}[/{color}]"

        content = []
        content.append(f"[bold]Error:[/bold] {error.message}")
        content.append(f"[bold]Category:[/bold] {error.category.value.title()}")
        content.append(f"[bold]Severity:[/bold] [{color}]{error.severity.value.upper()}[/{color}]")

        if error.context.command:
            cmd_parts = [error.context.command]
            if error.context.subcommand:
                cmd_parts.append(error.context.subcommand)
            if error.context.arguments:
                cmd_parts.extend(error.context.arguments)

            content.append(f"[bold]Command:[/bold] [cyan]{' '.join(cmd_parts)}[/cyan]")

        error_panel = Panel(
            "\n".join(content),
            title=title,
            border_style=color,
            padding=(1, 2)
        )

        console.print("\n")
        console.print(error_panel)

        # Show traceback if requested and available
        if show_traceback and error.original_exception:
            console.print("\n[dim]Stack trace:[/dim]")
            console.print(f"[dim]{traceback.format_exception(type(error.original_exception), error.original_exception, error.original_exception.__traceback__)}[/dim]")

    def _suggest_recovery(self, error: WqmError) -> None:
        """Suggest and optionally execute recovery actions."""
        if not error.recovery_actions:
            return

        console.print("\n💡 [bold blue]Suggested Solutions:[/bold blue]")

        # Group actions by auto-applicability
        auto_actions = [a for a in error.recovery_actions if a.auto_applicable and not a.requires_confirmation]
        manual_actions = [a for a in error.recovery_actions if not a.auto_applicable or a.requires_confirmation]

        # Execute auto actions first
        for action in auto_actions:
            if action.command:
                console.print(f"🔍 Running: [cyan]{action.command}[/cyan]")
                try:
                    result = subprocess.run(
                        action.command.split(),
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    if result.returncode == 0:
                        console.print(f"✅ {action.description}: [green]OK[/green]")
                        if result.stdout.strip():
                            console.print(f"   [dim]{result.stdout.strip()}[/dim]")
                    else:
                        console.print(f"❌ {action.description}: [red]Failed[/red]")
                        if result.stderr.strip():
                            console.print(f"   [red]{result.stderr.strip()}[/red]")
                except subprocess.TimeoutExpired:
                    console.print(f"⏱️ {action.description}: [yellow]Timeout[/yellow]")
                except Exception as e:
                    console.print(f"❌ {action.description}: [red]{e}[/red]")
            else:
                console.print(f"ℹ️  {action.description}")

        # Show manual actions
        if manual_actions:
            console.print(f"\n🛠️  [bold]Manual Actions:[/bold]")

            for i, action in enumerate(manual_actions, 1):
                console.print(f"  {i}. [yellow]{action.description}[/yellow]")
                if action.command:
                    console.print(f"     Command: [cyan]{action.command}[/cyan]")

            # Ask user if they want to execute actions
            try:
                if Confirm.ask("\nWould you like to execute any of these actions?"):
                    self._execute_manual_actions(manual_actions)
            except (KeyboardInterrupt, EOFError):
                console.print("\n[yellow]Recovery cancelled by user[/yellow]")

        # Show related commands and documentation
        if error.related_commands:
            related = ", ".join(f"[cyan]wqm {cmd}[/cyan]" for cmd in error.related_commands)
            console.print(f"\n📚 [bold]Related commands:[/bold] {related}")

        if error.documentation_links:
            console.print("\n📖 [bold]Documentation:[/bold]")
            for link in error.documentation_links:
                console.print(f"  • [blue]{link}[/blue]")

        console.print()

    def _execute_manual_actions(self, actions: List[RecoveryAction]) -> None:
        """Execute manual recovery actions based on user selection."""
        while True:
            try:
                choice = Prompt.ask(
                    "Select action number (or 'q' to quit)",
                    choices=[str(i) for i in range(1, len(actions) + 1)] + ['q']
                )

                if choice == 'q':
                    break

                action = actions[int(choice) - 1]

                if action.requires_confirmation:
                    if not Confirm.ask(f"Execute: {action.description}?"):
                        continue

                if action.command:
                    console.print(f"🔧 Executing: [cyan]{action.command}[/cyan]")
                    try:
                        result = subprocess.run(
                            action.command.split(),
                            timeout=60
                        )
                        if result.returncode == 0:
                            console.print("[green]✅ Command completed successfully[/green]")
                        else:
                            console.print(f"[red]❌ Command failed with exit code {result.returncode}[/red]")
                    except subprocess.TimeoutExpired:
                        console.print("[yellow]⏱️ Command timed out[/yellow]")
                    except Exception as e:
                        console.print(f"[red]❌ Error executing command: {e}[/red]")
                else:
                    console.print(f"ℹ️  Please manually: {action.description}")

            except (KeyboardInterrupt, EOFError):
                console.print("\n[yellow]Recovery cancelled[/yellow]")
                break
            except (ValueError, IndexError):
                console.print("[red]Invalid selection[/red]")

    @contextmanager
    def error_context(self, command: str, subcommand: Optional[str] = None, **kwargs):
        """Context manager for handling errors within a command."""
        context = ErrorContext(
            command=command,
            subcommand=subcommand,
            **kwargs
        )

        try:
            yield context
        except Exception as e:
            self.handle_exception(e, context)
            raise typer.Exit(1)

    def get_error_history(self) -> List[WqmError]:
        """Get recent error history."""
        return self.last_errors.copy()

    def clear_error_history(self) -> None:
        """Clear error history."""
        self.last_errors.clear()


# Global error handler instance
error_handler = ErrorHandler()


def handle_cli_error(exception: Exception, context: Optional[ErrorContext] = None, show_traceback: bool = False) -> None:
    """Convenience function for handling CLI errors."""
    if context is None:
        context = ErrorContext(command="unknown")

    error_handler.handle_exception(exception, context, show_traceback)


# Exception hook for uncaught exceptions
def setup_exception_hook():
    """Setup global exception hook for CLI."""
    original_hook = sys.excepthook

    def cli_exception_hook(exc_type, exc_value, exc_traceback):
        """Custom exception hook for CLI errors."""
        if exc_type == KeyboardInterrupt:
            console.print("\n[yellow]Operation cancelled by user[/yellow]")
            sys.exit(1)
        elif exc_type == typer.Exit:
            # Let typer handle its own exits
            original_hook(exc_type, exc_value, exc_traceback)
        else:
            # Handle other exceptions through our error handler
            context = ErrorContext(command=sys.argv[0] if sys.argv else "unknown")
            error_handler.handle_exception(exc_value, context, show_traceback=True)
            sys.exit(1)

    sys.excepthook = cli_exception_hook