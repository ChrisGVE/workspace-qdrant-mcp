"""Rich-based formatting utilities for consistent CLI output.

This module provides Rich console formatting for all wqm CLI commands
to ensure consistent, professional-looking output across the interface.
Standardizes on Rich Panel and Table formatting used by service commands.
"""

from typing import Any, Dict, List, Optional, Union

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


# Create global console instance for consistent formatting
console = Console()


def success_panel(message: str, title: str = "Success") -> None:
    """Display a success message in a green panel.
    
    Args:
        message: Success message content
        title: Panel title (default: "Success")
    """
    console.print(
        Panel.fit(
            f"✅ {message}",
            title=title,
            style="green",
        )
    )


def error_panel(message: str, title: str = "Error") -> None:
    """Display an error message in a red panel.
    
    Args:
        message: Error message content
        title: Panel title (default: "Error")
    """
    console.print(
        Panel.fit(
            f"❌ {message}",
            title=title,
            style="red",
        )
    )


def warning_panel(message: str, title: str = "Warning") -> None:
    """Display a warning message in a yellow panel.
    
    Args:
        message: Warning message content
        title: Panel title (default: "Warning")
    """
    console.print(
        Panel.fit(
            f"⚠️ {message}",
            title=title,
            style="yellow",
        )
    )


def info_panel(message: str, title: str = "Information") -> None:
    """Display an info message in a blue panel.
    
    Args:
        message: Info message content
        title: Panel title (default: "Information")
    """
    console.print(
        Panel.fit(
            f"ℹ️ {message}",
            title=title,
            style="blue",
        )
    )


def simple_success(message: str) -> None:
    """Display a simple success message without panel.
    
    Args:
        message: Success message content
    """
    console.print(f"✅ {message}", style="green")


def simple_error(message: str) -> None:
    """Display a simple error message without panel.
    
    Args:
        message: Error message content
    """
    console.print(f"❌ {message}", style="red")


def simple_warning(message: str) -> None:
    """Display a simple warning message without panel.
    
    Args:
        message: Warning message content
    """
    console.print(f"⚠️ {message}", style="yellow")


def simple_info(message: str) -> None:
    """Display a simple info message without panel.
    
    Args:
        message: Info message content
    """
    console.print(f"ℹ️ {message}", style="blue")


def create_status_table(title: str) -> Table:
    """Create a standardized status table with consistent styling.
    
    Args:
        title: Table title
        
    Returns:
        Configured Rich Table instance
    """
    table = Table(title=title)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")
    return table


def create_data_table(title: str, headers: List[str]) -> Table:
    """Create a standardized data table with consistent styling.
    
    Args:
        title: Table title
        headers: Column headers
        
    Returns:
        Configured Rich Table instance
    """
    table = Table(title=title)
    for header in headers:
        table.add_column(header, style="white")
    return table


def display_table_or_empty(table: Table, empty_message: str) -> None:
    """Display a table if it has rows, otherwise show empty message.
    
    Args:
        table: Rich Table instance
        empty_message: Message to show if table is empty
    """
    if len(table.rows) > 0:
        console.print(table)
    else:
        simple_info(empty_message)


def format_rule_summary(rules_count: int, tokens_estimate: int) -> str:
    """Format a summary string for memory rules.
    
    Args:
        rules_count: Number of rules
        tokens_estimate: Estimated token count
        
    Returns:
        Formatted summary string
    """
    return f"Total: {rules_count} rules, ~{tokens_estimate} tokens"


def format_status_text(status: str, is_running: bool) -> Text:
    """Format status text with appropriate color coding.
    
    Args:
        status: Status string
        is_running: Whether the item is running
        
    Returns:
        Rich Text with color styling
    """
    if is_running:
        if "manual" in status.lower():
            return Text("Running (Manual)", style="green")
        else:
            return Text("Running", style="green")
    elif status.lower() == "stopped":
        return Text("Stopped", style="yellow")
    elif status.lower() == "failed":
        return Text("Failed", style="red")
    elif "not_loaded" in status.lower():
        return Text("Not Loaded", style="dim")
    else:
        return Text(status.title(), style="dim")


def format_boolean_text(value: bool, true_text: str = "Yes", false_text: str = "No") -> str:
    """Format boolean values consistently.
    
    Args:
        value: Boolean value
        true_text: Text to show for True (default: "Yes")
        false_text: Text to show for False (default: "No")
        
    Returns:
        Formatted boolean string
    """
    return true_text if value else false_text


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to maximum length with suffix.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncated (default: "...")
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def format_list_items(items: List[str], prefix: str = "  • ") -> List[str]:
    """Format a list of items with consistent bullet points.
    
    Args:
        items: List of items to format
        prefix: Prefix for each item (default: "  • ")
        
    Returns:
        List of formatted strings
    """
    return [f"{prefix}{item}" for item in items]


def print_section_header(title: str, char: str = "=") -> None:
    """Print a section header with underline.
    
    Args:
        title: Section title
        char: Character to use for underline (default: "=")
    """
    console.print(f"\n{title}")
    console.print(char * len(title))


def print_subsection_header(title: str, char: str = "-") -> None:
    """Print a subsection header with underline.
    
    Args:
        title: Subsection title
        char: Character to use for underline (default: "-")
    """
    console.print(f"\n{title}")
    console.print(char * len(title))


def display_operation_result(
    success: bool,
    success_message: str,
    error_message: str,
    success_title: str = "Success",
    error_title: str = "Error"
) -> None:
    """Display operation result with appropriate formatting.
    
    Args:
        success: Whether operation succeeded
        success_message: Message to show on success
        error_message: Message to show on error
        success_title: Title for success panel
        error_title: Title for error panel
    """
    if success:
        success_panel(success_message, success_title)
    else:
        error_panel(error_message, error_title)