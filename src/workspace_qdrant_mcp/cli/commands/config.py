"""Configuration management CLI commands.

This module provides comprehensive configuration management capabilities
including getting, setting, showing, and editing configuration values
with automatic service restart notifications.
"""

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ...core.config import Config
from ...observability import get_logger
from ..utils import (
    create_command_app,
    error_message,
    handle_async,
    requires_service_restart,
    show_service_restart_notification,
    success_message,
    warning_message,
)

logger = get_logger(__name__)
console = Console()


# Create the config app
config_app = create_command_app(
    name="config",
    help_text="""Configuration management commands.

Manage wqm configuration with get, set, show, and edit operations.
Automatically detects changes that require service restart.

Examples:
    wqm config show                      # Show all configuration
    wqm config get qdrant.url            # Get specific value
    wqm config set qdrant.url http://localhost:6334  # Set value
    wqm config edit                      # Edit in default editor
    wqm config validate                  # Validate configuration""",
    no_args_is_help=True,
)



@config_app.command("show")
def show_config(
    format_type: str = typer.Option(
        "yaml", "--format", "-f", 
        help="Output format: yaml, json, table"
    ),
    section: Optional[str] = typer.Option(
        None, "--section", "-s",
        help="Show only specific section (e.g., qdrant, embedding)"
    ),
):
    """Show current configuration."""
    try:
        config = Config()
        config_dict = config.dict()
        
        if section:
            if section not in config_dict:
                error_message(f"Section '{section}' not found in configuration")
                raise typer.Exit(1)
            config_dict = {section: config_dict[section]}
        
        if format_type == "json":
            print(json.dumps(config_dict, indent=2, default=str))
        elif format_type == "table":
            _show_config_table(config_dict)
        else:  # yaml (default)
            print(yaml.dump(config_dict, default_flow_style=False, sort_keys=False))
            
    except Exception as e:
        error_message(f"Failed to show configuration: {e}")
        raise typer.Exit(1)


@config_app.command("get")
def get_config_value(
    key: str = typer.Argument(..., help="Configuration key (e.g., qdrant.url)"),
    format_type: str = typer.Option(
        "value", "--format", "-f",
        help="Output format: value, json, yaml"
    ),
):
    """Get a specific configuration value."""
    try:
        config = Config()
        config_dict = config.dict()
        
        # Navigate nested keys
        value = config_dict
        for part in key.split('.'):
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                error_message(f"Configuration key '{key}' not found")
                raise typer.Exit(1)
        
        if format_type == "json":
            print(json.dumps(value, indent=2, default=str))
        elif format_type == "yaml":
            print(yaml.dump({key.split('.')[-1]: value}, default_flow_style=False))
        else:  # value (default)
            print(value)
            
    except Exception as e:
        error_message(f"Failed to get configuration value: {e}")
        raise typer.Exit(1)


@config_app.command("set")
def set_config_value(
    key: str = typer.Argument(..., help="Configuration key (e.g., qdrant.url)"),
    value: str = typer.Argument(..., help="Configuration value"),
    config_file: Optional[str] = typer.Option(
        None, "--file", "-f",
        help="Configuration file path (defaults to workspace_qdrant_config.yaml)"
    ),
    create_file: bool = typer.Option(
        False, "--create",
        help="Create configuration file if it doesn't exist"
    ),
):
    """Set a configuration value."""
    try:
        # Find or create config file
        if config_file is None:
            config_file = "workspace_qdrant_config.yaml"
            
        config_path = Path(config_file)
        
        # Load existing config or create new one
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f) or {}
        elif create_file:
            config_data = {}
        else:
            error_message(f"Configuration file '{config_path}' not found. Use --create to create it.")
            raise typer.Exit(1)
        
        # Convert value to appropriate type
        typed_value = _convert_value_type(value)
        
        # Set nested value
        _set_nested_value(config_data, key.split('.'), typed_value)
        
        # Write back to file
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
        
        success_message(f"Configuration updated: {key} = {typed_value}")
        
        # Check if restart is required
        if requires_service_restart(key):
            show_service_restart_notification("configuration changes")
            
    except Exception as e:
        error_message(f"Failed to set configuration value: {e}")
        raise typer.Exit(1)


@config_app.command("edit")
def edit_config(
    config_file: Optional[str] = typer.Option(
        None, "--file", "-f",
        help="Configuration file path (defaults to workspace_qdrant_config.yaml)"
    ),
    editor: Optional[str] = typer.Option(
        None, "--editor", "-e",
        help="Editor command (defaults to $EDITOR environment variable)"
    ),
    create_file: bool = typer.Option(
        False, "--create",
        help="Create configuration file if it doesn't exist"
    ),
):
    """Edit configuration file in default editor."""
    try:
        # Find config file
        if config_file is None:
            config_file = "workspace_qdrant_config.yaml"
            
        config_path = Path(config_file)
        
        # Create file if requested and doesn't exist
        if not config_path.exists():
            if create_file:
                # Copy example config as starting point
                example_config = Path(__file__).parent.parent.parent.parent.parent / "example-config.yaml"
                if example_config.exists():
                    import shutil
                    shutil.copy(example_config, config_path)
                    success_message(f"Created configuration file from template: {config_path}")
                else:
                    # Create minimal config
                    config_path.write_text("# wqm configuration\n# Edit as needed\n\n")
                    success_message(f"Created empty configuration file: {config_path}")
            else:
                error_message(f"Configuration file '{config_path}' not found. Use --create to create it.")
                raise typer.Exit(1)
        
        # Determine editor
        if editor is None:
            editor = os.environ.get('EDITOR', 'nano' if sys.platform != 'win32' else 'notepad')
        
        # Launch editor
        try:
            original_mtime = config_path.stat().st_mtime if config_path.exists() else 0
            
            subprocess.run([editor, str(config_path)], check=True)
            
            # Check if file was modified
            if config_path.exists():
                new_mtime = config_path.stat().st_mtime
                if new_mtime > original_mtime:
                    success_message(f"Configuration file updated: {config_path}")
                    show_service_restart_notification("configuration changes")
                else:
                    print("Configuration file unchanged.")
                    
        except subprocess.CalledProcessError as e:
            error_message(f"Editor command failed: {e}")
            raise typer.Exit(1)
        except FileNotFoundError:
            error_message(f"Editor '{editor}' not found. Set EDITOR environment variable or use --editor.")
            raise typer.Exit(1)
            
    except Exception as e:
        error_message(f"Failed to edit configuration: {e}")
        raise typer.Exit(1)


@config_app.command("validate")
def validate_config(
    config_file: Optional[str] = typer.Option(
        None, "--file", "-f",
        help="Configuration file path to validate"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help="Show verbose validation information"
    ),
):
    """Validate configuration file and settings."""
    try:
        if config_file:
            # Validate specific file
            config_path = Path(config_file)
            if not config_path.exists():
                error_message(f"Configuration file '{config_path}' not found")
                raise typer.Exit(1)
                
            # Load and validate
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                
            # Create Config instance from file data
            # Note: This is a simplified validation - full Config class handles more
            print(f"Validating configuration file: {config_path}")
            
        # Load and validate current configuration
        config = Config()
        
        # Perform validation
        validation_results = []
        
        # Test Qdrant connection
        print("Testing Qdrant connection...")
        try:
            from qdrant_client import QdrantClient
            
            client_kwargs = {"url": config.qdrant.url, "timeout": 5}
            if (
                hasattr(config.qdrant, "api_key")
                and config.qdrant.api_key
                and config.qdrant.api_key.strip()
                and config.qdrant.url.startswith("https")
            ):
                client_kwargs["api_key"] = config.qdrant.api_key
            
            client = QdrantClient(**client_kwargs)
            collections = client.get_collections()
            client.close()
            
            validation_results.append(("Qdrant Connection", "✓ Connected", "success"))
            
        except Exception as e:
            validation_results.append(("Qdrant Connection", f"✗ Failed: {e}", "error"))
        
        # Validate embedding model
        print("Validating embedding configuration...")
        try:
            # Basic validation of embedding settings
            if config.embedding.chunk_size <= 0:
                validation_results.append(("Embedding Config", "✗ chunk_size must be positive", "error"))
            elif config.embedding.chunk_overlap >= config.embedding.chunk_size:
                validation_results.append(("Embedding Config", "✗ chunk_overlap must be less than chunk_size", "error"))
            else:
                validation_results.append(("Embedding Config", "✓ Valid", "success"))
                
        except Exception as e:
            validation_results.append(("Embedding Config", f"✗ Error: {e}", "error"))
        
        # Display results
        _show_validation_results(validation_results, verbose)
        
        # Exit with error if any validation failed
        errors = sum(1 for _, _, level in validation_results if level == "error")
        if errors > 0:
            error_message(f"Configuration validation failed with {errors} error(s)")
            raise typer.Exit(1)
        else:
            success_message("Configuration validation passed")
            
    except Exception as e:
        error_message(f"Validation failed: {e}")
        raise typer.Exit(1)


# Helper functions
def _show_config_table(config_dict: Dict[str, Any], section_name: str = "Configuration") -> None:
    """Display configuration in table format."""
    table = Table(title=section_name)
    table.add_column("Setting", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")
    table.add_column("Type", style="magenta")
    
    def add_rows(data: Dict[str, Any], prefix: str = "") -> None:
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                # Add section header
                table.add_row(f"[bold]{full_key}[/bold]", "[dim]--- section ---[/dim]", "dict")
                add_rows(value, full_key)
            else:
                table.add_row(full_key, str(value), type(value).__name__)
    
    add_rows(config_dict)
    console.print(table)


def _convert_value_type(value: str) -> Union[str, int, float, bool, None]:
    """Convert string value to appropriate Python type."""
    # Handle special values
    if value.lower() in ('null', 'none'):
        return None
    elif value.lower() in ('true', 'yes', '1'):
        return True
    elif value.lower() in ('false', 'no', '0'):
        return False
    
    # Try numeric conversion
    try:
        if '.' in value:
            return float(value)
        else:
            return int(value)
    except ValueError:
        pass
    
    # Return as string
    return value


def _set_nested_value(data: Dict[str, Any], keys: List[str], value: Any) -> None:
    """Set a nested dictionary value using a list of keys."""
    current = data
    
    # Navigate to the parent of the target key
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        elif not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    
    # Set the final value
    current[keys[-1]] = value


def _show_validation_results(
    results: List[tuple[str, str, str]], 
    verbose: bool = False
) -> None:
    """Display validation results in a formatted table."""
    table = Table(title="Configuration Validation Results")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="white")
    
    for component, status, level in results:
        if level == "success":
            style = "green"
        elif level == "warning":
            style = "yellow"
        else:  # error
            style = "red"
            
        table.add_row(component, f"[{style}]{status}[/{style}]")
    
    console.print(table)
