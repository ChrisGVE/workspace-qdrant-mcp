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

from common.core.config import Config
from common.core.unified_config import UnifiedConfigManager, ConfigFormat, ConfigValidationError, ConfigFormatError
from common.core.ingestion_config import IngestionConfigManager, IngestionConfig
from common.logging.loguru_config import get_logger
from ..utils import (
    create_command_app,
    error_message,
    get_configured_client,
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
        config_dict = config.model_dump()
        
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
        config_dict = config.model_dump()
        
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
            # Use unified config manager for validation
            config_path = Path(config_file)
            if not config_path.exists():
                error_message(f"Configuration file '{config_path}' not found")
                raise typer.Exit(1)
                
            config_manager = UnifiedConfigManager()
            issues = config_manager.validate_config_file(config_path)
            
            if issues:
                error_message(f"Configuration validation failed for {config_path}:")
                for issue in issues:
                    console.print(f"  - {issue}")
                raise typer.Exit(1)
            else:
                success_message(f"Configuration file is valid: {config_path}")
                return
            
        # Load and validate current configuration
        config = Config()
        
        # Perform validation
        validation_results = []
        
        # Test Qdrant connection
        print("Testing Qdrant connection...")
        try:
            client = get_configured_client(config)
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


@config_app.command("info")
def config_info(
    config_dir: Optional[str] = typer.Option(
        None, "--config-dir", "-d",
        help="Configuration directory to search"
    ),
    format_type: Optional[str] = typer.Option(
        None, "--format", "-f",
        help="Preferred configuration format (toml, yaml, json)"
    ),
):
    """Display configuration information and discovered sources."""
    try:
        prefer_format = ConfigFormat(format_type) if format_type else None
        config_manager = UnifiedConfigManager(config_dir=config_dir)
        
        info_data = config_manager.get_config_info()
        
        # Create display
        console.print("\n[bold blue]Configuration Information[/bold blue]")
        console.print(f"Config Directory: [green]{info_data['config_dir']}[/green]")
        console.print(f"Environment Prefix: [cyan]{info_data['env_prefix']}[/cyan]")
        
        # Sources table
        table = Table(title="Discovered Configuration Sources")
        table.add_column("File", style="cyan")
        table.add_column("Format", style="magenta")
        table.add_column("Exists", style="green")
        table.add_column("Last Modified", style="yellow")
        
        for source in info_data['sources']:
            exists_icon = "✓" if source['exists'] else "✗"
            exists_style = "green" if source['exists'] else "red"
            last_mod = source['last_modified']
            last_mod_str = f"{last_mod:.0f}" if last_mod else "N/A"
            
            table.add_row(
                source['file_path'],
                source['format'],
                f"[{exists_style}]{exists_icon}[/{exists_style}]",
                last_mod_str
            )
        
        console.print(table)
        
        # Preferred source
        if info_data['preferred_source']:
            console.print(f"\n[bold green]Preferred Source:[/bold green] {info_data['preferred_source']}")
        else:
            console.print("\n[bold red]No configuration files found[/bold red]")
        
        # Current config status
        config_loaded = info_data['current_config_loaded']
        status_text = "Loaded" if config_loaded else "Not loaded"
        status_color = "green" if config_loaded else "yellow"
        console.print(f"Configuration Status: [{status_color}]{status_text}[/{status_color}]")
        
    except Exception as e:
        error_message(f"Error getting configuration info: {e}")
        raise typer.Exit(1)


@config_app.command("convert")
def convert_config(
    source_file: str = typer.Argument(..., help="Source configuration file"),
    target_file: str = typer.Argument(..., help="Target configuration file"),
    target_format: Optional[str] = typer.Option(
        None, "--target-format", "-f",
        help="Target format - toml, yaml, or json (auto-detect from extension if not specified)"
    ),
    validate_first: bool = typer.Option(
        True, "--validate/--no-validate",
        help="Validate source before conversion"
    ),
):
    """Convert configuration file between formats."""
    try:
        source_path = Path(source_file)
        target_path = Path(target_file)
        
        if not source_path.exists():
            error_message(f"Source configuration file not found: {source_path}")
            raise typer.Exit(1)
        
        config_manager = UnifiedConfigManager()
        
        if validate_first:
            console.print("Validating source configuration...")
            issues = config_manager.validate_config_file(source_path)
            if issues:
                error_message("Source configuration is invalid:")
                for issue in issues:
                    console.print(f"  - {issue}")
                raise typer.Exit(1)
            success_message("✓ Source configuration is valid")
        
        target_fmt = ConfigFormat(target_format) if target_format else None
        
        console.print(f"Converting [cyan]{source_file}[/cyan] to [cyan]{target_file}[/cyan]...")
        config_manager.convert_config(source_path, target_path, target_fmt)
        
        success_message("✓ Configuration converted successfully")
        
        # Validate target file
        target_issues = config_manager.validate_config_file(target_path)
        if target_issues:
            warning_message("Warning: Converted configuration has issues:")
            for issue in target_issues:
                console.print(f"  - {issue}")
        else:
            success_message("✓ Converted configuration is valid")
            
    except Exception as e:
        error_message(f"Error converting configuration: {e}")
        raise typer.Exit(1)


@config_app.command("init-unified")
def init_unified_config(
    config_dir: Optional[str] = typer.Option(
        None, "--config-dir", "-d",
        help="Configuration directory"
    ),
    formats: List[str] = typer.Option(
        ["toml", "yaml"], "--format", "-f",
        help="Formats to create (can specify multiple)"
    ),
    force: bool = typer.Option(
        False, "--force",
        help="Overwrite existing files"
    ),
):
    """Initialize default configuration files in multiple formats."""
    try:
        config_manager = UnifiedConfigManager(config_dir=config_dir)
        
        # Validate format choices
        valid_formats = ['toml', 'yaml', 'json']
        invalid_formats = [f for f in formats if f not in valid_formats]
        if invalid_formats:
            error_message(f"Invalid formats: {invalid_formats}. Valid options: {valid_formats}")
            raise typer.Exit(1)
        
        format_objs = [ConfigFormat(f) for f in formats]
        
        if force:
            # Remove existing files if force is specified
            for fmt in format_objs:
                if fmt == ConfigFormat.TOML:
                    file_path = config_manager.config_dir / "workspace_qdrant_config.toml"
                elif fmt == ConfigFormat.YAML:
                    file_path = config_manager.config_dir / "workspace_qdrant_config.yaml"
                elif fmt == ConfigFormat.JSON:
                    file_path = config_manager.config_dir / "workspace_qdrant_config.json"
                
                if file_path.exists():
                    file_path.unlink()
                    warning_message(f"Removed existing: {file_path}")
        
        created_files = config_manager.create_default_configs(format_objs)
        
        if created_files:
            success_message("Created configuration files:")
            for fmt, file_path in created_files.items():
                console.print(f"  {fmt.value}: [cyan]{file_path}[/cyan]")
        else:
            warning_message("All specified configuration files already exist")
            console.print("Use --force to overwrite existing files")
        
    except Exception as e:
        error_message(f"Error initializing configuration: {e}")
        raise typer.Exit(1)


@config_app.command("watch")
def watch_config(
    config_file: Optional[str] = typer.Option(
        None, "--file", "-f",
        help="Specific config file to watch"
    ),
    config_dir: Optional[str] = typer.Option(
        None, "--config-dir", "-d",
        help="Configuration directory to search"
    ),
    interval: int = typer.Option(
        1, "--interval", "-i",
        help="Check interval in seconds"
    ),
):
    """Watch configuration file for changes and validate on change."""
    try:
        config_manager = UnifiedConfigManager(config_dir=config_dir)
        
        if config_file:
            config_path = Path(config_file)
            if not config_path.exists():
                error_message(f"Configuration file not found: {config_path}")
                raise typer.Exit(1)
            console.print(f"Watching configuration file: [cyan]{config_file}[/cyan]")
        else:
            source = config_manager.get_preferred_config_source()
            if source and source.exists:
                console.print(f"Watching configuration file: [cyan]{source.file_path}[/cyan]")
            else:
                error_message("No configuration file found to watch")
                raise typer.Exit(1)
        
        def on_config_change(new_config: Config):
            from datetime import datetime
            timestamp = datetime.now().strftime("%H:%M:%S")
            console.print(f"\n[yellow][{timestamp}] Configuration changed[/yellow]")
            issues = new_config.validate_config()
            if issues:
                error_message(f"Configuration has {len(issues)} issues:")
                for issue in issues:
                    console.print(f"  - {issue}")
                
                # Show service restart notification for config changes
                show_service_restart_notification("configuration changes")
            else:
                success_message("✓ Configuration is valid")
        
        console.print("Press Ctrl+C to stop watching...")
        config_manager.watch_config(on_config_change)
        
        try:
            while True:
                import time
                time.sleep(interval)
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopping configuration watcher...[/yellow]")
            config_manager.stop_watching()
        
    except Exception as e:
        error_message(f"Error watching configuration: {e}")
        raise typer.Exit(1)


@config_app.command("env-vars")
def show_env_vars(
    config_dir: Optional[str] = typer.Option(
        None, "--config-dir", "-d",
        help="Configuration directory to search"
    ),
):
    """Show environment variables that can override configuration."""
    try:
        config_manager = UnifiedConfigManager(config_dir=config_dir)
        
        console.print("\n[bold blue]Environment Variable Overrides[/bold blue]")
        console.print(f"Prefix: [cyan]{config_manager.env_prefix}[/cyan]")
        
        # Create sections for different config areas
        sections = [
            ("Server Configuration", [
                (f"{config_manager.env_prefix}HOST", "Server host address"),
                (f"{config_manager.env_prefix}PORT", "Server port number"),
                (f"{config_manager.env_prefix}DEBUG", "Enable debug mode (true/false)"),
            ]),
            ("Qdrant Configuration", [
                (f"{config_manager.env_prefix}QDRANT__URL", "Qdrant server URL"),
                (f"{config_manager.env_prefix}QDRANT__API_KEY", "Qdrant API key"),
                (f"{config_manager.env_prefix}QDRANT__TIMEOUT", "Connection timeout"),
                (f"{config_manager.env_prefix}QDRANT__PREFER_GRPC", "Use gRPC protocol"),
            ]),
            ("Embedding Configuration", [
                (f"{config_manager.env_prefix}EMBEDDING__MODEL", "Embedding model name"),
                (f"{config_manager.env_prefix}EMBEDDING__ENABLE_SPARSE_VECTORS", "Enable sparse vectors"),
                (f"{config_manager.env_prefix}EMBEDDING__CHUNK_SIZE", "Text chunk size"),
                (f"{config_manager.env_prefix}EMBEDDING__CHUNK_OVERLAP", "Chunk overlap size"),
                (f"{config_manager.env_prefix}EMBEDDING__BATCH_SIZE", "Processing batch size"),
            ]),
            ("Workspace Configuration", [
                (f"{config_manager.env_prefix}WORKSPACE__COLLECTION_SUFFIXES", "Collection suffixes (comma-separated)"),
                (f"{config_manager.env_prefix}WORKSPACE__GLOBAL_COLLECTIONS", "Global collections (comma-separated)"),
                (f"{config_manager.env_prefix}WORKSPACE__GITHUB_USER", "GitHub username"),
                (f"{config_manager.env_prefix}WORKSPACE__COLLECTION_PREFIX", "Collection name prefix"),
                (f"{config_manager.env_prefix}WORKSPACE__MAX_COLLECTIONS", "Maximum collections limit"),
                (f"{config_manager.env_prefix}WORKSPACE__AUTO_CREATE_COLLECTIONS", "Auto-create collections"),
            ]),
            ("Auto-Ingestion Configuration", [
                (f"{config_manager.env_prefix}AUTO_INGESTION__ENABLED", "Enable auto-ingestion"),
                (f"{config_manager.env_prefix}AUTO_INGESTION__AUTO_CREATE_WATCHES", "Auto-create file watches"),
                (f"{config_manager.env_prefix}AUTO_INGESTION__TARGET_COLLECTION_SUFFIX", "Target collection suffix"),
            ]),
            ("gRPC Configuration", [
                (f"{config_manager.env_prefix}GRPC__ENABLED", "Enable gRPC"),
                (f"{config_manager.env_prefix}GRPC__HOST", "gRPC server host"),
                (f"{config_manager.env_prefix}GRPC__PORT", "gRPC server port"),
            ]),
        ]
        
        for section_name, env_vars in sections:
            table = Table(title=section_name)
            table.add_column("Environment Variable", style="cyan")
            table.add_column("Description", style="white")
            
            for env_var, description in env_vars:
                table.add_row(env_var, description)
            
            console.print(table)
            console.print()  # Add spacing between tables
        
        # Show example
        console.print("[bold yellow]Example Usage:[/bold yellow]")
        console.print(f"export {config_manager.env_prefix}QDRANT__URL=http://localhost:6333")
        console.print(f"export {config_manager.env_prefix}EMBEDDING__MODEL=sentence-transformers/all-MiniLM-L6-v2")
        console.print(f"export {config_manager.env_prefix}DEBUG=true")
        
    except Exception as e:
        error_message(f"Error displaying environment variables: {e}")
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


# Ingestion configuration commands

@config_app.command("ingestion", help="Ingestion configuration management")
def ingestion_command():
    """Ingestion configuration management commands.
    
    Use subcommands:
    - wqm config ingestion show    # Show current ingestion config
    - wqm config ingestion edit    # Edit ingestion config
    - wqm config ingestion validate # Validate ingestion config
    - wqm config ingestion reset   # Reset to defaults
    """
    console.print("[yellow]Use subcommands: show, edit, validate, reset[/yellow]")
    console.print("Example: [cyan]wqm config ingestion show[/cyan]")


@config_app.command("ingestion-show")
def show_ingestion_config(
    format_type: str = typer.Option(
        "yaml", "--format", "-f",
        help="Output format: yaml, json, table"
    ),
    section: Optional[str] = typer.Option(
        None, "--section", "-s",
        help="Show specific section (ignore_patterns, performance, languages)"
    ),
):
    """Show current ingestion configuration."""
    try:
        manager = IngestionConfigManager()
        config = manager.load_config()
        config_dict = config.model_dump()
        
        if section:
            if section not in config_dict:
                error_message(f"Section '{section}' not found in ingestion configuration")
                raise typer.Exit(1)
            config_dict = {section: config_dict[section]}
        
        if format_type == "json":
            print(json.dumps(config_dict, indent=2, default=str))
        elif format_type == "table":
            _show_ingestion_config_table(config_dict)
        else:  # yaml (default)
            print(yaml.dump(config_dict, default_flow_style=False, sort_keys=False))
            
    except Exception as e:
        error_message(f"Failed to show ingestion configuration: {e}")
        raise typer.Exit(1)


@config_app.command("ingestion-edit")
def edit_ingestion_config(
    editor: Optional[str] = typer.Option(
        None, "--editor", "-e",
        help="Editor to use (defaults to $EDITOR environment variable)"
    ),
):
    """Edit ingestion configuration in your preferred editor."""
    try:
        manager = IngestionConfigManager()
        
        # Find existing config or create template
        config_file = manager._find_ingestion_config()
        if not config_file:
            # Create default config file
            config_file = Path.cwd() / "ingestion.yaml"
            default_config = manager._get_default_config()
            with config_file.open('w') as f:
                yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)
            success_message(f"Created default ingestion config: {config_file}")
        
        # Determine editor
        editor_cmd = editor or os.getenv('EDITOR')
        if not editor_cmd:
            if platform.system() == 'Darwin':  # macOS
                editor_cmd = 'open -t'
            elif platform.system() == 'Windows':
                editor_cmd = 'notepad'
            else:  # Linux/Unix
                editor_cmd = 'nano'
        
        console.print(f"Opening [cyan]{config_file}[/cyan] with [yellow]{editor_cmd}[/yellow]")
        
        # Open editor
        if platform.system() == 'Darwin' and editor_cmd == 'open -t':
            subprocess.run(['open', '-t', str(config_file)], check=True)
        else:
            subprocess.run([editor_cmd, str(config_file)], check=True)
        
        # Validate after editing
        try:
            manager.load_config(config_file)
            success_message("Ingestion configuration is valid")
        except Exception as e:
            warning_message(f"Configuration validation failed: {e}")
            
    except subprocess.CalledProcessError:
        error_message(f"Failed to open editor: {editor_cmd}")
        raise typer.Exit(1)
    except Exception as e:
        error_message(f"Failed to edit ingestion configuration: {e}")
        raise typer.Exit(1)


@config_app.command("ingestion-validate")
def validate_ingestion_config(
    config_file: Optional[str] = typer.Option(
        None, "--config", "-c",
        help="Specific config file to validate"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help="Show detailed validation information"
    ),
):
    """Validate ingestion configuration."""
    try:
        manager = IngestionConfigManager()
        
        if config_file:
            config_path = Path(config_file)
            if not config_path.exists():
                error_message(f"Configuration file not found: {config_file}")
                raise typer.Exit(1)
            config = manager.load_config(config_path)
        else:
            config = manager.load_config()
        
        # Validate configuration
        issues = config.validate_config()
        
        if not issues:
            success_message("✓ Ingestion configuration is valid")
            
            if verbose:
                info = manager.get_config_info()
                console.print("\n[bold blue]Configuration Summary[/bold blue]")
                console.print(f"Status: [green]{info['status']}[/green]")
                console.print(f"Enabled: [cyan]{info['enabled']}[/cyan]")
                console.print(f"Languages supported: [cyan]{info['languages_supported']}[/cyan]")
                console.print(f"Ignore directories: [cyan]{info['ignore_directories']}[/cyan]")
                console.print(f"Ignore extensions: [cyan]{info['ignore_extensions']}[/cyan]")
                console.print(f"Pattern cache size: [cyan]{info['pattern_cache_size']}[/cyan]")
        else:
            error_message("✗ Ingestion configuration validation failed:")
            for issue in issues:
                console.print(f"  [red]•[/red] {issue}")
            raise typer.Exit(1)
            
    except Exception as e:
        error_message(f"Failed to validate ingestion configuration: {e}")
        raise typer.Exit(1)


@config_app.command("ingestion-reset")
def reset_ingestion_config(
    target_file: Optional[str] = typer.Option(
        None, "--output", "-o",
        help="Output file for reset config (default: ingestion.yaml)"
    ),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="Overwrite existing file without confirmation"
    ),
):
    """Reset ingestion configuration to defaults."""
    try:
        manager = IngestionConfigManager()
        
        # Determine output file
        if target_file:
            output_path = Path(target_file)
        else:
            output_path = Path.cwd() / "ingestion.yaml"
        
        # Check if file exists and confirm overwrite
        if output_path.exists() and not force:
            overwrite = typer.confirm(
                f"File {output_path} already exists. Overwrite?"
            )
            if not overwrite:
                console.print("Operation cancelled.")
                return
        
        # Create default configuration
        default_config = manager._get_default_config()
        
        # Write to file
        with output_path.open('w') as f:
            yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)
        
        success_message(f"✓ Reset ingestion configuration to defaults: {output_path}")
        console.print("\n[yellow]Note:[/yellow] Edit the file to customize patterns for your project")
        console.print(f"[cyan]wqm config ingestion-edit[/cyan] to edit")
        
    except Exception as e:
        error_message(f"Failed to reset ingestion configuration: {e}")
        raise typer.Exit(1)


@config_app.command("ingestion-info")
def show_ingestion_info():
    """Show ingestion system information and statistics."""
    try:
        manager = IngestionConfigManager()
        info = manager.get_config_info()
        
        if info["status"] == "not_loaded":
            warning_message("Ingestion configuration not loaded")
            console.print("Run [cyan]wqm config ingestion-reset[/cyan] to create default config")
            return
        
        # Show configuration info
        console.print("\n[bold blue]Ingestion System Information[/bold blue]")
        
        status_style = "green" if info["enabled"] else "red"
        console.print(f"Status: [{status_style}]{info['status'].upper()}[/{status_style}]")
        console.print(f"Enabled: [{status_style}]{info['enabled']}[/{status_style}]")
        
        console.print(f"\n[bold]Pattern Statistics:[/bold]")
        console.print(f"• Languages supported: [cyan]{info['languages_supported']}[/cyan]")
        console.print(f"• Ignore directories: [cyan]{info['ignore_directories']}[/cyan]")
        console.print(f"• Ignore extensions: [cyan]{info['ignore_extensions']}[/cyan]")
        console.print(f"• Pattern cache size: [cyan]{info['pattern_cache_size']}[/cyan]")
        
        console.print(f"\n[bold]Performance Settings:[/bold]")
        perf = info["performance_settings"]
        console.print(f"• Max file size: [cyan]{perf['max_file_size_mb']} MB[/cyan]")
        console.print(f"• Max files per directory: [cyan]{perf['max_files_per_directory']}[/cyan]")
        console.print(f"• Max files per batch: [cyan]{perf['max_files_per_batch']}[/cyan]")
        console.print(f"• Debounce seconds: [cyan]{perf['debounce_seconds']}[/cyan]")
        
        # Show supported languages
        console.print(f"\n[bold]Supported Languages:[/bold]")
        languages = [
            "JavaScript/TypeScript", "Python", "Rust", "Java/JVM", "Go", "C/C++",
            "C#/.NET", "Ruby", "PHP", "Swift", "Dart/Flutter", "R", "Julia",
            "Haskell", "Elixir", "And 10+ more..."
        ]
        for i, lang in enumerate(languages[:8], 1):
            console.print(f"  {i:2}. [cyan]{lang}[/cyan]")
        if len(languages) > 8:
            console.print(f"  ... [cyan]{languages[-1]}[/cyan]")
            
    except Exception as e:
        error_message(f"Failed to show ingestion information: {e}")
        raise typer.Exit(1)


def _show_ingestion_config_table(config_dict: Dict[str, Any]) -> None:
    """Display ingestion configuration in a formatted table."""
    
    # Overview table
    overview_table = Table(title="Ingestion Configuration Overview")
    overview_table.add_column("Setting", style="cyan")
    overview_table.add_column("Value", style="white")
    
    if "enabled" in config_dict:
        status = "[green]Enabled[/green]" if config_dict["enabled"] else "[red]Disabled[/red]"
        overview_table.add_row("Status", status)
    
    console.print(overview_table)
    
    # Ignore patterns table if present
    if "ignore_patterns" in config_dict:
        patterns = config_dict["ignore_patterns"]
        
        patterns_table = Table(title="Ignore Patterns")
        patterns_table.add_column("Type", style="cyan")
        patterns_table.add_column("Count", style="white")
        patterns_table.add_column("Examples", style="dim")
        
        if "directories" in patterns:
            dirs = patterns["directories"]
            examples = ", ".join(dirs[:3]) + ("..." if len(dirs) > 3 else "")
            patterns_table.add_row("Directories", str(len(dirs)), examples)
            
        if "file_extensions" in patterns:
            exts = patterns["file_extensions"]
            examples = ", ".join(exts[:3]) + ("..." if len(exts) > 3 else "")
            patterns_table.add_row("File Extensions", str(len(exts)), examples)
            
        console.print(patterns_table)
    
    # Performance settings table if present
    if "performance" in config_dict:
        perf = config_dict["performance"]
        
        perf_table = Table(title="Performance Settings")
        perf_table.add_column("Setting", style="cyan")
        perf_table.add_column("Value", style="white")
        
        for key, value in perf.items():
            display_key = key.replace("_", " ").title()
            if "mb" in key.lower():
                display_value = f"{value} MB"
            elif "seconds" in key.lower():
                display_value = f"{value} seconds"
            else:
                display_value = str(value)
            perf_table.add_row(display_key, display_value)
            
        console.print(perf_table)
