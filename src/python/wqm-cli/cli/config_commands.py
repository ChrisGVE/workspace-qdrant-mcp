"""
CLI commands for configuration management.

This module provides command-line interface commands for managing unified
configuration files including validation, format conversion, and information display.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree

from ..core.unified_config import UnifiedConfigManager, ConfigFormat, ConfigValidationError, ConfigFormatError

console = Console()
logger = logging.getLogger(__name__)


@click.group(name='config')
def config_group():
    """Configuration management commands."""
    pass


@config_group.command()
@click.option('--config-dir', '-d', type=click.Path(exists=True), 
              help='Configuration directory to search')
@click.option('--format', '-f', type=click.Choice(['toml', 'yaml', 'json']),
              help='Preferred configuration format')
def info(config_dir: Optional[str], format: Optional[str]):
    """Display configuration information and discovered sources."""
    try:
        prefer_format = ConfigFormat(format) if format else None
        config_manager = UnifiedConfigManager(config_dir=config_dir)
        
        info_data = config_manager.get_config_info()
        
        # Create rich display
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
        console.print(f"[red]Error getting configuration info: {e}[/red]", err=True)
        sys.exit(1)


@config_group.command()
@click.option('--config-dir', '-d', type=click.Path(), 
              help='Configuration directory')
@click.option('--format', '-f', type=click.Choice(['toml', 'yaml', 'json']), 
              multiple=True, default=['toml', 'yaml'],
              help='Formats to create (can specify multiple)')
@click.option('--force', is_flag=True, help='Overwrite existing files')
def init(config_dir: Optional[str], format: tuple, force: bool):
    """Initialize default configuration files."""
    try:
        config_manager = UnifiedConfigManager(config_dir=config_dir)
        formats = [ConfigFormat(f) for f in format]
        
        if force:
            # Remove existing files if force is specified
            for fmt in formats:
                if fmt == ConfigFormat.TOML:
                    file_path = config_manager.config_dir / "workspace_qdrant_config.toml"
                elif fmt == ConfigFormat.YAML:
                    file_path = config_manager.config_dir / "workspace_qdrant_config.yaml"
                elif fmt == ConfigFormat.JSON:
                    file_path = config_manager.config_dir / "workspace_qdrant_config.json"
                
                if file_path.exists():
                    file_path.unlink()
                    console.print(f"[yellow]Removed existing: {file_path}[/yellow]")
        
        created_files = config_manager.create_default_configs(formats)
        
        if created_files:
            console.print("\n[bold green]Created configuration files:[/bold green]")
            for fmt, file_path in created_files.items():
                console.print(f"  {fmt.value}: [cyan]{file_path}[/cyan]")
        else:
            console.print("[yellow]All specified configuration files already exist[/yellow]")
            console.print("Use --force to overwrite existing files")
        
    except Exception as e:
        console.print(f"[red]Error initializing configuration: {e}[/red]", err=True)
        sys.exit(1)


@config_group.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--format', '-f', type=click.Choice(['toml', 'yaml', 'json']),
              help='Override format detection')
def validate(config_file: str, format: Optional[str]):
    """Validate configuration file."""
    try:
        config_path = Path(config_file)
        
        # Use override format if specified
        if format:
            # Temporarily change extension for format detection override
            original_suffix = config_path.suffix
            temp_path = config_path.with_suffix(f'.{format}')
            config_path.rename(temp_path)
            try:
                issues = UnifiedConfigManager().validate_config_file(temp_path)
            finally:
                temp_path.rename(config_path)
        else:
            issues = UnifiedConfigManager().validate_config_file(config_path)
        
        if not issues:
            console.print(f"[green]✓ Configuration file is valid: {config_file}[/green]")
        else:
            console.print(f"[red]✗ Configuration validation failed: {config_file}[/red]")
            console.print("\n[bold red]Issues found:[/bold red]")
            for i, issue in enumerate(issues, 1):
                console.print(f"  {i}. {issue}")
            sys.exit(1)
            
    except Exception as e:
        console.print(f"[red]Error validating configuration: {e}[/red]", err=True)
        sys.exit(1)


@config_group.command()
@click.argument('source_file', type=click.Path(exists=True))
@click.argument('target_file', type=click.Path())
@click.option('--target-format', '-f', type=click.Choice(['toml', 'yaml', 'json']),
              help='Target format (auto-detect from extension if not specified)')
@click.option('--validate', is_flag=True, help='Validate before conversion')
def convert(source_file: str, target_file: str, target_format: Optional[str], validate: bool):
    """Convert configuration file between formats."""
    try:
        source_path = Path(source_file)
        target_path = Path(target_file)
        
        config_manager = UnifiedConfigManager()
        
        if validate:
            console.print("Validating source configuration...")
            issues = config_manager.validate_config_file(source_path)
            if issues:
                console.print(f"[red]Source configuration is invalid:[/red]")
                for issue in issues:
                    console.print(f"  - {issue}")
                sys.exit(1)
            console.print("[green]✓ Source configuration is valid[/green]")
        
        target_fmt = ConfigFormat(target_format) if target_format else None
        
        console.print(f"Converting [cyan]{source_file}[/cyan] to [cyan]{target_file}[/cyan]...")
        config_manager.convert_config(source_path, target_path, target_fmt)
        
        console.print(f"[green]✓ Configuration converted successfully[/green]")
        
        # Validate target file
        target_issues = config_manager.validate_config_file(target_path)
        if target_issues:
            console.print(f"[yellow]Warning: Converted configuration has issues:[/yellow]")
            for issue in target_issues:
                console.print(f"  - {issue}")
        else:
            console.print(f"[green]✓ Converted configuration is valid[/green]")
            
    except Exception as e:
        console.print(f"[red]Error converting configuration: {e}[/red]", err=True)
        sys.exit(1)


@config_group.command()
@click.argument('config_file', type=click.Path(exists=True), required=False)
@click.option('--config-dir', '-d', type=click.Path(exists=True),
              help='Configuration directory to search')
@click.option('--format', '-f', type=click.Choice(['toml', 'yaml', 'json']),
              help='Preferred configuration format for auto-discovery')
@click.option('--output', '-o', type=click.Choice(['yaml', 'json']), default='yaml',
              help='Output format for display')
def show(config_file: Optional[str], config_dir: Optional[str], 
         format: Optional[str], output: str):
    """Load and display configuration."""
    try:
        prefer_format = ConfigFormat(format) if format else None
        config_manager = UnifiedConfigManager(config_dir=config_dir)
        
        if config_file:
            config = config_manager.load_config(config_file)
            console.print(f"Loaded configuration from: [cyan]{config_file}[/cyan]")
        else:
            config = config_manager.load_config(prefer_format=prefer_format)
            source = config_manager.get_preferred_config_source(prefer_format)
            if source:
                console.print(f"Loaded configuration from: [cyan]{source.file_path}[/cyan]")
            else:
                console.print("Using default configuration (no config file found)")
        
        # Display configuration
        config_dict = config_manager._config_to_dict(config)
        
        if output == 'json':
            import json
            console.print("\n[bold blue]Configuration (JSON):[/bold blue]")
            console.print(json.dumps(config_dict, indent=2))
        else:
            import yaml
            console.print("\n[bold blue]Configuration (YAML):[/bold blue]")
            console.print(yaml.dump(config_dict, default_flow_style=False, sort_keys=False))
        
        # Show validation status
        issues = config.validate_config()
        if issues:
            console.print(f"\n[yellow]Configuration has {len(issues)} validation issues:[/yellow]")
            for issue in issues:
                console.print(f"  - {issue}")
        else:
            console.print(f"\n[green]✓ Configuration is valid[/green]")
            
    except Exception as e:
        console.print(f"[red]Error loading configuration: {e}[/red]", err=True)
        sys.exit(1)


@config_group.command()
@click.argument('config_file', type=click.Path(exists=True), required=False)
@click.option('--config-dir', '-d', type=click.Path(exists=True),
              help='Configuration directory to search')  
@click.option('--interval', '-i', type=int, default=1,
              help='Check interval in seconds')
def watch(config_file: Optional[str], config_dir: Optional[str], interval: int):
    """Watch configuration file for changes and validate on change."""
    try:
        config_manager = UnifiedConfigManager(config_dir=config_dir)
        
        if config_file:
            config = config_manager.load_config(config_file)
            console.print(f"Watching configuration file: [cyan]{config_file}[/cyan]")
        else:
            config = config_manager.load_config()
            source = config_manager.get_preferred_config_source()
            if source:
                console.print(f"Watching configuration file: [cyan]{source.file_path}[/cyan]")
            else:
                console.print("[red]No configuration file found to watch[/red]")
                return
        
        def on_config_change(new_config):
            console.print(f"\n[yellow]Configuration changed at {new_config}[/yellow]")
            issues = new_config.validate_config()
            if issues:
                console.print(f"[red]Configuration has {len(issues)} issues:[/red]")
                for issue in issues:
                    console.print(f"  - {issue}")
            else:
                console.print("[green]✓ Configuration is valid[/green]")
        
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
        console.print(f"[red]Error watching configuration: {e}[/red]", err=True)
        sys.exit(1)


@config_group.command()
@click.option('--config-dir', '-d', type=click.Path(exists=True),
              help='Configuration directory to search')
@click.option('--format', '-f', type=click.Choice(['toml', 'yaml', 'json']),
              help='Preferred configuration format')  
def env_vars(config_dir: Optional[str], format: Optional[str]):
    """Show environment variables that can override configuration."""
    try:
        config_manager = UnifiedConfigManager(config_dir=config_dir)
        
        console.print("\n[bold blue]Environment Variable Overrides[/bold blue]")
        console.print(f"Prefix: [cyan]{config_manager.env_prefix}[/cyan]")
        
        # Create a tree structure for better visualization
        tree = Tree("Environment Variables")
        
        # Server level vars
        server_branch = tree.add("[bold green]Server Configuration[/bold green]")
        server_branch.add(f"[cyan]{config_manager.env_prefix}HOST[/cyan] - Server host address")
        server_branch.add(f"[cyan]{config_manager.env_prefix}PORT[/cyan] - Server port number")
        server_branch.add(f"[cyan]{config_manager.env_prefix}DEBUG[/cyan] - Enable debug mode (true/false)")
        
        # Qdrant vars
        qdrant_branch = tree.add("[bold green]Qdrant Configuration[/bold green]")
        qdrant_branch.add(f"[cyan]{config_manager.env_prefix}QDRANT__URL[/cyan] - Qdrant server URL")
        qdrant_branch.add(f"[cyan]{config_manager.env_prefix}QDRANT__API_KEY[/cyan] - Qdrant API key")
        qdrant_branch.add(f"[cyan]{config_manager.env_prefix}QDRANT__TIMEOUT[/cyan] - Connection timeout")
        qdrant_branch.add(f"[cyan]{config_manager.env_prefix}QDRANT__PREFER_GRPC[/cyan] - Use gRPC protocol")
        
        # Embedding vars
        embed_branch = tree.add("[bold green]Embedding Configuration[/bold green]")
        embed_branch.add(f"[cyan]{config_manager.env_prefix}EMBEDDING__MODEL[/cyan] - Embedding model name")
        embed_branch.add(f"[cyan]{config_manager.env_prefix}EMBEDDING__ENABLE_SPARSE_VECTORS[/cyan] - Enable sparse vectors")
        embed_branch.add(f"[cyan]{config_manager.env_prefix}EMBEDDING__CHUNK_SIZE[/cyan] - Text chunk size")
        embed_branch.add(f"[cyan]{config_manager.env_prefix}EMBEDDING__CHUNK_OVERLAP[/cyan] - Chunk overlap size")
        embed_branch.add(f"[cyan]{config_manager.env_prefix}EMBEDDING__BATCH_SIZE[/cyan] - Processing batch size")
        
        # Workspace vars
        workspace_branch = tree.add("[bold green]Workspace Configuration[/bold green]")
        workspace_branch.add(f"[cyan]{config_manager.env_prefix}WORKSPACE__COLLECTION_SUFFIXES[/cyan] - Collection suffixes (comma-separated)")
        workspace_branch.add(f"[cyan]{config_manager.env_prefix}WORKSPACE__GLOBAL_COLLECTIONS[/cyan] - Global collections (comma-separated)")
        workspace_branch.add(f"[cyan]{config_manager.env_prefix}WORKSPACE__GITHUB_USER[/cyan] - GitHub username")
        workspace_branch.add(f"[cyan]{config_manager.env_prefix}WORKSPACE__COLLECTION_PREFIX[/cyan] - Collection name prefix")
        workspace_branch.add(f"[cyan]{config_manager.env_prefix}WORKSPACE__MAX_COLLECTIONS[/cyan] - Maximum collections limit")
        workspace_branch.add(f"[cyan]{config_manager.env_prefix}WORKSPACE__AUTO_CREATE_COLLECTIONS[/cyan] - Auto-create collections")
        
        # Auto-ingestion vars
        auto_branch = tree.add("[bold green]Auto-Ingestion Configuration[/bold green]")
        auto_branch.add(f"[cyan]{config_manager.env_prefix}AUTO_INGESTION__ENABLED[/cyan] - Enable auto-ingestion")
        auto_branch.add(f"[cyan]{config_manager.env_prefix}AUTO_INGESTION__AUTO_CREATE_WATCHES[/cyan] - Auto-create file watches")
        auto_branch.add(f"[cyan]{config_manager.env_prefix}AUTO_INGESTION__TARGET_COLLECTION_SUFFIX[/cyan] - Target collection suffix")
        
        # gRPC vars
        grpc_branch = tree.add("[bold green]gRPC Configuration[/bold green]")
        grpc_branch.add(f"[cyan]{config_manager.env_prefix}GRPC__ENABLED[/cyan] - Enable gRPC")
        grpc_branch.add(f"[cyan]{config_manager.env_prefix}GRPC__HOST[/cyan] - gRPC server host")
        grpc_branch.add(f"[cyan]{config_manager.env_prefix}GRPC__PORT[/cyan] - gRPC server port")
        
        console.print(tree)
        
        # Show example
        console.print("\n[bold yellow]Example Usage:[/bold yellow]")
        console.print(f"export {config_manager.env_prefix}QDRANT__URL=http://localhost:6333")
        console.print(f"export {config_manager.env_prefix}EMBEDDING__MODEL=sentence-transformers/all-MiniLM-L6-v2")
        console.print(f"export {config_manager.env_prefix}DEBUG=true")
        
    except Exception as e:
        console.print(f"[red]Error displaying environment variables: {e}[/red]", err=True)
        sys.exit(1)


if __name__ == '__main__':
    config_group()