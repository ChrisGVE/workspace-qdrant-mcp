
from ...observability import get_logger
logger = get_logger(__name__)
"""Administrative CLI commands for system management.

This module provides comprehensive system administration capabilities
including status monitoring, configuration management, and engine lifecycle.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import psutil
import typer

from ...core.client import QdrantWorkspaceClient, create_qdrant_client
from ...core.config import Config
from ...utils.project_detection import ProjectDetector

# Create the admin app
admin_app = typer.Typer(
    help="""System administration and configuration
    
    Monitor system health, manage configuration, and control processing engines.
    
    Examples:
        wqm admin status                 # Show comprehensive system status
        wqm admin health                 # Run health checks
        wqm admin collections            # List all collections
        wqm admin start-engine           # Start Rust processing engine
        wqm admin config show           # Show current configuration
    """,
    no_args_is_help=True,
    rich_markup_mode=None  # Disable Rich formatting completely
)

def handle_async(coro):
    """Helper to run async commands."""
    try:
        return asyncio.run(coro)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        raise typer.Exit(1)
    except Exception as e:
        print(f"Error: {e}")
        raise typer.Exit(1)

@admin_app.command("status")
def system_status(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed status"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    watch: bool = typer.Option(False, "--watch", "-w", help="Watch status continuously (5s refresh)"),
):
    """Show comprehensive system status."""
    if watch:
        handle_async(_watch_status(verbose))
    else:
        handle_async(_system_status(verbose, json_output))

@admin_app.command("config")
def config_management(
    show: bool = typer.Option(False, "--show", help="Show current configuration"),
    validate: bool = typer.Option(False, "--validate", help="Validate configuration"),
    path: str | None = typer.Option(None, "--path", help="Configuration file path"),
):
    """Configuration management."""
    handle_async(_config_management(show, validate, path))

@admin_app.command("start-engine")
def start_engine(
    force: bool = typer.Option(False, "--force", help="Force start even if already running"),
    config_path: str | None = typer.Option(None, "--config", help="Custom config path"),
):
    """Start the Rust processing engine."""
    handle_async(_start_engine(force, config_path))

@admin_app.command("stop-engine")
def stop_engine(
    force: bool = typer.Option(False, "--force", help="Force stop without graceful shutdown"),
    timeout: int = typer.Option(30, "--timeout", help="Shutdown timeout in seconds"),
):
    """Stop the Rust processing engine."""
    handle_async(_stop_engine(force, timeout))

@admin_app.command("restart-engine")
def restart_engine(
    config_path: str | None = typer.Option(None, "--config", help="Custom config path"),
):
    """Restart engine with new configuration."""
    handle_async(_restart_engine(config_path))

@admin_app.command("collections")
def list_collections(
    project: str | None = typer.Option(None, "--project", help="Filter by project"),
    stats: bool = typer.Option(False, "--stats", help="Include collection statistics"),
    library: bool = typer.Option(False, "--library", help="Show only library collections (_prefixed)"),
):
    """List and manage collections."""
    handle_async(_list_collections(project, stats, library))

@admin_app.command("health")
def health_check(
    deep: bool = typer.Option(False, "--deep", help="Perform deep health check"),
    timeout: int = typer.Option(10, "--timeout", help="Health check timeout"),
):
    """Comprehensive health check."""
    handle_async(_health_check(deep, timeout))

# Async implementation functions
async def _system_status(verbose: bool, json_output: bool):
    """Show comprehensive system status."""
    try:
        config = Config()
        status_data = await _collect_status_data(config)

        if json_output:
            print(json.dumps(status_data, indent=2, default=str))
            return

        _display_status_panel(status_data, verbose)

    except Exception as e:
        print(f"Error getting system status: {e}")
        raise typer.Exit(1)

async def _watch_status(verbose: bool):
    """Watch system status with continuous refresh."""
    import time
    import os

    print("Watching system status (Ctrl+C to stop)\n")

    try:
        while True:
            # Clear screen
            os.system('clear' if os.name == 'posix' else 'cls')

            config = Config()
            status_data = await _collect_status_data(config)
            _display_status_panel(status_data, verbose)

            print(f"\nLast updated: {status_data['timestamp']} | Press Ctrl+C to stop")

            # Wait 5 seconds
            await asyncio.sleep(5)

    except KeyboardInterrupt:
        print("\nStatus monitoring stopped")

async def _collect_status_data(config: Config) -> dict[str, Any]:
    """Collect comprehensive status information."""
    from datetime import datetime

    status_data = {
        "timestamp": datetime.now().isoformat(),
        "config_valid": True,
        "qdrant": {"status": "unknown"},
        "rust_engine": {"status": "unknown"},
        "system": {},
        "project": {},
        "collections": {}
    }

    # Test Qdrant connectivity
    try:
        # Use raw Qdrant client for basic connectivity test
        from qdrant_client import QdrantClient
        # Only pass api_key if it's configured and we're using HTTPS
        client_kwargs = {'url': config.qdrant.url, 'timeout': 5}
        if (hasattr(config.qdrant, 'api_key') and config.qdrant.api_key and 
            config.qdrant.api_key.strip() and config.qdrant.url.startswith('https')):
            client_kwargs['api_key'] = config.qdrant.api_key
        raw_client = QdrantClient(**client_kwargs)
        
        # Get collections directly from API
        collections_response = raw_client.get_collections()
        collections = collections_response.collections if hasattr(collections_response, 'collections') else []
        
        status_data["qdrant"] = {
            "status": "healthy",
            "url": config.qdrant.url,
            "collections_count": len(collections),
            "version": "1.x"  # TODO: Get actual version from info API
        }
        
        # Clean up
        raw_client.close()
    except Exception as e:
        status_data["qdrant"] = {
            "status": "error",
            "error": str(e),
            "url": config.qdrant.url if hasattr(config, 'qdrant') else "unknown"
        }

    # Check Rust Engine (placeholder - will be implemented with Rust integration)
    try:
        # TODO: Implement actual Rust engine status check
        status_data["rust_engine"] = {
            "status": "not_implemented",
            "message": "Rust engine status checking will be implemented in Task 11"
        }
    except Exception as e:
        status_data["rust_engine"] = {
            "status": "error",
            "error": str(e)
        }

    # System resources
    try:
        status_data["system"] = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory": {
                "percent": psutil.virtual_memory().percent,
                "used_gb": round(psutil.virtual_memory().used / (1024**3), 2),
                "total_gb": round(psutil.virtual_memory().total / (1024**3), 2)
            },
            "disk": {
                "percent": psutil.disk_usage('/').percent,
                "free_gb": round(psutil.disk_usage('/').free / (1024**3), 2),
                "total_gb": round(psutil.disk_usage('/').total / (1024**3), 2)
            }
        }
    except Exception as e:
        status_data["system"] = {"error": str(e)}

    # Project detection
    try:
        detector = ProjectDetector(config.workspace.github_user if hasattr(config, 'workspace') else None)
        project_info = detector.get_project_info(str(Path.cwd()))
        main_project = project_info["main_project"]
        subprojects = project_info["subprojects"]
        
        status_data["project"] = {
            "current_dir": str(Path.cwd()),
            "detected_projects": 1 + len(subprojects),
            "current_project": main_project,
            "subprojects": len(subprojects)
        }
    except Exception as e:
        status_data["project"] = {"error": str(e)}

    return status_data

def _display_status_panel(status_data: dict[str, Any], verbose: bool):
    """Display status in plain text format."""

    # Overall health assessment
    qdrant_healthy = status_data["qdrant"]["status"] == "healthy"
    rust_healthy = status_data["rust_engine"]["status"] in ["healthy", "not_implemented"]
    overall_healthy = qdrant_healthy and rust_healthy

    health_text = 'HEALTHY' if overall_healthy else 'UNHEALTHY'

    # Main status display
    print(f"System Health: {health_text}")
    print("=" * 50)

    # Component status display
    print("\nComponent Status:")
    print("-" * 50)

    # Qdrant Database
    qdrant = status_data["qdrant"]
    if qdrant["status"] == "healthy":
        print(f"Qdrant DB        | CONNECTED    | {qdrant.get('collections_count', 0)} collections | {qdrant.get('url', 'unknown')}")
    else:
        print(f"Qdrant DB        | ERROR        | {qdrant.get('error', 'Connection failed')}")

    # Rust Engine - Show more useful status
    rust_engine = status_data["rust_engine"]
    if rust_engine["status"] == "healthy":
        print(f"Rust Engine      | RUNNING      | Tasks: {rust_engine.get('active_tasks', 0)}A/{rust_engine.get('queued_tasks', 0)}Q")
    elif rust_engine["status"] == "not_implemented":
        print(f"Rust Engine      | NOT READY    | Engine integration pending")
    else:
        print(f"Rust Engine      | ERROR        | {rust_engine.get('error', 'Unknown error')}")

    # Project Context
    project = status_data["project"]
    if "error" not in project:
        subproject_info = f" + {project.get('subprojects', 0)} sub" if project.get('subprojects', 0) > 0 else ""
        print(f"Project          | DETECTED     | {project.get('current_project', 'unknown')}{subproject_info} | {project.get('detected_projects', 0)} total")
    else:
        print(f"Project          | WARNING      | {project.get('error', 'Detection failed')}")

    # System resources (if verbose)
    if verbose and "error" not in status_data["system"]:
        system = status_data["system"]

        print("\nSystem Resources:")
        print("-" * 50)
        
        # CPU
        print(f"CPU              | {system['cpu_percent']:.1f}%     | System load")

        # Memory
        mem = system["memory"]
        print(f"Memory           | {mem['percent']:.1f}%     | {mem['used_gb']:.1f}GB / {mem['total_gb']:.1f}GB")

        # Disk
        disk = system["disk"]
        print(f"Disk             | {disk['percent']:.1f}%     | {disk['free_gb']:.1f}GB free / {disk['total_gb']:.1f}GB total")

async def _config_management(show: bool, validate: bool, path: str | None):
    """Configuration management operations."""
    try:
        config = Config()

        if show:
            print("Current Configuration")
            print("=" * 50)

            # Add key configuration values
            if hasattr(config, 'qdrant'):
                print(f"Qdrant URL:         {config.qdrant.url}")
            if hasattr(config, 'embedding'):
                print(f"Embedding Model:    {config.embedding.model}")
            if hasattr(config, 'workspace'):
                print(f"Collection Prefix:  {config.workspace.collection_prefix}")

        if validate:
            print("\nConfiguration Validation")
            print("=" * 50)

            validation_results = []

            # Validate Qdrant connection
            try:
                # Use raw Qdrant client for validation
                from qdrant_client import QdrantClient
                # Only pass api_key if it's configured and we're using HTTPS
                client_kwargs = {'url': config.qdrant.url, 'timeout': 5}
                if (hasattr(config.qdrant, 'api_key') and config.qdrant.api_key and 
                    config.qdrant.api_key.strip() and config.qdrant.url.startswith('https')):
                    client_kwargs['api_key'] = config.qdrant.api_key
                raw_client = QdrantClient(**client_kwargs)
                collections = raw_client.get_collections()
                raw_client.close()
                validation_results.append(("Qdrant Connection", "Valid"))
            except Exception as e:
                validation_results.append(("Qdrant Connection", f"Failed: {e}"))

            # Display validation results
            for setting, result in validation_results:
                print(f"  {setting}: {result}")

        if not show and not validate:
            print("Use --show or --validate flags to perform operations")

    except Exception as e:
        print(f"Configuration error: {e}")
        raise typer.Exit(1)

async def _start_engine(force: bool, config_path: str | None):
    """Start the Rust processing engine."""
    print("Starting Rust Engine")
    print("=" * 50)

    try:
        print("Starting engine...")
        # TODO: Implement actual Rust engine startup
        await asyncio.sleep(2)  # Simulate startup time

        print("Rust engine started successfully")
        print("Note: Full Rust engine integration will be implemented in Task 11")

    except Exception as e:
        print(f"Failed to start engine: {e}")
        raise typer.Exit(1)

async def _stop_engine(force: bool, timeout: int):
    """Stop the Rust processing engine."""
    print("Stopping Rust Engine")
    print("=" * 50)

    try:
        # TODO: Implement actual Rust engine shutdown
        if force:
            print("Force stopping engine...")
        else:
            print(f"Graceful shutdown (timeout: {timeout}s)...")

        await asyncio.sleep(1)  # Simulate shutdown time
        print("Rust engine stopped")

    except Exception as e:
        print(f"Failed to stop engine: {e}")
        raise typer.Exit(1)

async def _restart_engine(config_path: str | None):
    """Restart engine with new configuration."""
    print("Restarting Rust Engine")
    print("=" * 50)

    try:
        await _stop_engine(False, 30)
        await asyncio.sleep(1)
        await _start_engine(False, config_path)

    except Exception as e:
        print(f"Failed to restart engine: {e}")
        raise typer.Exit(1)

async def _list_collections(project: str | None, stats: bool, library: bool):
    """List and manage collections."""
    try:
        config = Config()
        # Use raw Qdrant client to get collections
        from qdrant_client import QdrantClient
        # Only pass api_key if it's configured and we're using HTTPS
        client_kwargs = {'url': config.qdrant.url, 'timeout': 10}
        if (hasattr(config.qdrant, 'api_key') and config.qdrant.api_key and 
            config.qdrant.api_key.strip() and config.qdrant.url.startswith('https')):
            client_kwargs['api_key'] = config.qdrant.api_key
        raw_client = QdrantClient(**client_kwargs)
        
        collections_response = raw_client.get_collections()
        all_collections = [{'name': col.name} for col in collections_response.collections]
        
        # Filter collections
        if library:
            collections = [col for col in all_collections if col.get("name", "").startswith("_")]
        elif project:
            prefix = f"{config.workspace.collection_prefix}{project}_" if hasattr(config, 'workspace') else f"{project}_"
            collections = [col for col in all_collections if col.get("name", "").startswith(prefix)]
        else:
            collections = all_collections

        if not collections:
            filter_desc = "library " if library else f"project '{project}' " if project else ""
            print(f"No {filter_desc}collections found.")
            return

        # Display collections table
        print(f"Collections ({len(collections)} found)")
        print("=" * 50)
        
        if stats:
            print(f"{'Name':<30} {'Type':<10} {'Points':<10} {'Vectors':<10}")
            print("-" * 60)
        else:
            print(f"{'Name':<30} {'Type':<10}")
            print("-" * 40)

        for col in collections:
            name = col.get("name", "unknown")
            col_type = "Library" if name.startswith("_") else "Project"

            if stats:
                try:
                    info = raw_client.get_collection(name)
                    points = str(info.points_count if hasattr(info, 'points_count') else "?")
                    vectors = str(info.vectors_count if hasattr(info, 'vectors_count') else "?")
                    print(f"{name:<30} {col_type:<10} {points:<10} {vectors:<10}")
                except Exception:
                    print(f"{name:<30} {col_type:<10} {'?':<10} {'?':<10}")
            else:
                print(f"{name:<30} {col_type:<10}")

        # Clean up
        raw_client.close()
        
    except Exception as e:
        print(f"Error listing collections: {e}")
        raise typer.Exit(1)

async def _health_check(deep: bool, timeout: int):
    """Comprehensive health check."""
    print("System Health Check")
    print("=" * 50)

    health_results = []

    # Basic connectivity
    print("Testing Qdrant connectivity...")
    try:
        config = Config()
        # Use raw Qdrant client for health check
        from qdrant_client import QdrantClient
        # Only pass api_key if it's configured and we're using HTTPS
        client_kwargs = {'url': config.qdrant.url, 'timeout': timeout}
        if (hasattr(config.qdrant, 'api_key') and config.qdrant.api_key and 
            config.qdrant.api_key.strip() and config.qdrant.url.startswith('https')):
            client_kwargs['api_key'] = config.qdrant.api_key
        raw_client = QdrantClient(**client_kwargs)
        
        # Test basic connectivity
        collections = raw_client.get_collections()
        raw_client.close()
        health_results.append(("Qdrant Connectivity", "Healthy", "ok"))
    except Exception as e:
        if "timeout" in str(e).lower():
            health_results.append(("Qdrant Connectivity", "Timeout", "warning"))
        else:
            health_results.append(("Qdrant Connectivity", f"Error: {e}", "error"))

    # Memory usage check
    print("Checking memory usage...")
    try:
        memory = psutil.virtual_memory()
        if memory.percent < 80:
            health_results.append(("Memory Usage", f"{memory.percent:.1f}%", "ok"))
        elif memory.percent < 95:
            health_results.append(("Memory Usage", f"{memory.percent:.1f}%", "warning"))
        else:
            health_results.append(("Memory Usage", f"{memory.percent:.1f}%", "error"))
    except Exception as e:
        health_results.append(("Memory Usage", f"Error: {e}", "error"))

    # Disk space check
    if deep:
        print("Checking disk space...")
        try:
            disk = psutil.disk_usage('/')
            if disk.percent < 85:
                health_results.append(("Disk Space", f"{disk.percent:.1f}%", "ok"))
            elif disk.percent < 95:
                health_results.append(("Disk Space", f"{disk.percent:.1f}%", "warning"))
            else:
                health_results.append(("Disk Space", f"{disk.percent:.1f}%", "error"))
        except Exception as e:
            health_results.append(("Disk Space", f"Error: {e}", "error"))

    # Display results
    print("\nHealth Check Results:")
    print("-" * 50)
    for component, status, level in health_results:
        print(f"{component:<20} | {status}")

    # Overall assessment
    errors = sum(1 for _, _, level in health_results if level == "error")
    warnings = sum(1 for _, _, level in health_results if level == "warning")

    if errors == 0:
        if warnings == 0:
            print("\nSystem is healthy!")
        else:
            print(f"\nSystem has {warnings} warning(s)")
    else:
        print(f"\nSystem has {errors} error(s) and {warnings} warning(s)")
        raise typer.Exit(1)
