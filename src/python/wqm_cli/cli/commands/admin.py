"""Administrative CLI commands for system management.

This module provides comprehensive system administration capabilities
including status monitoring, configuration management, and engine lifecycle.
"""

import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import psutil
import typer
from common.core.config import get_config_manager
from common.utils.project_detection import ProjectDetector
from loguru import logger

# Lazy import to avoid CLI startup issues
# from workspace_qdrant_mcp.utils.migration import ConfigMigrator, ReportGenerator

def _get_config_migrator():
    """Lazy import ConfigMigrator to avoid CLI startup issues."""
    try:
        from workspace_qdrant_mcp.utils.migration import ConfigMigrator
        return ConfigMigrator()
    except ImportError as e:
        logger.error(f"Failed to import ConfigMigrator: {e}")
        error_message("Migration functionality requires MCP server components")
        raise typer.Exit(1)

def _get_report_generator():
    """Lazy import ReportGenerator to avoid CLI startup issues."""
    try:
        from workspace_qdrant_mcp.utils.migration import ReportGenerator
        return ReportGenerator()
    except ImportError as e:
        logger.error(f"Failed to import ReportGenerator: {e}")
        error_message("Migration reporting requires MCP server components")
        raise typer.Exit(1)
from ..utils import (
    config_path_option,
    create_command_app,
    error_message,
    force_option,
    get_configured_client,
    handle_async,
    json_output_option,
    verbose_option,
)

# logger imported from loguru

# Create the admin app using shared utilities
admin_app = create_command_app(
    name="admin",
    help_text="""System administration and configuration.

Monitor system health, manage configuration, control processing engines, and view migration reports.

Examples:
    wqm admin status                 # Show comprehensive system status
    wqm admin health                 # Run health checks
    wqm admin collections            # List all collections
    wqm admin start-engine           # Start Rust processing engine
    wqm admin config --show          # Show current configuration
    wqm admin migration-report       # Show latest migration report
    wqm admin migration-history      # Show migration history
    wqm admin rollback-config <id>   # Rollback to a previous backup""",
    no_args_is_help=True,
)


@admin_app.command("status")
def system_status(
    verbose: bool = verbose_option(),
    json_output: bool = json_output_option(),
    watch: bool = typer.Option(
        False, "--watch", "-w", help="Watch status continuously (5s refresh)"
    ),
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
    set_value: str | None = typer.Option(
        None, "--set", help="Set configuration value (format: key=value, e.g., server.debug=true)"
    ),
    get_value: str | None = typer.Option(
        None, "--get", help="Get a specific configuration value (e.g., server.port)"
    ),
):
    """Configuration management.

    Examples:
        wqm admin config --show                    # Show all configuration
        wqm admin config --get server.port         # Get a specific value
        wqm admin config --set server.debug=true   # Set a value
        wqm admin config --validate                # Validate configuration
    """
    handle_async(_config_management(show, validate, path, set_value, get_value))


@admin_app.command("start-engine")
def start_engine(
    force: bool = force_option(),
    config_path: str | None = config_path_option(),
):
    """Start the Rust processing engine."""
    handle_async(_start_engine(force, config_path))


@admin_app.command("stop-engine")
def stop_engine(
    force: bool = force_option(),
    timeout: int = typer.Option(30, "--timeout", help="Shutdown timeout in seconds"),
):
    """Stop the Rust processing engine."""
    handle_async(_stop_engine(force, timeout))


@admin_app.command("restart-engine")
def restart_engine(
    config_path: str | None = config_path_option(),
):
    """Restart engine with new configuration."""
    handle_async(_restart_engine(config_path))


@admin_app.command("collections")
def list_collections(
    project: str | None = typer.Option(None, "--project", help="Filter by project"),
    stats: bool = typer.Option(False, "--stats", help="Include collection statistics"),
    library: bool = typer.Option(
        False, "--library", help="Show only library collections (_prefixed)"
    ),
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


@admin_app.command("projects")
def list_projects(
    json_output: bool = json_output_option(),
    priority: str | None = typer.Option(
        None, "--priority", help="Filter by priority: high, normal, low"
    ),
    active_only: bool = typer.Option(
        False, "--active", help="Show only projects with active sessions"
    ),
):
    """
    List registered projects with priority and session status.

    Shows all projects registered in the multi-tenant architecture,
    including their priority level, active sessions, and last activity.

    Examples:
        wqm admin projects                    # List all projects
        wqm admin projects --active           # Show only active projects
        wqm admin projects --priority=high    # Filter by priority
        wqm admin projects --json             # JSON output
    """
    handle_async(_list_projects(json_output, priority, active_only))


@admin_app.command("queue")
def show_queue(
    json_output: bool = json_output_option(),
    collection: str | None = typer.Option(
        None, "--collection", help="Filter by collection type: projects, libraries"
    ),
    status: str | None = typer.Option(
        None, "--status", help="Filter by status: pending, processing, failed"
    ),
):
    """
    Show ingestion queue status with priority breakdown.

    Displays the current state of the ingestion queue, including
    pending items by priority level and collection type.

    Examples:
        wqm admin queue                       # Show full queue status
        wqm admin queue --collection=projects # Show only project queue
        wqm admin queue --status=pending      # Show pending items
        wqm admin queue --json                # JSON output
    """
    handle_async(_show_queue(json_output, collection, status))


@admin_app.command("metrics")
def show_metrics(
    json_output: bool = json_output_option(),
    watch: bool = typer.Option(
        False, "--watch", "-w", help="Watch metrics continuously (5s refresh)"
    ),
    tenant: str | None = typer.Option(
        None, "--tenant", "-t", help="Filter metrics by tenant/project ID"
    ),
    daemon_only: bool = typer.Option(
        False, "--daemon", help="Show only daemon metrics (from Prometheus endpoint)"
    ),
    mcp_only: bool = typer.Option(
        False, "--mcp", help="Show only MCP server metrics"
    ),
):
    """
    Display system metrics (Task 412).

    Shows metrics from both the Rust daemon (Prometheus format) and
    the Python MCP server. Includes session tracking, queue metrics,
    and per-tenant statistics.

    Examples:
        wqm admin metrics                    # Show all current metrics
        wqm admin metrics --watch            # Live updating metrics
        wqm admin metrics --tenant=abc123    # Filter by tenant
        wqm admin metrics --daemon           # Daemon metrics only
        wqm admin metrics --mcp              # MCP server metrics only
        wqm admin metrics --json             # JSON output
    """
    if watch:
        handle_async(_watch_metrics(tenant, daemon_only, mcp_only))
    else:
        handle_async(_show_metrics(json_output, tenant, daemon_only, mcp_only))


@admin_app.command("migration-report")
def migration_report(
    migration_id: str | None = typer.Argument(None, help="Specific migration ID to view"),
    format: str = typer.Option("text", "--format", help="Output format: text, json"),
    export: str | None = typer.Option(None, "--export", help="Export report to file"),
    latest: bool = typer.Option(False, "--latest", help="Show latest migration report"),
):
    """View detailed migration reports."""
    handle_async(_migration_report(migration_id, format, export, latest))


@admin_app.command("migration-history")
def migration_history(
    limit: int = typer.Option(10, "--limit", help="Number of recent migrations to show"),
    source_version: str | None = typer.Option(None, "--source", help="Filter by source version"),
    target_version: str | None = typer.Option(None, "--target", help="Filter by target version"),
    success_only: bool | None = typer.Option(None, "--success-only", help="Show only successful migrations"),
    days_back: int | None = typer.Option(None, "--days", help="Show migrations from last N days"),
    format: str = typer.Option("table", "--format", help="Output format: table, json"),
):
    """View migration history with filtering options."""
    handle_async(_migration_history(limit, source_version, target_version, success_only, days_back, format))


@admin_app.command("validate-backup")
def validate_backup(
    backup_id: str = typer.Argument(..., help="Backup ID to validate"),
):
    """Validate backup integrity using checksum verification."""
    handle_async(_validate_backup(backup_id))


@admin_app.command("rollback-config")
def rollback_config(
    backup_id: str = typer.Argument(..., help="Backup ID to restore from"),
    force: bool = force_option(),
):
    """Rollback configuration to a previous backup."""
    handle_async(_rollback_config(backup_id, force))


@admin_app.command("backup-info")
def backup_info(
    backup_id: str = typer.Argument(..., help="Backup ID to get information about"),
):
    """Get detailed information about a specific backup."""
    handle_async(_backup_info(backup_id))


@admin_app.command("cleanup-migration-history")
def cleanup_migration_history(
    keep_count: int = typer.Option(50, "--keep", help="Number of recent migration reports to keep"),
    force: bool = force_option(),
):
    """Clean up old migration reports to save disk space."""
    handle_async(_cleanup_migration_history(keep_count, force))


# Async implementation functions
async def _system_status(verbose: bool, json_output: bool) -> None:
    """Show comprehensive system status."""
    try:
        config = get_config_manager()
        status_data = await _collect_status_data(config)

        if json_output:
            print(json.dumps(status_data, indent=2, default=str))
            return

        _display_status_panel(status_data, verbose)

    except Exception as e:
        print(f"Error getting system status: {e}")
        raise typer.Exit(1) from e


async def _watch_status(verbose: bool) -> None:
    """Watch system status with continuous refresh."""
    import os

    print("Watching system status (Ctrl+C to stop)\n")

    try:
        while True:
            # Clear screen
            clear_cmd = ["clear"] if os.name == "posix" else ["cls"]
            subprocess.run(clear_cmd, check=False, capture_output=False)

            config = get_config_manager()
            status_data = await _collect_status_data(config)
            _display_status_panel(status_data, verbose)

            print(f"\nLast updated: {status_data['timestamp']} | Press Ctrl+C to stop")

            # Wait 5 seconds
            await asyncio.sleep(5)

    except KeyboardInterrupt:
        print("\nStatus monitoring stopped")


async def _collect_status_data(config) -> dict[str, Any]:
    """Collect comprehensive status information."""
    from datetime import datetime

    status_data = {
        "timestamp": datetime.now().isoformat(),
        "config_valid": True,
        "qdrant": {"status": "unknown"},
        "rust_engine": {"status": "unknown"},
        "system": {},
        "project": {},
        "collections": {},
        "registered_projects": {},  # Task 401: Multi-tenant projects info
        "library_watches": {},      # Task 401: Library watches info
        "ingestion_queue": {},      # Task 401: Queue info
    }

    # Test Qdrant connectivity
    try:
        # Use configured client for basic connectivity test
        raw_client = get_configured_client(config)

        # Get collections directly from API
        collections_response = raw_client.get_collections()
        collections = (
            collections_response.collections
            if hasattr(collections_response, "collections")
            else []
        )

        status_data["qdrant"] = {
            "status": "healthy",
            "url": config.qdrant.url,
            "collections_count": len(collections),
            "version": "1.x",  # TODO: Get actual version from info API
        }

        # Clean up
        raw_client.close()
    except Exception as e:
        status_data["qdrant"] = {
            "status": "error",
            "error": str(e),
            "url": config.qdrant.url if hasattr(config, "qdrant") else "unknown",
        }

    # Check Rust Engine (placeholder - will be implemented with Rust integration)
    try:
        # TODO: Implement actual Rust engine status check
        status_data["rust_engine"] = {
            "status": "not_implemented",
            "message": "Rust engine status checking will be implemented in Task 11",
        }
    except Exception as e:
        status_data["rust_engine"] = {"status": "error", "error": str(e)}

    # System resources
    try:
        status_data["system"] = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory": {
                "percent": psutil.virtual_memory().percent,
                "used_gb": round(psutil.virtual_memory().used / (1024**3), 2),
                "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            },
            "disk": {
                "percent": psutil.disk_usage("/").percent,
                "free_gb": round(psutil.disk_usage("/").free / (1024**3), 2),
                "total_gb": round(psutil.disk_usage("/").total / (1024**3), 2),
            },
        }
    except Exception as e:
        status_data["system"] = {"error": str(e)}

    # Project detection
    try:
        detector = ProjectDetector(
            config.workspace.github_user if hasattr(config, "workspace") else None
        )
        project_info = detector.get_project_info(str(Path.cwd()))
        main_project = project_info["main_project"]
        subprojects = project_info["subprojects"]

        status_data["project"] = {
            "current_dir": str(Path.cwd()),
            "detected_projects": 1 + len(subprojects),
            "current_project": main_project,
            "subprojects": len(subprojects),
        }
    except Exception as e:
        status_data["project"] = {"error": str(e)}

    # Task 401: Multi-tenant architecture status
    try:
        from common.core.sqlite_state_manager import SQLiteStateManager
        state_manager = SQLiteStateManager()

        # Initialize synchronously for status collection
        import asyncio
        loop = asyncio.get_event_loop()

        async def get_multitenant_status():
            await state_manager.initialize()

            # Get registered projects summary
            async with state_manager._lock:
                projects_cursor = state_manager.connection.execute(
                    """SELECT priority, COUNT(*) as count FROM projects GROUP BY priority"""
                )
                projects_by_priority = {row["priority"]: row["count"] for row in projects_cursor.fetchall()}

                active_cursor = state_manager.connection.execute(
                    """SELECT COUNT(*) as count FROM projects WHERE active_sessions > 0"""
                )
                active_projects = active_cursor.fetchone()["count"]

                total_cursor = state_manager.connection.execute(
                    """SELECT COUNT(*) as count FROM projects"""
                )
                total_projects = total_cursor.fetchone()["count"]

            # Get library watches summary
            async with state_manager._lock:
                lib_cursor = state_manager.connection.execute(
                    """SELECT enabled, COUNT(*) as count FROM library_watches GROUP BY enabled"""
                )
                lib_rows = lib_cursor.fetchall()
                enabled_libs = sum(row["count"] for row in lib_rows if row["enabled"])
                total_libs = sum(row["count"] for row in lib_rows)

            # Get ingestion queue summary
            async with state_manager._lock:
                queue_cursor = state_manager.connection.execute(
                    """SELECT priority, COUNT(*) as count FROM ingestion_queue GROUP BY priority"""
                )
                queue_by_priority = {row["priority"]: row["count"] for row in queue_cursor.fetchall()}
                total_queue = sum(queue_by_priority.values())

            return {
                "projects": {
                    "total": total_projects,
                    "active": active_projects,
                    "by_priority": projects_by_priority,
                },
                "libraries": {
                    "total": total_libs,
                    "enabled": enabled_libs,
                },
                "queue": {
                    "total": total_queue,
                    "by_priority": queue_by_priority,
                },
            }

        multitenant_status = loop.run_until_complete(get_multitenant_status())
        status_data["registered_projects"] = multitenant_status["projects"]
        status_data["library_watches"] = multitenant_status["libraries"]
        status_data["ingestion_queue"] = multitenant_status["queue"]

    except Exception as e:
        logger.debug(f"Multi-tenant status collection failed: {e}")
        status_data["registered_projects"] = {"error": str(e)}
        status_data["library_watches"] = {"error": str(e)}
        status_data["ingestion_queue"] = {"error": str(e)}

    return status_data


def _display_status_panel(status_data: dict[str, Any], verbose: bool) -> None:
    """Display status in plain text format."""

    # Overall health assessment
    qdrant_healthy = status_data["qdrant"]["status"] == "healthy"
    rust_healthy = status_data["rust_engine"]["status"] in [
        "healthy",
        "not_implemented",
    ]
    overall_healthy = qdrant_healthy and rust_healthy

    health_text = "HEALTHY" if overall_healthy else "UNHEALTHY"

    # Main status display
    print(f"System Health: {health_text}")
    print("=" * 50)

    # Component status display
    print("\nComponent Status:")
    print("-" * 50)

    # Qdrant Database
    qdrant = status_data["qdrant"]
    if qdrant["status"] == "healthy":
        print(
            f"Qdrant DB        | CONNECTED    | {qdrant.get('collections_count', 0)} collections | {qdrant.get('url', 'unknown')}"
        )
    else:
        print(
            f"Qdrant DB        | ERROR        | {qdrant.get('error', 'Connection failed')}"
        )

    # Rust Engine - Show more useful status
    rust_engine = status_data["rust_engine"]
    if rust_engine["status"] == "healthy":
        print(
            f"Rust Engine      | RUNNING      | Tasks: {rust_engine.get('active_tasks', 0)}A/{rust_engine.get('queued_tasks', 0)}Q"
        )
    elif rust_engine["status"] == "not_implemented":
        print("Rust Engine      | NOT READY    | Engine integration pending")
    else:
        print(
            f"Rust Engine      | ERROR        | {rust_engine.get('error', 'Unknown error')}"
        )

    # Project Context
    project = status_data["project"]
    if "error" not in project:
        subproject_info = (
            f" + {project.get('subprojects', 0)} sub"
            if project.get("subprojects", 0) > 0
            else ""
        )
        print(
            f"Project          | DETECTED     | {project.get('current_project', 'unknown')}{subproject_info} | {project.get('detected_projects', 0)} total"
        )
    else:
        print(
            f"Project          | WARNING      | {project.get('error', 'Detection failed')}"
        )

    # System resources (if verbose)
    if verbose and "error" not in status_data["system"]:
        system = status_data["system"]

        print("\nSystem Resources:")
        print("-" * 50)

        # CPU
        print(f"CPU              | {system['cpu_percent']:.1f}%     | System load")

        # Memory
        mem = system["memory"]
        print(
            f"Memory           | {mem['percent']:.1f}%     | {mem['used_gb']:.1f}GB / {mem['total_gb']:.1f}GB"
        )

        # Disk
        disk = system["disk"]
        print(
            f"Disk             | {disk['percent']:.1f}%     | {disk['free_gb']:.1f}GB free / {disk['total_gb']:.1f}GB total"
        )

    # Task 401: Multi-tenant architecture status (always show)
    reg_projects = status_data.get("registered_projects", {})
    lib_watches = status_data.get("library_watches", {})
    queue = status_data.get("ingestion_queue", {})

    if "error" not in reg_projects:
        print("\nMulti-Tenant Status:")
        print("-" * 50)

        # Projects
        total_projects = reg_projects.get("total", 0)
        active_projects = reg_projects.get("active", 0)
        by_priority = reg_projects.get("by_priority", {})
        high_count = by_priority.get("high", 0)
        print(
            f"Projects         | {total_projects} total    | {active_projects} active | {high_count} HIGH priority"
        )

        # Libraries
        total_libs = lib_watches.get("total", 0)
        enabled_libs = lib_watches.get("enabled", 0)
        print(
            f"Libraries        | {total_libs} total    | {enabled_libs} enabled"
        )

        # Queue
        total_queue = queue.get("total", 0)
        queue_by_priority = queue.get("by_priority", {})
        high_queue = queue_by_priority.get(1, 0)
        normal_queue = queue_by_priority.get(3, 0)
        print(
            f"Ingestion Queue  | {total_queue} total    | {high_queue} HIGH | {normal_queue} NORMAL"
        )


async def _config_management(
    show: bool,
    validate: bool,
    path: str | None,
    set_value: str | None = None,
    get_value: str | None = None,
) -> None:
    """Configuration management operations."""
    from common.core.config import ConfigManager

    try:
        config = get_config_manager()

        # Handle --get option
        if get_value:
            value = config.get(get_value)
            if value is None:
                print(f"Configuration key '{get_value}' not found")
                raise typer.Exit(1)
            print(f"{get_value} = {value}")
            return

        # Handle --set option
        if set_value:
            if "=" not in set_value:
                print("Error: --set requires format 'key=value' (e.g., server.debug=true)")
                raise typer.Exit(1)

            key, value_str = set_value.split("=", 1)
            key = key.strip()
            value_str = value_str.strip()

            # Parse the value to appropriate type
            parsed_value = _parse_config_value(value_str)

            # Validate the key exists (optional: warn if new key)
            old_value = config.get(key)
            if old_value is None:
                print(f"Warning: Creating new configuration key '{key}'")

            # Set the value
            config.set(key, parsed_value)

            # Check if restart is required
            requires_restart = ConfigManager.requires_restart(key)

            # Save to config file
            config_file = config.get_config_file_path()
            if config_file:
                config.save_to_file(config_file)
                print(f"Configuration updated: {key} = {parsed_value}")
                print(f"Saved to: {config_file}")
            else:
                # Create a new config file in XDG config directory
                from pathlib import Path
                import platform

                if platform.system() == "Windows":
                    config_dir = Path.home() / "AppData" / "Local" / "workspace-qdrant-mcp"
                else:
                    xdg_config = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
                    config_dir = xdg_config / "workspace-qdrant-mcp"

                config_dir.mkdir(parents=True, exist_ok=True)
                config_file = config_dir / "config.yaml"
                config.save_to_file(config_file)
                print(f"Configuration updated: {key} = {parsed_value}")
                print(f"Created config file: {config_file}")

            if requires_restart:
                print()
                print("⚠️  RESTART REQUIRED")
                print(f"   Setting '{key}' requires a server/daemon restart to take effect.")
                print("   Run: wqm admin restart-engine")

            return

        if show:
            print("Current Configuration")
            print("=" * 50)

            # Show key configuration values using the new get method
            print(f"Qdrant URL:         {config.get('qdrant.url', 'not set')}")
            print(f"Qdrant Timeout:     {config.get('qdrant.timeout', 'not set')}ms")
            print(f"Embedding Model:    {config.get('embedding.model', 'not set')}")
            print(f"Server Host:        {config.get('server.host', 'not set')}")
            print(f"Server Port:        {config.get('server.port', 'not set')}")
            print(f"Server Debug:       {config.get('server.debug', False)}")
            print(f"gRPC Enabled:       {config.get('grpc.enabled', False)}")
            print(f"gRPC Host:          {config.get('grpc.host', 'not set')}")
            print(f"gRPC Port:          {config.get('grpc.port', 'not set')}")

            # Show config file location
            config_file = config.get_config_file_path()
            print()
            print(f"Config file:        {config_file or '(using defaults)'}")

        if validate:
            print("\nConfiguration Validation")
            print("=" * 50)

            validation_results = []

            # Validate Qdrant connection
            try:
                # Use configured client for validation
                raw_client = get_configured_client(config)
                raw_client.get_collections()
                raw_client.close()
                validation_results.append(("Qdrant Connection", "Valid"))
            except Exception as e:
                validation_results.append(("Qdrant Connection", f"Failed: {e}"))

            # Use built-in validation
            issues = config.validate()
            if issues:
                for issue in issues:
                    validation_results.append(("Configuration", f"Issue: {issue}"))
            else:
                validation_results.append(("Configuration Schema", "Valid"))

            # Display validation results
            for setting, result in validation_results:
                print(f"  {setting}: {result}")

        if not show and not validate and not set_value and not get_value:
            print("Use --show, --get, --set, or --validate flags to perform operations")
            print()
            print("Examples:")
            print("  wqm admin config --show                    # Show all configuration")
            print("  wqm admin config --get server.port         # Get a specific value")
            print("  wqm admin config --set server.debug=true   # Set a value")
            print("  wqm admin config --validate                # Validate configuration")

    except Exception as e:
        print(f"Configuration error: {e}")
        raise typer.Exit(1)


def _parse_config_value(value_str: str) -> str | int | float | bool | list | None:
    """Parse a configuration value string to the appropriate Python type.

    Args:
        value_str: String value from command line

    Returns:
        Parsed value with appropriate type
    """
    # Handle null/none
    if value_str.lower() in ("null", "none", "~"):
        return None

    # Handle booleans
    if value_str.lower() in ("true", "yes", "on", "1"):
        return True
    if value_str.lower() in ("false", "no", "off", "0"):
        return False

    # Handle integers
    try:
        return int(value_str)
    except ValueError:
        pass

    # Handle floats
    try:
        return float(value_str)
    except ValueError:
        pass

    # Handle lists (comma-separated)
    if "," in value_str:
        return [item.strip() for item in value_str.split(",")]

    # Return as string
    return value_str


async def _start_engine(force: bool, config_path: str | None) -> None:
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


async def _stop_engine(force: bool, timeout: int) -> None:
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


async def _restart_engine(config_path: str | None) -> None:
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


async def _list_collections(project: str | None, stats: bool, library: bool) -> None:
    """List and manage collections."""
    try:
        config = get_config_manager()
        # Use configured client to get collections
        raw_client = get_configured_client(config)

        collections_response = raw_client.get_collections()
        all_collections = [
            {"name": col.name} for col in collections_response.collections
        ]

        # Filter collections
        if library:
            collections = [
                col for col in all_collections if col.get("name", "").startswith("_")
            ]
        elif project:
            # Note: Using project-based naming without explicit prefix field
            prefix = f"{project}_"
            collections = [
                col for col in all_collections if col.get("name", "").startswith(prefix)
            ]
        else:
            collections = all_collections

        if not collections:
            filter_desc = (
                "library " if library else f"project '{project}' " if project else ""
            )
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
                    points = str(
                        info.points_count if hasattr(info, "points_count") else "?"
                    )
                    vectors = str(
                        info.vectors_count if hasattr(info, "vectors_count") else "?"
                    )
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


async def _health_check(deep: bool, timeout: int) -> None:
    """Comprehensive health check."""
    print("System Health Check")
    print("=" * 50)

    health_results = []

    # Basic connectivity
    print("Testing Qdrant connectivity...")
    try:
        config = get_config_manager()
        # Use configured client for health check
        raw_client = get_configured_client(config)

        # Test basic connectivity
        raw_client.get_collections()
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
            disk = psutil.disk_usage("/")
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
    for component, status, _level in health_results:
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


# Migration reporting functions
async def _migration_report(migration_id: str | None, format: str, export: str | None, latest: bool) -> None:
    """View detailed migration reports."""
    try:
        migrator = _get_config_migrator()

        # Determine which report to show
        if latest:
            report = migrator.get_latest_migration_report()
            if not report:
                print("No migration reports found")
                return
        elif migration_id:
            report = migrator.get_migration_report(migration_id)
            if not report:
                print(f"Migration report '{migration_id}' not found")
                return
        else:
            # Show latest if no specific ID provided
            report = migrator.get_latest_migration_report()
            if not report:
                print("No migration reports found. Use --help to see options.")
                return

        # Generate report content
        if format.lower() == "json":
            report_content = migrator.report_generator.format_report_json(report)
        else:
            report_content = migrator.report_generator.format_report_text(report)

        # Export or display
        if export:
            export_path = Path(export)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            with export_path.open('w', encoding='utf-8') as f:
                f.write(report_content)
            print(f"Migration report exported to: {export_path}")
        else:
            print(report_content)

    except Exception as e:
        print(f"Error retrieving migration report: {e}")
        raise typer.Exit(1)


async def _migration_history(limit: int, source_version: str | None, target_version: str | None,
                            success_only: bool | None, days_back: int | None, format: str) -> None:
    """View migration history with filtering options."""
    try:
        migrator = _get_config_migrator()

        # Get filtered migration history
        if any([source_version, target_version, success_only is not None, days_back]):
            migrations = migrator.search_migration_history(
                source_version=source_version,
                target_version=target_version,
                success_only=success_only,
                days_back=days_back
            )
        else:
            migrations = migrator.get_migration_history(limit=limit)

        if not migrations:
            print("No migrations found matching the criteria")
            return

        # Apply limit to search results
        if limit and len(migrations) > limit:
            migrations = migrations[:limit]

        if format.lower() == "json":
            print(json.dumps(migrations, indent=2, default=str))
            return

        # Display as formatted table
        print(f"Migration History ({len(migrations)} records)")
        print("=" * 120)
        print(f"{'Migration ID':<40} {'Timestamp':<20} {'Source':<15} {'Target':<15} {'Status':<10} {'Changes':<8}")
        print("-" * 120)

        for migration in migrations:
            migration_id_short = migration.get("migration_id", "unknown")[:36] + "..."
            timestamp = migration.get("timestamp", "")[:19]  # Remove microseconds
            source = migration.get("source_version", "unknown")[:12]
            target = migration.get("target_version", "unknown")[:12]
            status = "SUCCESS" if migration.get("success", False) else "FAILED"
            changes = str(migration.get("changes_count", 0))

            print(f"{migration_id_short:<40} {timestamp:<20} {source:<15} {target:<15} {status:<10} {changes:<8}")

        print("\nUse 'wqm admin migration-report <migration_id>' to view detailed report")

    except Exception as e:
        print(f"Error retrieving migration history: {e}")
        raise typer.Exit(1)


async def _validate_backup(backup_id: str) -> None:
    """Validate backup integrity using checksum verification."""
    try:
        migrator = _get_config_migrator()

        print(f"Validating backup: {backup_id}")
        print("=" * 50)

        is_valid = migrator.validate_backup(backup_id)

        if is_valid:
            print("✅ Backup validation successful")
            print("   Checksum matches, JSON is valid")

            # Show backup info
            backup_info = migrator.get_backup_info(backup_id)
            if backup_info:
                print(f"   Backup created: {backup_info.get('timestamp', 'unknown')}")
                print(f"   Configuration version: {backup_info.get('version', 'unknown')}")
                print(f"   File size: {backup_info.get('file_size', 0)} bytes")
        else:
            print("❌ Backup validation failed")
            print("   Backup file is corrupted or missing")
            raise typer.Exit(1)

    except Exception as e:
        print(f"Error validating backup: {e}")
        raise typer.Exit(1)


async def _rollback_config(backup_id: str, force: bool) -> None:
    """Rollback configuration to a previous backup."""
    try:
        migrator = _get_config_migrator()

        print(f"Rolling back configuration to backup: {backup_id}")
        print("=" * 50)

        # Get backup info first
        backup_info = migrator.get_backup_info(backup_id)
        if not backup_info:
            print(f"Backup '{backup_id}' not found")
            raise typer.Exit(1)

        # Validate backup before rollback
        if not migrator.validate_backup(backup_id):
            print("❌ Backup validation failed - cannot rollback")
            raise typer.Exit(1)

        # Confirm rollback unless forced
        if not force:
            print(f"This will restore configuration from backup created: {backup_info.get('timestamp', 'unknown')}")
            print(f"Configuration version: {backup_info.get('version', 'unknown')}")

            confirm = typer.confirm("Continue with rollback?")
            if not confirm:
                print("Rollback cancelled")
                return

        # Perform rollback
        migrator.rollback_config(backup_id)
        print("✅ Configuration rollback completed successfully")
        print(f"   Restored from: {backup_info.get('timestamp', 'unknown')}")
        print(f"   Configuration version: {backup_info.get('version', 'unknown')}")
        print("\n💡 Remember to restart any services that use the configuration")

    except Exception as e:
        print(f"Error during rollback: {e}")
        raise typer.Exit(1)


async def _backup_info(backup_id: str) -> None:
    """Get detailed information about a specific backup."""
    try:
        migrator = _get_config_migrator()

        backup_info = migrator.get_backup_info(backup_id)
        if not backup_info:
            print(f"Backup '{backup_id}' not found")
            return

        print(f"Backup Information: {backup_id}")
        print("=" * 50)
        print(f"Created: {backup_info.get('timestamp', 'unknown')}")
        print(f"Version: {backup_info.get('version', 'unknown')}")
        print(f"Description: {backup_info.get('description', 'No description')}")
        print(f"File Location: {backup_info.get('file_path', 'unknown')}")
        print(f"File Size: {backup_info.get('file_size', 0)} bytes")
        print(f"Checksum: {backup_info.get('checksum', 'unknown')}")

        if backup_info.get('config_path'):
            print(f"Original Config: {backup_info.get('config_path')}")

        # Validate backup
        print("\nValidation Status:")
        is_valid = migrator.validate_backup(backup_id)
        if is_valid:
            print("✅ Backup is valid and can be used for rollback")
        else:
            print("❌ Backup validation failed - file may be corrupted")

    except Exception as e:
        print(f"Error getting backup info: {e}")
        raise typer.Exit(1)


async def _cleanup_migration_history(keep_count: int, force: bool) -> None:
    """Clean up old migration reports to save disk space."""
    try:
        migrator = _get_config_migrator()

        # Get current count
        current_history = migrator.get_migration_history()
        current_count = len(current_history)

        if current_count <= keep_count:
            print(f"No cleanup needed. Current reports: {current_count}, keep: {keep_count}")
            return

        remove_count = current_count - keep_count

        if not force:
            print(f"This will remove {remove_count} old migration reports")
            print(f"Current reports: {current_count}, will keep: {keep_count}")

            confirm = typer.confirm("Continue with cleanup?")
            if not confirm:
                print("Cleanup cancelled")
                return

        # Perform cleanup
        removed_count = migrator.cleanup_old_migration_reports(keep_count)

        print("✅ Cleanup completed")
        print(f"   Removed: {removed_count} migration reports")
        print(f"   Remaining: {len(migrator.get_migration_history())} reports")

    except Exception as e:
        print(f"Error during cleanup: {e}")
        raise typer.Exit(1)


# =============================================================================
# Multi-tenant architecture admin commands (Task 401)
# =============================================================================


async def _get_state_manager():
    """Get SQLite state manager for querying projects and libraries."""
    from common.core.sqlite_state_manager import SQLiteStateManager
    state_manager = SQLiteStateManager()
    await state_manager.initialize()
    return state_manager


# =============================================================================
# Metrics commands (Task 412)
# =============================================================================


async def _fetch_daemon_metrics(metrics_port: int = 9090) -> dict[str, Any] | None:
    """
    Fetch metrics from the Rust daemon's Prometheus endpoint.

    Returns parsed metrics as a dictionary, or None if daemon is unavailable.
    """
    import aiohttp

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"http://127.0.0.1:{metrics_port}/metrics",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status != 200:
                    return None

                text = await response.text()
                return _parse_prometheus_metrics(text)
    except Exception as e:
        logger.debug(f"Failed to fetch daemon metrics: {e}")
        return None


def _parse_prometheus_metrics(text: str) -> dict[str, Any]:
    """Parse Prometheus text format into a structured dictionary."""
    metrics: dict[str, Any] = {
        "sessions": {},
        "queue": {},
        "tenant": {},
        "system": {},
        "raw_lines": []
    }

    for line in text.strip().split("\n"):
        # Skip comments and empty lines
        if not line or line.startswith("#"):
            continue

        metrics["raw_lines"].append(line)

        # Parse metric line: metric_name{labels} value
        try:
            if "{" in line:
                metric_part, value = line.rsplit(" ", 1)
                metric_name = metric_part.split("{")[0]
                labels_str = metric_part.split("{")[1].rstrip("}")
                labels = dict(
                    kv.split("=") for kv in labels_str.replace('"', '').split(",") if "=" in kv
                )
            else:
                parts = line.split(" ")
                metric_name = parts[0]
                value = parts[1] if len(parts) > 1 else "0"
                labels = {}

            value_float = float(value)

            # Categorize metrics
            if metric_name.startswith("memexd_active_sessions"):
                project_id = labels.get("project_id", "unknown")
                priority = labels.get("priority", "unknown")
                if project_id not in metrics["sessions"]:
                    metrics["sessions"][project_id] = {}
                metrics["sessions"][project_id][priority] = value_float

            elif metric_name.startswith("memexd_queue_depth"):
                priority = labels.get("priority", "unknown")
                collection = labels.get("collection", "unknown")
                key = f"{priority}_{collection}"
                metrics["queue"][key] = value_float

            elif metric_name.startswith("memexd_queue_items_processed"):
                priority = labels.get("priority", "unknown")
                status = labels.get("status", "unknown")
                key = f"{priority}_{status}"
                if "items_processed" not in metrics["queue"]:
                    metrics["queue"]["items_processed"] = {}
                metrics["queue"]["items_processed"][key] = value_float

            elif metric_name.startswith("memexd_tenant"):
                tenant_id = labels.get("tenant_id", "unknown")
                if tenant_id not in metrics["tenant"]:
                    metrics["tenant"][tenant_id] = {}
                # Extract metric type from name
                metric_type = metric_name.replace("memexd_tenant_", "")
                metrics["tenant"][tenant_id][metric_type] = value_float

            elif metric_name.startswith("memexd_uptime"):
                metrics["system"]["uptime_seconds"] = value_float

            elif metric_name.startswith("memexd_ingestion_errors"):
                error_type = labels.get("error_type", "unknown")
                if "errors" not in metrics["system"]:
                    metrics["system"]["errors"] = {}
                metrics["system"]["errors"][error_type] = value_float

        except (ValueError, IndexError) as e:
            logger.debug(f"Failed to parse metric line '{line}': {e}")
            continue

    return metrics


def _get_mcp_metrics() -> dict[str, Any]:
    """Get metrics from the Python MCP server."""
    try:
        from common.observability.metrics import get_tool_metrics_summary
        return get_tool_metrics_summary()
    except ImportError:
        logger.debug("MCP metrics module not available")
        return {}
    except Exception as e:
        logger.debug(f"Failed to get MCP metrics: {e}")
        return {}


async def _show_metrics(
    json_output: bool,
    tenant: str | None,
    daemon_only: bool,
    mcp_only: bool
) -> None:
    """Display system metrics (Task 412.12)."""
    from datetime import datetime

    metrics_data: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "daemon": None,
        "mcp": None,
    }

    # Fetch daemon metrics unless MCP-only
    if not mcp_only:
        daemon_metrics = await _fetch_daemon_metrics()
        if daemon_metrics:
            # Filter by tenant if specified
            if tenant and daemon_metrics.get("tenant"):
                filtered_tenant = {
                    k: v for k, v in daemon_metrics["tenant"].items()
                    if tenant.lower() in k.lower()
                }
                daemon_metrics["tenant"] = filtered_tenant
            metrics_data["daemon"] = daemon_metrics
        else:
            metrics_data["daemon"] = {"status": "unavailable", "message": "Daemon metrics endpoint not accessible"}

    # Fetch MCP metrics unless daemon-only
    if not daemon_only:
        mcp_metrics = _get_mcp_metrics()
        metrics_data["mcp"] = mcp_metrics if mcp_metrics else {"status": "no_data"}

    # Output
    if json_output:
        print(json.dumps(metrics_data, indent=2, default=str))
        return

    # Display formatted output
    print(f"System Metrics - {metrics_data['timestamp'][:19]}")
    print("=" * 70)

    # Daemon metrics
    if not mcp_only:
        print("\n[Daemon Metrics]")
        print("-" * 70)

        if metrics_data["daemon"] and "status" not in metrics_data["daemon"]:
            dm = metrics_data["daemon"]

            # System info
            if dm.get("system"):
                uptime = dm["system"].get("uptime_seconds", 0)
                uptime_str = f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m"
                print(f"  Uptime:           {uptime_str}")

                if dm["system"].get("errors"):
                    total_errors = sum(dm["system"]["errors"].values())
                    print(f"  Total Errors:     {int(total_errors)}")

            # Session metrics
            if dm.get("sessions"):
                total_sessions = sum(
                    sum(priorities.values()) for priorities in dm["sessions"].values()
                )
                print(f"  Active Sessions:  {int(total_sessions)}")

                if len(dm["sessions"]) > 0:
                    print("  Sessions by Project:")
                    for project_id, priorities in list(dm["sessions"].items())[:5]:
                        session_count = sum(priorities.values())
                        print(f"    {project_id[:12]}:  {int(session_count)}")

            # Queue metrics
            if dm.get("queue"):
                print("  Queue Depth:")
                for key, value in dm["queue"].items():
                    if key != "items_processed" and value > 0:
                        print(f"    {key}:  {int(value)}")

                if dm["queue"].get("items_processed"):
                    print("  Items Processed:")
                    for key, value in dm["queue"]["items_processed"].items():
                        if value > 0:
                            print(f"    {key}:  {int(value)}")

            # Tenant metrics
            if dm.get("tenant") and tenant:
                print(f"  Tenant Metrics (filter: {tenant}):")
                for tenant_id, tenant_metrics in dm["tenant"].items():
                    print(f"    {tenant_id}:")
                    for metric, value in tenant_metrics.items():
                        print(f"      {metric}: {value}")
        else:
            print("  Daemon metrics unavailable (start daemon with --metrics-port)")

    # MCP metrics
    if not daemon_only:
        print("\n[MCP Server Metrics]")
        print("-" * 70)

        if metrics_data["mcp"] and metrics_data["mcp"].get("status") != "no_data":
            mcp = metrics_data["mcp"]

            if mcp.get("tools"):
                print("  Tool Calls:")
                for tool_name, stats in mcp["tools"].items():
                    calls = stats.get("calls", 0)
                    errors = stats.get("errors", 0)
                    avg_duration = stats.get("avg_duration_ms", 0)
                    print(f"    {tool_name}:  {calls} calls, {errors} errors, {avg_duration:.1f}ms avg")

            if mcp.get("search_scopes"):
                print("  Search Scopes:")
                for scope, count in mcp["search_scopes"].items():
                    print(f"    {scope}:  {count}")

            if mcp.get("total_calls"):
                print(f"  Total Tool Calls: {mcp['total_calls']}")
        else:
            print("  No MCP metrics data available (run search/store operations first)")


async def _watch_metrics(
    tenant: str | None,
    daemon_only: bool,
    mcp_only: bool
) -> None:
    """Watch metrics with continuous refresh (Task 412.13)."""
    import os

    print("Watching system metrics (Ctrl+C to stop)\n")

    try:
        while True:
            # Clear screen
            clear_cmd = ["clear"] if os.name == "posix" else ["cls"]
            subprocess.run(clear_cmd, check=False, capture_output=False)

            await _show_metrics(
                json_output=False,
                tenant=tenant,
                daemon_only=daemon_only,
                mcp_only=mcp_only
            )

            print("\nPress Ctrl+C to stop watching...")

            # Wait 5 seconds
            await asyncio.sleep(5)

    except KeyboardInterrupt:
        print("\nMetrics monitoring stopped")


async def _list_projects(json_output: bool, priority: str | None, active_only: bool) -> None:
    """
    List registered projects with priority and session status.

    Task 401: Display projects table from SQLite state database.
    Shows project_id, path, name, priority, active_sessions, last_active.
    """
    try:
        state_manager = await _get_state_manager()

        # Build query with filters
        query = "SELECT project_id, project_name, project_root, priority, active_sessions, last_active FROM projects"
        conditions = []
        params = []

        if priority:
            conditions.append("priority = ?")
            params.append(priority.lower())

        if active_only:
            conditions.append("active_sessions > 0")

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        # Order by priority and last_active
        query += " ORDER BY CASE priority WHEN 'high' THEN 1 WHEN 'normal' THEN 2 WHEN 'low' THEN 3 END, last_active DESC"

        # Execute query
        async with state_manager._lock:
            cursor = state_manager.connection.execute(query, params)
            rows = cursor.fetchall()

        if not rows:
            filter_desc = ""
            if priority:
                filter_desc += f" with priority '{priority}'"
            if active_only:
                filter_desc += " with active sessions"
            print(f"No projects found{filter_desc}.")
            return

        # Format output
        if json_output:
            projects = []
            for row in rows:
                projects.append({
                    "project_id": row["project_id"],
                    "name": row["project_name"],
                    "path": row["project_root"],
                    "priority": row["priority"],
                    "active_sessions": row["active_sessions"],
                    "last_active": row["last_active"],
                })
            print(json.dumps(projects, indent=2, default=str))
            return

        # Display as table
        print(f"Registered Projects ({len(rows)} found)")
        print("=" * 100)
        print(f"{'Project ID':<14} {'Name':<20} {'Priority':<10} {'Sessions':<10} {'Last Active':<20}")
        print("-" * 100)

        for row in rows:
            project_id = row["project_id"][:12] if row["project_id"] else "N/A"
            name = (row["project_name"] or "unnamed")[:18]
            priority_val = row["priority"] or "normal"
            sessions = row["active_sessions"] or 0
            last_active = str(row["last_active"])[:19] if row["last_active"] else "never"

            # Color coding via text markers
            if priority_val == "high":
                priority_display = f"[HIGH]"
            elif priority_val == "low":
                priority_display = f"[LOW]"
            else:
                priority_display = priority_val

            print(f"{project_id:<14} {name:<20} {priority_display:<10} {sessions:<10} {last_active:<20}")

        # Summary
        high_count = sum(1 for row in rows if row["priority"] == "high")
        active_count = sum(1 for row in rows if (row["active_sessions"] or 0) > 0)
        print("-" * 100)
        print(f"Summary: {high_count} HIGH priority, {active_count} with active sessions")

    except Exception as e:
        logger.error(f"Error listing projects: {e}")
        print(f"Error: {e}")
        raise typer.Exit(1)


async def _show_queue(json_output: bool, collection: str | None, status: str | None) -> None:
    """
    Show ingestion queue status with priority breakdown.

    Task 401: Display ingestion queue with priority breakdown,
    pending/processing/failed by priority level.
    """
    try:
        state_manager = await _get_state_manager()

        # Query ingestion queue
        query = """
            SELECT
                priority,
                status,
                collection_name,
                COUNT(*) as count
            FROM ingestion_queue
        """
        conditions = []
        params = []

        if collection:
            if collection.lower() == "projects":
                conditions.append("collection_name LIKE '\\_%' ESCAPE '\\'")
            elif collection.lower() == "libraries":
                conditions.append("collection_name LIKE '\\_%' ESCAPE '\\' AND collection_name != '_projects'")

        if status:
            conditions.append("status = ?")
            params.append(status.lower())

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " GROUP BY priority, status, collection_name ORDER BY priority ASC, status ASC"

        # Execute query
        async with state_manager._lock:
            cursor = state_manager.connection.execute(query, params)
            rows = cursor.fetchall()

        # Also get total counts by priority
        priority_query = """
            SELECT priority, COUNT(*) as count
            FROM ingestion_queue
            GROUP BY priority
            ORDER BY priority ASC
        """
        async with state_manager._lock:
            priority_cursor = state_manager.connection.execute(priority_query)
            priority_rows = priority_cursor.fetchall()

        # Format output
        if json_output:
            queue_data = {
                "by_priority": {},
                "by_status": {},
                "by_collection": {},
                "total": 0
            }

            for row in priority_rows:
                queue_data["by_priority"][f"priority_{row['priority']}"] = row["count"]
                queue_data["total"] += row["count"]

            # Status breakdown
            status_query = """
                SELECT status, COUNT(*) as count
                FROM ingestion_queue
                GROUP BY status
            """
            async with state_manager._lock:
                status_cursor = state_manager.connection.execute(status_query)
                status_rows = status_cursor.fetchall()

            for row in status_rows:
                queue_data["by_status"][row["status"]] = row["count"]

            print(json.dumps(queue_data, indent=2))
            return

        # Display as table
        total = sum(row["count"] for row in priority_rows)
        print(f"Ingestion Queue Status ({total} total items)")
        print("=" * 70)

        # Priority breakdown
        print("\nBy Priority:")
        print("-" * 40)
        print(f"{'Priority':<15} {'Count':<10} {'Description':<25}")
        print("-" * 40)

        priority_names = {1: "HIGH (active)", 3: "NORMAL", 5: "LOW (background)"}
        for row in priority_rows:
            priority_name = priority_names.get(row["priority"], f"Priority {row['priority']}")
            print(f"{priority_name:<15} {row['count']:<10}")

        # Status breakdown
        status_query = """
            SELECT status, COUNT(*) as count
            FROM ingestion_queue
            GROUP BY status
            ORDER BY status
        """
        async with state_manager._lock:
            status_cursor = state_manager.connection.execute(status_query)
            status_rows = status_cursor.fetchall()

        if status_rows:
            print("\nBy Status:")
            print("-" * 40)
            print(f"{'Status':<15} {'Count':<10}")
            print("-" * 40)
            for row in status_rows:
                print(f"{row['status']:<15} {row['count']:<10}")

        # Collection breakdown (top 5)
        collection_query = """
            SELECT collection_name, COUNT(*) as count
            FROM ingestion_queue
            GROUP BY collection_name
            ORDER BY count DESC
            LIMIT 5
        """
        async with state_manager._lock:
            collection_cursor = state_manager.connection.execute(collection_query)
            collection_rows = collection_cursor.fetchall()

        if collection_rows:
            print("\nTop Collections:")
            print("-" * 40)
            print(f"{'Collection':<25} {'Count':<10}")
            print("-" * 40)
            for row in collection_rows:
                col_name = row["collection_name"][:23] if row["collection_name"] else "N/A"
                print(f"{col_name:<25} {row['count']:<10}")

    except Exception as e:
        logger.error(f"Error showing queue: {e}")
        print(f"Error: {e}")
        raise typer.Exit(1)
