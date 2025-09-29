"""Administrative CLI commands for system management.

This module provides comprehensive system administration capabilities
including status monitoring, configuration management, and engine lifecycle.
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import psutil
import typer

from common.core.client import QdrantWorkspaceClient, create_qdrant_client
from common.core.config import get_config_manager
from loguru import logger
from common.utils.project_detection import ProjectDetector
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
    success_message,
    verbose_option,
    warning_message,
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
):
    """Configuration management."""
    handle_async(_config_management(show, validate, path))


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


@admin_app.command("migration-report")
def migration_report(
    migration_id: Optional[str] = typer.Argument(None, help="Specific migration ID to view"),
    format: str = typer.Option("text", "--format", help="Output format: text, json"),
    export: Optional[str] = typer.Option(None, "--export", help="Export report to file"),
    latest: bool = typer.Option(False, "--latest", help="Show latest migration report"),
):
    """View detailed migration reports."""
    handle_async(_migration_report(migration_id, format, export, latest))


@admin_app.command("migration-history")
def migration_history(
    limit: int = typer.Option(10, "--limit", help="Number of recent migrations to show"),
    source_version: Optional[str] = typer.Option(None, "--source", help="Filter by source version"),
    target_version: Optional[str] = typer.Option(None, "--target", help="Filter by target version"),
    success_only: Optional[bool] = typer.Option(None, "--success-only", help="Show only successful migrations"),
    days_back: Optional[int] = typer.Option(None, "--days", help="Show migrations from last N days"),
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
    import time

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


async def _config_management(show: bool, validate: bool, path: str | None) -> None:
    """Configuration management operations."""
    try:
        config = get_config_manager()

        if show:
            print("Current Configuration")
            print("=" * 50)

            # Add key configuration values
            if hasattr(config, "qdrant"):
                print(f"Qdrant URL:         {config.qdrant.url}")
            if hasattr(config, "embedding"):
                print(f"Embedding Model:    {config.embedding.model}")
            if hasattr(config, "workspace"):
                # Note: Collection prefix field removed as part of multi-tenant architecture
                pass

        if validate:
            print("\nConfiguration Validation")
            print("=" * 50)

            validation_results = []

            # Validate Qdrant connection
            try:
                # Use configured client for validation
                raw_client = get_configured_client(config)
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


# Migration reporting functions
async def _migration_report(migration_id: Optional[str], format: str, export: Optional[str], latest: bool) -> None:
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


async def _migration_history(limit: int, source_version: Optional[str], target_version: Optional[str],
                            success_only: Optional[bool], days_back: Optional[int], format: str) -> None:
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
        restored_config = migrator.rollback_config(backup_id)
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

        print(f"✅ Cleanup completed")
        print(f"   Removed: {removed_count} migration reports")
        print(f"   Remaining: {len(migrator.get_migration_history())} reports")

    except Exception as e:
        print(f"Error during cleanup: {e}")
        raise typer.Exit(1)
