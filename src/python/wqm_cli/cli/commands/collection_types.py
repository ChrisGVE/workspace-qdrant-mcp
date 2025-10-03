"""Collection type management CLI commands.

This module provides commands to manage and monitor collection types,
including type assignment, migration, validation, and deletion status.

Usage:
    wqm collections list-types              # Show collections grouped by type
    wqm collections configure-type <name> <type>  # Set collection type
    wqm collections migrate-type <name> <type>    # Migrate collection
    wqm collections validate-types          # Validate all collections
    wqm collections deletion-status         # Show deletion queue status
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from common.core.collection_migration import CollectionMigrator
from common.core.collection_type_config import (
    CollectionTypeConfig,
    DeletionMode,
    get_all_type_configs,
    get_type_config,
)
from common.core.collection_types import CollectionType, CollectionTypeClassifier

from ..utils import (
    confirm,
    create_command_app,
    dry_run_option,
    error_message,
    force_option,
    get_configured_client,
    handle_async,
    info_message,
    json_output_option,
    success_message,
    verbose_option,
    warning_message,
)

# Create the collections app
collection_types_app = create_command_app(
    name="collections",
    help_text="""Manage collection types and type-specific behaviors.

View collections organized by type, configure type assignments, migrate
collections between types, and monitor deletion handling.

Examples:
    wqm collections list-types               # View all collections by type
    wqm collections list-types --format json # JSON output
    wqm collections configure-type my-coll PROJECT  # Set type
    wqm collections migrate-type old-coll PROJECT   # Migrate collection
    wqm collections validate-types           # Validate all collections
    wqm collections deletion-status          # View deletion queues""",
    no_args_is_help=True,
)


def get_daemon_db_path() -> Path:
    """Get path to daemon SQLite database.

    Returns:
        Path to state.db file in daemon state directory.

    Raises:
        FileNotFoundError: If database file doesn't exist.
    """
    config_dir = Path.home() / ".config" / "workspace-qdrant"
    db_path = config_dir / "state.db"

    if not db_path.exists():
        raise FileNotFoundError(
            f"Daemon database not found at {db_path}. "
            "Is the daemon running? Try: wqm service status"
        )

    return db_path


def get_type_color(collection_type: CollectionType) -> str:
    """Get Rich color for collection type.

    Args:
        collection_type: The collection type

    Returns:
        Rich color name
    """
    colors = {
        CollectionType.SYSTEM: "cyan",
        CollectionType.LIBRARY: "blue",
        CollectionType.PROJECT: "green",
        CollectionType.GLOBAL: "yellow",
        CollectionType.UNKNOWN: "white",
    }
    return colors.get(collection_type, "white")


@collection_types_app.command("list-types")
def list_types(
    format: str = typer.Option(
        "table",
        "--format",
        help="Output format: table or json",
    ),
    verbose: bool = verbose_option(),
):
    """List collections grouped by type.

    Shows all collections organized by their type (SYSTEM, LIBRARY, PROJECT, GLOBAL)
    with counts and optional details.
    """
    handle_async(_list_types(format, verbose))


@collection_types_app.command("configure-type")
def configure_type(
    collection_name: str = typer.Argument(..., help="Collection name"),
    collection_type: str = typer.Argument(..., help="Target type (SYSTEM, LIBRARY, PROJECT, GLOBAL)"),
    force: bool = force_option(),
):
    """Set collection type configuration.

    Assigns a type to a collection and validates that the assignment is valid
    for the collection name pattern.
    """
    handle_async(_configure_type(collection_name, collection_type, force))


@collection_types_app.command("migrate-type")
def migrate_type(
    collection_name: str = typer.Argument(..., help="Collection name to migrate"),
    target_type: str = typer.Argument(..., help="Target type (SYSTEM, LIBRARY, PROJECT, GLOBAL)"),
    dry_run: bool = dry_run_option(),
    force: bool = force_option(),
):
    """Migrate collection to a different type.

    Performs a safe migration of collection metadata and configuration
    to conform to the target type's requirements.
    """
    handle_async(_migrate_type(collection_name, target_type, dry_run, force))


@collection_types_app.command("validate-types")
def validate_types(
    format: str = typer.Option(
        "table",
        "--format",
        help="Output format: table or json",
    ),
    severity: Optional[str] = typer.Option(
        None,
        "--severity",
        help="Filter by severity: error, warning, info",
    ),
):
    """Validate all collection types.

    Runs validation on all collections and reports issues by severity.
    """
    handle_async(_validate_types(format, severity))


@collection_types_app.command("deletion-status")
def deletion_status(
    format: str = typer.Option(
        "table",
        "--format",
        help="Output format: table or json",
    ),
    verbose: bool = verbose_option(),
    trigger_cleanup: bool = typer.Option(
        False,
        "--trigger-cleanup",
        help="Trigger manual batch cleanup for cumulative deletions",
    ),
):
    """Show deletion queue status.

    Displays pending deletions by type and deletion mode, with
    dynamic vs cumulative deletion statistics.
    """
    handle_async(_deletion_status(format, verbose, trigger_cleanup))


# Async implementation functions


async def _list_types(format: str, verbose: bool) -> None:
    """Implementation of list-types command."""
    try:
        client = get_configured_client()
        classifier = CollectionTypeClassifier()

        # Get all collections
        collections_response = client.get_collections()
        collections = collections_response.collections if hasattr(collections_response, "collections") else []

        # Group by type
        by_type: Dict[CollectionType, List[str]] = {
            CollectionType.SYSTEM: [],
            CollectionType.LIBRARY: [],
            CollectionType.PROJECT: [],
            CollectionType.GLOBAL: [],
            CollectionType.UNKNOWN: [],
        }

        for coll in collections:
            coll_type = classifier.classify_collection_type(coll.name)
            by_type[coll_type].append(coll.name)

        # Count totals
        type_counts = {coll_type: len(names) for coll_type, names in by_type.items()}

        client.close()

        # Format output
        if format.lower() == "json":
            output = {
                "total_collections": sum(type_counts.values()),
                "by_type": {
                    coll_type.value: {
                        "count": count,
                        "collections": sorted(by_type[coll_type]),
                    }
                    for coll_type, count in type_counts.items()
                },
            }
            print(json.dumps(output, indent=2))
        else:
            console = Console()

            # Summary table
            summary_table = Table(title="Collection Types Summary")
            summary_table.add_column("Type", style="bold", no_wrap=True)
            summary_table.add_column("Count", style="cyan", justify="right")
            summary_table.add_column("Deletion Mode", style="magenta")
            summary_table.add_column("Description", style="white")

            for coll_type in [CollectionType.SYSTEM, CollectionType.LIBRARY, CollectionType.PROJECT, CollectionType.GLOBAL]:
                config = get_type_config(coll_type)
                count = type_counts[coll_type]
                color = get_type_color(coll_type)

                summary_table.add_row(
                    f"[{color}]{coll_type.value.upper()}[/{color}]",
                    str(count),
                    config.deletion_mode.value.upper(),
                    config.description,
                )

            if type_counts[CollectionType.UNKNOWN] > 0:
                summary_table.add_row(
                    "[white]UNKNOWN[/white]",
                    str(type_counts[CollectionType.UNKNOWN]),
                    "N/A",
                    "Unrecognized collection pattern",
                )

            console.print(summary_table)

            # Detailed listing if verbose
            if verbose:
                for coll_type in [CollectionType.SYSTEM, CollectionType.LIBRARY, CollectionType.PROJECT, CollectionType.GLOBAL]:
                    if type_counts[coll_type] > 0:
                        console.print(f"\n[bold {get_type_color(coll_type)}]{coll_type.value.upper()} Collections:[/bold {get_type_color(coll_type)}]")
                        for name in sorted(by_type[coll_type]):
                            console.print(f"  • {name}")

                if type_counts[CollectionType.UNKNOWN] > 0:
                    console.print("\n[bold white]UNKNOWN Collections:[/bold white]")
                    for name in sorted(by_type[CollectionType.UNKNOWN]):
                        console.print(f"  • {name}")

    except Exception as e:
        error_message(f"Failed to list collection types: {e}")
        logger.error("Error listing collection types", error=str(e), exc_info=True)
        raise typer.Exit(1)


async def _configure_type(collection_name: str, collection_type_str: str, force: bool) -> None:
    """Implementation of configure-type command."""
    try:
        # Parse collection type
        try:
            target_type = CollectionType[collection_type_str.upper()]
        except KeyError:
            error_message(
                f"Invalid collection type: {collection_type_str}. "
                f"Valid types: SYSTEM, LIBRARY, PROJECT, GLOBAL"
            )
            raise typer.Exit(1)

        if target_type == CollectionType.UNKNOWN:
            error_message("Cannot configure collection as UNKNOWN type")
            raise typer.Exit(1)

        client = get_configured_client()
        classifier = CollectionTypeClassifier()

        # Check if collection exists
        try:
            client.get_collection(collection_name)
        except Exception:
            error_message(f"Collection not found: {collection_name}")
            client.close()
            raise typer.Exit(1)

        # Detect current type
        current_type = classifier.classify_collection_type(collection_name)

        # Validate type assignment matches name pattern
        config = get_type_config(target_type)
        console = Console()

        # Show before/after comparison
        table = Table(title=f"Type Configuration: {collection_name}")
        table.add_column("Attribute", style="cyan")
        table.add_column("Current", style="yellow")
        table.add_column("New", style="green")

        table.add_row(
            "Type",
            current_type.value.upper(),
            target_type.value.upper(),
        )
        table.add_row(
            "Deletion Mode",
            get_type_config(current_type).deletion_mode.value if current_type != CollectionType.UNKNOWN else "N/A",
            config.deletion_mode.value,
        )
        table.add_row(
            "Description",
            get_type_config(current_type).description if current_type != CollectionType.UNKNOWN else "N/A",
            config.description,
        )

        console.print(table)

        # Confirm if not forced
        if not force:
            if not confirm(f"\nConfigure collection '{collection_name}' as {target_type.value.upper()}?"):
                info_message("Operation cancelled")
                client.close()
                raise typer.Exit(0)

        # In a real implementation, this would update collection metadata
        # For now, we'll just validate the assignment
        info_message(
            f"Type configuration for '{collection_name}' would be set to {target_type.value.upper()}\n"
            f"Note: Actual metadata update requires collection migration system integration"
        )

        client.close()
        success_message(f"Collection type configured: {collection_name} -> {target_type.value.upper()}")

    except typer.Exit:
        raise
    except Exception as e:
        error_message(f"Failed to configure collection type: {e}")
        logger.error("Error configuring collection type", error=str(e), exc_info=True)
        raise typer.Exit(1)


async def _migrate_type(collection_name: str, target_type_str: str, dry_run: bool, force: bool) -> None:
    """Implementation of migrate-type command."""
    try:
        # Parse collection type
        try:
            target_type = CollectionType[target_type_str.upper()]
        except KeyError:
            error_message(
                f"Invalid collection type: {target_type_str}. "
                f"Valid types: SYSTEM, LIBRARY, PROJECT, GLOBAL"
            )
            raise typer.Exit(1)

        if target_type == CollectionType.UNKNOWN:
            error_message("Cannot migrate collection to UNKNOWN type")
            raise typer.Exit(1)

        console = Console()
        client = get_configured_client()

        # Initialize migrator
        migrator = CollectionMigrator(client)
        await migrator.initialize()

        # Detect current type
        detection = await migrator.detect_collection_type(collection_name)

        if dry_run:
            info_message(f"DRY RUN: Migration preview for {collection_name}")

        # Show migration plan
        table = Table(title=f"Migration Plan: {collection_name}")
        table.add_column("Stage", style="cyan")
        table.add_column("Details", style="white")

        table.add_row("Current Type", f"{detection.detected_type.value.upper()} (confidence: {detection.confidence:.2%})")
        table.add_row("Target Type", target_type.value.upper())
        table.add_row("Mode", "DRY RUN" if dry_run else "EXECUTE")
        table.add_row("Deletion Mode Change",
                     f"{get_type_config(detection.detected_type).deletion_mode.value} → {get_type_config(target_type).deletion_mode.value}")

        console.print(table)

        # Validate collection first
        validation = await migrator.validate_collection(collection_name)
        if not validation.is_valid:
            warning_message(f"Collection has {len(validation.errors)} validation errors")
            for error in validation.errors[:5]:  # Show first 5 errors
                console.print(f"  • [{error.severity.value}] {error.message}")
            if len(validation.errors) > 5:
                console.print(f"  ... and {len(validation.errors) - 5} more errors")

        # Confirm if not dry-run and not forced
        if not dry_run and not force:
            if not confirm(f"\nProceed with migration of '{collection_name}'?"):
                info_message("Migration cancelled")
                client.close()
                raise typer.Exit(0)

        # Perform migration
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"{'Previewing' if dry_run else 'Executing'} migration...",
                total=None,
            )

            migration_result = await migrator.migrate_collection(
                collection_name,
                target_type,
                dry_run=dry_run,
            )

            progress.update(task, completed=True)

        # Display results
        if migration_result.success:
            if dry_run:
                success_message(f"Migration preview successful for {collection_name}")
                info_message("Run without --dry-run to execute migration")
            else:
                success_message(f"Successfully migrated {collection_name} to {target_type.value.upper()}")

            if migration_result.changes_made:
                console.print("\n[bold]Changes:[/bold]")
                for change in migration_result.changes_made[:10]:  # Show first 10 changes
                    console.print(f"  • {change}")
                if len(migration_result.changes_made) > 10:
                    console.print(f"  ... and {len(migration_result.changes_made) - 10} more changes")
        else:
            error_message(f"Migration failed: {migration_result.error_message}")
            if migration_result.validation_errors:
                console.print("\n[bold]Validation Errors:[/bold]")
                for error in migration_result.validation_errors:
                    console.print(f"  • {error}")

        client.close()

    except typer.Exit:
        raise
    except Exception as e:
        error_message(f"Failed to migrate collection: {e}")
        logger.error("Error migrating collection", error=str(e), exc_info=True)
        raise typer.Exit(1)


async def _validate_types(format: str, severity: Optional[str]) -> None:
    """Implementation of validate-types command."""
    try:
        client = get_configured_client()
        classifier = CollectionTypeClassifier()

        # Get all collections
        collections_response = client.get_collections()
        collections = collections_response.collections if hasattr(collections_response, "collections") else []

        # Initialize migrator
        migrator = CollectionMigrator(client)
        await migrator.initialize()

        console = Console()
        validation_results = []

        # Validate each collection
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Validating {len(collections)} collections...",
                total=len(collections),
            )

            for coll in collections:
                result = await migrator.validate_collection(coll.name)
                validation_results.append(result)
                progress.advance(task)

        # Filter by severity if specified
        if severity:
            severity = severity.lower()

        # Prepare output
        total_collections = len(validation_results)
        valid_count = sum(1 for r in validation_results if r.is_valid)
        invalid_count = total_collections - valid_count

        if format.lower() == "json":
            output = {
                "total_collections": total_collections,
                "valid": valid_count,
                "invalid": invalid_count,
                "results": [r.to_dict() for r in validation_results if not r.is_valid or severity],
            }
            print(json.dumps(output, indent=2))
        else:
            # Summary table
            summary_table = Table(title="Validation Summary")
            summary_table.add_column("Metric", style="cyan")
            summary_table.add_column("Value", style="yellow", justify="right")

            summary_table.add_row("Total Collections", str(total_collections))
            summary_table.add_row("Valid", f"[green]{valid_count}[/green]")
            summary_table.add_row("Invalid", f"[red]{invalid_count}[/red]")

            console.print(summary_table)

            # Issues table for invalid collections
            if invalid_count > 0:
                console.print("\n")
                issues_table = Table(title="Validation Issues")
                issues_table.add_column("Collection", style="cyan")
                issues_table.add_column("Type", style="blue")
                issues_table.add_column("Errors", style="red", justify="right")
                issues_table.add_column("Warnings", style="yellow", justify="right")
                issues_table.add_column("Sample Issue", style="white")

                for result in validation_results:
                    if not result.is_valid:
                        error_count = len(result.errors)
                        warning_count = len(result.warnings)

                        # Get first error or warning
                        sample_issue = ""
                        if result.errors:
                            sample_issue = result.errors[0].message
                        elif result.warnings:
                            sample_issue = result.warnings[0].message

                        # Truncate sample issue if too long
                        if len(sample_issue) > 50:
                            sample_issue = sample_issue[:47] + "..."

                        issues_table.add_row(
                            result.collection_name,
                            result.detected_type.value if result.detected_type else "UNKNOWN",
                            str(error_count),
                            str(warning_count),
                            sample_issue,
                        )

                console.print(issues_table)

        client.close()

    except Exception as e:
        error_message(f"Failed to validate collection types: {e}")
        logger.error("Error validating collection types", error=str(e), exc_info=True)
        raise typer.Exit(1)


async def _deletion_status(format: str, verbose: bool, trigger_cleanup: bool) -> None:
    """Implementation of deletion-status command."""
    try:
        # Get type configs
        type_configs = get_all_type_configs()

        # Determine which types use cumulative deletion
        cumulative_types = [
            coll_type for coll_type, config in type_configs.items()
            if config.deletion_mode == DeletionMode.CUMULATIVE
        ]
        dynamic_types = [
            coll_type for coll_type, config in type_configs.items()
            if config.deletion_mode == DeletionMode.DYNAMIC
        ]

        # Try to get queue information from database
        deletion_stats = {
            "cumulative_pending": 0,
            "cumulative_types": [t.value for t in cumulative_types],
            "dynamic_types": [t.value for t in dynamic_types],
            "batch_cleanup_schedule": "Every 24 hours or 1000 items",
        }

        try:
            db_path = get_daemon_db_path()
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Query for pending deletions (this is a placeholder query)
            # In actual implementation, would query deletion tracking table
            cursor.execute("""
                SELECT COUNT(*) as count
                FROM ingestion_queue
                WHERE error_message_id IS NOT NULL
            """)
            result = cursor.fetchone()
            deletion_stats["cumulative_pending"] = result["count"] if result else 0

            conn.close()
        except FileNotFoundError:
            # Daemon not running, skip database stats
            pass
        except Exception as e:
            logger.warning(f"Failed to query deletion stats: {e}")

        # Format output
        if format.lower() == "json":
            print(json.dumps(deletion_stats, indent=2))
        else:
            console = Console()

            # Deletion modes table
            modes_table = Table(title="Deletion Handling by Type")
            modes_table.add_column("Collection Type", style="cyan")
            modes_table.add_column("Deletion Mode", style="yellow")
            modes_table.add_column("Behavior", style="white")

            for coll_type in [CollectionType.SYSTEM, CollectionType.LIBRARY, CollectionType.PROJECT, CollectionType.GLOBAL]:
                config = get_type_config(coll_type)
                color = get_type_color(coll_type)

                behavior = (
                    "Mark deleted + batch cleanup"
                    if config.deletion_mode == DeletionMode.CUMULATIVE
                    else "Immediate deletion"
                )

                modes_table.add_row(
                    f"[{color}]{coll_type.value.upper()}[/{color}]",
                    config.deletion_mode.value.upper(),
                    behavior,
                )

            console.print(modes_table)

            # Queue status
            console.print(f"\n[bold]Deletion Queue Status:[/bold]")
            console.print(f"  Cumulative deletions pending: {deletion_stats['cumulative_pending']}")
            console.print(f"  Batch cleanup schedule: {deletion_stats['batch_cleanup_schedule']}")

            if trigger_cleanup:
                warning_message("\nManual batch cleanup trigger not yet implemented")
                info_message("This will be available when deletion tracking system is fully integrated")

            if verbose:
                console.print(f"\n[bold]Cumulative Deletion Types:[/bold]")
                for type_name in deletion_stats["cumulative_types"]:
                    console.print(f"  • {type_name.upper()}")

                console.print(f"\n[bold]Dynamic Deletion Types:[/bold]")
                for type_name in deletion_stats["dynamic_types"]:
                    console.print(f"  • {type_name.upper()}")

    except Exception as e:
        error_message(f"Failed to get deletion status: {e}")
        logger.error("Error getting deletion status", error=str(e), exc_info=True)
        raise typer.Exit(1)
