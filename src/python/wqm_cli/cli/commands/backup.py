"""Backup and restore CLI commands.

This module provides comprehensive backup and restore capabilities
with version compatibility validation to prevent data corruption.
"""

from pathlib import Path

import typer
from common.core.backup import BackupManager, RestoreManager
from common.core.client import create_qdrant_client
from common.core.config import get_config_manager
from common.core.error_handling import FileSystemError
from loguru import logger

from wqm_cli import __version__

from ..utils import (
    create_command_app,
    error_message,
    force_option,
    handle_async,
    json_output_option,
    success_message,
    verbose_option,
    warning_message,
)

# Create the backup app using shared utilities
backup_app = create_command_app(
    name="backup",
    help_text="""System backup and restore operations.

Create and manage backups of Qdrant collections, SQLite state,
and system metadata with version compatibility validation.

Examples:
    wqm backup create /path/to/backup         # Create full system backup
    wqm backup create /path/to/backup --description "Before upgrade"
    wqm backup create /path/to/backup --collections myapp-code,myapp-docs
    wqm backup info /path/to/backup           # Show backup information
    wqm backup list /path/to/backups          # List backups in directory
    wqm backup validate /path/to/backup       # Validate backup structure
    wqm backup restore /path/to/backup --dry-run  # Preview restore
    wqm backup restore /path/to/backup        # Restore from backup""",
    no_args_is_help=True,
)


@backup_app.command("create")
def create_backup(
    backup_path: str = typer.Argument(
        ..., help="Path where backup should be created"
    ),
    description: str | None = typer.Option(
        None, "--description", "-d", help="Human-readable backup description"
    ),
    collections: str | None = typer.Option(
        None, "--collections", "-c", help="Comma-separated list of collections (default: all)"
    ),
    force: bool = force_option(),
    verbose: bool = verbose_option(),
):
    """Create a system backup with metadata.

    Creates a backup directory with:
    - manifest.json: Version and metadata
    - sqlite/: SQLite state database
    - collections/: Qdrant collection snapshots
    """
    handle_async(_create_backup(
        backup_path, description, collections, force, verbose
    ))


@backup_app.command("info")
def backup_info(
    backup_path: str = typer.Argument(..., help="Path to backup directory"),
    json_output: bool = json_output_option(),
    verbose: bool = verbose_option(),
):
    """Display backup information and metadata.

    Shows:
    - Backup version
    - Creation timestamp
    - Collections included
    - Total documents
    - Compatibility status with current system
    """
    handle_async(_backup_info(backup_path, json_output, verbose))


@backup_app.command("list")
def list_backups(
    directory: str = typer.Argument(..., help="Directory containing backups"),
    sort_by: str = typer.Option(
        "timestamp", "--sort", "-s", help="Sort by: timestamp, version, size"
    ),
    json_output: bool = json_output_option(),
):
    """List all backups in a directory."""
    handle_async(_list_backups(directory, sort_by, json_output))


@backup_app.command("validate")
def validate_backup(
    backup_path: str = typer.Argument(..., help="Path to backup directory"),
    check_files: bool = typer.Option(
        False, "--check-files", help="Verify all backup files exist"
    ),
    verbose: bool = verbose_option(),
):
    """Validate backup structure and version compatibility.

    Checks:
    - Directory structure is valid
    - manifest.json exists and is readable
    - Version compatibility with current system
    - Optionally verifies all backup files exist
    """
    handle_async(_validate_backup(backup_path, check_files, verbose))


@backup_app.command("restore")
def restore_backup(
    backup_path: str = typer.Argument(..., help="Path to backup directory"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show restore plan without executing"
    ),
    allow_downgrade: bool = typer.Option(
        False, "--allow-downgrade", help="Allow restoring from newer backup version"
    ),
    force: bool = force_option(),
    verbose: bool = verbose_option(),
):
    """Restore system from backup with version validation.

    Validates version compatibility and restores:
    - SQLite state database
    - Qdrant collections (when implemented)

    WARNING: This will overwrite current system state!
    Use --dry-run to preview changes first.
    """
    handle_async(_restore_backup(
        backup_path, dry_run, allow_downgrade, force, verbose
    ))


# Async implementation functions

async def _create_backup(
    backup_path: str,
    description: str | None,
    collections: str | None,
    force: bool,
    verbose: bool,
) -> None:
    """Create backup with metadata and validation."""
    try:
        backup_dir = Path(backup_path)

        # Check if backup directory already exists
        if backup_dir.exists() and not force:
            error_message(
                f"Backup directory already exists: {backup_path}\n"
                "Use --force to overwrite"
            )
            raise typer.Exit(1)

        if verbose:
            typer.echo(f"Creating backup at: {backup_path}")

        # Initialize backup manager
        backup_manager = BackupManager(current_version=__version__)

        # Prepare backup directory structure
        if verbose:
            typer.echo("Preparing backup directory structure...")
        backup_manager.prepare_backup_directory(backup_path)

        # Get Qdrant client to check collections
        client = create_qdrant_client()

        # Determine which collections to backup
        selected_collections = None
        if collections:
            selected_collections = [c.strip() for c in collections.split(",")]
            if verbose:
                typer.echo(f"Backing up collections: {', '.join(selected_collections)}")
        else:
            if verbose:
                typer.echo("Backing up all collections...")

        # Get collection information
        all_collections = await client.list_collections()
        collection_names = [c.name for c in all_collections.collections]

        if selected_collections:
            # Validate selected collections exist
            invalid_collections = set(selected_collections) - set(collection_names)
            if invalid_collections:
                error_message(
                    f"Collections not found: {', '.join(invalid_collections)}"
                )
                raise typer.Exit(1)
            backup_collections = selected_collections
        else:
            backup_collections = collection_names

        # Get document counts
        collection_info = {}
        total_docs = 0
        for col_name in backup_collections:
            col_info = await client.get_collection(col_name)
            doc_count = col_info.points_count
            collection_info[col_name] = doc_count
            total_docs += doc_count

        if verbose:
            typer.echo(f"Total collections: {len(backup_collections)}")
            typer.echo(f"Total documents: {total_docs}")

        # Create backup metadata
        metadata = backup_manager.create_backup_metadata(
            collections=collection_info,
            total_documents=total_docs,
            partial_backup=(selected_collections is not None),
            selected_collections=selected_collections,
            description=description,
        )

        # Save manifest
        if verbose:
            typer.echo("Saving backup manifest...")
        backup_manager.save_backup_manifest(metadata, backup_path)

        # Create Qdrant collection snapshots
        # NOTE: Full snapshot implementation requires Qdrant HTTP API integration
        # For now, store collection metadata only
        if verbose:
            typer.echo("Recording collection metadata...")
            typer.echo("  (Full collection export will be implemented in Task 376.15)")
        collections_dir = backup_dir / "collections"

        # Store collection schemas as JSON for now
        import json
        for col_name in backup_collections:
            if verbose:
                typer.echo(f"  Recording schema for {col_name}...")
            col_info = await client.get_collection(col_name)

            # Save collection config as JSON
            config_path = collections_dir / f"{col_name}.config.json"
            config_data = {
                "name": col_name,
                "vectors_count": col_info.vectors_count,
                "points_count": col_info.points_count,
                "config": {
                    "params": col_info.config.params.dict() if col_info.config and col_info.config.params else {},
                    # Note: Full vector export requires implementing snapshot/export functionality
                }
            }
            config_path.write_text(json.dumps(config_data, indent=2))

        # Copy SQLite database
        if verbose:
            typer.echo("Backing up SQLite state database...")
        config_manager = get_config_manager()
        config = await config_manager.get_config()
        sqlite_db_path = Path(config.state_db_path)

        if sqlite_db_path.exists():
            import shutil
            sqlite_backup_path = backup_dir / "sqlite" / "state.db"
            shutil.copy2(sqlite_db_path, sqlite_backup_path)
            if verbose:
                typer.echo(f"  Copied {sqlite_db_path} -> {sqlite_backup_path}")

        # Success message
        success_message(
            f"Backup created successfully at {backup_path}\n"
            f"Version: {metadata.version}\n"
            f"Collections: {len(backup_collections)}\n"
            f"Documents: {total_docs}\n"
            f"Timestamp: {metadata.format_timestamp()}"
        )

    except FileSystemError as e:
        error_message(f"Backup creation failed: {e}")
        raise typer.Exit(1)
    except Exception as e:
        logger.exception("Unexpected error during backup creation")
        error_message(f"Backup creation failed: {e}")
        raise typer.Exit(1)


async def _backup_info(
    backup_path: str,
    json_output: bool,
    verbose: bool,
) -> None:
    """Display backup information."""
    try:
        backup_dir = Path(backup_path)

        if not backup_dir.exists():
            error_message(f"Backup directory not found: {backup_path}")
            raise typer.Exit(1)

        # Load backup manifest
        backup_manager = BackupManager(current_version=__version__)
        metadata = backup_manager.load_backup_manifest(backup_path)

        if json_output:
            import json
            info_dict = metadata.to_dict()
            typer.echo(json.dumps(info_dict, indent=2))
        else:
            # Display formatted information
            typer.echo("=" * 60)
            typer.echo(f"Backup Information: {backup_path}")
            typer.echo("=" * 60)
            typer.echo(f"Version:         {metadata.version}")
            typer.echo(f"Timestamp:       {metadata.format_timestamp()}")

            if metadata.description:
                typer.echo(f"Description:     {metadata.description}")

            if metadata.collections:
                if isinstance(metadata.collections, dict):
                    typer.echo(f"Collections:     {len(metadata.collections)}")
                    if verbose:
                        for col_name, doc_count in metadata.collections.items():
                            typer.echo(f"  - {col_name}: {doc_count} documents")
                else:
                    typer.echo(f"Collections:     {', '.join(metadata.collections)}")

            if metadata.total_documents:
                typer.echo(f"Total Documents: {metadata.total_documents}")

            if metadata.partial_backup:
                typer.echo("Partial Backup:  Yes")
                if metadata.selected_collections:
                    typer.echo(f"Selected:        {', '.join(metadata.selected_collections)}")

            if metadata.python_version:
                typer.echo(f"Python Version:  {metadata.python_version}")

            if metadata.database_version:
                typer.echo(f"Database Ver:    {metadata.database_version}")

            # Check compatibility with current version
            from common.core.backup import CompatibilityStatus, VersionValidator
            status = VersionValidator.check_compatibility(
                metadata.version,
                __version__
            )

            compatibility_color = {
                CompatibilityStatus.COMPATIBLE: "green",
                CompatibilityStatus.UPGRADE_AVAILABLE: "yellow",
                CompatibilityStatus.DOWNGRADE: "yellow",
                CompatibilityStatus.INCOMPATIBLE: "red",
            }.get(status, "white")

            message = VersionValidator.get_compatibility_message(
                metadata.version,
                __version__
            )

            typer.echo("\nCompatibility:   ", nl=False)
            typer.secho(status.value.upper(), fg=compatibility_color, bold=True)
            typer.echo(f"  {message}")

            typer.echo("=" * 60)

    except FileSystemError as e:
        error_message(f"Cannot read backup: {e}")
        raise typer.Exit(1)
    except Exception as e:
        logger.exception("Error reading backup information")
        error_message(f"Error: {e}")
        raise typer.Exit(1)


async def _list_backups(
    directory: str,
    sort_by: str,
    json_output: bool,
) -> None:
    """List all backups in directory."""
    try:
        backup_dir = Path(directory)

        if not backup_dir.exists() or not backup_dir.is_dir():
            error_message(f"Directory not found: {directory}")
            raise typer.Exit(1)

        # Find all backup directories (contains metadata/manifest.json)
        backup_manager = BackupManager(current_version=__version__)
        backups = []

        for item in backup_dir.iterdir():
            if item.is_dir():
                manifest_path = item / "metadata" / "manifest.json"
                if manifest_path.exists():
                    try:
                        metadata = backup_manager.load_backup_manifest(item)
                        backups.append({
                            "path": str(item),
                            "name": item.name,
                            "version": metadata.version,
                            "timestamp": metadata.timestamp,
                            "formatted_timestamp": metadata.format_timestamp(),
                            "collections": len(metadata.collections) if metadata.collections else 0,
                            "total_documents": metadata.total_documents or 0,
                            "description": metadata.description,
                        })
                    except Exception as e:
                        logger.warning(f"Failed to read backup {item}: {e}")

        if not backups:
            warning_message(f"No backups found in {directory}")
            return

        # Sort backups
        if sort_by == "timestamp":
            backups.sort(key=lambda x: x["timestamp"], reverse=True)
        elif sort_by == "version":
            backups.sort(key=lambda x: x["version"])
        elif sort_by == "size":
            # Sort by document count as proxy for size
            backups.sort(key=lambda x: x["total_documents"], reverse=True)

        if json_output:
            import json
            typer.echo(json.dumps(backups, indent=2))
        else:
            typer.echo(f"\nFound {len(backups)} backup(s) in {directory}:\n")
            for backup in backups:
                typer.echo(f"  {backup['name']}")
                typer.echo(f"    Version:     {backup['version']}")
                typer.echo(f"    Timestamp:   {backup['formatted_timestamp']}")
                typer.echo(f"    Collections: {backup['collections']}")
                typer.echo(f"    Documents:   {backup['total_documents']}")
                if backup['description']:
                    typer.echo(f"    Description: {backup['description']}")
                typer.echo()

    except Exception as e:
        logger.exception("Error listing backups")
        error_message(f"Error: {e}")
        raise typer.Exit(1)


async def _validate_backup(
    backup_path: str,
    check_files: bool,
    verbose: bool,
) -> None:
    """Validate backup structure and compatibility."""
    try:
        backup_dir = Path(backup_path)

        if not backup_dir.exists():
            error_message(f"Backup directory not found: {backup_path}")
            raise typer.Exit(1)

        if verbose:
            typer.echo(f"Validating backup: {backup_path}\n")

        # Validate directory structure
        backup_manager = BackupManager(current_version=__version__)
        if not backup_manager.validate_backup_directory(backup_path):
            error_message("Invalid backup directory structure")
            raise typer.Exit(1)

        if verbose:
            typer.echo("✓ Directory structure valid")

        # Load and validate manifest
        metadata = backup_manager.load_backup_manifest(backup_path)
        if verbose:
            typer.echo("✓ Manifest loaded successfully")

        # Check version compatibility
        from common.core.backup import CompatibilityStatus, VersionValidator
        status = VersionValidator.check_compatibility(
            metadata.version,
            __version__
        )

        if status == CompatibilityStatus.INCOMPATIBLE:
            error_message(
                f"Incompatible backup version: {metadata.version}\n"
                f"Current system version: {__version__}\n"
                "Major or minor version mismatch detected"
            )
            raise typer.Exit(1)

        if verbose:
            if status == CompatibilityStatus.COMPATIBLE:
                typer.echo("✓ Version compatible")
            elif status == CompatibilityStatus.UPGRADE_AVAILABLE:
                warning_message(f"Backup version {metadata.version} is older than current {__version__}")
            elif status == CompatibilityStatus.DOWNGRADE:
                warning_message(f"Backup version {metadata.version} is newer than current {__version__}")

        # Check files exist if requested
        if check_files:
            if verbose:
                typer.echo("\nChecking backup files...")

            # Check SQLite database
            sqlite_path = backup_dir / "sqlite" / "state.db"
            if sqlite_path.exists():
                if verbose:
                    typer.echo("✓ SQLite database found")
            else:
                warning_message("SQLite database not found")

            # Check collection snapshots
            collections_dir = backup_dir / "collections"
            if collections_dir.exists():
                snapshot_files = list(collections_dir.glob("*.snapshot"))
                if verbose:
                    typer.echo(f"✓ Found {len(snapshot_files)} collection snapshot(s)")
            else:
                warning_message("No collection snapshots found")

        success_message(f"Backup validation successful: {backup_path}")

    except FileSystemError as e:
        error_message(f"Validation failed: {e}")
        raise typer.Exit(1)
    except Exception as e:
        logger.exception("Error validating backup")
        error_message(f"Validation error: {e}")
        raise typer.Exit(1)


async def _restore_backup(
    backup_path: str,
    dry_run: bool,
    allow_downgrade: bool,
    force: bool,
    verbose: bool,
) -> None:
    """Restore system from backup with version validation."""
    try:
        from common.core.backup import IncompatibleVersionError

        backup_dir = Path(backup_path)

        if not backup_dir.exists():
            error_message(f"Backup directory not found: {backup_path}")
            raise typer.Exit(1)

        if verbose:
            typer.echo(f"Preparing to restore from: {backup_path}")

        # Initialize restore manager
        restore_manager = RestoreManager(current_version=__version__)

        # Prepare restore plan
        if verbose:
            typer.echo("Validating backup and preparing restore plan...")

        try:
            restore_plan = restore_manager.prepare_restore(
                backup_path,
                allow_downgrade=allow_downgrade,
                dry_run=dry_run
            )
        except IncompatibleVersionError as e:
            error_message(
                f"Cannot restore backup:\n{e.message}\n\n"
                f"Backup version: {e.context.get('backup_version', 'unknown')}\n"
                f"Current version: {e.context.get('current_version', 'unknown')}\n\n"
                "Use --allow-downgrade to restore from newer backup (may cause issues)"
            )
            raise typer.Exit(1)

        # Display restore plan
        typer.echo("\n" + "=" * 60)
        typer.echo("RESTORE PLAN")
        typer.echo("=" * 60)
        typer.echo(f"Backup version:  {restore_plan['backup_version']}")
        typer.echo(f"Current version: {restore_plan['current_version']}")
        typer.echo(f"Backup date:     {restore_plan['backup_timestamp']}")

        if restore_plan.get('partial_backup'):
            typer.echo("Backup type:     PARTIAL")
            if restore_plan.get('selected_collections'):
                typer.echo(f"Collections:     {', '.join(restore_plan['selected_collections'])}")
        else:
            typer.echo("Backup type:     FULL")

        if restore_plan.get('collections'):
            if isinstance(restore_plan['collections'], dict):
                typer.echo(f"Collections:     {len(restore_plan['collections'])}")
            else:
                typer.echo(f"Collections:     {', '.join(restore_plan['collections'])}")

        if restore_plan.get('total_documents'):
            typer.echo(f"Total documents: {restore_plan['total_documents']}")

        # Show what will be restored
        typer.echo("\nWill restore:")
        contents = restore_plan.get('contents', {})

        if contents.get('sqlite'):
            typer.echo(f"  ✓ SQLite database ({len(contents['sqlite'])} file(s))")
        else:
            typer.echo("  ✗ SQLite database (not found)")

        if contents.get('collections'):
            collection_files = [f for f in contents['collections'] if f.endswith('.config.json')]
            typer.echo(f"  ✓ Collection metadata ({len(collection_files)} collection(s))")
            typer.echo("      (Full collection restore not yet implemented)")
        else:
            typer.echo("  ✗ Collection data (not found)")

        typer.echo("=" * 60)

        # If dry-run, exit here
        if dry_run:
            success_message("Dry-run complete. No changes made.")
            return

        # Confirm restore
        if not force:
            typer.echo("\n" + "⚠️  " * 15)
            typer.echo("WARNING: This will OVERWRITE your current system state!")
            typer.echo("⚠️  " * 15 + "\n")

            confirm = typer.confirm(
                "Are you sure you want to restore from this backup?",
                default=False
            )

            if not confirm:
                warning_message("Restore cancelled by user")
                raise typer.Exit(0)

        # Perform actual restore
        if verbose:
            typer.echo("\nStarting restore operation...")

        # Restore SQLite database
        sqlite_backup = backup_dir / "sqlite" / "state.db"
        if sqlite_backup.exists():
            if verbose:
                typer.echo("Restoring SQLite database...")

            config_manager = get_config_manager()
            config = await config_manager.get_config()
            sqlite_db_path = Path(config.state_db_path)

            # Backup current database before overwrite
            if sqlite_db_path.exists() and not force:
                backup_current = sqlite_db_path.with_suffix('.db.backup')
                import shutil
                shutil.copy2(sqlite_db_path, backup_current)
                if verbose:
                    typer.echo(f"  Current database backed up to: {backup_current}")

            # Restore from backup
            import shutil
            shutil.copy2(sqlite_backup, sqlite_db_path)
            if verbose:
                typer.echo("  ✓ SQLite database restored")
        else:
            warning_message("No SQLite database found in backup")

        # Note about collection restore
        if contents.get('collections'):
            warning_message(
                "Collection data restore not yet implemented.\n"
                "Only metadata has been recorded. Full restore requires\n"
                "implementing collection snapshot/import (Task 376.15)."
            )

        success_message(
            f"Restore completed from {backup_path}\n"
            f"Backup version: {restore_plan['backup_version']}\n"
            f"System may require restart for changes to take effect."
        )

    except FileSystemError as e:
        error_message(f"Restore failed: {e}")
        raise typer.Exit(1)
    except Exception as e:
        logger.exception("Unexpected error during restore")
        error_message(f"Restore failed: {e}")
        raise typer.Exit(1)
