"""
Project collection alias management commands.

This module provides CLI commands for managing project collection aliases,
enabling smooth migrations when project IDs change (e.g., when a local
project gains a git remote).

Commands:
    - update-remote: Detect remote change and create alias
    - list-aliases: Show all active collection aliases
    - remove-alias: Remove a specific alias

Example usage:
    # Update project remote (creates alias automatically)
    wqm project update-remote --project-path /path/to/project

    # List all aliases
    wqm project list-aliases

    # Remove specific alias
    wqm project remove-alias --old-collection _path_abc123def456
"""

import asyncio
import hashlib
import json as json_module
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
from loguru import logger
from tabulate import tabulate

from ....common.core.client import QdrantWorkspaceClient
from ....common.core.collection_aliases import AliasManager
from ....common.core.collection_naming import build_project_collection_name
from ....common.core.config import get_config

# Create Typer app for project commands
project_app = typer.Typer(
    name="project",
    help="Project collection alias management",
    no_args_is_help=True
)


def generate_project_id(project_path: Path) -> str:
    """
    Generate a 12-character hex project ID from project path.

    Args:
        project_path: Path to the project directory

    Returns:
        12-character hex hash
    """
    path_str = str(project_path.resolve())
    return hashlib.sha256(path_str.encode('utf-8')).hexdigest()[:12]


def get_git_remote_url(project_path: Path) -> Optional[str]:
    """
    Get the git remote URL for a project.

    Args:
        project_path: Path to the project directory

    Returns:
        Git remote URL if available, None otherwise
    """
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=str(project_path),
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()

    except Exception as e:
        logger.debug(f"Failed to get git remote URL: {e}")

    return None


def sanitize_remote_url_to_project_id(remote_url: str) -> str:
    """
    Convert a git remote URL to a valid 12-char project ID.

    Args:
        remote_url: Git remote URL

    Returns:
        12-character hex hash based on URL
    """
    # Hash the URL to get consistent project ID
    return hashlib.sha256(remote_url.encode('utf-8')).hexdigest()[:12]


@project_app.command(name="update-remote")
def update_remote_command(
    project_path: Path = typer.Option(
        ...,
        "--project-path",
        help="Path to the project directory",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Force alias creation even if remote hasn't changed"
    )
):
    """
    Update project remote configuration and create alias if needed.

    This command detects when a project's git remote has changed (e.g., a local
    project gained a git remote URL) and automatically creates a collection alias
    to maintain zero-downtime access during the migration.

    The command will:
    1. Check the current git remote URL
    2. Calculate the old (path-based) and new (remote-based) project IDs
    3. Create an alias from old collection name to new collection name
    4. Preserve existing data access during migration

    Example:
        wqm project update-remote --project-path /path/to/project
    """
    asyncio.run(_update_remote(project_path, force))


async def _update_remote(project_path: Path, force: bool):
    """
    Implementation of update-remote command.

    Args:
        project_path: Path to the project directory
        force: Whether to force alias creation
    """
    try:
        typer.echo(f"Checking project remote configuration: {project_path}")

        # Get current git remote
        remote_url = get_git_remote_url(project_path)

        if not remote_url and not force:
            typer.echo("No git remote URL found for this project.")
            typer.echo("Use --force to create alias based on path hash only.")
            raise typer.Exit(1)

        # Calculate old project ID (path-based)
        old_project_id = generate_project_id(project_path)
        old_collection_name = build_project_collection_name(old_project_id)

        typer.echo(f"Old collection name (path-based): {old_collection_name}")

        # Calculate new project ID
        if remote_url:
            new_project_id = sanitize_remote_url_to_project_id(remote_url)
            typer.echo(f"Remote URL detected: {remote_url}")
        else:
            # Force mode without remote - use a different hash to simulate change
            new_project_id = hashlib.sha256(
                f"forced_{project_path}".encode('utf-8')
            ).hexdigest()[:12]
            typer.echo("Force mode: Using alternative hash as new project ID")

        new_collection_name = build_project_collection_name(new_project_id)
        typer.echo(f"New collection name (remote-based): {new_collection_name}")

        # Check if they're the same
        if old_collection_name == new_collection_name and not force:
            typer.echo("Collection names are identical. No alias needed.")
            raise typer.Exit(0)

        # Initialize client and alias manager
        config = get_config()
        client = QdrantWorkspaceClient()
        await client.initialize()

        alias_manager = AliasManager(client.client, client.state_manager)
        await alias_manager.initialize()

        # Check if new collection exists
        collections = client.client.get_collections().collections
        collection_names = [col.name for col in collections]

        if new_collection_name not in collection_names:
            typer.echo(f"\nWarning: New collection '{new_collection_name}' does not exist yet.")
            typer.echo("You may need to create it or wait for auto-ingestion to create it.")
            if not typer.confirm("Create alias anyway?"):
                typer.echo("Aborted.")
                raise typer.Exit(1)

        # Check if alias already exists
        existing_alias = await alias_manager.get_alias(old_collection_name)
        if existing_alias:
            if existing_alias.collection_name == new_collection_name:
                typer.echo(f"\nAlias already exists: {old_collection_name} -> {new_collection_name}")
                raise typer.Exit(0)
            else:
                typer.echo(
                    f"\nWarning: Alias '{old_collection_name}' already points to "
                    f"'{existing_alias.collection_name}'"
                )
                if not typer.confirm(f"Update alias to point to '{new_collection_name}'?"):
                    typer.echo("Aborted.")
                    raise typer.Exit(1)

                # Delete old alias first
                await alias_manager.delete_alias(old_collection_name)

        # Create alias
        typer.echo(f"\nCreating alias: {old_collection_name} -> {new_collection_name}")

        await alias_manager.create_alias(
            alias_name=old_collection_name,
            collection_name=new_collection_name,
            created_by="cli",
            metadata={
                "project_path": str(project_path),
                "remote_url": remote_url,
                "migration_reason": "remote_update"
            }
        )

        typer.echo("✓ Alias created successfully")
        typer.echo("\nQueries using the old collection name will now transparently")
        typer.echo("access the new collection. The alias can be removed after migration")
        typer.echo("is complete using:")
        typer.echo(f"  wqm project remove-alias --old-collection {old_collection_name}")

        await client.close()

    except typer.Exit:
        raise
    except Exception as e:
        logger.error(f"Failed to update project remote: {e}")
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@project_app.command(name="list-aliases")
def list_aliases_command(
    format: str = typer.Option(
        "table",
        "--format",
        help="Output format (table, json, simple)",
        case_sensitive=False
    )
):
    """
    List all active collection aliases.

    This command displays all collection aliases currently configured in the
    system, showing the alias name, the actual collection it points to, when
    it was created, and by what source.

    Example:
        wqm project list-aliases
        wqm project list-aliases --format json
    """
    asyncio.run(_list_aliases(format))


async def _list_aliases(format: str):
    """
    Implementation of list-aliases command.

    Args:
        format: Output format (table, json, simple)
    """
    try:
        # Initialize client and alias manager
        config = get_config()
        client = QdrantWorkspaceClient()
        await client.initialize()

        alias_manager = AliasManager(client.client, client.state_manager)
        await alias_manager.initialize()

        # Get all aliases
        aliases = await alias_manager.list_aliases()

        if not aliases:
            typer.echo("No collection aliases found.")
            await client.close()
            return

        # Format output
        if format == "json":
            data = [
                {
                    "alias_name": alias.alias_name,
                    "collection_name": alias.collection_name,
                    "created_at": alias.created_at.isoformat(),
                    "created_by": alias.created_by,
                    "metadata": alias.metadata or {}
                }
                for alias in aliases
            ]
            typer.echo(json_module.dumps(data, indent=2))

        elif format == "simple":
            for alias in aliases:
                typer.echo(f"{alias.alias_name} -> {alias.collection_name}")

        else:  # table format
            headers = ["Alias Name", "Collection Name", "Created At", "Created By"]
            rows = [
                [
                    alias.alias_name,
                    alias.collection_name,
                    alias.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                    alias.created_by
                ]
                for alias in aliases
            ]

            typer.echo(tabulate(rows, headers=headers, tablefmt="grid"))

            # Show metadata if any alias has it
            for alias in aliases:
                if alias.metadata:
                    typer.echo(f"\nMetadata for {alias.alias_name}:")
                    for key, value in alias.metadata.items():
                        typer.echo(f"  {key}: {value}")

        await client.close()

    except Exception as e:
        logger.error(f"Failed to list aliases: {e}")
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@project_app.command(name="remove-alias")
def remove_alias_command(
    old_collection: str = typer.Option(
        ...,
        "--old-collection",
        help="The alias name (old collection name) to remove"
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Skip confirmation prompt"
    )
):
    """
    Remove a collection alias.

    This command removes a collection alias after migration is complete.
    The alias name is typically the old collection name that was aliased
    to the new collection name.

    Example:
        wqm project remove-alias --old-collection _path_abc123def456
        wqm project remove-alias --old-collection _path_abc123def456 --force
    """
    asyncio.run(_remove_alias(old_collection, force))


async def _remove_alias(old_collection: str, force: bool):
    """
    Implementation of remove-alias command.

    Args:
        old_collection: The alias name to remove
        force: Skip confirmation prompt
    """
    try:
        # Initialize client and alias manager
        config = get_config()
        client = QdrantWorkspaceClient()
        await client.initialize()

        alias_manager = AliasManager(client.client, client.state_manager)
        await alias_manager.initialize()

        # Check if alias exists
        alias = await alias_manager.get_alias(old_collection)
        if not alias:
            typer.echo(f"Alias '{old_collection}' not found.")
            await client.close()
            raise typer.Exit(1)

        typer.echo(f"Alias found: {alias.alias_name} -> {alias.collection_name}")
        typer.echo(f"Created at: {alias.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        typer.echo(f"Created by: {alias.created_by}")

        if alias.metadata:
            typer.echo("Metadata:")
            for key, value in alias.metadata.items():
                typer.echo(f"  {key}: {value}")

        # Confirm deletion
        if not force:
            if not typer.confirm(f"\nRemove alias '{old_collection}'?"):
                typer.echo("Aborted.")
                raise typer.Exit(0)

        # Delete alias
        typer.echo(f"\nRemoving alias: {old_collection}")
        await alias_manager.delete_alias(old_collection)

        typer.echo("✓ Alias removed successfully")
        typer.echo("\nNote: The actual collection remains unchanged. Only the alias was removed.")

        await client.close()

    except typer.Exit:
        raise
    except Exception as e:
        logger.error(f"Failed to remove alias: {e}")
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


# Export the Typer app
__all__ = ["project_app"]
