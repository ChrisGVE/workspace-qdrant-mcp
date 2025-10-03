"""
Branch management commands for workspace-qdrant-mcp.

This module provides CLI commands for managing Git branches within project collections,
enabling operations like delete, rename, and list for branch-specific document filtering.

Commands:
    - delete: Delete all documents with specified branch from collection
    - rename: Update branch metadata for all affected documents
    - list: List all branches in project collection with document counts

Example usage:
    # Delete all documents from a specific branch
    wqm branch delete --project abc123def456 --branch feature/old-api

    # Rename a branch across all documents
    wqm branch rename --project abc123def456 --from old-name --to new-name

    # List all branches in a project collection
    wqm branch list --project abc123def456
"""

import asyncio
import json as json_module
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import typer
from loguru import logger
from qdrant_client.models import (
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointIdsList,
)
from tabulate import tabulate

from ....common.core.client import QdrantWorkspaceClient
from ....common.core.collection_naming import build_project_collection_name
from ....common.core.config import get_config
from ....common.utils.git_utils import get_current_branch

# Create Typer app for branch commands
branch_app = typer.Typer(
    name="branch", help="Branch management for project collections", no_args_is_help=True
)


def get_project_collection(project_id: str) -> str:
    """
    Get the collection name for a project ID.

    Args:
        project_id: The 12-character hex project ID

    Returns:
        Collection name in format _{project_id}
    """
    return build_project_collection_name(project_id)


async def find_documents_by_branch(
    client: QdrantWorkspaceClient, collection: str, branch: str
) -> List[str]:
    """
    Find all document IDs for a specific branch in a collection.

    Args:
        client: Initialized Qdrant workspace client
        collection: Collection name to search
        branch: Branch name to filter by

    Returns:
        List of document point IDs matching the branch
    """
    try:
        # Create filter for branch metadata
        branch_filter = Filter(
            must=[FieldCondition(key="branch", match=MatchValue(value=branch))]
        )

        # Scroll through all matching documents
        point_ids = []
        offset = None
        batch_size = 100

        while True:
            # Scroll with filter
            scroll_result = client.client.scroll(
                collection_name=collection,
                scroll_filter=branch_filter,
                limit=batch_size,
                offset=offset,
                with_payload=False,  # We only need IDs
                with_vectors=False,
            )

            records, next_offset = scroll_result

            # Extract point IDs
            for record in records:
                point_ids.append(str(record.id))

            # Check if we're done
            if next_offset is None or not records:
                break

            offset = next_offset

        logger.debug(
            f"Found {len(point_ids)} documents with branch '{branch}' in collection '{collection}'"
        )
        return point_ids

    except Exception as e:
        logger.error(f"Failed to find documents by branch: {e}")
        raise


async def get_all_branches(
    client: QdrantWorkspaceClient, collection: str
) -> Dict[str, int]:
    """
    Get all unique branches in a collection with document counts.

    Args:
        client: Initialized Qdrant workspace client
        collection: Collection name to analyze

    Returns:
        Dictionary mapping branch names to document counts
    """
    try:
        branch_counts: Dict[str, int] = defaultdict(int)
        offset = None
        batch_size = 100

        while True:
            # Scroll through all documents
            scroll_result = client.client.scroll(
                collection_name=collection,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            records, next_offset = scroll_result

            # Count branches from metadata
            for record in records:
                if record.payload and "branch" in record.payload:
                    branch = record.payload["branch"]
                    branch_counts[branch] += 1
                else:
                    # Documents without branch metadata
                    branch_counts["<no-branch>"] += 1

            # Check if we're done
            if next_offset is None or not records:
                break

            offset = next_offset

        logger.debug(
            f"Found {len(branch_counts)} unique branches in collection '{collection}'"
        )
        return dict(branch_counts)

    except Exception as e:
        logger.error(f"Failed to get all branches: {e}")
        raise


@branch_app.command(name="delete")
def delete_branch_command(
    project: str = typer.Option(
        ...,
        "--project",
        help="Project ID (12-character hex hash)",
        metavar="PROJECT_ID",
    ),
    branch: str = typer.Option(
        ..., "--branch", help="Branch name to delete documents from", metavar="BRANCH"
    ),
    force: bool = typer.Option(
        False, "--force", help="Skip confirmation prompt and delete immediately"
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be deleted without actually deleting",
    ),
):
    """
    Delete all documents with specified branch from project collection.

    This command removes all documents that have the specified branch in their
    metadata from the project collection. Use with caution as this operation
    cannot be undone.

    Example:
        wqm branch delete --project abc123def456 --branch feature/old-api
        wqm branch delete --project abc123def456 --branch main --force
        wqm branch delete --project abc123def456 --branch test --dry-run
    """
    asyncio.run(_delete_branch(project, branch, force, dry_run))


async def _delete_branch(project: str, branch: str, force: bool, dry_run: bool):
    """
    Implementation of delete branch command.

    Args:
        project: Project ID
        branch: Branch name to delete
        force: Skip confirmation
        dry_run: Show what would be deleted without deleting
    """
    try:
        collection = get_project_collection(project)
        typer.echo(f"Deleting documents from branch '{branch}' in collection '{collection}'")

        # Initialize client
        config = get_config()
        client = QdrantWorkspaceClient()
        await client.initialize()

        # Check if collection exists
        collections = client.client.get_collections().collections
        collection_names = [col.name for col in collections]

        if collection not in collection_names:
            typer.echo(f"Error: Collection '{collection}' does not exist", err=True)
            await client.close()
            raise typer.Exit(1)

        # Find documents by branch
        typer.echo(f"Finding documents with branch '{branch}'...")
        point_ids = await find_documents_by_branch(client, collection, branch)

        if not point_ids:
            typer.echo(f"No documents found with branch '{branch}'")
            await client.close()
            return

        typer.echo(f"Found {len(point_ids)} documents to delete")

        # Dry run mode
        if dry_run:
            typer.echo("\n[DRY RUN] Would delete the following documents:")
            for i, point_id in enumerate(point_ids[:10], 1):
                typer.echo(f"  {i}. Point ID: {point_id}")
            if len(point_ids) > 10:
                typer.echo(f"  ... and {len(point_ids) - 10} more")
            typer.echo(f"\nTotal: {len(point_ids)} documents would be deleted")
            await client.close()
            return

        # Confirmation prompt (unless --force)
        if not force:
            confirm_msg = f"\nDelete {len(point_ids)} documents from branch '{branch}'?"
            if not typer.confirm(confirm_msg):
                typer.echo("Aborted.")
                await client.close()
                raise typer.Exit(0)

        # Delete documents
        typer.echo(f"Deleting {len(point_ids)} documents...")

        # Use FilterSelector for efficient deletion
        branch_filter = Filter(
            must=[FieldCondition(key="branch", match=MatchValue(value=branch))]
        )

        client.client.delete(
            collection_name=collection, points_selector=FilterSelector(filter=branch_filter)
        )

        typer.echo(f"✓ Successfully deleted {len(point_ids)} documents from branch '{branch}'")

        await client.close()

    except typer.Exit:
        raise
    except Exception as e:
        logger.error(f"Failed to delete branch: {e}")
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@branch_app.command(name="rename")
def rename_branch_command(
    project: str = typer.Option(
        ...,
        "--project",
        help="Project ID (12-character hex hash)",
        metavar="PROJECT_ID",
    ),
    from_branch: str = typer.Option(
        ..., "--from", help="Current branch name", metavar="OLD_BRANCH"
    ),
    to_branch: str = typer.Option(
        ..., "--to", help="New branch name", metavar="NEW_BRANCH"
    ),
    force: bool = typer.Option(
        False, "--force", help="Skip confirmation prompt and rename immediately"
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be renamed without actually renaming",
    ),
):
    """
    Rename branch metadata for all affected documents in project collection.

    This command updates the branch metadata field from the old branch name
    to the new branch name for all matching documents. This is useful when
    a Git branch is renamed and you want to keep the metadata in sync.

    Example:
        wqm branch rename --project abc123def456 --from old-name --to new-name
        wqm branch rename --project abc123def456 --from feature/old --to feature/new --force
        wqm branch rename --project abc123def456 --from test --to staging --dry-run
    """
    asyncio.run(_rename_branch(project, from_branch, to_branch, force, dry_run))


async def _rename_branch(
    project: str, from_branch: str, to_branch: str, force: bool, dry_run: bool
):
    """
    Implementation of rename branch command.

    Args:
        project: Project ID
        from_branch: Current branch name
        to_branch: New branch name
        force: Skip confirmation
        dry_run: Show what would be renamed without renaming
    """
    try:
        collection = get_project_collection(project)
        typer.echo(
            f"Renaming branch '{from_branch}' to '{to_branch}' in collection '{collection}'"
        )

        # Initialize client
        config = get_config()
        client = QdrantWorkspaceClient()
        await client.initialize()

        # Check if collection exists
        collections = client.client.get_collections().collections
        collection_names = [col.name for col in collections]

        if collection not in collection_names:
            typer.echo(f"Error: Collection '{collection}' does not exist", err=True)
            await client.close()
            raise typer.Exit(1)

        # Find documents by old branch name
        typer.echo(f"Finding documents with branch '{from_branch}'...")
        point_ids = await find_documents_by_branch(client, collection, from_branch)

        if not point_ids:
            typer.echo(f"No documents found with branch '{from_branch}'")
            await client.close()
            return

        typer.echo(f"Found {len(point_ids)} documents to rename")

        # Dry run mode
        if dry_run:
            typer.echo("\n[DRY RUN] Would rename branch metadata for:")
            for i, point_id in enumerate(point_ids[:10], 1):
                typer.echo(f"  {i}. Point ID: {point_id}")
            if len(point_ids) > 10:
                typer.echo(f"  ... and {len(point_ids) - 10} more")
            typer.echo(
                f"\nTotal: {len(point_ids)} documents would have branch renamed from '{from_branch}' to '{to_branch}'"
            )
            await client.close()
            return

        # Confirmation prompt (unless --force)
        if not force:
            confirm_msg = f"\nRename branch for {len(point_ids)} documents from '{from_branch}' to '{to_branch}'?"
            if not typer.confirm(confirm_msg):
                typer.echo("Aborted.")
                await client.close()
                raise typer.Exit(0)

        # Update branch metadata
        typer.echo(f"Updating branch metadata for {len(point_ids)} documents...")

        # Update in batches for better performance
        batch_size = 100
        for i in range(0, len(point_ids), batch_size):
            batch = point_ids[i : i + batch_size]

            client.client.set_payload(
                collection_name=collection,
                payload={"branch": to_branch},
                points=batch,
            )

        typer.echo(
            f"✓ Successfully renamed branch from '{from_branch}' to '{to_branch}' for {len(point_ids)} documents"
        )

        await client.close()

    except typer.Exit:
        raise
    except Exception as e:
        logger.error(f"Failed to rename branch: {e}")
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@branch_app.command(name="list")
def list_branches_command(
    project: str = typer.Option(
        ...,
        "--project",
        help="Project ID (12-character hex hash)",
        metavar="PROJECT_ID",
    ),
    format: str = typer.Option(
        "table",
        "--format",
        help="Output format (table, json, simple)",
        case_sensitive=False,
    ),
    show_current: bool = typer.Option(
        False,
        "--show-current",
        help="Highlight the current Git branch if in a repository",
    ),
):
    """
    List all branches in project collection with document counts.

    This command displays all unique branch names found in the project collection's
    document metadata, along with the count of documents for each branch.

    Example:
        wqm branch list --project abc123def456
        wqm branch list --project abc123def456 --format json
        wqm branch list --project abc123def456 --show-current
    """
    asyncio.run(_list_branches(project, format, show_current))


async def _list_branches(project: str, format: str, show_current: bool):
    """
    Implementation of list branches command.

    Args:
        project: Project ID
        format: Output format (table, json, simple)
        show_current: Whether to highlight current Git branch
    """
    try:
        collection = get_project_collection(project)
        typer.echo(f"Listing branches in collection '{collection}'")

        # Initialize client
        config = get_config()
        client = QdrantWorkspaceClient()
        await client.initialize()

        # Check if collection exists
        collections = client.client.get_collections().collections
        collection_names = [col.name for col in collections]

        if collection not in collection_names:
            typer.echo(f"Error: Collection '{collection}' does not exist", err=True)
            await client.close()
            raise typer.Exit(1)

        # Get all branches with counts
        typer.echo("Analyzing branch metadata...")
        branch_counts = await get_all_branches(client, collection)

        if not branch_counts:
            typer.echo("No documents found in collection")
            await client.close()
            return

        # Get current Git branch if requested
        current_branch = None
        if show_current:
            try:
                from pathlib import Path

                current_branch = get_current_branch(Path.cwd())
                logger.debug(f"Current Git branch: {current_branch}")
            except Exception as e:
                logger.debug(f"Could not detect current Git branch: {e}")

        # Format output
        if format == "json":
            data = {
                "collection": collection,
                "project_id": project,
                "total_branches": len(branch_counts),
                "current_branch": current_branch,
                "branches": [
                    {"branch": branch, "document_count": count, "is_current": branch == current_branch}
                    for branch, count in sorted(branch_counts.items(), key=lambda x: x[1], reverse=True)
                ],
            }
            typer.echo(json_module.dumps(data, indent=2))

        elif format == "simple":
            for branch, count in sorted(
                branch_counts.items(), key=lambda x: x[1], reverse=True
            ):
                marker = " *" if show_current and branch == current_branch else ""
                typer.echo(f"{branch}: {count}{marker}")

        else:  # table format
            headers = ["Branch", "Documents"]
            if show_current:
                headers.append("Current")

            rows = []
            for branch, count in sorted(
                branch_counts.items(), key=lambda x: x[1], reverse=True
            ):
                row = [branch, count]
                if show_current:
                    row.append("✓" if branch == current_branch else "")
                rows.append(row)

            typer.echo("\n" + tabulate(rows, headers=headers, tablefmt="grid"))

            # Summary
            total_docs = sum(branch_counts.values())
            typer.echo(f"\nTotal: {len(branch_counts)} branches, {total_docs} documents")
            if show_current and current_branch:
                typer.echo(f"Current Git branch: {current_branch}")

        await client.close()

    except typer.Exit:
        raise
    except Exception as e:
        logger.error(f"Failed to list branches: {e}")
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


# Export the Typer app
__all__ = ["branch_app"]
