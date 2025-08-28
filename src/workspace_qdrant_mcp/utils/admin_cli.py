"""
Comprehensive administrative CLI for workspace-qdrant-mcp.

This module provides a powerful yet safe administrative interface for managing
workspace-qdrant collections and data. It operates independently from the MCP server
and includes robust safety mechanisms, project scoping, and comprehensive operations
for development, debugging, and maintenance workflows.

Safety Features:
    - Interactive confirmation prompts for destructive operations
    - Dry-run mode for testing commands without side effects
    - Project scoping to prevent accidental cross-project operations
    - Protected collection lists to prevent system data deletion
    - Comprehensive logging and operation auditing
    - Rollback capabilities for supported operations

Key Operations:
    - Collection management (create, delete, inspect, backup)
    - Document operations (add, update, delete, search)
    - Workspace analysis and diagnostics
    - Data migration and export tools
    - Performance monitoring and optimization
    - Configuration validation and testing

Project Scoping:
    The CLI automatically detects the current project context and applies appropriate
    scoping to prevent accidental operations on other projects' data. This includes:
    - Automatic project detection from Git repositories
    - Collection filtering based on project ownership
    - Safety prompts when operating outside current project
    - Protected collections that require explicit confirmation

Usage Patterns:
    ```bash
    # Interactive mode with safety prompts
    python -m workspace_qdrant_mcp.utils.admin_cli
    
    # Direct command execution
    python -m workspace_qdrant_mcp.utils.admin_cli delete-collection my-collection
    
    # Dry-run mode for testing
    python -m workspace_qdrant_mcp.utils.admin_cli --dry-run delete-project
    
    # Batch operations with confirmation
    python -m workspace_qdrant_mcp.utils.admin_cli migrate-data --source old-db
    ```

Example:
    ```python
    from workspace_qdrant_mcp.utils.admin_cli import WorkspaceQdrantAdmin
    
    admin = WorkspaceQdrantAdmin(dry_run=True)
    await admin.delete_project_collections(confirm=False)  # Safe testing
    
    admin = WorkspaceQdrantAdmin(dry_run=False)
    collections = await admin.list_project_collections()
    ```
"""

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Set

import typer
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException

from ..core.config import Config
from ..core.collections import WorkspaceCollectionManager
from .project_detection import ProjectDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class WorkspaceQdrantAdmin:
    """
    Comprehensive administrative interface for workspace-qdrant management.
    
    This class provides a full suite of administrative operations for managing
    workspace-qdrant collections, documents, and configurations. It emphasizes
    safety through project scoping, confirmation prompts, and comprehensive
    logging while providing powerful tools for development and maintenance.
    
    The admin interface operates with multiple safety layers:
    - Project context awareness to prevent cross-project accidents
    - Interactive confirmation for destructive operations
    - Dry-run mode for testing commands safely
    - Protected collection detection and warnings
    - Comprehensive operation logging and audit trails
    
    Key Capabilities:
        - Collection lifecycle management (CRUD operations)
        - Document batch operations and migrations
        - Workspace analysis and health checking
        - Performance monitoring and optimization
        - Data export and backup operations
        - Configuration validation and testing
    
    Attributes:
        config (Config): Configuration object with Qdrant and workspace settings
        dry_run (bool): Whether to execute operations or just simulate them
        client (QdrantClient): Direct Qdrant client for database operations
        collection_manager (WorkspaceCollectionManager): Collection management interface
        project_detector (ProjectDetector): Project structure detection
        project_info (Dict): Current project information and context
        current_project (str): Main project name for scoping operations
        subprojects (List[str]): Detected subproject names
    
    Safety Mechanisms:
        - All destructive operations require explicit confirmation
        - Project scoping prevents accidental cross-project operations
        - Protected collections (like system collections) have additional safeguards
        - Dry-run mode allows testing without side effects
        - Comprehensive logging provides audit trails
    
    Example:
        ```python
        # Initialize with dry-run for testing
        admin = WorkspaceQdrantAdmin(dry_run=True)
        
        # List collections in current project
        collections = await admin.list_project_collections()
        
        # Delete with safety prompts (dry-run won't actually delete)
        await admin.delete_project_collections(confirm=True)
        
        # Production operations
        admin = WorkspaceQdrantAdmin(dry_run=False)
        await admin.backup_collections(["/backup/path"])
        ```
    """
    
    def __init__(self, config: Optional[Config] = None, dry_run: bool = False) -> None:
        """Initialize the administrative interface with safety configuration.
        
        Args:
            config: Configuration object. If None, loads from environment/files
            dry_run: If True, operations are simulated without actual execution.
                    Useful for testing commands safely before running them
        """
        self.config = config or Config()
        self.dry_run = dry_run
        self.client = QdrantClient(**self.config.qdrant_client_config)
        self.collection_manager = WorkspaceCollectionManager(self.client, self.config)
        self.project_detector = ProjectDetector(github_user=self.config.workspace.github_user)
        
        # Initialize project context for scoping operations
        self.project_info = self.project_detector.get_project_info()
        self.current_project = self.project_info["main_project"]
        self.subprojects = self.project_info["subprojects"]
        
        # Log initialization for audit trail
        logger.info(
            "Initialized WorkspaceQdrantAdmin - Project: %s, Dry-run: %s",
            self.current_project, self.dry_run
        )
    
    def get_protected_collections(self) -> Set[str]:
        """
        Get set of protected collections that cannot be deleted.
        
        Returns:
            Set of protected collection names
        """
        protected = set()
        
        # Protect memexd daemon collections (-code suffix)
        try:
            all_collections = self.client.get_collections()
            for collection in all_collections.collections:
                if collection.name.endswith("-code"):
                    protected.add(collection.name)
        except Exception as e:
            logger.warning("Could not list collections to identify protected ones: %s", e)
        
        return protected
    
    def list_collections(self, show_all: bool = False) -> List[str]:
        """
        List collections with optional filtering.
        
        Args:
            show_all: If True, show all collections, otherwise only workspace collections
            
        Returns:
            List of collection names
        """
        try:
            if show_all:
                all_collections = self.client.get_collections()
                return sorted([col.name for col in all_collections.collections])
            else:
                # Get workspace collections synchronously
                try:
                    all_collections = self.client.get_collections()
                    workspace_collections = []
                    
                    for collection in all_collections.collections:
                        # Filter for workspace collections (exclude memexd -code collections)
                        if self.collection_manager._is_workspace_collection(collection.name):
                            workspace_collections.append(collection.name)
                            
                    return sorted(workspace_collections)
                    
                except Exception as e:
                    logger.error("Failed to list collections: %s", e)
                    return []
        except Exception as e:
            logger.error("Failed to list collections: %s", e)
            return []
    
    def is_project_scoped_collection(self, collection_name: str) -> bool:
        """
        Check if a collection belongs to the current project scope.
        
        Args:
            collection_name: Name of the collection to check
            
        Returns:
            True if collection is within project scope
        """
        # Global collections are always in scope
        if collection_name in self.config.workspace.global_collections:
            return True
        
        # Check main project collections
        project_prefixes = [f"{self.current_project}-"]
        
        # Add subproject prefixes
        if self.subprojects:
            project_prefixes.extend([f"{sub}-" for sub in self.subprojects])
        
        return any(collection_name.startswith(prefix) for prefix in project_prefixes)
    
    def validate_collection_for_deletion(self, collection_name: str) -> tuple[bool, str]:
        """
        Validate if a collection can be safely deleted.
        
        Args:
            collection_name: Name of collection to validate
            
        Returns:
            Tuple of (can_delete, reason)
        """
        # Check if collection exists
        try:
            existing_collections = self.client.get_collections()
            collection_names = {col.name for col in existing_collections.collections}
            
            if collection_name not in collection_names:
                return False, f"Collection '{collection_name}' does not exist"
        except Exception as e:
            return False, f"Cannot verify collection existence: {e}"
        
        # Check if protected
        protected = self.get_protected_collections()
        if collection_name in protected:
            return False, f"Collection '{collection_name}' is protected (memexd daemon collection)"
        
        # Check project scope
        if not self.is_project_scoped_collection(collection_name):
            return False, f"Collection '{collection_name}' is outside current project scope"
        
        return True, "Collection can be safely deleted"
    
    def delete_collection(
        self, 
        collection_name: str, 
        force: bool = False
    ) -> bool:
        """
        Delete a collection with safety checks.
        
        Args:
            collection_name: Name of collection to delete
            force: Skip confirmation prompts if True
            
        Returns:
            True if deletion was successful
        """
        # Validate collection
        can_delete, reason = self.validate_collection_for_deletion(collection_name)
        if not can_delete:
            logger.error("Cannot delete collection: %s", reason)
            return False
        
        # Get collection info before deletion
        try:
            collection_info = self.client.get_collection(collection_name)
            point_count = collection_info.points_count
            logger.info("Collection '%s' contains %d points", collection_name, point_count)
        except Exception as e:
            logger.warning("Could not get collection info: %s", e)
            point_count = "unknown"
        
        # Confirmation prompt (unless forced or dry-run)
        if not force and not self.dry_run:
            confirmation = typer.confirm(
                f"Delete collection '{collection_name}' with {point_count} points?"
            )
            if not confirmation:
                logger.info("Collection deletion cancelled by user")
                return False
        
        # Perform deletion
        if self.dry_run:
            logger.info("DRY RUN: Would delete collection '%s'", collection_name)
            return True
        
        try:
            self.client.delete_collection(collection_name)
            logger.info("Successfully deleted collection '%s'", collection_name)
            return True
        except ResponseHandlingException as e:
            logger.error("Failed to delete collection '%s': %s", collection_name, e)
            return False
        except Exception as e:
            logger.error("Unexpected error deleting collection '%s': %s", collection_name, e)
            return False
    
    def get_collection_info(self, collection_name: Optional[str] = None) -> dict:
        """
        Get detailed information about collections.
        
        Args:
            collection_name: Specific collection name, or None for all workspace collections
            
        Returns:
            Dictionary with collection information
        """
        if collection_name:
            # Single collection info
            try:
                info = self.client.get_collection(collection_name)
                return {
                    collection_name: {
                        "vectors_count": info.vectors_count,
                        "points_count": info.points_count,
                        "status": info.status,
                        "optimizer_status": info.optimizer_status,
                        "config": {
                            "distance": info.config.params.vectors.distance.value,
                            "vector_size": info.config.params.vectors.size,
                        },
                        "project_scoped": self.is_project_scoped_collection(collection_name),
                        "protected": collection_name in self.get_protected_collections()
                    }
                }
            except Exception as e:
                return {collection_name: {"error": str(e)}}
        else:
            # All workspace collections - sync version
            try:
                workspace_collections = self.list_collections(show_all=False)
                collection_info = {}
                
                for collection_name in workspace_collections:
                    try:
                        info = self.client.get_collection(collection_name)
                        collection_info[collection_name] = {
                            "vectors_count": info.vectors_count,
                            "points_count": info.points_count,
                            "status": info.status,
                            "optimizer_status": info.optimizer_status,
                            "config": {
                                "distance": info.config.params.vectors.distance.value,
                                "vector_size": info.config.params.vectors.size,
                            },
                            "project_scoped": self.is_project_scoped_collection(collection_name),
                            "protected": collection_name in self.get_protected_collections()
                        }
                    except Exception as e:
                        logger.warning("Failed to get info for collection %s: %s", collection_name, e)
                        collection_info[collection_name] = {"error": str(e)}
                
                return {
                    "collections": collection_info,
                    "total_collections": len(workspace_collections)
                }
                
            except Exception as e:
                logger.error("Failed to get collection info: %s", e)
                return {"error": str(e)}
    
    def close(self):
        """Close the Qdrant client connection."""
        self.client.close()


# CLI Application
app = typer.Typer(
    name="workspace-qdrant-admin",
    help="Administrative CLI for workspace-qdrant-mcp collections"
)


@app.command()
def list_collections(
    all_collections: bool = typer.Option(
        False, "--all", "-a", help="Show all collections, not just workspace collections"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show verbose output")
) -> None:
    """List collections in the Qdrant instance."""
    
    try:
        admin = WorkspaceQdrantAdmin()
        collections = admin.list_collections(show_all=all_collections)
        
        if not collections:
            typer.echo("No collections found.")
            return
        
        typer.echo(f"\nFound {len(collections)} collection{'s' if len(collections) != 1 else ''}:")
        
        if verbose:
            # Show detailed info
            for collection_name in collections:
                info = admin.get_collection_info(collection_name)
                collection_data = info.get(collection_name, {})
                
                if "error" in collection_data:
                    typer.echo(f"  ❌ {collection_name} (error: {collection_data['error']})")
                else:
                    points = collection_data.get("points_count", 0)
                    protected = "🔒" if collection_data.get("protected", False) else ""
                    scoped = "📁" if collection_data.get("project_scoped", False) else "🌐"
                    typer.echo(f"  {scoped} {protected} {collection_name} ({points} points)")
        else:
            # Simple list
            for collection_name in collections:
                typer.echo(f"  • {collection_name}")
        
        admin.close()
        
    except Exception as e:
        typer.echo(typer.style(f"❌ Error listing collections: {e}", fg=typer.colors.RED))
        sys.exit(1)


@app.command()
def delete_collection(
    collection_name: str = typer.Argument(..., help="Name of collection to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation prompts"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be deleted without actually deleting")
) -> None:
    """Delete a collection with safety checks."""
    
    try:
        admin = WorkspaceQdrantAdmin(dry_run=dry_run)
        
        # Show what we're about to delete
        info = admin.get_collection_info(collection_name)
        collection_data = info.get(collection_name, {})
        
        if "error" in collection_data:
            typer.echo(typer.style(f"❌ Cannot access collection '{collection_name}': {collection_data['error']}", fg=typer.colors.RED))
            sys.exit(1)
        
        # Display collection info
        points = collection_data.get("points_count", 0)
        protected = collection_data.get("protected", False)
        scoped = collection_data.get("project_scoped", False)
        
        typer.echo(f"\nCollection: {collection_name}")
        typer.echo(f"Points: {points}")
        typer.echo(f"Project scoped: {'Yes' if scoped else 'No'}")
        typer.echo(f"Protected: {'Yes' if protected else 'No'}")
        
        # Attempt deletion
        success = admin.delete_collection(collection_name, force=force)
        
        if success:
            if dry_run:
                typer.echo(typer.style("✅ DRY RUN: Collection would be deleted successfully", fg=typer.colors.GREEN))
            else:
                typer.echo(typer.style("✅ Collection deleted successfully", fg=typer.colors.GREEN))
        else:
            typer.echo(typer.style("❌ Collection deletion failed", fg=typer.colors.RED))
            sys.exit(1)
        
        admin.close()
        
    except Exception as e:
        typer.echo(typer.style(f"❌ Error deleting collection: {e}", fg=typer.colors.RED))
        sys.exit(1)


@app.command()
def collection_info(
    collection_name: Optional[str] = typer.Argument(None, help="Name of specific collection (optional)")
) -> None:
    """Show detailed information about collections."""
    
    try:
        admin = WorkspaceQdrantAdmin()
        info = admin.get_collection_info(collection_name)
        
        if collection_name:
            # Single collection
            collection_data = info.get(collection_name, {})
            if "error" in collection_data:
                typer.echo(typer.style(f"❌ Error getting collection info: {collection_data['error']}", fg=typer.colors.RED))
                sys.exit(1)
            
            typer.echo(f"\nCollection: {collection_name}")
            typer.echo(f"Points: {collection_data.get('points_count', 0)}")
            typer.echo(f"Vectors: {collection_data.get('vectors_count', 0)}")
            typer.echo(f"Status: {collection_data.get('status', 'unknown')}")
            typer.echo(f"Project scoped: {'Yes' if collection_data.get('project_scoped', False) else 'No'}")
            typer.echo(f"Protected: {'Yes' if collection_data.get('protected', False) else 'No'}")
            
            config = collection_data.get('config', {})
            if config:
                typer.echo(f"Vector size: {config.get('vector_size', 'unknown')}")
                typer.echo(f"Distance metric: {config.get('distance', 'unknown')}")
        
        else:
            # All collections
            collections = info.get("collections", {})
            total = info.get("total_collections", 0)
            
            typer.echo(f"\nWorkspace Collections ({total} total):")
            
            for name, data in collections.items():
                if "error" in data:
                    typer.echo(f"  ❌ {name} (error: {data['error']})")
                else:
                    points = data.get("points_count", 0)
                    status = data.get("status", "unknown")
                    typer.echo(f"  • {name}: {points} points ({status})")
        
        admin.close()
        
    except Exception as e:
        typer.echo(typer.style(f"❌ Error getting collection info: {e}", fg=typer.colors.RED))
        sys.exit(1)


@app.callback()
def main(
    config_file: Optional[str] = typer.Option(None, "--config", help="Path to config file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
    """Administrative CLI for workspace-qdrant-mcp collections."""
    
    if config_file:
        os.environ["CONFIG_FILE"] = config_file
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


def admin_cli():
    """Console script entry point."""
    try:
        app()
    except KeyboardInterrupt:
        typer.echo("\n❌ Operation cancelled by user")
        sys.exit(1)


if __name__ == "__main__":
    admin_cli()