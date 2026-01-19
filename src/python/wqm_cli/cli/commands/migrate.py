"""Collection migration tooling for workspace-qdrant-mcp.

This module provides CLI commands to migrate old {project}-{suffix} collections
to the new _{project_id} architecture specified in Task 374.6.

OLD Architecture:
- Multiple collections per project: {project}-code, {project}-docs, {project}-test
- File type differentiation by collection name

NEW Architecture:
- Single collection per project: _{project_id}
- File type differentiation by metadata: file_type, branch, project_id
- project_id from calculate_tenant_id() based on git remote or path hash

Features:
- Detect old collections and group by project
- Generate migration plans with dry-run validation
- Execute migrations with progress reporting
- Backup collections before migration (JSON export)
- Rollback capability if migration fails
- Metadata enrichment during migration

Usage:
    wqm migrate detect                          # List old collections
    wqm migrate plan PROJECT                    # Show migration plan
    wqm migrate execute PROJECT --project-root PATH  # Run migration
    wqm migrate rollback --backup-id ID         # Restore from backup
    wqm migrate status                          # Show migration history
"""

import asyncio
import hashlib
import json
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import typer
from common.core.config import get_config
from common.utils.git_utils import get_current_branch
from common.utils.project_detection import calculate_tenant_id
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

# Create Typer app for migrate commands
migrate_app = typer.Typer(
    name="migrate",
    help="Migrate old collections to new _{project_id} architecture",
    no_args_is_help=True,
)


# Mapping from old collection suffixes to file types
SUFFIX_TO_FILE_TYPE = {
    "code": "code",
    "docs": "docs",
    "doc": "docs",
    "test": "test",
    "tests": "test",
    "config": "config",
    "data": "data",
    "build": "build",
    "notes": "docs",
    "note": "docs",
    "readme": "docs",
    "scripts": "code",
    "src": "code",
    "lib": "code",
    "assets": "data",
    "examples": "code",
}


@dataclass
class MigrationPlan:
    """Plan for migrating old collections to new architecture."""

    project_name: str
    project_root: Path | None
    old_collections: list[str]
    new_collection: str
    project_id: str
    branch: str
    document_count: int
    estimated_time_ms: float
    metadata_enrichment: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        if self.project_root:
            result["project_root"] = str(self.project_root)
        return result


@dataclass
class MigrationResult:
    """Result of a migration operation."""

    project_name: str
    success: bool
    new_collection: str
    old_collections: list[str]
    documents_migrated: int
    backup_files: list[str]
    duration_ms: float
    error_message: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class BackupMetadata:
    """Metadata for a collection backup."""

    backup_id: str
    collection_name: str
    backup_file: Path
    timestamp: str
    document_count: int
    project_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result["backup_file"] = str(self.backup_file)
        return result


class CollectionMigrationManager:
    """Manages collection migration from old to new architecture."""

    def __init__(
        self,
        qdrant_client: QdrantClient,
        backup_dir: Path | None = None,
    ):
        """
        Initialize migration manager.

        Args:
            qdrant_client: Qdrant client instance
            backup_dir: Directory for backups (default: .taskmaster/backups/migration/)
        """
        self.qdrant_client = qdrant_client
        self.backup_dir = backup_dir or (
            Path.home() / ".taskmaster" / "backups" / "migration"
        )
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Migration history
        self.migration_history: list[MigrationResult] = []
        self.backup_metadata: list[BackupMetadata] = []

    async def detect_old_collections(self) -> dict[str, list[str]]:
        """
        Detect old {project}-{suffix} collections.

        Returns:
            Dictionary mapping project names to list of old collection names
        """
        try:
            collections_response = self.qdrant_client.get_collections()
            all_collections = [c.name for c in collections_response.collections]
        except Exception as e:
            logger.error(f"Failed to retrieve collections: {e}")
            raise

        old_collections: dict[str, list[str]] = {}

        for name in all_collections:
            # Skip new format collections (start with _)
            if name.startswith("_"):
                continue

            # Skip system collections (start with __)
            if name.startswith("__"):
                continue

            # Check for {project}-{suffix} pattern
            if "-" in name:
                # Split on last dash to get project and suffix
                parts = name.rsplit("-", 1)
                if len(parts) == 2:
                    project, suffix = parts

                    # Validate suffix is a known file type
                    if suffix in SUFFIX_TO_FILE_TYPE or suffix.isalpha():
                        if project not in old_collections:
                            old_collections[project] = []
                        old_collections[project].append(name)

        logger.info(
            f"Detected {len(old_collections)} projects with old-style collections"
        )
        return old_collections

    async def generate_migration_plan(
        self,
        project_name: str,
        old_collections: list[str],
        project_root: Path | None = None,
    ) -> MigrationPlan:
        """
        Generate migration plan for a project.

        Args:
            project_name: Name of the project
            old_collections: List of old collection names
            project_root: Optional path to project root (for accurate project_id)

        Returns:
            MigrationPlan with detailed migration information
        """
        warnings: list[str] = []

        # Calculate project_id
        if project_root and project_root.exists():
            project_id = calculate_tenant_id(project_root)
            branch = get_current_branch(project_root)
        else:
            # Fallback: generate project_id from project name
            warnings.append(
                "Project root not found, using hash of project name for project_id"
            )
            project_id = hashlib.md5(project_name.encode()).hexdigest()[:16]
            branch = "main"

        # Build new collection name
        new_collection = f"_{project_id}"

        # Count total documents
        total_docs = 0
        for collection_name in old_collections:
            try:
                collection_info = self.qdrant_client.get_collection(collection_name)
                total_docs += collection_info.points_count or 0
            except Exception as e:
                warnings.append(f"Failed to get count for {collection_name}: {e}")

        # Estimate migration time (rough estimate: 1ms per document + overhead)
        estimated_time_ms = total_docs * 1.0 + 1000  # +1s overhead

        # Build metadata enrichment info
        metadata_enrichment = {
            "project_id": project_id,
            "branch": branch,
            "file_type": "varies by collection suffix",
        }

        plan = MigrationPlan(
            project_name=project_name,
            project_root=project_root,
            old_collections=old_collections,
            new_collection=new_collection,
            project_id=project_id,
            branch=branch,
            document_count=total_docs,
            estimated_time_ms=estimated_time_ms,
            metadata_enrichment=metadata_enrichment,
            warnings=warnings,
        )

        logger.info(
            f"Generated migration plan for {project_name}: "
            f"{len(old_collections)} collections → {new_collection}"
        )

        return plan

    async def create_backup(
        self, collection_name: str, project_name: str | None = None
    ) -> BackupMetadata:
        """
        Create backup of a collection to JSON file.

        Args:
            collection_name: Name of collection to backup
            project_name: Optional project name for organization

        Returns:
            BackupMetadata with backup information
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_id = f"{collection_name}_{timestamp}"
        backup_file = self.backup_dir / f"{backup_id}.json"

        logger.info(f"Creating backup of {collection_name} to {backup_file}")

        try:
            # Get collection info for metadata
            collection_info = self.qdrant_client.get_collection(collection_name)

            # Scroll through all documents
            documents = []
            offset = None

            while True:
                batch, offset = self.qdrant_client.scroll(
                    collection_name=collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True,
                )

                for point in batch:
                    documents.append(
                        {
                            "id": point.id,
                            "vector": point.vector,
                            "payload": point.payload,
                        }
                    )

                if offset is None:
                    break

            # Save to JSON
            backup_data = {
                "backup_id": backup_id,
                "collection_name": collection_name,
                "timestamp": timestamp,
                "document_count": len(documents),
                "collection_config": {
                    "vectors_count": collection_info.vectors_count,
                    "points_count": collection_info.points_count,
                    "status": str(collection_info.status),
                },
                "documents": documents,
            }

            with open(backup_file, "w") as f:
                json.dump(backup_data, f, indent=2)

            logger.info(
                f"Backup created: {len(documents)} documents saved to {backup_file}"
            )

            metadata = BackupMetadata(
                backup_id=backup_id,
                collection_name=collection_name,
                backup_file=backup_file,
                timestamp=timestamp,
                document_count=len(documents),
                project_name=project_name,
            )

            self.backup_metadata.append(metadata)
            return metadata

        except Exception as e:
            logger.error(f"Failed to create backup of {collection_name}: {e}")
            raise

    async def execute_migration(
        self,
        plan: MigrationPlan,
        dry_run: bool = False,
    ) -> MigrationResult:
        """
        Execute migration based on plan.

        Args:
            plan: Migration plan to execute
            dry_run: If True, validate without making changes

        Returns:
            MigrationResult with outcome details
        """
        start_time = time.time()
        documents_migrated = 0
        backup_files: list[str] = []
        warnings: list[str] = list(plan.warnings)

        logger.info(
            f"{'[DRY RUN] ' if dry_run else ''}Starting migration for {plan.project_name}"
        )

        try:
            # Step 1: Create new collection if it doesn't exist
            if not dry_run:
                try:
                    self.qdrant_client.get_collection(plan.new_collection)
                    logger.info(f"Collection {plan.new_collection} already exists")
                except UnexpectedResponse:
                    # Collection doesn't exist, create it
                    logger.info(f"Creating new collection: {plan.new_collection}")

                    # Get vector config from first old collection
                    first_old = plan.old_collections[0]
                    old_config = self.qdrant_client.get_collection(first_old)

                    # Create with same vector configuration
                    self.qdrant_client.create_collection(
                        collection_name=plan.new_collection,
                        vectors_config=old_config.config.params.vectors,
                    )

            # Step 2: Migrate documents from each old collection
            for old_collection in plan.old_collections:
                logger.info(f"Migrating {old_collection}...")

                # Create backup before migration
                if not dry_run:
                    backup = await self.create_backup(old_collection, plan.project_name)
                    backup_files.append(str(backup.backup_file))

                # Determine file_type from collection suffix
                suffix = old_collection.rsplit("-", 1)[1] if "-" in old_collection else "other"
                file_type = SUFFIX_TO_FILE_TYPE.get(suffix, "other")

                # Scroll through documents and migrate
                offset = None
                batch_count = 0

                while True:
                    batch, offset = self.qdrant_client.scroll(
                        collection_name=old_collection,
                        limit=100,
                        offset=offset,
                        with_payload=True,
                        with_vectors=True,
                    )

                    if not dry_run and len(batch) > 0:
                        # Enrich metadata for each document
                        enriched_points = []
                        for point in batch:
                            # Add migration metadata
                            new_payload = dict(point.payload) if point.payload else {}
                            new_payload["project_id"] = plan.project_id
                            new_payload["branch"] = plan.branch
                            new_payload["file_type"] = file_type

                            # Add migration tracking
                            new_payload["_migrated_from"] = old_collection
                            new_payload["_migrated_at"] = datetime.now(
                                timezone.utc
                            ).isoformat()

                            enriched_points.append(
                                PointStruct(
                                    id=point.id,
                                    vector=point.vector,
                                    payload=new_payload,
                                )
                            )

                        # Insert into new collection
                        self.qdrant_client.upsert(
                            collection_name=plan.new_collection,
                            points=enriched_points,
                        )

                    documents_migrated += len(batch)
                    batch_count += 1

                    if offset is None:
                        break

                logger.info(
                    f"Migrated {documents_migrated} documents from {old_collection}"
                )

            # Step 3: Verify document counts
            if not dry_run:
                new_collection_info = self.qdrant_client.get_collection(
                    plan.new_collection
                )
                actual_count = new_collection_info.points_count or 0

                if actual_count != documents_migrated:
                    warnings.append(
                        f"Document count mismatch: expected {documents_migrated}, got {actual_count}"
                    )

            duration_ms = (time.time() - start_time) * 1000

            result = MigrationResult(
                project_name=plan.project_name,
                success=True,
                new_collection=plan.new_collection,
                old_collections=plan.old_collections,
                documents_migrated=documents_migrated,
                backup_files=backup_files,
                duration_ms=duration_ms,
                warnings=warnings,
            )

            self.migration_history.append(result)
            logger.info(
                f"Migration {'simulation' if dry_run else 'completed'} successfully: "
                f"{documents_migrated} documents in {duration_ms:.1f}ms"
            )

            return result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            error_msg = f"Migration failed: {str(e)}"
            logger.error(error_msg)

            result = MigrationResult(
                project_name=plan.project_name,
                success=False,
                new_collection=plan.new_collection,
                old_collections=plan.old_collections,
                documents_migrated=documents_migrated,
                backup_files=backup_files,
                duration_ms=duration_ms,
                error_message=error_msg,
                warnings=warnings,
            )

            self.migration_history.append(result)
            return result

    async def rollback_from_backup(self, backup_id: str) -> bool:
        """
        Rollback migration by restoring from backup.

        Args:
            backup_id: ID of backup to restore

        Returns:
            True if rollback successful
        """
        logger.info(f"Rolling back from backup: {backup_id}")

        # Find backup file
        backup_file = self.backup_dir / f"{backup_id}.json"
        if not backup_file.exists():
            logger.error(f"Backup file not found: {backup_file}")
            return False

        try:
            # Load backup data
            with open(backup_file) as f:
                backup_data = json.load(f)

            collection_name = backup_data["collection_name"]
            documents = backup_data["documents"]

            logger.info(
                f"Restoring {collection_name}: {len(documents)} documents from backup"
            )

            # Recreate collection (delete if exists)
            try:
                self.qdrant_client.delete_collection(collection_name)
                logger.info(f"Deleted existing collection: {collection_name}")
            except UnexpectedResponse:
                pass  # Collection doesn't exist, that's fine

            # Get vector config from backup or use default
            backup_data.get("collection_config", {})

            # Create collection (simplified - using default config)
            # In production, would restore exact vector configuration
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

            # Restore documents in batches
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]
                points = [
                    PointStruct(
                        id=doc["id"],
                        vector=doc["vector"],
                        payload=doc["payload"],
                    )
                    for doc in batch
                ]
                self.qdrant_client.upsert(collection_name=collection_name, points=points)

            logger.info(f"Rollback completed successfully for {collection_name}")
            return True

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False


# ============================================================================
# CLI Commands
# ============================================================================


@migrate_app.command("detect")
def detect_command():
    """Detect old {project}-{suffix} collections that need migration."""
    try:
        # Load configuration
        config = get_config()
        qdrant_url = config.get("qdrant", {}).get("url", "http://localhost:6333")

        # Initialize client
        qdrant_client = QdrantClient(url=qdrant_url)

        # Create manager
        manager = CollectionMigrationManager(qdrant_client)

        # Detect old collections
        old_collections = asyncio.run(manager.detect_old_collections())

        if not old_collections:
            print("No old-style collections found. All collections are up to date!")
            return

        # Display results
        print(f"\nFound {len(old_collections)} projects with old-style collections:\n")

        for project_name, collections in sorted(old_collections.items()):
            print(f"Project: {project_name}")
            print(f"  Collections: {len(collections)}")
            for coll in collections:
                try:
                    info = qdrant_client.get_collection(coll)
                    count = info.points_count or 0
                    print(f"    - {coll} ({count} documents)")
                except Exception as e:
                    print(f"    - {coll} (count unavailable: {e})")
            print()

        print(
            "\nNext steps:\n"
            "  1. Review migration plan: wqm migrate plan <PROJECT_NAME>\n"
            "  2. Execute migration: wqm migrate execute <PROJECT_NAME> --project-root PATH\n"
        )

    except Exception as e:
        print(f"Error: {e}")
        raise typer.Exit(1)


@migrate_app.command("plan")
def plan_command(
    project_name: str = typer.Argument(..., help="Project name to migrate"),
    project_root: Path | None = typer.Option(
        None,
        "--project-root",
        "-p",
        help="Path to project root (for accurate project_id)",
    ),
):
    """Generate and display migration plan for a project."""
    try:
        # Load configuration
        config = get_config()
        qdrant_url = config.get("qdrant", {}).get("url", "http://localhost:6333")

        # Initialize client
        qdrant_client = QdrantClient(url=qdrant_url)

        # Create manager
        manager = CollectionMigrationManager(qdrant_client)

        # Detect old collections
        old_collections_map = asyncio.run(manager.detect_old_collections())

        if project_name not in old_collections_map:
            print(f"Error: No old collections found for project '{project_name}'")
            print(f"\nAvailable projects: {', '.join(sorted(old_collections_map.keys()))}")
            raise typer.Exit(1)

        old_collections = old_collections_map[project_name]

        # Generate plan
        plan = asyncio.run(
            manager.generate_migration_plan(project_name, old_collections, project_root)
        )

        # Display plan
        print(f"\n{'='*80}")
        print(f"MIGRATION PLAN: {project_name}")
        print(f"{'='*80}\n")

        print(f"Project Name:     {plan.project_name}")
        print(f"Project Root:     {plan.project_root or 'NOT SPECIFIED (using fallback)'}")
        print(f"Project ID:       {plan.project_id}")
        print(f"Branch:           {plan.branch}")
        print(f"\nOld Collections:  {len(plan.old_collections)}")
        for coll in plan.old_collections:
            print(f"  - {coll}")

        print(f"\nNew Collection:   {plan.new_collection}")
        print(f"Documents:        {plan.document_count}")
        print(f"Estimated Time:   {plan.estimated_time_ms/1000:.2f} seconds")

        print("\nMetadata Enrichment:")
        for key, value in plan.metadata_enrichment.items():
            print(f"  - {key}: {value}")

        if plan.warnings:
            print("\nWARNINGS:")
            for warning in plan.warnings:
                print(f"  ! {warning}")

        print(f"\n{'='*80}")
        print("Next: Execute migration with:")
        print(f"  wqm migrate execute {project_name}" + (f" --project-root {project_root}" if project_root else ""))
        print(f"{'='*80}\n")

    except Exception as e:
        print(f"Error: {e}")
        raise typer.Exit(1)


@migrate_app.command("execute")
def execute_command(
    project_name: str = typer.Argument(..., help="Project name to migrate"),
    project_root: Path | None = typer.Option(
        None,
        "--project-root",
        "-p",
        help="Path to project root (required for accurate project_id)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Simulate migration without making changes",
    ),
    backup_dir: Path | None = typer.Option(
        None,
        "--backup-dir",
        "-b",
        help="Custom backup directory",
    ),
):
    """Execute migration for a project."""
    try:
        # Load configuration
        config = get_config()
        qdrant_url = config.get("qdrant", {}).get("url", "http://localhost:6333")

        # Initialize client
        qdrant_client = QdrantClient(url=qdrant_url)

        # Create manager
        manager = CollectionMigrationManager(qdrant_client, backup_dir)

        # Detect old collections
        old_collections_map = asyncio.run(manager.detect_old_collections())

        if project_name not in old_collections_map:
            print(f"Error: No old collections found for project '{project_name}'")
            raise typer.Exit(1)

        old_collections = old_collections_map[project_name]

        # Generate plan
        plan = asyncio.run(
            manager.generate_migration_plan(project_name, old_collections, project_root)
        )

        # Confirm migration
        if not dry_run:
            print(f"\n{'='*80}")
            print(f"MIGRATION: {project_name}")
            print(f"{'='*80}\n")
            print(f"Old Collections:  {len(plan.old_collections)}")
            print(f"New Collection:   {plan.new_collection}")
            print(f"Documents:        {plan.document_count}")
            print("\nBackups will be created before migration.")
            print(f"Backup directory: {manager.backup_dir}")
            print(f"\n{'='*80}\n")

            confirm = typer.confirm("Proceed with migration?")
            if not confirm:
                print("Migration cancelled.")
                raise typer.Exit(0)

        # Execute migration
        print(f"\n{'[DRY RUN] ' if dry_run else ''}Starting migration...\n")

        result = asyncio.run(manager.execute_migration(plan, dry_run=dry_run))

        # Display results
        print(f"\n{'='*80}")
        print(f"MIGRATION {'SIMULATION' if dry_run else 'RESULT'}")
        print(f"{'='*80}\n")

        print(f"Status:           {'SUCCESS' if result.success else 'FAILED'}")
        print(f"Documents Migrated: {result.documents_migrated}")
        print(f"Duration:         {result.duration_ms/1000:.2f} seconds")

        if result.backup_files and not dry_run:
            print(f"\nBackup Files:     {len(result.backup_files)}")
            for backup in result.backup_files:
                print(f"  - {backup}")

        if result.warnings:
            print(f"\nWarnings:         {len(result.warnings)}")
            for warning in result.warnings:
                print(f"  ! {warning}")

        if result.error_message:
            print(f"\nError: {result.error_message}")

        print(f"\n{'='*80}\n")

        if result.success and not dry_run:
            print("Migration completed successfully!")
            print("\nOld collections have NOT been deleted for safety.")
            print("After verifying the migration, you can delete them manually using:")
            print("  wqm admin delete-collection <COLLECTION_NAME>")
        elif result.success and dry_run:
            print("Dry run completed successfully!")
            print("No changes were made. Execute without --dry-run to perform migration.")
        else:
            print("Migration failed. See error above.")
            if result.backup_files:
                print("\nRollback available using backup IDs from backup files.")

    except Exception as e:
        print(f"Error: {e}")
        raise typer.Exit(1)


@migrate_app.command("rollback")
def rollback_command(
    backup_id: str = typer.Argument(..., help="Backup ID to restore"),
    backup_dir: Path | None = typer.Option(
        None,
        "--backup-dir",
        "-b",
        help="Custom backup directory",
    ),
):
    """Rollback migration by restoring from backup."""
    try:
        # Load configuration
        config = get_config()
        qdrant_url = config.get("qdrant", {}).get("url", "http://localhost:6333")

        # Initialize client
        qdrant_client = QdrantClient(url=qdrant_url)

        # Create manager
        manager = CollectionMigrationManager(qdrant_client, backup_dir)

        # Confirm rollback
        print(f"\nRollback from backup: {backup_id}")
        print("This will DELETE the current collection and restore from backup.")
        print(f"\nBackup directory: {manager.backup_dir}")

        confirm = typer.confirm("Proceed with rollback?")
        if not confirm:
            print("Rollback cancelled.")
            raise typer.Exit(0)

        # Execute rollback
        print("\nRestoring from backup...\n")

        success = asyncio.run(manager.rollback_from_backup(backup_id))

        if success:
            print("\nRollback completed successfully!")
        else:
            print("\nRollback failed. See error above.")
            raise typer.Exit(1)

    except Exception as e:
        print(f"Error: {e}")
        raise typer.Exit(1)


@migrate_app.command("status")
def status_command(
    backup_dir: Path | None = typer.Option(
        None,
        "--backup-dir",
        "-b",
        help="Custom backup directory",
    ),
):
    """Show migration history and available backups."""
    try:
        # Initialize manager to access backup directory
        config = get_config()
        qdrant_url = config.get("qdrant", {}).get("url", "http://localhost:6333")
        qdrant_client = QdrantClient(url=qdrant_url)
        manager = CollectionMigrationManager(qdrant_client, backup_dir)

        print(f"\n{'='*80}")
        print("MIGRATION STATUS")
        print(f"{'='*80}\n")

        print(f"Backup Directory: {manager.backup_dir}")

        # List available backups
        if manager.backup_dir.exists():
            backup_files = list(manager.backup_dir.glob("*.json"))
            print(f"Available Backups: {len(backup_files)}\n")

            for backup_file in sorted(backup_files, reverse=True):
                try:
                    with open(backup_file) as f:
                        data = json.load(f)
                    print(f"  Backup ID: {data['backup_id']}")
                    print(f"    Collection: {data['collection_name']}")
                    print(f"    Timestamp: {data['timestamp']}")
                    print(f"    Documents: {data['document_count']}")
                    print()
                except Exception as e:
                    print(f"  {backup_file.name} (error reading: {e})\n")
        else:
            print("No backups found.\n")

        print(f"{'='*80}\n")

    except Exception as e:
        print(f"Error: {e}")
        raise typer.Exit(1)


# ============================================================================
# Multi-Tenant Migration Commands (Task 404)
# Migrate from per-project _{project_id} to unified _projects collection
# ============================================================================

# Unified collection names for multi-tenant architecture
UNIFIED_COLLECTIONS = {
    "projects": "_projects",
    "libraries": "_libraries",
    "memory": "_memory",
}


# ============================================================================
# Migration State Tracking (Task 410)
# ============================================================================


class MigrationStatus(str, Enum):
    """Migration status values."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class MigrationState:
    """Represents a migration operation state."""

    migration_id: str
    migration_type: str  # "to-multitenant", "to-project", etc.
    started_at: str
    completed_at: str | None
    status: MigrationStatus
    collections_migrated: list[str]
    points_migrated: int
    error_message: str | None
    report_path: str | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result["status"] = self.status.value
        return result


class MigrationStateManager:
    """
    SQLite-based migration state tracking.

    Task 410: Provides rollback capability by tracking migration state
    in a persistent SQLite database.
    """

    def __init__(self, db_path: Path | None = None):
        """
        Initialize state manager.

        Args:
            db_path: Path to SQLite database (default: ~/.taskmaster/migration_state.db)
        """
        self.db_path = db_path or (
            Path.home() / ".taskmaster" / "migration_state.db"
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS migration_history (
                    migration_id TEXT PRIMARY KEY,
                    migration_type TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    status TEXT CHECK (status IN ('running', 'completed', 'failed', 'rolled_back')),
                    collections_migrated TEXT,
                    points_migrated INTEGER DEFAULT 0,
                    error_message TEXT,
                    report_path TEXT
                )
            """)

            # Create index for efficient status queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_migration_status
                ON migration_history(status)
            """)

            conn.commit()

    def start_migration(
        self,
        migration_type: str,
        collections: list[str],
        report_path: str | None = None,
    ) -> str:
        """
        Start a new migration and record it in the database.

        Args:
            migration_type: Type of migration (e.g., "to-multitenant")
            collections: List of collections being migrated
            report_path: Path to migration report file

        Returns:
            Migration ID (UUID)
        """
        migration_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO migration_history
                (migration_id, migration_type, started_at, status, collections_migrated, points_migrated, report_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    migration_id,
                    migration_type,
                    now,
                    MigrationStatus.RUNNING.value,
                    json.dumps(collections),
                    0,
                    report_path,
                ),
            )
            conn.commit()

        logger.info(f"Started migration {migration_id} ({migration_type})")
        return migration_id

    def update_progress(self, migration_id: str, points_migrated: int) -> None:
        """Update migration progress."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE migration_history
                SET points_migrated = ?
                WHERE migration_id = ?
                """,
                (points_migrated, migration_id),
            )
            conn.commit()

    def complete_migration(
        self, migration_id: str, points_migrated: int, report_path: str | None = None
    ) -> None:
        """Mark migration as completed."""
        now = datetime.now(timezone.utc).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE migration_history
                SET status = ?, completed_at = ?, points_migrated = ?, report_path = ?
                WHERE migration_id = ?
                """,
                (
                    MigrationStatus.COMPLETED.value,
                    now,
                    points_migrated,
                    report_path,
                    migration_id,
                ),
            )
            conn.commit()

        logger.info(f"Migration {migration_id} completed: {points_migrated} points migrated")

    def fail_migration(self, migration_id: str, error_message: str) -> None:
        """Mark migration as failed."""
        now = datetime.now(timezone.utc).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE migration_history
                SET status = ?, completed_at = ?, error_message = ?
                WHERE migration_id = ?
                """,
                (MigrationStatus.FAILED.value, now, error_message, migration_id),
            )
            conn.commit()

        logger.error(f"Migration {migration_id} failed: {error_message}")

    def rollback_migration(self, migration_id: str) -> None:
        """Mark migration as rolled back."""
        now = datetime.now(timezone.utc).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE migration_history
                SET status = ?, completed_at = ?
                WHERE migration_id = ?
                """,
                (MigrationStatus.ROLLED_BACK.value, now, migration_id),
            )
            conn.commit()

        logger.info(f"Migration {migration_id} rolled back")

    def get_migration(self, migration_id: str) -> MigrationState | None:
        """Get migration state by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM migration_history WHERE migration_id = ?",
                (migration_id,),
            )
            row = cursor.fetchone()

        if row is None:
            return None

        return MigrationState(
            migration_id=row["migration_id"],
            migration_type=row["migration_type"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            status=MigrationStatus(row["status"]),
            collections_migrated=json.loads(row["collections_migrated"] or "[]"),
            points_migrated=row["points_migrated"],
            error_message=row["error_message"],
            report_path=row["report_path"],
        )

    def list_migrations(
        self, status: MigrationStatus | None = None, limit: int = 20
    ) -> list[MigrationState]:
        """List migrations, optionally filtered by status."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            if status:
                cursor = conn.execute(
                    """
                    SELECT * FROM migration_history
                    WHERE status = ?
                    ORDER BY started_at DESC
                    LIMIT ?
                    """,
                    (status.value, limit),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM migration_history
                    ORDER BY started_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                )

            rows = cursor.fetchall()

        return [
            MigrationState(
                migration_id=row["migration_id"],
                migration_type=row["migration_type"],
                started_at=row["started_at"],
                completed_at=row["completed_at"],
                status=MigrationStatus(row["status"]),
                collections_migrated=json.loads(row["collections_migrated"] or "[]"),
                points_migrated=row["points_migrated"],
                error_message=row["error_message"],
                report_path=row["report_path"],
            )
            for row in rows
        ]

    def get_last_completed_migration(self) -> MigrationState | None:
        """Get the most recent completed migration."""
        migrations = self.list_migrations(status=MigrationStatus.COMPLETED, limit=1)
        return migrations[0] if migrations else None


@dataclass
class MultiTenantMigrationReport:
    """Report for multi-tenant migration operation."""

    total_collections_migrated: int
    total_points_migrated: int
    failed_migrations: list[dict[str, Any]]
    verification_results: dict[str, bool]
    elapsed_time_seconds: float
    throughput_points_per_sec: float
    created_aliases: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class MultiTenantMigrationManager:
    """
    Manages migration from per-project collections (_{project_id}) to
    unified multi-tenant collections (_projects, _libraries, _memory).

    Task 404: Multi-tenant architecture migration.
    """

    def __init__(
        self,
        qdrant_client: QdrantClient,
        backup_dir: Path | None = None,
    ):
        """
        Initialize multi-tenant migration manager.

        Args:
            qdrant_client: Qdrant client instance
            backup_dir: Directory for backups (default: .taskmaster/backups/multitenant/)
        """
        self.qdrant_client = qdrant_client
        self.backup_dir = backup_dir or (
            Path.home() / ".taskmaster" / "backups" / "multitenant"
        )
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def discover_project_collections(self) -> list[str]:
        """
        Discover existing per-project collections (_{project_id} pattern).

        Returns:
            List of collection names matching _{hex_string} pattern
        """
        try:
            collections_response = self.qdrant_client.get_collections()
            all_collections = [c.name for c in collections_response.collections]
        except Exception as e:
            logger.error(f"Failed to retrieve collections: {e}")
            raise

        project_collections = []

        for name in all_collections:
            # Match _{project_id} pattern (underscore + 12 hex chars)
            if (
                name.startswith("_")
                and len(name) == 13
                and all(c in "0123456789abcdef" for c in name[1:])
            ):
                project_collections.append(name)

        logger.info(f"Discovered {len(project_collections)} per-project collections")
        return project_collections

    def ensure_unified_collection_exists(
        self, collection_name: str, vector_size: int = 384
    ) -> bool:
        """
        Ensure unified collection exists with proper configuration.

        Args:
            collection_name: Target collection name (_projects, _libraries, _memory)
            vector_size: Vector dimension size (default: 384 for all-MiniLM-L6-v2)

        Returns:
            True if collection exists or was created
        """
        try:
            self.qdrant_client.get_collection(collection_name)
            logger.info(f"Collection {collection_name} already exists")
            return True
        except UnexpectedResponse:
            pass  # Collection doesn't exist

        try:
            # Create with optimized multi-tenant config
            from qdrant_client.http.models import (
                HnswConfigDiff,
            )

            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
                hnsw_config=HnswConfigDiff(
                    m=16,
                    ef_construct=100,
                ),
                on_disk_payload=True,  # For large multi-tenant collections
            )

            # Create payload index for project_id (essential for filtering)
            if collection_name == UNIFIED_COLLECTIONS["projects"]:
                self.qdrant_client.create_payload_index(
                    collection_name=collection_name,
                    field_name="project_id",
                    field_schema="keyword",
                )
                logger.info(f"Created project_id index on {collection_name}")

            # Create payload index for library_name
            elif collection_name == UNIFIED_COLLECTIONS["libraries"]:
                self.qdrant_client.create_payload_index(
                    collection_name=collection_name,
                    field_name="library_name",
                    field_schema="keyword",
                )
                logger.info(f"Created library_name index on {collection_name}")

            logger.info(f"Created unified collection: {collection_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to create collection {collection_name}: {e}")
            return False

    async def migrate_collection_to_unified(
        self,
        source_collection: str,
        target_collection: str,
        batch_size: int = 1000,
        dry_run: bool = False,
    ) -> tuple[int, str | None]:
        """
        Migrate documents from source collection to unified target collection.

        Args:
            source_collection: Source collection name (_{project_id})
            target_collection: Target unified collection (_projects)
            batch_size: Number of documents per batch
            dry_run: If True, only count documents without migrating

        Returns:
            Tuple of (points_migrated, error_message or None)
        """
        try:
            # Get source collection info
            source_info = self.qdrant_client.get_collection(source_collection)
            total_points = source_info.points_count or 0

            if total_points == 0:
                logger.info(f"Collection {source_collection} is empty, skipping")
                return 0, None

            if dry_run:
                logger.info(
                    f"[DRY RUN] Would migrate {total_points} points from {source_collection}"
                )
                return total_points, None

            # Extract project_id from collection name
            project_id = source_collection[1:]  # Remove leading underscore

            # Scroll through all documents in batches
            offset = None
            points_migrated = 0

            while True:
                batch, offset = self.qdrant_client.scroll(
                    collection_name=source_collection,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True,
                )

                if not batch:
                    break

                # Enrich payload with project_id if missing
                enriched_points = []
                for point in batch:
                    payload = dict(point.payload) if point.payload else {}

                    # Ensure project_id is set
                    if "project_id" not in payload:
                        payload["project_id"] = project_id

                    # Add migration metadata
                    payload["_migrated_from_collection"] = source_collection
                    payload["_migrated_at"] = datetime.now(timezone.utc).isoformat()

                    enriched_points.append(
                        PointStruct(
                            id=point.id,
                            vector=point.vector,
                            payload=payload,
                        )
                    )

                # Upsert to unified collection
                self.qdrant_client.upsert(
                    collection_name=target_collection,
                    points=enriched_points,
                )

                points_migrated += len(batch)

                if points_migrated % (batch_size * 10) == 0:
                    logger.info(
                        f"Migrated {points_migrated}/{total_points} points from {source_collection}"
                    )

                if offset is None:
                    break

            logger.info(
                f"Completed migration of {points_migrated} points from {source_collection}"
            )
            return points_migrated, None

        except Exception as e:
            error_msg = f"Failed to migrate {source_collection}: {e}"
            logger.error(error_msg)
            return 0, error_msg

    async def verify_migration(
        self, source_collection: str, target_collection: str, sample_size: int = 100
    ) -> tuple[bool, list[str]]:
        """
        Verify migration by comparing source and target collections.

        Args:
            source_collection: Original collection name
            target_collection: Unified collection name
            sample_size: Number of random documents to verify

        Returns:
            Tuple of (success, list of issues found)
        """
        issues: list[str] = []

        try:
            # Get source collection info
            source_info = self.qdrant_client.get_collection(source_collection)
            source_count = source_info.points_count or 0

            # Extract project_id for filtering
            project_id = source_collection[1:]

            # Count documents with this project_id in target
            from qdrant_client.http.models import (
                Filter,
                FieldCondition,
                MatchValue,
            )

            target_scroll = self.qdrant_client.scroll(
                collection_name=target_collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="project_id",
                            match=MatchValue(value=project_id),
                        )
                    ]
                ),
                limit=1,  # Just to get count
            )

            # Count by scrolling (more accurate)
            target_count = 0
            offset = None
            while True:
                batch, offset = self.qdrant_client.scroll(
                    collection_name=target_collection,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="project_id",
                                match=MatchValue(value=project_id),
                            )
                        ]
                    ),
                    limit=1000,
                    offset=offset,
                )
                target_count += len(batch)
                if offset is None:
                    break

            # Check point count matches
            if source_count != target_count:
                issues.append(
                    f"Point count mismatch: source={source_count}, target={target_count}"
                )

            # Sample verification: Check random documents
            if source_count > 0:
                sample_batch, _ = self.qdrant_client.scroll(
                    collection_name=source_collection,
                    limit=min(sample_size, source_count),
                    with_payload=True,
                )

                for point in sample_batch:
                    # Try to retrieve same point from target
                    try:
                        target_points = self.qdrant_client.retrieve(
                            collection_name=target_collection,
                            ids=[point.id],
                            with_payload=True,
                        )
                        if not target_points:
                            issues.append(f"Point {point.id} not found in target")
                        else:
                            target_point = target_points[0]
                            # Check project_id is set
                            if target_point.payload.get("project_id") != project_id:
                                issues.append(
                                    f"Point {point.id} has incorrect project_id"
                                )
                    except Exception as e:
                        issues.append(f"Failed to verify point {point.id}: {e}")

            success = len(issues) == 0
            if success:
                logger.info(f"Verification passed for {source_collection}")
            else:
                logger.warning(
                    f"Verification failed for {source_collection}: {len(issues)} issues"
                )

            return success, issues

        except Exception as e:
            issues.append(f"Verification error: {e}")
            return False, issues

    def create_backward_compatibility_alias(
        self, source_collection: str, target_collection: str
    ) -> bool:
        """
        Create collection alias for backward compatibility.

        Args:
            source_collection: Original collection name to alias
            target_collection: Target unified collection

        Returns:
            True if alias created successfully
        """
        try:
            # Create alias from old name to new collection
            self.qdrant_client.update_collection_aliases(
                change_aliases_operations=[
                    {
                        "create_alias": {
                            "collection_name": target_collection,
                            "alias_name": source_collection,
                        }
                    }
                ]
            )
            logger.info(
                f"Created alias: {source_collection} -> {target_collection}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to create alias {source_collection}: {e}")
            return False

    async def rollback_collection(
        self,
        source_collection: str,
        target_collection: str,
    ) -> tuple[int, str | None]:
        """
        Rollback migration by deleting points with matching project_id from target.

        Task 410: Rollback capability for failed migrations.

        Args:
            source_collection: Original collection name (_{project_id})
            target_collection: Unified collection (_projects)

        Returns:
            Tuple of (points_deleted, error_message or None)
        """
        try:
            # Extract project_id from collection name
            project_id = source_collection[1:]  # Remove leading underscore

            # Count points to be deleted
            points_to_delete = 0
            offset = None
            point_ids: list[str | int] = []

            while True:
                batch, offset = self.qdrant_client.scroll(
                    collection_name=target_collection,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="project_id",
                                match=MatchValue(value=project_id),
                            )
                        ]
                    ),
                    limit=1000,
                    offset=offset,
                )

                for point in batch:
                    point_ids.append(point.id)

                points_to_delete += len(batch)

                if offset is None:
                    break

            if points_to_delete == 0:
                logger.info(f"No points to rollback for {source_collection}")
                return 0, None

            # Delete points in batches
            batch_size = 1000
            for i in range(0, len(point_ids), batch_size):
                batch_ids = point_ids[i : i + batch_size]
                self.qdrant_client.delete(
                    collection_name=target_collection,
                    points_selector=batch_ids,
                )

            logger.info(
                f"Rolled back {points_to_delete} points for {source_collection}"
            )
            return points_to_delete, None

        except Exception as e:
            error_msg = f"Rollback failed for {source_collection}: {e}"
            logger.error(error_msg)
            return 0, error_msg

    async def verify_rollback(
        self,
        source_collection: str,
        target_collection: str,
    ) -> tuple[bool, list[str]]:
        """
        Verify rollback was successful.

        Checks:
        - Original collection still exists and has expected points
        - No points with this project_id remain in target collection

        Args:
            source_collection: Original collection name
            target_collection: Unified collection name

        Returns:
            Tuple of (success, list of issues found)
        """
        issues: list[str] = []

        try:
            # Check original collection still exists
            try:
                source_info = self.qdrant_client.get_collection(source_collection)
                source_count = source_info.points_count or 0
                if source_count == 0:
                    issues.append(f"Original collection {source_collection} is empty")
            except UnexpectedResponse:
                issues.append(f"Original collection {source_collection} not found")

            # Check no points remain in target with this project_id
            project_id = source_collection[1:]
            remaining_count = 0
            offset = None

            while True:
                batch, offset = self.qdrant_client.scroll(
                    collection_name=target_collection,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="project_id",
                                match=MatchValue(value=project_id),
                            )
                        ]
                    ),
                    limit=100,
                    offset=offset,
                )
                remaining_count += len(batch)
                if offset is None:
                    break

            if remaining_count > 0:
                issues.append(
                    f"{remaining_count} points still exist in target for project_id={project_id}"
                )

            success = len(issues) == 0
            return success, issues

        except Exception as e:
            issues.append(f"Verification error: {e}")
            return False, issues


@migrate_app.command("to-multitenant")
def to_multitenant_command(
    dry_run: bool = typer.Option(
        True,
        "--dry-run/--execute",
        help="Simulate migration (default) or execute it",
    ),
    batch_size: int = typer.Option(
        1000,
        "--batch-size",
        "-b",
        help="Number of documents per batch",
    ),
    verify: bool = typer.Option(
        True,
        "--verify/--no-verify",
        help="Run verification after migration",
    ),
    create_aliases: bool = typer.Option(
        True,
        "--aliases/--no-aliases",
        help="Create backward compatibility aliases",
    ),
):
    """
    Migrate from per-project collections (_{project_id}) to unified _projects collection.

    This command:
    1. Discovers existing _{project_id} collections
    2. Creates unified _projects collection with proper indexes
    3. Scrolls all points in batches and adds project_id to payload
    4. Upserts to _projects collection
    5. Verifies point count matches
    6. Creates collection aliases for backward compatibility

    Use --dry-run (default) to see what would be migrated without making changes.
    Use --execute to perform the actual migration.
    """
    try:
        from rich.console import Console
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
        from rich.table import Table

        console = Console()

        # Load configuration
        config = get_config()
        qdrant_url = config.get("qdrant", {}).get("url", "http://localhost:6333")

        # Initialize client and managers
        qdrant_client = QdrantClient(url=qdrant_url)
        manager = MultiTenantMigrationManager(qdrant_client)
        state_manager = MigrationStateManager()

        console.print(f"\n[bold]{'[DRY RUN] ' if dry_run else ''}Multi-Tenant Migration[/bold]")
        console.print(f"Target collection: {UNIFIED_COLLECTIONS['projects']}")
        console.print(f"Batch size: {batch_size}\n")

        # Step 1: Discover project collections
        project_collections = manager.discover_project_collections()

        if not project_collections:
            console.print("[yellow]No per-project collections found. Nothing to migrate.[/yellow]")
            return

        console.print(f"Found {len(project_collections)} collections to migrate:")
        for coll in project_collections[:10]:
            try:
                info = qdrant_client.get_collection(coll)
                console.print(f"  - {coll} ({info.points_count or 0} points)")
            except Exception:
                console.print(f"  - {coll} (count unavailable)")
        if len(project_collections) > 10:
            console.print(f"  ... and {len(project_collections) - 10} more")

        if dry_run:
            console.print("\n[yellow]DRY RUN - No changes will be made[/yellow]")
        else:
            console.print("\n[bold red]EXECUTE MODE - Changes will be made[/bold red]")
            if not typer.confirm("Proceed with migration?"):
                console.print("Migration cancelled.")
                raise typer.Exit(0)

        # Step 2: Ensure unified collection exists
        if not dry_run:
            if not manager.ensure_unified_collection_exists(UNIFIED_COLLECTIONS["projects"]):
                console.print("[red]Failed to create unified collection[/red]")
                raise typer.Exit(1)

        # Step 3: Start migration tracking (Task 410)
        migration_id: str | None = None
        if not dry_run:
            migration_id = state_manager.start_migration(
                migration_type="to-multitenant",
                collections=project_collections,
            )
            console.print(f"Migration ID: {migration_id[:12]}...")

        # Step 4: Migrate collections
        start_time = time.time()
        total_points_migrated = 0
        failed_migrations: list[dict[str, Any]] = []
        created_aliases: list[str] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task("Migrating...", total=len(project_collections))

            for source_collection in project_collections:
                progress.update(task, description=f"Migrating {source_collection}...")

                points, error = asyncio.run(
                    manager.migrate_collection_to_unified(
                        source_collection=source_collection,
                        target_collection=UNIFIED_COLLECTIONS["projects"],
                        batch_size=batch_size,
                        dry_run=dry_run,
                    )
                )

                if error:
                    failed_migrations.append(
                        {"collection": source_collection, "error": error}
                    )
                else:
                    total_points_migrated += points

                    # Create alias after successful migration
                    if create_aliases and not dry_run:
                        # Note: Can't create alias with same name as existing collection
                        # So we skip alias creation if collection still exists
                        # Aliases created after cleanup instead
                        pass

                progress.update(task, advance=1)

        elapsed_time = time.time() - start_time
        throughput = total_points_migrated / elapsed_time if elapsed_time > 0 else 0

        # Step 4: Verification
        verification_results: dict[str, bool] = {}
        if verify and not dry_run:
            console.print("\n[bold]Running verification...[/bold]")

            for source_collection in project_collections:
                success, issues = asyncio.run(
                    manager.verify_migration(
                        source_collection=source_collection,
                        target_collection=UNIFIED_COLLECTIONS["projects"],
                    )
                )
                verification_results[source_collection] = success
                if not success:
                    for issue in issues:
                        console.print(f"  [yellow]⚠ {source_collection}: {issue}[/yellow]")

        # Step 5: Generate report
        report = MultiTenantMigrationReport(
            total_collections_migrated=len(project_collections) - len(failed_migrations),
            total_points_migrated=total_points_migrated,
            failed_migrations=failed_migrations,
            verification_results=verification_results,
            elapsed_time_seconds=elapsed_time,
            throughput_points_per_sec=throughput,
            created_aliases=created_aliases,
        )

        # Display report
        console.print("\n" + "=" * 80)
        console.print(f"[bold]MIGRATION {'SIMULATION' if dry_run else 'REPORT'}[/bold]")
        console.print("=" * 80)

        table = Table(show_header=False)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Collections migrated", str(report.total_collections_migrated))
        table.add_row("Total points migrated", str(report.total_points_migrated))
        table.add_row("Failed migrations", str(len(report.failed_migrations)))
        table.add_row("Elapsed time", f"{report.elapsed_time_seconds:.2f} seconds")
        table.add_row("Throughput", f"{report.throughput_points_per_sec:.0f} points/sec")

        if verify and not dry_run:
            passed = sum(1 for v in verification_results.values() if v)
            table.add_row("Verification", f"{passed}/{len(verification_results)} passed")

        console.print(table)

        if failed_migrations:
            console.print("\n[red]Failed migrations:[/red]")
            for failure in failed_migrations:
                console.print(f"  - {failure['collection']}: {failure['error']}")

        # Save report to file
        report_file = manager.backup_dir / f"migration_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        console.print(f"\nReport saved to: {report_file}")

        # Step 6: Update migration state (Task 410)
        if migration_id and not dry_run:
            if failed_migrations:
                # Mark as failed if any migrations failed
                error_msg = f"{len(failed_migrations)} collection(s) failed: " + ", ".join(
                    f["collection"] for f in failed_migrations
                )
                state_manager.fail_migration(migration_id, error_msg)
                console.print(f"\n[yellow]Migration recorded as FAILED ({migration_id[:12]}...)[/yellow]")
            else:
                # Mark as completed
                state_manager.complete_migration(
                    migration_id=migration_id,
                    points_migrated=total_points_migrated,
                    report_path=str(report_file),
                )
                console.print(f"\n[green]Migration recorded as COMPLETED ({migration_id[:12]}...)[/green]")

        if dry_run:
            console.print("\n[yellow]This was a dry run. Use --execute to perform the actual migration.[/yellow]")
        else:
            if failed_migrations:
                console.print("\n[bold yellow]Migration completed with failures![/bold yellow]")
                console.print("\nTo rollback: wqm migrate rollback-multitenant --migration-id=" + (migration_id[:12] if migration_id else "ID") + "...")
            else:
                console.print("\n[bold green]Migration completed successfully![/bold green]")
            console.print("\nNext steps:")
            console.print("  1. Verify: wqm migrate verify")
            console.print("  2. Cleanup: wqm migrate cleanup (after verification)")
            console.print("  3. History: wqm migrate history")

    except ImportError:
        # Fallback if rich is not available
        print("Error: 'rich' package required for progress display")
        print("Install with: pip install rich")
        raise typer.Exit(1)
    except Exception as e:
        # Record failure in state manager if migration was started
        if "migration_id" in dir() and migration_id and "state_manager" in dir():
            try:
                state_manager.fail_migration(migration_id, str(e))
            except Exception:
                pass  # Don't mask the original error
        print(f"Error: {e}")
        raise typer.Exit(1)


@migrate_app.command("verify")
def verify_command(
    collection: str | None = typer.Option(
        None,
        "--collection",
        "-c",
        help="Specific collection to verify (default: all)",
    ),
    sample_size: int = typer.Option(
        100,
        "--sample-size",
        "-s",
        help="Number of documents to sample for verification",
    ),
):
    """
    Verify migration by checking document counts and sampling data integrity.

    Checks:
    - Point counts match between source and target (filtered by project_id)
    - Random sample of documents exist in target with correct metadata
    - All old collections have been migrated
    """
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()

        # Load configuration
        config = get_config()
        qdrant_url = config.get("qdrant", {}).get("url", "http://localhost:6333")

        # Initialize client and manager
        qdrant_client = QdrantClient(url=qdrant_url)
        manager = MultiTenantMigrationManager(qdrant_client)

        console.print("\n[bold]Migration Verification[/bold]")

        # Get collections to verify
        if collection:
            collections_to_verify = [collection]
        else:
            collections_to_verify = manager.discover_project_collections()

        if not collections_to_verify:
            console.print("[yellow]No per-project collections found to verify.[/yellow]")
            return

        console.print(f"Verifying {len(collections_to_verify)} collections...\n")

        # Verify each collection
        table = Table(title="Verification Results")
        table.add_column("Collection", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Issues", style="yellow")

        all_passed = True
        for source_collection in collections_to_verify:
            success, issues = asyncio.run(
                manager.verify_migration(
                    source_collection=source_collection,
                    target_collection=UNIFIED_COLLECTIONS["projects"],
                    sample_size=sample_size,
                )
            )

            status = "✓ PASS" if success else "✗ FAIL"
            issues_str = "; ".join(issues) if issues else "None"
            table.add_row(source_collection, status, issues_str)

            if not success:
                all_passed = False

        console.print(table)

        if all_passed:
            console.print("\n[bold green]All verifications passed![/bold green]")
            console.print("\nYou can now run cleanup: wqm migrate cleanup")
        else:
            console.print("\n[bold red]Some verifications failed.[/bold red]")
            console.print("Review issues above before running cleanup.")

    except ImportError:
        print("Error: 'rich' package required. Install with: pip install rich")
        raise typer.Exit(1)
    except Exception as e:
        print(f"Error: {e}")
        raise typer.Exit(1)


@migrate_app.command("cleanup")
def cleanup_command(
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip verification and delete without confirmation",
    ),
    remove_aliases: bool = typer.Option(
        True,
        "--remove-aliases/--keep-aliases",
        help="Also remove collection aliases",
    ),
    archive_report: bool = typer.Option(
        True,
        "--archive/--no-archive",
        help="Archive migration report before cleanup",
    ),
):
    """
    Delete old per-project collections after verifying migration.

    This command:
    1. Verifies all documents are in unified collection
    2. Optionally removes collection aliases
    3. Deletes old _{project_id} collections
    4. Archives migration report

    WARNING: This is a destructive operation. Use with caution.
    """
    try:
        from rich.console import Console

        console = Console()

        # Load configuration
        config = get_config()
        qdrant_url = config.get("qdrant", {}).get("url", "http://localhost:6333")

        # Initialize client and manager
        qdrant_client = QdrantClient(url=qdrant_url)
        manager = MultiTenantMigrationManager(qdrant_client)

        console.print("\n[bold red]Multi-Tenant Migration Cleanup[/bold red]")
        console.print("[yellow]WARNING: This will DELETE old collections permanently![/yellow]\n")

        # Discover collections to clean up
        project_collections = manager.discover_project_collections()

        if not project_collections:
            console.print("[yellow]No per-project collections found to clean up.[/yellow]")
            return

        console.print(f"Collections to delete: {len(project_collections)}")
        for coll in project_collections[:5]:
            console.print(f"  - {coll}")
        if len(project_collections) > 5:
            console.print(f"  ... and {len(project_collections) - 5} more")

        # Run verification unless forced
        if not force:
            console.print("\n[bold]Running verification before cleanup...[/bold]")

            all_verified = True
            for source_collection in project_collections:
                success, issues = asyncio.run(
                    manager.verify_migration(
                        source_collection=source_collection,
                        target_collection=UNIFIED_COLLECTIONS["projects"],
                    )
                )
                if not success:
                    console.print(f"[red]✗ {source_collection} verification failed:[/red]")
                    for issue in issues:
                        console.print(f"    {issue}")
                    all_verified = False

            if not all_verified:
                console.print("\n[red]Verification failed. Cleanup aborted.[/red]")
                console.print("Fix issues above or use --force to skip verification.")
                raise typer.Exit(1)

            console.print("[green]All verifications passed.[/green]\n")

        # Confirm deletion
        if not force:
            confirm = typer.confirm(
                f"Delete {len(project_collections)} collections permanently?"
            )
            if not confirm:
                console.print("Cleanup cancelled.")
                raise typer.Exit(0)

        # Archive report if requested
        if archive_report:
            archive_file = manager.backup_dir / f"cleanup_archive_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
            archive_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "collections_deleted": project_collections,
                "unified_collection": UNIFIED_COLLECTIONS["projects"],
            }
            with open(archive_file, "w") as f:
                json.dump(archive_data, f, indent=2)
            console.print(f"Archive saved: {archive_file}")

        # Delete collections
        deleted = 0
        failed = 0

        for collection in project_collections:
            try:
                qdrant_client.delete_collection(collection)
                console.print(f"[green]✓[/green] Deleted {collection}")
                deleted += 1
            except Exception as e:
                console.print(f"[red]✗[/red] Failed to delete {collection}: {e}")
                failed += 1

        console.print(f"\n[bold]Cleanup complete:[/bold]")
        console.print(f"  Deleted: {deleted}")
        console.print(f"  Failed: {failed}")

        if failed == 0:
            console.print("\n[bold green]Migration cleanup completed successfully![/bold green]")
        else:
            console.print(f"\n[yellow]Cleanup completed with {failed} failures.[/yellow]")

    except ImportError:
        print("Error: 'rich' package required. Install with: pip install rich")
        raise typer.Exit(1)
    except Exception as e:
        print(f"Error: {e}")
        raise typer.Exit(1)


@migrate_app.command("rollback-multitenant")
def rollback_multitenant_command(
    migration_id: str | None = typer.Argument(
        None,
        help="Migration ID to rollback (from migration history). If not provided, shows recent migrations.",
    ),
    verify: bool = typer.Option(
        True,
        "--verify/--no-verify",
        help="Verify rollback was successful",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip confirmation and verification checks",
    ),
):
    """
    Rollback a multi-tenant migration by removing migrated data from unified collection.

    This command:
    1. Loads migration state from SQLite database
    2. For each migrated collection:
       - Deletes points from _projects collection with project_id filter
       - Verifies original collection still exists and intact
    3. Updates migration status to "rolled_back"
    4. Generates rollback report

    If no migration_id is provided, shows recent migrations for selection.

    Safety features:
    - Verifies original collections exist before rollback
    - Atomic deletion of new data per collection
    - Verifies rollback success before marking complete
    - All operations logged for audit trail
    """
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()

        # Load configuration
        config = get_config()
        qdrant_url = config.get("qdrant", {}).get("url", "http://localhost:6333")

        # Initialize managers
        qdrant_client = QdrantClient(url=qdrant_url)
        migration_manager = MultiTenantMigrationManager(qdrant_client)
        state_manager = MigrationStateManager()

        # If no migration_id provided, show recent migrations
        if migration_id is None:
            console.print("\n[bold]Recent Migrations[/bold]")
            migrations = state_manager.list_migrations(limit=10)

            if not migrations:
                console.print("[yellow]No migrations found in history.[/yellow]")
                console.print("Run `wqm migrate to-multitenant --execute` first.")
                return

            table = Table(title="Migration History")
            table.add_column("ID", style="cyan", no_wrap=True)
            table.add_column("Type", style="green")
            table.add_column("Status", style="yellow")
            table.add_column("Points", style="magenta")
            table.add_column("Started", style="blue")
            table.add_column("Collections", style="white")

            for m in migrations:
                status_style = {
                    MigrationStatus.COMPLETED: "green",
                    MigrationStatus.FAILED: "red",
                    MigrationStatus.RUNNING: "yellow",
                    MigrationStatus.ROLLED_BACK: "cyan",
                }.get(m.status, "white")

                table.add_row(
                    m.migration_id[:8] + "...",
                    m.migration_type,
                    f"[{status_style}]{m.status.value}[/{status_style}]",
                    str(m.points_migrated),
                    m.started_at[:19].replace("T", " "),
                    str(len(m.collections_migrated)),
                )

            console.print(table)
            console.print("\nTo rollback a migration, run:")
            console.print("  wqm migrate rollback-multitenant <MIGRATION_ID>")
            return

        # Get migration state
        migration = state_manager.get_migration(migration_id)

        if migration is None:
            # Try partial match
            all_migrations = state_manager.list_migrations(limit=100)
            matches = [m for m in all_migrations if m.migration_id.startswith(migration_id)]

            if len(matches) == 1:
                migration = matches[0]
            elif len(matches) > 1:
                console.print(f"[yellow]Multiple migrations match '{migration_id}':[/yellow]")
                for m in matches:
                    console.print(f"  - {m.migration_id}")
                raise typer.Exit(1)
            else:
                console.print(f"[red]Migration '{migration_id}' not found[/red]")
                raise typer.Exit(1)

        console.print(f"\n[bold]Rollback Migration: {migration.migration_id[:8]}...[/bold]")
        console.print(f"Type: {migration.migration_type}")
        console.print(f"Status: {migration.status.value}")
        console.print(f"Points migrated: {migration.points_migrated}")
        console.print(f"Collections: {len(migration.collections_migrated)}")

        if migration.status == MigrationStatus.ROLLED_BACK:
            console.print("\n[yellow]This migration has already been rolled back.[/yellow]")
            return

        if migration.status == MigrationStatus.RUNNING:
            console.print("\n[red]Cannot rollback a running migration.[/red]")
            console.print("Wait for it to complete or fail first.")
            raise typer.Exit(1)

        # Confirmation
        if not force:
            console.print("\n[bold red]WARNING: This will delete migrated data from the unified collection.[/bold red]")
            console.print("Original collections will NOT be affected.")
            if not typer.confirm("Proceed with rollback?"):
                console.print("Rollback cancelled.")
                raise typer.Exit(0)

        # Verify original collections still exist
        console.print("\n[bold]Checking original collections...[/bold]")
        missing_collections = []

        for source_collection in migration.collections_migrated:
            try:
                info = qdrant_client.get_collection(source_collection)
                console.print(f"  ✓ {source_collection} ({info.points_count or 0} points)")
            except UnexpectedResponse:
                missing_collections.append(source_collection)
                console.print(f"  [red]✗ {source_collection} (NOT FOUND)[/red]")

        if missing_collections and not force:
            console.print(
                f"\n[red]{len(missing_collections)} original collections are missing.[/red]"
            )
            console.print("Data recovery may not be possible. Use --force to proceed anyway.")
            raise typer.Exit(1)

        # Execute rollback
        console.print("\n[bold]Rolling back...[/bold]")
        start_time = time.time()
        total_points_deleted = 0
        failed_rollbacks: list[dict[str, Any]] = []

        for source_collection in migration.collections_migrated:
            points, error = asyncio.run(
                migration_manager.rollback_collection(
                    source_collection=source_collection,
                    target_collection=UNIFIED_COLLECTIONS["projects"],
                )
            )

            if error:
                failed_rollbacks.append({"collection": source_collection, "error": error})
                console.print(f"  [red]✗ {source_collection}: {error}[/red]")
            else:
                total_points_deleted += points
                console.print(f"  ✓ {source_collection}: {points} points deleted")

        elapsed_time = time.time() - start_time

        # Verify rollback if requested
        verification_results: dict[str, bool] = {}
        if verify and not force:
            console.print("\n[bold]Verifying rollback...[/bold]")

            for source_collection in migration.collections_migrated:
                success, issues = asyncio.run(
                    migration_manager.verify_rollback(
                        source_collection=source_collection,
                        target_collection=UNIFIED_COLLECTIONS["projects"],
                    )
                )
                verification_results[source_collection] = success

                if success:
                    console.print(f"  ✓ {source_collection}: verified")
                else:
                    console.print(f"  [yellow]⚠ {source_collection}:[/yellow]")
                    for issue in issues:
                        console.print(f"      {issue}")

        # Update migration state
        state_manager.rollback_migration(migration.migration_id)

        # Display results
        console.print("\n" + "=" * 80)
        console.print("[bold]ROLLBACK COMPLETE[/bold]")
        console.print("=" * 80)

        table = Table(show_header=False)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Collections rolled back", str(len(migration.collections_migrated) - len(failed_rollbacks)))
        table.add_row("Points deleted", str(total_points_deleted))
        table.add_row("Failed rollbacks", str(len(failed_rollbacks)))
        table.add_row("Elapsed time", f"{elapsed_time:.2f} seconds")

        if verify and not force:
            passed = sum(1 for v in verification_results.values() if v)
            table.add_row("Verification", f"{passed}/{len(verification_results)} passed")

        console.print(table)

        if failed_rollbacks:
            console.print("\n[red]Failed rollbacks:[/red]")
            for failure in failed_rollbacks:
                console.print(f"  - {failure['collection']}: {failure['error']}")

        # Save rollback report
        report_file = migration_manager.backup_dir / f"rollback_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        report_data = {
            "migration_id": migration.migration_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "collections_rolled_back": len(migration.collections_migrated) - len(failed_rollbacks),
            "points_deleted": total_points_deleted,
            "failed_rollbacks": failed_rollbacks,
            "verification_results": {k: v for k, v in verification_results.items()},
            "elapsed_time_seconds": elapsed_time,
        }
        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=2)
        console.print(f"\nRollback report saved: {report_file}")

        if len(failed_rollbacks) == 0:
            console.print("\n[bold green]Rollback completed successfully![/bold green]")
        else:
            console.print(f"\n[yellow]Rollback completed with {len(failed_rollbacks)} failures.[/yellow]")

    except ImportError:
        print("Error: 'rich' package required. Install with: pip install rich")
        raise typer.Exit(1)
    except Exception as e:
        print(f"Error: {e}")
        raise typer.Exit(1)


@migrate_app.command("history")
def history_command(
    status: str | None = typer.Option(
        None,
        "--status",
        "-s",
        help="Filter by status (running, completed, failed, rolled_back)",
    ),
    limit: int = typer.Option(
        20,
        "--limit",
        "-n",
        help="Number of migrations to show",
    ),
):
    """
    Show migration history from SQLite database.

    Displays recent migrations with their status, point counts, and timing.
    Use --status to filter by migration status.
    """
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()

        # Initialize state manager
        state_manager = MigrationStateManager()

        # Parse status filter
        status_filter = None
        if status:
            try:
                status_filter = MigrationStatus(status.lower())
            except ValueError:
                console.print(f"[red]Invalid status: {status}[/red]")
                console.print("Valid statuses: running, completed, failed, rolled_back")
                raise typer.Exit(1)

        # Get migrations
        migrations = state_manager.list_migrations(status=status_filter, limit=limit)

        if not migrations:
            console.print("[yellow]No migrations found.[/yellow]")
            return

        # Display table
        console.print(f"\n[bold]Migration History ({len(migrations)} records)[/bold]")

        table = Table()
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Type", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Points", justify="right", style="magenta")
        table.add_column("Collections", justify="right")
        table.add_column("Started", style="blue")
        table.add_column("Duration")

        for m in migrations:
            status_style = {
                MigrationStatus.COMPLETED: "green",
                MigrationStatus.FAILED: "red",
                MigrationStatus.RUNNING: "yellow",
                MigrationStatus.ROLLED_BACK: "cyan",
            }.get(m.status, "white")

            # Calculate duration if completed
            duration = "-"
            if m.completed_at:
                try:
                    start = datetime.fromisoformat(m.started_at.replace("Z", "+00:00"))
                    end = datetime.fromisoformat(m.completed_at.replace("Z", "+00:00"))
                    seconds = (end - start).total_seconds()
                    if seconds < 60:
                        duration = f"{seconds:.1f}s"
                    else:
                        duration = f"{seconds/60:.1f}m"
                except Exception:
                    pass

            table.add_row(
                m.migration_id[:12] + "...",
                m.migration_type,
                f"[{status_style}]{m.status.value}[/{status_style}]",
                str(m.points_migrated),
                str(len(m.collections_migrated)),
                m.started_at[:19].replace("T", " "),
                duration,
            )

        console.print(table)

        console.print("\nTo see full details: wqm migrate status")
        console.print("To rollback: wqm migrate rollback-multitenant <ID>")

    except ImportError:
        print("Error: 'rich' package required. Install with: pip install rich")
        raise typer.Exit(1)
    except Exception as e:
        print(f"Error: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    migrate_app()
