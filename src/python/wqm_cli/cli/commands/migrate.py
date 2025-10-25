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
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
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


if __name__ == "__main__":
    migrate_app()
