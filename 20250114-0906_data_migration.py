#!/usr/bin/env python3
"""
Data Migration Utility for Multi-Tenant Collection Architecture

This script migrates existing collections from the prefix-based naming system
to the new metadata-based multi-tenant architecture. It preserves all existing
data while reorganizing it to use the new project isolation system.

Usage:
    python 20250114-0906_data_migration.py --analyze     # Analyze existing collections
    python 20250114-0906_data_migration.py --migrate     # Perform data migration
    python 20250114-0906_data_migration.py --cleanup     # Clean up old collections
"""

import argparse
import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import sys

# Add Python path
sys.path.append('src/python')

from common.core.config import Config
from common.core.client import QdrantVectorDB
from common.utils.project_detection import ProjectDetector
from qdrant_client import QdrantClient
from qdrant_client.http import models


class DataMigration:
    """Handles migration of existing collection data to multi-tenant architecture."""

    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.config = Config()
        self.client = None
        self.project_detector = ProjectDetector()
        self.migration_log = []
        self.migration_stats = {
            "collections_analyzed": 0,
            "documents_migrated": 0,
            "collections_created": 0,
            "collections_removed": 0,
            "errors": 0
        }

    async def initialize(self):
        """Initialize Qdrant client connection."""
        try:
            self.client = QdrantClient(**self.config.qdrant_client_config)
            # Test connection
            await asyncio.get_event_loop().run_in_executor(
                None, self.client.get_collections
            )
            print("✅ Connected to Qdrant database")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Qdrant: {e}")
            return False

    async def analyze_existing_collections(self) -> Dict[str, Any]:
        """Analyze existing collections to plan migration."""
        print("Analyzing existing collections...")

        try:
            collections_response = await asyncio.get_event_loop().run_in_executor(
                None, self.client.get_collections
            )
            collections = collections_response.collections

            analysis = {
                "total_collections": len(collections),
                "prefix_based_collections": [],
                "multi_tenant_collections": [],
                "global_collections": [],
                "unknown_collections": [],
                "migration_plan": {},
                "estimated_complexity": "low"
            }

            for collection in collections:
                collection_name = collection.name
                self.migration_stats["collections_analyzed"] += 1

                # Analyze collection naming pattern
                if self._is_prefix_based_collection(collection_name):
                    project, collection_type = self._parse_prefix_collection(collection_name)
                    analysis["prefix_based_collections"].append({
                        "name": collection_name,
                        "project": project,
                        "type": collection_type,
                        "points_count": collection.points_count
                    })

                elif collection_name in self.config.workspace.global_collections:
                    analysis["global_collections"].append({
                        "name": collection_name,
                        "points_count": collection.points_count
                    })

                elif collection_name in self.config.workspace.collection_types:
                    analysis["multi_tenant_collections"].append({
                        "name": collection_name,
                        "points_count": collection.points_count
                    })

                else:
                    analysis["unknown_collections"].append({
                        "name": collection_name,
                        "points_count": collection.points_count
                    })

            # Create migration plan
            analysis["migration_plan"] = self._create_migration_plan(analysis)

            # Estimate complexity
            if len(analysis["prefix_based_collections"]) > 10:
                analysis["estimated_complexity"] = "high"
            elif len(analysis["prefix_based_collections"]) > 3:
                analysis["estimated_complexity"] = "medium"

            return analysis

        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            return {"error": str(e)}

    def _is_prefix_based_collection(self, collection_name: str) -> bool:
        """Check if collection uses prefix-based naming."""
        # Look for patterns like "project_type" or "prefix_project_type"
        pattern = r'^[a-zA-Z0-9_-]+_[a-zA-Z0-9_-]+$'
        if not re.match(pattern, collection_name):
            return False

        # Split and check if it looks like project_type
        parts = collection_name.split('_')
        if len(parts) >= 2:
            potential_type = parts[-1]
            # Check if the last part is a known collection type
            known_types = ['docs', 'notes', 'scratchbook', 'code', 'refs', 'references']
            return potential_type in known_types or len(potential_type) > 2

        return False

    def _parse_prefix_collection(self, collection_name: str) -> Tuple[str, str]:
        """Parse prefix-based collection name into project and type."""
        parts = collection_name.split('_')
        if len(parts) >= 2:
            collection_type = parts[-1]
            project = '_'.join(parts[:-1])
            return project, collection_type
        return collection_name, "unknown"

    def _create_migration_plan(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed migration plan."""
        plan = {
            "target_collections": {},
            "data_moves": [],
            "collections_to_create": set(),
            "collections_to_remove": []
        }

        # Plan migration for prefix-based collections
        for collection_info in analysis["prefix_based_collections"]:
            project = collection_info["project"]
            collection_type = collection_info["type"]
            source_name = collection_info["name"]

            # Map to target collection
            target_collection = collection_type
            if target_collection not in plan["target_collections"]:
                plan["target_collections"][target_collection] = []

            plan["target_collections"][target_collection].append({
                "source_collection": source_name,
                "project_id": project,
                "points_count": collection_info["points_count"]
            })

            plan["data_moves"].append({
                "source": source_name,
                "target": target_collection,
                "project_id": project,
                "operation": "migrate_with_metadata"
            })

            plan["collections_to_create"].add(target_collection)
            plan["collections_to_remove"].append(source_name)

        plan["collections_to_create"] = list(plan["collections_to_create"])
        return plan

    async def perform_migration(self, analysis: Dict[str, Any]) -> bool:
        """Perform the actual data migration."""
        print("Starting data migration...")

        if "error" in analysis:
            print(f"❌ Cannot migrate due to analysis error: {analysis['error']}")
            return False

        migration_plan = analysis["migration_plan"]

        try:
            # Step 1: Create target collections
            for target_collection in migration_plan["collections_to_create"]:
                await self._create_target_collection(target_collection)

            # Step 2: Migrate data with metadata
            for move in migration_plan["data_moves"]:
                await self._migrate_collection_data(
                    move["source"],
                    move["target"],
                    move["project_id"]
                )

            # Step 3: Verify migration
            verification_passed = await self._verify_migration(migration_plan)

            if verification_passed:
                print("✅ Data migration completed successfully")
                self._log_migration_stats()
                return True
            else:
                print("❌ Migration verification failed")
                return False

        except Exception as e:
            print(f"❌ Migration failed: {e}")
            self.migration_stats["errors"] += 1
            return False

    async def _create_target_collection(self, collection_name: str) -> bool:
        """Create a target collection with proper vector configuration."""
        if self.dry_run:
            print(f"[DRY RUN] Would create collection: {collection_name}")
            return True

        try:
            # Check if collection already exists
            existing_collections = await asyncio.get_event_loop().run_in_executor(
                None, self.client.get_collections
            )

            existing_names = [col.name for col in existing_collections.collections]
            if collection_name in existing_names:
                print(f"✅ Collection {collection_name} already exists")
                return True

            # Create collection with vector configuration
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.client.create_collection,
                collection_name,
                models.VectorParams(
                    size=384,  # Standard size for all-MiniLM-L6-v2
                    distance=models.Distance.COSINE
                )
            )

            print(f"✅ Created collection: {collection_name}")
            self.migration_stats["collections_created"] += 1
            self.migration_log.append(f"Created collection: {collection_name}")
            return True

        except Exception as e:
            print(f"❌ Failed to create collection {collection_name}: {e}")
            self.migration_stats["errors"] += 1
            return False

    async def _migrate_collection_data(self, source: str, target: str, project_id: str) -> bool:
        """Migrate data from source to target collection with project metadata."""
        print(f"Migrating {source} → {target} (project: {project_id})")

        if self.dry_run:
            print(f"[DRY RUN] Would migrate data from {source} to {target}")
            return True

        try:
            # Get all points from source collection
            scroll_result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.client.scroll,
                source,
                limit=1000,  # Process in batches
                with_payload=True,
                with_vectors=True
            )

            points = scroll_result[0]
            if not points:
                print(f"  No data to migrate from {source}")
                return True

            # Prepare points for target collection with metadata
            migrated_points = []
            for point in points:
                # Add project metadata to payload
                payload = point.payload.copy() if point.payload else {}
                payload.update({
                    "project_id": project_id,
                    "migrated_from": source,
                    "migration_timestamp": datetime.now().isoformat()
                })

                # Create new point for target collection
                migrated_point = models.PointStruct(
                    id=point.id,
                    vector=point.vector,
                    payload=payload
                )
                migrated_points.append(migrated_point)

            # Upload to target collection
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.client.upsert,
                target,
                migrated_points
            )

            print(f"  ✅ Migrated {len(migrated_points)} documents")
            self.migration_stats["documents_migrated"] += len(migrated_points)
            self.migration_log.append(f"Migrated {len(migrated_points)} documents from {source} to {target}")

            return True

        except Exception as e:
            print(f"  ❌ Failed to migrate data from {source} to {target}: {e}")
            self.migration_stats["errors"] += 1
            return False

    async def _verify_migration(self, migration_plan: Dict[str, Any]) -> bool:
        """Verify that migration completed successfully."""
        print("Verifying migration...")

        try:
            for move in migration_plan["data_moves"]:
                source = move["source"]
                target = move["target"]
                project_id = move["project_id"]

                # Count documents in source
                source_count = await self._count_collection_points(source)

                # Count migrated documents in target with project filter
                target_count = await self._count_project_points(target, project_id)

                if source_count != target_count:
                    print(f"❌ Verification failed: {source} ({source_count}) != {target}:{project_id} ({target_count})")
                    return False

                print(f"✅ Verified: {source} → {target}:{project_id} ({target_count} documents)")

            return True

        except Exception as e:
            print(f"❌ Verification failed: {e}")
            return False

    async def _count_collection_points(self, collection_name: str) -> int:
        """Count total points in a collection."""
        try:
            info = await asyncio.get_event_loop().run_in_executor(
                None,
                self.client.get_collection,
                collection_name
            )
            return info.points_count
        except:
            return 0

    async def _count_project_points(self, collection_name: str, project_id: str) -> int:
        """Count points for a specific project in a collection."""
        try:
            # Use scroll to count points with project filter
            count = 0
            next_page_offset = None

            while True:
                scroll_result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.client.scroll,
                    collection_name,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="project_id",
                                match=models.MatchValue(value=project_id)
                            )
                        ]
                    ),
                    limit=100,
                    offset=next_page_offset
                )

                points, next_page_offset = scroll_result
                count += len(points)

                if next_page_offset is None:
                    break

            return count

        except Exception as e:
            print(f"Error counting project points: {e}")
            return 0

    async def cleanup_old_collections(self, migration_plan: Dict[str, Any]) -> bool:
        """Clean up old collections after successful migration."""
        print("Cleaning up old collections...")

        if self.dry_run:
            print("[DRY RUN] Would remove old collections:")
            for collection_name in migration_plan["collections_to_remove"]:
                print(f"  - {collection_name}")
            return True

        try:
            for collection_name in migration_plan["collections_to_remove"]:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.client.delete_collection,
                    collection_name
                )
                print(f"✅ Removed old collection: {collection_name}")
                self.migration_stats["collections_removed"] += 1
                self.migration_log.append(f"Removed old collection: {collection_name}")

            return True

        except Exception as e:
            print(f"❌ Cleanup failed: {e}")
            self.migration_stats["errors"] += 1
            return False

    def _log_migration_stats(self):
        """Log migration statistics."""
        print("\n=== Migration Statistics ===")
        for key, value in self.migration_stats.items():
            print(f"{key.replace('_', ' ').title()}: {value}")

        # Save detailed log
        if not self.dry_run:
            log_path = Path("20250114-0906_data_migration.log")
            with log_path.open('w') as f:
                f.write(f"Data Migration Log - {datetime.now().isoformat()}\n\n")
                f.write("Statistics:\n")
                for key, value in self.migration_stats.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\nDetailed Log:\n")
                for entry in self.migration_log:
                    f.write(f"  {entry}\n")
            print(f"Detailed log saved to: {log_path}")


async def main():
    parser = argparse.ArgumentParser(description="Multi-Tenant Data Migration Tool")
    parser.add_argument("--analyze", action="store_true", help="Analyze existing collections")
    parser.add_argument("--migrate", action="store_true", help="Perform data migration")
    parser.add_argument("--cleanup", action="store_true", help="Clean up old collections")
    parser.add_argument("--dry-run", action="store_true", help="Perform dry run without making changes")
    parser.add_argument("--plan-file", help="Use migration plan from file")

    args = parser.parse_args()

    migration = DataMigration(dry_run=args.dry_run)

    if not await migration.initialize():
        print("Failed to initialize migration tool")
        return False

    if args.analyze:
        print("=== Analyzing Existing Collections ===")
        analysis = await migration.analyze_existing_collections()

        if "error" not in analysis:
            print(f"\nAnalysis Results:")
            print(f"  Total collections: {analysis['total_collections']}")
            print(f"  Prefix-based collections: {len(analysis['prefix_based_collections'])}")
            print(f"  Multi-tenant collections: {len(analysis['multi_tenant_collections'])}")
            print(f"  Global collections: {len(analysis['global_collections'])}")
            print(f"  Unknown collections: {len(analysis['unknown_collections'])}")
            print(f"  Migration complexity: {analysis['estimated_complexity']}")

            # Save analysis for later use
            analysis_file = Path("20250114-0906_migration_analysis.json")
            with analysis_file.open('w') as f:
                json.dump(analysis, f, indent=2)
            print(f"\nAnalysis saved to: {analysis_file}")

            if analysis['prefix_based_collections']:
                print("\nCollections requiring migration:")
                for collection in analysis['prefix_based_collections']:
                    print(f"  - {collection['name']} → project:{collection['project']}, type:{collection['type']}")
        else:
            print(f"Analysis failed: {analysis['error']}")

    elif args.migrate:
        # Load analysis
        analysis_file = Path(args.plan_file or "20250114-0906_migration_analysis.json")
        if not analysis_file.exists():
            print("No migration analysis found. Run --analyze first.")
            return False

        with analysis_file.open('r') as f:
            analysis = json.load(f)

        print("=== Performing Data Migration ===")
        success = await migration.perform_migration(analysis)

        if success:
            print("🎉 Data migration completed successfully!")
        else:
            print("❌ Data migration failed. Check logs for details.")

    elif args.cleanup:
        # Load analysis for cleanup
        analysis_file = Path(args.plan_file or "20250114-0906_migration_analysis.json")
        if not analysis_file.exists():
            print("No migration analysis found. Run --analyze first.")
            return False

        with analysis_file.open('r') as f:
            analysis = json.load(f)

        print("=== Cleaning Up Old Collections ===")
        if input("Are you sure you want to delete old collections? (yes/no): ").lower() == 'yes':
            success = await migration.cleanup_old_collections(analysis["migration_plan"])
            if success:
                print("🎉 Cleanup completed successfully!")
            else:
                print("❌ Cleanup failed. Check logs for details.")
        else:
            print("Cleanup cancelled.")

    else:
        parser.print_help()

    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)