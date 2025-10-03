#!/usr/bin/env python3
"""
Migration script to populate collection_type for existing queue items.

This script:
1. Detects existing queue items without collection_type
2. Uses CollectionTypeClassifier to infer type from collection name
3. Updates database with detected types
4. Logs migration statistics

Usage:
    python migrate_collection_types.py [--db-path PATH] [--dry-run]
"""

import argparse
import asyncio
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.python.common.core.collection_types import CollectionTypeClassifier
from src.python.common.utils.os_directories import OSDirectories


def get_default_db_path() -> str:
    """Get default database path from OS directories."""
    os_dirs = OSDirectories()
    os_dirs.ensure_directories()
    return str(os_dirs.get_state_file("workspace_state.db"))


def classify_existing_items(db_path: str) -> List[Tuple[str, str, str]]:
    """
    Classify existing queue items without collection_type.

    Args:
        db_path: Path to SQLite database

    Returns:
        List of (file_path, collection_name, detected_type) tuples
    """
    classifier = CollectionTypeClassifier()
    results = []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Find items without collection_type
    cursor.execute("""
        SELECT file_absolute_path, collection_name
        FROM ingestion_queue
        WHERE collection_type IS NULL
    """)

    for row in cursor.fetchall():
        file_path = row["file_absolute_path"]
        collection_name = row["collection_name"]

        # Classify collection type
        collection_type_enum = classifier.classify_collection_type(collection_name)
        collection_type = collection_type_enum.value

        results.append((file_path, collection_name, collection_type))

    conn.close()
    return results


def update_collection_types(
    db_path: str,
    items: List[Tuple[str, str, str]],
    dry_run: bool = False
) -> Dict[str, int]:
    """
    Update collection_type for items.

    Args:
        db_path: Path to SQLite database
        items: List of (file_path, collection_name, type) tuples
        dry_run: If True, don't actually update database

    Returns:
        Dictionary with statistics by type
    """
    stats: Dict[str, int] = {}

    if dry_run:
        print("DRY RUN MODE - No changes will be made")
        for file_path, collection_name, collection_type in items:
            stats[collection_type] = stats.get(collection_type, 0) + 1
            print(f"  Would update: {collection_name} -> {collection_type}")
        return stats

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for file_path, collection_name, collection_type in items:
        cursor.execute(
            """
            UPDATE ingestion_queue
            SET collection_type = ?
            WHERE file_absolute_path = ?
            """,
            (collection_type, file_path)
        )
        stats[collection_type] = stats.get(collection_type, 0) + 1

    conn.commit()
    conn.close()

    return stats


def verify_migration(db_path: str) -> Dict[str, int]:
    """
    Verify migration results.

    Args:
        db_path: Path to SQLite database

    Returns:
        Dictionary with count by collection_type
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT collection_type, COUNT(*) as count
        FROM ingestion_queue
        GROUP BY collection_type
    """)

    results = {row[0] or "NULL": row[1] for row in cursor.fetchall()}
    conn.close()

    return results


def main():
    """Main migration entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate collection_type for existing queue items"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to SQLite database (default: system default)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying database"
    )

    args = parser.parse_args()

    db_path = args.db_path or get_default_db_path()
    print(f"Database: {db_path}")

    # Check if database exists
    if not Path(db_path).exists():
        print(f"ERROR: Database not found at {db_path}")
        return 1

    # Check if migration is needed
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT collection_type FROM ingestion_queue LIMIT 1")
    except sqlite3.OperationalError:
        print("ERROR: collection_type column does not exist. Run migration SQL first:")
        print("  src/python/common/core/migrations/add_collection_type_to_queue.sql")
        conn.close()
        return 1
    conn.close()

    # Classify existing items
    print("\nClassifying existing queue items...")
    items = classify_existing_items(db_path)

    if not items:
        print("No items need migration (all have collection_type set)")
        return 0

    print(f"Found {len(items)} items to migrate")

    # Update collection types
    print("\nUpdating collection types...")
    stats = update_collection_types(db_path, items, dry_run=args.dry_run)

    # Print statistics
    print("\nMigration Statistics:")
    for collection_type, count in sorted(stats.items()):
        print(f"  {collection_type}: {count} items")
    print(f"  TOTAL: {sum(stats.values())} items")

    if not args.dry_run:
        # Verify migration
        print("\nVerification:")
        verification = verify_migration(db_path)
        for collection_type, count in sorted(verification.items()):
            print(f"  {collection_type}: {count} items")

        null_count = verification.get("NULL", 0)
        if null_count > 0:
            print(f"\nWARNING: {null_count} items still have NULL collection_type")
        else:
            print("\n✓ Migration completed successfully - all items have collection_type")

    return 0


if __name__ == "__main__":
    sys.exit(main())
