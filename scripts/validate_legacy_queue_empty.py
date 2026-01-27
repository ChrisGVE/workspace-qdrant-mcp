#!/usr/bin/env python3
"""
Legacy Queue Migration Validator

Validates that legacy queue tables (ingestion_queue, content_ingestion_queue)
are empty before removal in Phase 4 cleanup.

This script should be run before executing DROP TABLE statements to ensure
no data is lost.

Usage:
    python scripts/validate_legacy_queue_empty.py
    python scripts/validate_legacy_queue_empty.py --db-path /path/to/state.db

Exit codes:
    0 - All legacy queues are empty (safe to remove)
    1 - Legacy queues contain data (migration incomplete)
    2 - Database error
"""

import argparse
import sqlite3
import sys
from pathlib import Path


def get_default_db_path() -> str:
    """Get default database path based on OS."""
    import platform

    system = platform.system()

    if system == "Darwin":  # macOS
        return str(Path.home() / "Library/Application Support/workspace-qdrant-mcp/state.db")
    elif system == "Windows":
        local_app_data = Path.home() / "AppData/Local/workspace-qdrant-mcp"
        return str(local_app_data / "state.db")
    else:  # Linux and others
        xdg_data = Path.home() / ".local/share/workspace-qdrant-mcp"
        return str(xdg_data / "state.db")


def check_table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """Check if a table exists in the database."""
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,)
    )
    return cursor.fetchone() is not None


def count_table_rows(conn: sqlite3.Connection, table_name: str) -> int:
    """Count rows in a table."""
    cursor = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
    return cursor.fetchone()[0]


def get_row_breakdown(conn: sqlite3.Connection, table_name: str) -> dict:
    """Get breakdown of rows by status."""
    cursor = conn.execute(
        f"SELECT status, COUNT(*) FROM {table_name} GROUP BY status"
    )
    return {row[0]: row[1] for row in cursor.fetchall()}


def validate_legacy_queues(db_path: str) -> tuple[bool, dict]:
    """
    Validate that legacy queues are empty.

    Returns:
        tuple: (is_safe, details)
            is_safe: True if all legacy queues are empty
            details: Dictionary with validation details
    """
    details = {
        "db_path": db_path,
        "ingestion_queue": {"exists": False, "count": 0, "by_status": {}},
        "content_ingestion_queue": {"exists": False, "count": 0, "by_status": {}},
        "is_safe": True,
        "errors": [],
    }

    try:
        conn = sqlite3.connect(db_path)
    except Exception as e:
        details["errors"].append(f"Failed to connect to database: {e}")
        details["is_safe"] = False
        return False, details

    try:
        # Check ingestion_queue
        if check_table_exists(conn, "ingestion_queue"):
            details["ingestion_queue"]["exists"] = True
            count = count_table_rows(conn, "ingestion_queue")
            details["ingestion_queue"]["count"] = count
            if count > 0:
                details["ingestion_queue"]["by_status"] = get_row_breakdown(conn, "ingestion_queue")
                details["is_safe"] = False

        # Check content_ingestion_queue
        if check_table_exists(conn, "content_ingestion_queue"):
            details["content_ingestion_queue"]["exists"] = True
            count = count_table_rows(conn, "content_ingestion_queue")
            details["content_ingestion_queue"]["count"] = count
            if count > 0:
                details["content_ingestion_queue"]["by_status"] = get_row_breakdown(conn, "content_ingestion_queue")
                details["is_safe"] = False

    except Exception as e:
        details["errors"].append(f"Query error: {e}")
        details["is_safe"] = False

    finally:
        conn.close()

    return details["is_safe"], details


def main():
    parser = argparse.ArgumentParser(
        description="Validate legacy queue tables are empty before removal"
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="Path to SQLite state database (default: OS-specific location)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )

    args = parser.parse_args()

    db_path = args.db_path or get_default_db_path()

    # Check if database exists
    if not Path(db_path).exists():
        if args.json:
            import json
            print(json.dumps({"error": f"Database not found: {db_path}", "is_safe": True}))
        else:
            print(f"Database not found: {db_path}")
            print("No legacy data to migrate - safe to proceed.")
        sys.exit(0)

    is_safe, details = validate_legacy_queues(db_path)

    if args.json:
        import json
        print(json.dumps(details, indent=2))
    else:
        print(f"\nLegacy Queue Validation Report")
        print(f"{'=' * 40}")
        print(f"Database: {db_path}")
        print()

        # ingestion_queue
        iq = details["ingestion_queue"]
        if iq["exists"]:
            print(f"ingestion_queue:")
            print(f"  Total rows: {iq['count']}")
            if iq["by_status"]:
                for status, count in iq["by_status"].items():
                    print(f"    {status}: {count}")
        else:
            print("ingestion_queue: NOT FOUND (already removed or never created)")

        print()

        # content_ingestion_queue
        ciq = details["content_ingestion_queue"]
        if ciq["exists"]:
            print(f"content_ingestion_queue:")
            print(f"  Total rows: {ciq['count']}")
            if ciq["by_status"]:
                for status, count in ciq["by_status"].items():
                    print(f"    {status}: {count}")
        else:
            print("content_ingestion_queue: NOT FOUND (already removed or never created)")

        print()
        print("=" * 40)

        if is_safe:
            print("\n✅ SAFE TO REMOVE: Legacy queues are empty or don't exist.")
            print("   You can proceed with DROP TABLE statements.")
        else:
            print("\n❌ NOT SAFE: Legacy queues contain data!")
            print("   Complete the migration before removing tables.")
            print("\n   Options:")
            print("   1. Run phase3_cutover.sh to complete migration")
            print("   2. Manually process remaining items")
            print("   3. Use --force to drop tables anyway (DATA LOSS)")

        if details["errors"]:
            print("\nErrors encountered:")
            for error in details["errors"]:
                print(f"  - {error}")

    sys.exit(0 if is_safe else 1)


if __name__ == "__main__":
    main()
