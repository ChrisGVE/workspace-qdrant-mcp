#!/usr/bin/env python3
"""
Simple test script to verify update_retry functionality.
"""

import asyncio
import tempfile
from pathlib import Path
from datetime import datetime, timedelta, timezone

from src.python.common.core.sqlite_state_manager import SQLiteStateManager


async def test_update_retry():
    """Test update_retry method for retry scheduling."""

    # Create temporary database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"

        # Initialize state manager
        state_manager = SQLiteStateManager(str(db_path))
        await state_manager.initialize()

        try:
            # Test 1: Try to use update_retry method
            file_path = "/test/file.py"
            queue_id = str(Path(file_path).resolve())

            # First enqueue a file (using the new ingestion_queue table)
            conn = await state_manager._get_connection()
            conn.execute(
                """
                INSERT INTO ingestion_queue (
                    file_absolute_path, collection_name, operation, priority
                ) VALUES (?, ?, ?, ?)
                """,
                (queue_id, "test-collection", "ingest", 5)
            )
            conn.commit()
            print(f"✓ Enqueued file: {queue_id}")

            # Test if update_retry method exists
            if hasattr(state_manager, 'update_retry'):
                # Test updating retry schedule
                retry_after = datetime.now(timezone.utc) + timedelta(seconds=60)
                success = await state_manager.update_retry(
                    queue_id=queue_id,
                    retry_count=1,
                    retry_after=retry_after
                )
                print(f"✓ update_retry exists and returned: {success}")
            else:
                print("✗ update_retry method does NOT exist yet")
                print("  Need to implement it!")

        finally:
            await state_manager.close()
            print("✓ Test completed")


if __name__ == "__main__":
    asyncio.run(test_update_retry())
