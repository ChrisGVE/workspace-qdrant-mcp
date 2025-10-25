"""
Integration Tests for Queue Error Handling Integration.

Tests the complete integration of queue error handling with automatic
error categorization, retry management, and error message persistence.

Test Coverage:
    - Error recording during queue operations
    - Error context preservation
    - Automatic categorization integration
    - Error message retrieval and filtering
    - Complete workflow tests (enqueue → fail → retry → success)
"""

import asyncio
import json
import socket
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pytest

from src.python.common.core.error_categorization import (
    ErrorCategorizer,
    ErrorCategory,
    ErrorSeverity,
)
from src.python.common.core.error_message_manager import ErrorMessageManager
from src.python.common.core.queue_client import QueueOperation, SQLiteQueueClient

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
async def temp_db():
    """Create temporary database for testing."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
        db_path = tmp.name

    yield db_path

    # Cleanup
    try:
        Path(db_path).unlink()
        Path(f"{db_path}-wal").unlink(missing_ok=True)
        Path(f"{db_path}-shm").unlink(missing_ok=True)
    except Exception:
        pass


@pytest.fixture
async def initialized_db(temp_db):
    """Initialize database with queue and error message schemas."""
    conn = sqlite3.connect(temp_db)

    # Load queue schema
    queue_schema_path = Path(__file__).parent.parent.parent / "src" / "python" / "common" / "core" / "queue_schema.sql"
    with open(queue_schema_path) as f:
        conn.executescript(f.read())

    # Load error messages schema
    error_schema_path = Path(__file__).parent.parent.parent / "src" / "python" / "common" / "core" / "error_messages_schema.sql"
    with open(error_schema_path) as f:
        conn.executescript(f.read())

    conn.commit()
    conn.close()

    yield temp_db


@pytest.fixture
async def queue_client(initialized_db):
    """Initialize SQLiteQueueClient with database."""
    client = SQLiteQueueClient(db_path=initialized_db)
    await client.initialize()
    yield client
    await client.close()


@pytest.fixture
async def error_manager(initialized_db):
    """Initialize ErrorMessageManager with database."""
    manager = ErrorMessageManager(db_path=initialized_db)
    success = await manager.initialize()
    assert success, "ErrorMessageManager initialization failed"
    yield manager
    await manager.close()


@pytest.fixture
def test_file(tmp_path):
    """Create a test file."""
    test_file = tmp_path / "test_file.py"
    test_file.write_text("# Test file\nprint('hello')\n")
    return str(test_file)


@pytest.fixture
def test_files(tmp_path):
    """Create multiple test files."""
    files = []
    for i in range(5):
        test_file = tmp_path / f"test_file_{i}.py"
        test_file.write_text(f"# Test file {i}\nprint({i})\n")
        files.append(str(test_file))
    return files


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


async def enqueue_test_file(
    queue_client: SQLiteQueueClient,
    file_path: str,
    collection: str = "test-collection",
    tenant_id: str = "default",
    branch: str = "main",
    priority: int = 5,
) -> str:
    """Helper to enqueue a test file."""
    file_absolute_path = str(Path(file_path).resolve())

    async with queue_client.connection_pool.get_connection_async() as conn:
        conn.execute(
            """
            INSERT INTO ingestion_queue
            (file_absolute_path, collection_name, tenant_id, branch, operation, priority)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (file_absolute_path, collection, tenant_id, branch, QueueOperation.INGEST.value, priority)
        )
        conn.commit()

    return file_absolute_path


async def get_queue_item(
    queue_client: SQLiteQueueClient,
    file_path: str
) -> dict[str, Any] | None:
    """Get queue item by file path."""
    file_absolute_path = str(Path(file_path).resolve())

    async with queue_client.connection_pool.get_connection_async() as conn:
        cursor = conn.execute(
            """
            SELECT file_absolute_path, collection_name, tenant_id, branch,
                   operation, priority, retry_count, error_message_id
            FROM ingestion_queue
            WHERE file_absolute_path = ?
            """,
            (file_absolute_path,)
        )
        row = cursor.fetchone()

        if row:
            return {
                "file_absolute_path": row[0],
                "collection_name": row[1],
                "tenant_id": row[2],
                "branch": row[3],
                "operation": row[4],
                "priority": row[5],
                "retry_count": row[6],
                "error_message_id": row[7],
            }
        return None


async def get_error_message(
    queue_client: SQLiteQueueClient,
    error_id: int
) -> dict[str, Any] | None:
    """Get error message by ID."""
    async with queue_client.connection_pool.get_connection_async() as conn:
        cursor = conn.execute(
            """
            SELECT id, timestamp, severity, category, message, context, retry_count
            FROM messages_enhanced
            WHERE id = ?
            """,
            (error_id,)
        )
        row = cursor.fetchone()

        if row:
            context = json.loads(row[5]) if row[5] else {}
            return {
                "id": row[0],
                "timestamp": row[1],
                "severity": row[2],
                "category": row[3],
                "message": row[4],
                "context": context,
                "retry_count": row[6],
            }
        return None


async def count_error_messages(
    queue_client: SQLiteQueueClient,
    severity: str | None = None,
    category: str | None = None,
) -> int:
    """Count error messages with optional filters."""
    query = "SELECT COUNT(*) FROM messages_enhanced WHERE 1=1"
    params = []

    if severity:
        query += " AND severity = ?"
        params.append(severity)

    if category:
        query += " AND category = ?"
        params.append(category)

    async with queue_client.connection_pool.get_connection_async() as conn:
        cursor = conn.execute(query, params)
        return cursor.fetchone()[0]


# =============================================================================
# SECTION A: Error Recording During Queue Operations
# =============================================================================


@pytest.mark.asyncio
async def test_mark_error_records_to_messages_table(queue_client, test_file):
    """Test that mark_error() records errors to messages table."""
    # Enqueue test file
    file_path = await enqueue_test_file(queue_client, test_file)

    # Mark error
    exception = FileNotFoundError("File not found")
    should_retry, error_id = await queue_client.mark_error(
        file_path=file_path,
        exception=exception,
        max_retries=3
    )

    # Verify error was created
    assert error_id is not None
    assert should_retry is True  # Should retry on first error

    # Verify error message exists in database
    error_msg = await get_error_message(queue_client, error_id)
    assert error_msg is not None
    assert error_msg["message"] == "File not found"
    assert error_msg["category"] == "file_corrupt"
    assert error_msg["severity"] == "error"

    # Verify queue item was updated
    queue_item = await get_queue_item(queue_client, file_path)
    assert queue_item is not None
    assert queue_item["error_message_id"] == error_id
    assert queue_item["retry_count"] == 1


@pytest.mark.asyncio
async def test_error_messages_have_proper_severity_and_category(queue_client, test_files):
    """Test that different exception types get correct severity/category."""
    test_cases = [
        (FileNotFoundError("File not found"), "file_corrupt", "error"),
        (PermissionError("Access denied"), "permission_denied", "error"),
        (TimeoutError("Connection timeout"), "timeout", "error"),
        (TimeoutError("Socket timeout"), "timeout", "error"),
        (ConnectionError("Network error"), "network", "error"),
    ]

    for i, (exception, expected_category, expected_severity) in enumerate(test_cases):
        file_path = await enqueue_test_file(queue_client, test_files[i])

        should_retry, error_id = await queue_client.mark_error(
            file_path=file_path,
            exception=exception,
            max_retries=3
        )

        error_msg = await get_error_message(queue_client, error_id)
        assert error_msg is not None
        assert error_msg["category"] == expected_category, \
            f"Expected category {expected_category}, got {error_msg['category']}"
        assert error_msg["severity"] == expected_severity, \
            f"Expected severity {expected_severity}, got {error_msg['severity']}"


@pytest.mark.asyncio
async def test_retry_count_incrementation_with_error_recording(queue_client, test_file):
    """Test that retry count increments with each error recording."""
    file_path = await enqueue_test_file(queue_client, test_file)

    exception = RuntimeError("Processing failed")

    # First error
    should_retry_1, error_id_1 = await queue_client.mark_error(
        file_path=file_path,
        exception=exception,
        max_retries=3
    )

    assert should_retry_1 is True
    queue_item_1 = await get_queue_item(queue_client, file_path)
    assert queue_item_1["retry_count"] == 1

    error_msg_1 = await get_error_message(queue_client, error_id_1)
    assert error_msg_1["context"]["retry_count"] == 1

    # Second error
    should_retry_2, error_id_2 = await queue_client.mark_error(
        file_path=file_path,
        exception=exception,
        max_retries=3
    )

    assert should_retry_2 is True
    queue_item_2 = await get_queue_item(queue_client, file_path)
    assert queue_item_2["retry_count"] == 2

    error_msg_2 = await get_error_message(queue_client, error_id_2)
    assert error_msg_2["context"]["retry_count"] == 2

    # Third error
    should_retry_3, error_id_3 = await queue_client.mark_error(
        file_path=file_path,
        exception=exception,
        max_retries=3
    )

    # Max retries reached, should not retry
    assert should_retry_3 is False

    # Queue item should be removed
    queue_item_3 = await get_queue_item(queue_client, file_path)
    assert queue_item_3 is None

    # Error message should still exist
    error_msg_3 = await get_error_message(queue_client, error_id_3)
    assert error_msg_3 is not None
    assert error_msg_3["context"]["retry_count"] == 3


@pytest.mark.asyncio
async def test_max_retries_behavior_with_error_recording(queue_client, test_file):
    """Test that max retries removes item from queue but keeps error message."""
    file_path = await enqueue_test_file(queue_client, test_file)

    exception = RuntimeError("Persistent error")
    error_ids = []

    # Create 3 errors with max_retries=3
    for i in range(3):
        should_retry, error_id = await queue_client.mark_error(
            file_path=file_path,
            exception=exception,
            max_retries=3
        )
        error_ids.append(error_id)

        if i < 2:
            assert should_retry is True
        else:
            assert should_retry is False

    # Verify item removed from queue
    queue_item = await get_queue_item(queue_client, file_path)
    assert queue_item is None

    # Verify all error messages still exist
    for error_id in error_ids:
        error_msg = await get_error_message(queue_client, error_id)
        assert error_msg is not None


# =============================================================================
# SECTION B: Error Context Preservation
# =============================================================================


@pytest.mark.asyncio
async def test_context_preservation_file_path_collection_tenant(queue_client, test_file):
    """Test that file_path, collection_name, tenant_id are captured in context."""
    file_path = await enqueue_test_file(
        queue_client,
        test_file,
        collection="test-project",
        tenant_id="tenant-123",
        branch="feature/test"
    )

    exception = RuntimeError("Test error")
    should_retry, error_id = await queue_client.mark_error(
        file_path=file_path,
        exception=exception,
        max_retries=3
    )

    error_msg = await get_error_message(queue_client, error_id)
    context = error_msg["context"]

    assert context["file_path"] == file_path
    assert context["collection"] == "test-project"
    assert context["tenant_id"] == "tenant-123"
    assert context["branch"] == "feature/test"


@pytest.mark.asyncio
async def test_context_preservation_queue_type_and_retry_count(queue_client, test_file):
    """Test that queue type and retry count are preserved in context."""
    file_path = await enqueue_test_file(queue_client, test_file)

    exception = RuntimeError("Test error")

    # First error
    should_retry_1, error_id_1 = await queue_client.mark_error(
        file_path=file_path,
        exception=exception,
        max_retries=3
    )

    error_msg_1 = await get_error_message(queue_client, error_id_1)
    assert error_msg_1["context"]["queue_type"] == "ingestion_queue"
    assert error_msg_1["context"]["retry_count"] == 1

    # Second error
    should_retry_2, error_id_2 = await queue_client.mark_error(
        file_path=file_path,
        exception=exception,
        max_retries=3
    )

    error_msg_2 = await get_error_message(queue_client, error_id_2)
    assert error_msg_2["context"]["queue_type"] == "ingestion_queue"
    assert error_msg_2["context"]["retry_count"] == 2


@pytest.mark.asyncio
async def test_context_preservation_branch_information(queue_client, test_file):
    """Test that branch information is captured in context."""
    file_path = await enqueue_test_file(
        queue_client,
        test_file,
        branch="feature/new-feature"
    )

    exception = RuntimeError("Test error")
    should_retry, error_id = await queue_client.mark_error(
        file_path=file_path,
        exception=exception,
        max_retries=3
    )

    error_msg = await get_error_message(queue_client, error_id)
    assert error_msg["context"]["branch"] == "feature/new-feature"


# =============================================================================
# SECTION C: Automatic Categorization Integration
# =============================================================================


@pytest.mark.asyncio
async def test_categorization_file_not_found_error(queue_client, test_file):
    """Test FileNotFoundError → FILE_CORRUPT categorization."""
    file_path = await enqueue_test_file(queue_client, test_file)

    exception = FileNotFoundError("/path/to/missing/file.txt")
    should_retry, error_id = await queue_client.mark_error(
        file_path=file_path,
        exception=exception,
        max_retries=3
    )

    error_msg = await get_error_message(queue_client, error_id)
    assert error_msg["category"] == "file_corrupt"
    assert error_msg["severity"] == "error"


@pytest.mark.asyncio
async def test_categorization_permission_error(queue_client, test_file):
    """Test PermissionError → PERMISSION_DENIED categorization."""
    file_path = await enqueue_test_file(queue_client, test_file)

    exception = PermissionError("Permission denied")
    should_retry, error_id = await queue_client.mark_error(
        file_path=file_path,
        exception=exception,
        max_retries=3
    )

    error_msg = await get_error_message(queue_client, error_id)
    assert error_msg["category"] == "permission_denied"
    assert error_msg["severity"] == "error"


@pytest.mark.asyncio
async def test_categorization_timeout_error(queue_client, test_file):
    """Test TimeoutError → TIMEOUT categorization."""
    file_path = await enqueue_test_file(queue_client, test_file)

    exception = TimeoutError("Connection timed out")
    should_retry, error_id = await queue_client.mark_error(
        file_path=file_path,
        exception=exception,
        max_retries=3
    )

    error_msg = await get_error_message(queue_client, error_id)
    assert error_msg["category"] == "timeout"
    assert error_msg["severity"] == "error"


@pytest.mark.asyncio
async def test_categorization_network_error(queue_client, test_file):
    """Test ConnectionError → NETWORK categorization."""
    file_path = await enqueue_test_file(queue_client, test_file)

    exception = ConnectionError("Connection refused")
    should_retry, error_id = await queue_client.mark_error(
        file_path=file_path,
        exception=exception,
        max_retries=3
    )

    error_msg = await get_error_message(queue_client, error_id)
    assert error_msg["category"] == "network"
    assert error_msg["severity"] == "error"


@pytest.mark.asyncio
async def test_categorization_generic_exception(queue_client, test_file):
    """Test generic Exception → UNKNOWN categorization."""
    file_path = await enqueue_test_file(queue_client, test_file)

    # Use a truly generic exception
    exception = Exception("Generic error")
    should_retry, error_id = await queue_client.mark_error(
        file_path=file_path,
        exception=exception,
        max_retries=3
    )

    error_msg = await get_error_message(queue_client, error_id)
    # Generic exceptions typically fall into processing_failed or unknown
    assert error_msg["category"] in ["processing_failed", "unknown"]
    assert error_msg["severity"] == "error"


# =============================================================================
# SECTION D: Error Message Retrieval and Filtering
# =============================================================================


@pytest.mark.asyncio
async def test_filter_errors_by_severity(queue_client, error_manager, test_files):
    """Test filtering error messages by severity."""
    # Create errors with different severities by using error_manager directly
    for i, file_path in enumerate(test_files[:3]):
        await enqueue_test_file(queue_client, file_path)

        # Create errors with different severities
        if i == 0:
            # Error severity
            await error_manager.record_error(
                exception=RuntimeError("Error severity"),
                context={"file_path": file_path}
            )
        elif i == 1:
            # Warning severity (using manual override)
            await error_manager.record_error(
                message_override="Warning message",
                context={"file_path": file_path},
                severity_override=ErrorSeverity.WARNING
            )
        else:
            # Info severity
            await error_manager.record_error(
                message_override="Info message",
                context={"file_path": file_path},
                severity_override=ErrorSeverity.INFO
            )

    # Count by severity
    error_count = await count_error_messages(queue_client, severity="error")
    warning_count = await count_error_messages(queue_client, severity="warning")
    info_count = await count_error_messages(queue_client, severity="info")

    assert error_count >= 1
    assert warning_count >= 1
    assert info_count >= 1


@pytest.mark.asyncio
async def test_filter_errors_by_category(queue_client, test_files):
    """Test filtering error messages by category."""
    # Create errors with different categories
    exceptions = [
        FileNotFoundError("File not found"),  # file_corrupt
        PermissionError("Access denied"),     # permission_denied
        TimeoutError("Timeout"),              # timeout
    ]

    for i, exception in enumerate(exceptions):
        file_path = await enqueue_test_file(queue_client, test_files[i])
        await queue_client.mark_error(
            file_path=file_path,
            exception=exception,
            max_retries=3
        )

    # Count by category
    file_corrupt_count = await count_error_messages(queue_client, category="file_corrupt")
    permission_denied_count = await count_error_messages(queue_client, category="permission_denied")
    timeout_count = await count_error_messages(queue_client, category="timeout")

    assert file_corrupt_count >= 1
    assert permission_denied_count >= 1
    assert timeout_count >= 1


@pytest.mark.asyncio
async def test_error_statistics_generation(queue_client, test_files):
    """Test error statistics calculation."""
    # Create multiple errors
    for file_path in test_files:
        await enqueue_test_file(queue_client, file_path)
        await queue_client.mark_error(
            file_path=file_path,
            exception=RuntimeError("Test error"),
            max_retries=3
        )

    # Get total error count
    total_errors = await count_error_messages(queue_client)
    assert total_errors >= len(test_files)


# =============================================================================
# SECTION E: Complete Workflow Tests
# =============================================================================


@pytest.mark.asyncio
async def test_workflow_enqueue_fail_retry_success(queue_client, test_file):
    """Test complete workflow: enqueue → fail → mark error → retry → success."""
    # Step 1: Enqueue file
    file_path = await enqueue_test_file(queue_client, test_file)
    queue_item = await get_queue_item(queue_client, file_path)
    assert queue_item is not None
    assert queue_item["retry_count"] == 0

    # Step 2: First processing fails
    exception = RuntimeError("Temporary processing error")
    should_retry, error_id = await queue_client.mark_error(
        file_path=file_path,
        exception=exception,
        max_retries=3
    )

    # Step 3: Verify error recorded and retry_count incremented
    assert should_retry is True
    queue_item = await get_queue_item(queue_client, file_path)
    assert queue_item["retry_count"] == 1
    assert queue_item["error_message_id"] == error_id

    # Error message should exist
    error_msg = await get_error_message(queue_client, error_id)
    assert error_msg is not None

    # Step 4: Second processing succeeds - mark as complete
    await queue_client.mark_complete(file_path)

    # Step 5: Verify item removed from queue
    queue_item = await get_queue_item(queue_client, file_path)
    assert queue_item is None

    # Error record should still exist
    error_msg = await get_error_message(queue_client, error_id)
    assert error_msg is not None


@pytest.mark.asyncio
async def test_workflow_missing_metadata_error_retry_success(queue_client, error_manager, test_file):
    """Test workflow: missing metadata → error → metadata added → retry → success."""
    # Step 1: Enqueue with "missing" metadata (simulated)
    file_path = await enqueue_test_file(queue_client, test_file)

    # Step 2: Mark error for missing metadata
    exception = ValueError("Missing required metadata")
    should_retry, error_id = await queue_client.mark_error(
        file_path=file_path,
        exception=exception,
        error_context={"reason": "missing_metadata"},
        max_retries=3
    )

    assert should_retry is True
    error_msg = await get_error_message(queue_client, error_id)
    assert error_msg is not None
    # ValueError typically categorized as parse_error
    assert error_msg["category"] in ["parse_error", "metadata_invalid"]

    # Step 3: Simulate metadata being added (update queue item)
    # In real workflow, metadata would be added externally
    queue_item = await get_queue_item(queue_client, file_path)
    assert queue_item["retry_count"] == 1

    # Step 4: Retry succeeds - mark as complete
    await queue_client.mark_complete(file_path)

    # Verify item removed from queue
    queue_item = await get_queue_item(queue_client, file_path)
    assert queue_item is None


@pytest.mark.asyncio
async def test_workflow_max_retries_reached_item_removed(queue_client, test_file):
    """Test workflow: max retries reached → item removed from queue."""
    file_path = await enqueue_test_file(queue_client, test_file)

    exception = RuntimeError("Persistent error")
    error_ids = []
    max_retries = 3

    # Fail max_retries times
    for i in range(max_retries):
        should_retry, error_id = await queue_client.mark_error(
            file_path=file_path,
            exception=exception,
            max_retries=max_retries
        )
        error_ids.append(error_id)

        if i < max_retries - 1:
            # Should still retry
            assert should_retry is True
            queue_item = await get_queue_item(queue_client, file_path)
            assert queue_item is not None
            assert queue_item["retry_count"] == i + 1
        else:
            # Max retries reached
            assert should_retry is False
            queue_item = await get_queue_item(queue_client, file_path)
            assert queue_item is None

    # Verify all error messages exist
    assert len(error_ids) == max_retries
    for error_id in error_ids:
        error_msg = await get_error_message(queue_client, error_id)
        assert error_msg is not None


# =============================================================================
# RUN TESTS
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
