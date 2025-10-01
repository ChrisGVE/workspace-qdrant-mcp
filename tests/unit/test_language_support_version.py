"""Unit tests for language support version tracking and hash comparison.

Tests the LanguageSupportVersionTracker class functionality including:
- Version and hash retrieval
- Hash calculation
- Update detection
- Database operations
"""

import hashlib
import sqlite3
import tempfile
from pathlib import Path

import pytest

from src.python.common.core.language_support_version import (
    LanguageSupportVersionTracker,
)


@pytest.fixture
def temp_db():
    """Create a temporary database with language_support_version table."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    # Create the table structure
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE language_support_version (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            yaml_hash TEXT NOT NULL UNIQUE,
            loaded_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            language_count INTEGER NOT NULL DEFAULT 0,
            last_checked_at TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()

    yield db_path

    # Cleanup
    db_path.unlink(missing_ok=True)


@pytest.fixture
def temp_yaml_file():
    """Create a temporary YAML file for testing."""
    with tempfile.NamedTemporaryFile(
        suffix=".yaml", delete=False, mode="w"
    ) as f:
        f.write("file_extensions:\n  .py: python\n  .rs: rust\n")
        yaml_path = Path(f.name)

    yield yaml_path

    # Cleanup
    yaml_path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_initialization(temp_db):
    """Test tracker initialization."""
    tracker = LanguageSupportVersionTracker(temp_db)
    await tracker.initialize()

    assert tracker.connection is not None
    assert tracker.db_path == temp_db

    await tracker.close()
    assert tracker.connection is None


@pytest.mark.asyncio
async def test_get_current_version_empty(temp_db):
    """Test getting version when database is empty."""
    tracker = LanguageSupportVersionTracker(temp_db)
    await tracker.initialize()

    version = await tracker.get_current_version()
    assert version is None

    await tracker.close()


@pytest.mark.asyncio
async def test_get_content_hash_empty(temp_db):
    """Test getting content hash when database is empty."""
    tracker = LanguageSupportVersionTracker(temp_db)
    await tracker.initialize()

    hash_value = await tracker.get_content_hash()
    assert hash_value is None

    await tracker.close()


@pytest.mark.asyncio
async def test_update_version(temp_db):
    """Test updating version and hash."""
    tracker = LanguageSupportVersionTracker(temp_db)
    await tracker.initialize()

    test_hash = "a3f5c8b2e4d6f1a2c3b4d5e6f7a8b9c0"
    await tracker.update_version("1.0.0", test_hash, language_count=500)

    # Verify version was stored
    version = await tracker.get_current_version()
    assert version == test_hash

    # Verify content hash
    content_hash = await tracker.get_content_hash()
    assert content_hash == test_hash

    await tracker.close()


@pytest.mark.asyncio
async def test_update_version_replace(temp_db):
    """Test that update_version replaces existing record with same hash."""
    tracker = LanguageSupportVersionTracker(temp_db)
    await tracker.initialize()

    test_hash = "a3f5c8b2e4d6f1a2c3b4d5e6f7a8b9c0"

    # Insert first version
    await tracker.update_version("1.0.0", test_hash, language_count=500)

    # Update with same hash but different language count
    await tracker.update_version("1.0.0", test_hash, language_count=600)

    # Verify only one record exists with updated count
    cursor = tracker.connection.cursor()
    cursor.execute("SELECT COUNT(*), language_count FROM language_support_version")
    row = cursor.fetchone()
    cursor.close()

    assert row[0] == 1  # Only one record
    assert row[1] == 600  # Updated language count

    await tracker.close()


def test_calculate_file_hash(temp_yaml_file):
    """Test file hash calculation."""
    tracker = LanguageSupportVersionTracker(":memory:")

    # Calculate hash
    hash_value = tracker.calculate_file_hash(temp_yaml_file)

    # Verify it's a valid SHA256 hash (64 hex characters)
    assert len(hash_value) == 64
    assert all(c in "0123456789abcdef" for c in hash_value)

    # Verify hash is deterministic
    hash_value2 = tracker.calculate_file_hash(temp_yaml_file)
    assert hash_value == hash_value2

    # Verify hash matches expected value
    with open(temp_yaml_file, "rb") as f:
        expected_hash = hashlib.sha256(f.read()).hexdigest()
    assert hash_value == expected_hash


def test_calculate_file_hash_nonexistent():
    """Test hash calculation with non-existent file."""
    tracker = LanguageSupportVersionTracker(":memory:")

    with pytest.raises(FileNotFoundError):
        tracker.calculate_file_hash(Path("/nonexistent/file.yaml"))


@pytest.mark.asyncio
async def test_needs_update_no_stored_hash(temp_db, temp_yaml_file):
    """Test needs_update when no hash is stored."""
    tracker = LanguageSupportVersionTracker(temp_db)
    await tracker.initialize()

    # Should return True when no hash exists
    needs_update = await tracker.needs_update(temp_yaml_file)
    assert needs_update is True

    await tracker.close()


@pytest.mark.asyncio
async def test_needs_update_unchanged_file(temp_db, temp_yaml_file):
    """Test needs_update when file hasn't changed."""
    tracker = LanguageSupportVersionTracker(temp_db)
    await tracker.initialize()

    # Calculate and store initial hash
    current_hash = tracker.calculate_file_hash(temp_yaml_file)
    await tracker.update_version("1.0.0", current_hash, language_count=2)

    # Check if update needed (should be False since file unchanged)
    needs_update = await tracker.needs_update(temp_yaml_file)
    assert needs_update is False

    await tracker.close()


@pytest.mark.asyncio
async def test_needs_update_changed_file(temp_db, temp_yaml_file):
    """Test needs_update when file has changed."""
    tracker = LanguageSupportVersionTracker(temp_db)
    await tracker.initialize()

    # Calculate and store initial hash
    initial_hash = tracker.calculate_file_hash(temp_yaml_file)
    await tracker.update_version("1.0.0", initial_hash, language_count=2)

    # Modify the file
    with open(temp_yaml_file, "a") as f:
        f.write("  .js: javascript\n")

    # Check if update needed (should be True since file changed)
    needs_update = await tracker.needs_update(temp_yaml_file)
    assert needs_update is True

    # Verify the hash is actually different
    new_hash = tracker.calculate_file_hash(temp_yaml_file)
    assert new_hash != initial_hash

    await tracker.close()


@pytest.mark.asyncio
async def test_needs_update_updates_last_checked(temp_db, temp_yaml_file):
    """Test that needs_update updates last_checked_at timestamp."""
    tracker = LanguageSupportVersionTracker(temp_db)
    await tracker.initialize()

    # Store initial version
    current_hash = tracker.calculate_file_hash(temp_yaml_file)
    await tracker.update_version("1.0.0", current_hash, language_count=2)

    # Get initial last_checked_at
    cursor = tracker.connection.cursor()
    cursor.execute(
        "SELECT last_checked_at FROM language_support_version ORDER BY loaded_at DESC LIMIT 1"
    )
    initial_checked = cursor.fetchone()[0]
    cursor.close()

    # Wait a moment and check again
    import asyncio
    await asyncio.sleep(0.1)

    # Call needs_update (should update last_checked_at)
    await tracker.needs_update(temp_yaml_file)

    # Verify last_checked_at was updated
    cursor = tracker.connection.cursor()
    cursor.execute(
        "SELECT last_checked_at FROM language_support_version ORDER BY loaded_at DESC LIMIT 1"
    )
    new_checked = cursor.fetchone()[0]
    cursor.close()

    assert new_checked != initial_checked

    await tracker.close()


@pytest.mark.asyncio
async def test_error_on_uninitialized_connection(temp_db):
    """Test that operations fail on uninitialized connection."""
    tracker = LanguageSupportVersionTracker(temp_db)

    # Should raise RuntimeError before initialization
    with pytest.raises(RuntimeError, match="not initialized"):
        await tracker.get_current_version()

    with pytest.raises(RuntimeError, match="not initialized"):
        await tracker.get_content_hash()

    with pytest.raises(RuntimeError, match="not initialized"):
        await tracker.update_version("1.0.0", "hash123")


@pytest.mark.asyncio
async def test_multiple_versions(temp_db):
    """Test storing multiple versions with different hashes."""
    tracker = LanguageSupportVersionTracker(temp_db)
    await tracker.initialize()

    # Store first version
    hash1 = "a" * 64
    await tracker.update_version("1.0.0", hash1, language_count=100)

    # Store second version with different hash
    hash2 = "b" * 64
    await tracker.update_version("2.0.0", hash2, language_count=200)

    # Should return the most recent version
    current = await tracker.get_current_version()
    assert current == hash2

    # Verify both records exist
    cursor = tracker.connection.cursor()
    cursor.execute("SELECT COUNT(*) FROM language_support_version")
    count = cursor.fetchone()[0]
    cursor.close()

    assert count == 2

    await tracker.close()


@pytest.mark.asyncio
async def test_context_manager_cleanup(temp_db):
    """Test that connection is properly cleaned up."""
    tracker = LanguageSupportVersionTracker(temp_db)
    await tracker.initialize()

    # Verify connection is open
    assert tracker.connection is not None

    # Close and verify cleanup
    await tracker.close()
    assert tracker.connection is None

    # Should be able to reinitialize
    await tracker.initialize()
    assert tracker.connection is not None

    await tracker.close()
