"""
Unit tests for Task 432: Library additive deletion policy.
Tests library deletion tracking without removing vectors from Qdrant.
"""

import pytest
import tempfile
from pathlib import Path

from common.core.sqlite_state_manager import SQLiteStateManager


@pytest.fixture
async def state_manager():
    """Create a test SQLiteStateManager with temporary database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_state.db"
        manager = SQLiteStateManager(db_path=str(db_path))
        await manager.initialize()
        yield manager
        await manager.close()


class TestMarkLibraryDeleted:
    """Tests for marking library files as deleted."""

    @pytest.mark.asyncio
    async def test_mark_library_deleted_basic(self, state_manager):
        """Test marking a library file as deleted."""
        success = await state_manager.mark_library_deleted(
            library_name="numpy",
            file_path="numpy/linalg/norm.py",
        )

        assert success is True

        # Verify it's marked as deleted
        is_deleted = await state_manager.is_library_file_deleted(
            library_name="numpy",
            file_path="numpy/linalg/norm.py",
        )
        assert is_deleted is True

    @pytest.mark.asyncio
    async def test_mark_library_deleted_with_metadata(self, state_manager):
        """Test marking a file as deleted with metadata."""
        metadata = {"reason": "outdated", "deleted_by": "admin"}

        success = await state_manager.mark_library_deleted(
            library_name="pandas",
            file_path="pandas/core/frame.py",
            metadata=metadata,
        )

        assert success is True

        # Verify metadata was stored
        deletions = await state_manager.list_library_deletions(
            library_name="pandas"
        )
        assert len(deletions) == 1
        assert deletions[0]["metadata"] == metadata

    @pytest.mark.asyncio
    async def test_mark_same_file_twice_resets_status(self, state_manager):
        """Test that marking the same file again resets deletion status."""
        # Mark as deleted
        await state_manager.mark_library_deleted("numpy", "file.py")

        # Mark as re-ingested
        await state_manager.re_ingest_library_file("numpy", "file.py")

        # Verify it's now not deleted
        is_deleted = await state_manager.is_library_file_deleted("numpy", "file.py")
        assert is_deleted is False

        # Mark as deleted again - should reset
        await state_manager.mark_library_deleted("numpy", "file.py")

        # Should be deleted again
        is_deleted = await state_manager.is_library_file_deleted("numpy", "file.py")
        assert is_deleted is True


class TestIsLibraryFileDeleted:
    """Tests for checking library file deletion status."""

    @pytest.mark.asyncio
    async def test_file_not_deleted(self, state_manager):
        """Test checking a file that was never deleted."""
        is_deleted = await state_manager.is_library_file_deleted(
            library_name="unknown",
            file_path="unknown/file.py",
        )
        assert is_deleted is False

    @pytest.mark.asyncio
    async def test_deleted_file_shows_as_deleted(self, state_manager):
        """Test checking a file that was deleted."""
        await state_manager.mark_library_deleted("numpy", "file.py")

        is_deleted = await state_manager.is_library_file_deleted("numpy", "file.py")
        assert is_deleted is True

    @pytest.mark.asyncio
    async def test_re_ingested_file_not_deleted(self, state_manager):
        """Test that re-ingested files are not considered deleted."""
        await state_manager.mark_library_deleted("numpy", "file.py")
        await state_manager.re_ingest_library_file("numpy", "file.py")

        is_deleted = await state_manager.is_library_file_deleted("numpy", "file.py")
        assert is_deleted is False


class TestReIngestLibraryFile:
    """Tests for marking deleted files as re-ingested."""

    @pytest.mark.asyncio
    async def test_re_ingest_deleted_file(self, state_manager):
        """Test marking a deleted file for re-ingestion."""
        await state_manager.mark_library_deleted("numpy", "file.py")

        success = await state_manager.re_ingest_library_file("numpy", "file.py")
        assert success is True

        # Verify it's no longer deleted
        is_deleted = await state_manager.is_library_file_deleted("numpy", "file.py")
        assert is_deleted is False

    @pytest.mark.asyncio
    async def test_re_ingest_non_existent_file(self, state_manager):
        """Test re-ingesting a file that was never deleted."""
        success = await state_manager.re_ingest_library_file("unknown", "file.py")
        assert success is False

    @pytest.mark.asyncio
    async def test_re_ingest_already_re_ingested_file(self, state_manager):
        """Test re-ingesting a file that was already re-ingested."""
        await state_manager.mark_library_deleted("numpy", "file.py")
        await state_manager.re_ingest_library_file("numpy", "file.py")

        # Try to re-ingest again
        success = await state_manager.re_ingest_library_file("numpy", "file.py")
        assert success is False  # Already re-ingested


class TestListLibraryDeletions:
    """Tests for listing deleted library files."""

    @pytest.mark.asyncio
    async def test_list_all_deletions(self, state_manager):
        """Test listing all deleted files."""
        await state_manager.mark_library_deleted("numpy", "file1.py")
        await state_manager.mark_library_deleted("pandas", "file2.py")
        await state_manager.mark_library_deleted("numpy", "file3.py")

        deletions = await state_manager.list_library_deletions()
        assert len(deletions) == 3

    @pytest.mark.asyncio
    async def test_list_deletions_by_library(self, state_manager):
        """Test listing deleted files filtered by library."""
        await state_manager.mark_library_deleted("numpy", "file1.py")
        await state_manager.mark_library_deleted("pandas", "file2.py")
        await state_manager.mark_library_deleted("numpy", "file3.py")

        deletions = await state_manager.list_library_deletions(library_name="numpy")
        assert len(deletions) == 2
        for d in deletions:
            assert d["library_name"] == "numpy"

    @pytest.mark.asyncio
    async def test_list_excludes_re_ingested_by_default(self, state_manager):
        """Test that re-ingested files are excluded by default."""
        await state_manager.mark_library_deleted("numpy", "file1.py")
        await state_manager.mark_library_deleted("numpy", "file2.py")
        await state_manager.re_ingest_library_file("numpy", "file2.py")

        deletions = await state_manager.list_library_deletions()
        assert len(deletions) == 1
        assert deletions[0]["file_path"] == "file1.py"

    @pytest.mark.asyncio
    async def test_list_includes_re_ingested_when_requested(self, state_manager):
        """Test listing with re-ingested files included."""
        await state_manager.mark_library_deleted("numpy", "file1.py")
        await state_manager.mark_library_deleted("numpy", "file2.py")
        await state_manager.re_ingest_library_file("numpy", "file2.py")

        deletions = await state_manager.list_library_deletions(include_re_ingested=True)
        assert len(deletions) == 2


class TestClearLibraryDeletion:
    """Tests for clearing library deletion records."""

    @pytest.mark.asyncio
    async def test_clear_deletion_record(self, state_manager):
        """Test clearing a deletion record entirely."""
        await state_manager.mark_library_deleted("numpy", "file.py")

        success = await state_manager.clear_library_deletion("numpy", "file.py")
        assert success is True

        # Verify it's completely gone (not in history either)
        deletions = await state_manager.list_library_deletions(include_re_ingested=True)
        assert len(deletions) == 0

    @pytest.mark.asyncio
    async def test_clear_non_existent_record(self, state_manager):
        """Test clearing a record that doesn't exist."""
        success = await state_manager.clear_library_deletion("unknown", "file.py")
        assert success is False


class TestGetLibraryDeletionCount:
    """Tests for counting deleted library files."""

    @pytest.mark.asyncio
    async def test_count_all_deletions(self, state_manager):
        """Test counting all deleted files."""
        await state_manager.mark_library_deleted("numpy", "file1.py")
        await state_manager.mark_library_deleted("pandas", "file2.py")
        await state_manager.mark_library_deleted("numpy", "file3.py")

        count = await state_manager.get_library_deletion_count()
        assert count == 3

    @pytest.mark.asyncio
    async def test_count_by_library(self, state_manager):
        """Test counting deleted files by library."""
        await state_manager.mark_library_deleted("numpy", "file1.py")
        await state_manager.mark_library_deleted("pandas", "file2.py")
        await state_manager.mark_library_deleted("numpy", "file3.py")

        count = await state_manager.get_library_deletion_count(library_name="numpy")
        assert count == 2

    @pytest.mark.asyncio
    async def test_count_excludes_re_ingested(self, state_manager):
        """Test that count excludes re-ingested files by default."""
        await state_manager.mark_library_deleted("numpy", "file1.py")
        await state_manager.mark_library_deleted("numpy", "file2.py")
        await state_manager.re_ingest_library_file("numpy", "file2.py")

        count = await state_manager.get_library_deletion_count()
        assert count == 1

    @pytest.mark.asyncio
    async def test_count_includes_re_ingested_when_requested(self, state_manager):
        """Test counting with re-ingested files included."""
        await state_manager.mark_library_deleted("numpy", "file1.py")
        await state_manager.mark_library_deleted("numpy", "file2.py")
        await state_manager.re_ingest_library_file("numpy", "file2.py")

        count = await state_manager.get_library_deletion_count(include_re_ingested=True)
        assert count == 2


class TestAdditiveDeletionWorkflow:
    """Integration tests for the additive deletion workflow."""

    @pytest.mark.asyncio
    async def test_full_workflow(self, state_manager):
        """Test the full additive deletion workflow."""
        # 1. Mark file as deleted
        await state_manager.mark_library_deleted("numpy", "core/function.py")

        # 2. Verify it's deleted
        assert await state_manager.is_library_file_deleted("numpy", "core/function.py")

        # 3. Simulate daemon checking before ingestion - file should be skipped
        # (This is what the daemon would do)
        if await state_manager.is_library_file_deleted("numpy", "core/function.py"):
            skipped = True  # Daemon would skip this file
        else:
            skipped = False

        assert skipped is True

        # 4. User explicitly requests re-ingestion
        await state_manager.re_ingest_library_file("numpy", "core/function.py")

        # 5. Now the file should be ingested
        assert not await state_manager.is_library_file_deleted("numpy", "core/function.py")

        # 6. Deletion history is preserved (with include_re_ingested)
        deletions = await state_manager.list_library_deletions(include_re_ingested=True)
        assert len(deletions) == 1
        assert deletions[0]["re_ingested"] is True
        assert deletions[0]["re_ingested_at"] is not None

    @pytest.mark.asyncio
    async def test_multiple_libraries_independent(self, state_manager):
        """Test that deletions in different libraries are independent."""
        # Delete file in numpy
        await state_manager.mark_library_deleted("numpy", "shared_name.py")

        # Same filename in pandas should not be affected
        assert not await state_manager.is_library_file_deleted("pandas", "shared_name.py")

        # Delete in pandas too
        await state_manager.mark_library_deleted("pandas", "shared_name.py")

        # Both should be deleted
        assert await state_manager.is_library_file_deleted("numpy", "shared_name.py")
        assert await state_manager.is_library_file_deleted("pandas", "shared_name.py")

        # Re-ingest only numpy
        await state_manager.re_ingest_library_file("numpy", "shared_name.py")

        # Numpy should be cleared, pandas still deleted
        assert not await state_manager.is_library_file_deleted("numpy", "shared_name.py")
        assert await state_manager.is_library_file_deleted("pandas", "shared_name.py")
