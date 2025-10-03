"""
Unit tests for metadata enforcement system.

Tests validation, metadata generation, enforcement workflow, and statistics
for all collection types.
"""

import asyncio
import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.python.common.core.metadata_enforcement import (
    MetadataEnforcer,
    ValidationResult,
    EnforcementResult,
    EnforcementStatistics,
)
from src.python.common.core.collection_types import CollectionType
from src.python.common.core.queue_client import QueueItem, QueueOperation


@pytest.fixture
def mock_state_manager():
    """Create mock state manager."""
    manager = MagicMock()
    manager.get_current_branch = AsyncMock(return_value="main")
    return manager


@pytest.fixture
def enforcer(mock_state_manager):
    """Create metadata enforcer instance."""
    return MetadataEnforcer(mock_state_manager)


@pytest.fixture
def sample_queue_item():
    """Create sample queue item for testing."""
    return QueueItem(
        file_absolute_path="/path/to/test.py",
        collection_name="test-project",
        tenant_id="default",
        branch="main",
        operation=QueueOperation.INGEST,
        priority=5,
        collection_type="project",
    )


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_valid_result(self):
        """Test valid validation result."""
        result = ValidationResult(
            is_valid=True,
            collection_type=CollectionType.SYSTEM,
        )

        assert result.is_valid
        assert result.collection_type == CollectionType.SYSTEM
        assert len(result.missing_fields) == 0
        assert len(result.invalid_fields) == 0

    def test_invalid_result_with_errors(self):
        """Test invalid result with missing and invalid fields."""
        result = ValidationResult(
            is_valid=False,
            collection_type=CollectionType.LIBRARY,
            missing_fields=["language", "symbols"],
            invalid_fields={"version": "Invalid version format"},
        )

        assert not result.is_valid
        assert result.collection_type == CollectionType.LIBRARY
        assert "language" in result.missing_fields
        assert "symbols" in result.missing_fields
        assert "version" in result.invalid_fields

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = ValidationResult(
            is_valid=False,
            collection_type=CollectionType.PROJECT,
            missing_fields=["project_id"],
            invalid_fields={"priority": "Must be integer"},
        )

        result_dict = result.to_dict()

        assert result_dict["is_valid"] is False
        assert result_dict["collection_type"] == "project"
        assert "project_id" in result_dict["missing_fields"]
        assert "priority" in result_dict["invalid_fields"]


class TestEnforcementResult:
    """Test EnforcementResult dataclass."""

    def test_successful_enforcement(self):
        """Test successful enforcement result."""
        validation = ValidationResult(
            is_valid=True,
            collection_type=CollectionType.SYSTEM,
        )

        result = EnforcementResult(
            success=True,
            validation_result=validation,
            completed_metadata={"collection_name": "__test", "created_at": "2024-01-01"},
        )

        assert result.success
        assert not result.metadata_generated
        assert not result.moved_to_missing_metadata_queue
        assert result.error_message is None

    def test_enforcement_with_metadata_generation(self):
        """Test enforcement with metadata generation."""
        validation = ValidationResult(
            is_valid=True,
            collection_type=CollectionType.LIBRARY,
        )

        result = EnforcementResult(
            success=True,
            validation_result=validation,
            metadata_generated=True,
            completed_metadata={
                "collection_name": "_mylib",
                "language": "python",
                "symbols": ["func1", "func2"],
            },
        )

        assert result.success
        assert result.metadata_generated
        assert "symbols" in result.completed_metadata

    def test_enforcement_moved_to_queue(self):
        """Test enforcement moved to missing metadata queue."""
        validation = ValidationResult(
            is_valid=False,
            collection_type=CollectionType.LIBRARY,
            missing_fields=["symbols", "dependencies"],
        )

        result = EnforcementResult(
            success=False,
            validation_result=validation,
            moved_to_missing_metadata_queue=True,
            error_message="Missing required tools: LSP=True, Tree-sitter=False",
        )

        assert not result.success
        assert result.moved_to_missing_metadata_queue
        assert "LSP=True" in result.error_message

    def test_to_dict(self):
        """Test conversion to dictionary."""
        validation = ValidationResult(
            is_valid=True,
            collection_type=CollectionType.GLOBAL,
        )

        result = EnforcementResult(
            success=True,
            validation_result=validation,
            completed_metadata={"collection_name": "algorithms"},
        )

        result_dict = result.to_dict()

        assert result_dict["success"] is True
        assert "validation" in result_dict
        assert result_dict["validation"]["is_valid"] is True


class TestEnforcementStatistics:
    """Test EnforcementStatistics dataclass."""

    def test_empty_stats(self):
        """Test empty statistics."""
        stats = EnforcementStatistics()

        assert stats.total_enforced == 0
        assert stats.successful == 0
        assert stats.failed == 0
        assert stats.metadata_generated == 0
        assert stats.moved_to_queue == 0

    def test_stats_with_data(self):
        """Test statistics with data."""
        stats = EnforcementStatistics(
            total_enforced=100,
            successful=85,
            failed=10,
            metadata_generated=50,
            moved_to_queue=5,
        )

        assert stats.total_enforced == 100
        assert stats.successful == 85
        assert stats.failed == 10

    def test_to_dict_with_success_rate(self):
        """Test conversion to dict includes success rate calculation."""
        stats = EnforcementStatistics(
            total_enforced=100,
            successful=85,
            failed=15,
        )

        stats_dict = stats.to_dict()

        assert stats_dict["total_enforced"] == 100
        assert stats_dict["successful"] == 85
        assert stats_dict["failed"] == 15
        assert stats_dict["success_rate"] == 85.0

    def test_zero_division_in_success_rate(self):
        """Test success rate calculation with zero total."""
        stats = EnforcementStatistics(total_enforced=0)

        stats_dict = stats.to_dict()

        assert stats_dict["success_rate"] == 0.0


class TestMetadataEnforcer:
    """Test MetadataEnforcer class."""

    @pytest.mark.asyncio
    async def test_validate_metadata_system_valid(self, enforcer):
        """Test validation of valid SYSTEM collection metadata."""
        metadata = {
            "collection_name": "__user_prefs",
            "created_at": "2024-01-01T00:00:00Z",
            "collection_category": "system",
        }

        result = await enforcer.validate_metadata(CollectionType.SYSTEM, metadata)

        assert result.is_valid
        assert result.collection_type == CollectionType.SYSTEM
        assert len(result.missing_fields) == 0

    @pytest.mark.asyncio
    async def test_validate_metadata_system_missing_fields(self, enforcer):
        """Test validation with missing required fields."""
        metadata = {
            "collection_name": "__user_prefs",
            # Missing: created_at, collection_category
        }

        result = await enforcer.validate_metadata(CollectionType.SYSTEM, metadata)

        assert not result.is_valid
        assert "created_at" in result.missing_fields
        assert "collection_category" in result.missing_fields

    @pytest.mark.asyncio
    async def test_validate_metadata_library_valid(self, enforcer):
        """Test validation of valid LIBRARY collection metadata."""
        metadata = {
            "collection_name": "_python_stdlib",
            "created_at": "2024-01-01T00:00:00Z",
            "collection_category": "library",
            "language": "python",
        }

        result = await enforcer.validate_metadata(CollectionType.LIBRARY, metadata)

        assert result.is_valid
        assert result.collection_type == CollectionType.LIBRARY

    @pytest.mark.asyncio
    async def test_validate_metadata_library_missing_language(self, enforcer):
        """Test LIBRARY validation missing language field."""
        metadata = {
            "collection_name": "_mylib",
            "created_at": "2024-01-01T00:00:00Z",
            "collection_category": "library",
            # Missing: language (required for LIBRARY)
        }

        result = await enforcer.validate_metadata(CollectionType.LIBRARY, metadata)

        assert not result.is_valid
        assert "language" in result.missing_fields

    @pytest.mark.asyncio
    async def test_validate_metadata_project_valid(self, enforcer):
        """Test validation of valid PROJECT collection metadata."""
        metadata = {
            "project_name": "myproject",
            "project_id": "abcdefghijkl",  # 12 chars
            "collection_type": "docs",
            "created_at": "2024-01-01T00:00:00Z",
        }

        result = await enforcer.validate_metadata(CollectionType.PROJECT, metadata)

        assert result.is_valid
        assert result.collection_type == CollectionType.PROJECT

    @pytest.mark.asyncio
    async def test_validate_metadata_project_invalid_id_length(self, enforcer):
        """Test PROJECT validation with invalid project_id length."""
        metadata = {
            "project_name": "myproject",
            "project_id": "short",  # Not 12 chars
            "collection_type": "docs",
            "created_at": "2024-01-01T00:00:00Z",
        }

        result = await enforcer.validate_metadata(CollectionType.PROJECT, metadata)

        assert not result.is_valid
        assert "project_id" in result.invalid_fields

    @pytest.mark.asyncio
    async def test_generate_missing_metadata_system(self, enforcer, tmp_path):
        """Test metadata generation for SYSTEM collection."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        metadata = {"collection_name": "__test"}

        completed = await enforcer.generate_missing_metadata(
            str(test_file),
            CollectionType.SYSTEM,
            metadata
        )

        assert "created_at" in completed
        assert "updated_at" in completed
        assert "file_size" in completed
        assert "file_extension" in completed
        assert completed["file_extension"] == ".txt"

    @pytest.mark.asyncio
    async def test_generate_missing_metadata_library(self, enforcer, tmp_path):
        """Test metadata generation for LIBRARY collection."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def hello(): pass")

        metadata = {"collection_name": "_mylib"}

        # Mock LSP and tree-sitter availability
        with patch.object(enforcer.tracker, "check_lsp_available") as mock_lsp, \
             patch.object(enforcer.tracker, "check_tree_sitter_available") as mock_ts:

            mock_lsp.return_value = {"available": True, "path": "/usr/bin/pylsp"}
            mock_ts.return_value = {"available": True, "path": "/usr/bin/tree-sitter"}

            completed = await enforcer.generate_missing_metadata(
                str(test_file),
                CollectionType.LIBRARY,
                metadata
            )

        assert "language" in completed
        assert completed["language"] == "python"
        assert "lsp_available" in completed
        assert completed["lsp_available"] is True
        assert "tree_sitter_available" in completed

    @pytest.mark.asyncio
    async def test_generate_missing_metadata_project(self, enforcer, tmp_path, mock_state_manager):
        """Test metadata generation for PROJECT collection."""
        test_file = tmp_path / "test.js"
        test_file.write_text("console.log('test');")

        metadata = {"project_name": "myproject"}

        # Mock branch detection
        mock_state_manager.get_current_branch = AsyncMock(return_value="feature/test")

        completed = await enforcer.generate_missing_metadata(
            str(test_file),
            CollectionType.PROJECT,
            metadata
        )

        assert "branch" in completed
        assert completed["branch"] == "feature/test"
        assert "file_type" in completed
        assert completed["file_type"] == "js"

    @pytest.mark.asyncio
    async def test_detect_language_from_extension(self, enforcer):
        """Test language detection from file extension."""
        test_cases = [
            (Path("/test.py"), "python"),
            (Path("/test.rs"), "rust"),
            (Path("/test.js"), "javascript"),
            (Path("/test.ts"), "typescript"),
            (Path("/test.java"), "java"),
            (Path("/test.go"), "go"),
            (Path("/test.unknown"), None),
        ]

        for file_path, expected in test_cases:
            result = enforcer._detect_language_from_extension(file_path)
            assert result == expected, f"Failed for {file_path.suffix}"

    @pytest.mark.asyncio
    async def test_enforce_metadata_valid_existing(self, enforcer, sample_queue_item):
        """Test enforcement with already valid metadata."""
        # Mock validation to return valid
        with patch.object(enforcer, "validate_metadata") as mock_validate:
            mock_validate.return_value = ValidationResult(
                is_valid=True,
                collection_type=CollectionType.PROJECT,
            )

            result = await enforcer.enforce_metadata(sample_queue_item)

        assert result.success
        assert not result.metadata_generated
        assert not result.moved_to_missing_metadata_queue

    @pytest.mark.asyncio
    async def test_enforce_metadata_generation_success(self, enforcer, sample_queue_item):
        """Test successful metadata generation during enforcement."""
        # First validation fails, second succeeds
        validation_results = [
            ValidationResult(
                is_valid=False,
                collection_type=CollectionType.PROJECT,
                missing_fields=["file_type"],
            ),
            ValidationResult(
                is_valid=True,
                collection_type=CollectionType.PROJECT,
            ),
        ]

        with patch.object(enforcer, "validate_metadata") as mock_validate, \
             patch.object(enforcer, "generate_missing_metadata") as mock_generate:

            mock_validate.side_effect = validation_results
            mock_generate.return_value = {
                "collection_name": "test-project",
                "tenant_id": "default",
                "branch": "main",
                "file_type": "py",
            }

            result = await enforcer.enforce_metadata(sample_queue_item)

        assert result.success
        assert result.metadata_generated
        assert "file_type" in result.completed_metadata

    @pytest.mark.asyncio
    async def test_enforce_metadata_missing_tools(self, enforcer, sample_queue_item):
        """Test enforcement when LSP/tree-sitter unavailable."""
        # Both validations fail
        validation_result = ValidationResult(
            is_valid=False,
            collection_type=CollectionType.LIBRARY,
            missing_fields=["symbols", "dependencies"],
        )

        metadata_with_language = {
            "collection_name": "_mylib",
            "tenant_id": "default",
            "branch": "main",
            "language": "python",
        }

        with patch.object(enforcer, "validate_metadata") as mock_validate, \
             patch.object(enforcer, "generate_missing_metadata") as mock_generate, \
             patch.object(enforcer.tracker, "check_lsp_available") as mock_lsp, \
             patch.object(enforcer.tracker, "check_tree_sitter_available") as mock_ts, \
             patch.object(enforcer.tracker, "track_missing_metadata") as mock_track:

            mock_validate.return_value = validation_result
            mock_generate.return_value = metadata_with_language
            mock_lsp.return_value = {"available": False, "path": None}
            mock_ts.return_value = {"available": False, "path": None}

            # Update queue item to library type
            sample_queue_item.collection_name = "_mylib"
            sample_queue_item.collection_type = "library"

            result = await enforcer.enforce_metadata(sample_queue_item)

        assert not result.success
        assert result.moved_to_missing_metadata_queue
        assert "Missing required tools" in result.error_message
        mock_track.assert_called_once()

    @pytest.mark.asyncio
    async def test_enforce_metadata_tools_available_but_incomplete(self, enforcer, sample_queue_item):
        """Test enforcement when tools available but metadata still incomplete."""
        validation_result = ValidationResult(
            is_valid=False,
            collection_type=CollectionType.LIBRARY,
            missing_fields=["symbols"],
        )

        metadata_with_language = {
            "collection_name": "_mylib",
            "language": "python",
        }

        with patch.object(enforcer, "validate_metadata") as mock_validate, \
             patch.object(enforcer, "generate_missing_metadata") as mock_generate, \
             patch.object(enforcer.tracker, "check_lsp_available") as mock_lsp, \
             patch.object(enforcer.tracker, "check_tree_sitter_available") as mock_ts:

            mock_validate.return_value = validation_result
            mock_generate.return_value = metadata_with_language
            mock_lsp.return_value = {"available": True, "path": "/usr/bin/pylsp"}
            mock_ts.return_value = {"available": True, "path": "/usr/bin/tree-sitter"}

            sample_queue_item.collection_name = "_mylib"
            sample_queue_item.collection_type = "library"

            result = await enforcer.enforce_metadata(sample_queue_item)

        assert not result.success
        assert not result.moved_to_missing_metadata_queue
        assert "Could not generate required metadata" in result.error_message

    @pytest.mark.asyncio
    async def test_get_enforcement_stats(self, enforcer):
        """Test getting enforcement statistics."""
        # Manually update stats
        enforcer.stats.total_enforced = 100
        enforcer.stats.successful = 85
        enforcer.stats.failed = 10
        enforcer.stats.metadata_generated = 50
        enforcer.stats.moved_to_queue = 5
        enforcer.stats.common_missing_fields = {
            "language": 30,
            "symbols": 20,
            "project_id": 15,
        }

        stats = enforcer.get_enforcement_stats()

        assert stats.total_enforced == 100
        assert stats.successful == 85
        assert stats.failed == 10
        assert stats.metadata_generated == 50
        assert stats.moved_to_queue == 5
        assert stats.common_missing_fields["language"] == 30

    @pytest.mark.asyncio
    async def test_update_type_stats(self, enforcer):
        """Test per-type statistics tracking."""
        enforcer._update_type_stats("library", "successful", 5)
        enforcer._update_type_stats("library", "generated", 3)
        enforcer._update_type_stats("project", "successful", 10)

        assert enforcer.stats.by_collection_type["library"]["successful"] == 5
        assert enforcer.stats.by_collection_type["library"]["generated"] == 3
        assert enforcer.stats.by_collection_type["project"]["successful"] == 10

    def test_reset_stats(self, enforcer):
        """Test resetting statistics."""
        # Set some stats
        enforcer.stats.total_enforced = 100
        enforcer.stats.successful = 85

        # Reset
        enforcer.reset_stats()

        # Verify reset
        assert enforcer.stats.total_enforced == 0
        assert enforcer.stats.successful == 0

    @pytest.mark.asyncio
    async def test_enforce_metadata_error_handling(self, enforcer, sample_queue_item):
        """Test error handling during enforcement."""
        # Mock validation to raise exception
        with patch.object(enforcer, "validate_metadata") as mock_validate:
            mock_validate.side_effect = Exception("Test error")

            result = await enforcer.enforce_metadata(sample_queue_item)

        assert not result.success
        assert result.error_message == "Test error"
        assert "enforcement_error" in result.validation_result.invalid_fields

    @pytest.mark.asyncio
    async def test_enforce_metadata_statistics_tracking(self, enforcer, sample_queue_item):
        """Test that statistics are properly tracked during enforcement."""
        initial_total = enforcer.stats.total_enforced

        with patch.object(enforcer, "validate_metadata") as mock_validate:
            mock_validate.return_value = ValidationResult(
                is_valid=True,
                collection_type=CollectionType.PROJECT,
            )

            await enforcer.enforce_metadata(sample_queue_item)

        assert enforcer.stats.total_enforced == initial_total + 1
        assert enforcer.stats.successful >= 1

    @pytest.mark.asyncio
    async def test_enforce_metadata_tracks_missing_fields(self, enforcer, sample_queue_item):
        """Test that common missing fields are tracked."""
        validation_results = [
            ValidationResult(
                is_valid=False,
                collection_type=CollectionType.LIBRARY,
                missing_fields=["language", "symbols"],
            ),
            ValidationResult(
                is_valid=False,
                collection_type=CollectionType.LIBRARY,
                missing_fields=["language", "symbols"],
            ),
        ]

        with patch.object(enforcer, "validate_metadata") as mock_validate, \
             patch.object(enforcer, "generate_missing_metadata") as mock_generate, \
             patch.object(enforcer.tracker, "check_lsp_available") as mock_lsp:

            mock_validate.side_effect = validation_results
            mock_generate.return_value = {"collection_name": "_mylib", "language": "python"}
            mock_lsp.return_value = {"available": True, "path": "/usr/bin/pylsp"}

            sample_queue_item.collection_name = "_mylib"
            sample_queue_item.collection_type = "library"

            await enforcer.enforce_metadata(sample_queue_item)

        # Check that missing fields were tracked
        assert "language" in enforcer.stats.common_missing_fields
        assert "symbols" in enforcer.stats.common_missing_fields


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_validate_metadata_unknown_collection_type(self, enforcer):
        """Test validation with unknown collection type."""
        # Error handling catches the ValueError and returns ValidationResult
        result = await enforcer.validate_metadata(
            CollectionType.UNKNOWN,
            {"collection_name": "unknown"}
        )

        # Should return invalid result instead of raising
        assert not result.is_valid
        assert "validation_error" in result.invalid_fields

    @pytest.mark.asyncio
    async def test_generate_metadata_nonexistent_file(self, enforcer):
        """Test metadata generation for non-existent file."""
        metadata = {"collection_name": "__test"}

        completed = await enforcer.generate_missing_metadata(
            "/nonexistent/file.py",
            CollectionType.SYSTEM,
            metadata
        )

        # Should still add timestamps even if file doesn't exist
        assert "created_at" in completed
        assert "updated_at" in completed
        # But file-specific metadata won't be added
        assert "file_size" not in completed

    @pytest.mark.asyncio
    async def test_enforce_metadata_no_collection_type(self, enforcer):
        """Test enforcement when queue item has no collection_type."""
        queue_item = QueueItem(
            file_absolute_path="/path/to/test.py",
            collection_name="test-project",
            tenant_id="default",
            branch="main",
            operation=QueueOperation.INGEST,
            priority=5,
            collection_type=None,  # No type specified
        )

        with patch.object(enforcer, "validate_metadata") as mock_validate:
            mock_validate.return_value = ValidationResult(
                is_valid=True,
                collection_type=CollectionType.PROJECT,
            )

            result = await enforcer.enforce_metadata(queue_item)

        # Should classify from collection name
        assert result.success

    @pytest.mark.asyncio
    async def test_add_library_metadata_no_language(self, enforcer, tmp_path):
        """Test library metadata generation when language cannot be detected."""
        test_file = tmp_path / "test.unknown"
        test_file.write_text("content")

        metadata = {"collection_name": "_mylib"}

        await enforcer._add_library_metadata(test_file, metadata)

        # Should not crash, but won't add language-specific metadata
        assert "language" not in metadata

    @pytest.mark.asyncio
    async def test_add_project_metadata_no_git(self, enforcer, tmp_path, mock_state_manager):
        """Test project metadata when git info unavailable."""
        test_file = tmp_path / "test.py"
        test_file.write_text("print('test')")

        metadata = {"project_name": "test"}

        # Mock branch detection to fail
        mock_state_manager.get_current_branch.side_effect = Exception("No git")

        await enforcer._add_project_metadata(test_file, metadata)

        # Should still add file_type
        assert "file_type" in metadata
        assert metadata["file_type"] == "py"
        # But not branch
        assert "branch" not in metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
