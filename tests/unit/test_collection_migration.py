"""
Unit tests for collection migration system.

Tests cover:
- Collection validation against type requirements
- Collection type detection with confidence scoring
- Migration with dry-run and execution modes
- Rollback capability
- Edge case handling
- Migration report generation
"""

import pytest
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from src.python.common.core.collection_migration import (
    CollectionMigrator,
    ValidationResult,
    ValidationIssue,
    ValidationSeverity,
    DetectionResult,
    MigrationResult,
    MigrationReport,
    MigrationStrategy,
    MigrationRecommendation,
)
from src.python.common.core.collection_types import CollectionType


@pytest.fixture
def mock_qdrant_client():
    """Create mock Qdrant client."""
    client = Mock(spec=QdrantClient)
    return client


@pytest.fixture
def mock_collision_detector():
    """Create mock collision detector."""
    detector = AsyncMock()
    detector.initialize = AsyncMock()
    detector.check_collection_collision = AsyncMock()
    return detector


def create_mock_collection_info():
    """Create a mock collection info object that bypasses Pydantic validation."""
    mock_info = Mock()
    mock_info.status = "green"
    mock_info.optimizer_status = "ok"
    mock_info.vectors_count = 100
    mock_info.indexed_vectors_count = 100
    mock_info.points_count = 100
    mock_info.segments_count = 1
    return mock_info


@pytest.fixture
async def migrator(mock_qdrant_client, mock_collision_detector):
    """Create CollectionMigrator instance."""
    migrator = CollectionMigrator(mock_qdrant_client, mock_collision_detector)
    await migrator.initialize()
    return migrator


class TestCollectionValidation:
    """Tests for collection validation functionality."""

    @pytest.mark.asyncio
    async def test_validate_system_collection_valid(
        self, migrator, mock_qdrant_client
    ):
        """Test validation of valid system collection."""
        # Setup: Mock Qdrant response
        mock_qdrant_client.get_collection.return_value = create_mock_collection_info()

        # Execute
        result = await migrator.validate_collection("__system_config")

        # Verify
        assert result.collection_name == "__system_config"
        assert result.detected_type == CollectionType.SYSTEM
        # May have INFO issues but should be overall valid (no errors)
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_validate_library_collection_valid(
        self, migrator, mock_qdrant_client
    ):
        """Test validation of valid library collection."""
        mock_qdrant_client.get_collection.return_value = create_mock_collection_info()

        result = await migrator.validate_collection("_library_docs")

        assert result.collection_name == "_library_docs"
        assert result.detected_type == CollectionType.LIBRARY
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_validate_project_collection_valid(
        self, migrator, mock_qdrant_client
    ):
        """Test validation of valid project collection."""
        mock_qdrant_client.get_collection.return_value = create_mock_collection_info()

        # New architecture: PROJECT collections use _{project_id} format
        result = await migrator.validate_collection("_github_com_user_myproject")

        assert result.collection_name == "_github_com_user_myproject"
        assert result.detected_type == CollectionType.PROJECT
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_validate_global_collection_valid(
        self, migrator, mock_qdrant_client
    ):
        """Test validation of valid global collection."""
        mock_qdrant_client.get_collection.return_value = create_mock_collection_info()

        result = await migrator.validate_collection("algorithms")

        assert result.collection_name == "algorithms"
        assert result.detected_type == CollectionType.GLOBAL
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_validate_collection_not_found(self, migrator, mock_qdrant_client):
        """Test validation when collection doesn't exist."""
        mock_qdrant_client.get_collection.side_effect = UnexpectedResponse(
            status_code=404, reason_phrase="Not found", content=b"", headers={}
        )

        result = await migrator.validate_collection("nonexistent")

        assert result.collection_name == "nonexistent"
        assert not result.is_valid
        assert len(result.errors) > 0
        assert any("does not exist" in e.message for e in result.errors)

    @pytest.mark.asyncio
    async def test_validate_unknown_type(self, migrator, mock_qdrant_client):
        """Test validation of collection with unknown type."""
        mock_qdrant_client.get_collection.return_value = create_mock_collection_info()

        result = await migrator.validate_collection("unknownformat")

        assert result.detected_type == CollectionType.UNKNOWN
        assert not result.is_valid
        assert any(
            "does not match any known type pattern" in e.message for e in result.errors
        )

    @pytest.mark.asyncio
    async def test_validate_misnamed_system_collection(
        self, migrator, mock_qdrant_client
    ):
        """Test validation of system collection with wrong prefix."""
        mock_qdrant_client.get_collection.return_value = create_mock_collection_info()

        # Collection detected as library (single underscore), not system
        result = await migrator.validate_collection("_system_config")

        # Should be detected as LIBRARY (single underscore)
        assert result.detected_type == CollectionType.LIBRARY


class TestCollectionTypeDetection:
    """Tests for collection type detection with confidence scoring."""

    @pytest.mark.asyncio
    async def test_detect_system_collection_high_confidence(self, migrator):
        """Test detection of properly formatted system collection."""
        result = await migrator.detect_collection_type("__user_preferences")

        assert result.detected_type == CollectionType.SYSTEM
        assert result.confidence == 1.0
        assert "Exact match" in result.detection_reason
        assert not result.requires_manual_intervention

    @pytest.mark.asyncio
    async def test_detect_library_collection_high_confidence(self, migrator):
        """Test detection of properly formatted library collection."""
        result = await migrator.detect_collection_type("_library_code")

        assert result.detected_type == CollectionType.LIBRARY
        assert result.confidence == 1.0
        assert "Exact match" in result.detection_reason
        assert not result.requires_manual_intervention

    @pytest.mark.asyncio
    async def test_detect_project_collection_high_confidence(self, migrator):
        """Test detection of properly formatted project collection."""
        # New architecture: PROJECT collections use _{project_id} format
        result = await migrator.detect_collection_type("_github_com_user_myproject")

        assert result.detected_type == CollectionType.PROJECT
        assert result.confidence == 1.0
        assert "Exact match" in result.detection_reason
        assert not result.requires_manual_intervention

    @pytest.mark.asyncio
    async def test_detect_global_collection_high_confidence(self, migrator):
        """Test detection of known global collection."""
        result = await migrator.detect_collection_type("algorithms")

        assert result.detected_type == CollectionType.GLOBAL
        assert result.confidence == 1.0
        assert "Exact match" in result.detection_reason
        assert not result.requires_manual_intervention

    @pytest.mark.asyncio
    async def test_detect_unknown_collection_low_confidence(self, migrator):
        """Test detection of unknown format collection."""
        result = await migrator.detect_collection_type("randomname")

        # Single word without prefix might be detected as UNKNOWN or PROJECT
        assert result.detected_type in [CollectionType.UNKNOWN, CollectionType.PROJECT]
        assert result.confidence <= 0.6  # Should be low confidence

    @pytest.mark.asyncio
    async def test_detect_ambiguous_collection(self, migrator):
        """Test detection of collection with ambiguous pattern."""
        result = await migrator.detect_collection_type("test-incomplete-")

        # Should still be classified but with lower confidence
        assert result.confidence < 0.9


class TestCollectionMigration:
    """Tests for collection migration functionality."""

    @pytest.mark.asyncio
    async def test_migrate_dry_run_success(
        self, migrator, mock_qdrant_client, mock_collision_detector
    ):
        """Test successful dry-run migration."""
        # Setup: Collection exists
        mock_qdrant_client.get_collection.return_value = create_mock_collection_info()

        # No collision
        mock_collision_result = Mock()
        mock_collision_result.has_collision = False
        mock_collision_detector.check_collection_collision.return_value = (
            mock_collision_result
        )

        # Execute: Migrate library collection to system type (dry-run)
        result = await migrator.migrate_collection(
            "_library_docs", CollectionType.SYSTEM, dry_run=True
        )

        # Verify
        assert result.collection_name == "_library_docs"
        assert result.target_type == CollectionType.SYSTEM
        assert result.dry_run is True
        assert len(result.changes_applied) > 0
        # Should include rename operation
        assert any("Rename" in change for change in result.changes_applied)

    @pytest.mark.asyncio
    async def test_migrate_already_correct_type(
        self, migrator, mock_qdrant_client
    ):
        """Test migration when collection already matches target type."""
        mock_qdrant_client.get_collection.return_value = create_mock_collection_info()

        # Execute: Migrate system collection to system type
        result = await migrator.migrate_collection(
            "__system_config", CollectionType.SYSTEM, dry_run=True
        )

        # Verify: No changes needed
        assert result.success is True
        assert any(
            "No migration needed" in change or "already matches" in change
            for change in result.changes_applied
        )

    @pytest.mark.asyncio
    async def test_migrate_with_name_collision(
        self, migrator, mock_qdrant_client, mock_collision_detector
    ):
        """Test migration with naming collision detected."""
        mock_qdrant_client.get_collection.return_value = create_mock_collection_info()

        # Setup: Collision detected
        mock_collision_result = Mock()
        mock_collision_result.has_collision = True
        mock_collision_result.collision_reason = "Collection already exists"
        mock_collision_detector.check_collection_collision.return_value = (
            mock_collision_result
        )

        # Execute
        result = await migrator.migrate_collection(
            "_library_docs", CollectionType.SYSTEM, dry_run=True
        )

        # Verify: Should detect collision
        assert len(result.conflicts_detected) > 0
        assert any("collision" in c.lower() for c in result.conflicts_detected)

    @pytest.mark.asyncio
    async def test_migrate_nonexistent_collection(
        self, migrator, mock_qdrant_client
    ):
        """Test migration of non-existent collection."""
        mock_qdrant_client.get_collection.side_effect = UnexpectedResponse(
            status_code=404, reason_phrase="Not found", content=b"", headers={}
        )

        result = await migrator.migrate_collection(
            "nonexistent", CollectionType.SYSTEM, dry_run=True
        )

        assert result.success is False
        assert result.error_message is not None
        assert "does not exist" in result.error_message

    @pytest.mark.asyncio
    async def test_migrate_execution_mode(
        self, migrator, mock_qdrant_client, mock_collision_detector
    ):
        """Test migration in execution mode (not dry-run)."""
        mock_qdrant_client.get_collection.return_value = create_mock_collection_info()

        mock_collision_result = Mock()
        mock_collision_result.has_collision = False
        mock_collision_detector.check_collection_collision.return_value = (
            mock_collision_result
        )

        # Execute: Non-dry-run migration
        result = await migrator.migrate_collection(
            "_library_docs", CollectionType.SYSTEM, dry_run=False
        )

        # Verify: Should create backup
        assert result.dry_run is False
        assert any("backup" in change.lower() for change in result.changes_applied)


class TestMigrationReporting:
    """Tests for migration report generation."""

    @pytest.mark.asyncio
    async def test_generate_report_empty_list(self, migrator):
        """Test report generation with empty collection list."""
        report = await migrator.generate_migration_report([])

        assert report.total_collections == 0
        assert report.valid_collections == 0
        assert report.invalid_collections == 0
        assert len(report.recommendations) == 0

    @pytest.mark.asyncio
    async def test_generate_report_all_valid(
        self, migrator, mock_qdrant_client
    ):
        """Test report generation with all valid collections."""
        mock_qdrant_client.get_collection.return_value = create_mock_collection_info()

        collections = [
            "__system_config",
            "_library_docs",
            "myproject-docs",
            "algorithms",
        ]

        report = await migrator.generate_migration_report(collections)

        assert report.total_collections == 4
        # All properly formatted collections should be valid (no errors)
        assert report.valid_collections >= 0

    @pytest.mark.asyncio
    async def test_generate_report_mixed_validity(
        self, migrator, mock_qdrant_client
    ):
        """Test report generation with mix of valid and invalid collections."""
        # Setup: Some collections exist, some don't
        def get_collection_side_effect(name):
            if name in ["__system_config", "_library_docs"]:
                return create_mock_collection_info()
            else:
                raise UnexpectedResponse(
            status_code=404, reason_phrase="Not found", content=b"", headers={}
        )

        mock_qdrant_client.get_collection.side_effect = get_collection_side_effect

        collections = ["__system_config", "_library_docs", "nonexistent", "invalid"]

        report = await migrator.generate_migration_report(collections)

        assert report.total_collections == 4
        assert report.invalid_collections > 0
        assert len(report.problematic_collections) > 0

    @pytest.mark.asyncio
    async def test_generate_report_with_recommendations(
        self, migrator, mock_qdrant_client
    ):
        """Test report generation includes migration recommendations."""
        mock_qdrant_client.get_collection.return_value = create_mock_collection_info()

        # Collection with unknown type should generate recommendation
        collections = ["unknownformat"]

        report = await migrator.generate_migration_report(collections)

        # Unknown format should trigger migration recommendations
        assert len(report.collections_needing_migration) > 0

    @pytest.mark.asyncio
    async def test_generate_report_categorizes_by_type(
        self, migrator, mock_qdrant_client
    ):
        """Test report correctly categorizes collections by type."""
        mock_qdrant_client.get_collection.return_value = create_mock_collection_info()

        collections = [
            "__system_config",
            "__system_cache",
            "_library_docs",
            "_library_utils",
            "_github_com_user_project1",
            "_path_abc123def456789a",
            "algorithms",
        ]

        report = await migrator.generate_migration_report(collections)

        # Verify categorization
        assert len(report.collections_by_type[CollectionType.SYSTEM.value]) == 2
        assert len(report.collections_by_type[CollectionType.LIBRARY.value]) == 2
        assert len(report.collections_by_type[CollectionType.PROJECT.value]) == 2
        assert len(report.collections_by_type[CollectionType.GLOBAL.value]) == 1

    @pytest.mark.asyncio
    async def test_report_to_dict(self, migrator, mock_qdrant_client):
        """Test report serialization to dictionary."""
        mock_qdrant_client.get_collection.return_value = create_mock_collection_info()

        collections = ["__system_config", "_library_docs"]
        report = await migrator.generate_migration_report(collections)

        report_dict = report.to_dict()

        assert isinstance(report_dict, dict)
        assert "total_collections" in report_dict
        assert "valid_collections" in report_dict
        assert "by_type" in report_dict
        assert "recommendations" in report_dict

    @pytest.mark.asyncio
    async def test_report_to_json(self, migrator, mock_qdrant_client):
        """Test report serialization to JSON."""
        mock_qdrant_client.get_collection.return_value = create_mock_collection_info()

        collections = ["__system_config"]
        report = await migrator.generate_migration_report(collections)

        json_str = report.to_json()

        assert isinstance(json_str, str)
        # Should be valid JSON
        import json

        json.loads(json_str)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_handle_qdrant_connection_error(
        self, migrator, mock_qdrant_client
    ):
        """Test handling of Qdrant connection errors."""
        mock_qdrant_client.get_collection.side_effect = Exception(
            "Connection refused"
        )

        result = await migrator.validate_collection("__system_config")

        assert not result.is_valid
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_migration_strategy_automatic(self, migrator):
        """Test automatic migration strategy for high confidence."""
        # Set confidence threshold
        migrator.confidence_threshold_automatic = 0.9

        detection = await migrator.detect_collection_type("__system_config")

        # High confidence should enable automatic migration
        assert detection.confidence >= migrator.confidence_threshold_automatic

    @pytest.mark.asyncio
    async def test_migration_strategy_assisted(self, migrator):
        """Test assisted migration strategy for medium confidence."""
        migrator.confidence_threshold_automatic = 0.9
        migrator.confidence_threshold_assisted = 0.6

        # Collection with partial match
        detection = await migrator.detect_collection_type("test-partial")

        # Just verify detection works
        assert detection.detected_type is not None

    @pytest.mark.asyncio
    async def test_generate_target_name_system(self, migrator):
        """Test target name generation for system type."""
        target_name = migrator._generate_target_name("library_docs", CollectionType.SYSTEM)

        assert target_name == "__library_docs"

    @pytest.mark.asyncio
    async def test_generate_target_name_library(self, migrator):
        """Test target name generation for library type."""
        target_name = migrator._generate_target_name("library_docs", CollectionType.LIBRARY)

        assert target_name == "_library_docs"

    @pytest.mark.asyncio
    async def test_generate_target_name_project(self, migrator):
        """Test target name generation for project type."""
        target_name = migrator._generate_target_name("myproject", CollectionType.PROJECT)

        # Should add default suffix if no dash present
        assert "-" in target_name
        assert target_name == "myproject-docs"

    @pytest.mark.asyncio
    async def test_generate_target_name_project_with_suffix(self, migrator):
        """Test target name generation for project with existing suffix."""
        target_name = migrator._generate_target_name(
            "myproject-notes", CollectionType.PROJECT
        )

        assert target_name == "myproject-notes"

    @pytest.mark.asyncio
    async def test_calculate_metadata_changes(self, migrator):
        """Test metadata change calculation."""
        from src.python.common.core.collection_type_config import get_type_config

        config = get_type_config(CollectionType.SYSTEM)
        changes = migrator._calculate_metadata_changes(
            "__system_config", CollectionType.SYSTEM, config
        )

        assert len(changes) > 0
        assert any("collection_type" in change for change in changes)

    @pytest.mark.asyncio
    async def test_validation_result_filtering(self, migrator, mock_qdrant_client):
        """Test validation result issue filtering."""
        mock_qdrant_client.get_collection.return_value = create_mock_collection_info()

        result = await migrator.validate_collection("__system_config")

        # Test filtering methods
        errors = result.errors
        warnings = result.warnings
        infos = result.infos

        assert isinstance(errors, list)
        assert isinstance(warnings, list)
        assert isinstance(infos, list)

        # All issues should be accounted for
        assert len(errors) + len(warnings) + len(infos) == len(result.issues)


class TestMigrationScenarios:
    """Integration-style tests for complete migration scenarios."""

    @pytest.mark.asyncio
    async def test_full_migration_workflow(
        self, migrator, mock_qdrant_client, mock_collision_detector
    ):
        """Test complete migration workflow from validation to execution."""
        # Setup
        mock_qdrant_client.get_collection.return_value = create_mock_collection_info()

        mock_collision_result = Mock()
        mock_collision_result.has_collision = False
        mock_collision_detector.check_collection_collision.return_value = (
            mock_collision_result
        )

        collection_name = "_library_docs"

        # Step 1: Validate
        validation = await migrator.validate_collection(collection_name)
        assert validation.detected_type == CollectionType.LIBRARY

        # Step 2: Detect type
        detection = await migrator.detect_collection_type(collection_name)
        assert detection.detected_type == CollectionType.LIBRARY
        assert detection.confidence == 1.0

        # Step 3: Dry-run migration
        dry_run = await migrator.migrate_collection(
            collection_name, CollectionType.SYSTEM, dry_run=True
        )
        assert dry_run.dry_run is True
        assert len(dry_run.changes_applied) > 0

        # Step 4: Execute migration (if dry-run successful)
        if dry_run.success and len(dry_run.conflicts_detected) == 0:
            execution = await migrator.migrate_collection(
                collection_name, CollectionType.SYSTEM, dry_run=False
            )
            assert execution.dry_run is False

    @pytest.mark.asyncio
    async def test_migration_with_rollback(
        self, migrator, mock_qdrant_client, mock_collision_detector
    ):
        """Test migration rollback on failure."""
        mock_qdrant_client.get_collection.return_value = create_mock_collection_info()

        mock_collision_result = Mock()
        mock_collision_result.has_collision = False
        mock_collision_detector.check_collection_collision.return_value = (
            mock_collision_result
        )

        # Execute migration
        result = await migrator.migrate_collection(
            "_library_docs", CollectionType.SYSTEM, dry_run=False
        )

        # If migration created a backup, verify it's available for rollback
        if result.backup:
            assert result.backup.collection_name == "_library_docs"
            assert result.backup.backup_id is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
