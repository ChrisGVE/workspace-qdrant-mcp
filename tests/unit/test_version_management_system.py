"""
Comprehensive unit tests for the Version Management System.

Tests Task 262 implementation including edge cases:
- Document type detection with ambiguous content
- Version comparison with invalid/malformed versions
- Conflict detection and resolution strategies
- Archive management with storage failures
- User workflow integration
"""

import asyncio
import hashlib

# Import the modules we're testing
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add src/python to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src" / "python"))

# Now we can import our modules
try:
    from qdrant_client.http import models
    from workspace_qdrant_mcp.version_management import (
        ArchiveEntry,
        ArchiveManager,
        ArchivePolicy,
        ArchiveStatus,
        ConflictType,
        DocumentType,
        FileFormat,
        ResolutionStrategy,
        UserDecision,
        UserPrompt,
        VersionConflict,
        VersionInfo,
        VersionManager,
        WorkflowIntegrator,
        WorkflowStatus,
    )

    from python.common.core.client import QdrantWorkspaceClient
except ImportError as e:
    pytest.skip(f"Could not import required modules: {e}", allow_module_level=True)


class TestVersionManagerCore:
    """Test core VersionManager functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_client = MagicMock(spec=QdrantWorkspaceClient)
        self.manager = VersionManager(self.mock_client)

    def test_init_with_client(self):
        """Test initializing version manager with client."""
        assert self.manager.client is self.mock_client
        assert DocumentType.BOOK in self.manager.type_configs
        assert DocumentType.CODE_FILE in self.manager.type_configs
        assert len(self.manager.format_precedence) > 0

    def test_document_type_detection_explicit(self):
        """Test detection with explicit document_type in metadata."""
        metadata = {"document_type": "book"}
        doc_type = self.manager.detect_document_type(metadata)
        assert doc_type == DocumentType.BOOK

    def test_document_type_detection_from_extension(self):
        """Test detection from file extension."""
        metadata = {"file_path": "/path/to/script.py"}
        doc_type = self.manager.detect_document_type(metadata)
        assert doc_type == DocumentType.CODE_FILE

    def test_document_type_detection_from_metadata_fields(self):
        """Test detection from metadata field patterns."""
        # Book detection
        metadata = {"isbn": "978-0123456789", "title": "Test Book"}
        doc_type = self.manager.detect_document_type(metadata)
        assert doc_type == DocumentType.BOOK

        # Scientific article detection
        metadata = {"doi": "10.1000/test.doi", "journal": "Nature"}
        doc_type = self.manager.detect_document_type(metadata)
        assert doc_type == DocumentType.SCIENTIFIC_ARTICLE

        # Web page detection
        metadata = {"url": "https://example.com/page"}
        doc_type = self.manager.detect_document_type(metadata)
        assert doc_type == DocumentType.WEB_PAGE

    def test_document_type_detection_generic_fallback(self):
        """Test fallback to generic type."""
        metadata = {"title": "Unknown Document"}
        doc_type = self.manager.detect_document_type(metadata)
        assert doc_type == DocumentType.GENERIC

    def test_version_extraction_semantic_version(self):
        """Test extracting semantic version from code content."""
        content = """
        # Version: v1.2.3
        def main():
            pass
        """
        version = self.manager.extract_version_from_content(content, DocumentType.CODE_FILE)
        assert version == "1.2.3"

    def test_version_extraction_book_edition(self):
        """Test extracting edition from book content."""
        content = "This is the 2nd edition of the comprehensive guide."
        version = self.manager.extract_version_from_content(content, DocumentType.BOOK)
        assert version == "2"

    def test_version_extraction_publication_date(self):
        """Test extracting publication date from article."""
        content = "Published on 2024-03-15 in Nature journal."
        version = self.manager.extract_version_from_content(content, DocumentType.SCIENTIFIC_ARTICLE)
        assert version == "2024-03-15"

    def test_version_extraction_no_pattern(self):
        """Test extraction when no pattern matches."""
        content = "This document has no version information."
        version = self.manager.extract_version_from_content(content, DocumentType.GENERIC)
        assert version is None

    def test_version_comparison_semantic_versions(self):
        """Test semantic version comparison."""
        with patch('workspace_qdrant_mcp.version_management.version_manager.semver') as mock_semver:
            v1_mock = MagicMock()
            v1_mock.compare.return_value = 1
            v2_mock = MagicMock()

            mock_semver.VersionInfo.parse.side_effect = [v1_mock, v2_mock]

            result = self.manager.compare_versions("2.0.0", "1.0.0", DocumentType.CODE_FILE)
            assert result == 1

    def test_version_comparison_book_editions(self):
        """Test book edition comparison."""
        result = self.manager.compare_versions("3rd edition", "1st edition", DocumentType.BOOK)
        assert result == 1  # 3rd > 1st

        result = self.manager.compare_versions("1st edition", "2nd edition", DocumentType.BOOK)
        assert result == -1  # 1st < 2nd

    def test_version_comparison_malformed_editions(self):
        """Test handling malformed book editions."""
        result = self.manager.compare_versions("special edition", "limited edition", DocumentType.BOOK)
        assert result == 0  # No numbers found, fallback

    def test_file_format_detection(self):
        """Test file format detection."""
        metadata = {"file_path": "/path/to/document.pdf"}
        format = self.manager.get_file_format(metadata)
        assert format == FileFormat.PDF

        metadata = {"format": "docx"}
        format = self.manager.get_file_format(metadata)
        assert format == FileFormat.DOCX

        metadata = {"file_path": "/path/to/file.unknown"}
        format = self.manager.get_file_format(metadata)
        assert format == FileFormat.UNKNOWN

    def test_content_hash_calculation(self):
        """Test content hash calculation."""
        content = "test content"
        expected_hash = hashlib.sha256(content.encode()).hexdigest()
        actual_hash = self.manager.calculate_content_hash(content)
        assert actual_hash == expected_hash

    def test_content_hash_unicode(self):
        """Test content hash with unicode characters."""
        content = "test content with émojis 🎉"
        hash_result = self.manager.calculate_content_hash(content)
        assert isinstance(hash_result, str)
        assert len(hash_result) == 64  # SHA256 hex length


class TestConflictDetection:
    """Test version conflict detection and analysis."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_client = MagicMock(spec=QdrantWorkspaceClient)
        self.mock_client.initialized = True
        self.manager = VersionManager(self.mock_client)

    def _create_version_info(self, **kwargs):
        """Helper to create VersionInfo instances."""
        defaults = {
            "version_string": "1.0.0",
            "version_type": "semantic",
            "document_type": DocumentType.CODE_FILE,
            "authority_level": 0.7,
            "timestamp": datetime.now(timezone.utc),
            "content_hash": "hash123",
            "format": FileFormat.TXT,
            "metadata": {},
            "point_id": "point123",
            "supersedes": []
        }
        defaults.update(kwargs)
        return VersionInfo(**defaults)

    def test_analyze_version_conflict_content_conflict(self):
        """Test detecting content conflict with same version."""
        new_version = self._create_version_info(
            version_string="1.0.0",
            content_hash="hash1"
        )
        existing_version = self._create_version_info(
            version_string="1.0.0",
            content_hash="hash2",
            point_id="existing-point"
        )

        conflict = self.manager._analyze_version_conflict(new_version, existing_version)

        assert conflict is not None
        assert ConflictType.CONTENT_CONFLICT in [conflict.conflict_type]
        assert conflict.conflict_severity >= 0.9

    def test_analyze_version_conflict_format_conflict(self):
        """Test detecting format conflict with same content."""
        new_version = self._create_version_info(
            format=FileFormat.PDF,
            content_hash="same_hash"
        )
        existing_version = self._create_version_info(
            format=FileFormat.TXT,
            content_hash="same_hash",
            point_id="existing-point"
        )

        conflict = self.manager._analyze_version_conflict(new_version, existing_version)

        assert conflict is not None
        assert ConflictType.FORMAT_CONFLICT in [conflict.conflict_type]

    def test_analyze_version_conflict_temporal_conflict(self):
        """Test detecting temporal conflict."""
        base_time = datetime.now(timezone.utc)

        new_version = self._create_version_info(
            timestamp=base_time,
            content_hash="hash1"
        )
        existing_version = self._create_version_info(
            timestamp=base_time,  # Same time
            content_hash="hash2",  # Different content
            point_id="existing-point"
        )

        conflict = self.manager._analyze_version_conflict(new_version, existing_version)

        assert conflict is not None
        assert ConflictType.TEMPORAL_CONFLICT in [conflict.conflict_type]

    def test_analyze_no_conflict(self):
        """Test when there's no conflict between versions."""
        new_version = self._create_version_info(
            version_string="2.0.0",
            content_hash="hash1"
        )
        existing_version = self._create_version_info(
            version_string="1.0.0",
            content_hash="hash2",
            point_id="existing-point"
        )

        conflict = self.manager._analyze_version_conflict(new_version, existing_version)

        assert conflict is None

    def test_recommend_resolution_strategy_high_severity(self):
        """Test strategy recommendation for high-severity conflicts."""
        conflict_types = [ConflictType.CONTENT_CONFLICT]
        severity = 0.9

        new_version = self._create_version_info()
        existing_version = self._create_version_info(point_id="existing")

        strategy = self.manager._recommend_resolution_strategy(
            conflict_types, new_version, existing_version, severity
        )

        assert strategy == ResolutionStrategy.USER_DECISION

    def test_recommend_resolution_strategy_format_conflict(self):
        """Test strategy recommendation for format conflicts."""
        conflict_types = [ConflictType.FORMAT_CONFLICT]
        severity = 0.4

        new_version = self._create_version_info()
        existing_version = self._create_version_info(point_id="existing")

        strategy = self.manager._recommend_resolution_strategy(
            conflict_types, new_version, existing_version, severity
        )

        assert strategy == ResolutionStrategy.FORMAT_PRECEDENCE


class TestArchiveManager:
    """Test archive management functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_client = MagicMock(spec=QdrantWorkspaceClient)
        self.mock_client.initialized = True
        self.manager = ArchiveManager(self.mock_client)

    def test_init_with_client(self):
        """Test initializing archive manager with client."""
        assert self.manager.client is self.mock_client
        assert self.manager.archive_suffix == "_archive"
        assert len(self.manager.default_policies) > 0

    def test_get_archive_collection_name(self):
        """Test archive collection naming."""
        assert self.manager.get_archive_collection_name("test-collection") == "test-collection_archive"

    @pytest.mark.asyncio
    async def test_ensure_archive_collection_exists(self):
        """Test ensuring archive collection when it already exists."""
        self.mock_client.list_collections.return_value = ["test-collection", "test-collection_archive"]

        result = await self.manager.ensure_archive_collection("test-collection")

        assert result is True
        # Should not attempt to create collection
        self.mock_client.client.create_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_ensure_archive_collection_creates_new(self):
        """Test creating new archive collection."""
        self.mock_client.list_collections.return_value = ["test-collection"]

        # Mock collection info
        mock_collection_info = MagicMock()
        mock_collection_info.config.params.vectors = {"default": "config"}
        mock_collection_info.config.params.sparse_vectors = {"sparse": "config"}
        self.mock_client.client.get_collection = AsyncMock(return_value=mock_collection_info)
        self.mock_client.client.create_collection = AsyncMock()

        result = await self.manager.ensure_archive_collection("test-collection")

        assert result is True
        self.mock_client.client.create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_archive_document_version_success(self):
        """Test successful document archiving."""
        with patch.object(self.manager, 'ensure_archive_collection', return_value=True):
            # Mock document retrieval
            mock_point = MagicMock()
            mock_point.id = "point123"
            mock_point.payload = {"document_id": "doc1", "version": "1.0.0"}
            mock_point.vector = {"dense": [1, 2, 3]}

            self.mock_client.client.retrieve = AsyncMock(return_value=[mock_point])
            self.mock_client.client.upsert = AsyncMock()
            self.mock_client.client.delete = AsyncMock()

            result = await self.manager.archive_document_version("point123", "test-collection")

            assert result["success"] is True
            assert result["archived_point_id"] == "point123"
            assert result["archive_collection"] == "test-collection_archive"

    @pytest.mark.asyncio
    async def test_archive_document_version_not_found(self):
        """Test archiving non-existent document."""
        with patch.object(self.manager, 'ensure_archive_collection', return_value=True):
            self.mock_client.client.retrieve = AsyncMock(return_value=[])

            result = await self.manager.archive_document_version("nonexistent", "test-collection")

            assert "error" in result
            assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_retrieve_archived_versions_success(self):
        """Test successful retrieval of archived versions."""
        self.mock_client.list_collections.return_value = ["test-collection_archive"]

        # Mock archived versions
        mock_points = []
        for i in range(3):
            mock_point = MagicMock()
            mock_point.id = f"point{i}"
            mock_point.payload = {
                "document_id": "doc1",
                "version": f"1.{i}.0",
                "archive_date": f"2024-01-{i+10:02d}T00:00:00+00:00",
                "archive_status": ArchiveStatus.ARCHIVED.value,
                "importance_score": 0.5
            }
            mock_points.append(mock_point)

        self.mock_client.client.scroll = AsyncMock(return_value=(mock_points, None))

        entries = await self.manager.retrieve_archived_versions("doc1", "test-collection")

        assert len(entries) == 3
        assert all(isinstance(entry, ArchiveEntry) for entry in entries)

    @pytest.mark.asyncio
    async def test_restore_archived_version_success(self):
        """Test successful version restoration."""
        # Mock archived document
        mock_point = MagicMock()
        mock_point.id = "archived_point"
        mock_point.payload = {
            "document_id": "doc1",
            "version": "1.0.0",
            "archive_date": "2024-01-01T00:00:00+00:00",
            "archive_status": ArchiveStatus.ARCHIVED.value
        }
        mock_point.vector = {"dense": [1, 2, 3]}

        self.mock_client.client.retrieve = AsyncMock(return_value=[mock_point])
        self.mock_client.client.upsert = AsyncMock()
        self.mock_client.client.set_payload = AsyncMock()

        result = await self.manager.restore_archived_version(
            "archived_point", "test_archive", "test-collection"
        )

        assert result["success"] is True
        assert result["restored_point_id"] == "archived_point"
        assert result["target_collection"] == "test-collection"

    @pytest.mark.asyncio
    async def test_cleanup_by_time(self):
        """Test time-based cleanup implementation."""
        # Mock old entries
        mock_points = []
        for i in range(5):
            mock_point = MagicMock()
            mock_point.id = f"old_point{i}"
            mock_points.append(mock_point)

        self.mock_client.client.scroll = AsyncMock(return_value=(mock_points, None))
        self.mock_client.client.delete = AsyncMock()

        cleaned_count = await self.manager._cleanup_by_time(
            "test_archive",
            {"retention_days": 30}
        )

        assert cleaned_count == 5
        self.mock_client.client.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_archive_statistics_success(self):
        """Test successful statistics generation."""
        self.mock_client.list_collections.return_value = ["test_archive"]

        # Mock archived documents
        mock_points = []
        archive_dates = ["2024-01-01T00:00:00+00:00", "2024-01-15T00:00:00+00:00", "2024-02-01T00:00:00+00:00"]

        for i, date in enumerate(archive_dates):
            mock_point = MagicMock()
            mock_point.id = f"point{i}"
            mock_point.payload = {
                "document_id": f"doc{i % 2}",
                "archive_date": date,
                "archive_status": ArchiveStatus.ARCHIVED.value
            }
            mock_points.append(mock_point)

        self.mock_client.client.scroll = AsyncMock(return_value=(mock_points, None))

        stats = await self.manager.get_archive_statistics("test")

        assert stats["total_archived"] == 3
        assert stats["unique_documents"] == 2
        assert stats["oldest_entry"] == "2024-01-01T00:00:00+00:00"
        assert stats["newest_entry"] == "2024-02-01T00:00:00+00:00"


class TestWorkflowIntegrator:
    """Test workflow integration functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_client = MagicMock(spec=QdrantWorkspaceClient)
        self.mock_version_manager = MagicMock(spec=VersionManager)
        self.mock_archive_manager = MagicMock(spec=ArchiveManager)

        self.integrator = WorkflowIntegrator(
            self.mock_client,
            self.mock_version_manager,
            self.mock_archive_manager
        )

    def test_init_with_managers(self):
        """Test initializing workflow integrator."""
        assert self.integrator.client is self.mock_client
        assert self.integrator.version_manager is self.mock_version_manager
        assert self.integrator.archive_manager is self.mock_archive_manager

    def test_register_notification_callback(self):
        """Test registering notification callbacks."""
        callback = MagicMock()
        self.integrator.register_notification_callback(callback)

        assert len(self.integrator.notification_callbacks) == 1
        assert callback in self.integrator.notification_callbacks

    @pytest.mark.asyncio
    async def test_process_document_ingestion_no_conflicts(self):
        """Test document ingestion with no conflicts."""
        # Mock version manager methods
        self.mock_version_manager.detect_document_type.return_value = DocumentType.CODE_FILE
        self.mock_version_manager.extract_version_from_content.return_value = "1.0.0"
        self.mock_version_manager.calculate_content_hash.return_value = "hash123"
        self.mock_version_manager.get_file_format.return_value = FileFormat.TXT
        self.mock_version_manager.find_conflicting_versions = AsyncMock(return_value=[])

        # Mock document ingestion
        with patch('workspace_qdrant_mcp.tools.documents.ingest_new_version') as mock_ingest:
            mock_ingest.return_value = {
                "success": True,
                "document_id": "test-doc",
                "chunks_added": 1
            }

            result = await self.integrator.process_document_ingestion_with_workflow(
                content="test content",
                collection="test-collection",
                metadata={"title": "Test Document"},
                document_id="test-doc"
            )

            assert result["success"] is True
            assert "workflow_id" in result
            assert result["document_id"] == "test-doc"
            assert result["conflicts_resolved"] == 0

    def test_generate_user_message(self):
        """Test user message generation."""
        # Create mock conflict
        mock_conflict = MagicMock(spec=VersionConflict)
        mock_conflict.user_message = "Version conflict detected"
        mock_conflict.conflict_severity = 0.7
        mock_conflict.recommended_strategy = ResolutionStrategy.FORMAT_PRECEDENCE

        # Create mock version info
        existing_version = MagicMock()
        existing_version.version_string = "1.0.0"
        existing_version.format = FileFormat.TXT

        new_version = MagicMock()
        new_version.version_string = "2.0.0"
        new_version.format = FileFormat.PDF

        mock_conflict.conflicting_versions = [new_version, existing_version]

        message = self.integrator._generate_user_message(mock_conflict, new_version)

        assert "Version conflict detected" in message
        assert "Existing version: 1.0.0 (txt)" in message
        assert "New version: 2.0.0 (pdf)" in message
        assert "Conflict severity: 0.70" in message

    def test_generate_resolution_options(self):
        """Test resolution options generation."""
        mock_conflict = MagicMock(spec=VersionConflict)
        mock_conflict.resolution_options = [
            {
                "action": "keep_new",
                "description": "Keep new version",
                "strategy": ResolutionStrategy.FORMAT_PRECEDENCE
            }
        ]
        mock_conflict.recommended_strategy = ResolutionStrategy.FORMAT_PRECEDENCE

        options = self.integrator._generate_resolution_options(mock_conflict)

        assert len(options) >= 3  # At least original + manual_review + skip

        # Check required fields
        for option in options:
            assert "value" in option
            assert "label" in option
            assert "strategy" in option
            assert "recommended" in option

    def test_get_workflow_status_not_found(self):
        """Test getting status of non-existent workflow."""
        status = self.integrator.get_workflow_status("nonexistent_workflow")

        assert "error" in status
        assert "not found" in status["error"]

    def test_list_pending_prompts_empty(self):
        """Test listing pending prompts when none exist."""
        prompts = self.integrator.list_pending_prompts()
        assert prompts == []


class TestEdgeCases:
    """Test various edge cases and error conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_client = MagicMock(spec=QdrantWorkspaceClient)

    def test_version_manager_client_not_initialized(self):
        """Test version manager when client not initialized."""
        self.mock_client.initialized = False
        manager = VersionManager(self.mock_client)

        # This should not crash, just return empty results
        assert manager.client is self.mock_client

    def test_archive_manager_client_not_initialized(self):
        """Test archive manager when client not initialized."""
        self.mock_client.initialized = False
        manager = ArchiveManager(self.mock_client)

        assert manager.client is self.mock_client

    @pytest.mark.asyncio
    async def test_version_manager_find_versions_client_error(self):
        """Test handling client errors during version search."""
        self.mock_client.initialized = True
        self.mock_client.client.scroll = AsyncMock(side_effect=Exception("Database error"))

        manager = VersionManager(self.mock_client)

        VersionInfo(
            version_string="1.0.0",
            version_type="semantic",
            document_type=DocumentType.CODE_FILE,
            authority_level=0.7,
            timestamp=datetime.now(timezone.utc),
            content_hash="hash",
            format=FileFormat.TXT,
            metadata={},
            point_id=""
        )

        versions = await manager._get_existing_versions("doc1", "collection")
        assert versions == []

    @pytest.mark.asyncio
    async def test_archive_manager_error_handling(self):
        """Test archive manager error handling."""
        self.mock_client.initialized = False
        manager = ArchiveManager(self.mock_client)

        result = await manager.archive_document_version("point", "collection")

        assert "error" in result
        assert "not initialized" in result["error"]

    def test_malformed_version_comparison(self):
        """Test version comparison with malformed data."""
        manager = VersionManager(self.mock_client)

        # Should not crash with empty strings
        result = manager.compare_versions("", "", DocumentType.CODE_FILE)
        assert result == 0

        # Should handle malformed dates
        result = manager.compare_versions("invalid-date", "2024-01-01", DocumentType.SCIENTIFIC_ARTICLE)
        # Should fall back to string comparison
        assert isinstance(result, int)

    def test_unicode_content_handling(self):
        """Test handling of unicode content."""
        manager = VersionManager(self.mock_client)

        content = "Test with unicode: 🎉 émojis and special chars"
        hash_result = manager.calculate_content_hash(content)

        assert isinstance(hash_result, str)
        assert len(hash_result) == 64  # SHA256 hex length


if __name__ == "__main__":
    pytest.main([__file__])
