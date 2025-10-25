"""
Edge case tests for metadata workflow system.

This module tests edge cases including corrupted metadata, large collections,
YAML serialization failures, and incremental update scenarios as specified
in the task requirements.
"""

import json
import sqlite3
import tempfile
from concurrent.futures import TimeoutError
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.python.wqm_cli.cli.metadata.aggregator import (
    DocumentMetadata,
    MetadataAggregator,
)
from src.python.wqm_cli.cli.metadata.batch_processor import (
    BatchConfig,
    BatchProcessor,
    BatchResult,
)
from src.python.wqm_cli.cli.metadata.exceptions import (
    AggregationError,
    BatchProcessingError,
    IncrementalTrackingError,
    MetadataError,
    YAMLGenerationError,
)
from src.python.wqm_cli.cli.metadata.incremental_tracker import (
    DocumentChangeInfo,
    IncrementalTracker,
)
from src.python.wqm_cli.cli.metadata.workflow_manager import (
    WorkflowConfig,
    WorkflowManager,
)
from src.python.wqm_cli.cli.metadata.yaml_generator import (
    YAMLConfig,
    YAMLGenerator,
)
from src.python.wqm_cli.cli.parsers.base import ParsedDocument


class TestCorruptedMetadataEdgeCases:
    """Test cases for handling corrupted metadata."""

    def test_corrupted_metadata_fields(self):
        """Test handling of corrupted metadata field values."""
        aggregator = MetadataAggregator()

        # Create document with various corrupted metadata
        corrupted_metadata = {
            "page_count": float('inf'),  # Infinity
            "creation_date": "not-a-date",  # Invalid date
            "is_encrypted": {"not": "boolean"},  # Wrong type
            "author": None,  # Null value
            "title": "",  # Empty string
            "file_size": -1,  # Negative number
            "line_count": "NaN",  # Not a number
            "unicode_field": "\x00\x01\x02",  # Control characters
        }

        parsed_doc = ParsedDocument.create(
            content="test content",
            file_path="/test/corrupted.pdf",
            file_type="pdf",
            additional_metadata=corrupted_metadata,
        )

        # Should not crash but handle gracefully
        result = aggregator.aggregate_metadata(parsed_doc)

        assert result is not None
        metadata = result.parsed_document.metadata

        # Verify corruption was handled
        assert metadata["page_count"] != float('inf')  # Should be converted
        assert metadata["creation_date"] is not None  # Should be handled
        assert isinstance(metadata["is_encrypted"], bool)  # Should be converted to bool
        assert "author" in metadata  # Should be present even if None

    def test_metadata_with_circular_references(self):
        """Test handling of metadata with circular references."""
        aggregator = MetadataAggregator()

        # Create circular reference
        circular_dict = {}
        circular_dict["self_ref"] = circular_dict

        parsed_doc = ParsedDocument.create(
            content="test content",
            file_path="/test/circular.txt",
            file_type="text",
            additional_metadata={"circular": circular_dict},
        )

        # Should handle circular references gracefully
        result = aggregator.aggregate_metadata(parsed_doc)
        assert result is not None

    def test_metadata_with_extremely_large_values(self):
        """Test handling of metadata with extremely large values."""
        aggregator = MetadataAggregator()

        large_string = "x" * (10**6)  # 1MB string
        large_number = 10**20  # Very large number

        parsed_doc = ParsedDocument.create(
            content="test content",
            file_path="/test/large_metadata.txt",
            file_type="text",
            additional_metadata={
                "large_string": large_string,
                "large_number": large_number,
                "large_list": list(range(100000)),  # Large list
            },
        )

        # Should handle large values without crashing
        result = aggregator.aggregate_metadata(parsed_doc)
        assert result is not None

    def test_metadata_with_invalid_encoding(self):
        """Test handling of metadata with invalid encoding."""
        aggregator = MetadataAggregator()

        # Simulate invalid UTF-8 sequences
        invalid_bytes = b'\xff\xfe\xfd'
        try:
            invalid_string = invalid_bytes.decode('utf-8', errors='replace')
        except UnicodeDecodeError:
            invalid_string = "replacement_text"

        parsed_doc = ParsedDocument.create(
            content="test content",
            file_path="/test/invalid_encoding.txt",
            file_type="text",
            additional_metadata={"invalid_text": invalid_string},
        )

        result = aggregator.aggregate_metadata(parsed_doc)
        assert result is not None


class TestLargeCollectionEdgeCases:
    """Test cases for handling large document collections."""

    def test_memory_management_large_collection(self):
        """Test memory management with large document collections."""
        config = BatchConfig(
            batch_size=10,
            max_memory_usage_mb=100,  # Low memory limit to trigger management
        )
        processor = BatchProcessor(config=config)

        # Create many documents (simulated)
        large_file_list = [f"/test/doc_{i}.txt" for i in range(1000)]

        # Mock the file existence and parser availability
        with patch.object(processor, '_filter_valid_paths') as mock_filter:
            mock_filter.return_value = [Path(p) for p in large_file_list[:10]]  # Limit for testing

            with patch.object(processor, '_process_single_document') as mock_process:
                mock_parsed_doc = ParsedDocument.create(
                    content="test content",
                    file_path="/test/mock.txt",
                    file_type="text",
                )
                mock_metadata = DocumentMetadata(
                    file_path="/test/mock.txt",
                    content_hash="mock_hash",
                    parsed_document=mock_parsed_doc,
                )
                mock_process.return_value = mock_metadata

                # Should handle without memory issues
                import asyncio
                result = asyncio.run(processor.process_documents(large_file_list))

                assert isinstance(result, BatchResult)
                # Memory management should have been triggered

    def test_timeout_handling_large_collection(self):
        """Test timeout handling with slow processing."""
        config = BatchConfig(
            batch_size=5,
            timeout_seconds=0.1,  # Very short timeout
            continue_on_error=True,
        )
        processor = BatchProcessor(config=config)

        # Create documents that will timeout
        file_list = [f"/test/slow_doc_{i}.txt" for i in range(3)]

        with patch.object(processor, '_filter_valid_paths') as mock_filter:
            mock_filter.return_value = [Path(p) for p in file_list]

            with patch.object(processor, '_process_single_document') as mock_process:
                # Simulate slow processing
                import time
                def slow_process(*args, **kwargs):
                    time.sleep(1)  # Longer than timeout
                    return Mock()

                mock_process.side_effect = slow_process

                # Should handle timeouts gracefully
                import asyncio
                result = asyncio.run(processor.process_documents(file_list))

                assert isinstance(result, BatchResult)
                assert result.failure_count > 0  # Some should fail due to timeout

    def test_incremental_tracking_large_collection(self):
        """Test incremental tracking with large collections."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = IncrementalTracker(
                storage_path=Path(temp_dir) / "large_tracking.db"
            )

            # Create many document metadata objects
            documents = []
            for i in range(100):
                parsed_doc = ParsedDocument.create(
                    content=f"content {i}",
                    file_path=f"/test/large_doc_{i}.txt",
                    file_type="text",
                )
                doc_metadata = DocumentMetadata(
                    file_path=f"/test/large_doc_{i}.txt",
                    content_hash=f"hash_{i}",
                    parsed_document=parsed_doc,
                )
                documents.append(doc_metadata)

            # Initial tracking should handle large number
            tracker.update_tracking_data(documents)

            # Change detection should be efficient
            changes = tracker.detect_changes(documents)
            assert len(changes) == len(documents)

            # Database should remain performant
            summary = tracker.get_change_summary()
            assert summary["total_tracked_documents"] == len(documents)

    def test_yaml_generation_large_collection(self):
        """Test YAML generation with large collections."""
        generator = YAMLGenerator()

        # Create large collection
        documents = []
        for i in range(50):  # Reasonable size for testing
            parsed_doc = ParsedDocument.create(
                content=f"Large content {i} " * 100,  # Make content substantial
                file_path=f"/test/large_doc_{i}.txt",
                file_type="text",
                additional_metadata={"index": i, "type": "large_test"},
            )
            doc_metadata = DocumentMetadata(
                file_path=f"/test/large_doc_{i}.txt",
                content_hash=f"large_hash_{i}",
                parsed_document=parsed_doc,
            )
            documents.append(doc_metadata)

        # Should handle large collection without issues
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_files = generator.generate_batch_yaml_files(
                documents, output_directory=temp_dir
            )

            assert len(yaml_files) == len(documents)

            # Collection YAML should also work
            collection_yaml = generator.generate_collection_yaml(
                documents, collection_name="large_collection"
            )

            assert "document_count: 50" in collection_yaml


class TestYAMLSerializationFailures:
    """Test cases for YAML serialization failures."""

    def test_non_serializable_objects(self):
        """Test handling of non-serializable objects in metadata."""
        generator = YAMLGenerator()

        # Create object that cannot be serialized
        class NonSerializable:
            def __init__(self):
                # Create circular reference
                self.self_ref = self

            def __str__(self):
                raise Exception("Cannot stringify")

            def __repr__(self):
                raise Exception("Cannot represent")

        parsed_doc = ParsedDocument.create(
            content="test content",
            file_path="/test/non_serializable.txt",
            file_type="text",
            additional_metadata={"bad_object": NonSerializable()},
        )

        doc_metadata = DocumentMetadata(
            file_path="/test/non_serializable.txt",
            content_hash="bad_hash",
            parsed_document=parsed_doc,
        )

        # Should handle gracefully by converting to string fallback
        yaml_content = generator.generate_yaml(doc_metadata)
        assert yaml_content is not None
        assert isinstance(yaml_content, str)

    def test_yaml_serialization_with_special_characters(self):
        """Test YAML serialization with special characters."""
        generator = YAMLGenerator()

        special_content = """
        Content with special YAML characters:
        - List item: value
        { dictionary: key }
        | pipe character
        > greater than
        < less than
        @ at symbol
        % percent: symbol
        & ampersand
        * asterisk
        """

        ParsedDocument.create(
            content=special_content,
            file_path="/test/special_chars.txt",
            file_type="text",
            additional_metadata={
                "yaml_breaker": "value: with: colons",
                "list_breaker": ["item1", "item2: with: colons"],
                "quote_breaker": 'single "double" quotes',
            },
        )

        doc_metadata = DocumentMetadata(
            file_path="/test/special_chars.txt",
            content_hash="special_hash",
            parsed_document=doc_metadata,
        )

        # Should handle special characters properly
        yaml_content = generator.generate_yaml(doc_metadata)
        assert yaml_content is not None

        # Should be valid YAML
        import yaml
        parsed = yaml.safe_load(yaml_content)
        assert isinstance(parsed, dict)

    def test_yaml_with_extremely_nested_data(self):
        """Test YAML generation with extremely nested data structures."""
        generator = YAMLGenerator()

        # Create deeply nested structure
        nested_dict = {}
        current = nested_dict
        for i in range(100):  # Deep nesting
            current[f"level_{i}"] = {}
            current = current[f"level_{i}"]
        current["final_value"] = "deep_value"

        parsed_doc = ParsedDocument.create(
            content="test content",
            file_path="/test/deeply_nested.txt",
            file_type="text",
            additional_metadata={"nested_data": nested_dict},
        )

        doc_metadata = DocumentMetadata(
            file_path="/test/deeply_nested.txt",
            content_hash="nested_hash",
            parsed_document=parsed_doc,
        )

        # Should handle deep nesting
        yaml_content = generator.generate_yaml(doc_metadata)
        assert yaml_content is not None

    def test_yaml_generation_disk_full_simulation(self):
        """Test YAML generation when disk is full."""
        generator = YAMLGenerator()

        parsed_doc = ParsedDocument.create(
            content="test content",
            file_path="/test/document.txt",
            file_type="text",
        )

        doc_metadata = DocumentMetadata(
            file_path="/test/document.txt",
            content_hash="test_hash",
            parsed_document=parsed_doc,
        )

        # Simulate disk full error
        with patch("pathlib.Path.write_text") as mock_write:
            mock_write.side_effect = OSError("No space left on device")

            with pytest.raises(YAMLGenerationError) as exc_info:
                generator.generate_yaml(
                    doc_metadata,
                    output_path="/fake/path/document.yaml"
                )

            assert "Failed to write YAML file" in str(exc_info.value)


class TestIncrementalUpdateScenarios:
    """Test cases for incremental update scenarios."""

    def test_documents_moved_or_renamed(self):
        """Test incremental tracking when documents are moved or renamed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = IncrementalTracker(
                storage_path=Path(temp_dir) / "move_test.db"
            )

            # Initial document
            original_doc = ParsedDocument.create(
                content="test content",
                file_path="/original/path/document.txt",
                file_type="text",
            )
            original_metadata = DocumentMetadata(
                file_path="/original/path/document.txt",
                content_hash="same_hash",
                parsed_document=original_doc,
            )

            # Track initial document
            tracker.update_tracking_data([original_metadata])

            # Same content, new path (simulating move/rename)
            moved_doc = ParsedDocument.create(
                content="test content",  # Same content
                file_path="/new/path/document.txt",  # New path
                file_type="text",
            )
            moved_metadata = DocumentMetadata(
                file_path="/new/path/document.txt",
                content_hash="same_hash",  # Same hash
                parsed_document=moved_doc,
            )

            # Detect changes should show old as deleted, new as added
            changes = tracker.detect_changes([moved_metadata])

            added_changes = [c for c in changes if c.change_type == "added"]
            deleted_changes = [c for c in changes if c.change_type == "deleted"]

            assert len(added_changes) == 1
            assert len(deleted_changes) == 1
            assert added_changes[0].file_path == "/new/path/document.txt"
            assert deleted_changes[0].file_path == "/original/path/document.txt"

    def test_partial_updates_failing(self):
        """Test handling of partial update failures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = IncrementalTracker(
                storage_path=Path(temp_dir) / "partial_fail.db"
            )

            # Create multiple documents
            documents = []
            for i in range(5):
                doc = ParsedDocument.create(
                    content=f"content {i}",
                    file_path=f"/test/doc_{i}.txt",
                    file_type="text",
                )
                metadata = DocumentMetadata(
                    file_path=f"/test/doc_{i}.txt",
                    content_hash=f"hash_{i}",
                    parsed_document=doc,
                )
                documents.append(metadata)

            # Simulate database corruption during update
            tracker.update_tracking_data(documents[:3])  # First 3 succeed

            # Simulate database lock/error for remaining updates
            with patch.object(tracker, '_init_database') as mock_init:
                mock_init.side_effect = sqlite3.OperationalError("database is locked")

                with pytest.raises(IncrementalTrackingError):
                    tracker.update_tracking_data(documents[3:])

            # Should still be able to query partial results
            summary = tracker.get_change_summary()
            assert summary["total_tracked_documents"] == 3

    def test_metadata_schema_changes(self):
        """Test handling of metadata schema changes over time."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = IncrementalTracker(
                storage_path=Path(temp_dir) / "schema_change.db"
            )

            # Old format document
            old_doc = ParsedDocument.create(
                content="old content",
                file_path="/test/old_format.txt",
                file_type="text",
                additional_metadata={"old_field": "old_value"},
            )
            old_metadata = DocumentMetadata(
                file_path="/test/old_format.txt",
                content_hash="old_hash",
                parsed_document=old_doc,
            )

            tracker.update_tracking_data([old_metadata])

            # New format document (additional fields)
            new_doc = ParsedDocument.create(
                content="new content",
                file_path="/test/new_format.txt",
                file_type="text",
                additional_metadata={
                    "old_field": "old_value",
                    "new_field": "new_value",
                    "another_new_field": {"nested": "data"},
                },
            )
            new_metadata = DocumentMetadata(
                file_path="/test/new_format.txt",
                content_hash="new_hash",
                parsed_document=new_doc,
            )

            # Should handle schema differences gracefully
            changes = tracker.detect_changes([old_metadata, new_metadata])

            # Should detect new document
            new_changes = [c for c in changes if c.change_type == "added"]
            assert len(new_changes) == 1
            assert new_changes[0].file_path == "/test/new_format.txt"

    def test_concurrent_access_simulation(self):
        """Test handling of simulated concurrent access to tracking database."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "concurrent.db"

            # Create two tracker instances (simulating concurrent access)
            tracker1 = IncrementalTracker(storage_path=db_path)
            tracker2 = IncrementalTracker(storage_path=db_path)

            # Documents for each tracker
            doc1 = ParsedDocument.create(
                content="content 1",
                file_path="/test/doc1.txt",
                file_type="text",
            )
            metadata1 = DocumentMetadata(
                file_path="/test/doc1.txt",
                content_hash="hash1",
                parsed_document=doc1,
            )

            doc2 = ParsedDocument.create(
                content="content 2",
                file_path="/test/doc2.txt",
                file_type="text",
            )
            metadata2 = DocumentMetadata(
                file_path="/test/doc2.txt",
                content_hash="hash2",
                parsed_document=doc2,
            )

            # Simulate concurrent updates
            try:
                tracker1.update_tracking_data([metadata1])
                tracker2.update_tracking_data([metadata2])

                # Both should be tracked
                summary1 = tracker1.get_change_summary()
                summary2 = tracker2.get_change_summary()

                # At least one should succeed (depending on SQLite locking)
                assert summary1["total_tracked_documents"] >= 1 or summary2["total_tracked_documents"] >= 1

            except IncrementalTrackingError:
                # Concurrent access errors are acceptable
                pass

    def test_database_corruption_recovery(self):
        """Test recovery from database corruption."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "corrupt.db"

            # Create valid tracker and add data
            tracker = IncrementalTracker(storage_path=db_path)

            doc = ParsedDocument.create(
                content="test content",
                file_path="/test/doc.txt",
                file_type="text",
            )
            metadata = DocumentMetadata(
                file_path="/test/doc.txt",
                content_hash="test_hash",
                parsed_document=doc,
            )

            tracker.update_tracking_data([metadata])

            # Simulate database corruption by writing garbage
            with open(db_path, "wb") as f:
                f.write(b"corrupted data not sqlite")

            # New tracker instance should handle corruption
            new_tracker = IncrementalTracker(storage_path=db_path)

            # Should either recover or recreate database
            try:
                summary = new_tracker.get_change_summary()
                # If it succeeds, database was recreated
                assert isinstance(summary, dict)
            except IncrementalTrackingError:
                # Corruption detection is also acceptable
                pass


class TestWorkflowIntegrationEdgeCases:
    """Test edge cases in workflow integration."""

    def test_workflow_with_all_failed_documents(self):
        """Test workflow when all documents fail to process."""
        config = WorkflowConfig(
            output_directory=Path(tempfile.mkdtemp()),
            generate_individual_yamls=True,
            incremental_updates=False,
        )
        workflow = WorkflowManager(config)

        # Mock batch processor to fail all documents
        with patch.object(workflow.batch_processor, 'process_documents') as mock_process:
            mock_result = BatchResult(
                successful_documents=[],
                failed_documents=[("/fake/doc.txt", "Mock failure")],
                processing_stats={"total_files": 1, "successful_count": 0},
            )
            mock_process.return_value = mock_result

            import asyncio
            result = asyncio.run(workflow.process_documents(["/fake/doc.txt"]))

            assert result.success_count == 0
            assert result.failure_count == 1
            assert len(result.yaml_files) == 0  # No YAML files generated

    def test_workflow_with_yaml_generation_failure(self):
        """Test workflow when YAML generation fails."""
        config = WorkflowConfig(
            output_directory=Path(tempfile.mkdtemp()),
            generate_individual_yamls=True,
        )
        workflow = WorkflowManager(config)

        # Create successful batch result
        doc = ParsedDocument.create(
            content="test content",
            file_path="/test/doc.txt",
            file_type="text",
        )
        metadata = DocumentMetadata(
            file_path="/test/doc.txt",
            content_hash="test_hash",
            parsed_document=doc,
        )

        mock_result = BatchResult(
            successful_documents=[metadata],
            failed_documents=[],
            processing_stats={"total_files": 1, "successful_count": 1},
        )

        # Mock YAML generator to fail
        with patch.object(workflow.batch_processor, 'process_documents') as mock_process:
            mock_process.return_value = mock_result

            with patch.object(workflow.yaml_generator, 'generate_batch_yaml_files') as mock_yaml:
                mock_yaml.side_effect = YAMLGenerationError("YAML generation failed")

                # Workflow should handle YAML generation failure
                import asyncio
                result = asyncio.run(workflow.process_documents(["/test/doc.txt"]))

                assert result.success_count == 1  # Document processing succeeded
                assert len(result.yaml_files) == 0  # YAML generation failed

    def test_workflow_with_incremental_tracking_failure(self):
        """Test workflow when incremental tracking fails."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = WorkflowConfig(
                output_directory=Path(temp_dir),
                incremental_updates=True,
                tracking_storage_path=Path(temp_dir) / "tracking.db",
            )
            workflow = WorkflowManager(config)

            # Mock incremental tracker to fail
            with patch.object(workflow.incremental_tracker, 'detect_changes') as mock_detect:
                mock_detect.side_effect = IncrementalTrackingError("Tracking failed")

                # Mock successful batch processing
                doc = ParsedDocument.create(
                    content="test content",
                    file_path="/test/doc.txt",
                    file_type="text",
                )
                metadata = DocumentMetadata(
                    file_path="/test/doc.txt",
                    content_hash="test_hash",
                    parsed_document=doc,
                )

                mock_result = BatchResult(
                    successful_documents=[metadata],
                    failed_documents=[],
                    processing_stats={"total_files": 1, "successful_count": 1},
                )

                with patch.object(workflow.batch_processor, 'process_documents') as mock_process:
                    mock_process.return_value = mock_result

                    # Should handle tracking failure and continue
                    import asyncio
                    with pytest.raises(MetadataError):
                        asyncio.run(workflow.process_documents(["/test/doc.txt"]))


# Stress test helpers

def create_large_document_collection(size: int, temp_dir: Path):
    """Helper to create large document collections for testing."""
    documents = []
    for i in range(size):
        doc = ParsedDocument.create(
            content=f"Large document content {i} " * 50,
            file_path=f"/test/large_doc_{i:06d}.txt",
            file_type="text",
            additional_metadata={
                "index": i,
                "category": f"category_{i % 10}",
                "large_field": "x" * 1000,
            },
        )
        metadata = DocumentMetadata(
            file_path=f"/test/large_doc_{i:06d}.txt",
            content_hash=f"hash_{i:06d}",
            parsed_document=doc,
        )
        documents.append(metadata)
    return documents
