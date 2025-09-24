"""
Unit tests for MetadataAggregator.

This module tests the metadata aggregation functionality, including parser-specific
metadata normalization, batch processing, and validation capabilities.
"""

import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.python.wqm_cli.cli.metadata.aggregator import (
    MetadataAggregator,
    DocumentMetadata,
)
from src.python.wqm_cli.cli.metadata.exceptions import AggregationError
from src.python.wqm_cli.cli.parsers.base import ParsedDocument
from src.python.common.core.metadata_schema import (
    MultiTenantMetadataSchema,
    CollectionCategory,
)


class TestDocumentMetadata:
    """Test cases for DocumentMetadata class."""

    def test_init_with_basic_data(self):
        """Test DocumentMetadata initialization with basic data."""
        # Create mock parsed document
        parsed_doc = ParsedDocument.create(
            content="test content",
            file_path="/test/document.txt",
            file_type="text",
            additional_metadata={"encoding": "utf-8"},
        )

        doc_metadata = DocumentMetadata(
            file_path="/test/document.txt",
            content_hash="abc123",
            parsed_document=parsed_doc,
        )

        assert doc_metadata.file_path == "/test/document.txt"
        assert doc_metadata.content_hash == "abc123"
        assert doc_metadata.parsed_document == parsed_doc
        assert doc_metadata.collection_metadata is None
        assert doc_metadata.aggregated_at  # Should be set

    def test_init_with_collection_metadata(self):
        """Test DocumentMetadata initialization with collection metadata."""
        parsed_doc = ParsedDocument.create(
            content="test content",
            file_path="/test/document.txt",
            file_type="text",
        )

        collection_metadata = MultiTenantMetadataSchema.create_for_project(
            project_name="test_project",
            collection_type="docs",
        )

        doc_metadata = DocumentMetadata(
            file_path="/test/document.txt",
            content_hash="abc123",
            parsed_document=parsed_doc,
            collection_metadata=collection_metadata,
        )

        assert doc_metadata.collection_metadata == collection_metadata

    def test_to_dict_with_content(self):
        """Test converting DocumentMetadata to dict with content."""
        parsed_doc = ParsedDocument.create(
            content="test content",
            file_path="/test/document.txt",
            file_type="text",
            additional_metadata={"encoding": "utf-8"},
        )

        doc_metadata = DocumentMetadata(
            file_path="/test/document.txt",
            content_hash="abc123",
            parsed_document=parsed_doc,
        )

        result = doc_metadata.to_dict(include_content=True)

        assert result["file_path"] == "/test/document.txt"
        assert result["content_hash"] == "abc123"
        assert result["file_type"] == "text"
        assert result["content"] == "test content"
        assert "metadata" in result
        assert "parsing_info" in result
        assert result["aggregated_at"]

    def test_to_dict_without_content(self):
        """Test converting DocumentMetadata to dict without content."""
        parsed_doc = ParsedDocument.create(
            content="test content",
            file_path="/test/document.txt",
            file_type="text",
        )

        doc_metadata = DocumentMetadata(
            file_path="/test/document.txt",
            content_hash="abc123",
            parsed_document=parsed_doc,
        )

        result = doc_metadata.to_dict(include_content=False)

        assert "content" not in result
        assert result["file_path"] == "/test/document.txt"

    def test_to_dict_with_collection_metadata(self):
        """Test converting DocumentMetadata to dict with collection metadata."""
        parsed_doc = ParsedDocument.create(
            content="test content",
            file_path="/test/document.txt",
            file_type="text",
        )

        collection_metadata = MultiTenantMetadataSchema.create_for_project(
            project_name="test_project",
            collection_type="docs",
        )

        doc_metadata = DocumentMetadata(
            file_path="/test/document.txt",
            content_hash="abc123",
            parsed_document=parsed_doc,
            collection_metadata=collection_metadata,
        )

        result = doc_metadata.to_dict()

        assert "collection_metadata" in result
        assert result["collection_metadata"]["project_name"] == "test_project"


class TestMetadataAggregator:
    """Test cases for MetadataAggregator class."""

    def test_init(self):
        """Test MetadataAggregator initialization."""
        aggregator = MetadataAggregator()

        assert aggregator.parser_registry
        assert "pdf" in aggregator.parser_registry
        assert "epub" in aggregator.parser_registry
        assert "web" in aggregator.parser_registry

    def test_register_known_parsers(self):
        """Test that known parsers are registered correctly."""
        aggregator = MetadataAggregator()

        # Check PDF parser fields
        pdf_fields = aggregator.parser_registry["pdf"]
        assert "page_count" in pdf_fields
        assert "title" in pdf_fields
        assert "author" in pdf_fields
        assert "is_encrypted" in pdf_fields

        # Check web parser fields
        web_fields = aggregator.parser_registry["web"]
        assert "url" in web_fields
        assert "domain" in web_fields
        assert "title" in web_fields
        assert "status_code" in web_fields

    def test_aggregate_metadata_basic(self):
        """Test basic metadata aggregation."""
        aggregator = MetadataAggregator()

        parsed_doc = ParsedDocument.create(
            content="test content",
            file_path="/test/document.txt",
            file_type="text",
            additional_metadata={"encoding": "utf-8", "line_count": 5},
        )

        result = aggregator.aggregate_metadata(parsed_doc)

        assert isinstance(result, DocumentMetadata)
        assert result.file_path == "/test/document.txt"
        assert result.parsed_document.file_type == "text"
        assert result.parsed_document.metadata["parser_type"] == "text"
        assert "known_fields" in result.parsed_document.metadata

    def test_aggregate_metadata_with_project(self):
        """Test metadata aggregation with project name."""
        aggregator = MetadataAggregator()

        parsed_doc = ParsedDocument.create(
            content="test content",
            file_path="/test/document.txt",
            file_type="text",
        )

        result = aggregator.aggregate_metadata(
            parsed_doc, project_name="test_project", collection_type="docs"
        )

        assert result.collection_metadata is not None
        assert result.collection_metadata.project_name == "test_project"
        assert result.collection_metadata.collection_type == "docs"

    def test_aggregate_metadata_invalid_document(self):
        """Test aggregation with invalid document."""
        aggregator = MetadataAggregator()

        # Create invalid document (no content)
        parsed_doc = ParsedDocument(
            content="",  # Empty content
            file_path="/test/document.txt",
            file_type="text",
            metadata={},
            content_hash="abc123",
            parsed_at=datetime.now(timezone.utc).isoformat(),
            file_size=0,
        )

        with pytest.raises(AggregationError) as exc_info:
            aggregator.aggregate_metadata(parsed_doc)

        assert "has no content" in str(exc_info.value)

    def test_aggregate_batch_metadata(self):
        """Test batch metadata aggregation."""
        aggregator = MetadataAggregator()

        # Create multiple parsed documents
        parsed_docs = []
        for i in range(3):
            doc = ParsedDocument.create(
                content=f"test content {i}",
                file_path=f"/test/document_{i}.txt",
                file_type="text",
            )
            parsed_docs.append(doc)

        results = aggregator.aggregate_batch_metadata(parsed_docs, project_name="test_project")

        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.file_path == f"/test/document_{i}.txt"
            assert result.collection_metadata.project_name == "test_project"

    def test_aggregate_batch_metadata_with_failures(self):
        """Test batch aggregation with some failures."""
        aggregator = MetadataAggregator()

        # Create mix of valid and invalid documents
        parsed_docs = []

        # Valid document
        valid_doc = ParsedDocument.create(
            content="valid content",
            file_path="/test/valid.txt",
            file_type="text",
        )
        parsed_docs.append(valid_doc)

        # Invalid document (no content)
        invalid_doc = ParsedDocument(
            content="",
            file_path="/test/invalid.txt",
            file_type="text",
            metadata={},
            content_hash="abc123",
            parsed_at=datetime.now(timezone.utc).isoformat(),
            file_size=0,
        )
        parsed_docs.append(invalid_doc)

        # This should return partial results (only valid document)
        results = aggregator.aggregate_batch_metadata(parsed_docs)

        assert len(results) == 1
        assert results[0].file_path == "/test/valid.txt"

    def test_aggregate_batch_metadata_all_failures(self):
        """Test batch aggregation with all failures."""
        aggregator = MetadataAggregator()

        # Create only invalid documents
        parsed_docs = []
        for i in range(2):
            invalid_doc = ParsedDocument(
                content="",  # Empty content
                file_path=f"/test/invalid_{i}.txt",
                file_type="text",
                metadata={},
                content_hash=f"abc{i}",
                parsed_at=datetime.now(timezone.utc).isoformat(),
                file_size=0,
            )
            parsed_docs.append(invalid_doc)

        with pytest.raises(AggregationError) as exc_info:
            aggregator.aggregate_batch_metadata(parsed_docs)

        assert "Failed to aggregate metadata for entire batch" in str(exc_info.value)

    def test_normalize_pdf_metadata(self):
        """Test PDF-specific metadata normalization."""
        aggregator = MetadataAggregator()

        parsed_doc = ParsedDocument.create(
            content="pdf content",
            file_path="/test/document.pdf",
            file_type="pdf",
            additional_metadata={
                "page_count": 10,
                "title": "Test PDF",
                "pdf_version": "4",  # Should be normalized to 1.4
                "creation_date": "2023-01-01T00:00:00",
            },
        )

        result = aggregator.aggregate_metadata(parsed_doc)

        metadata = result.parsed_document.metadata
        assert metadata["parser_type"] == "pdf"
        assert metadata["has_text_layer"] is True  # Default value
        assert metadata["requires_ocr"] is False  # Default value
        assert metadata["pdf_version"] == "1.4"  # Normalized
        assert metadata["page_count"] == 10  # Integer

    def test_normalize_web_metadata(self):
        """Test web content metadata normalization."""
        aggregator = MetadataAggregator()

        parsed_doc = ParsedDocument.create(
            content="web content",
            file_path="https://example.com/page.html",
            file_type="web",
            additional_metadata={
                "url": "example.com/page.html",  # Missing protocol
                "canonical_url": "example.com/canonical",  # Missing protocol
                "title": "Test Page",
                "link_count": "5",  # String number
            },
        )

        result = aggregator.aggregate_metadata(parsed_doc)

        metadata = result.parsed_document.metadata
        assert metadata["parser_type"] == "web"
        assert metadata["url"] == "https://example.com/page.html"
        assert metadata["canonical_url"] == "https://example.com/canonical"
        assert metadata["status_code"] == 200  # Default value
        assert metadata["link_count"] == 5  # Converted to int

    def test_normalize_ebook_metadata(self):
        """Test EPUB/MOBI metadata normalization."""
        aggregator = MetadataAggregator()

        parsed_doc = ParsedDocument.create(
            content="epub content",
            file_path="/test/book.epub",
            file_type="epub",
            additional_metadata={
                "title": "Test Book",
                "author": "Test Author",
                "isbn": "978-0-123456-78-9",  # With hyphens
                "spine_items": ["chapter1", "chapter2", "chapter3"],
            },
        )

        result = aggregator.aggregate_metadata(parsed_doc)

        metadata = result.parsed_document.metadata
        assert metadata["parser_type"] == "epub"
        assert metadata["isbn"] == "9780123456789"  # Hyphens removed
        assert metadata["chapter_count"] == 3  # Calculated from spine_items

    def test_normalize_code_metadata(self):
        """Test code parser metadata normalization."""
        aggregator = MetadataAggregator()

        parsed_doc = ParsedDocument.create(
            content="print('hello world')",
            file_path="/test/script.py",
            file_type="code",
            additional_metadata={
                "language": "python",
                "function_count": "2",  # String number
                "class_count": "1",  # String number
                "complexity_score": "3.5",  # String float
                "line_count": 10,
            },
        )

        result = aggregator.aggregate_metadata(parsed_doc)

        metadata = result.parsed_document.metadata
        assert metadata["parser_type"] == "code"
        assert metadata["function_count"] == 2
        assert metadata["class_count"] == 1
        assert metadata["complexity_score"] == 3.5
        assert metadata["line_count"] == 10

    def test_normalize_date_fields(self):
        """Test date field normalization."""
        aggregator = MetadataAggregator()

        # Test with various date formats
        test_dates = {
            "creation_date": "2023-01-01T00:00:00Z",
            "modification_date": datetime(2023, 1, 2, 12, 0, 0, tzinfo=timezone.utc),
            "publication_date": "2023-01-03",
        }

        parsed_doc = ParsedDocument.create(
            content="test content",
            file_path="/test/document.txt",
            file_type="text",
            additional_metadata=test_dates,
        )

        result = aggregator.aggregate_metadata(parsed_doc)
        metadata = result.parsed_document.metadata

        # All dates should be normalized to ISO format strings
        assert isinstance(metadata["creation_date"], str)
        assert isinstance(metadata["modification_date"], str)
        assert isinstance(metadata["publication_date"], str)

    def test_get_supported_parsers(self):
        """Test getting supported parser types."""
        aggregator = MetadataAggregator()

        supported = aggregator.get_supported_parsers()

        assert isinstance(supported, list)
        assert "pdf" in supported
        assert "epub" in supported
        assert "web" in supported
        assert "text" in supported

    def test_get_parser_fields(self):
        """Test getting parser fields."""
        aggregator = MetadataAggregator()

        pdf_fields = aggregator.get_parser_fields("pdf")
        assert isinstance(pdf_fields, set)
        assert "page_count" in pdf_fields
        assert "title" in pdf_fields

        # Test unknown parser
        unknown_fields = aggregator.get_parser_fields("unknown")
        assert isinstance(unknown_fields, set)
        assert len(unknown_fields) == 0

    def test_validate_metadata_completeness(self):
        """Test metadata completeness validation."""
        aggregator = MetadataAggregator()

        # Create document with partial metadata
        parsed_doc = ParsedDocument.create(
            content="test content",
            file_path="/test/document.pdf",
            file_type="pdf",
            additional_metadata={
                "page_count": 5,
                "title": "Test PDF",
                # Missing: author, creation_date, etc.
            },
        )

        doc_metadata = aggregator.aggregate_metadata(parsed_doc)
        validation_report = aggregator.validate_metadata_completeness(doc_metadata)

        assert validation_report["parser_type"] == "pdf"
        assert validation_report["expected_fields"] > 0
        assert validation_report["actual_fields"] > 0
        assert "missing_fields" in validation_report
        assert "extra_fields" in validation_report
        assert 0 <= validation_report["completeness_score"] <= 1

    def test_edge_case_empty_metadata(self):
        """Test handling of empty metadata."""
        aggregator = MetadataAggregator()

        parsed_doc = ParsedDocument.create(
            content="test content",
            file_path="/test/document.txt",
            file_type="text",
            additional_metadata={},  # No additional metadata
        )

        result = aggregator.aggregate_metadata(parsed_doc)

        # Should still work and add parser-specific fields
        metadata = result.parsed_document.metadata
        assert metadata["parser_type"] == "text"
        assert "known_fields" in metadata

    def test_edge_case_malformed_data(self):
        """Test handling of malformed metadata values."""
        aggregator = MetadataAggregator()

        parsed_doc = ParsedDocument.create(
            content="test content",
            file_path="/test/document.pdf",
            file_type="pdf",
            additional_metadata={
                "page_count": "not_a_number",  # Invalid integer
                "creation_date": None,  # Null date
                "is_encrypted": "maybe",  # Invalid boolean
                "pdf_version": "",  # Empty string
            },
        )

        # Should not raise exception but handle gracefully
        result = aggregator.aggregate_metadata(parsed_doc)

        metadata = result.parsed_document.metadata
        assert metadata["page_count"] == 0  # Default for invalid int
        assert metadata["creation_date"] is None  # Null date handled
        assert metadata["is_encrypted"] is True  # Truthy value converted
        assert metadata["pdf_version"] == "1."  # Empty string handled

    def test_unicode_handling(self):
        """Test handling of Unicode content and metadata."""
        aggregator = MetadataAggregator()

        parsed_doc = ParsedDocument.create(
            content="Test content with émojis 🎉 and üñíçødé",
            file_path="/test/unicode_document.txt",
            file_type="text",
            additional_metadata={
                "title": "Tëst Dòcümënt",
                "author": "Ñâmé wïth Àccénts",
            },
        )

        result = aggregator.aggregate_metadata(parsed_doc)

        # Should handle Unicode without errors
        metadata = result.parsed_document.metadata
        assert "title" in metadata
        assert "author" in metadata
        assert result.parsed_document.content  # Content preserved


# Test fixtures and helpers

@pytest.fixture
def sample_parsed_document():
    """Create a sample ParsedDocument for testing."""
    return ParsedDocument.create(
        content="Sample test content for metadata aggregation testing.",
        file_path="/test/sample_document.txt",
        file_type="text",
        additional_metadata={
            "encoding": "utf-8",
            "line_count": 1,
            "word_count": 8,
        },
    )


@pytest.fixture
def metadata_aggregator():
    """Create a MetadataAggregator instance for testing."""
    return MetadataAggregator()


@pytest.fixture
def mock_collection_metadata():
    """Create mock collection metadata."""
    return MultiTenantMetadataSchema.create_for_project(
        project_name="test_project",
        collection_type="test_docs",
        created_by="test_user",
    )