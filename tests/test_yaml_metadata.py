"""
Tests for the YAML metadata workflow system.

This test module validates the core functionality of the YAML metadata completion
system, including document type detection, metadata extraction, YAML generation,
and processing workflows.
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
import yaml

from workspace_qdrant_mcp.cli.parsers.base import ParsedDocument
from workspace_qdrant_mcp.core.yaml_metadata import (
    DOCUMENT_SCHEMAS,
    DocumentTypeDetector,
    DocumentTypeSchema,
    MetadataExtractor,
    PendingDocument,
    YamlMetadataFile,
    YamlMetadataWorkflow,
)


class TestDocumentTypeSchema:
    """Test DocumentTypeSchema validation and functionality."""

    def test_validate_metadata_complete(self):
        """Test validation with complete metadata."""
        schema = DOCUMENT_SCHEMAS['book']
        metadata = {
            'title': 'Test Book',
            'author': 'Test Author',
            'edition': '1st'
        }

        result = schema.validate_metadata(metadata)

        assert result['missing_required'] == []
        assert result['unknown_fields'] == []

    def test_validate_metadata_missing_required(self):
        """Test validation with missing required fields."""
        schema = DOCUMENT_SCHEMAS['book']
        metadata = {
            'title': 'Test Book',
            # Missing author and edition
        }

        result = schema.validate_metadata(metadata)

        assert 'author' in result['missing_required']
        assert 'edition' in result['missing_required']
        assert len(result['missing_required']) == 2

    def test_validate_metadata_question_marks(self):
        """Test validation treats '?' as missing."""
        schema = DOCUMENT_SCHEMAS['book']
        metadata = {
            'title': 'Test Book',
            'author': '?',
            'edition': '?'
        }

        result = schema.validate_metadata(metadata)

        assert 'author' in result['missing_required']
        assert 'edition' in result['missing_required']

    def test_validate_metadata_unknown_fields(self):
        """Test validation detects unknown fields."""
        schema = DOCUMENT_SCHEMAS['book']
        metadata = {
            'title': 'Test Book',
            'author': 'Test Author',
            'edition': '1st',
            'unknown_field': 'value',
            '_internal_field': 'value'  # Should be ignored (starts with _)
        }

        result = schema.validate_metadata(metadata)

        assert result['missing_required'] == []
        assert 'unknown_field' in result['unknown_fields']
        assert '_internal_field' not in result['unknown_fields']


class TestDocumentTypeDetector:
    """Test document type detection functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = DocumentTypeDetector()

    @pytest.mark.asyncio
    async def test_detect_book_pdf(self):
        """Test detection of book PDF."""
        parsed_doc = ParsedDocument.create(
            content="Chapter 1\n\nThis is the first chapter of our book. ISBN: 978-0123456789",
            file_path="test_book.pdf",
            file_type="pdf",
            additional_metadata={'title': 'Test Book Title'}
        )

        doc_type, confidence = await self.detector.detect_document_type(
            parsed_doc, Path("test_book.pdf")
        )

        assert doc_type == 'book'
        assert confidence > 0.3

    @pytest.mark.asyncio
    async def test_detect_scientific_article(self):
        """Test detection of scientific article."""
        parsed_doc = ParsedDocument.create(
            content="""Abstract

            This paper presents a novel approach to machine learning.
            Keywords: machine learning, AI, research
            DOI: 10.1234/example.2023.001
            Published in Journal of AI Research""",
            file_path="research_paper.pdf",
            file_type="pdf"
        )

        doc_type, confidence = await self.detector.detect_document_type(
            parsed_doc, Path("research_paper.pdf")
        )

        assert doc_type == 'scientific_article'
        assert confidence > 0.3

    @pytest.mark.asyncio
    async def test_detect_manual(self):
        """Test detection of manual/guide."""
        parsed_doc = ParsedDocument.create(
            content="""User Manual

            Step 1: Install the software
            Step 2: Configure settings

            This guide will help you get started.""",
            file_path="user_manual.pdf",
            file_type="pdf"
        )

        doc_type, confidence = await self.detector.detect_document_type(
            parsed_doc, Path("user_manual.pdf")
        )

        assert doc_type == 'manual'
        assert confidence > 0.3

    @pytest.mark.asyncio
    async def test_detect_unknown_low_confidence(self):
        """Test fallback to unknown for low confidence."""
        parsed_doc = ParsedDocument.create(
            content="Some generic content without clear indicators",
            file_path="generic.txt",
            file_type="text"
        )

        doc_type, confidence = await self.detector.detect_document_type(
            parsed_doc, Path("generic.txt")
        )

        # Should fall back to unknown if confidence is too low
        assert doc_type in ['unknown'] or confidence >= 0.3


class TestMetadataExtractor:
    """Test metadata extraction functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = MetadataExtractor()

    def test_extract_title_from_content(self):
        """Test title extraction."""
        content = """Advanced Python Programming

        This book covers advanced topics in Python development."""

        title = self.extractor._extract_title_from_content(content)
        assert title == "Advanced Python Programming"

    def test_extract_author_from_content(self):
        """Test author extraction."""
        content = """Programming Guide
        by John Smith

        This guide covers programming concepts."""

        author = self.extractor._extract_author_from_content(content)
        assert author == "John Smith"

    def test_extract_isbn_from_content(self):
        """Test ISBN extraction."""
        content = """This book is published with ISBN: 978-0123456789"""

        isbn = self.extractor._extract_isbn_from_content(content)
        assert isbn == "9780123456789"

    def test_extract_year_from_content(self):
        """Test year extraction."""
        content = """Copyright 2023 by Publisher Inc."""

        year = self.extractor._extract_year_from_content(content)
        assert year == "2023"

    def test_extract_doi_from_content(self):
        """Test DOI extraction."""
        content = """Published paper
        DOI: 10.1234/example.2023.001
        Abstract follows..."""

        doi = self.extractor._extract_doi_from_content(content)
        assert doi == "10.1234/example.2023.001"


class TestPendingDocument:
    """Test PendingDocument functionality."""

    def test_is_complete_true(self):
        """Test complete document detection."""
        doc = PendingDocument(
            path="/test/book.pdf",
            detected_metadata={'title': 'Test Book'},
            required_metadata={
                'title': 'Test Book',
                'author': 'Test Author',
                'edition': '1st'
            },
            document_type='book'
        )

        assert doc.is_complete()

    def test_is_complete_false(self):
        """Test incomplete document detection."""
        doc = PendingDocument(
            path="/test/book.pdf",
            detected_metadata={'title': 'Test Book'},
            required_metadata={
                'title': 'Test Book',
                'author': '?',
                'edition': '?'
            },
            document_type='book'
        )

        assert not doc.is_complete()


class TestYamlMetadataFile:
    """Test YAML file serialization and deserialization."""

    def test_to_yaml_dict(self):
        """Test YAML serialization."""
        pending_doc = PendingDocument(
            path="/test/book.pdf",
            detected_metadata={'title': 'Test Book'},
            required_metadata={'title': 'Test Book', 'author': '?', 'edition': '?'},
            document_type='book',
            confidence=0.8
        )

        yaml_file = YamlMetadataFile(
            generated_at="2023-01-01T00:00:00",
            engine_version="1.0.0",
            library_collection="_test_library",
            pending_files=[pending_doc]
        )

        yaml_dict = yaml_file.to_yaml_dict()

        assert yaml_dict['metadata']['library_collection'] == "_test_library"
        assert len(yaml_dict['pending_files']) == 1
        assert yaml_dict['pending_files'][0]['path'] == "/test/book.pdf"
        assert yaml_dict['pending_files'][0]['document_type'] == "book"

    def test_from_yaml_dict(self):
        """Test YAML deserialization."""
        yaml_data = {
            'metadata': {
                'generated_at': '2023-01-01T00:00:00',
                'engine_version': '1.0.0',
                'library_collection': '_test_library'
            },
            'pending_files': [{
                'path': '/test/book.pdf',
                'document_type': 'book',
                'confidence': 0.8,
                'detected_metadata': {'title': 'Test Book'},
                'required_metadata': {'title': 'Test Book', 'author': '?', 'edition': '?'},
                'extraction_errors': []
            }],
            'completed_files': []
        }

        yaml_file = YamlMetadataFile.from_yaml_dict(yaml_data)

        assert yaml_file.library_collection == "_test_library"
        assert len(yaml_file.pending_files) == 1
        assert yaml_file.pending_files[0].path == "/test/book.pdf"
        assert yaml_file.pending_files[0].document_type == "book"


class TestYamlMetadataWorkflow:
    """Test the main workflow functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_client = Mock()
        self.workflow = YamlMetadataWorkflow(self.mock_client)

    @pytest.mark.asyncio
    async def test_find_documents(self):
        """Test document discovery."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test files
            (temp_path / "book1.pdf").touch()
            (temp_path / "article.pdf").touch()
            (temp_path / "notes.txt").touch()
            (temp_path / "readme.md").touch()
            (temp_path / "ignored.doc").touch()  # Unsupported format

            # Create subdirectory
            sub_dir = temp_path / "subdir"
            sub_dir.mkdir()
            (sub_dir / "book2.pdf").touch()

            documents = await self.workflow._find_documents(
                temp_path, formats=['pdf', 'txt', 'md']
            )

            # Should find pdf, txt, md files recursively
            assert len(documents) == 4
            assert any('book1.pdf' in str(doc) for doc in documents)
            assert any('article.pdf' in str(doc) for doc in documents)
            assert any('notes.txt' in str(doc) for doc in documents)
            assert any('readme.md' in str(doc) for doc in documents)
            assert any('book2.pdf' in str(doc) for doc in documents)

    @pytest.mark.asyncio
    async def test_save_and_load_yaml_file(self):
        """Test YAML file I/O operations."""
        pending_doc = PendingDocument(
            path="/test/book.pdf",
            detected_metadata={'title': 'Test Book'},
            required_metadata={'title': 'Test Book', 'author': '?', 'edition': '?'},
            document_type='book',
            confidence=0.8
        )

        yaml_file = YamlMetadataFile(
            generated_at="2023-01-01T00:00:00",
            engine_version="1.0.0",
            library_collection="_test_library",
            pending_files=[pending_doc]
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Save
            await self.workflow._save_yaml_file(yaml_file, temp_path)
            assert temp_path.exists()

            # Load
            loaded_file = await self.workflow._load_yaml_file(temp_path)

            assert loaded_file.library_collection == "_test_library"
            assert len(loaded_file.pending_files) == 1
            assert loaded_file.pending_files[0].path == "/test/book.pdf"

        finally:
            temp_path.unlink()


@pytest.mark.integration
class TestYamlMetadataIntegration:
    """Integration tests for the complete YAML metadata workflow."""

    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """Test complete workflow from generation to processing."""
        # This would require actual files and a test database
        # For now, this is a placeholder for integration tests
        pass


if __name__ == "__main__":
    pytest.main([__file__])
