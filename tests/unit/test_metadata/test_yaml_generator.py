"""
Unit tests for YAMLGenerator.

This module tests YAML generation functionality including serialization,
error handling, and various output formats for document metadata.
"""

import pytest
import tempfile
import yaml
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from src.python.wqm_cli.cli.metadata.yaml_generator import (
    YAMLGenerator,
    YAMLConfig,
)
from src.python.wqm_cli.cli.metadata.aggregator import DocumentMetadata
from src.python.wqm_cli.cli.metadata.exceptions import YAMLGenerationError
from src.python.wqm_cli.cli.parsers.base import ParsedDocument
from src.python.common.core.metadata_schema import MultiTenantMetadataSchema


class TestYAMLConfig:
    """Test cases for YAMLConfig class."""

    def test_init_with_defaults(self):
        """Test YAMLConfig initialization with default values."""
        config = YAMLConfig()

        assert config.include_content is True
        assert config.include_collection_metadata is True
        assert config.pretty_format is True
        assert config.max_content_length is None
        assert config.indent_size == 2
        assert config.sort_keys is True
        assert config.safe_serialization is True
        assert config.line_width == 120

    def test_init_with_custom_values(self):
        """Test YAMLConfig initialization with custom values."""
        config = YAMLConfig(
            include_content=False,
            max_content_length=1000,
            indent_size=4,
            sort_keys=False,
            line_width=80,
        )

        assert config.include_content is False
        assert config.max_content_length == 1000
        assert config.indent_size == 4
        assert config.sort_keys is False
        assert config.line_width == 80


class TestYAMLGenerator:
    """Test cases for YAMLGenerator class."""

    def test_init_with_default_config(self):
        """Test YAMLGenerator initialization with default config."""
        generator = YAMLGenerator()

        assert generator.config is not None
        assert generator.config.include_content is True
        assert generator._dumper_class is not None
        assert generator._dump_kwargs is not None

    def test_init_with_custom_config(self):
        """Test YAMLGenerator initialization with custom config."""
        config = YAMLConfig(include_content=False, indent_size=4)
        generator = YAMLGenerator(config)

        assert generator.config == config
        assert generator.config.include_content is False
        assert generator.config.indent_size == 4

    def test_generate_yaml_basic(self, sample_document_metadata):
        """Test basic YAML generation."""
        generator = YAMLGenerator()

        yaml_content = generator.generate_yaml(sample_document_metadata)

        assert isinstance(yaml_content, str)
        assert "file_path" in yaml_content
        assert "content" in yaml_content
        assert "metadata" in yaml_content

        # Verify it's valid YAML
        parsed = yaml.safe_load(yaml_content)
        assert isinstance(parsed, dict)

    def test_generate_yaml_without_content(self, sample_document_metadata):
        """Test YAML generation without content."""
        config = YAMLConfig(include_content=False)
        generator = YAMLGenerator(config)

        yaml_content = generator.generate_yaml(sample_document_metadata)

        assert "content:" not in yaml_content  # Content should not be included
        assert "file_path" in yaml_content
        assert "metadata" in yaml_content

    def test_generate_yaml_with_content_truncation(self, sample_document_metadata):
        """Test YAML generation with content truncation."""
        config = YAMLConfig(max_content_length=10)
        generator = YAMLGenerator(config)

        yaml_content = generator.generate_yaml(sample_document_metadata)

        parsed = yaml.safe_load(yaml_content)
        content = parsed.get("content", "")

        if len(content) > 10:
            assert "truncated" in content
            assert parsed.get("content_truncated") is True

    def test_generate_yaml_with_pretty_format(self, sample_document_metadata):
        """Test YAML generation with pretty formatting."""
        config = YAMLConfig(pretty_format=True)
        generator = YAMLGenerator(config)

        yaml_content = generator.generate_yaml(sample_document_metadata)

        # Pretty format should include header comments
        lines = yaml_content.split("\n")
        assert any(line.startswith("#") for line in lines)
        assert "Document Metadata YAML" in yaml_content

    def test_generate_yaml_without_pretty_format(self, sample_document_metadata):
        """Test YAML generation without pretty formatting."""
        config = YAMLConfig(pretty_format=False)
        generator = YAMLGenerator(config)

        yaml_content = generator.generate_yaml(sample_document_metadata)

        # Should not include header comments
        assert not yaml_content.startswith("#")
        assert "Document Metadata YAML" not in yaml_content

    def test_generate_yaml_with_file_output(self, sample_document_metadata):
        """Test YAML generation with file output."""
        generator = YAMLGenerator()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            yaml_content = generator.generate_yaml(
                sample_document_metadata,
                output_path=temp_path
            )

            # File should be created and contain the YAML
            assert temp_path.exists()
            file_content = temp_path.read_text(encoding="utf-8")
            assert file_content == yaml_content

            # Verify content
            parsed = yaml.safe_load(file_content)
            assert isinstance(parsed, dict)

        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_generate_collection_yaml(self, sample_document_metadata_list):
        """Test collection YAML generation."""
        generator = YAMLGenerator()

        yaml_content = generator.generate_collection_yaml(
            sample_document_metadata_list,
            collection_name="test_collection"
        )

        parsed = yaml.safe_load(yaml_content)

        assert "collection" in parsed
        collection = parsed["collection"]
        assert collection["name"] == "test_collection"
        assert collection["document_count"] == len(sample_document_metadata_list)
        assert "documents" in collection
        assert len(collection["documents"]) == len(sample_document_metadata_list)

    def test_generate_collection_yaml_empty_list(self):
        """Test collection YAML generation with empty document list."""
        generator = YAMLGenerator()

        yaml_content = generator.generate_collection_yaml(
            [],
            collection_name="empty_collection"
        )

        parsed = yaml.safe_load(yaml_content)
        collection = parsed["collection"]
        assert collection["document_count"] == 0
        assert collection["documents"] == []

    def test_generate_collection_yaml_with_file_output(self, sample_document_metadata_list):
        """Test collection YAML generation with file output."""
        generator = YAMLGenerator()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            yaml_content = generator.generate_collection_yaml(
                sample_document_metadata_list,
                output_path=temp_path,
                collection_name="file_collection"
            )

            assert temp_path.exists()
            file_content = temp_path.read_text(encoding="utf-8")
            assert file_content == yaml_content

        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_generate_batch_yaml_files(self, sample_document_metadata_list):
        """Test batch YAML file generation."""
        generator = YAMLGenerator()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            generated_files = generator.generate_batch_yaml_files(
                sample_document_metadata_list,
                output_directory=output_dir
            )

            assert len(generated_files) == len(sample_document_metadata_list)

            # Check each generated file
            for file_path in generated_files:
                path_obj = Path(file_path)
                assert path_obj.exists()
                assert path_obj.suffix == ".yaml"

                # Verify content
                content = path_obj.read_text(encoding="utf-8")
                parsed = yaml.safe_load(content)
                assert isinstance(parsed, dict)
                assert "file_path" in parsed

    def test_generate_batch_yaml_files_custom_template(self, sample_document_metadata_list):
        """Test batch YAML file generation with custom filename template."""
        generator = YAMLGenerator()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            generated_files = generator.generate_batch_yaml_files(
                sample_document_metadata_list,
                output_directory=output_dir,
                filename_template="custom_{file_type}_{file_stem}.yaml"
            )

            # Check filename format
            for file_path in generated_files:
                filename = Path(file_path).name
                assert filename.startswith("custom_")
                assert filename.endswith(".yaml")

    def test_serialization_error_handling(self):
        """Test handling of YAML serialization errors."""
        generator = YAMLGenerator()

        # Create document metadata with non-serializable object
        class NonSerializable:
            def __str__(self):
                raise Exception("Cannot convert to string")

        parsed_doc = ParsedDocument.create(
            content="test content",
            file_path="/test/document.txt",
            file_type="text",
            additional_metadata={"bad_object": NonSerializable()},
        )

        doc_metadata = DocumentMetadata(
            file_path="/test/document.txt",
            content_hash="abc123",
            parsed_document=parsed_doc,
        )

        # Should handle the error gracefully by converting to string
        yaml_content = generator.generate_yaml(doc_metadata)
        assert isinstance(yaml_content, str)

    def test_unicode_handling(self):
        """Test Unicode content handling in YAML generation."""
        generator = YAMLGenerator()

        parsed_doc = ParsedDocument.create(
            content="Tëst cöntënt wîth ümläuts and émöjïs 🎉",
            file_path="/test/ünïcödë_document.txt",
            file_type="text",
            additional_metadata={"title": "Ünïcödë Tëst"},
        )

        doc_metadata = DocumentMetadata(
            file_path="/test/ünïcödë_document.txt",
            content_hash="abc123",
            parsed_document=parsed_doc,
        )

        yaml_content = generator.generate_yaml(doc_metadata)

        # Should handle Unicode without errors
        assert isinstance(yaml_content, str)
        parsed = yaml.safe_load(yaml_content)
        assert "Tëst cöntënt" in parsed["content"]
        assert parsed["metadata"]["title"] == "Ünïcödë Tëst"

    def test_datetime_serialization(self):
        """Test datetime object serialization."""
        config = YAMLConfig(date_format="%Y-%m-%d %H:%M:%S")
        generator = YAMLGenerator(config)

        test_datetime = datetime(2023, 1, 15, 14, 30, 0, tzinfo=timezone.utc)

        parsed_doc = ParsedDocument.create(
            content="test content",
            file_path="/test/document.txt",
            file_type="text",
            additional_metadata={"creation_date": test_datetime},
        )

        doc_metadata = DocumentMetadata(
            file_path="/test/document.txt",
            content_hash="abc123",
            parsed_document=parsed_doc,
        )

        yaml_content = generator.generate_yaml(doc_metadata)
        parsed = yaml.safe_load(yaml_content)

        # Date should be formatted according to config
        assert "2023-01-15 14:30:00" in parsed["metadata"]["creation_date"]

    def test_none_value_serialization(self):
        """Test None value serialization."""
        generator = YAMLGenerator()

        parsed_doc = ParsedDocument.create(
            content="test content",
            file_path="/test/document.txt",
            file_type="text",
            additional_metadata={"nullable_field": None},
        )

        doc_metadata = DocumentMetadata(
            file_path="/test/document.txt",
            content_hash="abc123",
            parsed_document=parsed_doc,
        )

        yaml_content = generator.generate_yaml(doc_metadata)

        # Should handle None values
        assert "null" in yaml_content
        parsed = yaml.safe_load(yaml_content)
        assert parsed["metadata"]["nullable_field"] is None

    def test_validate_yaml_output_valid(self, sample_document_metadata):
        """Test YAML output validation with valid content."""
        generator = YAMLGenerator()

        yaml_content = generator.generate_yaml(sample_document_metadata)
        validation_result = generator.validate_yaml_output(yaml_content)

        assert validation_result["valid"] is True
        assert validation_result["parseable"] is True
        assert validation_result["has_content"] is True
        assert validation_result["has_metadata"] is True
        assert validation_result["line_count"] > 0
        assert validation_result["size_bytes"] > 0
        assert validation_result["structure_valid"] is True

    def test_validate_yaml_output_invalid(self):
        """Test YAML output validation with invalid content."""
        generator = YAMLGenerator()

        # Invalid YAML content
        invalid_yaml = "invalid: yaml: content: ["

        validation_result = generator.validate_yaml_output(invalid_yaml)

        assert validation_result["valid"] is False
        assert validation_result["parseable"] is False
        assert "error" in validation_result

    def test_file_write_error(self, sample_document_metadata):
        """Test file writing error handling."""
        generator = YAMLGenerator()

        # Try to write to invalid path
        invalid_path = "/invalid/nonexistent/directory/file.yaml"

        with pytest.raises(YAMLGenerationError) as exc_info:
            generator.generate_yaml(sample_document_metadata, output_path=invalid_path)

        assert "Failed to write YAML file" in str(exc_info.value)

    def test_collection_yaml_with_serialization_errors(self):
        """Test collection YAML generation with some serialization errors."""
        generator = YAMLGenerator()

        # Create mix of valid and problematic documents
        documents = []

        # Valid document
        valid_doc = ParsedDocument.create(
            content="valid content",
            file_path="/test/valid.txt",
            file_type="text",
        )
        valid_metadata = DocumentMetadata(
            file_path="/test/valid.txt",
            content_hash="abc123",
            parsed_document=valid_doc,
        )
        documents.append(valid_metadata)

        # Document with serialization issues (but should still work)
        class ProblematicObject:
            def __str__(self):
                return "problematic_value"

        problematic_doc = ParsedDocument.create(
            content="content with issues",
            file_path="/test/problematic.txt",
            file_type="text",
            additional_metadata={"problematic": ProblematicObject()},
        )
        problematic_metadata = DocumentMetadata(
            file_path="/test/problematic.txt",
            content_hash="def456",
            parsed_document=problematic_doc,
        )
        documents.append(problematic_metadata)

        # Should still generate YAML, possibly with warnings
        yaml_content = generator.generate_collection_yaml(
            documents, collection_name="mixed_collection"
        )

        parsed = yaml.safe_load(yaml_content)
        collection = parsed["collection"]

        # Should have at least the valid document
        assert collection["document_count"] >= 1
        assert len(collection["documents"]) >= 1

    def test_edge_case_empty_content(self):
        """Test handling of document with empty content."""
        generator = YAMLGenerator()

        parsed_doc = ParsedDocument.create(
            content="",  # Empty content
            file_path="/test/empty.txt",
            file_type="text",
        )

        doc_metadata = DocumentMetadata(
            file_path="/test/empty.txt",
            content_hash="empty_hash",
            parsed_document=parsed_doc,
        )

        yaml_content = generator.generate_yaml(doc_metadata)

        # Should handle empty content without errors
        parsed = yaml.safe_load(yaml_content)
        assert parsed["content"] == ""

    def test_edge_case_very_large_content(self):
        """Test handling of very large content."""
        config = YAMLConfig(max_content_length=100)
        generator = YAMLGenerator(config)

        large_content = "x" * 1000  # 1000 characters

        parsed_doc = ParsedDocument.create(
            content=large_content,
            file_path="/test/large.txt",
            file_type="text",
        )

        doc_metadata = DocumentMetadata(
            file_path="/test/large.txt",
            content_hash="large_hash",
            parsed_document=parsed_doc,
        )

        yaml_content = generator.generate_yaml(doc_metadata)

        parsed = yaml.safe_load(yaml_content)
        content = parsed["content"]

        # Content should be truncated
        assert len(content) <= 200  # 100 + truncation message
        assert "truncated" in content
        assert parsed["content_truncated"] is True

    def test_config_dict_generation(self):
        """Test configuration dictionary generation."""
        config = YAMLConfig(
            include_content=False,
            max_content_length=500,
            safe_serialization=False
        )
        generator = YAMLGenerator(config)

        config_dict = generator._get_config_dict()

        assert config_dict["include_content"] is False
        assert config_dict["max_content_length"] == 500
        assert config_dict["safe_serialization"] is False


# Test fixtures and helpers

@pytest.fixture
def sample_document_metadata():
    """Create a sample DocumentMetadata for testing."""
    parsed_doc = ParsedDocument.create(
        content="Sample test content for YAML generation testing.",
        file_path="/test/sample_document.txt",
        file_type="text",
        additional_metadata={
            "encoding": "utf-8",
            "line_count": 1,
            "word_count": 8,
        },
    )

    return DocumentMetadata(
        file_path="/test/sample_document.txt",
        content_hash="abc123def456",
        parsed_document=parsed_doc,
    )


@pytest.fixture
def sample_document_metadata_with_collection():
    """Create sample DocumentMetadata with collection metadata."""
    parsed_doc = ParsedDocument.create(
        content="Sample content with collection metadata.",
        file_path="/test/collection_document.txt",
        file_type="text",
    )

    collection_metadata = MultiTenantMetadataSchema.create_for_project(
        project_name="test_project",
        collection_type="docs",
    )

    return DocumentMetadata(
        file_path="/test/collection_document.txt",
        content_hash="collection123",
        parsed_document=parsed_doc,
        collection_metadata=collection_metadata,
    )


@pytest.fixture
def sample_document_metadata_list():
    """Create a list of sample DocumentMetadata for testing."""
    documents = []

    for i in range(3):
        parsed_doc = ParsedDocument.create(
            content=f"Sample test content {i} for YAML generation testing.",
            file_path=f"/test/sample_document_{i}.txt",
            file_type="text",
            additional_metadata={
                "document_index": i,
                "encoding": "utf-8",
            },
        )

        doc_metadata = DocumentMetadata(
            file_path=f"/test/sample_document_{i}.txt",
            content_hash=f"hash{i}",
            parsed_document=parsed_doc,
        )
        documents.append(doc_metadata)

    return documents