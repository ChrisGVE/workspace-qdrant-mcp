"""
YAML generation system for document metadata.

This module provides comprehensive YAML generation capabilities for document
metadata, supporting structured output with proper serialization handling,
custom formatting, and error recovery for complex data structures.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from .aggregator import DocumentMetadata
from .exceptions import YAMLGenerationError


class YAMLConfig:
    """Configuration for YAML generation."""

    def __init__(
        self,
        include_content: bool = True,
        include_collection_metadata: bool = True,
        pretty_format: bool = True,
        max_content_length: int | None = None,
        indent_size: int = 2,
        sort_keys: bool = True,
        safe_serialization: bool = True,
        line_width: int = 120,
        date_format: str = "%Y-%m-%d %H:%M:%S",
    ):
        """
        Initialize YAML configuration.

        Args:
            include_content: Whether to include document content in YAML
            include_collection_metadata: Whether to include collection metadata
            pretty_format: Whether to use pretty formatting with comments
            max_content_length: Maximum content length (None for no limit)
            indent_size: YAML indentation size
            sort_keys: Whether to sort dictionary keys
            safe_serialization: Use safe YAML serialization
            line_width: Maximum line width for YAML output
            date_format: Date format string for timestamps
        """
        self.include_content = include_content
        self.include_collection_metadata = include_collection_metadata
        self.pretty_format = pretty_format
        self.max_content_length = max_content_length
        self.indent_size = indent_size
        self.sort_keys = sort_keys
        self.safe_serialization = safe_serialization
        self.line_width = line_width
        self.date_format = date_format


class YAMLGenerator:
    """
    Generates structured YAML files from document metadata.

    This class provides comprehensive YAML generation capabilities with
    proper error handling, custom formatting, and support for complex
    data structures from document metadata aggregation.
    """

    def __init__(self, config: YAMLConfig | None = None):
        """
        Initialize YAML generator.

        Args:
            config: Optional YAML configuration
        """
        self.config = config or YAMLConfig()
        self._setup_yaml_dumper()

    def _setup_yaml_dumper(self) -> None:
        """Configure YAML dumper with custom settings."""
        # Create custom dumper class
        class CustomDumper(yaml.SafeDumper):
            pass

        # Custom representer for None values
        def represent_none(dumper, value):
            return dumper.represent_scalar("tag:yaml.org,2002:null", "null")

        # Custom representer for datetime objects
        def represent_datetime(dumper, value):
            return dumper.represent_scalar(
                "tag:yaml.org,2002:timestamp",
                value.strftime(self.config.date_format)
            )

        # Register custom representers
        CustomDumper.add_representer(type(None), represent_none)
        CustomDumper.add_representer(datetime, represent_datetime)

        # Configure dumper settings
        self._dumper_class = CustomDumper
        self._dump_kwargs = {
            "Dumper": CustomDumper,
            "default_flow_style": False,
            "indent": self.config.indent_size,
            "width": self.config.line_width,
            "sort_keys": self.config.sort_keys,
            "allow_unicode": True,
            # Remove encoding to get string output instead of bytes
        }

    def generate_yaml(
        self,
        document_metadata: DocumentMetadata,
        output_path: str | Path | None = None,
    ) -> str:
        """
        Generate YAML from document metadata.

        Args:
            document_metadata: Document metadata to serialize
            output_path: Optional path to write YAML file

        Returns:
            Generated YAML string

        Raises:
            YAMLGenerationError: If YAML generation fails
        """
        try:
            # Convert metadata to dictionary
            metadata_dict = self._prepare_metadata_dict(document_metadata)

            # Generate YAML string
            yaml_content = self._serialize_to_yaml(metadata_dict)

            # Add header comments if pretty format enabled
            if self.config.pretty_format:
                yaml_content = self._add_yaml_header(yaml_content, document_metadata)

            # Write to file if path provided
            if output_path:
                self._write_yaml_file(yaml_content, output_path)

            logger.debug(f"Generated YAML for document: {document_metadata.file_path}")
            return yaml_content

        except Exception as e:
            raise YAMLGenerationError(
                f"Failed to generate YAML for document: {document_metadata.file_path}",
                details={"original_error": str(e)},
            ) from e

    def generate_collection_yaml(
        self,
        document_metadata_list: list[DocumentMetadata],
        output_path: str | Path | None = None,
        collection_name: str = "document_collection",
    ) -> str:
        """
        Generate YAML for a collection of documents.

        Args:
            document_metadata_list: List of document metadata
            output_path: Optional path to write YAML file
            collection_name: Name of the collection

        Returns:
            Generated collection YAML string

        Raises:
            YAMLGenerationError: If collection YAML generation fails
        """
        try:
            # Prepare collection structure
            collection_dict = {
                "collection": {
                    "name": collection_name,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "document_count": len(document_metadata_list),
                    "generator_config": self._get_config_dict(),
                    "documents": []
                }
            }

            # Add documents to collection
            serialization_errors = []
            for doc_metadata in document_metadata_list:
                try:
                    doc_dict = self._prepare_metadata_dict(doc_metadata)
                    collection_dict["collection"]["documents"].append(doc_dict)
                except Exception as e:
                    serialization_errors.append(
                        f"Failed to serialize {doc_metadata.file_path}: {str(e)}"
                    )
                    logger.warning(f"Skipping document due to serialization error: {e}")

            # Check if any documents were successfully processed
            if not collection_dict["collection"]["documents"] and serialization_errors:
                raise YAMLGenerationError(
                    "Failed to serialize any documents in collection",
                    serialization_errors=serialization_errors,
                )

            # Generate YAML string
            yaml_content = self._serialize_to_yaml(collection_dict)

            # Add collection header if pretty format enabled
            if self.config.pretty_format:
                yaml_content = self._add_collection_header(
                    yaml_content, collection_name, len(document_metadata_list)
                )

            # Write to file if path provided
            if output_path:
                self._write_yaml_file(yaml_content, output_path)

            if serialization_errors:
                logger.warning(
                    f"Generated collection YAML with {len(serialization_errors)} errors"
                )

            return yaml_content

        except YAMLGenerationError:
            raise
        except Exception as e:
            raise YAMLGenerationError(
                f"Failed to generate collection YAML: {collection_name}",
                details={"original_error": str(e)},
            ) from e

    def generate_batch_yaml_files(
        self,
        document_metadata_list: list[DocumentMetadata],
        output_directory: str | Path,
        filename_template: str = "{file_stem}_metadata.yaml",
    ) -> list[str]:
        """
        Generate individual YAML files for each document.

        Args:
            document_metadata_list: List of document metadata
            output_directory: Directory to write YAML files
            filename_template: Template for YAML filenames

        Returns:
            List of generated YAML file paths

        Raises:
            YAMLGenerationError: If batch generation fails
        """
        try:
            output_dir = Path(output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)

            generated_files = []
            errors = []

            for doc_metadata in document_metadata_list:
                try:
                    # Generate filename from template
                    file_path = Path(doc_metadata.file_path)
                    filename = filename_template.format(
                        file_stem=file_path.stem,
                        file_name=file_path.name,
                        file_type=doc_metadata.parsed_document.file_type,
                    )

                    output_path = output_dir / filename
                    self.generate_yaml(doc_metadata, output_path)
                    generated_files.append(str(output_path))

                except Exception as e:
                    error_msg = f"Failed to generate YAML for {doc_metadata.file_path}: {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)

            if errors and not generated_files:
                raise YAMLGenerationError(
                    "Failed to generate any YAML files in batch",
                    serialization_errors=errors,
                )

            if errors:
                logger.warning(
                    f"Batch YAML generation completed with {len(errors)} errors"
                )

            return generated_files

        except YAMLGenerationError:
            raise
        except Exception as e:
            raise YAMLGenerationError(
                f"Failed to generate batch YAML files in {output_directory}",
                details={"original_error": str(e)},
            ) from e

    def _prepare_metadata_dict(self, document_metadata: DocumentMetadata) -> dict[str, Any]:
        """
        Prepare metadata dictionary for YAML serialization.

        Args:
            document_metadata: Document metadata to prepare

        Returns:
            Dictionary ready for YAML serialization
        """
        # Get base metadata dictionary
        metadata_dict = document_metadata.to_dict(
            include_content=self.config.include_content
        )

        # Remove collection metadata if not requested
        if not self.config.include_collection_metadata:
            metadata_dict.pop("collection_metadata", None)

        # Truncate content if max length specified
        if (
            self.config.max_content_length
            and "content" in metadata_dict
            and metadata_dict["content"]
        ):
            content = metadata_dict["content"]
            if len(content) > self.config.max_content_length:
                metadata_dict["content"] = (
                    content[: self.config.max_content_length]
                    + f"... [truncated from {len(content)} characters]"
                )
                metadata_dict["content_truncated"] = True

        # Ensure all values are serializable
        return self._ensure_serializable(metadata_dict)

    def _ensure_serializable(self, obj: Any) -> Any:
        """
        Ensure object is YAML serializable.

        Args:
            obj: Object to make serializable

        Returns:
            Serializable version of object
        """
        if obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, dict):
            return {str(k): self._ensure_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._ensure_serializable(item) for item in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, "__dict__"):
            # Handle custom objects by converting to dict
            return self._ensure_serializable(obj.__dict__)
        else:
            # Convert to string as fallback
            return str(obj)

    def _serialize_to_yaml(self, data: dict[str, Any]) -> str:
        """
        Serialize data to YAML string.

        Args:
            data: Data to serialize

        Returns:
            YAML string

        Raises:
            YAMLGenerationError: If serialization fails
        """
        try:
            if self.config.safe_serialization:
                return yaml.dump(data, **self._dump_kwargs)
            else:
                return yaml.dump(data, Dumper=yaml.Dumper, **self._dump_kwargs)

        except yaml.YAMLError as e:
            raise YAMLGenerationError(
                "YAML serialization failed",
                serialization_errors=[str(e)],
            ) from e

    def _add_yaml_header(self, yaml_content: str, document_metadata: DocumentMetadata) -> str:
        """
        Add header comments to YAML content.

        Args:
            yaml_content: YAML content string
            document_metadata: Document metadata for header info

        Returns:
            YAML content with header comments
        """
        header_lines = [
            "# Document Metadata YAML",
            f"# Generated on: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"# Source file: {document_metadata.file_path}",
            f"# File type: {document_metadata.parsed_document.file_type}",
            f"# Content hash: {document_metadata.content_hash}",
            "",
        ]

        return "\n".join(header_lines) + yaml_content

    def _add_collection_header(
        self, yaml_content: str, collection_name: str, document_count: int
    ) -> str:
        """
        Add header comments to collection YAML content.

        Args:
            yaml_content: YAML content string
            collection_name: Name of the collection
            document_count: Number of documents in collection

        Returns:
            Collection YAML content with header comments
        """
        header_lines = [
            "# Document Collection Metadata YAML",
            f"# Generated on: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"# Collection: {collection_name}",
            f"# Document count: {document_count}",
            f"# Content included: {self.config.include_content}",
            "",
        ]

        return "\n".join(header_lines) + yaml_content

    def _write_yaml_file(self, yaml_content: str, output_path: str | Path) -> None:
        """
        Write YAML content to file.

        Args:
            yaml_content: YAML content to write
            output_path: Path to output file

        Raises:
            YAMLGenerationError: If file writing fails
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(yaml_content)

            logger.debug(f"Wrote YAML file: {output_path}")

        except Exception as e:
            raise YAMLGenerationError(
                f"Failed to write YAML file: {output_path}",
                details={"original_error": str(e)},
            ) from e

    def _get_config_dict(self) -> dict[str, Any]:
        """
        Get configuration as dictionary for metadata.

        Returns:
            Configuration dictionary
        """
        return {
            "include_content": self.config.include_content,
            "include_collection_metadata": self.config.include_collection_metadata,
            "pretty_format": self.config.pretty_format,
            "max_content_length": self.config.max_content_length,
            "safe_serialization": self.config.safe_serialization,
        }

    def validate_yaml_output(self, yaml_content: str) -> dict[str, Any]:
        """
        Validate generated YAML content.

        Args:
            yaml_content: YAML content to validate

        Returns:
            Validation results dictionary
        """
        try:
            # Try to parse the YAML back
            parsed_data = yaml.safe_load(yaml_content)

            # Check structure
            has_content = "content" in str(yaml_content).lower()
            has_metadata = "metadata" in str(yaml_content).lower()
            line_count = len(yaml_content.splitlines())

            return {
                "valid": True,
                "parseable": True,
                "has_content": has_content,
                "has_metadata": has_metadata,
                "line_count": line_count,
                "size_bytes": len(yaml_content.encode("utf-8")),
                "structure_valid": isinstance(parsed_data, dict),
            }

        except yaml.YAMLError as e:
            return {
                "valid": False,
                "parseable": False,
                "error": str(e),
                "line_count": len(yaml_content.splitlines()),
                "size_bytes": len(yaml_content.encode("utf-8")),
            }
        except Exception as e:
            return {
                "valid": False,
                "parseable": False,
                "error": f"Validation error: {str(e)}",
            }


# Export main classes
__all__ = ["YAMLGenerator", "YAMLConfig"]
