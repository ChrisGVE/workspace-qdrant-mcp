"""
Metadata aggregation system for document processors.

This module provides comprehensive metadata extraction and aggregation
capabilities, collecting metadata from all document parser types and
normalizing it into a unified format for YAML generation and storage.
"""

from datetime import datetime, timezone
from typing import Any

from loguru import logger

from ....common.core.metadata_schema import (
    MultiTenantMetadataSchema,
)
from ..parsers.base import ParsedDocument
from .exceptions import AggregationError


class DocumentMetadata:
    """
    Unified metadata container for processed documents.

    This class represents aggregated metadata from various parser types,
    normalized into a consistent format for YAML generation and storage.
    """

    def __init__(
        self,
        file_path: str,
        content_hash: str,
        parsed_document: ParsedDocument,
        collection_metadata: MultiTenantMetadataSchema | None = None,
    ):
        """
        Initialize document metadata.

        Args:
            file_path: Path to the original document
            content_hash: SHA256 hash of document content
            parsed_document: Parsed document with content and metadata
            collection_metadata: Collection-level metadata schema
        """
        self.file_path = file_path
        self.content_hash = content_hash
        self.parsed_document = parsed_document
        self.collection_metadata = collection_metadata
        self.aggregated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self, include_content: bool = True) -> dict[str, Any]:
        """
        Convert metadata to dictionary format.

        Args:
            include_content: Whether to include document content

        Returns:
            Dictionary representation of metadata
        """
        result = {
            # Core document information
            "file_path": self.file_path,
            "file_type": self.parsed_document.file_type,
            "content_hash": self.content_hash,
            "file_size": self.parsed_document.file_size,
            "parsed_at": self.parsed_document.parsed_at,
            "aggregated_at": self.aggregated_at,

            # Document metadata
            "metadata": self.parsed_document.metadata.copy(),

            # Parsing information
            "parsing_info": self.parsed_document.parsing_info or {},
        }

        if include_content:
            result["content"] = self.parsed_document.content

        # Add collection metadata if available
        if self.collection_metadata:
            result["collection_metadata"] = self.collection_metadata.to_qdrant_payload()

        return result


class MetadataAggregator:
    """
    Aggregates metadata from all document parser types.

    This class provides a unified interface for extracting and normalizing
    metadata from documents processed by various parsers, ensuring consistent
    metadata structure across all document types.
    """

    def __init__(self):
        """Initialize the metadata aggregator."""
        self.parser_registry: dict[str, set[str]] = {}
        self._register_known_parsers()

    def _register_known_parsers(self) -> None:
        """Register known parser types and their metadata fields."""
        # PDF parser metadata fields
        self.parser_registry["pdf"] = {
            "page_count", "title", "author", "subject", "creator", "producer",
            "creation_date", "modification_date", "has_text_layer",
            "requires_ocr", "is_encrypted", "pdf_version"
        }

        # EPUB parser metadata fields
        self.parser_registry["epub"] = {
            "title", "author", "publisher", "publication_date", "language",
            "isbn", "description", "chapter_count", "image_count",
            "epub_version", "spine_items", "manifest_items"
        }

        # MOBI parser metadata fields
        self.parser_registry["mobi"] = {
            "title", "author", "publisher", "publication_date", "language",
            "asin", "description", "drm_protected", "mobi_version",
            "compression_type", "text_encoding"
        }

        # Web parser metadata fields
        self.parser_registry["web"] = {
            "url", "domain", "title", "meta_description", "meta_keywords",
            "author", "publication_date", "last_modified", "language",
            "canonical_url", "robots_meta", "content_type", "status_code",
            "crawl_depth", "link_count", "image_count", "security_scan_result"
        }

        # Code parser metadata fields
        self.parser_registry["code"] = {
            "language", "line_count", "function_count", "class_count",
            "import_count", "comment_lines", "blank_lines", "complexity_score",
            "lsp_analysis", "syntax_errors", "style_violations"
        }

        # HTML parser metadata fields
        self.parser_registry["html"] = {
            "title", "meta_description", "meta_keywords", "author",
            "language", "charset", "doctype", "head_tag_count",
            "body_tag_count", "link_count", "image_count", "form_count"
        }

        # DOCX parser metadata fields
        self.parser_registry["docx"] = {
            "title", "author", "subject", "creator", "last_modified_by",
            "creation_date", "modification_date", "word_count",
            "paragraph_count", "page_count", "revision", "category",
            "comments", "template", "company"
        }

        # PPTX parser metadata fields
        self.parser_registry["pptx"] = {
            "title", "author", "subject", "creator", "last_modified_by",
            "creation_date", "modification_date", "slide_count",
            "notes_count", "revision", "category", "company",
            "presentation_format", "template"
        }

        # Text parser metadata fields
        self.parser_registry["text"] = {
            "encoding", "line_endings", "has_bom", "detected_language",
            "paragraph_count", "sentence_count", "word_count",
            "character_count"
        }

        # Markdown parser metadata fields
        self.parser_registry["markdown"] = {
            "frontmatter", "heading_count", "link_count", "image_count",
            "code_block_count", "table_count", "list_count",
            "emphasis_count", "has_yaml_frontmatter", "has_toc"
        }

    def aggregate_metadata(
        self,
        parsed_document: ParsedDocument,
        project_name: str | None = None,
        collection_type: str = "documents",
    ) -> DocumentMetadata:
        """
        Aggregate metadata from a parsed document.

        Args:
            parsed_document: Parsed document with content and metadata
            project_name: Optional project name for collection metadata
            collection_type: Type of collection for metadata

        Returns:
            DocumentMetadata with aggregated information

        Raises:
            AggregationError: If metadata aggregation fails
        """
        try:
            # Validate parsed document
            self._validate_parsed_document(parsed_document)

            # Normalize parser-specific metadata
            normalized_metadata = self._normalize_parser_metadata(
                parsed_document.file_type,
                parsed_document.metadata
            )

            # Create updated parsed document with normalized metadata
            normalized_document = ParsedDocument(
                content=parsed_document.content,
                file_path=parsed_document.file_path,
                file_type=parsed_document.file_type,
                metadata=normalized_metadata,
                content_hash=parsed_document.content_hash,
                parsed_at=parsed_document.parsed_at,
                file_size=parsed_document.file_size,
                parsing_info=parsed_document.parsing_info,
            )

            # Create collection metadata if project specified
            collection_metadata = None
            if project_name:
                collection_metadata = self._create_collection_metadata(
                    project_name, collection_type
                )

            # Create aggregated metadata
            document_metadata = DocumentMetadata(
                file_path=parsed_document.file_path,
                content_hash=parsed_document.content_hash,
                parsed_document=normalized_document,
                collection_metadata=collection_metadata,
            )

            logger.debug(
                f"Aggregated metadata for {parsed_document.file_type} document: "
                f"{parsed_document.file_path}"
            )

            return document_metadata

        except Exception as e:
            raise AggregationError(
                f"Failed to aggregate metadata for document: {parsed_document.file_path}",
                parser_type=parsed_document.file_type,
                file_path=parsed_document.file_path,
                details={"original_error": str(e)},
            ) from e

    def aggregate_batch_metadata(
        self,
        parsed_documents: list[ParsedDocument],
        project_name: str | None = None,
        collection_type: str = "documents",
    ) -> list[DocumentMetadata]:
        """
        Aggregate metadata for a batch of parsed documents.

        Args:
            parsed_documents: List of parsed documents
            project_name: Optional project name for collection metadata
            collection_type: Type of collection for metadata

        Returns:
            List of DocumentMetadata objects

        Raises:
            AggregationError: If batch aggregation fails
        """
        results = []
        errors = []

        for parsed_doc in parsed_documents:
            try:
                metadata = self.aggregate_metadata(
                    parsed_doc, project_name, collection_type
                )
                results.append(metadata)
            except AggregationError as e:
                errors.append(str(e))
                logger.error(f"Failed to aggregate metadata for {parsed_doc.file_path}: {e}")

        if errors and not results:
            raise AggregationError(
                "Failed to aggregate metadata for entire batch",
                details={
                    "error_count": len(errors),
                    "errors": errors[:10],  # Limit error list
                    "total_documents": len(parsed_documents),
                }
            )

        if errors:
            logger.warning(
                f"Partial success in batch aggregation: {len(results)} succeeded, "
                f"{len(errors)} failed"
            )

        return results

    def _validate_parsed_document(self, parsed_document: ParsedDocument) -> None:
        """
        Validate parsed document for metadata aggregation.

        Args:
            parsed_document: Parsed document to validate

        Raises:
            AggregationError: If document validation fails
        """
        if not parsed_document.content:
            raise AggregationError(
                "Parsed document has no content",
                parser_type=parsed_document.file_type,
                file_path=parsed_document.file_path,
            )

        if not parsed_document.file_type:
            raise AggregationError(
                "Parsed document missing file_type",
                file_path=parsed_document.file_path,
            )

        if not parsed_document.content_hash:
            raise AggregationError(
                "Parsed document missing content_hash",
                parser_type=parsed_document.file_type,
                file_path=parsed_document.file_path,
            )

    def _normalize_parser_metadata(
        self, parser_type: str, metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Normalize parser-specific metadata fields.

        Args:
            parser_type: Type of parser (pdf, epub, web, etc.)
            metadata: Raw metadata dictionary

        Returns:
            Normalized metadata dictionary
        """
        # Create copy of metadata to avoid mutation
        normalized = metadata.copy()

        # Get known fields for this parser type
        known_fields = self.parser_registry.get(parser_type, set())

        # Add parser type identification
        normalized["parser_type"] = parser_type
        normalized["known_fields"] = list(known_fields)

        # Normalize common fields
        self._normalize_common_fields(normalized)

        # Apply parser-specific normalizations
        if parser_type == "pdf":
            self._normalize_pdf_metadata(normalized)
        elif parser_type in ["epub", "mobi"]:
            self._normalize_ebook_metadata(normalized)
        elif parser_type == "web":
            self._normalize_web_metadata(normalized)
        elif parser_type == "code":
            self._normalize_code_metadata(normalized)
        elif parser_type in ["docx", "pptx"]:
            self._normalize_office_metadata(normalized)

        return normalized

    def _normalize_common_fields(self, metadata: dict[str, Any]) -> None:
        """Normalize common metadata fields across all parsers."""
        # Standardize date fields
        for date_field in ["creation_date", "modification_date", "publication_date"]:
            if date_field in metadata:
                metadata[date_field] = self._normalize_date_field(
                    metadata[date_field]
                )

        # Standardize numeric fields
        for numeric_field in ["line_count", "word_count", "page_count"]:
            if numeric_field in metadata:
                metadata[numeric_field] = self._ensure_integer(
                    metadata[numeric_field]
                )

        # Standardize boolean fields
        for bool_field in ["is_encrypted", "has_text_layer", "drm_protected"]:
            if bool_field in metadata:
                metadata[bool_field] = bool(metadata[bool_field])

    def _normalize_pdf_metadata(self, metadata: dict[str, Any]) -> None:
        """Normalize PDF-specific metadata fields."""
        # Ensure required PDF fields have defaults
        if "has_text_layer" not in metadata:
            metadata["has_text_layer"] = True  # Assume text layer exists

        if "requires_ocr" not in metadata:
            metadata["requires_ocr"] = False

        # Normalize PDF version format
        if "pdf_version" in metadata:
            version = str(metadata["pdf_version"])
            if not version.startswith("1."):
                metadata["pdf_version"] = f"1.{version}"

    def _normalize_ebook_metadata(self, metadata: dict[str, Any]) -> None:
        """Normalize EPUB/MOBI-specific metadata fields."""
        # Ensure chapter count is available
        if "chapter_count" not in metadata:
            # Estimate from content if possible
            content_sections = metadata.get("spine_items", [])
            metadata["chapter_count"] = len(content_sections) if content_sections else 1

        # Normalize ISBN format
        if "isbn" in metadata and metadata["isbn"]:
            isbn = str(metadata["isbn"]).replace("-", "").replace(" ", "")
            metadata["isbn"] = isbn

    def _normalize_web_metadata(self, metadata: dict[str, Any]) -> None:
        """Normalize web content metadata fields."""
        # Ensure URL fields are properly formatted
        for url_field in ["url", "canonical_url"]:
            if url_field in metadata and metadata[url_field]:
                url = str(metadata[url_field])
                if not url.startswith(("http://", "https://")):
                    metadata[url_field] = f"https://{url}"

        # Set default status code if missing
        if "status_code" not in metadata:
            metadata["status_code"] = 200

    def _normalize_code_metadata(self, metadata: dict[str, Any]) -> None:
        """Normalize code parser metadata fields."""
        # Ensure complexity score is numeric
        if "complexity_score" in metadata:
            try:
                metadata["complexity_score"] = float(metadata["complexity_score"])
            except (ValueError, TypeError):
                metadata["complexity_score"] = 0.0

        # Ensure counts are integers
        for count_field in ["function_count", "class_count", "import_count"]:
            if count_field in metadata:
                metadata[count_field] = self._ensure_integer(metadata[count_field])

    def _normalize_office_metadata(self, metadata: dict[str, Any]) -> None:
        """Normalize DOCX/PPTX metadata fields."""
        # Handle revision field
        if "revision" in metadata:
            try:
                metadata["revision"] = int(metadata["revision"])
            except (ValueError, TypeError):
                metadata["revision"] = 1

        # Ensure company field is string
        if "company" in metadata and metadata["company"]:
            metadata["company"] = str(metadata["company"])

    def _normalize_date_field(self, date_value: Any) -> str | None:
        """
        Normalize date field to ISO format.

        Args:
            date_value: Date value to normalize

        Returns:
            ISO format date string or None
        """
        if not date_value:
            return None

        # If already a string in reasonable format, return as-is
        if isinstance(date_value, str):
            try:
                # Try to parse and reformat
                parsed = datetime.fromisoformat(date_value.replace("Z", "+00:00"))
                return parsed.isoformat()
            except ValueError:
                return date_value

        # If datetime object, convert to ISO
        if isinstance(date_value, datetime):
            return date_value.isoformat()

        # Otherwise return string representation
        return str(date_value)

    def _ensure_integer(self, value: Any) -> int:
        """
        Ensure value is integer.

        Args:
            value: Value to convert

        Returns:
            Integer value
        """
        try:
            return int(value)
        except (ValueError, TypeError):
            return 0

    def _create_collection_metadata(
        self, project_name: str, collection_type: str
    ) -> MultiTenantMetadataSchema:
        """
        Create collection metadata for project.

        Args:
            project_name: Name of the project
            collection_type: Type of collection

        Returns:
            MultiTenantMetadataSchema for the collection
        """
        return MultiTenantMetadataSchema.create_for_project(
            project_name=project_name,
            collection_type=collection_type,
            created_by="metadata_workflow",
            tags=["document_metadata", "automated"],
            category="document_collection",
        )

    def get_supported_parsers(self) -> list[str]:
        """
        Get list of supported parser types.

        Returns:
            List of supported parser type names
        """
        return list(self.parser_registry.keys())

    def get_parser_fields(self, parser_type: str) -> set[str]:
        """
        Get known metadata fields for a parser type.

        Args:
            parser_type: Parser type name

        Returns:
            Set of known metadata field names
        """
        return self.parser_registry.get(parser_type, set()).copy()

    def validate_metadata_completeness(
        self, document_metadata: DocumentMetadata
    ) -> dict[str, Any]:
        """
        Validate completeness of aggregated metadata.

        Args:
            document_metadata: Document metadata to validate

        Returns:
            Validation report dictionary
        """
        parser_type = document_metadata.parsed_document.file_type
        expected_fields = self.get_parser_fields(parser_type)
        actual_fields = set(document_metadata.parsed_document.metadata.keys())

        missing_fields = expected_fields - actual_fields
        extra_fields = actual_fields - expected_fields

        return {
            "parser_type": parser_type,
            "expected_fields": len(expected_fields),
            "actual_fields": len(actual_fields),
            "missing_fields": list(missing_fields),
            "extra_fields": list(extra_fields),
            "completeness_score": len(actual_fields & expected_fields) / len(expected_fields) if expected_fields else 1.0,
        }


# Export main classes
__all__ = ["MetadataAggregator", "DocumentMetadata"]
