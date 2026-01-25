"""
YAML metadata workflow for library document ingestion.

This module provides the core functionality for the YAML-based metadata completion
system, allowing users to provide missing metadata for library documents through
an intuitive YAML interface.

Features:
    - Document type detection (book, scientific_article, webpage, etc.)
    - YAML file generation with detected vs required fields
    - Iterative processing workflow (process complete, update remaining)
    - Schema validation for different document types
    - User-friendly error messages and guidance
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

# Import parsers with fallback for when wqm_cli is not available
try:
    from wqm_cli.cli.parsers import (
        DocumentParser,
        MarkdownParser,
        ParsedDocument,
        PDFParser,
        PptxParser,
        TextParser,
    )
    PARSERS_AVAILABLE = True
except ImportError:
    logger.warning("wqm_cli parsers not available - using fallback stubs")
    PARSERS_AVAILABLE = False

    # Fallback stub classes
    class ParsedDocument:
        def __init__(self, content="", content_hash="", metadata=None):
            import hashlib
            self.content = content
            self.content_hash = content_hash or hashlib.sha256(content.encode()).hexdigest()
            self.metadata = metadata or {}

    class DocumentParser:
        format_name = "Generic"
        supported_extensions: set = set()

    class TextParser(DocumentParser):
        format_name = "Text"
        supported_extensions = {".txt"}

    class MarkdownParser(DocumentParser):
        format_name = "Markdown"
        supported_extensions = {".md", ".markdown"}

    class PDFParser(DocumentParser):
        format_name = "PDF"
        supported_extensions = {".pdf"}

    class PptxParser(DocumentParser):
        format_name = "PPTX"
        supported_extensions = {".pptx", ".ppt"}

from ..core.client import QdrantWorkspaceClient

# logger imported from loguru


@dataclass
class DocumentTypeSchema:
    """Schema definition for a specific document type."""

    name: str
    primary_version: str
    required_metadata: list[str]
    optional_metadata: list[str] = field(default_factory=list)

    def validate_metadata(self, metadata: dict[str, Any]) -> dict[str, list[str]]:
        """Validate metadata against schema requirements.

        Returns:
            Dictionary with 'missing_required' and 'unknown_fields' lists
        """
        missing_required = []
        for field in self.required_metadata:
            if field not in metadata or metadata[field] in (None, "", "?"):
                missing_required.append(field)

        all_valid_fields = set(self.required_metadata + self.optional_metadata)
        unknown_fields = [
            field
            for field in metadata.keys()
            if field not in all_valid_fields and not field.startswith("_")
        ]

        return {"missing_required": missing_required, "unknown_fields": unknown_fields}


@dataclass
class PendingDocument:
    """Represents a document pending metadata completion."""

    path: str
    detected_metadata: dict[str, Any]
    required_metadata: dict[str, Any]
    document_type: str
    confidence: float = 0.0
    extraction_errors: list[str] = field(default_factory=list)

    def is_complete(self) -> bool:
        """Check if all required metadata is provided."""
        schema = DOCUMENT_SCHEMAS.get(self.document_type)
        if not schema:
            return False

        validation = schema.validate_metadata(self.required_metadata)
        return len(validation["missing_required"]) == 0


@dataclass
class YamlMetadataFile:
    """Represents a YAML metadata file for pending documents."""

    generated_at: str
    engine_version: str
    library_collection: str
    pending_files: list[PendingDocument]
    completed_files: list[str] = field(default_factory=list)

    def to_yaml_dict(self) -> dict[str, Any]:
        """Convert to YAML-serializable dictionary."""
        return {
            "metadata": {
                "generated_at": self.generated_at,
                "engine_version": self.engine_version,
                "library_collection": self.library_collection,
                "instructions": [
                    "Fill in the required metadata fields marked with '?'",
                    "You can modify detected metadata if it's incorrect",
                    "Remove files from pending_files if you don't want to process them",
                    "Run 'wqm ingest yaml <this-file>' when complete",
                ],
            },
            "document_types": {
                name: {
                    "primary_version": schema.primary_version,
                    "required_metadata": schema.required_metadata,
                    "optional_metadata": schema.optional_metadata,
                }
                for name, schema in DOCUMENT_SCHEMAS.items()
            },
            "pending_files": [
                {
                    "path": doc.path,
                    "document_type": doc.document_type,
                    "confidence": doc.confidence,
                    "detected_metadata": doc.detected_metadata,
                    "required_metadata": doc.required_metadata,
                    "extraction_errors": doc.extraction_errors
                    if doc.extraction_errors
                    else None,
                }
                for doc in self.pending_files
            ],
            "completed_files": self.completed_files,
        }

    @classmethod
    def from_yaml_dict(cls, data: dict[str, Any]) -> "YamlMetadataFile":
        """Create from YAML dictionary."""
        metadata = data.get("metadata", {})
        pending_files_data = data.get("pending_files", [])

        pending_files = []
        for file_data in pending_files_data:
            pending_files.append(
                PendingDocument(
                    path=file_data["path"],
                    detected_metadata=file_data.get("detected_metadata", {}),
                    required_metadata=file_data.get("required_metadata", {}),
                    document_type=file_data.get("document_type", "unknown"),
                    confidence=file_data.get("confidence", 0.0),
                    extraction_errors=file_data.get("extraction_errors", []) or [],
                )
            )

        return cls(
            generated_at=metadata.get("generated_at", ""),
            engine_version=metadata.get("engine_version", ""),
            library_collection=metadata.get("library_collection", ""),
            pending_files=pending_files,
            completed_files=data.get("completed_files", []),
        )


# Document type schemas based on PRD Task 6 research
DOCUMENT_SCHEMAS = {
    "book": DocumentTypeSchema(
        name="book",
        primary_version="edition",
        required_metadata=["title", "author", "edition"],
        optional_metadata=[
            "isbn",
            "publisher",
            "year",
            "language",
            "pages",
            "genre",
            "tags",
        ],
    ),
    "scientific_article": DocumentTypeSchema(
        name="scientific_article",
        primary_version="publication_date",
        required_metadata=["title", "authors", "journal", "publication_date"],
        optional_metadata=[
            "doi",
            "volume",
            "issue",
            "pages",
            "abstract",
            "keywords",
            "tags",
        ],
    ),
    "webpage": DocumentTypeSchema(
        name="webpage",
        primary_version="ingestion_date",
        required_metadata=["title", "url", "ingestion_date"],
        optional_metadata=[
            "author",
            "site_name",
            "description",
            "tags",
            "last_modified",
        ],
    ),
    "report": DocumentTypeSchema(
        name="report",
        primary_version="publication_date",
        required_metadata=["title", "author", "publication_date"],
        optional_metadata=[
            "organization",
            "report_number",
            "pages",
            "abstract",
            "tags",
        ],
    ),
    "presentation": DocumentTypeSchema(
        name="presentation",
        primary_version="date",
        required_metadata=["title", "author", "date"],
        optional_metadata=["event", "location", "slides", "duration", "tags"],
    ),
    "manual": DocumentTypeSchema(
        name="manual",
        primary_version="version",
        required_metadata=["title", "version"],
        optional_metadata=["author", "product", "company", "date", "pages", "tags"],
    ),
    "unknown": DocumentTypeSchema(
        name="unknown",
        primary_version="date",
        required_metadata=["title"],
        optional_metadata=["author", "date", "source", "type", "tags"],
    ),
}


class DocumentTypeDetector:
    """Detects document types based on content and metadata analysis."""

    def __init__(self):
        self.confidence_thresholds = {"high": 0.8, "medium": 0.5, "low": 0.3}

    async def detect_document_type(
        self, parsed_doc: ParsedDocument, file_path: Path
    ) -> tuple[str, float]:
        """
        Detect the document type and confidence level.

        Args:
            parsed_doc: Parsed document with content and metadata
            file_path: Path to the original file

        Returns:
            Tuple of (document_type, confidence_score)
        """
        content = parsed_doc.content.lower()
        metadata = parsed_doc.metadata
        filename = file_path.name.lower()

        # Initialize scores for each document type
        type_scores = {
            doc_type: 0.0
            for doc_type in DOCUMENT_SCHEMAS.keys()
            if doc_type != "unknown"
        }

        # File extension based scoring
        if file_path.suffix.lower() == ".pdf":
            type_scores["book"] += 0.2
            type_scores["scientific_article"] += 0.3
            type_scores["report"] += 0.3
            type_scores["manual"] += 0.2

        # Content pattern analysis
        patterns = {
            "book": [
                "chapter",
                "isbn",
                "publisher",
                "copyright",
                "edition",
                "table of contents",
                "bibliography",
                "index",
            ],
            "scientific_article": [
                "abstract",
                "keywords",
                "doi:",
                "journal",
                "volume",
                "reference",
                "methodology",
                "conclusion",
                "research",
            ],
            "webpage": [
                "http://",
                "https://",
                "www.",
                ".com",
                ".org",
                ".net",
                "website",
                "browser",
                "url",
                "link",
            ],
            "report": [
                "executive summary",
                "findings",
                "recommendations",
                "analysis",
                "report",
                "study",
                "survey",
                "assessment",
            ],
            "presentation": [
                "slide",
                "presentation",
                "powerpoint",
                "keynote",
                "agenda",
                "outline",
                "summary",
                "overview",
            ],
            "manual": [
                "manual",
                "guide",
                "instruction",
                "how to",
                "step by step",
                "user guide",
                "documentation",
                "procedure",
                "tutorial",
            ],
        }

        # Score based on pattern matches
        for doc_type, keywords in patterns.items():
            matches = sum(1 for keyword in keywords if keyword in content)
            type_scores[doc_type] += min(matches * 0.1, 0.5)

        # Filename analysis
        filename_patterns = {
            "book": ["book", "ebook", "novel", "guide"],
            "scientific_article": ["paper", "article", "journal", "research"],
            "webpage": ["web", "html", "site"],
            "report": ["report", "analysis", "study"],
            "presentation": ["slides", "presentation", "ppt"],
            "manual": ["manual", "guide", "instructions", "howto"],
        }

        for doc_type, keywords in filename_patterns.items():
            if any(keyword in filename for keyword in keywords):
                type_scores[doc_type] += 0.3

        # Metadata analysis
        if "title" in metadata:
            title_lower = str(metadata["title"]).lower()
            # Boost scores based on title analysis
            if any(word in title_lower for word in ["journal", "paper", "research"]):
                type_scores["scientific_article"] += 0.2
            elif any(
                word in title_lower for word in ["manual", "guide", "instructions"]
            ):
                type_scores["manual"] += 0.2
            elif any(word in title_lower for word in ["report", "analysis", "study"]):
                type_scores["report"] += 0.2

        # Find the best match
        best_type = max(type_scores.items(), key=lambda x: x[1])
        best_doc_type, confidence = best_type

        # Use 'unknown' if confidence is too low
        if confidence < self.confidence_thresholds["low"]:
            return "unknown", confidence

        return best_doc_type, min(confidence, 1.0)


class MetadataExtractor:
    """Enhanced metadata extraction from documents."""

    def __init__(self):
        self.parsers = {
            ".pdf": PDFParser(),
            ".md": MarkdownParser(),
            ".txt": TextParser(),
            ".pptx": PptxParser(),
        }

    async def extract_enhanced_metadata(
        self, file_path: Path, document_type: str
    ) -> dict[str, Any]:
        """
        Extract enhanced metadata based on document type.

        Args:
            file_path: Path to the document file
            document_type: Detected document type

        Returns:
            Dictionary of extracted metadata
        """
        # Get basic metadata from parser
        parser = self._get_parser(file_path)
        if not parser:
            return {}

        parsed_doc = await parser.parse(file_path)
        metadata = parsed_doc.metadata.copy()

        # Type-specific metadata extraction
        if document_type == "book":
            metadata.update(await self._extract_book_metadata(parsed_doc, file_path))
        elif document_type == "scientific_article":
            metadata.update(await self._extract_article_metadata(parsed_doc, file_path))
        elif document_type == "webpage":
            metadata.update(await self._extract_webpage_metadata(parsed_doc, file_path))

        # Clean and standardize metadata
        return self._clean_metadata(metadata)

    def _get_parser(self, file_path: Path) -> DocumentParser | None:
        """Get appropriate parser for file."""
        extension = file_path.suffix.lower()
        return self.parsers.get(extension)

    async def _extract_book_metadata(
        self, parsed_doc: ParsedDocument, file_path: Path
    ) -> dict[str, Any]:
        """Extract book-specific metadata."""
        content = parsed_doc.content
        metadata = {}

        # Try to extract title from content if not in PDF metadata
        if "title" not in parsed_doc.metadata or not parsed_doc.metadata["title"]:
            title = self._extract_title_from_content(content)
            if title:
                metadata["title"] = title

        # Try to extract author information
        author = self._extract_author_from_content(content)
        if author:
            metadata["author"] = author

        # Try to extract ISBN
        isbn = self._extract_isbn_from_content(content)
        if isbn:
            metadata["isbn"] = isbn

        # Try to extract publisher
        publisher = self._extract_publisher_from_content(content)
        if publisher:
            metadata["publisher"] = publisher

        # Extract year from various sources
        year = self._extract_year_from_content(content)
        if year:
            metadata["year"] = year

        # Estimate edition (default to 1st if not found)
        metadata["edition"] = "1st"

        return metadata

    async def _extract_article_metadata(
        self, parsed_doc: ParsedDocument, file_path: Path
    ) -> dict[str, Any]:
        """Extract scientific article-specific metadata."""
        content = parsed_doc.content
        metadata = {}

        # Extract title (usually at the beginning)
        title = self._extract_title_from_content(content)
        if title:
            metadata["title"] = title

        # Extract authors (multiple authors support)
        authors = self._extract_authors_from_content(content)
        if authors:
            metadata["authors"] = authors

        # Extract journal information
        journal = self._extract_journal_from_content(content)
        if journal:
            metadata["journal"] = journal

        # Extract DOI
        doi = self._extract_doi_from_content(content)
        if doi:
            metadata["doi"] = doi

        # Extract publication date
        pub_date = self._extract_publication_date_from_content(content)
        if pub_date:
            metadata["publication_date"] = pub_date

        return metadata

    async def _extract_webpage_metadata(
        self, parsed_doc: ParsedDocument, file_path: Path
    ) -> dict[str, Any]:
        """Extract webpage-specific metadata."""
        metadata = {}

        # For webpages, we'd need the original URL
        # For now, set ingestion date to current time
        metadata["ingestion_date"] = datetime.now(timezone.utc).isoformat()

        # Try to extract title
        title = self._extract_title_from_content(parsed_doc.content)
        if title:
            metadata["title"] = title

        return metadata

    def _extract_title_from_content(self, content: str) -> str | None:
        """Extract title from document content."""
        lines = content.split("\n")

        # Look for title in first few lines
        for _i, line in enumerate(lines[:10]):
            line = line.strip()
            if len(line) > 10 and len(line) < 200:
                # Simple heuristic: likely title if it's not too short or long
                # and doesn't contain common body text patterns
                if not any(
                    word in line.lower()
                    for word in ["the", "and", "this", "chapter", "section", "page"]
                ):
                    return line

        return None

    def _extract_author_from_content(self, content: str) -> str | None:
        """Extract author from document content."""
        import re

        # Look for "by [author]" pattern
        by_pattern = r"by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"
        match = re.search(by_pattern, content, re.IGNORECASE)
        if match:
            return match.group(1)

        # Look for "Author: [name]" pattern
        author_pattern = r"Author:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"
        match = re.search(author_pattern, content, re.IGNORECASE)
        if match:
            return match.group(1)

        return None

    def _extract_authors_from_content(self, content: str) -> str | None:
        """Extract multiple authors from scientific content."""
        import re

        # Look for author list patterns common in papers
        patterns = [
            r"Authors?:\s*([A-Z][^.]+)",
            r"Written by:?\s*([A-Z][^.]+)",
            r"^([A-Z][a-z]+(?:\s+[A-Z]\.\s*)?[A-Z][a-z]+(?:,\s*[A-Z][a-z]+(?:\s+[A-Z]\.\s*)?[A-Z][a-z]+)*)",
        ]

        lines = content.split("\n")[:20]  # Check first 20 lines

        for pattern in patterns:
            for line in lines:
                match = re.search(pattern, line.strip())
                if match:
                    return match.group(1).strip()

        return None

    def _extract_isbn_from_content(self, content: str) -> str | None:
        """Extract ISBN from content."""
        import re

        # ISBN-10 or ISBN-13 pattern
        isbn_pattern = r"ISBN(?:-1[03])?\s*:?\s*([0-9-]{10,17})"
        match = re.search(isbn_pattern, content, re.IGNORECASE)
        if match:
            return match.group(1).replace("-", "")

        return None

    def _extract_publisher_from_content(self, content: str) -> str | None:
        """Extract publisher from content."""
        import re

        patterns = [
            r"Publisher:\s*([^.\n]+)",
            r"Published by\s+([^.\n]+)",
            r"©\s*\d{4}\s+([^.\n]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                publisher = match.group(1).strip()
                if len(publisher) > 2 and len(publisher) < 100:
                    return publisher

        return None

    def _extract_year_from_content(self, content: str) -> str | None:
        """Extract publication year from content."""
        import re

        # Look for 4-digit years in common contexts
        year_patterns = [
            r"©\s*(\d{4})",
            r"Published in (\d{4})",
            r"Copyright (\d{4})",
            r"\b(19|20)\d{2}\b",
        ]

        for pattern in year_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                year = match if isinstance(match, str) else match
                try:
                    year_int = int(year)
                    if 1800 <= year_int <= 2030:  # Reasonable year range
                        return str(year_int)
                except ValueError:
                    continue

        return None

    def _extract_journal_from_content(self, content: str) -> str | None:
        """Extract journal name from scientific article."""
        import re

        patterns = [
            r"Journal:\s*([^.\n]+)",
            r"Published in\s+([^.,\n]+)",
            r"Appeared in\s+([^.,\n]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                journal = match.group(1).strip()
                if len(journal) > 5 and len(journal) < 100:
                    return journal

        return None

    def _extract_doi_from_content(self, content: str) -> str | None:
        """Extract DOI from content."""
        import re

        doi_pattern = r"DOI:\s*(10\.\d+/[^\s]+)"
        match = re.search(doi_pattern, content, re.IGNORECASE)
        if match:
            return match.group(1)

        return None

    def _extract_publication_date_from_content(self, content: str) -> str | None:
        """Extract publication date from content."""
        import re

        # Look for various date formats
        date_patterns = [
            r"Published:\s*([A-Z][a-z]+ \d{1,2},? \d{4})",
            r"Date:\s*([A-Z][a-z]+ \d{4})",
            r"(\d{4}-\d{2}-\d{2})",
            r"([A-Z][a-z]+,? \d{4})",
        ]

        for pattern in date_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    def _clean_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Clean and standardize metadata values."""
        cleaned = {}

        for key, value in metadata.items():
            if value is None or value == "":
                continue

            # Convert to string and clean
            if isinstance(value, str):
                value = value.strip()
                if value and value != "?":
                    cleaned[key] = value
            else:
                cleaned[key] = value

        return cleaned


class YamlMetadataWorkflow:
    """Main workflow manager for YAML metadata completion system."""

    def __init__(self, client: QdrantWorkspaceClient):
        self.client = client
        self.type_detector = DocumentTypeDetector()
        self.metadata_extractor = MetadataExtractor()

    async def generate_yaml_file(
        self,
        library_path: Path,
        library_collection: str,
        output_path: Path | None = None,
        formats: list[str] | None = None,
    ) -> Path:
        """
        Generate YAML metadata file for documents in library path.

        Args:
            library_path: Path to library folder with documents
            library_collection: Target library collection name
            output_path: Optional output path for YAML file
            formats: File formats to process

        Returns:
            Path to generated YAML file
        """
        if not library_path.exists():
            raise FileNotFoundError(f"Library path not found: {library_path}")

        # Find documents to process
        documents = await self._find_documents(library_path, formats)

        if not documents:
            logger.warning(f"No documents found in {library_path}")
            return None

        logger.info(f"Found {len(documents)} documents to process")

        # Process each document
        pending_docs = []
        for doc_path in documents:
            try:
                pending_doc = await self._process_document(doc_path, library_collection)
                if pending_doc:
                    pending_docs.append(pending_doc)
            except Exception as e:
                logger.error(f"Failed to process {doc_path}: {e}")
                # Add as unknown type with error
                pending_docs.append(
                    PendingDocument(
                        path=str(doc_path),
                        detected_metadata={},
                        required_metadata={"title": "?"},
                        document_type="unknown",
                        confidence=0.0,
                        extraction_errors=[str(e)],
                    )
                )

        # Generate YAML file
        if not output_path:
            output_path = library_path / "metadata_completion.yaml"

        yaml_file = YamlMetadataFile(
            generated_at=datetime.now(timezone.utc).isoformat(),
            engine_version="1.0.0",
            library_collection=library_collection,
            pending_files=pending_docs,
        )

        await self._save_yaml_file(yaml_file, output_path)

        logger.info(f"Generated YAML metadata file: {output_path}")
        return output_path

    async def process_yaml_file(
        self, yaml_path: Path, dry_run: bool = False
    ) -> dict[str, Any]:
        """
        Process completed YAML metadata file and ingest documents.

        Args:
            yaml_path: Path to YAML metadata file
            dry_run: If True, validate but don't actually ingest

        Returns:
            Processing results summary
        """
        if not yaml_path.exists():
            raise FileNotFoundError(f"YAML file not found: {yaml_path}")

        # Load and validate YAML file
        yaml_file = await self._load_yaml_file(yaml_path)

        # Process documents with complete metadata
        results = {"processed": 0, "skipped": 0, "errors": [], "remaining": []}

        remaining_docs = []

        for doc in yaml_file.pending_files:
            try:
                if doc.is_complete():
                    if not dry_run:
                        await self._ingest_document_with_metadata(
                            doc, yaml_file.library_collection
                        )
                    results["processed"] += 1
                    logger.info(f"Processed: {doc.path}")
                else:
                    remaining_docs.append(doc)
                    results["skipped"] += 1
                    logger.warning(f"Skipped incomplete metadata: {doc.path}")

            except Exception as e:
                error_msg = f"Failed to process {doc.path}: {e}"
                results["errors"].append(error_msg)
                logger.error(error_msg)
                remaining_docs.append(doc)

        # Update YAML file with remaining documents
        if remaining_docs and not dry_run:
            yaml_file.pending_files = remaining_docs
            await self._save_yaml_file(yaml_file, yaml_path)

        results["remaining"] = len(remaining_docs)

        return results

    async def _find_documents(
        self, library_path: Path, formats: list[str] | None = None
    ) -> list[Path]:
        """Find documents in library path."""
        if not formats:
            formats = ["pdf", "txt", "md", "epub"]

        documents = []

        for fmt in formats:
            pattern = f"**/*.{fmt.lower()}"
            documents.extend(library_path.glob(pattern))
            if fmt != fmt.upper():
                pattern = f"**/*.{fmt.upper()}"
                documents.extend(library_path.glob(pattern))

        return sorted(set(documents))  # Remove duplicates and sort

    async def _process_document(
        self, doc_path: Path, library_collection: str
    ) -> PendingDocument | None:
        """Process a single document for metadata extraction."""
        try:
            # Parse document
            parser = self.metadata_extractor._get_parser(doc_path)
            if not parser:
                logger.warning(f"No parser available for {doc_path}")
                return None

            parsed_doc = await parser.parse(doc_path)

            # Detect document type
            doc_type, confidence = await self.type_detector.detect_document_type(
                parsed_doc, doc_path
            )

            # Extract enhanced metadata
            detected_metadata = await self.metadata_extractor.extract_enhanced_metadata(
                doc_path, doc_type
            )

            # Generate required metadata template
            schema = DOCUMENT_SCHEMAS[doc_type]
            required_metadata = {}

            for field in schema.required_metadata:
                if field in detected_metadata and detected_metadata[field]:
                    required_metadata[field] = detected_metadata[field]
                else:
                    required_metadata[field] = "?"

            # Add optional metadata if detected
            for field in schema.optional_metadata:
                if field in detected_metadata and detected_metadata[field]:
                    required_metadata[field] = detected_metadata[field]

            return PendingDocument(
                path=str(doc_path),
                detected_metadata=detected_metadata,
                required_metadata=required_metadata,
                document_type=doc_type,
                confidence=confidence,
            )

        except Exception as e:
            logger.error(f"Error processing {doc_path}: {e}")
            return PendingDocument(
                path=str(doc_path),
                detected_metadata={},
                required_metadata={"title": "?"},
                document_type="unknown",
                confidence=0.0,
                extraction_errors=[str(e)],
            )

    async def _save_yaml_file(self, yaml_file: YamlMetadataFile, output_path: Path):
        """Save YAML file to disk."""
        yaml_data = yaml_file.to_yaml_dict()

        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(
                yaml_data,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
                indent=2,
            )

    async def _load_yaml_file(self, yaml_path: Path) -> YamlMetadataFile:
        """Load YAML file from disk."""
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return YamlMetadataFile.from_yaml_dict(data)

    async def _ingest_document_with_metadata(
        self, doc: PendingDocument, collection: str
    ):
        """Ingest document with completed metadata into collection."""
        from ..tools.documents import add_document

        doc_path = Path(doc.path)
        parser = self.metadata_extractor._get_parser(doc_path)

        if not parser:
            raise ValueError(f"No parser available for {doc_path}")

        # Parse document again for content
        parsed_doc = await parser.parse(doc_path)

        # Combine detected metadata with user-provided metadata
        final_metadata = {}
        final_metadata.update(parsed_doc.metadata)
        final_metadata.update(doc.required_metadata)

        # Add document type information
        final_metadata["document_type"] = doc.document_type
        final_metadata["metadata_confidence"] = doc.confidence

        # Add to collection
        result = await add_document(
            client=self.client,
            content=parsed_doc.content,
            collection=collection,
            metadata=final_metadata,
            document_id=f"{doc_path.stem}_{parsed_doc.content_hash[:8]}",
            chunk_text=True,
        )

        if result.get("error"):
            raise RuntimeError(result["error"])

        return result
