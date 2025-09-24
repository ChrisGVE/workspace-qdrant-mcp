"""
Enhanced Version Management System for workspace-qdrant-mcp.

This module implements Task 262 requirements:
1. Document Type-Based Versioning
2. Conflict Resolution with Format Precedence
3. Archive Collections Management
4. User Workflow Integration

Key Features:
- Books: Edition-based versioning (1st edition, 2nd edition, etc.)
- Code files: Git tag precedence (semantic versioning)
- Scientific articles: Publication date precedence
- Web pages: Ingestion date with content hash validation
- Documentation: Version metadata from frontmatter
"""

import asyncio
import hashlib
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
try:
    import semver
except ImportError:
    semver = None
    logger.warning("semver library not installed, falling back to string comparison for semantic versions")

try:
    import yaml
except ImportError:
    yaml = None
    logger.warning("PyYAML not installed, frontmatter parsing disabled")

from python.common.core.client import QdrantWorkspaceClient
from qdrant_client.http import models


class DocumentType(Enum):
    """Document types with specific versioning strategies."""
    BOOK = "book"
    CODE_FILE = "code_file"
    SCIENTIFIC_ARTICLE = "scientific_article"
    WEB_PAGE = "web_page"
    DOCUMENTATION = "documentation"
    GENERIC = "generic"


class FileFormat(Enum):
    """File formats with precedence order."""
    PDF = ("pdf", 10)  # Highest precedence
    DOCX = ("docx", 9)
    DOC = ("doc", 8)
    RTF = ("rtf", 7)
    HTML = ("html", 6)
    MARKDOWN = ("md", 5)
    TXT = ("txt", 4)  # Lowest precedence
    UNKNOWN = ("unknown", 1)

    def __init__(self, extension: str, precedence: int):
        self.extension = extension
        self.precedence = precedence


class ConflictType(Enum):
    """Types of version conflicts."""
    VERSION_CONFLICT = "version_conflict"  # Different versions of same document
    FORMAT_CONFLICT = "format_conflict"    # Same content, different formats
    METADATA_CONFLICT = "metadata_conflict"  # Conflicting metadata
    CONTENT_CONFLICT = "content_conflict"   # Same ID, different content
    TEMPORAL_CONFLICT = "temporal_conflict"  # Timestamp inconsistencies


class ResolutionStrategy(Enum):
    """Conflict resolution strategies."""
    USER_DECISION = "user_decision"        # Require user input
    FORMAT_PRECEDENCE = "format_precedence"  # Use format precedence rules
    VERSION_PRECEDENCE = "version_precedence"  # Use version comparison
    TIMESTAMP_PRECEDENCE = "timestamp_precedence"  # Use newest timestamp
    AUTHORITY_PRECEDENCE = "authority_precedence"  # Use authority source
    ARCHIVE_ALL = "archive_all"           # Archive conflicts, keep all
    MANUAL_OVERRIDE = "manual_override"    # User override decision


@dataclass
class VersionInfo:
    """Version information for a document."""
    version_string: str
    version_type: str  # semantic, edition, date, timestamp, git_tag
    document_type: DocumentType
    authority_level: float  # 0.0 to 1.0
    timestamp: datetime
    content_hash: str
    format: FileFormat
    metadata: Dict[str, Any]
    point_id: str
    supersedes: List[str] = None

    def __post_init__(self):
        if self.supersedes is None:
            self.supersedes = []


@dataclass
class VersionConflict:
    """Represents a version conflict between documents."""
    conflict_type: ConflictType
    conflicting_versions: List[VersionInfo]
    recommended_strategy: ResolutionStrategy
    conflict_severity: float  # 0.0 to 1.0
    user_message: str
    resolution_options: List[Dict[str, Any]]


class VersionManager:
    """
    Comprehensive version management system with conflict resolution.

    Handles document type-based versioning, format precedence analysis,
    conflict detection and resolution, and archive management.
    """

    def __init__(self, client: QdrantWorkspaceClient):
        """Initialize version manager with workspace client."""
        self.client = client

        # Document type configurations
        self.type_configs = {
            DocumentType.BOOK: {
                "primary_version": "edition",
                "secondary_version": "date",
                "version_pattern": r"(\d+)(?:st|nd|rd|th)?\s*(?:edition|ed\.?)",
                "required_metadata": ["title", "author", "edition"],
                "optional_metadata": ["isbn", "publisher", "draft_status"],
                "precedence_rules": ["edition", "publication_date", "timestamp"],
                "retention_policy": "archive_superseded",
                "conflict_tolerance": 0.2,
            },
            DocumentType.CODE_FILE: {
                "primary_version": "git_tag",
                "secondary_version": "modification_date",
                "version_pattern": r"v?(\d+\.\d+\.\d+(?:-[\w\d]+)?(?:\+[\w\d]+)?)",
                "auto_metadata": True,
                "precedence_rules": ["semantic_version", "git_commit", "file_mtime"],
                "retention_policy": "keep_major_versions",
                "conflict_tolerance": 0.1,
            },
            DocumentType.SCIENTIFIC_ARTICLE: {
                "primary_version": "publication_date",
                "version_pattern": r"(\d{4}-\d{2}-\d{2})",
                "required_metadata": ["title", "authors", "journal", "publication_date"],
                "optional_metadata": ["doi", "volume", "issue", "pages"],
                "precedence_rules": ["publication_date", "doi", "journal_authority"],
                "retention_policy": "latest_only",
                "conflict_tolerance": 0.3,
            },
            DocumentType.WEB_PAGE: {
                "primary_version": "ingestion_date",
                "version_pattern": r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})",
                "required_metadata": ["title", "url", "ingestion_date"],
                "optional_metadata": ["site_authority", "content_type"],
                "precedence_rules": ["content_hash", "ingestion_date", "url_authority"],
                "retention_policy": "time_based_cleanup",
                "conflict_tolerance": 0.4,
            },
            DocumentType.DOCUMENTATION: {
                "primary_version": "frontmatter_version",
                "version_pattern": r"version:\s*['\"]?([^'\"\s]+)['\"]?",
                "required_metadata": ["title", "version"],
                "optional_metadata": ["date", "author", "category"],
                "precedence_rules": ["frontmatter_version", "file_mtime", "content_hash"],
                "retention_policy": "version_branches",
                "conflict_tolerance": 0.2,
            }
        }

        # Format precedence mapping
        self.format_precedence = {fmt.extension: fmt.precedence for fmt in FileFormat}

    def detect_document_type(self, metadata: Dict[str, Any], content: str = "") -> DocumentType:
        """
        Automatically detect document type from metadata and content.
        """
        # Check explicit document_type in metadata
        if "document_type" in metadata:
            try:
                return DocumentType(metadata["document_type"])
            except ValueError:
                pass

        # Check file extension
        if "file_path" in metadata:
            path = Path(metadata["file_path"])
            ext = path.suffix.lower().lstrip(".")

            if ext in ["py", "js", "ts", "java", "cpp", "c", "rs", "go"]:
                return DocumentType.CODE_FILE
            elif ext in ["md", "rst", "txt"] and self._has_frontmatter(content):
                return DocumentType.DOCUMENTATION

        # Check content patterns
        if "isbn" in metadata or "edition" in metadata:
            return DocumentType.BOOK

        if "doi" in metadata or "journal" in metadata:
            return DocumentType.SCIENTIFIC_ARTICLE

        if "url" in metadata:
            return DocumentType.WEB_PAGE

        return DocumentType.GENERIC

    def _has_frontmatter(self, content: str) -> bool:
        """Check if content has YAML frontmatter."""
        return content.strip().startswith("---\n")

    def extract_version_from_content(self, content: str, doc_type: DocumentType) -> Optional[str]:
        """
        Extract version information from document content.
        """
        config = self.type_configs.get(doc_type, {})
        pattern = config.get("version_pattern")

        if not pattern:
            return None

        match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1)

        # For documentation with frontmatter
        if doc_type == DocumentType.DOCUMENTATION and self._has_frontmatter(content) and yaml:
            try:
                # Extract YAML frontmatter
                parts = content.split("---", 2)
                if len(parts) >= 2:
                    frontmatter = yaml.safe_load(parts[1])
                    return frontmatter.get("version")
            except yaml.YAMLError:
                pass

        return None

    def compare_versions(self, version1: str, version2: str, doc_type: DocumentType) -> int:
        """
        Compare two versions for the given document type.

        Returns:
            -1 if version1 < version2
             0 if version1 == version2
             1 if version1 > version2
        """
        if doc_type == DocumentType.CODE_FILE and semver:
            try:
                # Use semantic versioning comparison
                v1 = semver.VersionInfo.parse(version1.lstrip("v"))
                v2 = semver.VersionInfo.parse(version2.lstrip("v"))
                return v1.compare(v2)
            except ValueError:
                # Fallback to string comparison
                pass

        elif doc_type == DocumentType.BOOK:
            # Extract edition numbers
            def extract_edition(v):
                match = re.search(r"(\d+)", v)
                return int(match.group(1)) if match else 0

            e1 = extract_edition(version1)
            e2 = extract_edition(version2)
            return (e1 > e2) - (e1 < e2)

        elif doc_type == DocumentType.SCIENTIFIC_ARTICLE:
            # Compare publication dates
            try:
                d1 = datetime.fromisoformat(version1.replace("Z", "+00:00"))
                d2 = datetime.fromisoformat(version2.replace("Z", "+00:00"))
                return (d1 > d2) - (d1 < d2)
            except ValueError:
                pass

        # Generic string comparison
        return (version1 > version2) - (version1 < version2)

    def get_file_format(self, metadata: Dict[str, Any]) -> FileFormat:
        """Determine file format from metadata."""
        if "file_path" in metadata:
            ext = Path(metadata["file_path"]).suffix.lower().lstrip(".")
            for fmt in FileFormat:
                if fmt.extension == ext:
                    return fmt

        if "format" in metadata:
            format_str = metadata["format"].lower()
            for fmt in FileFormat:
                if fmt.extension == format_str:
                    return fmt

        return FileFormat.UNKNOWN

    def calculate_content_hash(self, content: str) -> str:
        """Calculate SHA256 hash of content."""
        return hashlib.sha256(content.encode()).hexdigest()

    async def find_conflicting_versions(
        self,
        document_id: str,
        collection: str,
        new_version_info: VersionInfo
    ) -> List[VersionConflict]:
        """
        Find all versions that conflict with the new version.
        """
        # Get existing versions
        existing_versions = await self._get_existing_versions(document_id, collection)

        conflicts = []

        for existing in existing_versions:
            conflict = self._analyze_version_conflict(new_version_info, existing)
            if conflict:
                conflicts.append(conflict)

        return conflicts

    async def _get_existing_versions(self, document_id: str, collection: str) -> List[VersionInfo]:
        """Get all existing versions of a document."""
        if not self.client.initialized:
            return []

        try:
            # Search for documents with matching document_id
            points, _ = await self.client.client.scroll(
                collection_name=collection,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id)
                        )
                    ]
                ),
                with_payload=True,
                limit=100,
            )

            versions = []
            for point in points:
                payload = point.payload

                version_info = VersionInfo(
                    version_string=payload.get("version", ""),
                    version_type=payload.get("version_type", "timestamp"),
                    document_type=DocumentType(payload.get("document_type", "generic")),
                    authority_level=payload.get("authority_level", 0.5),
                    timestamp=datetime.fromisoformat(
                        payload.get("timestamp", datetime.now(timezone.utc).isoformat())
                    ),
                    content_hash=payload.get("content_hash", ""),
                    format=self.get_file_format(payload),
                    metadata=payload,
                    point_id=str(point.id),
                    supersedes=payload.get("supersedes", [])
                )
                versions.append(version_info)

            return versions

        except Exception as e:
            logger.error("Failed to get existing versions: %s", e)
            return []

    def _analyze_version_conflict(
        self,
        new_version: VersionInfo,
        existing_version: VersionInfo
    ) -> Optional[VersionConflict]:
        """
        Analyze potential conflict between two versions.
        """
        conflict_types = []
        severity = 0.0

        # Check for version conflicts
        if (new_version.version_string and existing_version.version_string and
            new_version.version_string != existing_version.version_string):

            version_cmp = self.compare_versions(
                new_version.version_string,
                existing_version.version_string,
                new_version.document_type
            )

            if version_cmp == 0:
                conflict_types.append(ConflictType.VERSION_CONFLICT)
                severity += 0.8

        # Check for format conflicts
        if (new_version.format != existing_version.format and
            new_version.content_hash == existing_version.content_hash):
            conflict_types.append(ConflictType.FORMAT_CONFLICT)
            severity += 0.4

        # Check for content conflicts (same ID, different content)
        if (new_version.content_hash != existing_version.content_hash and
            new_version.version_string == existing_version.version_string):
            conflict_types.append(ConflictType.CONTENT_CONFLICT)
            severity += 0.9

        # Check for temporal conflicts
        time_diff = abs((new_version.timestamp - existing_version.timestamp).total_seconds())
        if time_diff < 60 and new_version.content_hash != existing_version.content_hash:
            conflict_types.append(ConflictType.TEMPORAL_CONFLICT)
            severity += 0.3

        if not conflict_types:
            return None

        # Determine recommended strategy
        strategy = self._recommend_resolution_strategy(
            conflict_types, new_version, existing_version, severity
        )

        # Generate user message
        message = self._generate_conflict_message(
            conflict_types, new_version, existing_version
        )

        # Generate resolution options
        options = self._generate_resolution_options(
            conflict_types, new_version, existing_version
        )

        return VersionConflict(
            conflict_type=conflict_types[0],  # Primary conflict type
            conflicting_versions=[new_version, existing_version],
            recommended_strategy=strategy,
            conflict_severity=min(severity, 1.0),
            user_message=message,
            resolution_options=options
        )

    def _recommend_resolution_strategy(
        self,
        conflict_types: List[ConflictType],
        new_version: VersionInfo,
        existing_version: VersionInfo,
        severity: float
    ) -> ResolutionStrategy:
        """Recommend resolution strategy based on conflict analysis."""

        # High severity conflicts require user decision
        if severity > 0.8:
            return ResolutionStrategy.USER_DECISION

        # Format conflicts can use format precedence
        if ConflictType.FORMAT_CONFLICT in conflict_types:
            return ResolutionStrategy.FORMAT_PRECEDENCE

        # Version conflicts can use version precedence
        if ConflictType.VERSION_CONFLICT in conflict_types:
            return ResolutionStrategy.VERSION_PRECEDENCE

        # Temporal conflicts use timestamp precedence
        if ConflictType.TEMPORAL_CONFLICT in conflict_types:
            return ResolutionStrategy.TIMESTAMP_PRECEDENCE

        # Default to user decision for complex cases
        return ResolutionStrategy.USER_DECISION

    def _generate_conflict_message(
        self,
        conflict_types: List[ConflictType],
        new_version: VersionInfo,
        existing_version: VersionInfo
    ) -> str:
        """Generate human-readable conflict message."""

        type_messages = {
            ConflictType.VERSION_CONFLICT: f"Version conflict: '{new_version.version_string}' vs '{existing_version.version_string}'",
            ConflictType.FORMAT_CONFLICT: f"Format conflict: {new_version.format.extension} vs {existing_version.format.extension}",
            ConflictType.CONTENT_CONFLICT: "Content conflict: Same version, different content",
            ConflictType.TEMPORAL_CONFLICT: "Temporal conflict: Similar timestamps, different content"
        }

        messages = [type_messages.get(ct, str(ct)) for ct in conflict_types]
        return "; ".join(messages)

    def _generate_resolution_options(
        self,
        conflict_types: List[ConflictType],
        new_version: VersionInfo,
        existing_version: VersionInfo
    ) -> List[Dict[str, Any]]:
        """Generate available resolution options."""

        options = []

        # Always offer user decision
        options.append({
            "strategy": ResolutionStrategy.USER_DECISION,
            "description": "Let user decide which version to keep",
            "action": "prompt_user"
        })

        # Format precedence if applicable
        if ConflictType.FORMAT_CONFLICT in conflict_types:
            if new_version.format.precedence > existing_version.format.precedence:
                preferred = "new"
            else:
                preferred = "existing"

            options.append({
                "strategy": ResolutionStrategy.FORMAT_PRECEDENCE,
                "description": f"Keep {preferred} version based on format precedence",
                "action": f"keep_{preferred}"
            })

        # Version precedence if applicable
        if ConflictType.VERSION_CONFLICT in conflict_types:
            version_cmp = self.compare_versions(
                new_version.version_string,
                existing_version.version_string,
                new_version.document_type
            )

            if version_cmp > 0:
                preferred = "new"
            elif version_cmp < 0:
                preferred = "existing"
            else:
                preferred = "newest_timestamp"

            options.append({
                "strategy": ResolutionStrategy.VERSION_PRECEDENCE,
                "description": f"Keep {preferred} version based on version comparison",
                "action": f"keep_{preferred}"
            })

        # Archive all option
        options.append({
            "strategy": ResolutionStrategy.ARCHIVE_ALL,
            "description": "Archive existing version and keep new version",
            "action": "archive_and_replace"
        })

        return options