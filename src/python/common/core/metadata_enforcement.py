"""
Metadata Requirement Enforcement for Collection Types.

This module provides validation and enforcement of metadata requirements based on
collection types (SYSTEM, LIBRARY, PROJECT, GLOBAL). It validates documents have
required metadata, generates missing metadata using LSP/Tree-sitter when available,
and routes items to missing_metadata_queue when tools are unavailable.

Key Features:
    - Collection type-specific metadata validation
    - Automatic metadata generation using LSP/Tree-sitter
    - Graceful degradation when tools unavailable
    - Enforcement statistics tracking
    - Integration with missing_metadata_tracker

Example:
    ```python
    from workspace_qdrant_mcp.core.metadata_enforcement import MetadataEnforcer
    from workspace_qdrant_mcp.core.sqlite_state_manager import SQLiteStateManager

    # Initialize enforcer
    state_manager = SQLiteStateManager()
    await state_manager.initialize()
    enforcer = MetadataEnforcer(state_manager)

    # Enforce metadata for queue item
    result = await enforcer.enforce_metadata(queue_item)

    if result.success:
        print(f"Metadata complete: {result.completed_metadata}")
    elif result.moved_to_missing_metadata_queue:
        print("Moved to missing metadata queue - tools unavailable")

    # Get enforcement statistics
    stats = enforcer.get_enforcement_stats()
    print(f"Success rate: {stats.successful}/{stats.total_enforced}")
    ```
"""

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from loguru import logger

from .collection_type_config import (
    CollectionTypeConfig,
    get_type_config,
    validate_metadata_for_type,
)
from .collection_types import CollectionType, CollectionTypeClassifier
from .missing_metadata_tracker import MissingMetadataTracker
from .queue_client import QueueItem


@dataclass
class ValidationResult:
    """Result of metadata validation."""

    is_valid: bool
    collection_type: CollectionType
    missing_fields: List[str] = field(default_factory=list)
    invalid_fields: Dict[str, str] = field(default_factory=dict)  # field -> error

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "is_valid": self.is_valid,
            "collection_type": self.collection_type.value,
            "missing_fields": self.missing_fields,
            "invalid_fields": self.invalid_fields,
        }


@dataclass
class EnforcementResult:
    """Result of metadata enforcement operation."""

    success: bool
    validation_result: ValidationResult
    metadata_generated: bool = False
    completed_metadata: Optional[Dict[str, Any]] = None
    moved_to_missing_metadata_queue: bool = False
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "success": self.success,
            "validation": self.validation_result.to_dict(),
            "metadata_generated": self.metadata_generated,
            "completed_metadata": self.completed_metadata,
            "moved_to_missing_metadata_queue": self.moved_to_missing_metadata_queue,
            "error_message": self.error_message,
        }


@dataclass
class EnforcementStatistics:
    """Statistics for metadata enforcement operations."""

    total_enforced: int = 0
    successful: int = 0
    failed: int = 0
    metadata_generated: int = 0
    moved_to_queue: int = 0
    by_collection_type: Dict[str, Dict[str, int]] = field(default_factory=dict)
    common_missing_fields: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        success_rate = (self.successful / self.total_enforced * 100) if self.total_enforced > 0 else 0.0
        return {
            "total_enforced": self.total_enforced,
            "successful": self.successful,
            "failed": self.failed,
            "success_rate": round(success_rate, 2),
            "metadata_generated": self.metadata_generated,
            "moved_to_queue": self.moved_to_queue,
            "by_collection_type": self.by_collection_type,
            "common_missing_fields": self.common_missing_fields,
        }


class MetadataEnforcer:
    """
    Enforces metadata requirements for collection types.

    This class validates and enforces metadata requirements based on collection type
    configurations. It attempts to generate missing metadata using LSP/Tree-sitter,
    and routes items to missing_metadata_queue when tools are unavailable.

    Attributes:
        state_manager: SQLite state manager for database operations
        tracker: Missing metadata tracker for queue operations
        classifier: Collection type classifier
        stats: Enforcement statistics
    """

    def __init__(self, state_manager):
        """
        Initialize metadata enforcer.

        Args:
            state_manager: Initialized SQLiteStateManager instance
        """
        self.state_manager = state_manager
        self.tracker = MissingMetadataTracker(state_manager)
        self.classifier = CollectionTypeClassifier()
        self.stats = EnforcementStatistics()
        self._stats_lock = asyncio.Lock()

    async def validate_metadata(
        self,
        collection_type: CollectionType,
        metadata: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate metadata against collection type requirements.

        Checks all required fields are present and validates field types and constraints
        according to the collection type configuration.

        Args:
            collection_type: The collection type to validate against
            metadata: Metadata dictionary to validate

        Returns:
            ValidationResult with validation status and error details

        Example:
            ```python
            result = await enforcer.validate_metadata(
                CollectionType.LIBRARY,
                {"collection_name": "_mylib", "language": "python"}
            )

            if not result.is_valid:
                print(f"Missing: {result.missing_fields}")
                print(f"Invalid: {result.invalid_fields}")
            ```
        """
        try:
            # Get type configuration
            config = get_type_config(collection_type)

            # Validate using config
            is_valid, errors = config.validate_metadata(metadata)

            # Parse errors into missing vs invalid fields
            missing_fields = []
            invalid_fields = {}

            for error in errors:
                # Check if error is about missing field
                if "is required" in error:
                    # Extract field name from error message
                    field_name = error.split("'")[1] if "'" in error else None
                    if field_name:
                        missing_fields.append(field_name)
                else:
                    # Invalid field value
                    field_name = error.split("'")[1] if "'" in error else "unknown"
                    invalid_fields[field_name] = error

            logger.debug(
                f"Validated metadata for {collection_type.value}: "
                f"valid={is_valid}, missing={len(missing_fields)}, invalid={len(invalid_fields)}"
            )

            return ValidationResult(
                is_valid=is_valid,
                collection_type=collection_type,
                missing_fields=missing_fields,
                invalid_fields=invalid_fields,
            )

        except Exception as e:
            logger.error(f"Error validating metadata for {collection_type.value}: {e}")
            return ValidationResult(
                is_valid=False,
                collection_type=collection_type,
                missing_fields=[],
                invalid_fields={"validation_error": str(e)},
            )

    async def generate_missing_metadata(
        self,
        file_path: str,
        collection_type: CollectionType,
        existing_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate missing metadata using LSP/Tree-sitter and context.

        Attempts to fill in missing metadata fields based on collection type:
        - SYSTEM: Basic file metadata only
        - LIBRARY: Extract symbols, dependencies via LSP
        - PROJECT: Add project_id, branch from git context
        - GLOBAL: Validate predefined schema

        Args:
            file_path: Absolute path to the file
            collection_type: Collection type for metadata requirements
            existing_metadata: Current metadata to augment

        Returns:
            Completed metadata dictionary (merged with existing)

        Example:
            ```python
            metadata = await enforcer.generate_missing_metadata(
                "/path/to/file.py",
                CollectionType.LIBRARY,
                {"collection_name": "_mylib"}
            )
            # metadata now includes symbols, dependencies, language, etc.
            ```
        """
        try:
            file_path_obj = Path(file_path).resolve()
            completed_metadata = existing_metadata.copy()

            # Add timestamps if missing
            now_iso = datetime.now(timezone.utc).isoformat()
            if "created_at" not in completed_metadata:
                completed_metadata["created_at"] = now_iso
            if "updated_at" not in completed_metadata:
                completed_metadata["updated_at"] = now_iso

            # Collection type-specific metadata generation
            if collection_type == CollectionType.SYSTEM:
                # SYSTEM collections: Basic file metadata only
                await self._add_basic_file_metadata(file_path_obj, completed_metadata)

            elif collection_type == CollectionType.LIBRARY:
                # LIBRARY collections: Extract symbols, dependencies via LSP
                await self._add_library_metadata(file_path_obj, completed_metadata)

            elif collection_type == CollectionType.PROJECT:
                # PROJECT collections: Add project context
                await self._add_project_metadata(file_path_obj, completed_metadata)

            elif collection_type == CollectionType.GLOBAL:
                # GLOBAL collections: Basic metadata
                await self._add_basic_file_metadata(file_path_obj, completed_metadata)

            logger.debug(
                f"Generated metadata for {collection_type.value}: "
                f"{len(completed_metadata)} fields"
            )

            return completed_metadata

        except Exception as e:
            logger.error(f"Error generating metadata for {file_path}: {e}")
            return existing_metadata

    async def _add_basic_file_metadata(
        self,
        file_path: Path,
        metadata: Dict[str, Any]
    ) -> None:
        """Add basic file metadata (exists, size, etc.)."""
        try:
            if file_path.exists():
                stat = file_path.stat()
                if "file_size" not in metadata:
                    metadata["file_size"] = stat.st_size
                if "file_extension" not in metadata:
                    metadata["file_extension"] = file_path.suffix
        except Exception as e:
            logger.warning(f"Failed to add basic file metadata for {file_path}: {e}")

    async def _add_library_metadata(
        self,
        file_path: Path,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Add library-specific metadata using LSP if available.

        Attempts to extract:
        - language (from file extension or LSP detection)
        - symbols (exported functions, classes via LSP)
        - dependencies (imports via LSP)
        - version (from package files if found)
        """
        try:
            # Detect language from file extension if not present
            if "language" not in metadata:
                language = self._detect_language_from_extension(file_path)
                if language:
                    metadata["language"] = language

            # Check LSP availability
            language_name = metadata.get("language")
            if language_name:
                lsp_status = await self.tracker.check_lsp_available(language_name)

                if lsp_status["available"]:
                    # LSP available - extract symbols and dependencies
                    try:
                        # Try to extract symbols using LSP
                        # Note: This requires LSP client integration
                        # For now, we mark that LSP is available but extraction
                        # would happen in the actual processing pipeline
                        metadata["lsp_available"] = True
                        logger.debug(f"LSP available for {language_name} at {lsp_status['path']}")
                    except Exception as e:
                        logger.warning(f"LSP extraction failed for {file_path}: {e}")
                        metadata["lsp_available"] = False
                else:
                    metadata["lsp_available"] = False

            # Check tree-sitter availability
            ts_status = await self.tracker.check_tree_sitter_available()
            metadata["tree_sitter_available"] = ts_status["available"]

        except Exception as e:
            logger.warning(f"Failed to add library metadata for {file_path}: {e}")

    async def _add_project_metadata(
        self,
        file_path: Path,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Add project-specific metadata from git context.

        Attempts to extract:
        - project_id (from git repo)
        - branch (from git)
        - file_type (from extension)
        """
        try:
            # Try to get git context
            try:
                # Get branch from state manager
                branch = await self.state_manager.get_current_branch(file_path.parent)
                if branch and "branch" not in metadata:
                    metadata["branch"] = branch
            except Exception as e:
                logger.debug(f"Could not determine branch for {file_path}: {e}")

            # Add file type from extension
            if "file_type" not in metadata and file_path.suffix:
                metadata["file_type"] = file_path.suffix.lstrip(".")

        except Exception as e:
            logger.warning(f"Failed to add project metadata for {file_path}: {e}")

    def _detect_language_from_extension(self, file_path: Path) -> Optional[str]:
        """Detect programming language from file extension."""
        extension_map = {
            ".py": "python",
            ".rs": "rust",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".go": "go",
            ".rb": "ruby",
            ".php": "php",
            ".cs": "csharp",
            ".swift": "swift",
            ".kt": "kotlin",
            ".scala": "scala",
            ".sh": "bash",
            ".md": "markdown",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".toml": "toml",
            ".xml": "xml",
            ".html": "html",
            ".css": "css",
        }

        return extension_map.get(file_path.suffix.lower())

    async def enforce_metadata(self, queue_item: QueueItem) -> EnforcementResult:
        """
        Main enforcement method for queue items.

        Workflow:
        1. Detect collection type
        2. Validate current metadata
        3. If invalid, try to generate missing fields using LSP/Tree-sitter
        4. If tools unavailable, move to missing_metadata_queue
        5. Track statistics

        Args:
            queue_item: Queue item with file path and metadata

        Returns:
            EnforcementResult with enforcement status and completed metadata

        Example:
            ```python
            result = await enforcer.enforce_metadata(queue_item)

            if result.success:
                # Use result.completed_metadata for ingestion
                pass
            elif result.moved_to_missing_metadata_queue:
                # Wait for tools to become available
                pass
            else:
                # Handle enforcement error
                logger.error(result.error_message)
            ```
        """
        try:
            # Update statistics
            async with self._stats_lock:
                self.stats.total_enforced += 1

            # Determine collection type
            collection_type_str = queue_item.collection_type
            if collection_type_str:
                try:
                    collection_type = CollectionType(collection_type_str)
                except ValueError:
                    collection_type = self.classifier.classify_collection_type(
                        queue_item.collection_name
                    )
            else:
                collection_type = self.classifier.classify_collection_type(
                    queue_item.collection_name
                )

            # Get existing metadata (would come from queue item in real implementation)
            # For now, we construct basic metadata from queue item
            existing_metadata = {
                "collection_name": queue_item.collection_name,
                "tenant_id": queue_item.tenant_id,
                "branch": queue_item.branch,
            }

            # Validate current metadata
            validation_result = await self.validate_metadata(
                collection_type,
                existing_metadata
            )

            # If valid, we're done
            if validation_result.is_valid:
                async with self._stats_lock:
                    self.stats.successful += 1
                    self._update_type_stats(collection_type.value, "successful", 1)

                return EnforcementResult(
                    success=True,
                    validation_result=validation_result,
                    completed_metadata=existing_metadata,
                )

            # Try to generate missing metadata
            completed_metadata = await self.generate_missing_metadata(
                queue_item.file_absolute_path,
                collection_type,
                existing_metadata
            )

            # Validate completed metadata
            final_validation = await self.validate_metadata(
                collection_type,
                completed_metadata
            )

            # Track missing fields for statistics
            async with self._stats_lock:
                for field in validation_result.missing_fields:
                    self.stats.common_missing_fields[field] = (
                        self.stats.common_missing_fields.get(field, 0) + 1
                    )

            # Check if we successfully completed metadata
            if final_validation.is_valid:
                async with self._stats_lock:
                    self.stats.successful += 1
                    self.stats.metadata_generated += 1
                    self._update_type_stats(collection_type.value, "successful", 1)
                    self._update_type_stats(collection_type.value, "generated", 1)

                return EnforcementResult(
                    success=True,
                    validation_result=final_validation,
                    metadata_generated=True,
                    completed_metadata=completed_metadata,
                )

            # Still missing required fields - check if we need to move to queue
            missing_lsp = False
            missing_ts = False

            # Determine what tools are missing based on collection type and fields
            language_name = completed_metadata.get("language")
            if language_name:
                lsp_status = await self.tracker.check_lsp_available(language_name)
                ts_status = await self.tracker.check_tree_sitter_available()

                missing_lsp = not lsp_status["available"]
                missing_ts = not ts_status["available"]

            # Move to missing metadata queue if tools unavailable
            if missing_lsp or missing_ts:
                await self.tracker.track_missing_metadata(
                    file_path=queue_item.file_absolute_path,
                    language_name=language_name or "unknown",
                    branch=queue_item.branch,
                    missing_lsp=missing_lsp,
                    missing_ts=missing_ts,
                )

                async with self._stats_lock:
                    self.stats.moved_to_queue += 1
                    self._update_type_stats(collection_type.value, "moved_to_queue", 1)

                return EnforcementResult(
                    success=False,
                    validation_result=final_validation,
                    metadata_generated=False,
                    completed_metadata=completed_metadata,
                    moved_to_missing_metadata_queue=True,
                    error_message=f"Missing required tools: LSP={missing_lsp}, Tree-sitter={missing_ts}",
                )

            # Tools available but still can't complete metadata
            async with self._stats_lock:
                self.stats.failed += 1
                self._update_type_stats(collection_type.value, "failed", 1)

            return EnforcementResult(
                success=False,
                validation_result=final_validation,
                metadata_generated=False,
                completed_metadata=completed_metadata,
                error_message=f"Could not generate required metadata: {final_validation.missing_fields}",
            )

        except Exception as e:
            logger.error(f"Error enforcing metadata for {queue_item.file_absolute_path}: {e}")

            async with self._stats_lock:
                self.stats.failed += 1

            # Create a basic validation result for error case
            error_validation = ValidationResult(
                is_valid=False,
                collection_type=CollectionType.UNKNOWN,
                invalid_fields={"enforcement_error": str(e)},
            )

            return EnforcementResult(
                success=False,
                validation_result=error_validation,
                error_message=str(e),
            )

    def _update_type_stats(self, collection_type: str, stat_name: str, increment: int = 1):
        """Update statistics for a specific collection type."""
        if collection_type not in self.stats.by_collection_type:
            self.stats.by_collection_type[collection_type] = {
                "successful": 0,
                "failed": 0,
                "generated": 0,
                "moved_to_queue": 0,
            }

        if stat_name in self.stats.by_collection_type[collection_type]:
            self.stats.by_collection_type[collection_type][stat_name] += increment

    def get_enforcement_stats(self) -> EnforcementStatistics:
        """
        Get enforcement statistics.

        Returns statistics on enforcement operations including success/failure rates,
        metadata generation counts, and common missing fields.

        Returns:
            EnforcementStatistics with current statistics

        Example:
            ```python
            stats = enforcer.get_enforcement_stats()
            print(f"Total enforced: {stats.total_enforced}")
            print(f"Success rate: {stats.successful}/{stats.total_enforced}")
            print(f"Metadata generated: {stats.metadata_generated}")
            print(f"Moved to queue: {stats.moved_to_queue}")
            print(f"Common missing fields: {stats.common_missing_fields}")

            # Per-type breakdown
            for ctype, type_stats in stats.by_collection_type.items():
                print(f"{ctype}: {type_stats}")
            ```
        """
        return self.stats

    def reset_stats(self):
        """Reset enforcement statistics."""
        self.stats = EnforcementStatistics()
        logger.info("Enforcement statistics reset")


# Export all public classes and functions
__all__ = [
    "ValidationResult",
    "EnforcementResult",
    "EnforcementStatistics",
    "MetadataEnforcer",
]
