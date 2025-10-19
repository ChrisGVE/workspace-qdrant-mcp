"""
Backup and restore functionality for workspace-qdrant-mcp.

This module provides data structures and utilities for backing up and restoring
system state, including version compatibility validation.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class BackupMetadata:
    """
    Metadata for backup manifests.

    Contains version information, timestamps, and collection details
    required for backup creation and restoration.
    """

    # Required fields
    version: str  # System version (semver format: major.minor.patch)
    timestamp: float  # Unix timestamp when backup was created

    # Optional collection information
    collections: Optional[Union[List[str], Dict[str, int]]] = None  # Collection names or counts
    total_documents: Optional[int] = None  # Total number of documents across all collections

    # Optional system information
    database_version: Optional[str] = None  # SQLite or database version
    python_version: Optional[str] = None  # Python version used

    # Optional operational context
    during_operations: bool = False  # Whether backup was created during active operations
    partial_backup: bool = False  # Whether this is a partial/selective backup
    selected_collections: Optional[List[str]] = None  # Collections included in partial backup

    # Additional metadata
    description: Optional[str] = None  # Human-readable description
    created_by: Optional[str] = None  # User or system that created the backup
    backup_path: Optional[str] = None  # Path where backup is stored
    additional_metadata: Dict[str, Any] = field(default_factory=dict)  # Custom metadata

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metadata to dictionary for JSON serialization.

        Returns:
            Dictionary representation excluding None values.
        """
        data = asdict(self)
        # Remove None values to keep manifest clean
        result = {k: v for k, v in data.items() if v is not None and k != "additional_metadata"}

        # Merge additional_metadata at top level
        if self.additional_metadata:
            result.update(self.additional_metadata)

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BackupMetadata":
        """
        Create BackupMetadata from dictionary (e.g., loaded from JSON).

        Args:
            data: Dictionary with metadata fields

        Returns:
            BackupMetadata instance

        Raises:
            KeyError: If required fields are missing
            ValueError: If field types are invalid
        """
        # Extract known fields
        known_fields = {
            "version", "timestamp", "collections", "total_documents",
            "database_version", "python_version", "during_operations",
            "partial_backup", "selected_collections", "description",
            "created_by", "backup_path"
        }

        # Separate known fields from additional metadata
        metadata_fields = {k: v for k, v in data.items() if k in known_fields}
        additional_metadata = {k: v for k, v in data.items() if k not in known_fields}

        if additional_metadata:
            metadata_fields["additional_metadata"] = additional_metadata

        return cls(**metadata_fields)

    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """
        Save metadata to JSON file.

        Args:
            file_path: Path where manifest.json should be saved
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> "BackupMetadata":
        """
        Load metadata from JSON file.

        Args:
            file_path: Path to manifest.json file

        Returns:
            BackupMetadata instance

        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file contains invalid JSON
            KeyError: If required fields are missing
        """
        path = Path(file_path)
        data = json.loads(path.read_text())
        return cls.from_dict(data)

    def get_version_tuple(self) -> tuple[int, int, int]:
        """
        Parse version string into (major, minor, patch) tuple.

        Returns:
            Tuple of (major, minor, patch) version numbers

        Raises:
            ValueError: If version string is not valid semver format
        """
        parts = self.version.split(".")
        if len(parts) < 2:
            raise ValueError(f"Invalid version format: {self.version}")

        # Handle versions like "0.2.1dev1" by stripping non-numeric suffix
        major = int(parts[0])
        minor = int(parts[1])

        # Patch version is optional
        if len(parts) >= 3:
            # Extract leading numeric part only (e.g., "1dev1" -> 1, "123abc" -> 123)
            patch_str = parts[2]
            # Find where digits end
            numeric_part = ""
            for c in patch_str:
                if c.isdigit():
                    numeric_part += c
                else:
                    break  # Stop at first non-digit
            patch = int(numeric_part) if numeric_part else 0
        else:
            patch = 0

        return (major, minor, patch)

    def format_timestamp(self) -> str:
        """
        Format timestamp as human-readable string.

        Returns:
            ISO format timestamp string in UTC
        """
        from datetime import timezone
        return datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat()
