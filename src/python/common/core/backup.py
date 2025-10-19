"""
Backup and restore functionality for workspace-qdrant-mcp.

This module provides data structures and utilities for backing up and restoring
system state, including version compatibility validation.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from common.core.error_handling import IncompatibleVersionError


class CompatibilityStatus(Enum):
    """Version compatibility status."""
    COMPATIBLE = "compatible"  # Same major and minor version
    INCOMPATIBLE = "incompatible"  # Different major or minor version
    UPGRADE_AVAILABLE = "upgrade_available"  # Backup is older, patch version differs
    DOWNGRADE = "downgrade"  # Current version is older than backup


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


class VersionValidator:
    """
    Utility class for validating version compatibility between backups and current system.

    Uses semantic versioning rules (major.minor.patch) to determine compatibility:
    - Compatible: Same major and minor version (patch can differ)
    - Incompatible: Different major or minor version
    """

    @staticmethod
    def parse_version(version_str: str) -> tuple[int, int, int]:
        """
        Parse version string into (major, minor, patch) tuple.

        Args:
            version_str: Version string in semver format (e.g., "0.2.1", "1.0.0dev1")

        Returns:
            Tuple of (major, minor, patch) integers

        Raises:
            ValueError: If version format is invalid
        """
        parts = version_str.split(".")
        if len(parts) < 2:
            raise ValueError(f"Invalid version format: {version_str}")

        major = int(parts[0])
        minor = int(parts[1])

        # Patch version is optional and may have dev suffix
        if len(parts) >= 3:
            # Extract leading numeric part (e.g., "1dev1" -> 1)
            patch_str = parts[2]
            numeric_part = ""
            for c in patch_str:
                if c.isdigit():
                    numeric_part += c
                else:
                    break
            patch = int(numeric_part) if numeric_part else 0
        else:
            patch = 0

        return (major, minor, patch)

    @staticmethod
    def check_compatibility(
        backup_version: str,
        current_version: str,
        strict: bool = True
    ) -> CompatibilityStatus:
        """
        Check compatibility between backup version and current system version.

        Args:
            backup_version: Version of the backup
            current_version: Current system version
            strict: If True, only same major.minor is compatible.
                   If False, allow patch version differences as compatible.

        Returns:
            CompatibilityStatus enum value

        Examples:
            >>> VersionValidator.check_compatibility("0.2.1", "0.2.1")
            CompatibilityStatus.COMPATIBLE
            >>> VersionValidator.check_compatibility("0.2.0", "0.2.1")
            CompatibilityStatus.COMPATIBLE
            >>> VersionValidator.check_compatibility("0.1.0", "0.2.0")
            CompatibilityStatus.INCOMPATIBLE
        """
        backup_major, backup_minor, backup_patch = VersionValidator.parse_version(backup_version)
        current_major, current_minor, current_patch = VersionValidator.parse_version(current_version)

        # Check major and minor version compatibility
        if backup_major != current_major or backup_minor != current_minor:
            return CompatibilityStatus.INCOMPATIBLE

        # Same major.minor - check patch version
        if backup_patch == current_patch:
            return CompatibilityStatus.COMPATIBLE
        elif backup_patch < current_patch:
            return CompatibilityStatus.UPGRADE_AVAILABLE
        else:  # backup_patch > current_patch
            return CompatibilityStatus.DOWNGRADE

    @staticmethod
    def validate_compatibility(
        backup_metadata: BackupMetadata,
        current_version: str,
        allow_downgrade: bool = False
    ) -> None:
        """
        Validate backup version compatibility, raising exception if incompatible.

        Args:
            backup_metadata: Backup metadata containing version information
            current_version: Current system version
            allow_downgrade: If True, allow restoring from newer backup version

        Raises:
            IncompatibleVersionError: If versions are incompatible
        """
        status = VersionValidator.check_compatibility(
            backup_metadata.version,
            current_version
        )

        if status == CompatibilityStatus.INCOMPATIBLE:
            raise IncompatibleVersionError(
                f"Backup version {backup_metadata.version} is incompatible with "
                f"current version {current_version}. "
                f"Major or minor version mismatch detected.",
                backup_version=backup_metadata.version,
                current_version=current_version
            )

        if status == CompatibilityStatus.DOWNGRADE and not allow_downgrade:
            raise IncompatibleVersionError(
                f"Cannot restore from newer backup version {backup_metadata.version} "
                f"to older system version {current_version}. "
                f"Use allow_downgrade=True to override.",
                backup_version=backup_metadata.version,
                current_version=current_version
            )

    @staticmethod
    def get_compatibility_message(
        backup_version: str,
        current_version: str
    ) -> str:
        """
        Get human-readable compatibility message.

        Args:
            backup_version: Version of the backup
            current_version: Current system version

        Returns:
            Human-readable compatibility message
        """
        status = VersionValidator.check_compatibility(backup_version, current_version)

        messages = {
            CompatibilityStatus.COMPATIBLE: (
                f"Backup version {backup_version} is compatible with "
                f"current version {current_version}"
            ),
            CompatibilityStatus.UPGRADE_AVAILABLE: (
                f"Backup version {backup_version} is older than "
                f"current version {current_version} (patch difference only)"
            ),
            CompatibilityStatus.DOWNGRADE: (
                f"Backup version {backup_version} is newer than "
                f"current version {current_version} (patch difference only)"
            ),
            CompatibilityStatus.INCOMPATIBLE: (
                f"Backup version {backup_version} is incompatible with "
                f"current version {current_version} (major or minor version differs)"
            ),
        }

        return messages[status]
