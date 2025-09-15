"""Configuration migration utilities for workspace-qdrant-mcp.

This module provides the ConfigMigrator class for detecting configuration versions,
identifying deprecated fields, determining migration paths, and managing configuration
backups for safe migrations with rollback capability. It also includes comprehensive
migration reporting and user notification systems.
"""

import difflib
import hashlib
import json
import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Union
from enum import Enum


class ConfigVersion(Enum):
    """Configuration version enumeration."""
    V1_LEGACY = "v1_legacy"
    V2_CURRENT = "v2_current"
    UNKNOWN = "unknown"


class MigrationComplexity(Enum):
    """Migration complexity levels."""
    NONE = "none"
    SIMPLE = "simple"
    COMPLEX = "complex"
    MANUAL = "manual"


class NotificationLevel(Enum):
    """Notification level enumeration."""
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    INFO = "info"


class ChangeType(Enum):
    """Types of configuration changes."""
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    DEPRECATED_REMOVED = "deprecated_removed"
    MIGRATED = "migrated"
    VALIDATED = "validated"


@dataclass
class ChangeEntry:
    """Represents a single configuration change."""
    change_type: ChangeType
    field_path: str
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None
    reason: Optional[str] = None
    section: Optional[str] = None
    migration_method: Optional[str] = None
    timestamp: Optional[str] = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['change_type'] = data['change_type'].value if hasattr(data['change_type'], 'value') else data['change_type']
        return data


@dataclass
class ValidationResult:
    """Represents validation results from migration."""
    is_valid: bool
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class MigrationReport:
    """Comprehensive migration report with detailed change tracking."""
    # Identification
    migration_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Version information
    source_version: str = "unknown"
    target_version: str = "v2_current"

    # File information
    config_file_path: Optional[str] = None
    backup_id: Optional[str] = None
    backup_location: Optional[str] = None

    # Migration details
    changes_made: List[ChangeEntry] = field(default_factory=list)
    deprecated_fields_handled: Dict[str, str] = field(default_factory=dict)  # field_path -> replacement_info
    validation_results: List[ValidationResult] = field(default_factory=list)

    # Status
    success: bool = True
    migration_duration_seconds: Optional[float] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    # Additional metadata
    migration_methods_used: List[str] = field(default_factory=list)
    complexity: Optional[str] = None
    rollback_instructions: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert ChangeEntry objects to dictionaries
        data['changes_made'] = [change.to_dict() for change in self.changes_made]
        # Convert ValidationResult objects to dictionaries
        data['validation_results'] = [result.to_dict() for result in self.validation_results]
        return data

    def add_change(self, change: ChangeEntry) -> None:
        """Add a change entry to the report."""
        self.changes_made.append(change)

    def add_validation_result(self, result: ValidationResult) -> None:
        """Add a validation result to the report."""
        self.validation_results.append(result)

    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)

    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)
        self.success = False


class ReportGenerator:
    """Generates detailed migration reports with before/after comparisons."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize ReportGenerator.

        Args:
            logger: Optional logger instance. If None, creates a default logger.
        """
        self.logger = logger or logging.getLogger(__name__)

    def generate_diff(self, before: Dict[str, Any], after: Dict[str, Any]) -> str:
        """Generate a detailed diff between before and after configurations.

        Args:
            before: Configuration before migration
            after: Configuration after migration

        Returns:
            String containing unified diff output
        """
        try:
            before_json = json.dumps(before, indent=2, sort_keys=True)
            after_json = json.dumps(after, indent=2, sort_keys=True)

            before_lines = before_json.splitlines(keepends=True)
            after_lines = after_json.splitlines(keepends=True)

            diff_lines = list(difflib.unified_diff(
                before_lines,
                after_lines,
                fromfile="before_migration.json",
                tofile="after_migration.json",
                lineterm=""
            ))

            return "".join(diff_lines)

        except Exception as e:
            self.logger.error(f"Failed to generate diff: {e}")
            return f"Error generating diff: {e}"

    def format_report_text(self, report: MigrationReport, include_diff: bool = True) -> str:
        """Format migration report as human-readable text.

        Args:
            report: MigrationReport to format
            include_diff: Whether to include configuration diff

        Returns:
            Formatted report text
        """
        lines = []
        lines.append("=" * 80)
        lines.append("CONFIGURATION MIGRATION REPORT")
        lines.append("=" * 80)
        lines.append("")

        # Basic information
        lines.append(f"Migration ID: {report.migration_id}")
        lines.append(f"Timestamp: {report.timestamp}")
        lines.append(f"Status: {'SUCCESS' if report.success else 'FAILED'}")
        lines.append(f"Version: {report.source_version} → {report.target_version}")
        if report.migration_duration_seconds:
            lines.append(f"Duration: {report.migration_duration_seconds:.2f} seconds")
        lines.append("")

        # File information
        if report.config_file_path:
            lines.append(f"Configuration File: {report.config_file_path}")
        if report.backup_id:
            lines.append(f"Backup ID: {report.backup_id}")
        if report.backup_location:
            lines.append(f"Backup Location: {report.backup_location}")
        lines.append("")

        # Migration methods used
        if report.migration_methods_used:
            lines.append("Migration Methods Applied:")
            for method in report.migration_methods_used:
                lines.append(f"  • {method}")
            lines.append("")

        # Changes made
        if report.changes_made:
            lines.append(f"Changes Made ({len(report.changes_made)} total):")
            lines.append("-" * 40)

            # Group changes by section
            changes_by_section = {}
            for change in report.changes_made:
                section = change.section or "general"
                if section not in changes_by_section:
                    changes_by_section[section] = []
                changes_by_section[section].append(change)

            for section, changes in changes_by_section.items():
                lines.append(f"\n[{section.upper()}]")
                for change in changes:
                    change_symbol = self._get_change_symbol(change.change_type)
                    lines.append(f"  {change_symbol} {change.field_path}")
                    if change.old_value is not None:
                        lines.append(f"      Old: {change.old_value}")
                    if change.new_value is not None:
                        lines.append(f"      New: {change.new_value}")
                    if change.reason:
                        lines.append(f"      Reason: {change.reason}")
            lines.append("")

        # Deprecated fields handled
        if report.deprecated_fields_handled:
            lines.append("Deprecated Fields Migration:")
            lines.append("-" * 40)
            for field_path, replacement in report.deprecated_fields_handled.items():
                lines.append(f"  • {field_path} → {replacement}")
            lines.append("")

        # Validation results
        if report.validation_results:
            lines.append("Validation Results:")
            lines.append("-" * 40)
            for i, result in enumerate(report.validation_results, 1):
                status = "VALID" if result.is_valid else "INVALID"
                lines.append(f"  Validation {i}: {status}")
                if result.warnings:
                    lines.append("    Warnings:")
                    for warning in result.warnings:
                        lines.append(f"      • {warning}")
                if result.errors:
                    lines.append("    Errors:")
                    for error in result.errors:
                        lines.append(f"      • {error}")
                if result.recommendations:
                    lines.append("    Recommendations:")
                    for rec in result.recommendations:
                        lines.append(f"      • {rec}")
            lines.append("")

        # Warnings and errors
        if report.warnings:
            lines.append("Warnings:")
            lines.append("-" * 40)
            for warning in report.warnings:
                lines.append(f"  ⚠️  {warning}")
            lines.append("")

        if report.errors:
            lines.append("Errors:")
            lines.append("-" * 40)
            for error in report.errors:
                lines.append(f"  ❌ {error}")
            lines.append("")

        # Rollback instructions
        if report.rollback_instructions:
            lines.append("Rollback Instructions:")
            lines.append("-" * 40)
            lines.append(report.rollback_instructions)
            lines.append("")

        lines.append("=" * 80)

        return "\n".join(lines)

    def format_report_json(self, report: MigrationReport) -> str:
        """Format migration report as JSON.

        Args:
            report: MigrationReport to format

        Returns:
            JSON-formatted report string
        """
        try:
            return json.dumps(report.to_dict(), indent=2)
        except Exception as e:
            self.logger.error(f"Failed to format report as JSON: {e}")
            return json.dumps({"error": f"Failed to format report: {e}"}, indent=2)

    def _get_change_symbol(self, change_type: ChangeType) -> str:
        """Get symbol for change type.

        Args:
            change_type: Type of change

        Returns:
            Symbol representing the change
        """
        symbols = {
            ChangeType.ADDED: "✅",
            ChangeType.REMOVED: "❌",
            ChangeType.MODIFIED: "🔄",
            ChangeType.DEPRECATED_REMOVED: "🗑️",
            ChangeType.MIGRATED: "🔀",
            ChangeType.VALIDATED: "✔️"
        }
        return symbols.get(change_type, "📝")


class NotificationSystem:
    """Handles user notifications for migration processes."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize NotificationSystem.

        Args:
            logger: Optional logger instance. If None, creates a default logger.
        """
        self.logger = logger or logging.getLogger(__name__)

    def notify_migration_started(self, migration_id: str, source_version: str, target_version: str) -> None:
        """Notify that migration has started.

        Args:
            migration_id: Unique migration identifier
            source_version: Source configuration version
            target_version: Target configuration version
        """
        self.logger.info(f"🚀 Starting configuration migration {migration_id}")
        self.logger.info(f"   Migrating from {source_version} to {target_version}")

    def notify_migration_success(self, report: MigrationReport) -> None:
        """Notify successful migration completion.

        Args:
            report: Migration report with details
        """
        self.logger.info("✅ Configuration migration completed successfully")
        self.logger.info(f"   Migration ID: {report.migration_id}")
        self.logger.info(f"   Changes made: {len(report.changes_made)}")

        if report.backup_id:
            self.logger.info(f"   Configuration backed up: {report.backup_id}")

        if report.warnings:
            self.logger.warning(f"   ⚠️  {len(report.warnings)} warnings (see report for details)")

        if report.deprecated_fields_handled:
            self.logger.info(f"   🗑️  {len(report.deprecated_fields_handled)} deprecated fields migrated")

        self.logger.info("   💡 Use 'wqm admin migration-report' to view detailed report")

    def notify_migration_failure(self, report: MigrationReport) -> None:
        """Notify migration failure.

        Args:
            report: Migration report with error details
        """
        self.logger.error("❌ Configuration migration failed")
        self.logger.error(f"   Migration ID: {report.migration_id}")

        if report.errors:
            self.logger.error("   Errors encountered:")
            for error in report.errors[:3]:  # Show first 3 errors
                self.logger.error(f"     • {error}")
            if len(report.errors) > 3:
                self.logger.error(f"     ... and {len(report.errors) - 3} more errors")

        if report.backup_id:
            self.logger.error(f"   🔄 Configuration can be restored from backup: {report.backup_id}")
            self.logger.error(f"   🔄 Use 'wqm admin rollback-config {report.backup_id}' to restore")

        self.logger.error("   📄 Use 'wqm admin migration-report' to view detailed error report")

    def notify_warnings(self, warnings: List[str]) -> None:
        """Notify about migration warnings.

        Args:
            warnings: List of warning messages
        """
        if warnings:
            self.logger.warning("⚠️  Migration warnings:")
            for warning in warnings:
                self.logger.warning(f"   • {warning}")

    def notify_deprecated_features(self, deprecated_fields: Dict[str, str]) -> None:
        """Notify about deprecated features and their replacements.

        Args:
            deprecated_fields: Mapping of deprecated fields to replacement info
        """
        if deprecated_fields:
            self.logger.warning("🔄 Deprecated configuration fields updated:")
            for field_path, replacement in deprecated_fields.items():
                self.logger.warning(f"   • {field_path} → {replacement}")
            self.logger.warning("   📚 See migration documentation for details")

    def notify_rollback_available(self, backup_id: str, backup_location: str) -> None:
        """Notify that rollback is available.

        Args:
            backup_id: Backup identifier
            backup_location: Location of backup file
        """
        self.logger.info("🔄 Configuration rollback available:")
        self.logger.info(f"   Backup ID: {backup_id}")
        self.logger.info(f"   Location: {backup_location}")
        self.logger.info(f"   Command: wqm admin rollback-config {backup_id}")

    def format_notification(self, level: NotificationLevel, message: str, details: Optional[List[str]] = None) -> str:
        """Format a notification message.

        Args:
            level: Notification level
            message: Main message
            details: Optional list of detail messages

        Returns:
            Formatted notification string
        """
        symbols = {
            NotificationLevel.SUCCESS: "✅",
            NotificationLevel.WARNING: "⚠️",
            NotificationLevel.ERROR: "❌",
            NotificationLevel.INFO: "ℹ️"
        }

        symbol = symbols.get(level, "📝")
        lines = [f"{symbol} {message}"]

        if details:
            for detail in details:
                lines.append(f"   • {detail}")

        return "\n".join(lines)


class ConfigMigrator:
    """Handles configuration version detection and migration path planning.

    This class can detect different configuration schema versions based on
    field presence and structure, identify deprecated fields, and recommend
    appropriate migration strategies.
    """

    # Deprecated fields from v1 that should not appear in v2
    DEPRECATED_FIELDS = {
        "collection_prefix",
        "max_collections",
        "recursive_depth",
        "enable_legacy_mode",  # Additional deprecated field
        # Pattern-related deprecated fields
        "include_patterns",
        "exclude_patterns",
        "file_patterns",
        "ignore_patterns",
        "supported_extensions",
        "file_types",
        "pattern_priorities",
        "custom_ecosystems"
    }

    # Fields that indicate v2 current schema
    V2_INDICATOR_FIELDS = {
        "project_detection",
        "collections.project_suffixes",
        "collections.global_collections",
        "ingestion.max_depth",
        "ingestion.batch_size",
        "search.hybrid_search"
    }

    # Required top-level sections in v2
    V2_REQUIRED_SECTIONS = {
        "qdrant",
        "embeddings",
        "collections",
        "ingestion",
        "search"
    }

    def __init__(self, logger: Optional[logging.Logger] = None, config_path: Optional[str] = None):
        """Initialize ConfigMigrator.

        Args:
            logger: Optional logger instance. If None, creates a default logger.
            config_path: Optional path to the configuration file for backup operations.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.config_path = config_path
        self._backup_dir = None
        self._migration_history_dir = None

        # Initialize reporting components
        self.report_generator = ReportGenerator(self.logger)
        self.notification_system = NotificationSystem(self.logger)

        # Current migration tracking
        self._current_migration_report: Optional[MigrationReport] = None

    def detect_config_version(self, config_data: Dict[str, Any]) -> ConfigVersion:
        """Detect configuration version based on structure and field presence.

        Args:
            config_data: Dictionary containing configuration data

        Returns:
            ConfigVersion enum indicating the detected version
        """
        if not isinstance(config_data, dict):
            self.logger.warning("Config data is not a dictionary, returning UNKNOWN")
            return ConfigVersion.UNKNOWN

        if not config_data:
            self.logger.info("Empty configuration detected")
            return ConfigVersion.UNKNOWN

        deprecated_fields = self._detect_deprecated_fields(config_data)
        v2_indicators = self._detect_v2_indicators(config_data)
        v2_sections = self._detect_v2_sections(config_data)

        self.logger.debug(f"Found {len(deprecated_fields)} deprecated fields: {deprecated_fields}")
        self.logger.debug(f"Found {len(v2_indicators)} v2 indicators: {v2_indicators}")
        self.logger.debug(f"Found {len(v2_sections)} v2 sections: {v2_sections}")

        # Determine version based on field analysis
        if deprecated_fields and not v2_indicators:
            self.logger.info("Detected v1 legacy configuration (deprecated fields present)")
            return ConfigVersion.V1_LEGACY

        if v2_indicators and len(v2_sections) >= 3:  # At least 3 v2 sections present
            if deprecated_fields:
                self.logger.warning(
                    f"Mixed version state: v2 structure with deprecated fields {deprecated_fields}"
                )
            else:
                self.logger.info("Detected v2 current configuration")
            return ConfigVersion.V2_CURRENT

        if not deprecated_fields and not v2_indicators:
            self.logger.warning("Cannot determine version: no clear indicators found")
        else:
            self.logger.warning("Mixed or incomplete configuration structure")

        return ConfigVersion.UNKNOWN

    def needs_migration(self, config_data: Dict[str, Any]) -> bool:
        """Determine if configuration needs migration.

        Args:
            config_data: Dictionary containing configuration data

        Returns:
            True if migration is needed, False otherwise
        """
        version = self.detect_config_version(config_data)
        needs_it = version in [ConfigVersion.V1_LEGACY, ConfigVersion.UNKNOWN]

        self.logger.info(f"Configuration version {version.value}, migration needed: {needs_it}")
        return needs_it

    def get_migration_path(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed migration path information.

        Args:
            config_data: Dictionary containing configuration data

        Returns:
            Dictionary containing migration path details including:
            - from_version: Source version
            - to_version: Target version
            - complexity: Migration complexity level
            - deprecated_fields: List of deprecated fields found
            - steps: List of migration steps
            - backup_recommended: Whether backup is recommended
            - warnings: List of warnings or issues
        """
        from_version = self.detect_config_version(config_data)
        to_version = ConfigVersion.V2_CURRENT
        deprecated_fields = self._detect_deprecated_fields(config_data)
        complexity = self._assess_migration_complexity(from_version, deprecated_fields)

        migration_info = {
            "from_version": from_version.value,
            "to_version": to_version.value,
            "complexity": complexity.value,
            "deprecated_fields": deprecated_fields,
            "backup_recommended": complexity in [MigrationComplexity.COMPLEX, MigrationComplexity.MANUAL],
            "warnings": [],
            "steps": []
        }

        # Generate migration steps based on version
        if from_version == ConfigVersion.V1_LEGACY:
            migration_info["steps"] = [
                "Backup current configuration file",
                "Transform deprecated fields to new schema structure",
                "Migrate collection configuration to new format",
                "Migrate pattern configuration to custom pattern system",
                "Add missing v2 required sections",
                "Validate new configuration structure",
                "Test configuration with workspace-qdrant-mcp"
            ]
            if len(deprecated_fields) > 3:
                migration_info["warnings"].append("Large number of deprecated fields may require manual review")

        elif from_version == ConfigVersion.UNKNOWN:
            migration_info["steps"] = [
                "Backup existing configuration (if any)",
                "Create fresh v2 configuration with defaults",
                "Review and customize configuration for your needs",
                "Test configuration with workspace-qdrant-mcp"
            ]
            migration_info["warnings"].append("Cannot parse existing configuration - will create fresh config")

        elif from_version == ConfigVersion.V2_CURRENT:
            if deprecated_fields:
                migration_info["steps"] = [
                    "Remove deprecated fields from configuration using remove_deprecated_fields()",
                    "Validate updated configuration"
                ]
                migration_info["warnings"].append("Found deprecated fields in v2 config")
            else:
                migration_info["steps"] = ["No migration needed - configuration is current"]
                migration_info["complexity"] = MigrationComplexity.NONE.value

        self.logger.info(f"Generated migration path: {from_version.value} -> {to_version.value} "
                        f"(complexity: {complexity.value})")

        return migration_info

    def _detect_deprecated_fields(self, config_data: Dict[str, Any]) -> List[str]:
        """Detect deprecated fields in configuration.

        Args:
            config_data: Dictionary containing configuration data

        Returns:
            List of deprecated field names found
        """
        deprecated_found = []

        def _check_nested_dict(data: Dict[str, Any], path: str = "") -> None:
            """Recursively check for deprecated fields in nested dictionaries."""
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key

                if key in self.DEPRECATED_FIELDS:
                    deprecated_found.append(current_path)

                if isinstance(value, dict):
                    _check_nested_dict(value, current_path)

        _check_nested_dict(config_data)
        return deprecated_found

    def _detect_v2_indicators(self, config_data: Dict[str, Any]) -> List[str]:
        """Detect v2 schema indicator fields.

        Args:
            config_data: Dictionary containing configuration data

        Returns:
            List of v2 indicator field paths found
        """
        v2_found = []

        def _check_nested_path(data: Dict[str, Any], path_parts: List[str]) -> bool:
            """Check if a nested path exists in the data."""
            current = data
            for part in path_parts:
                if not isinstance(current, dict) or part not in current:
                    return False
                current = current[part]
            return True

        for indicator in self.V2_INDICATOR_FIELDS:
            if "." in indicator:
                # Handle nested field paths like "collections.project_suffixes"
                path_parts = indicator.split(".")
                if _check_nested_path(config_data, path_parts):
                    v2_found.append(indicator)
            else:
                # Handle top-level fields
                if indicator in config_data:
                    v2_found.append(indicator)

        return v2_found

    def _detect_v2_sections(self, config_data: Dict[str, Any]) -> List[str]:
        """Detect v2 required sections.

        Args:
            config_data: Dictionary containing configuration data

        Returns:
            List of v2 sections found
        """
        return [section for section in self.V2_REQUIRED_SECTIONS
                if section in config_data and isinstance(config_data[section], dict)]

    def migrate_collection_config(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate collection configuration from legacy formats to new schema.

        This method handles the migration of collection-related configuration fields:
        1. collection_suffixes → collection_types (with CollectionType metadata)
        2. collection_prefix mapping and integration
        3. Backward compatibility with existing collection references
        4. Integration with new CollectionType classification system

        Args:
            config_data: Dictionary containing configuration data to migrate

        Returns:
            Dictionary containing migrated configuration with updated collection settings

        Raises:
            ValueError: If migration encounters unrecoverable conflicts
        """
        if not isinstance(config_data, dict):
            self.logger.warning("Config data is not a dictionary, cannot migrate collections")
            return config_data

        migrated_config = config_data.copy()
        migration_applied = False

        # Check if we have workspace configuration to migrate
        workspace_config = migrated_config.get("workspace", {})
        if not isinstance(workspace_config, dict):
            workspace_config = {}

        # 1. Handle collection_suffixes → collection_types migration
        if "collection_suffixes" in workspace_config:
            collection_suffixes = workspace_config["collection_suffixes"]
            if isinstance(collection_suffixes, list):
                if "collection_types" not in workspace_config:
                    # Migrate collection_suffixes to collection_types
                    workspace_config["collection_types"] = collection_suffixes.copy()
                    self.logger.info(f"Migrated collection_suffixes {collection_suffixes} to collection_types")
                    migration_applied = True
                else:
                    # Both present - use collection_types and warn
                    self.logger.warning(
                        "Both collection_suffixes and collection_types present. "
                        "Using collection_types and ignoring collection_suffixes."
                    )

                # Remove deprecated field
                workspace_config.pop("collection_suffixes", None)
                migration_applied = True

        # 2. Handle collection_prefix migration and mapping
        collection_prefix = workspace_config.get("collection_prefix")
        if collection_prefix:
            self.logger.warning(
                f"Migrating deprecated collection_prefix '{collection_prefix}'. "
                "Collection prefixes are now handled automatically by the CollectionType system."
            )

            # If we have collection_types, try to preserve the prefix intent
            collection_types = workspace_config.get("collection_types", [])
            if collection_types and isinstance(collection_types, list):
                # Map prefixed collections to standard types
                migrated_types = self._map_prefixed_collections(collection_types, collection_prefix)
                if migrated_types != collection_types:
                    workspace_config["collection_types"] = migrated_types
                    self.logger.info(f"Mapped prefixed collection types: {collection_types} → {migrated_types}")
                    migration_applied = True

            # Remove deprecated collection_prefix
            workspace_config.pop("collection_prefix", None)
            migration_applied = True

        # 3. Handle max_collections deprecation (if present)
        if "max_collections" in workspace_config:
            max_collections = workspace_config.pop("max_collections")
            self.logger.warning(
                f"Removed deprecated max_collections setting ({max_collections}). "
                "Collection limits are now managed through the multi-tenant architecture."
            )
            migration_applied = True

        # 4. Ensure collection_types has sensible defaults if empty
        collection_types = workspace_config.get("collection_types", [])
        if not collection_types and migration_applied:
            # If we migrated something but ended up with no types, add default
            default_types = ["scratchbook"]
            workspace_config["collection_types"] = default_types
            self.logger.info(f"Added default collection_types {default_types} after migration")

        # 5. Validate migrated collection types
        if migration_applied:
            validation_result = self._validate_migrated_collection_types(
                workspace_config.get("collection_types", [])
            )
            if not validation_result["is_valid"]:
                self.logger.warning(f"Collection types validation warning: {validation_result['warning']}")

        # Update the main config with migrated workspace config
        if migration_applied:
            migrated_config["workspace"] = workspace_config
            self.logger.info("Collection configuration migration completed successfully")

        return migrated_config

    def _map_prefixed_collections(self, collection_types: List[str], prefix: str) -> List[str]:
        """Map prefixed collection names to standard collection types.

        Args:
            collection_types: List of collection type names
            prefix: Collection prefix that was being used

        Returns:
            List of mapped collection type names
        """
        if not prefix or not collection_types:
            return collection_types

        mapped_types = []
        prefix = prefix.rstrip("_")  # Remove trailing underscore if present

        for collection_type in collection_types:
            if isinstance(collection_type, str):
                # Remove prefix if it was applied to the type name
                if collection_type.startswith(f"{prefix}_"):
                    mapped_type = collection_type[len(f"{prefix}_"):]
                    self.logger.debug(f"Mapped prefixed collection: {collection_type} → {mapped_type}")
                    mapped_types.append(mapped_type)
                elif collection_type.startswith(prefix):
                    mapped_type = collection_type[len(prefix):]
                    self.logger.debug(f"Mapped prefixed collection: {collection_type} → {mapped_type}")
                    mapped_types.append(mapped_type)
                else:
                    # Keep as-is if no prefix found
                    mapped_types.append(collection_type)
            else:
                mapped_types.append(collection_type)

        return mapped_types

    def _validate_migrated_collection_types(self, collection_types: List[str]) -> Dict[str, Any]:
        """Validate migrated collection types for common issues.

        Args:
            collection_types: List of collection type names to validate

        Returns:
            Dictionary with validation result including is_valid and warning
        """
        if not isinstance(collection_types, list):
            return {
                "is_valid": False,
                "warning": "Collection types must be a list"
            }

        if not collection_types:
            return {
                "is_valid": True,
                "warning": None
            }

        # Check for common issues
        warnings = []
        valid_types = []

        for collection_type in collection_types:
            if not isinstance(collection_type, str):
                warnings.append(f"Collection type must be string, got {type(collection_type).__name__}")
                continue

            # Check for reserved patterns that might cause conflicts
            if collection_type.startswith("_") or collection_type.startswith("__"):
                warnings.append(f"Collection type '{collection_type}' uses reserved prefix pattern")
            elif collection_type in ["memory", "system", "admin"]:
                warnings.append(f"Collection type '{collection_type}' conflicts with reserved names")
            else:
                valid_types.append(collection_type)

        result = {
            "is_valid": len(valid_types) > 0 or len(collection_types) == 0,
            "warning": "; ".join(warnings) if warnings else None
        }

        return result

    def _assess_migration_complexity(self,
                                   from_version: ConfigVersion,
                                   deprecated_fields: List[str]) -> MigrationComplexity:
        """Assess migration complexity based on version and deprecated fields.

        Args:
            from_version: Source configuration version
            deprecated_fields: List of deprecated fields found

        Returns:
            MigrationComplexity enum indicating complexity level
        """
        if from_version == ConfigVersion.V2_CURRENT:
            return MigrationComplexity.NONE if not deprecated_fields else MigrationComplexity.SIMPLE

        if from_version == ConfigVersion.V1_LEGACY:
            if len(deprecated_fields) <= 2:
                return MigrationComplexity.SIMPLE
            elif len(deprecated_fields) <= 5:
                return MigrationComplexity.COMPLEX
            else:
                return MigrationComplexity.MANUAL

        # Unknown version requires manual review
        return MigrationComplexity.MANUAL

    def remove_deprecated_fields(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove deprecated fields with validation and user warnings.

        This method safely removes deprecated configuration fields while ensuring
        functionality is preserved. It provides warnings about removed fields and
        their new equivalents, and implements fallback handling for configurations
        that rely on deprecated fields.

        Args:
            config_data: Dictionary containing configuration data to clean

        Returns:
            Dictionary containing cleaned configuration with deprecated fields removed

        Raises:
            ValueError: If removing deprecated fields would break functionality
        """
        if not isinstance(config_data, dict):
            self.logger.warning("Config data is not a dictionary, cannot remove deprecated fields")
            return config_data

        cleaned_config = config_data.copy()
        fields_removed = []
        warnings_issued = []
        validation_results = []

        # Track what sections we process
        processed_sections = set()

        # 1. Remove deprecated fields and collect warnings
        removed_fields = self._remove_deprecated_fields_from_config(
            cleaned_config, fields_removed, warnings_issued
        )

        # 2. Validate that functionality is preserved after removal
        if removed_fields:
            validation_result = self._validate_functionality_preservation(
                config_data, cleaned_config, removed_fields
            )
            validation_results.append(validation_result)

            # 3. Issue user warnings about removed fields and their equivalents
            self._issue_deprecation_warnings(removed_fields, warnings_issued)

            # 4. Handle fallback scenarios for critical deprecated fields
            fallback_changes = self._apply_fallback_handling(cleaned_config, removed_fields)
            if fallback_changes:
                validation_results.extend(fallback_changes)

        # 5. Final validation that cleaned config is functional
        final_validation = self._validate_cleaned_config(cleaned_config)
        if not final_validation["is_valid"]:
            raise ValueError(f"Removing deprecated fields would break functionality: {final_validation['error']}")

        # Log summary if any changes were made
        if removed_fields:
            self.logger.info(
                f"Successfully removed {len(removed_fields)} deprecated fields: {', '.join(removed_fields)}"
            )
            if warnings_issued:
                self.logger.info(f"Issued {len(warnings_issued)} deprecation warnings to user")

        return cleaned_config

    def _remove_deprecated_fields_from_config(self,
                                            config_data: Dict[str, Any],
                                            fields_removed: List[str],
                                            warnings_issued: List[str]) -> List[str]:
        """Remove deprecated fields from configuration data.

        Args:
            config_data: Configuration to modify in place
            fields_removed: List to track removed fields
            warnings_issued: List to track warnings issued

        Returns:
            List of removed field paths
        """
        removed_fields = []

        def _remove_from_section(section_data: Dict[str, Any], section_path: str = "") -> None:
            """Remove deprecated fields from a configuration section."""
            if not isinstance(section_data, dict):
                return

            fields_to_remove = []
            for key, value in section_data.items():
                current_path = f"{section_path}.{key}" if section_path else key

                if key in self.DEPRECATED_FIELDS:
                    fields_to_remove.append(key)
                    removed_fields.append(current_path)
                    fields_removed.append(current_path)

                    # Generate specific warning for this field
                    warning = self._generate_field_specific_warning(key, value, current_path)
                    if warning:
                        warnings_issued.append(warning)

                elif isinstance(value, dict):
                    # Recursively process nested sections
                    _remove_from_section(value, current_path)

            # Remove deprecated fields from this section
            for field in fields_to_remove:
                section_data.pop(field, None)
                self.logger.debug(f"Removed deprecated field: {section_path}.{field}" if section_path else field)

        # Process all sections of the configuration
        _remove_from_section(config_data)
        return removed_fields

    def _generate_field_specific_warning(self, field_name: str, field_value: Any, field_path: str) -> str:
        """Generate specific warning message for a deprecated field.

        Args:
            field_name: Name of the deprecated field
            field_value: Value of the deprecated field
            field_path: Full path to the field

        Returns:
            Warning message string
        """
        warnings = {
            "collection_prefix": (
                f"Deprecated field '{field_path}' with value '{field_value}' has been removed. "
                "Collection prefixes are now handled automatically by the CollectionType system. "
                "Use 'collections.project_suffixes' in the new schema to define collection types."
            ),
            "max_collections": (
                f"Deprecated field '{field_path}' with value '{field_value}' has been removed. "
                "Collection limits are now managed through the multi-tenant architecture. "
                "Use 'collections.global_collections' to configure shared collections."
            ),
            "recursive_depth": (
                f"Deprecated field '{field_path}' with value '{field_value}' has been removed. "
                "File traversal depth is now controlled by 'ingestion.max_depth' setting. "
                f"Consider setting 'ingestion.max_depth: {field_value}' in your new configuration."
            ),
            "enable_legacy_mode": (
                f"Deprecated field '{field_path}' with value '{field_value}' has been removed. "
                "Legacy mode is no longer supported. All functionality has been integrated into the current system."
            ),
            # Pattern-related deprecated fields
            "include_patterns": (
                f"Deprecated field '{field_path}' has been removed. "
                "Use 'patterns.custom_include_patterns' in the new schema."
            ),
            "exclude_patterns": (
                f"Deprecated field '{field_path}' has been removed. "
                "Use 'patterns.custom_exclude_patterns' in the new schema."
            ),
            "file_patterns": (
                f"Deprecated field '{field_path}' has been removed. "
                "Use 'patterns.custom_include_patterns' in the new schema."
            ),
            "ignore_patterns": (
                f"Deprecated field '{field_path}' has been removed. "
                "Use 'patterns.custom_exclude_patterns' in the new schema."
            ),
            "supported_extensions": (
                f"Deprecated field '{field_path}' has been removed. "
                "File extensions are now handled through 'patterns.custom_include_patterns' using glob patterns."
            ),
            "file_types": (
                f"Deprecated field '{field_path}' has been removed. "
                "File types are now handled through 'patterns.custom_include_patterns' using glob patterns."
            ),
            "pattern_priorities": (
                f"Deprecated field '{field_path}' has been removed. "
                "Pattern ordering is now handled automatically based on specificity."
            ),
            "custom_ecosystems": (
                f"Deprecated field '{field_path}' has been removed. "
                "Use 'patterns.custom_project_indicators' in the new schema."
            ),
        }

        return warnings.get(field_name,
            f"Deprecated field '{field_path}' with value '{field_value}' has been removed. "
            "Please consult the documentation for the equivalent setting in the new schema.")

    def _validate_functionality_preservation(self,
                                           original_config: Dict[str, Any],
                                           cleaned_config: Dict[str, Any],
                                           removed_fields: List[str]) -> Dict[str, Any]:
        """Validate that functionality is preserved after removing deprecated fields.

        Args:
            original_config: Original configuration before field removal
            cleaned_config: Configuration after deprecated fields removed
            removed_fields: List of removed field paths

        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "is_valid": True,
            "warnings": [],
            "recommendations": []
        }

        # Check for critical functionality that might be affected
        for field_path in removed_fields:
            field_name = field_path.split('.')[-1]

            # Collection functionality validation
            if field_name in ["collection_prefix", "max_collections"]:
                if not self._has_collection_config_replacement(cleaned_config):
                    validation_result["warnings"].append(
                        f"Removed {field_name} but no equivalent collection configuration found. "
                        "Consider adding 'collections.project_suffixes' setting."
                    )

            # Ingestion depth validation
            elif field_name == "recursive_depth":
                if not self._has_ingestion_depth_config(cleaned_config):
                    validation_result["recommendations"].append(
                        "Removed recursive_depth but no 'ingestion.max_depth' found. "
                        "Consider adding this setting to control file traversal depth."
                    )

            # Pattern functionality validation
            elif field_name in ["include_patterns", "exclude_patterns", "file_patterns", "ignore_patterns"]:
                if not self._has_pattern_config_replacement(cleaned_config):
                    validation_result["warnings"].append(
                        f"Removed {field_name} but no equivalent pattern configuration found. "
                        "Consider adding 'patterns.custom_include_patterns' or 'patterns.custom_exclude_patterns'."
                    )

        return validation_result

    def _has_collection_config_replacement(self, config: Dict[str, Any]) -> bool:
        """Check if configuration has replacement for collection settings."""
        collections = config.get("collections", {})
        return bool(
            collections.get("project_suffixes") or
            collections.get("global_collections") or
            config.get("workspace", {}).get("collection_types")
        )

    def _has_ingestion_depth_config(self, config: Dict[str, Any]) -> bool:
        """Check if configuration has replacement for recursive depth."""
        ingestion = config.get("ingestion", {})
        return "max_depth" in ingestion

    def _has_pattern_config_replacement(self, config: Dict[str, Any]) -> bool:
        """Check if configuration has replacement for pattern settings."""
        patterns = config.get("patterns", {})
        return bool(
            patterns.get("custom_include_patterns") or
            patterns.get("custom_exclude_patterns")
        )

    def _issue_deprecation_warnings(self, removed_fields: List[str], warnings: List[str]) -> None:
        """Issue comprehensive deprecation warnings to the user.

        Args:
            removed_fields: List of removed field paths
            warnings: List of specific warnings to issue
        """
        if not removed_fields:
            return

        self.logger.warning("=" * 80)
        self.logger.warning("DEPRECATED CONFIGURATION FIELDS REMOVED")
        self.logger.warning("=" * 80)

        self.logger.warning(
            f"The following {len(removed_fields)} deprecated configuration fields have been removed "
            "from your configuration:"
        )

        for field in removed_fields:
            self.logger.warning(f"  - {field}")

        self.logger.warning("\nSpecific migration guidance:")
        for warning in warnings:
            self.logger.warning(f"  • {warning}")

        self.logger.warning(
            "\nFor complete migration guidance, see the documentation at: "
            "https://github.com/your-repo/workspace-qdrant-mcp/docs/migration.md"
        )
        self.logger.warning("=" * 80)

    def _apply_fallback_handling(self,
                               config: Dict[str, Any],
                               removed_fields: List[str]) -> List[Dict[str, Any]]:
        """Apply fallback handling for critical deprecated fields.

        Args:
            config: Configuration to potentially modify
            removed_fields: List of removed field paths

        Returns:
            List of validation results for applied fallbacks
        """
        fallback_results = []

        for field_path in removed_fields:
            field_name = field_path.split('.')[-1]

            # Apply fallbacks for critical fields that need replacement
            if field_name == "recursive_depth" and not self._has_ingestion_depth_config(config):
                # Set a reasonable default for ingestion depth
                if "ingestion" not in config:
                    config["ingestion"] = {}

                # Use a sensible default depth
                config["ingestion"]["max_depth"] = 10
                self.logger.info(
                    "Applied fallback: Set 'ingestion.max_depth: 10' to replace removed 'recursive_depth'"
                )
                fallback_results.append({
                    "is_valid": True,
                    "applied": "ingestion.max_depth = 10"
                })

        # Also apply fallbacks for fields that don't have direct replacements but are needed
        for field_path in removed_fields:
            field_name = field_path.split('.')[-1]

            # Always add ingestion section if it doesn't exist and we removed a field that would need it
            if field_name in ["recursive_depth", "max_collections"] and not self._has_ingestion_depth_config(config):
                if "ingestion" not in config:
                    config["ingestion"] = {}
                if "max_depth" not in config["ingestion"]:
                    config["ingestion"]["max_depth"] = 10
                    self.logger.info("Applied fallback: Set 'ingestion.max_depth: 10' for removed deprecated field")
                    fallback_results.append({
                        "is_valid": True,
                        "applied": "ingestion.max_depth = 10"
                    })

            elif field_name in ["collection_prefix", "max_collections"] and not self._has_collection_config_replacement(config):
                # Set basic collection configuration
                if "collections" not in config:
                    config["collections"] = {}

                if "project_suffixes" not in config["collections"]:
                    config["collections"]["project_suffixes"] = ["scratchbook"]

                self.logger.info(
                    "Applied fallback: Set 'collections.project_suffixes: [\"scratchbook\"]' for removed collection settings"
                )
                fallback_results.append({
                    "is_valid": True,
                    "applied": "collections.project_suffixes = [\"scratchbook\"]"
                })

        return fallback_results

    def _validate_cleaned_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that the cleaned configuration is functional.

        Args:
            config: Cleaned configuration to validate

        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "is_valid": True,
            "error": None
        }

        # Basic structure validation
        if not isinstance(config, dict):
            validation_result["is_valid"] = False
            validation_result["error"] = "Configuration must be a dictionary"
            return validation_result

        # Check for at least some essential configuration
        essential_sections = ["qdrant", "embeddings"]
        has_essential = any(section in config for section in essential_sections)

        if not has_essential and config:  # Only check if config is not empty
            validation_result["is_valid"] = False
            validation_result["error"] = f"Configuration missing essential sections: {essential_sections}"
            return validation_result

        # Check that we haven't accidentally removed all configuration
        if not config:
            validation_result["is_valid"] = False
            validation_result["error"] = "Configuration is empty after removing deprecated fields"
            return validation_result

        # Validate that no deprecated fields remain
        remaining_deprecated = self._detect_deprecated_fields(config)
        if remaining_deprecated:
            # This should not happen, but check anyway
            self.logger.warning(f"Some deprecated fields still remain: {remaining_deprecated}")

        return validation_result

    def migrate_pattern_config(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate pattern configuration from legacy formats to custom pattern system.

        This method handles the migration of pattern-related configuration fields:
        1. include_patterns/file_patterns → custom_include_patterns
        2. exclude_patterns/ignore_patterns → custom_exclude_patterns
        3. file_types/supported_extensions → include patterns for extensions
        4. custom_ecosystems → custom_project_indicators
        5. pattern_priorities → preserved through pattern ordering
        6. Backward compatibility with existing pattern references
        7. Integration with PatternManager custom pattern system

        Args:
            config_data: Dictionary containing configuration data to migrate

        Returns:
            Dictionary containing migrated configuration with updated pattern settings

        Raises:
            ValueError: If migration encounters unrecoverable conflicts
        """
        if not isinstance(config_data, dict):
            self.logger.warning("Config data is not a dictionary, cannot migrate patterns")
            return config_data

        migrated_config = config_data.copy()
        migration_applied = False

        # Check if we have pattern configuration to migrate
        pattern_config = migrated_config.get("patterns", {})
        if not isinstance(pattern_config, dict):
            pattern_config = {}

        # Also check workspace section for legacy pattern fields
        workspace_config = migrated_config.get("workspace", {})
        if not isinstance(workspace_config, dict):
            workspace_config = {}

        # 1. Handle include_patterns/file_patterns → custom_include_patterns migration
        legacy_include_fields = ["include_patterns", "file_patterns"]
        custom_include_patterns = []

        for field in legacy_include_fields:
            # Check both pattern_config and workspace_config sections
            for source_config in [pattern_config, workspace_config]:
                if field in source_config:
                    patterns = source_config[field]
                    if isinstance(patterns, list):
                        validated_patterns = self._validate_and_convert_patterns(patterns, "include")
                        custom_include_patterns.extend(validated_patterns)
                        self.logger.info(f"Migrated {field} ({len(validated_patterns)} patterns) to custom_include_patterns")
                        migration_applied = True

                    # Remove deprecated field
                    source_config.pop(field, None)

        # 2. Handle exclude_patterns/ignore_patterns → custom_exclude_patterns migration
        legacy_exclude_fields = ["exclude_patterns", "ignore_patterns"]
        custom_exclude_patterns = []

        for field in legacy_exclude_fields:
            for source_config in [pattern_config, workspace_config]:
                if field in source_config:
                    patterns = source_config[field]
                    if isinstance(patterns, list):
                        validated_patterns = self._validate_and_convert_patterns(patterns, "exclude")
                        custom_exclude_patterns.extend(validated_patterns)
                        self.logger.info(f"Migrated {field} ({len(validated_patterns)} patterns) to custom_exclude_patterns")
                        migration_applied = True

                    # Remove deprecated field
                    source_config.pop(field, None)

        # 3. Handle file_types/supported_extensions → include patterns
        extension_fields = ["file_types", "supported_extensions"]
        for field in extension_fields:
            for source_config in [pattern_config, workspace_config]:
                if field in source_config:
                    extensions = source_config[field]
                    if isinstance(extensions, list):
                        extension_patterns = self._convert_extensions_to_patterns(extensions)
                        custom_include_patterns.extend(extension_patterns)
                        self.logger.info(f"Migrated {field} ({len(extension_patterns)} patterns) to custom_include_patterns")
                        migration_applied = True

                    # Remove deprecated field
                    source_config.pop(field, None)

        # 4. Handle custom_ecosystems → custom_project_indicators migration
        custom_project_indicators = {}
        if "custom_ecosystems" in pattern_config:
            ecosystems = pattern_config["custom_ecosystems"]
            if isinstance(ecosystems, dict):
                custom_project_indicators = self._convert_ecosystems_to_indicators(ecosystems)
                self.logger.info(f"Migrated custom_ecosystems ({len(custom_project_indicators)} ecosystems) to custom_project_indicators")
                migration_applied = True

            # Remove deprecated field
            pattern_config.pop("custom_ecosystems", None)

        # 5. Handle pattern_priorities → apply ordering
        if "pattern_priorities" in pattern_config:
            priorities = pattern_config["pattern_priorities"]
            if isinstance(priorities, dict):
                custom_include_patterns = self._apply_pattern_priorities(custom_include_patterns, priorities, "include")
                custom_exclude_patterns = self._apply_pattern_priorities(custom_exclude_patterns, priorities, "exclude")
                self.logger.info("Applied pattern priorities to migrated patterns")
                migration_applied = True

            # Remove deprecated field
            pattern_config.pop("pattern_priorities", None)

        # 6. Store migrated patterns in the configuration
        if migration_applied:
            if not pattern_config:
                pattern_config = {}

            if custom_include_patterns:
                pattern_config["custom_include_patterns"] = custom_include_patterns

            if custom_exclude_patterns:
                pattern_config["custom_exclude_patterns"] = custom_exclude_patterns

            if custom_project_indicators:
                pattern_config["custom_project_indicators"] = custom_project_indicators

            # Update the main config
            migrated_config["patterns"] = pattern_config

            # Clean up empty workspace pattern fields
            if workspace_config and not any(field in workspace_config for field in self.DEPRECATED_FIELDS):
                # Keep workspace config if it has non-deprecated fields
                migrated_config["workspace"] = workspace_config

            self.logger.info("Pattern configuration migration completed successfully")

        return migrated_config

    def _validate_and_convert_patterns(self, patterns: List[Any], pattern_type: str) -> List[str]:
        """Validate and convert legacy patterns to custom pattern format.

        Args:
            patterns: List of pattern definitions (strings or dicts)
            pattern_type: Type of patterns ("include" or "exclude")

        Returns:
            List of validated pattern strings
        """
        validated_patterns = []

        for pattern in patterns:
            if isinstance(pattern, str):
                # Simple string pattern - validate and add
                if self._is_valid_pattern(pattern):
                    validated_patterns.append(pattern)
                else:
                    self.logger.warning(f"Skipping invalid {pattern_type} pattern: {pattern}")
            elif isinstance(pattern, dict):
                # Structured pattern - extract pattern string
                pattern_str = pattern.get("pattern", "")
                if isinstance(pattern_str, str) and self._is_valid_pattern(pattern_str):
                    validated_patterns.append(pattern_str)
                else:
                    self.logger.warning(f"Skipping invalid structured {pattern_type} pattern: {pattern}")
            else:
                self.logger.warning(f"Skipping unsupported {pattern_type} pattern type: {type(pattern).__name__}")

        return validated_patterns

    def _convert_extensions_to_patterns(self, extensions: List[Any]) -> List[str]:
        """Convert file extensions to glob patterns.

        Args:
            extensions: List of file extensions

        Returns:
            List of glob patterns for the extensions
        """
        patterns = []

        for ext in extensions:
            if isinstance(ext, str):
                # Clean up extension format
                ext = ext.strip()
                if not ext:
                    continue

                # Ensure extension starts with dot
                if not ext.startswith("."):
                    ext = f".{ext}"

                # Convert to glob pattern
                pattern = f"**/*{ext}"
                patterns.append(pattern)

            else:
                self.logger.warning(f"Skipping non-string extension: {ext}")

        return patterns

    def _convert_ecosystems_to_indicators(self, ecosystems: Dict[str, Any]) -> Dict[str, Any]:
        """Convert custom ecosystems to project indicators format.

        Args:
            ecosystems: Dictionary of custom ecosystem definitions

        Returns:
            Dictionary of custom project indicators
        """
        indicators = {}

        for ecosystem_name, ecosystem_def in ecosystems.items():
            if not isinstance(ecosystem_def, dict):
                self.logger.warning(f"Skipping invalid ecosystem definition for {ecosystem_name}")
                continue

            # Convert ecosystem definition to indicator format
            indicator = {}

            # Handle required files
            if "files" in ecosystem_def:
                indicator["required_files"] = ecosystem_def["files"]
            elif "required_files" in ecosystem_def:
                indicator["required_files"] = ecosystem_def["required_files"]

            # Handle optional files
            if "optional_files" in ecosystem_def:
                indicator["optional_files"] = ecosystem_def["optional_files"]

            # Handle minimum optional files requirement
            if "min_optional" in ecosystem_def:
                indicator["min_optional_files"] = ecosystem_def["min_optional"]
            elif "min_optional_files" in ecosystem_def:
                indicator["min_optional_files"] = ecosystem_def["min_optional_files"]

            if indicator:  # Only add if we have valid indicators
                indicators[ecosystem_name] = indicator

        return indicators

    def _apply_pattern_priorities(self, patterns: List[str], priorities: Dict[str, Any], pattern_type: str) -> List[str]:
        """Apply priority settings to pattern ordering.

        Args:
            patterns: List of pattern strings
            priorities: Dictionary of priority settings
            pattern_type: Type of patterns ("include" or "exclude")

        Returns:
            List of patterns ordered by priority
        """
        # Get priority settings for this pattern type
        type_priorities = priorities.get(pattern_type, {})
        if not isinstance(type_priorities, dict):
            return patterns

        # Create pattern priority mapping
        pattern_priority_map = {}
        for pattern in patterns:
            # Default priority is 0
            priority = 0

            # Check if pattern has specific priority
            for priority_pattern, priority_value in type_priorities.items():
                if isinstance(priority_value, (int, float)) and self._patterns_match(pattern, priority_pattern):
                    priority = priority_value
                    break

            pattern_priority_map[pattern] = priority

        # Sort patterns by priority (higher priority first)
        sorted_patterns = sorted(patterns, key=lambda p: pattern_priority_map.get(p, 0), reverse=True)
        return sorted_patterns

    def _is_valid_pattern(self, pattern: str) -> bool:
        """Validate a glob pattern.

        Args:
            pattern: Pattern string to validate

        Returns:
            True if pattern is valid
        """
        if not isinstance(pattern, str) or not pattern.strip():
            return False

        # Basic pattern validation
        pattern = pattern.strip()

        # Check for invalid characters or patterns
        invalid_patterns = ["", ".", "..", "/", "\\"]
        if pattern in invalid_patterns:
            return False

        # Check for potentially dangerous patterns
        if pattern.startswith("/") or "\\" in pattern:
            return False

        return True

    def _patterns_match(self, pattern: str, priority_pattern: str) -> bool:
        """Check if a pattern matches a priority pattern.

        Args:
            pattern: Pattern to check
            priority_pattern: Priority pattern to match against

        Returns:
            True if patterns match
        """
        import fnmatch

        try:
            return fnmatch.fnmatch(pattern, priority_pattern)
        except Exception:
            # Fallback to simple string matching
            return pattern == priority_pattern

    # Backup and Rollback System

    @property
    def backup_dir(self) -> Path:
        """Get the backup directory path, creating it if necessary."""
        if self._backup_dir is None:
            if self.config_path:
                config_dir = Path(self.config_path).parent
            else:
                config_dir = Path.cwd() / ".workspace-qdrant"

            self._backup_dir = config_dir / "backups"

        self._ensure_backup_directory()
        return self._backup_dir

    def set_backup_dir(self, backup_dir: Path) -> None:
        """Set a custom backup directory (useful for testing).

        Args:
            backup_dir: Path to the backup directory
        """
        self._backup_dir = backup_dir
        self._ensure_backup_directory()

    def backup_config(self, config_data: Dict[str, Any],
                     description: str = "Pre-migration backup") -> str:
        """Create a versioned backup of the configuration with timestamp and checksum.

        Args:
            config_data: Configuration data to backup
            description: Description of this backup

        Returns:
            Backup ID (timestamp-based identifier)

        Raises:
            OSError: If backup creation fails
            ValueError: If config_data is invalid
        """
        if not isinstance(config_data, dict):
            raise ValueError("Config data must be a dictionary")

        # Generate backup metadata with microseconds to avoid collisions
        timestamp = datetime.now()
        version = self.detect_config_version(config_data).value
        backup_id = f"backup_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}_v{version.replace('_', '')}"

        try:
            # Create backup file
            backup_file = self.backup_dir / f"{backup_id}.json"

            # Calculate checksum before writing
            config_json = json.dumps(config_data, indent=2, sort_keys=True)
            checksum = hashlib.sha256(config_json.encode()).hexdigest()

            # Write backup file
            with backup_file.open('w', encoding='utf-8') as f:
                f.write(config_json)

            # Create backup metadata - use relative path for portability
            metadata = {
                "id": backup_id,
                "timestamp": timestamp.isoformat(),
                "version": version,
                "checksum": f"sha256:{checksum}",
                "file_path": str(backup_file.resolve()),  # Use resolved absolute path
                "file_size": backup_file.stat().st_size,
                "description": description,
                "config_path": self.config_path
            }

            # Update metadata file
            self._update_backup_metadata(metadata)

            self.logger.info(f"Created backup {backup_id} with checksum {checksum[:8]}...")
            return backup_id

        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")
            raise OSError(f"Backup creation failed: {e}") from e

    def rollback_config(self, backup_id: str) -> Dict[str, Any]:
        """Restore configuration from a specific backup after validation.

        Args:
            backup_id: ID of the backup to restore

        Returns:
            Restored configuration data

        Raises:
            ValueError: If backup_id is invalid or backup is corrupted
            OSError: If backup file cannot be read
        """
        # Get backup metadata
        backup_info = self.get_backup_info(backup_id)
        if not backup_info:
            raise ValueError(f"Backup {backup_id} not found")

        # Validate backup integrity
        if not self.validate_backup(backup_id):
            raise ValueError(f"Backup {backup_id} is corrupted and cannot be restored")

        try:
            # Create safety backup of current config before rollback
            if self.config_path and Path(self.config_path).exists():
                current_config = self._load_current_config()
                safety_backup_id = self.backup_config(
                    current_config,
                    f"Safety backup before rollback to {backup_id}"
                )
                self.logger.info(f"Created safety backup {safety_backup_id}")

            # Load backup configuration
            backup_file = Path(backup_info["file_path"])
            with backup_file.open('r', encoding='utf-8') as f:
                restored_config = json.load(f)

            # Write restored config to original location if config_path is set
            if self.config_path:
                self._write_config_file(restored_config)

            self.logger.info(f"Successfully restored configuration from backup {backup_id}")
            return restored_config

        except Exception as e:
            self.logger.error(f"Failed to rollback to {backup_id}: {e}")
            raise OSError(f"Rollback failed: {e}") from e

    def list_backups(self) -> List[Dict[str, Any]]:
        """List all available backups with their metadata.

        Returns:
            List of backup information dictionaries
        """
        metadata_file = self.backup_dir / "metadata.json"

        if not metadata_file.exists():
            return []

        try:
            with metadata_file.open('r', encoding='utf-8') as f:
                metadata = json.load(f)

            backups = metadata.get("backups", [])

            # Sort by timestamp (most recent first)
            backups.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

            # Verify backup files still exist
            valid_backups = []
            for backup in backups:
                backup_file = Path(backup.get("file_path", ""))
                if backup_file.exists():
                    valid_backups.append(backup)
                else:
                    self.logger.warning(f"Backup file missing: {backup_file}")

            # Log for debugging
            self.logger.debug(f"Found {len(backups)} backups in metadata, {len(valid_backups)} with existing files")

            return valid_backups

        except Exception as e:
            self.logger.error(f"Failed to list backups: {e}")
            return []

    def validate_backup(self, backup_id: str) -> bool:
        """Validate backup integrity using checksum verification.

        Args:
            backup_id: ID of the backup to validate

        Returns:
            True if backup is valid, False otherwise
        """
        backup_info = self.get_backup_info(backup_id)
        if not backup_info:
            self.logger.error(f"Backup {backup_id} not found")
            return False

        try:
            backup_file = Path(backup_info["file_path"])
            if not backup_file.exists():
                self.logger.error(f"Backup file missing: {backup_file}")
                return False

            # Read backup file and calculate checksum
            with backup_file.open('r', encoding='utf-8') as f:
                content = f.read()

            actual_checksum = hashlib.sha256(content.encode()).hexdigest()
            expected_checksum = backup_info["checksum"].replace("sha256:", "")

            if actual_checksum != expected_checksum:
                self.logger.error(f"Checksum mismatch for backup {backup_id}")
                self.logger.error(f"Expected: {expected_checksum}")
                self.logger.error(f"Actual: {actual_checksum}")
                return False

            # Try to parse JSON to ensure it's valid
            try:
                json.loads(content)
            except json.JSONDecodeError as e:
                self.logger.error(f"Invalid JSON in backup {backup_id}: {e}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Failed to validate backup {backup_id}: {e}")
            return False

    def get_backup_info(self, backup_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific backup.

        Args:
            backup_id: ID of the backup

        Returns:
            Backup information dictionary or None if not found
        """
        backups = self.list_backups()
        for backup in backups:
            if backup.get("id") == backup_id:
                return backup
        return None

    def cleanup_old_backups(self, keep_count: int = 10) -> int:
        """Remove old backups, keeping only the most recent ones.

        Args:
            keep_count: Number of recent backups to keep

        Returns:
            Number of backups removed
        """
        if keep_count <= 0:
            raise ValueError("keep_count must be positive")

        backups = self.list_backups()
        if len(backups) <= keep_count:
            return 0

        # Get backups to remove (oldest first)
        backups_to_remove = backups[keep_count:]
        removed_count = 0

        for backup in backups_to_remove:
            try:
                backup_file = Path(backup["file_path"])
                if backup_file.exists():
                    backup_file.unlink()
                    self.logger.info(f"Removed old backup: {backup['id']}")
                    removed_count += 1
                else:
                    self.logger.warning(f"Backup file already missing: {backup['id']}")
                    removed_count += 1
            except Exception as e:
                self.logger.error(f"Failed to remove backup {backup['id']}: {e}")

        # Update metadata file to remove deleted backups
        if removed_count > 0:
            self._remove_backups_from_metadata([b["id"] for b in backups_to_remove])

            # Verify cleanup worked
            updated_backups = self.list_backups()
            self.logger.info(f"After cleanup: {len(updated_backups)} backups remaining")

        return removed_count

    def migrate_with_backup(self, config_data: Dict[str, Any],
                          backup_description: str = "Pre-migration backup") -> Dict[str, Any]:
        """Perform migration with automatic backup creation and comprehensive reporting.

        Args:
            config_data: Configuration data to migrate
            backup_description: Description for the backup

        Returns:
            Migrated configuration data

        Raises:
            ValueError: If migration fails
            OSError: If backup creation fails
        """
        start_time = time.time()

        # Create backup before migration
        backup_id = self.backup_config(config_data, backup_description)
        self.logger.info(f"Created backup {backup_id} before migration")

        # Initialize notification system and start migration
        source_version = self.detect_config_version(config_data).value
        migration_id = str(uuid.uuid4())
        self.notification_system.notify_migration_started(migration_id, source_version, "v2_current")

        try:
            # Perform migration steps
            migrated_config = config_data.copy()
            migration_methods = []

            # Apply all migration steps
            original_config = migrated_config.copy()
            migrated_config = self.migrate_collection_config(migrated_config)
            if migrated_config != original_config:
                migration_methods.append("migrate_collection_config")

            original_config = migrated_config.copy()
            migrated_config = self.migrate_pattern_config(migrated_config)
            if migrated_config != original_config:
                migration_methods.append("migrate_pattern_config")

            original_config = migrated_config.copy()
            migrated_config = self.remove_deprecated_fields(migrated_config)
            if migrated_config != original_config:
                migration_methods.append("remove_deprecated_fields")

            # Validate the migrated configuration
            final_validation = self._validate_cleaned_config(migrated_config)
            if not final_validation["is_valid"]:
                raise ValueError(f"Migration validation failed: {final_validation['error']}")

            # Calculate migration duration
            migration_duration = time.time() - start_time

            # Generate comprehensive migration report
            migration_report = self.generate_migration_report(
                before_config=config_data,
                after_config=migrated_config,
                migration_methods=migration_methods,
                backup_id=backup_id,
                migration_duration=migration_duration
            )

            # Send success notifications
            self.notification_system.notify_migration_success(migration_report)

            # Notify about deprecated features if any were handled
            if migration_report.deprecated_fields_handled:
                self.notification_system.notify_deprecated_features(migration_report.deprecated_fields_handled)

            # Notify about warnings if any
            if migration_report.warnings:
                self.notification_system.notify_warnings(migration_report.warnings)

            # Notify about rollback availability
            self.notification_system.notify_rollback_available(backup_id, migration_report.backup_location or "Unknown")

            self.logger.info("Migration completed successfully")
            return migrated_config

        except Exception as e:
            # Calculate migration duration even for failed migrations
            migration_duration = time.time() - start_time

            # Generate failure report
            migration_report = MigrationReport(
                migration_id=migration_id,
                source_version=source_version,
                target_version="v2_current",
                config_file_path=self.config_path,
                backup_id=backup_id,
                success=False,
                migration_duration_seconds=migration_duration,
                errors=[str(e)]
            )

            if backup_id:
                backup_info = self.get_backup_info(backup_id)
                if backup_info:
                    migration_report.backup_location = backup_info.get("file_path")
                    migration_report.rollback_instructions = self._generate_rollback_instructions(backup_id)

            # Save the failure report
            self._save_migration_report(migration_report)

            # Send failure notifications
            self.notification_system.notify_migration_failure(migration_report)

            self.logger.error(f"Migration failed: {e}")
            self.logger.info(f"Configuration can be restored from backup: {backup_id}")
            raise

    def _ensure_backup_directory(self) -> None:
        """Ensure backup directory exists."""
        if self._backup_dir is None:
            return

        try:
            self._backup_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.logger.error(f"Failed to create backup directory {self._backup_dir}: {e}")
            raise

    def _update_backup_metadata(self, backup_metadata: Dict[str, Any]) -> None:
        """Update the backup metadata file with new backup information.

        Args:
            backup_metadata: Metadata for the new backup
        """
        metadata_file = self.backup_dir / "metadata.json"

        # Load existing metadata
        if metadata_file.exists():
            try:
                with metadata_file.open('r', encoding='utf-8') as f:
                    metadata = json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to read metadata file, creating new: {e}")
                metadata = {"backups": []}
        else:
            metadata = {"backups": []}

        # Add new backup
        metadata["backups"].append(backup_metadata)

        # Write updated metadata
        try:
            with metadata_file.open('w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to update metadata file: {e}")
            raise

    def _remove_backups_from_metadata(self, backup_ids: List[str]) -> None:
        """Remove backup entries from metadata file.

        Args:
            backup_ids: List of backup IDs to remove
        """
        metadata_file = self.backup_dir / "metadata.json"

        if not metadata_file.exists():
            self.logger.warning(f"Metadata file does not exist: {metadata_file}")
            return

        try:
            # Read current metadata
            with metadata_file.open('r', encoding='utf-8') as f:
                metadata = json.load(f)

            original_count = len(metadata.get("backups", []))
            self.logger.debug(f"Removing {len(backup_ids)} backups from {original_count} total")

            # Filter out removed backups
            metadata["backups"] = [
                backup for backup in metadata.get("backups", [])
                if backup.get("id") not in backup_ids
            ]

            final_count = len(metadata["backups"])
            self.logger.debug(f"After removal: {final_count} backups remain")

            # Write updated metadata
            with metadata_file.open('w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)

            self.logger.debug(f"Successfully updated metadata file with {final_count} backups")

        except Exception as e:
            self.logger.error(f"Failed to update metadata after cleanup: {e}")
            raise  # Re-raise to help with debugging

    def _load_current_config(self) -> Dict[str, Any]:
        """Load current configuration from config_path.

        Returns:
            Current configuration data

        Raises:
            OSError: If config file cannot be read
        """
        if not self.config_path:
            raise OSError("No config_path set for loading current configuration")

        config_file = Path(self.config_path)
        if not config_file.exists():
            raise OSError(f"Config file does not exist: {config_file}")

        try:
            with config_file.open('r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise OSError(f"Failed to load config from {config_file}: {e}") from e

    def _write_config_file(self, config_data: Dict[str, Any]) -> None:
        """Write configuration data to config_path.

        Args:
            config_data: Configuration data to write

        Raises:
            OSError: If config file cannot be written
        """
        if not self.config_path:
            raise OSError("No config_path set for writing configuration")

        config_file = Path(self.config_path)

        try:
            # Create parent directories if needed
            config_file.parent.mkdir(parents=True, exist_ok=True)

            # Write configuration
            with config_file.open('w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2)

            self.logger.debug(f"Written configuration to {config_file}")

        except Exception as e:
            raise OSError(f"Failed to write config to {config_file}: {e}") from e

    # Migration Reporting and History Management

    @property
    def migration_history_dir(self) -> Path:
        """Get the migration history directory path, creating it if necessary."""
        if self._migration_history_dir is None:
            if self.config_path:
                config_dir = Path(self.config_path).parent
            else:
                config_dir = Path.cwd() / ".workspace-qdrant"

            self._migration_history_dir = config_dir / "migration_history"

        self._ensure_migration_history_directory()
        return self._migration_history_dir

    def set_migration_history_dir(self, history_dir: Path) -> None:
        """Set a custom migration history directory (useful for testing).

        Args:
            history_dir: Path to the migration history directory
        """
        self._migration_history_dir = history_dir
        self._ensure_migration_history_directory()

    def generate_migration_report(self,
                                before_config: Dict[str, Any],
                                after_config: Dict[str, Any],
                                migration_methods: List[str] = None,
                                backup_id: str = None,
                                migration_duration: Optional[float] = None) -> MigrationReport:
        """Generate a comprehensive migration report with detailed change tracking.

        This method creates a detailed report of all configuration changes made during
        migration, including before/after comparisons, deprecated field mappings,
        validation results, and user-friendly recommendations.

        Args:
            before_config: Configuration before migration
            after_config: Configuration after migration
            migration_methods: List of migration methods that were applied
            backup_id: ID of backup created before migration
            migration_duration: Duration of migration in seconds

        Returns:
            MigrationReport with comprehensive change details

        Raises:
            ValueError: If input configurations are invalid
        """
        if not isinstance(before_config, dict) or not isinstance(after_config, dict):
            raise ValueError("Both before_config and after_config must be dictionaries")

        # Create migration report
        report = MigrationReport(
            source_version=self.detect_config_version(before_config).value,
            target_version=self.detect_config_version(after_config).value,
            config_file_path=self.config_path,
            backup_id=backup_id,
            migration_duration_seconds=migration_duration
        )

        if backup_id:
            backup_info = self.get_backup_info(backup_id)
            if backup_info:
                report.backup_location = backup_info.get("file_path")

        if migration_methods:
            report.migration_methods_used = migration_methods

        # Assess migration complexity
        deprecated_fields = self._detect_deprecated_fields(before_config)
        complexity = self._assess_migration_complexity(
            self.detect_config_version(before_config),
            deprecated_fields
        )
        report.complexity = complexity.value

        try:
            # Generate detailed change analysis
            self._analyze_configuration_changes(before_config, after_config, report)

            # Analyze deprecated field handling
            self._analyze_deprecated_field_handling(before_config, after_config, report)

            # Perform validation analysis
            self._analyze_migration_validation(before_config, after_config, report)

            # Generate rollback instructions
            if backup_id:
                report.rollback_instructions = self._generate_rollback_instructions(backup_id)

            # Store the report in migration history
            self._save_migration_report(report)

            self.logger.info(f"Generated comprehensive migration report: {report.migration_id}")
            return report

        except Exception as e:
            report.add_error(f"Failed to generate complete migration report: {e}")
            self.logger.error(f"Error generating migration report: {e}")
            return report

    def _analyze_configuration_changes(self, before: Dict[str, Any], after: Dict[str, Any], report: MigrationReport) -> None:
        """Analyze and track all configuration changes.

        Args:
            before: Configuration before migration
            after: Configuration after migration
            report: Migration report to update
        """
        def _compare_nested_dict(before_dict: Dict[str, Any], after_dict: Dict[str, Any],
                               path: str = "", section: str = "configuration") -> None:
            """Recursively compare nested dictionaries and track changes."""
            # Check for removed fields
            for key, value in before_dict.items():
                current_path = f"{path}.{key}" if path else key

                if key not in after_dict:
                    # Field was removed
                    change = ChangeEntry(
                        change_type=ChangeType.DEPRECATED_REMOVED if key in self.DEPRECATED_FIELDS else ChangeType.REMOVED,
                        field_path=current_path,
                        old_value=value,
                        reason="Field removed during migration" if key not in self.DEPRECATED_FIELDS else "Deprecated field removed",
                        section=section
                    )
                    report.add_change(change)
                elif isinstance(value, dict) and isinstance(after_dict[key], dict):
                    # Recurse into nested dictionaries
                    _compare_nested_dict(value, after_dict[key], current_path, section)
                elif value != after_dict[key]:
                    # Field value changed
                    change = ChangeEntry(
                        change_type=ChangeType.MODIFIED,
                        field_path=current_path,
                        old_value=value,
                        new_value=after_dict[key],
                        reason="Value updated during migration",
                        section=section
                    )
                    report.add_change(change)

            # Check for added fields
            for key, value in after_dict.items():
                current_path = f"{path}.{key}" if path else key

                if key not in before_dict:
                    # Field was added
                    change = ChangeEntry(
                        change_type=ChangeType.ADDED,
                        field_path=current_path,
                        new_value=value,
                        reason="Field added during migration",
                        section=section
                    )
                    report.add_change(change)

        # Analyze changes in each top-level section
        all_sections = set(before.keys()) | set(after.keys())
        for section in all_sections:
            before_section = before.get(section, {})
            after_section = after.get(section, {})

            if isinstance(before_section, dict) and isinstance(after_section, dict):
                _compare_nested_dict(before_section, after_section, "", section)
            elif before_section != after_section:
                # Entire section changed
                change = ChangeEntry(
                    change_type=ChangeType.MODIFIED,
                    field_path=section,
                    old_value=before_section,
                    new_value=after_section,
                    reason="Section updated during migration",
                    section=section
                )
                report.add_change(change)

    def _analyze_deprecated_field_handling(self, before: Dict[str, Any], after: Dict[str, Any], report: MigrationReport) -> None:
        """Analyze how deprecated fields were handled.

        Args:
            before: Configuration before migration
            after: Configuration after migration
            report: Migration report to update
        """
        deprecated_found = self._detect_deprecated_fields(before)

        for field_path in deprecated_found:
            # Determine replacement information
            field_name = field_path.split('.')[-1]
            replacement_info = self._get_field_replacement_info(field_name)

            if replacement_info:
                report.deprecated_fields_handled[field_path] = replacement_info
            else:
                report.deprecated_fields_handled[field_path] = "Removed (no direct replacement)"

    def _analyze_migration_validation(self, before: Dict[str, Any], after: Dict[str, Any], report: MigrationReport) -> None:
        """Perform validation analysis on the migration.

        Args:
            before: Configuration before migration
            after: Configuration after migration
            report: Migration report to update
        """
        # Validate the final configuration
        final_validation = self._validate_cleaned_config(after)
        validation_result = ValidationResult(
            is_valid=final_validation["is_valid"],
            errors=[final_validation["error"]] if final_validation["error"] else []
        )
        report.add_validation_result(validation_result)

        # Check functionality preservation
        removed_fields = [change.field_path for change in report.changes_made
                         if change.change_type in [ChangeType.REMOVED, ChangeType.DEPRECATED_REMOVED]]

        if removed_fields:
            preservation_check = self._validate_functionality_preservation(before, after, removed_fields)
            preservation_result = ValidationResult(
                is_valid=preservation_check["is_valid"],
                warnings=preservation_check.get("warnings", []),
                recommendations=preservation_check.get("recommendations", [])
            )
            report.add_validation_result(preservation_result)

            # Add warnings from preservation check to main report
            for warning in preservation_check.get("warnings", []):
                report.add_warning(warning)

    def _get_field_replacement_info(self, field_name: str) -> str:
        """Get replacement information for a deprecated field.

        Args:
            field_name: Name of the deprecated field

        Returns:
            String describing the replacement
        """
        replacements = {
            "collection_prefix": "collections.project_suffixes (automatic collection naming)",
            "max_collections": "Multi-tenant architecture with metadata filtering",
            "recursive_depth": "ingestion.max_depth",
            "enable_legacy_mode": "Integrated into current system (no replacement needed)",
            "include_patterns": "patterns.custom_include_patterns",
            "exclude_patterns": "patterns.custom_exclude_patterns",
            "file_patterns": "patterns.custom_include_patterns",
            "ignore_patterns": "patterns.custom_exclude_patterns",
            "supported_extensions": "patterns.custom_include_patterns (as glob patterns)",
            "file_types": "patterns.custom_include_patterns (as glob patterns)",
            "pattern_priorities": "Automatic pattern ordering by specificity",
            "custom_ecosystems": "patterns.custom_project_indicators",
        }
        return replacements.get(field_name, "See documentation for replacement")

    def _generate_rollback_instructions(self, backup_id: str) -> str:
        """Generate rollback instructions for the migration.

        Args:
            backup_id: ID of the backup that can be used for rollback

        Returns:
            String with rollback instructions
        """
        instructions = [
            "To rollback this migration:",
            f"1. Use command: wqm admin rollback-config {backup_id}",
            f"2. Verify rollback: wqm admin validate-config",
            f"3. Restart services if needed",
            "",
            f"Backup validation: wqm admin validate-backup {backup_id}",
            f"View backup details: wqm admin backup-info {backup_id}"
        ]
        return "\n".join(instructions)

    def _save_migration_report(self, report: MigrationReport) -> None:
        """Save migration report to history.

        Args:
            report: Migration report to save
        """
        try:
            # Save individual report file
            report_file = self.migration_history_dir / f"{report.migration_id}.json"
            with report_file.open('w', encoding='utf-8') as f:
                json.dump(report.to_dict(), f, indent=2)

            # Update history index
            self._update_migration_history_index(report)

            self.logger.debug(f"Saved migration report to {report_file}")

        except Exception as e:
            self.logger.error(f"Failed to save migration report: {e}")

    def _update_migration_history_index(self, report: MigrationReport) -> None:
        """Update the migration history index with new report.

        Args:
            report: Migration report to add to index
        """
        index_file = self.migration_history_dir / "index.json"

        # Load existing index
        if index_file.exists():
            try:
                with index_file.open('r', encoding='utf-8') as f:
                    index = json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to read history index, creating new: {e}")
                index = {"migrations": []}
        else:
            index = {"migrations": []}

        # Add new migration entry
        index_entry = {
            "migration_id": report.migration_id,
            "timestamp": report.timestamp,
            "source_version": report.source_version,
            "target_version": report.target_version,
            "success": report.success,
            "changes_count": len(report.changes_made),
            "backup_id": report.backup_id,
            "config_file_path": report.config_file_path,
            "report_file": f"{report.migration_id}.json"
        }

        index["migrations"].append(index_entry)

        # Sort by timestamp (most recent first)
        index["migrations"].sort(key=lambda x: x["timestamp"], reverse=True)

        # Write updated index
        try:
            with index_file.open('w', encoding='utf-8') as f:
                json.dump(index, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to update migration history index: {e}")

    def get_migration_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get migration history with optional limit.

        Args:
            limit: Maximum number of migration records to return

        Returns:
            List of migration history entries
        """
        index_file = self.migration_history_dir / "index.json"

        if not index_file.exists():
            return []

        try:
            with index_file.open('r', encoding='utf-8') as f:
                index = json.load(f)

            migrations = index.get("migrations", [])

            # Apply limit if specified
            if limit is not None and limit > 0:
                migrations = migrations[:limit]

            return migrations

        except Exception as e:
            self.logger.error(f"Failed to get migration history: {e}")
            return []

    def get_migration_report(self, migration_id: str) -> Optional[MigrationReport]:
        """Get a specific migration report by ID.

        Args:
            migration_id: ID of the migration report to retrieve

        Returns:
            MigrationReport if found, None otherwise
        """
        report_file = self.migration_history_dir / f"{migration_id}.json"

        if not report_file.exists():
            return None

        try:
            with report_file.open('r', encoding='utf-8') as f:
                report_data = json.load(f)

            # Convert back to MigrationReport object
            return self._dict_to_migration_report(report_data)

        except Exception as e:
            self.logger.error(f"Failed to load migration report {migration_id}: {e}")
            return None

    def get_latest_migration_report(self) -> Optional[MigrationReport]:
        """Get the most recent migration report.

        Returns:
            Most recent MigrationReport if available, None otherwise
        """
        history = self.get_migration_history(limit=1)
        if history:
            return self.get_migration_report(history[0]["migration_id"])
        return None

    def search_migration_history(self,
                                source_version: Optional[str] = None,
                                target_version: Optional[str] = None,
                                success_only: Optional[bool] = None,
                                days_back: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search migration history with filters.

        Args:
            source_version: Filter by source version
            target_version: Filter by target version
            success_only: If True, only return successful migrations
            days_back: Only return migrations from the last N days

        Returns:
            List of filtered migration history entries
        """
        all_migrations = self.get_migration_history()

        # Apply filters
        filtered_migrations = []
        cutoff_time = None

        if days_back:
            from datetime import timedelta
            cutoff_time = datetime.now() - timedelta(days=days_back)

        for migration in all_migrations:
            # Check time filter
            if cutoff_time:
                try:
                    migration_time = datetime.fromisoformat(migration["timestamp"])
                    if migration_time < cutoff_time:
                        continue
                except (ValueError, KeyError):
                    continue

            # Check version filters
            if source_version and migration.get("source_version") != source_version:
                continue
            if target_version and migration.get("target_version") != target_version:
                continue

            # Check success filter
            if success_only is not None and migration.get("success") != success_only:
                continue

            filtered_migrations.append(migration)

        return filtered_migrations

    def cleanup_old_migration_reports(self, keep_count: int = 50) -> int:
        """Remove old migration reports, keeping only the most recent ones.

        Args:
            keep_count: Number of recent migration reports to keep

        Returns:
            Number of reports removed
        """
        if keep_count <= 0:
            raise ValueError("keep_count must be positive")

        history = self.get_migration_history()
        if len(history) <= keep_count:
            return 0

        # Get reports to remove (oldest first)
        reports_to_remove = history[keep_count:]
        removed_count = 0

        for migration in reports_to_remove:
            try:
                migration_id = migration["migration_id"]
                report_file = self.migration_history_dir / f"{migration_id}.json"

                if report_file.exists():
                    report_file.unlink()
                    self.logger.debug(f"Removed old migration report: {migration_id}")
                    removed_count += 1

            except Exception as e:
                self.logger.error(f"Failed to remove migration report {migration.get('migration_id', 'unknown')}: {e}")

        # Update history index to remove deleted reports
        if removed_count > 0:
            self._rebuild_migration_history_index()

        return removed_count

    def _rebuild_migration_history_index(self) -> None:
        """Rebuild migration history index from existing report files."""
        try:
            # Find all report files
            report_files = list(self.migration_history_dir.glob("*.json"))
            report_files = [f for f in report_files if f.name != "index.json"]

            migrations = []
            for report_file in report_files:
                try:
                    with report_file.open('r', encoding='utf-8') as f:
                        report_data = json.load(f)

                    # Create index entry
                    index_entry = {
                        "migration_id": report_data["migration_id"],
                        "timestamp": report_data["timestamp"],
                        "source_version": report_data["source_version"],
                        "target_version": report_data["target_version"],
                        "success": report_data["success"],
                        "changes_count": len(report_data.get("changes_made", [])),
                        "backup_id": report_data.get("backup_id"),
                        "config_file_path": report_data.get("config_file_path"),
                        "report_file": report_file.name
                    }
                    migrations.append(index_entry)

                except Exception as e:
                    self.logger.warning(f"Failed to process report file {report_file}: {e}")

            # Sort by timestamp
            migrations.sort(key=lambda x: x["timestamp"], reverse=True)

            # Write new index
            index_file = self.migration_history_dir / "index.json"
            with index_file.open('w', encoding='utf-8') as f:
                json.dump({"migrations": migrations}, f, indent=2)

            self.logger.debug(f"Rebuilt migration history index with {len(migrations)} reports")

        except Exception as e:
            self.logger.error(f"Failed to rebuild migration history index: {e}")

    def _dict_to_migration_report(self, data: Dict[str, Any]) -> MigrationReport:
        """Convert dictionary back to MigrationReport object.

        Args:
            data: Dictionary containing migration report data

        Returns:
            MigrationReport object
        """
        # Create base report
        report = MigrationReport(
            migration_id=data.get("migration_id", str(uuid.uuid4())),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            source_version=data.get("source_version", "unknown"),
            target_version=data.get("target_version", "v2_current"),
            config_file_path=data.get("config_file_path"),
            backup_id=data.get("backup_id"),
            backup_location=data.get("backup_location"),
            success=data.get("success", True),
            migration_duration_seconds=data.get("migration_duration_seconds"),
            warnings=data.get("warnings", []),
            errors=data.get("errors", []),
            migration_methods_used=data.get("migration_methods_used", []),
            complexity=data.get("complexity"),
            rollback_instructions=data.get("rollback_instructions"),
            deprecated_fields_handled=data.get("deprecated_fields_handled", {})
        )

        # Convert changes_made
        for change_data in data.get("changes_made", []):
            change_type_str = change_data.get("change_type", "modified")
            try:
                change_type = ChangeType(change_type_str)
            except ValueError:
                change_type = ChangeType.MODIFIED

            change = ChangeEntry(
                change_type=change_type,
                field_path=change_data.get("field_path", ""),
                old_value=change_data.get("old_value"),
                new_value=change_data.get("new_value"),
                reason=change_data.get("reason"),
                section=change_data.get("section"),
                migration_method=change_data.get("migration_method"),
                timestamp=change_data.get("timestamp")
            )
            report.add_change(change)

        # Convert validation_results
        for result_data in data.get("validation_results", []):
            validation_result = ValidationResult(
                is_valid=result_data.get("is_valid", True),
                warnings=result_data.get("warnings", []),
                errors=result_data.get("errors", []),
                recommendations=result_data.get("recommendations", [])
            )
            report.add_validation_result(validation_result)

        return report

    def _ensure_migration_history_directory(self) -> None:
        """Ensure migration history directory exists."""
        if self._migration_history_dir is None:
            return

        try:
            self._migration_history_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.logger.error(f"Failed to create migration history directory {self._migration_history_dir}: {e}")
            raise