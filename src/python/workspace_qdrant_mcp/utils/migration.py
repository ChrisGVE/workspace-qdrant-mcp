"""Configuration migration utilities for workspace-qdrant-mcp.

This module provides the ConfigMigrator class for detecting configuration versions,
identifying deprecated fields, and determining migration paths between different
configuration schema versions.
"""

import logging
from typing import Dict, List, Optional, Any, Set
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
        "enable_legacy_mode"  # Additional deprecated field
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

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize ConfigMigrator.

        Args:
            logger: Optional logger instance. If None, creates a default logger.
        """
        self.logger = logger or logging.getLogger(__name__)

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
                    "Remove deprecated fields from configuration",
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