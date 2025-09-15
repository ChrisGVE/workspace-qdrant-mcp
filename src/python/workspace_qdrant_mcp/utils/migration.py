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