"""
Asset Configuration Validation System for workspace-qdrant-mcp.

This module provides comprehensive validation and management for asset configurations
across multiple formats (YAML, JSON, TOML). It handles configuration loading, parsing,
schema validation, inheritance, environment variable substitution, asset discovery,
versioning, caching, security validation, and hot-reloading capabilities.

Features:
    - Multi-format configuration support (YAML, JSON, TOML)
    - Schema validation and compliance checking
    - Configuration inheritance and override resolution
    - Environment variable substitution with validation
    - Asset discovery and loading mechanisms
    - Asset versioning and compatibility checks
    - Asset caching with performance optimization
    - Asset security and integrity validation
    - Hot-reloading capabilities for runtime updates
    - Cross-component configuration consistency
    - Comprehensive error handling and reporting
"""

import hashlib
import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError, validator
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # Fallback for older Python versions
    except ImportError:
        tomllib = None

from loguru import logger


class ConfigFormat(Enum):
    """Supported configuration file formats."""
    YAML = "yaml"
    JSON = "json"
    TOML = "toml"


class AssetStatus(Enum):
    """Asset status enumeration."""
    VALID = "valid"
    INVALID = "invalid"
    MISSING = "missing"
    CORRUPTED = "corrupted"
    VERSION_MISMATCH = "version_mismatch"
    SECURITY_VIOLATION = "security_violation"


@dataclass
class AssetMetadata:
    """Metadata for asset configuration files."""
    path: Path
    format: ConfigFormat
    version: str = "1.0.0"
    checksum: str = ""
    last_modified: datetime = field(default_factory=datetime.now)
    size_bytes: int = 0
    permissions: int = 0
    status: AssetStatus = AssetStatus.VALID
    dependencies: list[str] = field(default_factory=list)
    tags: set[str] = field(default_factory=set)


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: AssetMetadata | None = None
    validated_config: dict[str, Any] | None = None


class AssetConfigSchema(BaseModel):
    """Base schema for asset configuration validation."""

    # Core configuration fields
    name: str = Field(..., description="Asset configuration name")
    version: str = Field(default="1.0.0", description="Configuration version")
    description: str | None = Field(None, description="Asset description")

    # Asset management fields
    enabled: bool = Field(default=True, description="Whether asset is enabled")
    priority: int = Field(default=0, description="Asset loading priority")
    dependencies: list[str] = Field(default_factory=list, description="Asset dependencies")

    # Security and validation fields
    checksum_algorithm: str = Field(default="sha256", description="Checksum algorithm")
    required_permissions: int | None = Field(None, description="Required file permissions")
    max_file_size: int = Field(default=100*1024*1024, description="Maximum file size in bytes")

    # Environment variable substitution
    allow_env_substitution: bool = Field(default=True, description="Allow environment variable substitution")
    required_env_vars: list[str] = Field(default_factory=list, description="Required environment variables")

    @validator('version')
    def validate_version(cls, v):
        """Validate semantic version format."""
        if not isinstance(v, str):
            raise ValueError("Version must be a string")

        parts = v.split('.')
        if len(parts) != 3:
            raise ValueError("Version must be in format X.Y.Z")

        for part in parts:
            if not part.isdigit():
                raise ValueError("Version parts must be numeric")

        return v

    @validator('priority')
    def validate_priority(cls, v):
        """Validate priority range."""
        if not isinstance(v, int):
            raise ValueError("Priority must be an integer")

        if v < 0 or v > 1000:
            raise ValueError("Priority must be between 0 and 1000")

        return v


class AssetConfigValidator:
    """Comprehensive asset configuration validator with multi-format support."""

    def __init__(self,
                 asset_directories: list[Path] = None,
                 cache_enabled: bool = True,
                 hot_reload_enabled: bool = False):
        """Initialize the asset configuration validator.

        Args:
            asset_directories: Directories to search for asset configurations
            cache_enabled: Whether to enable configuration caching
            hot_reload_enabled: Whether to enable hot-reloading of configurations
        """
        self.asset_directories = asset_directories or []
        self.cache_enabled = cache_enabled
        self.hot_reload_enabled = hot_reload_enabled

        # Internal state
        self._cache: dict[str, tuple[AssetMetadata, dict[str, Any]]] = {}
        self._cache_lock = threading.RLock()
        self._observers: list[Observer] = []
        self._loaded_assets: dict[str, AssetMetadata] = {}

        # Supported file extensions by format
        self._format_extensions = {
            ConfigFormat.YAML: {'.yaml', '.yml'},
            ConfigFormat.JSON: {'.json'},
            ConfigFormat.TOML: {'.toml', '.tml'}
        }

        # Initialize hot-reloading if enabled
        if self.hot_reload_enabled:
            self._setup_hot_reload()

    def discover_assets(self, directory: Path = None) -> list[AssetMetadata]:
        """Discover asset configuration files in specified directories.

        Args:
            directory: Specific directory to search, or None for all configured directories

        Returns:
            List of discovered asset metadata
        """
        discovered_assets = []
        search_dirs = [directory] if directory else self.asset_directories

        for search_dir in search_dirs:
            if not search_dir.exists() or not search_dir.is_dir():
                logger.warning(f"Asset directory not found or not a directory: {search_dir}")
                continue

            try:
                for file_path in search_dir.rglob('*'):
                    if file_path.is_file() and self._is_config_file(file_path):
                        asset_metadata = self._create_asset_metadata(file_path)
                        if asset_metadata:
                            discovered_assets.append(asset_metadata)
            except Exception as e:
                logger.error(f"Error discovering assets in {search_dir}: {e}")

        logger.info(f"Discovered {len(discovered_assets)} asset configuration files")
        return discovered_assets

    def validate_config_file(self,
                           file_path: Path,
                           schema: BaseModel | None = None,
                           enable_env_substitution: bool = True) -> ValidationResult:
        """Validate a single configuration file.

        Args:
            file_path: Path to configuration file
            schema: Optional Pydantic schema for validation
            enable_env_substitution: Whether to perform environment variable substitution

        Returns:
            ValidationResult with validation status and details
        """
        try:
            # Check if file exists and is readable
            if not file_path.exists():
                return ValidationResult(
                    is_valid=False,
                    errors=[f"Configuration file not found: {file_path}"]
                )

            if not file_path.is_file():
                return ValidationResult(
                    is_valid=False,
                    errors=[f"Path is not a file: {file_path}"]
                )

            # Create asset metadata
            asset_metadata = self._create_asset_metadata(file_path)
            if not asset_metadata:
                return ValidationResult(
                    is_valid=False,
                    errors=[f"Could not create metadata for: {file_path}"]
                )

            # Security validation
            security_result = self._validate_security(file_path, asset_metadata)
            if not security_result.is_valid:
                return security_result

            # Load and parse configuration
            config_data = self._load_config_file(file_path, asset_metadata.format)
            if config_data is None:
                return ValidationResult(
                    is_valid=False,
                    errors=[f"Failed to load configuration from: {file_path}"],
                    metadata=asset_metadata
                )

            # Environment variable substitution
            if enable_env_substitution:
                config_data = self._substitute_environment_variables(config_data)

            # Schema validation if provided
            validation_errors = []
            validation_warnings = []

            if schema:
                try:
                    validated_config = schema(**config_data)
                    config_data = validated_config.dict()
                except ValidationError as e:
                    validation_errors.extend([str(error) for error in e.errors])

            # Additional validation checks
            additional_validation = self._perform_additional_validation(config_data, asset_metadata)
            validation_errors.extend(additional_validation.errors)
            validation_warnings.extend(additional_validation.warnings)

            # Cache the validated configuration
            if self.cache_enabled and not validation_errors:
                self._cache_configuration(str(file_path), asset_metadata, config_data)

            return ValidationResult(
                is_valid=len(validation_errors) == 0,
                errors=validation_errors,
                warnings=validation_warnings,
                metadata=asset_metadata,
                validated_config=config_data if len(validation_errors) == 0 else None
            )

        except Exception as e:
            logger.error(f"Unexpected error validating {file_path}: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Unexpected validation error: {str(e)}"]
            )

    def validate_multiple_configs(self,
                                file_paths: list[Path],
                                schema: BaseModel | None = None,
                                fail_fast: bool = False) -> dict[str, ValidationResult]:
        """Validate multiple configuration files.

        Args:
            file_paths: List of configuration file paths
            schema: Optional Pydantic schema for validation
            fail_fast: Whether to stop on first validation failure

        Returns:
            Dictionary mapping file paths to validation results
        """
        results = {}

        for file_path in file_paths:
            try:
                result = self.validate_config_file(file_path, schema)
                results[str(file_path)] = result

                if fail_fast and not result.is_valid:
                    logger.warning(f"Stopping validation due to failure in: {file_path}")
                    break

            except Exception as e:
                logger.error(f"Error validating {file_path}: {e}")
                results[str(file_path)] = ValidationResult(
                    is_valid=False,
                    errors=[f"Validation error: {str(e)}"]
                )

                if fail_fast:
                    break

        return results

    def validate_configuration_inheritance(self,
                                         base_config: Path,
                                         override_configs: list[Path]) -> ValidationResult:
        """Validate configuration inheritance and overrides.

        Args:
            base_config: Base configuration file
            override_configs: List of override configuration files

        Returns:
            ValidationResult with merged and validated configuration
        """
        try:
            # Load base configuration
            base_result = self.validate_config_file(base_config)
            if not base_result.is_valid:
                return ValidationResult(
                    is_valid=False,
                    errors=[f"Base configuration is invalid: {base_config}"] + base_result.errors
                )

            merged_config = base_result.validated_config.copy()
            all_errors = []
            all_warnings = []

            # Apply overrides in order
            for override_config in override_configs:
                override_result = self.validate_config_file(override_config)

                if not override_result.is_valid:
                    error_msg = f"Override configuration is invalid: {override_config}"
                    all_errors.append(error_msg)
                    all_errors.extend(override_result.errors)
                    continue

                # Merge configurations
                merged_config = self._deep_merge_configs(merged_config, override_result.validated_config)
                all_warnings.extend(override_result.warnings)

            # Validate final merged configuration
            final_validation = self._validate_merged_config(merged_config)
            all_errors.extend(final_validation.errors)
            all_warnings.extend(final_validation.warnings)

            return ValidationResult(
                is_valid=len(all_errors) == 0,
                errors=all_errors,
                warnings=all_warnings,
                validated_config=merged_config if len(all_errors) == 0 else None
            )

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Error validating configuration inheritance: {str(e)}"]
            )

    def check_asset_integrity(self, file_path: Path) -> ValidationResult:
        """Check asset integrity using checksums and security validation.

        Args:
            file_path: Path to asset file

        Returns:
            ValidationResult with integrity check results
        """
        try:
            if not file_path.exists():
                return ValidationResult(
                    is_valid=False,
                    errors=[f"Asset file not found: {file_path}"]
                )

            # Calculate current checksum
            current_checksum = self._calculate_checksum(file_path)

            # Get cached metadata if available
            cached_metadata = self._get_cached_metadata(str(file_path))

            errors = []
            warnings = []

            # Compare checksums if we have cached metadata
            if cached_metadata and cached_metadata.checksum:
                if current_checksum != cached_metadata.checksum:
                    errors.append(f"Checksum mismatch for {file_path}")
                    errors.append(f"Expected: {cached_metadata.checksum}")
                    errors.append(f"Actual: {current_checksum}")

            # Check file permissions
            file_stat = file_path.stat()
            current_permissions = file_stat.st_mode & 0o777

            # Security checks
            if current_permissions & 0o002:  # World-writable
                warnings.append(f"Asset file is world-writable: {file_path}")

            if current_permissions & 0o004 == 0:  # Not world-readable
                warnings.append(f"Asset file is not world-readable: {file_path}")

            # Size validation
            file_size = file_stat.st_size
            max_size = 100 * 1024 * 1024  # 100MB default limit

            if file_size > max_size:
                errors.append(f"Asset file exceeds maximum size limit: {file_path}")
                errors.append(f"Size: {file_size} bytes, Limit: {max_size} bytes")

            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                metadata=AssetMetadata(
                    path=file_path,
                    format=self._detect_format(file_path),
                    checksum=current_checksum,
                    size_bytes=file_size,
                    permissions=current_permissions,
                    last_modified=datetime.fromtimestamp(file_stat.st_mtime)
                )
            )

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Error checking asset integrity: {str(e)}"]
            )

    def get_cached_config(self, file_path: str) -> dict[str, Any] | None:
        """Get cached configuration if available and valid.

        Args:
            file_path: Path to configuration file

        Returns:
            Cached configuration data or None if not available/invalid
        """
        if not self.cache_enabled:
            return None

        with self._cache_lock:
            cached_entry = self._cache.get(file_path)
            if not cached_entry:
                return None

            metadata, config_data = cached_entry

            # Check if file has been modified since caching
            try:
                file_path_obj = Path(file_path)
                if file_path_obj.exists():
                    current_mtime = datetime.fromtimestamp(file_path_obj.stat().st_mtime)
                    if current_mtime > metadata.last_modified:
                        # File has been modified, invalidate cache
                        del self._cache[file_path]
                        return None

                return config_data

            except Exception as e:
                logger.warning(f"Error checking cache validity for {file_path}: {e}")
                # Remove potentially corrupted cache entry
                if file_path in self._cache:
                    del self._cache[file_path]
                return None

    def clear_cache(self, file_path: str | None = None):
        """Clear configuration cache.

        Args:
            file_path: Specific file path to clear, or None to clear all
        """
        with self._cache_lock:
            if file_path:
                self._cache.pop(file_path, None)
                logger.debug(f"Cleared cache for: {file_path}")
            else:
                self._cache.clear()
                logger.debug("Cleared entire configuration cache")

    def enable_hot_reload(self):
        """Enable hot-reloading of configuration files."""
        if not self.hot_reload_enabled:
            self.hot_reload_enabled = True
            self._setup_hot_reload()
            logger.info("Hot-reload enabled for asset configurations")

    def disable_hot_reload(self):
        """Disable hot-reloading and cleanup observers."""
        if self.hot_reload_enabled:
            self.hot_reload_enabled = False
            self._cleanup_hot_reload()
            logger.info("Hot-reload disabled for asset configurations")

    # Private methods

    def _is_config_file(self, file_path: Path) -> bool:
        """Check if file is a supported configuration file."""
        suffix = file_path.suffix.lower()
        return any(suffix in extensions for extensions in self._format_extensions.values())

    def _detect_format(self, file_path: Path) -> ConfigFormat:
        """Detect configuration format from file extension."""
        suffix = file_path.suffix.lower()

        for format_type, extensions in self._format_extensions.items():
            if suffix in extensions:
                return format_type

        raise ValueError(f"Unsupported configuration format: {suffix}")

    def _create_asset_metadata(self, file_path: Path) -> AssetMetadata | None:
        """Create asset metadata for a configuration file."""
        try:
            format_type = self._detect_format(file_path)
            stat_info = file_path.stat()

            return AssetMetadata(
                path=file_path,
                format=format_type,
                checksum=self._calculate_checksum(file_path),
                last_modified=datetime.fromtimestamp(stat_info.st_mtime),
                size_bytes=stat_info.st_size,
                permissions=stat_info.st_mode & 0o777
            )
        except Exception as e:
            logger.error(f"Error creating metadata for {file_path}: {e}")
            return None

    def _calculate_checksum(self, file_path: Path, algorithm: str = "sha256") -> str:
        """Calculate file checksum."""
        hash_func = hashlib.new(algorithm)

        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)

        return hash_func.hexdigest()

    def _load_config_file(self, file_path: Path, format_type: ConfigFormat) -> dict[str, Any] | None:
        """Load configuration file based on format."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            if format_type == ConfigFormat.YAML:
                return yaml.safe_load(content)
            elif format_type == ConfigFormat.JSON:
                return json.loads(content)
            elif format_type == ConfigFormat.TOML:
                if tomllib is None:
                    raise ImportError("TOML support requires tomli or tomllib")
                return tomllib.loads(content)
            else:
                raise ValueError(f"Unsupported format: {format_type}")

        except Exception as e:
            logger.error(f"Error loading config file {file_path}: {e}")
            return None

    def _substitute_environment_variables(self, config_data: dict[str, Any]) -> dict[str, Any]:
        """Substitute environment variables in configuration data."""
        def substitute_value(value):
            if isinstance(value, str):
                # Simple ${VAR} and $VAR substitution
                import re

                def replace_var(match):
                    var_name = match.group(1) or match.group(2)
                    return os.environ.get(var_name, match.group(0))

                # Pattern matches ${VAR} or $VAR
                pattern = r'\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)'
                return re.sub(pattern, replace_var, value)
            elif isinstance(value, dict):
                return {k: substitute_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [substitute_value(item) for item in value]
            else:
                return value

        return substitute_value(config_data)

    def _validate_security(self, file_path: Path, metadata: AssetMetadata) -> ValidationResult:
        """Validate security aspects of configuration file."""
        errors = []
        warnings = []

        # Check file permissions
        if metadata.permissions & 0o002:  # World-writable
            errors.append(f"Configuration file is world-writable: {file_path}")

        if metadata.permissions & 0o044 == 0:  # Not readable by owner/group
            errors.append(f"Configuration file is not readable: {file_path}")

        # Check file size
        if metadata.size_bytes > 10 * 1024 * 1024:  # 10MB limit for config files
            errors.append(f"Configuration file too large: {file_path} ({metadata.size_bytes} bytes)")

        # Check for suspicious patterns in filename
        suspicious_patterns = ['.exe', '.bat', '.cmd', '.sh']
        if any(pattern in file_path.name.lower() for pattern in suspicious_patterns):
            warnings.append(f"Configuration file has suspicious extension: {file_path}")

        metadata.status = AssetStatus.VALID if not errors else AssetStatus.SECURITY_VIOLATION

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata
        )

    def _perform_additional_validation(self, config_data: dict[str, Any], metadata: AssetMetadata) -> ValidationResult:
        """Perform additional validation checks on configuration data."""
        errors = []
        warnings = []

        # Check for required top-level keys
        required_keys = ['name']  # Minimal requirement
        for key in required_keys:
            if key not in config_data:
                warnings.append(f"Configuration missing recommended key: {key}")

        # Validate data types
        if 'version' in config_data and not isinstance(config_data['version'], str):
            errors.append("Version must be a string")

        if 'enabled' in config_data and not isinstance(config_data['enabled'], bool):
            errors.append("Enabled must be a boolean")

        # Check for circular dependencies
        if 'dependencies' in config_data:
            deps = config_data['dependencies']
            if isinstance(deps, list):
                if config_data.get('name') in deps:
                    errors.append("Configuration has circular dependency on itself")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def _deep_merge_configs(self, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """Deep merge configuration dictionaries."""
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    def _validate_merged_config(self, merged_config: dict[str, Any]) -> ValidationResult:
        """Validate merged configuration for consistency."""
        errors = []
        warnings = []

        # Check for conflicting settings
        if merged_config.get('enabled') is False and merged_config.get('priority', 0) > 0:
            warnings.append("Configuration is disabled but has non-zero priority")

        # Validate dependencies exist (simplified check)
        dependencies = merged_config.get('dependencies', [])
        if dependencies and not isinstance(dependencies, list):
            errors.append("Dependencies must be a list")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def _cache_configuration(self, file_path: str, metadata: AssetMetadata, config_data: dict[str, Any]):
        """Cache validated configuration data."""
        with self._cache_lock:
            self._cache[file_path] = (metadata, config_data)
            logger.debug(f"Cached configuration: {file_path}")

    def _get_cached_metadata(self, file_path: str) -> AssetMetadata | None:
        """Get cached metadata for a file path."""
        with self._cache_lock:
            cached_entry = self._cache.get(file_path)
            return cached_entry[0] if cached_entry else None

    def _setup_hot_reload(self):
        """Setup file system watchers for hot-reloading."""
        if not self.asset_directories:
            return

        class ConfigChangeHandler(FileSystemEventHandler):
            def __init__(self, validator):
                self.validator = validator

            def on_modified(self, event):
                if not event.is_directory and self.validator._is_config_file(Path(event.src_path)):
                    logger.info(f"Configuration file changed: {event.src_path}")
                    self.validator.clear_cache(event.src_path)

        handler = ConfigChangeHandler(self)

        for directory in self.asset_directories:
            if directory.exists():
                observer = Observer()
                observer.schedule(handler, str(directory), recursive=True)
                observer.start()
                self._observers.append(observer)
                logger.debug(f"Started watching directory: {directory}")

    def _cleanup_hot_reload(self):
        """Cleanup file system watchers."""
        for observer in self._observers:
            observer.stop()
            observer.join()
        self._observers.clear()
        logger.debug("Stopped all configuration file watchers")

    def __del__(self):
        """Cleanup on destruction."""
        if self.hot_reload_enabled:
            self._cleanup_hot_reload()
