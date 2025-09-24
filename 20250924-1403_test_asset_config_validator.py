"""
Comprehensive unit tests for asset configuration validation system.

This test module provides comprehensive coverage of the AssetConfigValidator,
testing all functionality including multi-format support, schema validation,
inheritance, environment variable substitution, asset management features,
security validation, caching, hot-reloading, and edge cases.
"""

import sys
from pathlib import Path

# Add src/python to path for common module imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "python"))

import json
import os
import tempfile
import threading
import time
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

import pytest
from pydantic import BaseModel, Field, ValidationError

# Import the module under test
from common.core.asset_config_validator import (
    AssetConfigValidator,
    AssetConfigSchema,
    AssetMetadata,
    ValidationResult,
    ConfigFormat,
    AssetStatus,
)

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None


class TestAssetConfigSchema:
    """Test the base asset configuration schema."""

    def test_asset_config_schema_valid(self):
        """Test valid asset configuration schema."""
        config_data = {
            "name": "test-asset",
            "version": "1.2.3",
            "description": "Test asset configuration",
            "enabled": True,
            "priority": 100,
            "dependencies": ["dep1", "dep2"],
        }

        schema = AssetConfigSchema(**config_data)
        assert schema.name == "test-asset"
        assert schema.version == "1.2.3"
        assert schema.description == "Test asset configuration"
        assert schema.enabled is True
        assert schema.priority == 100
        assert schema.dependencies == ["dep1", "dep2"]

    def test_asset_config_schema_defaults(self):
        """Test asset configuration schema with defaults."""
        schema = AssetConfigSchema(name="test-asset")

        assert schema.name == "test-asset"
        assert schema.version == "1.0.0"
        assert schema.description is None
        assert schema.enabled is True
        assert schema.priority == 0
        assert schema.dependencies == []
        assert schema.checksum_algorithm == "sha256"
        assert schema.allow_env_substitution is True
        assert schema.required_env_vars == []

    def test_asset_config_schema_invalid_version(self):
        """Test invalid version format validation."""
        with pytest.raises(ValidationError) as exc_info:
            AssetConfigSchema(name="test", version="1.2")

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "Version must be in format X.Y.Z" in str(errors[0]["msg"])

    def test_asset_config_schema_invalid_version_non_numeric(self):
        """Test invalid non-numeric version parts."""
        with pytest.raises(ValidationError) as exc_info:
            AssetConfigSchema(name="test", version="1.2.a")

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "Version parts must be numeric" in str(errors[0]["msg"])

    def test_asset_config_schema_invalid_priority_range(self):
        """Test invalid priority range."""
        with pytest.raises(ValidationError) as exc_info:
            AssetConfigSchema(name="test", priority=1001)

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "Priority must be between 0 and 1000" in str(errors[0]["msg"])

    def test_asset_config_schema_negative_priority(self):
        """Test negative priority validation."""
        with pytest.raises(ValidationError) as exc_info:
            AssetConfigSchema(name="test", priority=-1)

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "Priority must be between 0 and 1000" in str(errors[0]["msg"])


class TestAssetMetadata:
    """Test asset metadata functionality."""

    def test_asset_metadata_creation(self):
        """Test asset metadata creation."""
        test_path = Path("/test/config.yaml")
        metadata = AssetMetadata(
            path=test_path,
            format=ConfigFormat.YAML,
            version="2.1.0",
            checksum="abc123",
            size_bytes=1024,
            permissions=0o644,
            status=AssetStatus.VALID,
            dependencies=["dep1"],
            tags={"test", "config"}
        )

        assert metadata.path == test_path
        assert metadata.format == ConfigFormat.YAML
        assert metadata.version == "2.1.0"
        assert metadata.checksum == "abc123"
        assert metadata.size_bytes == 1024
        assert metadata.permissions == 0o644
        assert metadata.status == AssetStatus.VALID
        assert metadata.dependencies == ["dep1"]
        assert metadata.tags == {"test", "config"}

    def test_asset_metadata_defaults(self):
        """Test asset metadata with default values."""
        test_path = Path("/test/config.json")
        metadata = AssetMetadata(path=test_path, format=ConfigFormat.JSON)

        assert metadata.path == test_path
        assert metadata.format == ConfigFormat.JSON
        assert metadata.version == "1.0.0"
        assert metadata.checksum == ""
        assert metadata.size_bytes == 0
        assert metadata.permissions == 0
        assert metadata.status == AssetStatus.VALID
        assert metadata.dependencies == []
        assert metadata.tags == set()
        assert isinstance(metadata.last_modified, datetime)


class TestValidationResult:
    """Test validation result functionality."""

    def test_validation_result_valid(self):
        """Test valid validation result."""
        result = ValidationResult(
            is_valid=True,
            warnings=["Minor warning"],
            validated_config={"key": "value"}
        )

        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == ["Minor warning"]
        assert result.metadata is None
        assert result.validated_config == {"key": "value"}

    def test_validation_result_invalid(self):
        """Test invalid validation result."""
        metadata = AssetMetadata(Path("/test"), ConfigFormat.YAML)
        result = ValidationResult(
            is_valid=False,
            errors=["Critical error"],
            warnings=["Warning"],
            metadata=metadata
        )

        assert result.is_valid is False
        assert result.errors == ["Critical error"]
        assert result.warnings == ["Warning"]
        assert result.metadata == metadata
        assert result.validated_config is None


class TestAssetConfigValidator:
    """Test the main AssetConfigValidator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.validator = AssetConfigValidator(
            asset_directories=[self.temp_dir],
            cache_enabled=True,
            hot_reload_enabled=False
        )

        # Create test configuration files
        self._create_test_files()

    def teardown_method(self):
        """Clean up test fixtures."""
        # Clean up temporary directory
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

        # Disable hot reload and cleanup
        if hasattr(self.validator, 'hot_reload_enabled') and self.validator.hot_reload_enabled:
            self.validator.disable_hot_reload()

    def _create_test_files(self):
        """Create test configuration files."""
        # Valid YAML config
        self.yaml_config = self.temp_dir / "valid_config.yaml"
        yaml_content = {
            "name": "test-yaml-config",
            "version": "1.0.0",
            "description": "Test YAML configuration",
            "enabled": True,
            "priority": 50,
            "dependencies": ["base-config"]
        }
        with open(self.yaml_config, "w") as f:
            yaml.dump(yaml_content, f)

        # Valid JSON config
        self.json_config = self.temp_dir / "valid_config.json"
        json_content = {
            "name": "test-json-config",
            "version": "2.1.0",
            "enabled": False,
            "priority": 10
        }
        with open(self.json_config, "w") as f:
            json.dump(json_content, f)

        # Valid TOML config (if TOML support is available)
        if tomllib is not None:
            self.toml_config = self.temp_dir / "valid_config.toml"
            toml_content = """
name = "test-toml-config"
version = "3.0.0"
enabled = true
priority = 75

[advanced]
setting = "value"
"""
            with open(self.toml_config, "w") as f:
                f.write(toml_content)

        # Invalid YAML config
        self.invalid_yaml = self.temp_dir / "invalid.yaml"
        with open(self.invalid_yaml, "w") as f:
            f.write("invalid: yaml: content: [\n")

        # Config with environment variables
        self.env_config = self.temp_dir / "env_config.yaml"
        env_content = {
            "name": "env-test",
            "database_url": "${DATABASE_URL}",
            "api_key": "$API_KEY",
            "fallback": "${MISSING_VAR:-default_value}"
        }
        with open(self.env_config, "w") as f:
            yaml.dump(env_content, f)

        # Large config file (for size testing)
        self.large_config = self.temp_dir / "large_config.json"
        large_content = {"data": "x" * (11 * 1024 * 1024)}  # 11MB
        with open(self.large_config, "w") as f:
            json.dump(large_content, f)

        # Config with inheritance
        self.base_config = self.temp_dir / "base_config.yaml"
        base_content = {
            "name": "base-config",
            "version": "1.0.0",
            "settings": {
                "timeout": 30,
                "retries": 3,
                "debug": False
            }
        }
        with open(self.base_config, "w") as f:
            yaml.dump(base_content, f)

        self.override_config = self.temp_dir / "override_config.yaml"
        override_content = {
            "settings": {
                "timeout": 60,
                "debug": True,
                "new_option": "enabled"
            },
            "extra": "override_value"
        }
        with open(self.override_config, "w") as f:
            yaml.dump(override_content, f)

    def test_validator_initialization(self):
        """Test validator initialization."""
        assert len(self.validator.asset_directories) == 1
        assert self.validator.cache_enabled is True
        assert self.validator.hot_reload_enabled is False
        assert self.validator._cache == {}
        assert self.validator._observers == []

    def test_validator_initialization_with_hot_reload(self):
        """Test validator initialization with hot-reloading enabled."""
        with patch('common.core.asset_config_validator.Observer') as mock_observer_class:
            mock_observer = Mock()
            mock_observer_class.return_value = mock_observer

            validator = AssetConfigValidator(
                asset_directories=[self.temp_dir],
                hot_reload_enabled=True
            )

            assert validator.hot_reload_enabled is True
            mock_observer_class.assert_called()

    def test_discover_assets(self):
        """Test asset discovery functionality."""
        discovered_assets = self.validator.discover_assets()

        # Should find at least the valid config files we created
        assert len(discovered_assets) >= 2
        asset_names = [asset.path.name for asset in discovered_assets]
        assert "valid_config.yaml" in asset_names
        assert "valid_config.json" in asset_names

        # Check metadata is properly populated
        yaml_asset = next(asset for asset in discovered_assets if asset.path.name == "valid_config.yaml")
        assert yaml_asset.format == ConfigFormat.YAML
        assert yaml_asset.checksum != ""
        assert yaml_asset.size_bytes > 0
        assert isinstance(yaml_asset.last_modified, datetime)

    def test_discover_assets_specific_directory(self):
        """Test asset discovery in a specific directory."""
        # Create a subdirectory with additional configs
        subdir = self.temp_dir / "subdir"
        subdir.mkdir()

        sub_config = subdir / "sub_config.json"
        with open(sub_config, "w") as f:
            json.dump({"name": "sub-config"}, f)

        # Discover assets in subdirectory only
        discovered_assets = self.validator.discover_assets(subdir)

        assert len(discovered_assets) == 1
        assert discovered_assets[0].path.name == "sub_config.json"

    def test_discover_assets_nonexistent_directory(self):
        """Test asset discovery with non-existent directory."""
        nonexistent_dir = Path("/nonexistent/directory")
        validator = AssetConfigValidator(asset_directories=[nonexistent_dir])

        with patch('common.core.asset_config_validator.logger') as mock_logger:
            discovered_assets = validator.discover_assets()

            assert len(discovered_assets) == 0
            mock_logger.warning.assert_called()

    def test_validate_config_file_yaml_valid(self):
        """Test validation of valid YAML configuration file."""
        result = self.validator.validate_config_file(self.yaml_config, AssetConfigSchema)

        assert result.is_valid is True
        assert len(result.errors) == 0
        assert result.metadata is not None
        assert result.metadata.format == ConfigFormat.YAML
        assert result.validated_config is not None
        assert result.validated_config["name"] == "test-yaml-config"

    def test_validate_config_file_json_valid(self):
        """Test validation of valid JSON configuration file."""
        result = self.validator.validate_config_file(self.json_config, AssetConfigSchema)

        assert result.is_valid is True
        assert len(result.errors) == 0
        assert result.metadata is not None
        assert result.metadata.format == ConfigFormat.JSON
        assert result.validated_config["name"] == "test-json-config"

    @pytest.mark.skipif(tomllib is None, reason="TOML support not available")
    def test_validate_config_file_toml_valid(self):
        """Test validation of valid TOML configuration file."""
        result = self.validator.validate_config_file(self.toml_config, AssetConfigSchema)

        assert result.is_valid is True
        assert len(result.errors) == 0
        assert result.metadata is not None
        assert result.metadata.format == ConfigFormat.TOML
        assert result.validated_config["name"] == "test-toml-config"

    def test_validate_config_file_nonexistent(self):
        """Test validation of non-existent configuration file."""
        nonexistent_file = self.temp_dir / "nonexistent.yaml"
        result = self.validator.validate_config_file(nonexistent_file)

        assert result.is_valid is False
        assert len(result.errors) == 1
        assert "Configuration file not found" in result.errors[0]

    def test_validate_config_file_invalid_yaml(self):
        """Test validation of invalid YAML file."""
        result = self.validator.validate_config_file(self.invalid_yaml)

        assert result.is_valid is False
        assert len(result.errors) >= 1
        assert result.metadata is not None

    def test_validate_config_file_with_schema_validation_error(self):
        """Test configuration file that fails schema validation."""
        invalid_schema_config = self.temp_dir / "invalid_schema.yaml"
        invalid_content = {
            "name": "test",
            "version": "invalid-version",  # This should fail validation
            "priority": 2000  # This should also fail validation
        }
        with open(invalid_schema_config, "w") as f:
            yaml.dump(invalid_content, f)

        result = self.validator.validate_config_file(invalid_schema_config, AssetConfigSchema)

        assert result.is_valid is False
        assert len(result.errors) >= 1
        assert any("Version must be in format X.Y.Z" in error for error in result.errors)

    def test_validate_config_file_security_violation(self):
        """Test security validation for world-writable files."""
        # Create a world-writable config file
        security_test_config = self.temp_dir / "security_test.yaml"
        with open(security_test_config, "w") as f:
            yaml.dump({"name": "security-test"}, f)

        # Make file world-writable
        os.chmod(security_test_config, 0o666)

        result = self.validator.validate_config_file(security_test_config)

        assert result.is_valid is False
        assert any("world-writable" in error for error in result.errors)

    def test_validate_config_file_large_file(self):
        """Test validation of oversized configuration file."""
        result = self.validator.validate_config_file(self.large_config)

        assert result.is_valid is False
        assert any("too large" in error for error in result.errors)

    @patch.dict(os.environ, {"DATABASE_URL": "postgres://localhost", "API_KEY": "secret123"}, clear=True)
    def test_environment_variable_substitution(self):
        """Test environment variable substitution in configuration."""
        result = self.validator.validate_config_file(self.env_config, enable_env_substitution=True)

        assert result.is_valid is True
        assert result.validated_config["database_url"] == "postgres://localhost"
        assert result.validated_config["api_key"] == "secret123"
        # Test fallback value for missing variable
        assert result.validated_config["fallback"] == "default_value"

    def test_environment_variable_substitution_disabled(self):
        """Test configuration with environment variable substitution disabled."""
        result = self.validator.validate_config_file(self.env_config, enable_env_substitution=False)

        assert result.is_valid is True
        # Variables should remain as-is
        assert result.validated_config["database_url"] == "${DATABASE_URL}"
        assert result.validated_config["api_key"] == "$API_KEY"

    def test_validate_multiple_configs(self):
        """Test validation of multiple configuration files."""
        config_files = [self.yaml_config, self.json_config]
        results = self.validator.validate_multiple_configs(config_files, AssetConfigSchema)

        assert len(results) == 2
        assert all(result.is_valid for result in results.values())

        # Check specific file results
        yaml_result = results[str(self.yaml_config)]
        json_result = results[str(self.json_config)]

        assert yaml_result.validated_config["name"] == "test-yaml-config"
        assert json_result.validated_config["name"] == "test-json-config"

    def test_validate_multiple_configs_fail_fast(self):
        """Test validation of multiple configs with fail-fast enabled."""
        config_files = [self.invalid_yaml, self.yaml_config]  # Invalid first
        results = self.validator.validate_multiple_configs(config_files, fail_fast=True)

        # Should only have result for the first (failed) file
        assert len(results) == 1
        assert str(self.invalid_yaml) in results
        assert not results[str(self.invalid_yaml)].is_valid

    def test_validate_configuration_inheritance(self):
        """Test configuration inheritance and merging."""
        result = self.validator.validate_configuration_inheritance(
            self.base_config,
            [self.override_config]
        )

        assert result.is_valid is True
        assert result.validated_config is not None

        # Check merged configuration
        config = result.validated_config
        assert config["name"] == "base-config"  # From base
        assert config["version"] == "1.0.0"  # From base
        assert config["settings"]["timeout"] == 60  # Overridden
        assert config["settings"]["retries"] == 3  # From base
        assert config["settings"]["debug"] is True  # Overridden
        assert config["settings"]["new_option"] == "enabled"  # New from override
        assert config["extra"] == "override_value"  # New from override

    def test_validate_configuration_inheritance_invalid_base(self):
        """Test configuration inheritance with invalid base configuration."""
        result = self.validator.validate_configuration_inheritance(
            self.invalid_yaml,  # Invalid base
            [self.override_config]
        )

        assert result.is_valid is False
        assert len(result.errors) >= 1
        assert "Base configuration is invalid" in result.errors[0]

    def test_validate_configuration_inheritance_invalid_override(self):
        """Test configuration inheritance with invalid override configuration."""
        result = self.validator.validate_configuration_inheritance(
            self.base_config,
            [self.invalid_yaml]  # Invalid override
        )

        assert result.is_valid is False
        assert any("Override configuration is invalid" in error for error in result.errors)

    def test_check_asset_integrity(self):
        """Test asset integrity checking."""
        result = self.validator.check_asset_integrity(self.yaml_config)

        assert result.is_valid is True
        assert result.metadata is not None
        assert result.metadata.checksum != ""
        assert result.metadata.size_bytes > 0
        assert len(result.warnings) >= 0  # May have permission warnings

    def test_check_asset_integrity_nonexistent_file(self):
        """Test asset integrity check for non-existent file."""
        nonexistent_file = self.temp_dir / "nonexistent.yaml"
        result = self.validator.check_asset_integrity(nonexistent_file)

        assert result.is_valid is False
        assert "Asset file not found" in result.errors[0]

    def test_check_asset_integrity_large_file(self):
        """Test asset integrity check for oversized file."""
        result = self.validator.check_asset_integrity(self.large_config)

        assert result.is_valid is False
        assert any("exceeds maximum size limit" in error for error in result.errors)

    def test_caching_functionality(self):
        """Test configuration caching."""
        # Validate a configuration (should cache it)
        result1 = self.validator.validate_config_file(self.yaml_config, AssetConfigSchema)
        assert result1.is_valid is True

        # Get cached configuration
        cached_config = self.validator.get_cached_config(str(self.yaml_config))
        assert cached_config is not None
        assert cached_config["name"] == "test-yaml-config"

        # Validate same configuration again (should use cache)
        result2 = self.validator.validate_config_file(self.yaml_config, AssetConfigSchema)
        assert result2.is_valid is True
        assert result2.validated_config == result1.validated_config

    def test_caching_disabled(self):
        """Test validator with caching disabled."""
        validator_no_cache = AssetConfigValidator(
            asset_directories=[self.temp_dir],
            cache_enabled=False
        )

        # Validate configuration
        result = validator_no_cache.validate_config_file(self.yaml_config, AssetConfigSchema)
        assert result.is_valid is True

        # Should not be cached
        cached_config = validator_no_cache.get_cached_config(str(self.yaml_config))
        assert cached_config is None

    def test_cache_invalidation_on_file_modification(self):
        """Test cache invalidation when file is modified."""
        # Validate and cache configuration
        result1 = self.validator.validate_config_file(self.yaml_config, AssetConfigSchema)
        assert result1.is_valid is True

        # Verify it's cached
        cached_config = self.validator.get_cached_config(str(self.yaml_config))
        assert cached_config is not None

        # Modify the file
        time.sleep(0.1)  # Ensure different timestamp
        modified_content = {
            "name": "modified-config",
            "version": "2.0.0"
        }
        with open(self.yaml_config, "w") as f:
            yaml.dump(modified_content, f)

        # Cache should be invalidated
        cached_config_after = self.validator.get_cached_config(str(self.yaml_config))
        assert cached_config_after is None

    def test_clear_cache_specific_file(self):
        """Test clearing cache for a specific file."""
        # Cache some configurations
        self.validator.validate_config_file(self.yaml_config, AssetConfigSchema)
        self.validator.validate_config_file(self.json_config, AssetConfigSchema)

        # Verify both are cached
        assert self.validator.get_cached_config(str(self.yaml_config)) is not None
        assert self.validator.get_cached_config(str(self.json_config)) is not None

        # Clear cache for specific file
        self.validator.clear_cache(str(self.yaml_config))

        # Only specific file should be cleared
        assert self.validator.get_cached_config(str(self.yaml_config)) is None
        assert self.validator.get_cached_config(str(self.json_config)) is not None

    def test_clear_cache_all(self):
        """Test clearing entire cache."""
        # Cache some configurations
        self.validator.validate_config_file(self.yaml_config, AssetConfigSchema)
        self.validator.validate_config_file(self.json_config, AssetConfigSchema)

        # Verify both are cached
        assert self.validator.get_cached_config(str(self.yaml_config)) is not None
        assert self.validator.get_cached_config(str(self.json_config)) is not None

        # Clear entire cache
        self.validator.clear_cache()

        # All should be cleared
        assert self.validator.get_cached_config(str(self.yaml_config)) is None
        assert self.validator.get_cached_config(str(self.json_config)) is None

    def test_hot_reload_enable_disable(self):
        """Test enabling and disabling hot-reload."""
        assert self.validator.hot_reload_enabled is False

        with patch('common.core.asset_config_validator.Observer') as mock_observer_class:
            mock_observer = Mock()
            mock_observer_class.return_value = mock_observer

            # Enable hot reload
            self.validator.enable_hot_reload()
            assert self.validator.hot_reload_enabled is True
            mock_observer_class.assert_called()

            # Disable hot reload
            self.validator.disable_hot_reload()
            assert self.validator.hot_reload_enabled is False
            mock_observer.stop.assert_called()
            mock_observer.join.assert_called()

    def test_format_detection(self):
        """Test configuration format detection."""
        yaml_format = self.validator._detect_format(Path("config.yaml"))
        json_format = self.validator._detect_format(Path("config.json"))

        assert yaml_format == ConfigFormat.YAML
        assert json_format == ConfigFormat.JSON

        if tomllib is not None:
            toml_format = self.validator._detect_format(Path("config.toml"))
            assert toml_format == ConfigFormat.TOML

    def test_format_detection_unsupported(self):
        """Test format detection for unsupported file types."""
        with pytest.raises(ValueError) as exc_info:
            self.validator._detect_format(Path("config.xml"))

        assert "Unsupported configuration format" in str(exc_info.value)

    def test_checksum_calculation(self):
        """Test checksum calculation."""
        checksum = self.validator._calculate_checksum(self.yaml_config)

        assert isinstance(checksum, str)
        assert len(checksum) == 64  # SHA256 hex length
        assert checksum != ""

        # Calculate again - should be the same
        checksum2 = self.validator._calculate_checksum(self.yaml_config)
        assert checksum == checksum2

    def test_checksum_calculation_different_algorithms(self):
        """Test checksum calculation with different algorithms."""
        sha256_checksum = self.validator._calculate_checksum(self.yaml_config, "sha256")
        md5_checksum = self.validator._calculate_checksum(self.yaml_config, "md5")

        assert len(sha256_checksum) == 64  # SHA256 hex length
        assert len(md5_checksum) == 32  # MD5 hex length
        assert sha256_checksum != md5_checksum

    def test_environment_variable_substitution_complex(self):
        """Test complex environment variable substitution patterns."""
        config_data = {
            "simple_var": "$VAR1",
            "braced_var": "${VAR2}",
            "nested_dict": {
                "deep_var": "${VAR3}",
                "mixed": "prefix_${VAR4}_suffix"
            },
            "list_with_vars": ["${VAR5}", "static", "$VAR6"],
            "no_substitution": "regular_string"
        }

        with patch.dict(os.environ, {
            "VAR1": "value1",
            "VAR2": "value2",
            "VAR3": "value3",
            "VAR4": "value4",
            "VAR5": "value5",
            "VAR6": "value6"
        }):
            result = self.validator._substitute_environment_variables(config_data)

        assert result["simple_var"] == "value1"
        assert result["braced_var"] == "value2"
        assert result["nested_dict"]["deep_var"] == "value3"
        assert result["nested_dict"]["mixed"] == "prefix_value4_suffix"
        assert result["list_with_vars"] == ["value5", "static", "value6"]
        assert result["no_substitution"] == "regular_string"

    def test_environment_variable_substitution_missing_vars(self):
        """Test environment variable substitution with missing variables."""
        config_data = {
            "missing_simple": "$MISSING_VAR",
            "missing_braced": "${MISSING_VAR2}"
        }

        result = self.validator._substitute_environment_variables(config_data)

        # Missing variables should remain unchanged
        assert result["missing_simple"] == "$MISSING_VAR"
        assert result["missing_braced"] == "${MISSING_VAR2}"

    def test_deep_merge_configs(self):
        """Test deep merging of configuration dictionaries."""
        base_config = {
            "level1": {
                "level2": {
                    "keep_this": "base_value",
                    "override_this": "base_override"
                },
                "keep_level2": "base_level2"
            },
            "keep_level1": "base_level1"
        }

        override_config = {
            "level1": {
                "level2": {
                    "override_this": "override_value",
                    "new_key": "new_value"
                },
                "new_level2": "override_level2"
            },
            "new_level1": "override_level1"
        }

        result = self.validator._deep_merge_configs(base_config, override_config)

        assert result["keep_level1"] == "base_level1"
        assert result["new_level1"] == "override_level1"
        assert result["level1"]["keep_level2"] == "base_level2"
        assert result["level1"]["new_level2"] == "override_level2"
        assert result["level1"]["level2"]["keep_this"] == "base_value"
        assert result["level1"]["level2"]["override_this"] == "override_value"
        assert result["level1"]["level2"]["new_key"] == "new_value"

    def test_validate_merged_config(self):
        """Test validation of merged configuration."""
        # Valid merged config
        valid_config = {
            "name": "merged-config",
            "enabled": True,
            "priority": 10,
            "dependencies": ["dep1", "dep2"]
        }

        result = self.validator._validate_merged_config(valid_config)
        assert result.is_valid is True
        assert len(result.errors) == 0

        # Config with warnings
        warning_config = {
            "enabled": False,
            "priority": 50  # Non-zero priority but disabled
        }

        result = self.validator._validate_merged_config(warning_config)
        assert result.is_valid is True
        assert len(result.warnings) > 0
        assert any("disabled but has non-zero priority" in warning for warning in result.warnings)

        # Config with errors
        error_config = {
            "dependencies": "not_a_list"  # Should be a list
        }

        result = self.validator._validate_merged_config(error_config)
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("Dependencies must be a list" in error for error in result.errors)

    def test_thread_safety(self):
        """Test thread safety of caching operations."""
        def validate_configs():
            """Function to run validation in multiple threads."""
            for i in range(10):
                result = self.validator.validate_config_file(self.yaml_config, AssetConfigSchema)
                assert result.is_valid is True

        # Run validation in multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=validate_configs)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Cache should still be consistent
        cached_config = self.validator.get_cached_config(str(self.yaml_config))
        assert cached_config is not None
        assert cached_config["name"] == "test-yaml-config"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.validator = AssetConfigValidator(asset_directories=[self.temp_dir])

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_empty_configuration_file(self):
        """Test handling of empty configuration files."""
        empty_config = self.temp_dir / "empty.yaml"
        empty_config.touch()

        result = self.validator.validate_config_file(empty_config)
        # Empty YAML file should load as None, then be handled gracefully
        assert result.is_valid is True  # Empty configs can be valid

    def test_binary_file_as_config(self):
        """Test handling of binary files mistakenly treated as config."""
        binary_file = self.temp_dir / "binary.yaml"
        with open(binary_file, "wb") as f:
            f.write(b"\x00\x01\x02\x03\x04\x05")

        result = self.validator.validate_config_file(binary_file)
        assert result.is_valid is False

    def test_very_large_configuration_data(self):
        """Test handling of configurations with very large data structures."""
        large_config = self.temp_dir / "large_data.json"
        large_data = {
            "name": "large-config",
            "large_array": ["item"] * 100000,  # Large array
            "deep_nesting": {}
        }

        # Create deeply nested structure
        current = large_data["deep_nesting"]
        for i in range(1000):
            current[f"level_{i}"] = {}
            current = current[f"level_{i}"]

        with open(large_config, "w") as f:
            json.dump(large_data, f)

        # This should work but might be slow
        result = self.validator.validate_config_file(large_config)
        assert result.is_valid is True
        assert result.validated_config["name"] == "large-config"

    def test_unicode_in_configuration(self):
        """Test handling of Unicode characters in configuration."""
        unicode_config = self.temp_dir / "unicode.yaml"
        unicode_content = {
            "name": "unicode-тест-🚀",
            "description": "Configuration with émojis 🎉 and ñoñó characters",
            "paths": ["/path/with/中文/characters", "/путь/с/кириллицей"]
        }

        with open(unicode_config, "w", encoding="utf-8") as f:
            yaml.dump(unicode_content, f, allow_unicode=True)

        result = self.validator.validate_config_file(unicode_config)
        assert result.is_valid is True
        assert result.validated_config["name"] == "unicode-тест-🚀"

    def test_circular_dependency_detection(self):
        """Test detection of circular dependencies in configurations."""
        # This test would need more sophisticated dependency resolution
        # For now, test basic self-dependency detection
        circular_config = self.temp_dir / "circular.yaml"
        circular_content = {
            "name": "circular-config",
            "dependencies": ["circular-config"]  # Self-dependency
        }

        with open(circular_config, "w") as f:
            yaml.dump(circular_content, f)

        result = self.validator.validate_config_file(circular_config, AssetConfigSchema)
        # The validator should detect this in additional validation
        assert any("circular dependency" in error.lower() for error in result.errors)

    def test_concurrent_cache_access(self):
        """Test concurrent access to cache with file modifications."""
        config_file = self.temp_dir / "concurrent_test.yaml"
        initial_content = {"name": "initial", "version": "1.0.0"}

        with open(config_file, "w") as f:
            yaml.dump(initial_content, f)

        def modify_and_validate():
            """Modify file and validate concurrently."""
            for i in range(5):
                # Modify file
                content = {"name": f"modified-{i}", "version": "1.0.0"}
                with open(config_file, "w") as f:
                    yaml.dump(content, f)

                # Validate (may hit cache or reload)
                result = self.validator.validate_config_file(config_file, AssetConfigSchema)
                assert result.is_valid is True

                time.sleep(0.01)  # Small delay

        # Run concurrent modifications and validations
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=modify_and_validate)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Final validation should succeed
        result = self.validator.validate_config_file(config_file, AssetConfigSchema)
        assert result.is_valid is True

    @pytest.mark.skipif(tomllib is None, reason="TOML support not available")
    def test_toml_file_without_toml_support(self):
        """Test TOML file handling when TOML support is not available."""
        with patch('common.core.asset_config_validator.tomllib', None):
            validator = AssetConfigValidator(asset_directories=[self.temp_dir])

            toml_config = self.temp_dir / "test.toml"
            with open(toml_config, "w") as f:
                f.write('name = "test"\n')

            result = validator.validate_config_file(toml_config)
            assert result.is_valid is False
            assert any("TOML support requires" in error for error in result.errors)

    def test_permission_error_handling(self):
        """Test handling of permission errors during file operations."""
        restricted_config = self.temp_dir / "restricted.yaml"
        with open(restricted_config, "w") as f:
            yaml.dump({"name": "restricted"}, f)

        # Make file unreadable
        os.chmod(restricted_config, 0o000)

        try:
            result = self.validator.validate_config_file(restricted_config)
            # Should handle permission error gracefully
            assert result.is_valid is False
        finally:
            # Restore permissions for cleanup
            os.chmod(restricted_config, 0o644)

    def test_cleanup_on_destruction(self):
        """Test proper cleanup when validator is destroyed."""
        with patch('common.core.asset_config_validator.Observer') as mock_observer_class:
            mock_observer = Mock()
            mock_observer_class.return_value = mock_observer

            # Create validator with hot reload enabled
            validator = AssetConfigValidator(
                asset_directories=[self.temp_dir],
                hot_reload_enabled=True
            )

            # Destroy validator
            del validator

            # Observer should be cleaned up (this is handled in __del__)
            # Note: __del__ cleanup is not guaranteed to be called immediately,
            # so we test the cleanup method directly
            validator = AssetConfigValidator(
                asset_directories=[self.temp_dir],
                hot_reload_enabled=True
            )

            validator._cleanup_hot_reload()
            mock_observer.stop.assert_called()
            mock_observer.join.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])