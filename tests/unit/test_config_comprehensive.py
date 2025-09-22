"""
Comprehensive unit tests for configuration management to achieve 100% coverage.

This test module provides comprehensive coverage of the config.py module,
testing all functionality including edge cases, error scenarios, and
environment variable handling with proper mocking.
"""

import sys
from pathlib import Path

# Add src/python to path for common module imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

import os
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from typing import Any, Dict

import pytest
from pydantic import ValidationError

from workspace_qdrant_mcp.core.config import (
    Config,
    EmbeddingConfig,
    QdrantConfig,
    WorkspaceConfig,
    GrpcConfig,
    AutoIngestionConfig,
    setup_stdio_environment,
)


class TestSetupStdioEnvironment:
    """Test early environment setup for MCP stdio mode."""

    @patch.dict(os.environ, {}, clear=True)
    def test_stdio_mode_detection_env_var(self):
        """Test stdio mode detection via WQM_STDIO_MODE env var."""
        with patch.dict(os.environ, {"WQM_STDIO_MODE": "true"}):
            setup_stdio_environment()
            assert os.environ.get("MCP_QUIET_MODE") == "true"
            assert os.environ.get("TOKENIZERS_PARALLELISM") == "false"

    @patch.dict(os.environ, {}, clear=True)
    def test_stdio_mode_detection_mcp_transport(self):
        """Test stdio mode detection via MCP_TRANSPORT env var."""
        with patch.dict(os.environ, {"MCP_TRANSPORT": "stdio"}):
            setup_stdio_environment()
            assert os.environ.get("MCP_QUIET_MODE") == "true"
            assert os.environ.get("GRPC_VERBOSITY") == "NONE"

    @patch.dict(os.environ, {}, clear=True)
    @patch('os.sys.argv', ['script', '--transport', 'stdio'])
    def test_stdio_mode_detection_command_args(self):
        """Test stdio mode detection via command line arguments."""
        with patch('hasattr', return_value=True):
            setup_stdio_environment()
            assert os.environ.get("MCP_QUIET_MODE") == "true"
            assert os.environ.get("TF_CPP_MIN_LOG_LEVEL") == "3"

    @patch.dict(os.environ, {}, clear=True)
    def test_non_stdio_mode(self):
        """Test that env vars are not set when not in stdio mode."""
        setup_stdio_environment()
        # Should not add stdio environment variables when not in stdio mode
        assert "MCP_QUIET_MODE" not in os.environ

    @patch.dict(os.environ, {"TOKENIZERS_PARALLELISM": "existing"}, clear=False)
    def test_existing_env_vars_not_overridden(self):
        """Test that existing environment variables are not overridden."""
        with patch.dict(os.environ, {"WQM_STDIO_MODE": "true"}):
            setup_stdio_environment()
            # Should not override existing values
            assert os.environ.get("TOKENIZERS_PARALLELISM") == "existing"


class TestEmbeddingConfigComprehensive:
    """Comprehensive tests for EmbeddingConfig."""

    def test_all_field_types(self):
        """Test all field types and their validation."""
        config = EmbeddingConfig(
            model="test-model",
            enable_sparse_vectors=False,
            chunk_size=1000,
            chunk_overlap=200,
            batch_size=25
        )
        assert config.model == "test-model"
        assert config.enable_sparse_vectors is False
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.batch_size == 25

    def test_edge_case_values(self):
        """Test edge case values for numeric fields."""
        config = EmbeddingConfig(
            chunk_size=1,
            chunk_overlap=0,
            batch_size=1
        )
        assert config.chunk_size == 1
        assert config.chunk_overlap == 0
        assert config.batch_size == 1


class TestQdrantConfigComprehensive:
    """Comprehensive tests for QdrantConfig."""

    def test_all_fields_with_values(self):
        """Test all fields with non-default values."""
        config = QdrantConfig(
            url="https://custom.qdrant.cloud:6334",
            api_key="test-key-123",
            timeout=120,
            prefer_grpc=False
        )
        assert config.url == "https://custom.qdrant.cloud:6334"
        assert config.api_key == "test-key-123"
        assert config.timeout == 120
        assert config.prefer_grpc is False

    def test_none_api_key(self):
        """Test handling of None API key."""
        config = QdrantConfig(api_key=None)
        assert config.api_key is None

    def test_union_type_handling(self):
        """Test string|None union type for api_key."""
        config1 = QdrantConfig(api_key="string-key")
        config2 = QdrantConfig(api_key=None)
        assert config1.api_key == "string-key"
        assert config2.api_key is None


class TestWorkspaceConfigComprehensive:
    """Comprehensive tests for WorkspaceConfig."""

    def test_all_fields_populated(self):
        """Test all fields with values."""
        config = WorkspaceConfig(
            collection_types=["docs", "notes", "scratchbook"],
            global_collections=["shared", "common"],
            github_user="testuser",
            auto_create_collections=True,
            memory_collection_name="__custom_memory",
            code_collection_name="__custom_code",
            custom_include_patterns=["**/*.md", "**/*.txt"],
            custom_exclude_patterns=["**/node_modules/**"],
            custom_project_indicators={"yarn.lock": {"pattern": "yarn.lock", "confidence": 0.9}}
        )

        assert config.collection_types == ["docs", "notes", "scratchbook"]
        assert config.global_collections == ["shared", "common"]
        assert config.github_user == "testuser"
        assert config.auto_create_collections is True
        assert config.memory_collection_name == "__custom_memory"
        assert config.code_collection_name == "__custom_code"
        assert config.custom_include_patterns == ["**/*.md", "**/*.txt"]
        assert config.custom_exclude_patterns == ["**/node_modules/**"]
        assert "yarn.lock" in config.custom_project_indicators

    def test_effective_collection_types(self):
        """Test effective_collection_types property."""
        config = WorkspaceConfig(collection_types=["test1", "test2"])
        assert config.effective_collection_types == ["test1", "test2"]

    def test_create_pattern_manager(self):
        """Test PatternManager creation with custom patterns."""
        config = WorkspaceConfig(
            custom_include_patterns=["*.custom"],
            custom_exclude_patterns=["*.exclude"],
            custom_project_indicators={"custom": {"pattern": "custom.file"}}
        )

        # Mock the PatternManager to avoid circular imports in tests
        with patch('workspace_qdrant_mcp.core.config.PatternManager') as mock_pm:
            config.create_pattern_manager()
            mock_pm.assert_called_once_with(
                custom_include_patterns=["*.custom"],
                custom_exclude_patterns=["*.exclude"],
                custom_project_indicators={"custom": {"pattern": "custom.file"}}
            )


class TestGrpcConfigComprehensive:
    """Comprehensive tests for GrpcConfig."""

    def test_all_fields_custom_values(self):
        """Test all gRPC configuration fields with custom values."""
        config = GrpcConfig(
            enabled=True,
            host="192.168.1.100",
            port=50052,
            fallback_to_direct=False,
            connection_timeout=30.0,
            max_retries=5,
            retry_backoff_multiplier=2.0,
            health_check_interval=60.0,
            max_message_length=200 * 1024 * 1024,
            keepalive_time=60
        )

        assert config.enabled is True
        assert config.host == "192.168.1.100"
        assert config.port == 50052
        assert config.fallback_to_direct is False
        assert config.connection_timeout == 30.0
        assert config.max_retries == 5
        assert config.retry_backoff_multiplier == 2.0
        assert config.health_check_interval == 60.0
        assert config.max_message_length == 200 * 1024 * 1024
        assert config.keepalive_time == 60


class TestAutoIngestionConfigComprehensive:
    """Comprehensive tests for AutoIngestionConfig."""

    def test_all_fields_custom_values(self):
        """Test all auto-ingestion fields with custom values."""
        config = AutoIngestionConfig(
            enabled=False,
            auto_create_watches=False,
            include_common_files=False,
            include_source_files=True,
            target_collection_suffix="custom",
            max_files_per_batch=10,
            batch_delay_seconds=5.0,
            max_file_size_mb=100,
            debounce_seconds=20
        )

        assert config.enabled is False
        assert config.auto_create_watches is False
        assert config.include_common_files is False
        assert config.include_source_files is True
        assert config.target_collection_suffix == "custom"
        assert config.max_files_per_batch == 10
        assert config.batch_delay_seconds == 5.0
        assert config.max_file_size_mb == 100
        assert config.debounce_seconds == 20

    def test_empty_target_collection_suffix(self):
        """Test handling of empty target collection suffix."""
        config = AutoIngestionConfig(target_collection_suffix="")
        assert config.target_collection_suffix == ""


class TestConfigComprehensive:
    """Comprehensive tests for the main Config class."""

    def test_yaml_config_loading_basic(self):
        """Test basic YAML configuration loading."""
        yaml_content = {
            "host": "yaml.host",
            "port": 9999,
            "debug": True,
            "qdrant": {
                "url": "https://yaml.qdrant.io",
                "api_key": "yaml-key",
                "timeout": 60
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(yaml_content, f)
            yaml_file = f.name

        try:
            config = Config(config_file=yaml_file)
            assert config.host == "yaml.host"
            assert config.port == 9999
            assert config.debug is True
            assert config.qdrant.url == "https://yaml.qdrant.io"
            assert config.qdrant.api_key == "yaml-key"
            assert config.qdrant.timeout == 60
        finally:
            os.unlink(yaml_file)

    def test_yaml_config_file_not_found(self):
        """Test error handling when YAML config file not found."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            Config(config_file="/nonexistent/config.yaml")

    def test_yaml_config_not_a_file(self):
        """Test error handling when config path is not a file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValueError, match="Configuration path is not a file"):
                Config(config_file=temp_dir)

    def test_yaml_config_invalid_yaml(self):
        """Test error handling for malformed YAML."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            yaml_file = f.name

        try:
            with pytest.raises(yaml.YAMLError):
                Config(config_file=yaml_file)
        finally:
            os.unlink(yaml_file)

    def test_yaml_config_not_dict(self):
        """Test error handling when YAML is not a dictionary."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(["list", "instead", "of", "dict"], f)
            yaml_file = f.name

        try:
            with pytest.raises(ValueError, match="YAML configuration must be a dictionary"):
                Config(config_file=yaml_file)
        finally:
            os.unlink(yaml_file)

    def test_yaml_config_empty_file(self):
        """Test handling of empty YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")
            yaml_file = f.name

        try:
            config = Config(config_file=yaml_file)
            # Should use defaults when YAML is empty
            assert config.host == "127.0.0.1"
            assert config.port == 8000
        finally:
            os.unlink(yaml_file)

    def test_yaml_config_none_content(self):
        """Test handling of YAML file with None content."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("~")  # YAML for None
            yaml_file = f.name

        try:
            config = Config(config_file=yaml_file)
            # Should use defaults when YAML loads as None
            assert config.host == "127.0.0.1"
        finally:
            os.unlink(yaml_file)

    def test_from_yaml_classmethod(self):
        """Test Config.from_yaml class method."""
        yaml_content = {"host": "classmethod.host", "port": 5555}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(yaml_content, f)
            yaml_file = f.name

        try:
            config = Config.from_yaml(yaml_file, debug=True)
            assert config.host == "classmethod.host"
            assert config.port == 5555
            assert config.debug is True  # From kwargs override
        finally:
            os.unlink(yaml_file)

    def test_to_yaml_method(self):
        """Test configuration export to YAML."""
        config = Config(host="export.host", port=7777, debug=True)
        config.qdrant.api_key = "export-key"
        config.workspace.github_user = "export-user"

        yaml_str = config.to_yaml()

        # Parse the YAML to verify content
        parsed = yaml.safe_load(yaml_str)
        assert parsed["host"] == "export.host"
        assert parsed["port"] == 7777
        assert parsed["debug"] is True
        assert parsed["qdrant"]["api_key"] == "export-key"
        assert parsed["workspace"]["github_user"] == "export-user"

    def test_to_yaml_with_file_path(self):
        """Test YAML export to file."""
        config = Config(host="file.host")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_file = f.name

        try:
            yaml_str = config.to_yaml(file_path=yaml_file)

            # Verify file was written
            assert Path(yaml_file).exists()
            with open(yaml_file, 'r') as f:
                file_content = f.read()
            assert file_content == yaml_str

            # Verify content
            parsed = yaml.safe_load(file_content)
            assert parsed["host"] == "file.host"
        finally:
            os.unlink(yaml_file)

    @patch('workspace_qdrant_mcp.core.config.Config._find_default_config_file')
    def test_auto_discover_config_file(self, mock_find):
        """Test automatic config file discovery."""
        mock_find.return_value = None  # No config file found

        config = Config()
        assert config.host == "127.0.0.1"  # Should use defaults

        mock_find.assert_called_once()

    @patch('workspace_qdrant_mcp.core.config.Config._find_default_config_file')
    @patch('workspace_qdrant_mcp.core.config.Config._load_yaml_config')
    def test_auto_discover_config_file_found(self, mock_load, mock_find):
        """Test automatic config file discovery when file is found."""
        mock_find.return_value = "/auto/config.yaml"
        mock_load.return_value = {"host": "auto.host"}

        config = Config()

        mock_find.assert_called_once()
        mock_load.assert_called_once_with("/auto/config.yaml")

    def test_nested_env_vars_all_types(self):
        """Test all nested environment variable types."""
        env_vars = {
            # Qdrant config
            "WORKSPACE_QDRANT_QDRANT__URL": "https://nested.qdrant.io",
            "WORKSPACE_QDRANT_QDRANT__API_KEY": "nested-key",
            "WORKSPACE_QDRANT_QDRANT__TIMEOUT": "90",
            "WORKSPACE_QDRANT_QDRANT__PREFER_GRPC": "false",

            # Embedding config
            "WORKSPACE_QDRANT_EMBEDDING__MODEL": "nested-model",
            "WORKSPACE_QDRANT_EMBEDDING__ENABLE_SPARSE_VECTORS": "false",
            "WORKSPACE_QDRANT_EMBEDDING__CHUNK_SIZE": "1200",
            "WORKSPACE_QDRANT_EMBEDDING__CHUNK_OVERLAP": "300",
            "WORKSPACE_QDRANT_EMBEDDING__BATCH_SIZE": "75",

            # Workspace config
            "WORKSPACE_QDRANT_WORKSPACE__COLLECTION_TYPES": "docs,notes,code",
            "WORKSPACE_QDRANT_WORKSPACE__GLOBAL_COLLECTIONS": "shared,common",
            "WORKSPACE_QDRANT_WORKSPACE__GITHUB_USER": "nested-user",
            "WORKSPACE_QDRANT_WORKSPACE__AUTO_CREATE_COLLECTIONS": "true",
            "WORKSPACE_QDRANT_WORKSPACE__MEMORY_COLLECTION_NAME": "__nested_memory",
            "WORKSPACE_QDRANT_WORKSPACE__CODE_COLLECTION_NAME": "__nested_code",
            "WORKSPACE_QDRANT_WORKSPACE__CUSTOM_INCLUDE_PATTERNS": "*.md,*.txt",
            "WORKSPACE_QDRANT_WORKSPACE__CUSTOM_EXCLUDE_PATTERNS": "*.log,*.tmp",

            # Auto-ingestion config
            "WORKSPACE_QDRANT_AUTO_INGESTION__ENABLED": "false",
            "WORKSPACE_QDRANT_AUTO_INGESTION__AUTO_CREATE_WATCHES": "false",
            "WORKSPACE_QDRANT_AUTO_INGESTION__INCLUDE_COMMON_FILES": "false",
            "WORKSPACE_QDRANT_AUTO_INGESTION__INCLUDE_SOURCE_FILES": "true",
            "WORKSPACE_QDRANT_AUTO_INGESTION__TARGET_COLLECTION_SUFFIX": "nested",
            "WORKSPACE_QDRANT_AUTO_INGESTION__MAX_FILES_PER_BATCH": "15",
            "WORKSPACE_QDRANT_AUTO_INGESTION__BATCH_DELAY_SECONDS": "3.5",
            "WORKSPACE_QDRANT_AUTO_INGESTION__MAX_FILE_SIZE_MB": "75",
            "WORKSPACE_QDRANT_AUTO_INGESTION__DEBOUNCE_SECONDS": "15",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            config = Config()

            # Verify Qdrant config
            assert config.qdrant.url == "https://nested.qdrant.io"
            assert config.qdrant.api_key == "nested-key"
            assert config.qdrant.timeout == 90
            assert config.qdrant.prefer_grpc is False

            # Verify embedding config
            assert config.embedding.model == "nested-model"
            assert config.embedding.enable_sparse_vectors is False
            assert config.embedding.chunk_size == 1200
            assert config.embedding.chunk_overlap == 300
            assert config.embedding.batch_size == 75

            # Verify workspace config
            assert config.workspace.collection_types == ["docs", "notes", "code"]
            assert config.workspace.global_collections == ["shared", "common"]
            assert config.workspace.github_user == "nested-user"
            assert config.workspace.auto_create_collections is True
            assert config.workspace.memory_collection_name == "__nested_memory"
            assert config.workspace.code_collection_name == "__nested_code"
            assert config.workspace.custom_include_patterns == ["*.md", "*.txt"]
            assert config.workspace.custom_exclude_patterns == ["*.log", "*.tmp"]

            # Verify auto-ingestion config
            assert config.auto_ingestion.enabled is False
            assert config.auto_ingestion.auto_create_watches is False
            assert config.auto_ingestion.include_common_files is False
            assert config.auto_ingestion.include_source_files is True
            assert config.auto_ingestion.target_collection_suffix == "nested"
            assert config.auto_ingestion.max_files_per_batch == 15
            assert config.auto_ingestion.batch_delay_seconds == 3.5
            assert config.auto_ingestion.max_file_size_mb == 75
            assert config.auto_ingestion.debounce_seconds == 15

    def test_legacy_env_vars_all_types(self):
        """Test all legacy environment variable types."""
        env_vars = {
            "QDRANT_URL": "https://legacy.qdrant.io",
            "QDRANT_API_KEY": "legacy-key",
            "FASTEMBED_MODEL": "legacy-model",
            "ENABLE_SPARSE_VECTORS": "true",
            "CHUNK_SIZE": "1000",
            "CHUNK_OVERLAP": "250",
            "BATCH_SIZE": "100",
            "COLLECTION_TYPES": "legacy1,legacy2",
            "GLOBAL_COLLECTIONS": "legacy_global",
            "GITHUB_USER": "legacy-user",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            config = Config()

            assert config.qdrant.url == "https://legacy.qdrant.io"
            assert config.qdrant.api_key == "legacy-key"
            assert config.embedding.model == "legacy-model"
            assert config.embedding.enable_sparse_vectors is True
            assert config.embedding.chunk_size == 1000
            assert config.embedding.chunk_overlap == 250
            assert config.embedding.batch_size == 100
            assert config.workspace.collection_types == ["legacy1", "legacy2"]
            assert config.workspace.global_collections == ["legacy_global"]
            assert config.workspace.github_user == "legacy-user"

    def test_comma_separated_list_parsing_empty_values(self):
        """Test parsing comma-separated lists with empty values."""
        env_vars = {
            "WORKSPACE_QDRANT_WORKSPACE__COLLECTION_TYPES": "docs,,notes,",
            "WORKSPACE_QDRANT_WORKSPACE__GLOBAL_COLLECTIONS": ",shared,,common,",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            config = Config()

            # Should filter out empty strings
            assert config.workspace.collection_types == ["docs", "notes"]
            assert config.workspace.global_collections == ["shared", "common"]

    def test_qdrant_client_config_complete(self):
        """Test complete Qdrant client configuration generation."""
        config = Config()
        config.qdrant.url = "https://test.qdrant.io"
        config.qdrant.api_key = "test-key"
        config.qdrant.timeout = 45
        config.qdrant.prefer_grpc = True

        client_config = config.qdrant_client_config

        expected = {
            "url": "https://test.qdrant.io",
            "timeout": 45,
            "prefer_grpc": True,
            "api_key": "test-key"
        }
        assert client_config == expected

    def test_qdrant_client_config_without_api_key(self):
        """Test Qdrant client config when API key is None."""
        config = Config()
        config.qdrant.api_key = None

        client_config = config.qdrant_client_config
        assert "api_key" not in client_config

    def test_validate_config_comprehensive(self):
        """Test comprehensive configuration validation."""
        config = Config()

        # Test invalid URL
        config.qdrant.url = "invalid-url"
        issues = config.validate_config()
        assert any("must start with http://" in issue for issue in issues)

        # Reset URL
        config.qdrant.url = "http://localhost:6333"

        # Test chunk size validation
        config.embedding.chunk_size = -1
        issues = config.validate_config()
        assert any("Chunk size must be positive" in issue for issue in issues)

        config.embedding.chunk_size = 15000
        issues = config.validate_config()
        assert any("should not exceed 10000" in issue for issue in issues)

        # Reset chunk size
        config.embedding.chunk_size = 800

        # Test batch size validation
        config.embedding.batch_size = 0
        issues = config.validate_config()
        assert any("Batch size must be positive" in issue for issue in issues)

        config.embedding.batch_size = 2000
        issues = config.validate_config()
        assert any("should not exceed 1000" in issue for issue in issues)

        # Reset batch size
        config.embedding.batch_size = 50

        # Test chunk overlap validation
        config.embedding.chunk_overlap = -1
        issues = config.validate_config()
        assert any("must be non-negative" in issue for issue in issues)

        config.embedding.chunk_overlap = 900  # Greater than chunk_size
        issues = config.validate_config()
        assert any("must be less than chunk size" in issue for issue in issues)

    def test_validate_config_limits(self):
        """Test validation of various limits."""
        config = Config()

        # Test collection types limit
        config.workspace.collection_types = [f"type{i}" for i in range(25)]
        issues = config.validate_config()
        assert any("Too many collection types" in issue for issue in issues)

        # Test global collections limit
        config.workspace.global_collections = [f"global{i}" for i in range(55)]
        issues = config.validate_config()
        assert any("Too many global collections" in issue for issue in issues)

        # Test custom pattern limits
        config.workspace.custom_include_patterns = [f"pattern{i}" for i in range(105)]
        issues = config.validate_config()
        assert any("Too many custom include patterns" in issue for issue in issues)

        config.workspace.custom_exclude_patterns = [f"exclude{i}" for i in range(105)]
        issues = config.validate_config()
        assert any("Too many custom exclude patterns" in issue for issue in issues)

        # Test custom project indicators limit
        config.workspace.custom_project_indicators = {f"indicator{i}": {"pattern": "test"} for i in range(25)}
        issues = config.validate_config()
        assert any("Too many custom project indicators" in issue for issue in issues)

    def test_validate_custom_project_indicators(self):
        """Test validation of custom project indicators structure."""
        config = Config()

        # Test non-dict indicator
        config.workspace.custom_project_indicators = {"bad": "not-a-dict"}
        issues = config.validate_config()
        assert any("must be a dictionary" in issue for issue in issues)

        # Test missing pattern field
        config.workspace.custom_project_indicators = {"missing_pattern": {"confidence": 0.8}}
        issues = config.validate_config()
        assert any("missing required 'pattern' field" in issue for issue in issues)

        # Test invalid confidence value
        config.workspace.custom_project_indicators = {
            "bad_confidence": {"pattern": "test", "confidence": 1.5}
        }
        issues = config.validate_config()
        assert any("confidence must be a number between 0.0 and 1.0" in issue for issue in issues)

        # Test non-numeric confidence
        config.workspace.custom_project_indicators = {
            "string_confidence": {"pattern": "test", "confidence": "invalid"}
        }
        issues = config.validate_config()
        assert any("confidence must be a number between 0.0 and 1.0" in issue for issue in issues)

    def test_auto_ingestion_validation_scenarios(self):
        """Test auto-ingestion validation scenarios."""
        config = Config()
        config.auto_ingestion.enabled = True

        # Scenario 1: target_suffix specified but not in collection_types
        config.auto_ingestion.target_collection_suffix = "docs"
        config.workspace.collection_types = ["notes", "code"]
        config.workspace.auto_create_collections = False
        issues = config.validate_config()
        assert any("not in workspace.collection_types" in issue for issue in issues)

        # Scenario 2: empty target_suffix but collection_types exist
        config.auto_ingestion.target_collection_suffix = ""
        config.workspace.collection_types = ["docs", "notes"]
        issues = config.validate_config()
        assert any("target_collection_suffix is empty but workspace.collection_types" in issue for issue in issues)

    @patch('workspace_qdrant_mcp.core.config.logger')
    def test_auto_ingestion_graceful_fallback_warning(self, mock_logger):
        """Test auto-ingestion graceful fallback warning."""
        config = Config()
        config.auto_ingestion.enabled = True
        config.auto_ingestion.target_collection_suffix = ""
        config.workspace.collection_types = []
        config.workspace.auto_create_collections = False

        issues = config.validate_config()

        # Should log warning but not add to issues (graceful fallback)
        mock_logger.warning.assert_called()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "Auto-ingestion enabled without explicit collection configuration" in warning_msg

    def test_get_auto_ingestion_diagnostics_all_scenarios(self):
        """Test auto-ingestion diagnostics for all scenarios."""
        config = Config()

        # Scenario: disabled
        config.auto_ingestion.enabled = False
        diag = config.get_auto_ingestion_diagnostics()
        assert diag["configuration_status"] == "disabled"
        assert "Auto-ingestion is disabled" in diag["summary"]

        # Scenario: invalid target suffix
        config.auto_ingestion.enabled = True
        config.auto_ingestion.target_collection_suffix = "missing"
        config.workspace.collection_types = ["docs", "notes"]
        diag = config.get_auto_ingestion_diagnostics()
        assert diag["configuration_status"] == "invalid_target_suffix"
        assert "not found in configured types" in diag["summary"]

        # Scenario: missing collection config
        config.auto_ingestion.target_collection_suffix = "docs"
        config.workspace.collection_types = []
        config.workspace.auto_create_collections = False
        diag = config.get_auto_ingestion_diagnostics()
        assert diag["configuration_status"] == "missing_collection_config"

        # Scenario: missing target suffix
        config.auto_ingestion.target_collection_suffix = ""
        config.workspace.collection_types = ["docs", "notes"]
        diag = config.get_auto_ingestion_diagnostics()
        assert diag["configuration_status"] == "missing_target_suffix"

        # Scenario: no collection config
        config.workspace.collection_types = []
        diag = config.get_auto_ingestion_diagnostics()
        assert diag["configuration_status"] == "no_collection_config"

        # Scenario: valid
        config.auto_ingestion.target_collection_suffix = "docs"
        config.workspace.collection_types = ["docs", "notes"]
        diag = config.get_auto_ingestion_diagnostics()
        assert diag["configuration_status"] == "valid"

    @patch('workspace_qdrant_mcp.core.config.Config._current_project_name')
    def test_get_effective_auto_ingestion_behavior(self, mock_project_name):
        """Test effective auto-ingestion behavior descriptions."""
        mock_project_name.return_value = "test-project"
        config = Config()

        # Disabled
        config.auto_ingestion.enabled = False
        behavior = config.get_effective_auto_ingestion_behavior()
        assert "Auto-ingestion is disabled" in behavior

        # Valid configuration
        config.auto_ingestion.enabled = True
        config.auto_ingestion.target_collection_suffix = "docs"
        config.workspace.collection_types = ["docs", "notes"]
        behavior = config.get_effective_auto_ingestion_behavior()
        assert "Will use collection 'test-project-docs'" in behavior

        # Auto-create scenario
        config.workspace.collection_types = []
        config.workspace.auto_create_collections = True
        behavior = config.get_effective_auto_ingestion_behavior()
        assert "Will create and use collection 'test-project-docs'" in behavior

        # Fallback scenario
        config.auto_ingestion.target_collection_suffix = ""
        config.workspace.auto_create_collections = False
        behavior = config.get_effective_auto_ingestion_behavior()
        assert "intelligent fallback selection" in behavior

        # Configuration issue
        config.auto_ingestion.target_collection_suffix = "missing"
        config.workspace.collection_types = ["docs"]
        behavior = config.get_effective_auto_ingestion_behavior()
        assert "Configuration may need adjustment" in behavior

    @patch('workspace_qdrant_mcp.core.config.ProjectDetector')
    def test_current_project_name_success(self, mock_detector_class):
        """Test successful project name detection."""
        mock_detector = Mock()
        mock_detector.get_project_info.return_value = {"main_project": "detected-project"}
        mock_detector_class.return_value = mock_detector

        config = Config()
        project_name = config._current_project_name()
        assert project_name == "detected-project"

    @patch('workspace_qdrant_mcp.core.config.ProjectDetector')
    def test_current_project_name_exception(self, mock_detector_class):
        """Test project name detection with exception."""
        mock_detector_class.side_effect = Exception("Import error")

        config = Config()
        project_name = config._current_project_name()
        assert project_name == "current-project"

    def test_find_default_config_file_none(self):
        """Test when no default config file is found."""
        config = Config()

        with patch.object(config, '_get_xdg_config_dirs', return_value=[]):
            with patch('pathlib.Path.cwd', return_value=Path("/nonexistent")):
                result = config._find_default_config_file()
                assert result is None

    @patch('platform.system')
    def test_get_xdg_config_dirs_all_platforms(self, mock_system):
        """Test XDG config directory detection for all platforms."""
        config = Config()

        # Test with XDG_CONFIG_HOME set
        with patch.dict(os.environ, {"XDG_CONFIG_HOME": "/custom/config"}):
            dirs = config._get_xdg_config_dirs()
            assert Path("/custom/config/workspace-qdrant") in dirs

        # Test macOS
        mock_system.return_value = "Darwin"
        with patch.dict(os.environ, {}, clear=True):
            with patch('pathlib.Path.home', return_value=Path("/Users/test")):
                dirs = config._get_xdg_config_dirs()
                expected = Path("/Users/test/Library/Application Support/workspace-qdrant")
                assert expected in dirs

        # Test Windows
        mock_system.return_value = "Windows"
        with patch.dict(os.environ, {"APPDATA": "/Users/test/AppData/Roaming"}, clear=True):
            dirs = config._get_xdg_config_dirs()
            expected = Path("/Users/test/AppData/Roaming/workspace-qdrant")
            assert expected in dirs

        # Test Windows without APPDATA
        with patch.dict(os.environ, {}, clear=True):
            with patch('pathlib.Path.home', return_value=Path("/Users/test")):
                dirs = config._get_xdg_config_dirs()
                expected = Path("/Users/test/AppData/Roaming/workspace-qdrant")
                assert expected in dirs

        # Test Linux
        mock_system.return_value = "Linux"
        with patch.dict(os.environ, {}, clear=True):
            with patch('pathlib.Path.home', return_value=Path("/home/test")):
                dirs = config._get_xdg_config_dirs()
                expected = Path("/home/test/.config/workspace-qdrant")
                assert expected in dirs

    @patch('workspace_qdrant_mcp.core.config.logger')
    def test_find_default_config_file_found_scenarios(self, mock_logger):
        """Test different scenarios of finding default config files."""
        config = Config()

        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "workspace-qdrant"
            config_dir.mkdir()

            # Test XDG standard config.yaml
            config_file = config_dir / "config.yaml"
            config_file.write_text("test: config")

            with patch.object(config, '_get_xdg_config_dirs', return_value=[config_dir]):
                result = config._find_default_config_file()
                assert result == str(config_file)
                mock_logger.info.assert_called_with(f"Auto-discovered XDG configuration file: {config_file}")

    def test_process_yaml_structure_comprehensive(self):
        """Test comprehensive YAML structure processing."""
        config = Config()

        # Test nested configurations
        yaml_data = {
            "host": "processed.host",
            "qdrant": {
                "url": "https://processed.qdrant.io",
                "timeout_ms": 60000,  # Should convert to seconds
                "transport": "grpc"   # Should set prefer_grpc
            },
            "embedding": {
                "model": "processed-model",
                "batch_size": 25
            },
            "workspace": {
                "collection_types": ["processed"],
                "github_user": "processed-user"
            },
            "auto_ingestion": {
                "enabled": True,
                "target_collection_suffix": "processed"
            },
            "grpc": {
                "enabled": True,
                "port": 50052
            },
            "daemon_specific_setting": "ignored",  # Should be ignored
            "unknown_section": {"data": "ignored"}  # Should be ignored
        }

        processed = config._process_yaml_structure(yaml_data)

        assert processed["host"] == "processed.host"
        assert isinstance(processed["qdrant"], QdrantConfig)
        assert processed["qdrant"].url == "https://processed.qdrant.io"
        assert processed["qdrant"].timeout == 60  # Converted from ms
        assert processed["qdrant"].prefer_grpc is True  # From transport
        assert isinstance(processed["embedding"], EmbeddingConfig)
        assert isinstance(processed["workspace"], WorkspaceConfig)
        assert isinstance(processed["auto_ingestion"], AutoIngestionConfig)
        assert isinstance(processed["grpc"], GrpcConfig)

        # Daemon-specific settings should not be in processed config
        assert "daemon_specific_setting" not in processed
        assert "unknown_section" not in processed

    def test_filter_qdrant_config(self):
        """Test Qdrant config filtering."""
        config = Config()

        daemon_config = {
            "url": "https://filtered.qdrant.io",
            "api_key": "filtered-key",
            "timeout_ms": 45000,
            "prefer_grpc": False,
            "transport": "http",
            "daemon_only_setting": "ignored"
        }

        filtered = config._filter_qdrant_config(daemon_config)

        assert filtered["url"] == "https://filtered.qdrant.io"
        assert filtered["api_key"] == "filtered-key"
        assert filtered["timeout"] == 45  # Converted from ms
        assert filtered["prefer_grpc"] is False
        assert "daemon_only_setting" not in filtered

        # Test transport-based prefer_grpc setting
        daemon_config_grpc = {"transport": "grpc"}
        filtered_grpc = config._filter_qdrant_config(daemon_config_grpc)
        assert filtered_grpc["prefer_grpc"] is True

    def test_filter_auto_ingestion_config(self):
        """Test auto-ingestion config filtering."""
        config = Config()

        daemon_config = {
            "enabled": True,
            "auto_create_watches": False,
            "include_common_files": True,
            "target_collection_suffix": "filtered",
            "max_files_per_batch": 10,
            "daemon_only_setting": "ignored"
        }

        filtered = config._filter_auto_ingestion_config(daemon_config)

        assert filtered["enabled"] is True
        assert filtered["auto_create_watches"] is False
        assert filtered["include_common_files"] is True
        assert filtered["target_collection_suffix"] == "filtered"
        assert filtered["max_files_per_batch"] == 10
        assert "daemon_only_setting" not in filtered

    @patch('workspace_qdrant_mcp.core.config.logger')
    def test_migrate_workspace_config(self, mock_logger):
        """Test workspace config migration with deprecated fields."""
        config = Config()

        # Test collection_suffixes -> collection_types migration
        workspace_config = {
            "collection_suffixes": ["old1", "old2"],
            "collection_prefix": "deprecated",
            "max_collections": 10
        }

        migrated = config._migrate_workspace_config(workspace_config)

        assert migrated["collection_types"] == ["old1", "old2"]
        assert "collection_suffixes" not in migrated
        assert "collection_prefix" not in migrated
        assert "max_collections" not in migrated

        # Check warnings were logged
        assert mock_logger.warning.call_count >= 2  # One for each deprecated field

    @patch('workspace_qdrant_mcp.core.config.logger')
    def test_migrate_workspace_config_both_fields(self, mock_logger):
        """Test migration when both old and new fields are present."""
        config = Config()

        workspace_config = {
            "collection_suffixes": ["old1", "old2"],
            "collection_types": ["new1", "new2"]
        }

        migrated = config._migrate_workspace_config(workspace_config)

        # Should use collection_types and ignore collection_suffixes
        assert migrated["collection_types"] == ["new1", "new2"]
        assert "collection_suffixes" not in migrated

        # Should warn about ignored collection_suffixes
        mock_logger.warning.assert_called()

    @patch('workspace_qdrant_mcp.core.config.logger')
    def test_migrate_auto_ingestion_config(self, mock_logger):
        """Test auto-ingestion config migration."""
        config = Config()

        auto_ingestion_config = {
            "enabled": True,
            "recursive_depth": 5,  # Deprecated
            "target_collection_suffix": "docs"
        }

        migrated = config._migrate_auto_ingestion_config(auto_ingestion_config)

        assert migrated["enabled"] is True
        assert migrated["target_collection_suffix"] == "docs"
        assert "recursive_depth" not in migrated

        # Should warn about deprecated field
        mock_logger.warning.assert_called()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "recursive_depth" in warning_msg

    def test_apply_yaml_overrides(self):
        """Test YAML configuration overrides."""
        config = Config()
        original_host = config.host

        # Test simple override
        yaml_config = {"host": "overridden.host", "port": 9999}
        config._apply_yaml_overrides(yaml_config)

        assert config.host == "overridden.host"
        assert config.port == 9999

        # Test that typed config objects are not overridden by raw dicts
        yaml_config_with_object = {
            "auto_ingestion": {"enabled": False}  # This should be skipped
        }
        config._apply_yaml_overrides(yaml_config_with_object)

        # auto_ingestion should still be an AutoIngestionConfig object
        assert isinstance(config.auto_ingestion, AutoIngestionConfig)

    def test_yaml_config_exception_handling(self):
        """Test YAML config loading exception handling."""
        config = Config()

        # Test generic exception during file loading
        with patch('pathlib.Path.open', side_effect=PermissionError("Access denied")):
            with pytest.raises(ValueError, match="Error loading configuration file"):
                config._load_yaml_config("test.yaml")


class TestConfigEdgeCases:
    """Test edge cases and error conditions."""

    def test_environment_variable_type_conversion_errors(self):
        """Test handling of invalid environment variable values."""
        # These should not raise exceptions but may produce unexpected results
        env_vars = {
            "WORKSPACE_QDRANT_QDRANT__TIMEOUT": "not-a-number",
            "WORKSPACE_QDRANT_EMBEDDING__CHUNK_SIZE": "invalid",
            "WORKSPACE_QDRANT_EMBEDDING__BATCH_SIZE": "also-invalid"
        }

        with patch.dict(os.environ, env_vars, clear=False):
            # Should handle gracefully (may use defaults or raise appropriate errors)
            try:
                Config()
            except (ValueError, TypeError):
                # Expected for invalid type conversions
                pass

    def test_boolean_environment_variable_edge_cases(self):
        """Test boolean environment variable parsing edge cases."""
        test_cases = [
            ("TRUE", True),
            ("True", True),
            ("true", True),
            ("FALSE", False),
            ("False", False),
            ("false", False),
            ("1", False),  # Only "true" (case-insensitive) should be True
            ("0", False),
            ("yes", False),
            ("no", False),
            ("", False),
        ]

        for env_value, expected in test_cases:
            env_vars = {"WORKSPACE_QDRANT_QDRANT__PREFER_GRPC": env_value}
            with patch.dict(os.environ, env_vars, clear=False):
                config = Config()
                assert config.qdrant.prefer_grpc == expected, f"Failed for '{env_value}'"

    def test_empty_environment_variables(self):
        """Test handling of empty environment variables."""
        env_vars = {
            "WORKSPACE_QDRANT_QDRANT__URL": "",
            "WORKSPACE_QDRANT_EMBEDDING__MODEL": "",
            "WORKSPACE_QDRANT_WORKSPACE__GITHUB_USER": "",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            config = Config()
            # Empty strings should be set as empty strings
            assert config.qdrant.url == ""
            assert config.embedding.model == ""
            assert config.workspace.github_user == ""

    def test_kwargs_override_precedence(self):
        """Test that kwargs take precedence over other sources."""
        yaml_content = {"host": "yaml.host", "port": 8888}
        env_vars = {"WORKSPACE_QDRANT_HOST": "env.host"}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(yaml_content, f)
            yaml_file = f.name

        try:
            with patch.dict(os.environ, env_vars, clear=False):
                config = Config(config_file=yaml_file, host="kwargs.host", debug=True)

                # kwargs should win
                assert config.host == "kwargs.host"
                assert config.debug is True
                # YAML should win over env for non-kwargs fields
                assert config.port == 8888
        finally:
            os.unlink(yaml_file)

    def test_model_dump_serialization(self):
        """Test model serialization for debugging."""
        config = Config(host="serialize.host", debug=True)
        config.qdrant.api_key = "serialize-key"
        config.workspace.github_user = "serialize-user"

        # Test that model_dump works (from Pydantic)
        config_dict = config.model_dump()

        assert isinstance(config_dict, dict)
        assert config_dict["host"] == "serialize.host"
        assert config_dict["debug"] is True
        assert config_dict["qdrant"]["api_key"] == "serialize-key"
        assert config_dict["workspace"]["github_user"] == "serialize-user"


if __name__ == "__main__":
    pytest.main([__file__])