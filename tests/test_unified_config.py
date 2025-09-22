"""
Test suite for unified configuration system.

This module tests the UnifiedConfigManager class and its integration with
both Python MCP server and Rust daemon configuration formats.
"""

import sys
from pathlib import Path

# Add src/python to path for common module imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

import pytest
import tempfile
import yaml
import toml
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from workspace_qdrant_mcp.core.unified_config import (
    UnifiedConfigManager,
    ConfigFormat,
    ConfigValidationError,
    ConfigFormatError,
)
from workspace_qdrant_mcp.core.config import Config


class TestUnifiedConfigManager:
    """Test suite for UnifiedConfigManager."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def sample_toml_config(self):
        """Sample TOML configuration."""
        return """
# Test TOML configuration
log_file = "/tmp/test.log"
max_concurrent_tasks = 4
enable_lsp = true
chunk_size = 1000
log_level = "info"

[qdrant]
url = "http://localhost:6333"
timeout_ms = 30000
transport = "Http"

[auto_ingestion]
enabled = true
target_collection_suffix = "test"
"""

    @pytest.fixture
    def sample_yaml_config(self):
        """Sample YAML configuration."""
        return """
# Test YAML configuration
host: "127.0.0.1"
port: 8000
debug: false

qdrant:
  url: "http://localhost:6333"
  timeout: 30
  prefer_grpc: false

embedding:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  chunk_size: 800
  chunk_overlap: 120

workspace:
  collection_suffixes: ["test"]
  auto_create_collections: false

auto_ingestion:
  enabled: true
  target_collection_suffix: "test"
"""

    @pytest.fixture
    def sample_json_config(self):
        """Sample JSON configuration."""
        return json.dumps({
            "host": "127.0.0.1",
            "port": 8000,
            "debug": False,
            "qdrant": {
                "url": "http://localhost:6333",
                "timeout": 30
            },
            "embedding": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "chunk_size": 800
            }
        }, indent=2)

    def test_format_detection(self):
        """Test automatic format detection."""
        assert ConfigFormat.TOML == UnifiedConfigManager()._detect_format(Path("config.toml"))
        assert ConfigFormat.YAML == UnifiedConfigManager()._detect_format(Path("config.yaml"))
        assert ConfigFormat.YAML == UnifiedConfigManager()._detect_format(Path("config.yml"))
        assert ConfigFormat.JSON == UnifiedConfigManager()._detect_format(Path("config.json"))
        assert ConfigFormat.YAML == UnifiedConfigManager()._detect_format(Path("config"))  # Default

    def test_config_source_discovery(self, temp_dir):
        """Test configuration source discovery."""
        config_manager = UnifiedConfigManager(config_dir=temp_dir)
        
        # Create test config files
        toml_file = temp_dir / "workspace_qdrant_config.toml"
        yaml_file = temp_dir / "workspace_qdrant_config.yaml"
        
        toml_file.write_text("# TOML config")
        yaml_file.write_text("# YAML config")
        
        config_manager._discover_config_sources()
        
        existing_sources = [s for s in config_manager.config_sources if s.exists]
        assert len(existing_sources) >= 2
        
        # Should prefer based on order
        preferred = config_manager.get_preferred_config_source()
        assert preferred is not None
        assert preferred.exists

    def test_toml_config_loading(self, temp_dir, sample_toml_config):
        """Test loading TOML configuration."""
        config_manager = UnifiedConfigManager(config_dir=temp_dir)
        toml_file = temp_dir / "workspace_qdrant_config.toml"
        toml_file.write_text(sample_toml_config)
        
        config = config_manager.load_config(config_file=toml_file)
        
        assert isinstance(config, Config)
        assert config.qdrant.url == "http://localhost:6333"

    def test_yaml_config_loading(self, temp_dir, sample_yaml_config):
        """Test loading YAML configuration."""
        config_manager = UnifiedConfigManager(config_dir=temp_dir)
        yaml_file = temp_dir / "workspace_qdrant_config.yaml"
        yaml_file.write_text(sample_yaml_config)
        
        config = config_manager.load_config(config_file=yaml_file)
        
        assert isinstance(config, Config)
        assert config.host == "127.0.0.1"
        assert config.port == 8000
        assert config.qdrant.url == "http://localhost:6333"
        assert config.embedding.model == "sentence-transformers/all-MiniLM-L6-v2"

    def test_json_config_loading(self, temp_dir, sample_json_config):
        """Test loading JSON configuration."""
        config_manager = UnifiedConfigManager(config_dir=temp_dir)
        json_file = temp_dir / "config.json"
        json_file.write_text(sample_json_config)
        
        config = config_manager.load_config(config_file=json_file)
        
        assert isinstance(config, Config)
        assert config.host == "127.0.0.1"
        assert config.port == 8000

    def test_environment_variable_overrides(self, temp_dir, sample_yaml_config):
        """Test environment variable overrides."""
        config_manager = UnifiedConfigManager(config_dir=temp_dir)
        yaml_file = temp_dir / "workspace_qdrant_config.yaml"
        yaml_file.write_text(sample_yaml_config)
        
        with patch.dict(os.environ, {
            'WORKSPACE_QDRANT_HOST': '192.168.1.100',
            'WORKSPACE_QDRANT_PORT': '9000',
            'WORKSPACE_QDRANT_QDRANT__URL': 'http://remote:6333',
            'WORKSPACE_QDRANT_DEBUG': 'true'
        }):
            config = config_manager.load_config(config_file=yaml_file)
            
            assert config.host == '192.168.1.100'
            assert config.port == 9000
            assert config.qdrant.url == 'http://remote:6333'
            assert config.debug is True

    def test_config_validation(self, temp_dir):
        """Test configuration validation."""
        config_manager = UnifiedConfigManager(config_dir=temp_dir)
        
        # Create invalid config
        invalid_yaml = """
host: "127.0.0.1"
port: 8000
qdrant:
  url: "invalid-url"  # Invalid URL
embedding:
  chunk_size: -1  # Invalid chunk size
"""
        yaml_file = temp_dir / "invalid_config.yaml"
        yaml_file.write_text(invalid_yaml)
        
        with pytest.raises(ConfigValidationError):
            config_manager.load_config(config_file=yaml_file)

    def test_config_file_validation(self, temp_dir, sample_yaml_config):
        """Test standalone config file validation."""
        config_manager = UnifiedConfigManager(config_dir=temp_dir)
        yaml_file = temp_dir / "workspace_qdrant_config.yaml"
        yaml_file.write_text(sample_yaml_config)
        
        issues = config_manager.validate_config_file(yaml_file)
        assert isinstance(issues, list)
        # Should be valid, so no issues
        assert len(issues) == 0

    def test_config_saving_toml(self, temp_dir, sample_yaml_config):
        """Test saving configuration to TOML format."""
        config_manager = UnifiedConfigManager(config_dir=temp_dir)
        
        # Load from YAML
        yaml_file = temp_dir / "source.yaml"
        yaml_file.write_text(sample_yaml_config)
        config = config_manager.load_config(config_file=yaml_file)
        
        # Save to TOML
        toml_file = temp_dir / "output.toml"
        config_manager.save_config(config, toml_file, ConfigFormat.TOML)
        
        assert toml_file.exists()
        
        # Verify TOML content can be loaded back
        loaded_toml = toml.loads(toml_file.read_text())
        assert "qdrant" in loaded_toml
        assert loaded_toml["host"] == "127.0.0.1"

    def test_config_saving_yaml(self, temp_dir):
        """Test saving configuration to YAML format."""
        config_manager = UnifiedConfigManager(config_dir=temp_dir)
        
        # Create config from defaults
        config = Config()
        
        # Save to YAML
        yaml_file = temp_dir / "output.yaml"
        config_manager.save_config(config, yaml_file, ConfigFormat.YAML)
        
        assert yaml_file.exists()
        
        # Verify YAML content can be loaded back
        loaded_yaml = yaml.safe_load(yaml_file.read_text())
        assert isinstance(loaded_yaml, dict)
        assert "qdrant" in loaded_yaml

    def test_config_conversion(self, temp_dir, sample_toml_config):
        """Test configuration format conversion."""
        config_manager = UnifiedConfigManager(config_dir=temp_dir)
        
        # Create source TOML file
        source_file = temp_dir / "source.toml"
        source_file.write_text(sample_toml_config)
        
        # Convert to YAML
        target_file = temp_dir / "target.yaml"
        config_manager.convert_config(source_file, target_file, ConfigFormat.YAML)
        
        assert target_file.exists()
        
        # Verify conversion worked
        yaml_data = yaml.safe_load(target_file.read_text())
        assert "qdrant" in yaml_data
        assert yaml_data["qdrant"]["url"] == "http://localhost:6333"

    def test_config_info(self, temp_dir):
        """Test configuration information retrieval."""
        config_manager = UnifiedConfigManager(config_dir=temp_dir)
        
        # Create some test files
        toml_file = temp_dir / "workspace_qdrant_config.toml"
        yaml_file = temp_dir / "workspace_qdrant_config.yaml"
        toml_file.write_text("# TOML")
        yaml_file.write_text("# YAML")
        
        info = config_manager.get_config_info()
        
        assert "config_dir" in info
        assert "env_prefix" in info
        assert "sources" in info
        assert info["env_prefix"] == "WORKSPACE_QDRANT_"
        assert str(temp_dir) == info["config_dir"]

    def test_default_config_creation(self, temp_dir):
        """Test creating default configuration files."""
        config_manager = UnifiedConfigManager(config_dir=temp_dir)
        
        formats = [ConfigFormat.TOML, ConfigFormat.YAML]
        created_files = config_manager.create_default_configs(formats)
        
        assert len(created_files) == 2
        assert ConfigFormat.TOML in created_files
        assert ConfigFormat.YAML in created_files
        
        # Verify files exist and are valid
        for format_type, file_path in created_files.items():
            assert file_path.exists()
            issues = config_manager.validate_config_file(file_path)
            assert len(issues) == 0  # Should be valid

    def test_config_watching(self, temp_dir, sample_yaml_config):
        """Test configuration file watching."""
        config_manager = UnifiedConfigManager(config_dir=temp_dir)
        yaml_file = temp_dir / "workspace_qdrant_config.yaml"
        yaml_file.write_text(sample_yaml_config)
        
        callback_calls = []
        
        def test_callback(new_config):
            callback_calls.append(new_config)
        
        config_manager.watch_config(test_callback)
        
        # Modify file to trigger watch (this is a basic test)
        # In a real scenario, file system events would trigger the callback
        assert config_manager.observer is not None
        
        config_manager.stop_watching()
        assert config_manager.observer is None

    def test_error_handling_malformed_toml(self, temp_dir):
        """Test error handling for malformed TOML."""
        config_manager = UnifiedConfigManager(config_dir=temp_dir)
        toml_file = temp_dir / "invalid.toml"
        toml_file.write_text("invalid toml content [[[")
        
        with pytest.raises(ConfigFormatError):
            config_manager._load_config_file(toml_file)

    def test_error_handling_malformed_yaml(self, temp_dir):
        """Test error handling for malformed YAML."""
        config_manager = UnifiedConfigManager(config_dir=temp_dir)
        yaml_file = temp_dir / "invalid.yaml"
        yaml_file.write_text("invalid: yaml: content: [[[")
        
        with pytest.raises(ConfigFormatError):
            config_manager._load_config_file(yaml_file)

    def test_error_handling_nonexistent_file(self, temp_dir):
        """Test error handling for nonexistent files."""
        config_manager = UnifiedConfigManager(config_dir=temp_dir)
        nonexistent_file = temp_dir / "nonexistent.yaml"
        
        with pytest.raises(FileNotFoundError):
            config_manager.load_config(config_file=nonexistent_file)

    def test_auto_discovery_preference(self, temp_dir, sample_toml_config, sample_yaml_config):
        """Test auto-discovery format preference."""
        config_manager = UnifiedConfigManager(config_dir=temp_dir)
        
        # Create both TOML and YAML files
        toml_file = temp_dir / "workspace_qdrant_config.toml"
        yaml_file = temp_dir / "workspace_qdrant_config.yaml"
        toml_file.write_text(sample_toml_config)
        yaml_file.write_text(sample_yaml_config)
        
        # Should prefer TOML when no preference specified (first in pattern list)
        preferred_default = config_manager.get_preferred_config_source()
        assert preferred_default is not None
        assert preferred_default.file_path.suffix == ".toml"
        
        # Should prefer YAML when explicitly requested
        preferred_yaml = config_manager.get_preferred_config_source(ConfigFormat.YAML)
        assert preferred_yaml is not None
        assert preferred_yaml.file_path.suffix == ".yaml"

    def test_context_manager(self, temp_dir):
        """Test context manager functionality."""
        with UnifiedConfigManager(config_dir=temp_dir) as config_manager:
            assert config_manager is not None
            assert config_manager.config_dir == temp_dir
        # Context manager should cleanup watchers if any were started

    def test_config_dict_conversion(self, temp_dir, sample_yaml_config):
        """Test configuration to dictionary conversion."""
        config_manager = UnifiedConfigManager(config_dir=temp_dir)
        yaml_file = temp_dir / "workspace_qdrant_config.yaml"
        yaml_file.write_text(sample_yaml_config)
        
        config = config_manager.load_config(config_file=yaml_file)
        config_dict = config_manager._config_to_dict(config)
        
        assert isinstance(config_dict, dict)
        assert "qdrant" in config_dict
        assert "embedding" in config_dict
        assert "workspace" in config_dict
        assert "auto_ingestion" in config_dict
        
        # Verify nested structure
        assert isinstance(config_dict["qdrant"], dict)
        assert "url" in config_dict["qdrant"]

    @patch('os.getenv')
    def test_complex_env_overrides(self, mock_getenv, temp_dir, sample_yaml_config):
        """Test complex environment variable override scenarios."""
        config_manager = UnifiedConfigManager(config_dir=temp_dir)
        yaml_file = temp_dir / "workspace_qdrant_config.yaml"
        yaml_file.write_text(sample_yaml_config)
        
        # Mock environment variables
        env_vars = {
            'WORKSPACE_QDRANT_QDRANT__URL': 'http://prod:6333',
            'WORKSPACE_QDRANT_EMBEDDING__BATCH_SIZE': '100',
            'WORKSPACE_QDRANT_WORKSPACE__COLLECTION_SUFFIXES': 'prod,staging,dev',
            'WORKSPACE_QDRANT_AUTO_INGESTION__ENABLED': 'false',
        }
        
        def mock_getenv_side_effect(key, default=None):
            return env_vars.get(key, default)
        
        mock_getenv.side_effect = mock_getenv_side_effect
        
        config = config_manager.load_config(config_file=yaml_file)
        
        assert config.qdrant.url == 'http://prod:6333'
        assert config.embedding.batch_size == 100
        assert config.workspace.collection_suffixes == ['prod', 'staging', 'dev']
        assert config.auto_ingestion.enabled is False

    @patch('os.getenv')
    def test_collection_types_env_overrides(self, mock_getenv, temp_dir, sample_yaml_config):
        """Test new collection_types environment variable overrides."""
        config_manager = UnifiedConfigManager(config_dir=temp_dir)
        yaml_file = temp_dir / "workspace_qdrant_config.yaml"
        yaml_file.write_text(sample_yaml_config)

        # Mock environment variables with new collection_types field
        env_vars = {
            'WORKSPACE_QDRANT_QDRANT__URL': 'http://prod:6333',
            'WORKSPACE_QDRANT_WORKSPACE__COLLECTION_TYPES': 'docs,notes,scratchbook',
            'WORKSPACE_QDRANT_AUTO_INGESTION__ENABLED': 'false',
        }

        def mock_getenv_side_effect(key, default=None):
            return env_vars.get(key, default)

        mock_getenv.side_effect = mock_getenv_side_effect

        config = config_manager.load_config(config_file=yaml_file)

        assert config.qdrant.url == 'http://prod:6333'
        assert config.workspace.collection_types == ['docs', 'notes', 'scratchbook']
        assert config.workspace.effective_collection_types == ['docs', 'notes', 'scratchbook']
        assert config.auto_ingestion.enabled is False


@pytest.mark.integration
class TestUnifiedConfigIntegration:
    """Integration tests for unified configuration system."""
    
    def test_rust_daemon_compatibility(self, temp_dir):
        """Test compatibility with Rust daemon configuration format."""
        config_manager = UnifiedConfigManager(config_dir=temp_dir)
        
        # Create a TOML config that matches Rust daemon expectations
        rust_config = """
# Rust daemon compatible configuration
log_file = "/tmp/memexd.log"
max_concurrent_tasks = 8
default_timeout_ms = 60000
enable_preemption = true
chunk_size = 2000
enable_lsp = false
log_level = "debug"
enable_metrics = true
metrics_interval_secs = 30

[qdrant]
url = "http://localhost:6333"
transport = "Grpc"
timeout_ms = 45000
max_retries = 5
retry_delay_ms = 2000
pool_size = 20
tls = false
dense_vector_size = 1536

[auto_ingestion]
enabled = true
auto_create_watches = true
include_common_files = true
include_source_files = false
target_collection_suffix = "daemon"
max_files_per_batch = 10
"""
        
        toml_file = temp_dir / "rust_daemon_config.toml"
        toml_file.write_text(rust_config)
        
        # This should load without issues
        config = config_manager.load_config(config_file=toml_file)
        assert config is not None

    def test_python_mcp_compatibility(self, temp_dir):
        """Test compatibility with Python MCP server configuration format."""
        config_manager = UnifiedConfigManager(config_dir=temp_dir)
        
        # Create a YAML config that matches Python MCP server expectations
        python_config = """
host: "0.0.0.0"
port: 8080
debug: true

qdrant:
  url: "https://cloud.qdrant.io"
  api_key: "secret-key"
  timeout: 60
  prefer_grpc: true

embedding:
  model: "sentence-transformers/all-mpnet-base-v2"
  enable_sparse_vectors: true
  chunk_size: 1200
  chunk_overlap: 200
  batch_size: 32

workspace:
  collection_types: ["docs", "code", "notes"]
  global_collections: ["shared", "reference"]
  github_user: "testuser"
  collection_prefix: "proj_"
  max_collections: 50
  auto_create_collections: true

grpc:
  enabled: true
  host: "127.0.0.1"
  port: 50051
  fallback_to_direct: true

auto_ingestion:
  enabled: true
  auto_create_watches: false
  include_common_files: true
  include_source_files: true
  target_collection_suffix: "docs"
"""
        
        yaml_file = temp_dir / "python_mcp_config.yaml"
        yaml_file.write_text(python_config)
        
        # This should load without issues
        config = config_manager.load_config(config_file=yaml_file)
        assert config is not None
        assert config.host == "0.0.0.0"
        assert config.port == 8080
        assert config.debug is True

    def test_format_conversion_fidelity(self, temp_dir):
        """Test that format conversion preserves data fidelity."""
        config_manager = UnifiedConfigManager(config_dir=temp_dir)
        
        # Create comprehensive source config
        source_yaml = """
host: "test.example.com"
port: 9999
debug: false

qdrant:
  url: "http://test:6333"
  timeout: 45
  prefer_grpc: true

embedding:
  model: "test-model"
  chunk_size: 512
  batch_size: 64

workspace:
  collection_types: ["test1", "test2"]
  max_collections: 25
"""
        
        source_file = temp_dir / "source.yaml"
        source_file.write_text(source_yaml)
        
        # Convert YAML -> TOML
        toml_file = temp_dir / "converted.toml"
        config_manager.convert_config(source_file, toml_file, ConfigFormat.TOML)
        
        # Convert TOML -> JSON
        json_file = temp_dir / "converted.json"
        config_manager.convert_config(toml_file, json_file, ConfigFormat.JSON)
        
        # Load all three and compare key values
        yaml_config = config_manager.load_config(config_file=source_file)
        toml_config = config_manager.load_config(config_file=toml_file)
        json_config = config_manager.load_config(config_file=json_file)
        
        # Verify critical values are preserved
        configs = [yaml_config, toml_config, json_config]
        for config in configs:
            assert config.host == "test.example.com"
            assert config.port == 9999
            assert config.qdrant.url == "http://test:6333"
            assert config.embedding.chunk_size == 512


if __name__ == "__main__":
    pytest.main([__file__, "-v"])