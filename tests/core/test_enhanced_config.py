"""
Tests for enhanced configuration management system.

This module provides comprehensive tests for the environment-based configuration
system including YAML file loading, environment variable overrides, validation,
hot-reload functionality, and security features.
"""

import sys
from pathlib import Path

# Add src/python to path for common module imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

import os
import tempfile
import pytest
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

# Skip yaml tests if PyYAML not available
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from workspace_qdrant_mcp.core.enhanced_config import (
    EnhancedConfig,
    EmbeddingConfig,
    QdrantConfig,
    WorkspaceConfig,
    SecurityConfig,
    MonitoringConfig,
    PerformanceConfig,
    DevelopmentConfig
)


class TestEmbeddingConfig:
    """Test embedding configuration validation."""
    
    def test_valid_embedding_config(self):
        """Test valid embedding configuration."""
        config = EmbeddingConfig(
            chunk_size=800,
            chunk_overlap=120,
            batch_size=50
        )
        assert config.chunk_size == 800
        assert config.chunk_overlap == 120
        assert config.batch_size == 50
    
    def test_chunk_overlap_validation(self):
        """Test chunk overlap validation."""
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            EmbeddingConfig(chunk_size=100, chunk_overlap=150)
    
    def test_batch_size_validation(self):
        """Test batch size validation."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            EmbeddingConfig(batch_size=0)
        
        with pytest.raises(ValueError, match="batch_size should not exceed 1000"):
            EmbeddingConfig(batch_size=1500)


class TestQdrantConfig:
    """Test Qdrant configuration validation."""
    
    def test_valid_qdrant_config(self):
        """Test valid Qdrant configuration."""
        config = QdrantConfig(
            url="https://example.com:6333",
            timeout=30
        )
        assert config.url == "https://example.com:6333"
        assert config.timeout == 30
    
    def test_url_validation(self):
        """Test URL format validation."""
        with pytest.raises(ValueError, match="URL must start with http:// or https://"):
            QdrantConfig(url="invalid-url")
    
    def test_timeout_validation(self):
        """Test timeout validation."""
        with pytest.raises(ValueError, match="timeout must be positive"):
            QdrantConfig(timeout=0)
        
        with pytest.raises(ValueError, match="timeout should not exceed 300 seconds"):
            QdrantConfig(timeout=500)


class TestWorkspaceConfig:
    """Test workspace configuration validation."""
    
    def test_valid_workspace_config(self):
        """Test valid workspace configuration."""
        config = WorkspaceConfig(
            collections=["project", "docs"],
            global_collections=["shared", "references"]
        )
        assert config.collections == ["project", "docs"]
        assert config.global_collections == ["shared", "references"]
    
    def test_empty_collections_validation(self):
        """Test empty collections validation."""
        with pytest.raises(ValueError, match="At least one collection must be configured"):
            WorkspaceConfig(collections=[])
    
    def test_too_many_collections_validation(self):
        """Test too many collections validation."""
        too_many = [f"collection_{i}" for i in range(60)]
        with pytest.raises(ValueError, match="Too many collections configured"):
            WorkspaceConfig(collections=too_many)


class TestEnhancedConfig:
    """Test enhanced configuration system."""
    
    def setup_method(self):
        """Set up test environment."""
        # Clear relevant environment variables
        env_vars_to_clear = [
            "APP_ENV", "WORKSPACE_QDRANT_HOST", "WORKSPACE_QDRANT_PORT",
            "WORKSPACE_QDRANT_QDRANT__URL", "QDRANT_URL", "FASTEMBED_MODEL"
        ]
        for var in env_vars_to_clear:
            if var in os.environ:
                del os.environ[var]
    
    def test_default_configuration(self):
        """Test default configuration loading."""
        config = EnhancedConfig()
        assert config.environment == "development"
        assert config.host == "127.0.0.1"
        assert config.port == 8000
        assert config.qdrant.url == "http://localhost:6333"
        assert config.embedding.model == "sentence-transformers/all-MiniLM-L6-v2"
    
    def test_environment_from_env_var(self):
        """Test environment setting from environment variable."""
        with patch.dict(os.environ, {"APP_ENV": "production"}):
            config = EnhancedConfig()
            assert config.environment == "production"
    
    def test_environment_parameter(self):
        """Test environment setting from parameter."""
        config = EnhancedConfig(environment="staging")
        assert config.environment == "staging"
    
    def test_legacy_env_vars(self):
        """Test legacy environment variable loading."""
        with patch.dict(os.environ, {
            "QDRANT_URL": "http://legacy:6333",
            "FASTEMBED_MODEL": "legacy-model",
            "GITHUB_USER": "legacy-user"
        }):
            config = EnhancedConfig()
            assert config.qdrant.url == "http://legacy:6333"
            assert config.embedding.model == "legacy-model"
            assert config.workspace.github_user == "legacy-user"
    
    def test_nested_env_vars(self):
        """Test nested environment variable loading."""
        with patch.dict(os.environ, {
            "WORKSPACE_QDRANT_QDRANT__URL": "http://nested:6333",
            "WORKSPACE_QDRANT_EMBEDDING__CHUNK_SIZE": "1000",
            "WORKSPACE_QDRANT_WORKSPACE__GITHUB_USER": "nested-user"
        }):
            config = EnhancedConfig()
            assert config.qdrant.url == "http://nested:6333"
            assert config.embedding.chunk_size == 1000
            assert config.workspace.github_user == "nested-user"
    
    def test_validation_success(self):
        """Test successful configuration validation."""
        config = EnhancedConfig()
        assert config.is_valid
        assert len(config.validation_errors) == 0
    
    def test_validation_failures(self):
        """Test configuration validation failures."""
        config = EnhancedConfig(
            environment="invalid-env",
            debug=True
        )
        config.environment = "production"  # Set after init to test production validation
        config.debug = True  # This should cause a production validation error
        
        errors = config.validate_config()
        assert len(errors) > 0
        assert any("Debug mode should be disabled in production" in error for error in errors)
    
    def test_qdrant_client_config(self):
        """Test Qdrant client configuration generation."""
        config = EnhancedConfig()
        config.qdrant.url = "https://test:6333"
        config.qdrant.api_key = "test-key"
        config.qdrant.timeout = 45
        
        client_config = config.qdrant_client_config
        
        assert client_config["url"] == "https://test:6333"
        assert client_config["api_key"] == "test-key"
        assert client_config["timeout"] == 45
        assert client_config["prefer_grpc"] is False
    
    def test_qdrant_client_config_no_api_key(self):
        """Test Qdrant client configuration without API key."""
        config = EnhancedConfig()
        client_config = config.qdrant_client_config
        assert "api_key" not in client_config
    
    def test_mask_sensitive_value(self):
        """Test sensitive value masking."""
        config = EnhancedConfig()
        config.security.mask_sensitive_logs = True
        
        # Test normal masking
        assert config.mask_sensitive_value("secretvalue") == "se****ue"
        
        # Test short values
        assert config.mask_sensitive_value("abc") == "***"
        
        # Test empty values
        assert config.mask_sensitive_value("") == ""
        
        # Test with masking disabled
        config.security.mask_sensitive_logs = False
        assert config.mask_sensitive_value("secretvalue") == "secretvalue"
    
    def test_config_summary(self):
        """Test configuration summary generation."""
        config = EnhancedConfig(environment="test")
        summary = config.get_config_summary()
        
        assert summary["environment"] == "test"
        assert summary["validation_status"] in ["valid", "invalid"]
        assert "validation_errors" in summary
        assert "qdrant_url" in summary
        assert "embedding_model" in summary


@pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not available")
class TestYAMLConfiguration:
    """Test YAML configuration file loading."""
    
    def test_yaml_config_loading(self):
        """Test loading configuration from YAML file."""
        yaml_content = """
server:
  host: "0.0.0.0"
  port: 9000
  debug: true

qdrant:
  url: "http://yaml-test:6333"
  timeout: 45

embedding:
  model: "yaml-model"
  chunk_size: 1200
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            yaml_file = config_dir / "development.yaml"
            
            with open(yaml_file, 'w') as f:
                f.write(yaml_content)
            
            config = EnhancedConfig(environment="development", config_dir=config_dir)
            
            assert config.host == "0.0.0.0"
            assert config.port == 9000
            assert config.debug is True
            assert config.qdrant.url == "http://yaml-test:6333"
            assert config.qdrant.timeout == 45
            assert config.embedding.model == "yaml-model"
            assert config.embedding.chunk_size == 1200
    
    def test_yaml_env_var_substitution(self):
        """Test environment variable substitution in YAML files."""
        yaml_content = """
server:
  host: "${SERVER_HOST:-localhost}"
  port: ${SERVER_PORT:-8000}

qdrant:
  url: "${QDRANT_URL}"
  api_key: "${QDRANT_API_KEY:-}"
"""
        with patch.dict(os.environ, {
            "SERVER_HOST": "yaml-host",
            "SERVER_PORT": "9001",
            "QDRANT_URL": "http://env-test:6333"
        }):
            with tempfile.TemporaryDirectory() as temp_dir:
                config_dir = Path(temp_dir)
                yaml_file = config_dir / "development.yaml"
                
                with open(yaml_file, 'w') as f:
                    f.write(yaml_content)
                
                config = EnhancedConfig(environment="development", config_dir=config_dir)
                
                assert config.host == "yaml-host"
                assert config.port == 9001
                assert config.qdrant.url == "http://env-test:6333"
                assert config.qdrant.api_key == ""
    
    def test_local_yaml_override(self):
        """Test local.yaml overriding environment configuration."""
        env_yaml = """
server:
  port: 8000
  debug: false

qdrant:
  url: "http://env:6333"
"""
        local_yaml = """
server:
  port: 8001
  debug: true

qdrant:
  timeout: 60
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            
            # Create environment config
            with open(config_dir / "development.yaml", 'w') as f:
                f.write(env_yaml)
            
            # Create local override
            with open(config_dir / "local.yaml", 'w') as f:
                f.write(local_yaml)
            
            config = EnhancedConfig(environment="development", config_dir=config_dir)
            
            # Local overrides should take precedence
            assert config.port == 8001
            assert config.debug is True
            assert config.qdrant.timeout == 60
            # Environment values should remain for non-overridden settings
            assert config.qdrant.url == "http://env:6333"


@pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not available")  
class TestConfigurationReload:
    """Test configuration hot-reload functionality."""
    
    def test_manual_config_reload(self):
        """Test manual configuration reload."""
        yaml_content_v1 = """
server:
  port: 8000
"""
        yaml_content_v2 = """
server:
  port: 9000
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            yaml_file = config_dir / "development.yaml"
            
            # Create initial config
            with open(yaml_file, 'w') as f:
                f.write(yaml_content_v1)
            
            config = EnhancedConfig(environment="development", config_dir=config_dir)
            assert config.port == 8000
            
            # Update config file
            with open(yaml_file, 'w') as f:
                f.write(yaml_content_v2)
            
            # Reload configuration
            config.reload_config()
            assert config.port == 9000


class TestConfigurationSecurity:
    """Test configuration security features."""
    
    def test_production_security_validation(self):
        """Test production environment security validation."""
        config = EnhancedConfig(environment="production")
        config.debug = True
        config.security.allow_http = True
        config.qdrant.url = "http://insecure:6333"
        config.security.mask_sensitive_logs = False
        
        errors = config.validate_config()
        
        assert any("Debug mode should be disabled in production" in error for error in errors)
        assert any("HTTPS should be used in production" in error for error in errors)
        assert any("Sensitive log masking should be enabled in production" in error for error in errors)
    
    def test_cors_validation(self):
        """Test CORS configuration validation."""
        config = EnhancedConfig()
        config.security.cors_enabled = True
        config.security.cors_origins = []
        
        errors = config.validate_config()
        assert any("CORS origins must be specified when CORS is enabled" in error for error in errors)


class TestConfigurationProfiles:
    """Test configuration profile functionality."""
    
    def test_environment_detection(self):
        """Test automatic environment detection."""
        environments = ["development", "staging", "production"]
        
        for env in environments:
            with patch.dict(os.environ, {"APP_ENV": env}):
                config = EnhancedConfig()
                assert config.environment == env
    
    def test_invalid_environment(self):
        """Test invalid environment handling."""
        config = EnhancedConfig(environment="invalid")
        errors = config.validate_config()
        assert any("Invalid environment" in error for error in errors)


@pytest.mark.integration
class TestConfigurationIntegration:
    """Integration tests for configuration system."""
    
    def test_full_configuration_loading(self):
        """Test complete configuration loading with all sources."""
        yaml_content = """
server:
  host: "yaml-host"
  port: 8080

qdrant:
  url: "${QDRANT_URL:-http://yaml:6333}"

development:
  hot_reload: true
"""
        with patch.dict(os.environ, {
            "APP_ENV": "development",
            "QDRANT_URL": "http://env-override:6333",
            "WORKSPACE_QDRANT_PORT": "9999",
            "FASTEMBED_MODEL": "legacy-model"
        }):
            with tempfile.TemporaryDirectory() as temp_dir:
                config_dir = Path(temp_dir)
                
                with open(config_dir / "development.yaml", 'w') as f:
                    f.write(yaml_content)
                
                config = EnhancedConfig(config_dir=config_dir)
                
                # Verify environment detection
                assert config.environment == "development"
                
                # Verify YAML loading
                assert config.host == "yaml-host"
                
                # Verify env var substitution in YAML
                assert config.qdrant.url == "http://env-override:6333"
                
                # Verify direct env var override (highest priority)
                assert config.port == 9999
                
                # Verify legacy env var support
                assert config.embedding.model == "legacy-model"
                
                # Verify YAML-specific settings
                assert config.development.hot_reload is True
    
    def test_configuration_validation_comprehensive(self):
        """Test comprehensive configuration validation."""
        # Create a config with multiple validation issues
        config = EnhancedConfig(environment="production")
        config.debug = True  # Should be false in production
        config.qdrant.url = "invalid-url"  # Invalid URL format
        config.embedding.chunk_overlap = 1000  # Greater than chunk_size
        config.embedding.chunk_size = 800
        config.embedding.batch_size = 2000  # Too large
        config.workspace.collections = []  # Empty collections
        config.security.cors_enabled = True
        config.security.cors_origins = []  # CORS enabled but no origins
        
        errors = config.validate_config()
        
        # Should have multiple validation errors
        assert len(errors) >= 5
        assert not config.is_valid


if __name__ == "__main__":
    pytest.main([__file__])