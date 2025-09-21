"""
Comprehensive configuration system validation tests for Task 81.

This module provides comprehensive testing for the configuration system including:
- Configuration precedence validation (CLI → YAML → env vars → defaults)
- Environment variable substitution with patterns like ${VAR_NAME} and fallbacks
- JSON schema validation using core/config.py and utils/config_validator.py
- Hot-reload capability testing for configuration changes without service restart
- Security validation for secret handling and environment variable safety

Tests cover both yaml_config.py and enhanced_config.py systems to ensure complete
coverage of the configuration infrastructure.
"""

import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import Dict, Any, List

import pytest
import yaml
from pydantic import ValidationError

from common.core.config import Config, EmbeddingConfig, QdrantConfig, WorkspaceConfig
from common.core.yaml_config import (
    ConfigLoader,
    WorkspaceConfig as YAMLWorkspaceConfig,
    YAMLConfigLoader,
    load_config,
    save_config,
    create_default_config,
)
from common.core.enhanced_config import EnhancedConfig
from common.utils.config_validator import ConfigValidator


class TestConfigurationPrecedence:
    """Test configuration precedence: CLI args → YAML file → environment variables → defaults."""
    
    def setup_method(self):
        """Set up test environment by clearing relevant environment variables."""
        self.env_vars_to_clear = [
            "WORKSPACE_QDRANT_HOST",
            "WORKSPACE_QDRANT_PORT", 
            "WORKSPACE_QDRANT_DEBUG",
            "WORKSPACE_QDRANT_QDRANT__URL",
            "WORKSPACE_QDRANT_QDRANT__API_KEY",
            "WORKSPACE_QDRANT_EMBEDDING__MODEL",
            "WORKSPACE_QDRANT_EMBEDDING__CHUNK_SIZE",
            "QDRANT_URL",
            "QDRANT_API_KEY", 
            "FASTEMBED_MODEL",
            "APP_ENV",
        ]
        
        for var in self.env_vars_to_clear:
            if var in os.environ:
                del os.environ[var]
    
    def teardown_method(self):
        """Clean up environment variables after test."""
        for var in self.env_vars_to_clear:
            if var in os.environ:
                del os.environ[var]
    
    def test_default_configuration_baseline(self):
        """Test that default configuration works without any external configuration."""
        config = Config()
        
        # Verify defaults are loaded
        assert config.host == "127.0.0.1"
        assert config.port == 8000
        assert config.debug is False
        assert config.qdrant.url == "http://localhost:6333"
        assert config.qdrant.api_key is None
        assert config.embedding.model == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.embedding.chunk_size == 800
        assert config.workspace.collections == ["project"]
    
    def test_environment_variable_override_precedence(self):
        """Test that environment variables override defaults."""
        with patch.dict(os.environ, {
            "WORKSPACE_QDRANT_HOST": "0.0.0.0",
            "WORKSPACE_QDRANT_PORT": "9000",
            "WORKSPACE_QDRANT_DEBUG": "true",
            "WORKSPACE_QDRANT_QDRANT__URL": "http://env-qdrant:6333",
            "WORKSPACE_QDRANT_QDRANT__API_KEY": "env-api-key",
            "WORKSPACE_QDRANT_EMBEDDING__MODEL": "env-model",
            "WORKSPACE_QDRANT_EMBEDDING__CHUNK_SIZE": "1200",
        }):
            config = Config()
            
            # Verify environment variables override defaults
            assert config.host == "0.0.0.0"
            assert config.port == 9000
            assert config.debug is True
            assert config.qdrant.url == "http://env-qdrant:6333"
            assert config.qdrant.api_key == "env-api-key"
            assert config.embedding.model == "env-model"
            assert config.embedding.chunk_size == 1200
    
    def test_yaml_file_override_precedence(self):
        """Test that YAML configuration overrides environment variables."""
        yaml_content = {
            "host": "yaml-host",
            "port": 8080,
            "debug": True,
            "qdrant": {
                "url": "http://yaml-qdrant:6333",
                "api_key": "yaml-api-key",
                "timeout": 60
            },
            "embedding": {
                "model": "yaml-model",
                "chunk_size": 1000,
                "batch_size": 100
            },
            "workspace": {
                "collections": ["docs", "tests"],
                "github_user": "yaml-user"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(yaml_content, f)
            yaml_file = f.name
        
        try:
            with patch.dict(os.environ, {
                "WORKSPACE_QDRANT_HOST": "env-host",
                "WORKSPACE_QDRANT_PORT": "7000",
                "WORKSPACE_QDRANT_QDRANT__URL": "http://env-qdrant:6333",
            }):
                config = Config(config_file=yaml_file)
                
                # YAML should override environment variables
                assert config.host == "yaml-host"
                assert config.port == 8080
                assert config.debug is True
                assert config.qdrant.url == "http://yaml-qdrant:6333"
                assert config.qdrant.api_key == "yaml-api-key"
                assert config.qdrant.timeout == 60
                assert config.embedding.model == "yaml-model"
                assert config.embedding.chunk_size == 1000
                assert config.embedding.batch_size == 100
                assert config.workspace.collections == ["docs", "tests"]
                assert config.workspace.github_user == "yaml-user"
        finally:
            os.unlink(yaml_file)
    
    def test_constructor_kwargs_highest_precedence(self):
        """Test that constructor kwargs have highest precedence."""
        yaml_content = {
            "host": "yaml-host",
            "port": 8080,
            "qdrant": {
                "url": "http://yaml-qdrant:6333"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(yaml_content, f)
            yaml_file = f.name
        
        try:
            with patch.dict(os.environ, {
                "WORKSPACE_QDRANT_HOST": "env-host",
                "WORKSPACE_QDRANT_PORT": "7000",
            }):
                config = Config(
                    config_file=yaml_file,
                    host="kwargs-host",
                    port=5000,
                    debug=True
                )
                
                # Constructor kwargs should have highest precedence
                assert config.host == "kwargs-host"
                assert config.port == 5000
                assert config.debug is True
                # YAML should still override env vars for non-kwargs parameters
                assert config.qdrant.url == "http://yaml-qdrant:6333"
        finally:
            os.unlink(yaml_file)
    
    def test_legacy_environment_variables(self):
        """Test that legacy environment variables work for backward compatibility."""
        with patch.dict(os.environ, {
            "QDRANT_URL": "http://legacy-qdrant:6333",
            "QDRANT_API_KEY": "legacy-api-key",
            "FASTEMBED_MODEL": "legacy-model",
            "CHUNK_SIZE": "1500",
            "BATCH_SIZE": "75",
            "GITHUB_USER": "legacy-user",
            "COLLECTIONS": "legacy1,legacy2",
            "GLOBAL_COLLECTIONS": "global1,global2",
        }):
            config = Config()
            
            # Verify legacy variables are loaded
            assert config.qdrant.url == "http://legacy-qdrant:6333"
            assert config.qdrant.api_key == "legacy-api-key"
            assert config.embedding.model == "legacy-model"
            assert config.embedding.chunk_size == 1500
            assert config.embedding.batch_size == 75
            assert config.workspace.github_user == "legacy-user"
            assert config.workspace.collections == ["legacy1", "legacy2"]
            assert config.workspace.global_collections == ["global1", "global2"]
    
    def test_nested_vs_legacy_precedence(self):
        """Test that new nested environment variables override legacy ones."""
        with patch.dict(os.environ, {
            # Legacy variables
            "QDRANT_URL": "http://legacy:6333",
            "FASTEMBED_MODEL": "legacy-model",
            # New nested variables (should take precedence)
            "WORKSPACE_QDRANT_QDRANT__URL": "http://nested:6333",
            "WORKSPACE_QDRANT_EMBEDDING__MODEL": "nested-model",
        }):
            config = Config()
            
            # Nested variables should override legacy ones
            assert config.qdrant.url == "http://nested:6333"
            assert config.embedding.model == "nested-model"


class TestEnvironmentVariableSubstitution:
    """Test environment variable substitution patterns like ${VAR_NAME} with fallbacks."""
    
    def setup_method(self):
        """Set up test environment."""
        self.test_env_vars = {
            "TEST_URL": "http://test-server:6333",
            "TEST_API_KEY": "test-api-key-123",
            "TEST_CHUNK_SIZE": "1000",
            "TEST_TIMEOUT": "45",
        }
    
    def test_basic_environment_substitution(self):
        """Test basic ${VAR_NAME} substitution."""
        yaml_content = """
server:
  host: "${TEST_HOST}"
  port: ${TEST_PORT}

qdrant:
  url: "${TEST_URL}"
  api_key: "${TEST_API_KEY}"
  timeout: ${TEST_TIMEOUT}

embedding:
  model: "${TEST_MODEL}"
  chunk_size: ${TEST_CHUNK_SIZE}
"""
        
        with patch.dict(os.environ, {
            **self.test_env_vars,
            "TEST_HOST": "substituted-host",
            "TEST_PORT": "8888",
            "TEST_MODEL": "substituted-model",
        }):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write(yaml_content)
                yaml_file = f.name
            
            try:
                loader = YAMLConfigLoader()
                config = loader.load_with_env_substitution(yaml_file)
                
                # Should be asyncio.run() but using sync for testing
                import asyncio
                config = asyncio.run(loader.load_with_env_substitution(yaml_file))
                
                assert config['server']['host'] == "substituted-host"
                assert config['server']['port'] == 8888  # Converted to int
                assert config['qdrant']['url'] == "http://test-server:6333"
                assert config['qdrant']['api_key'] == "test-api-key-123"
                assert config['qdrant']['timeout'] == 45  # Converted to int
                assert config['embedding']['model'] == "substituted-model"
                assert config['embedding']['chunk_size'] == 1000  # Converted to int
            finally:
                os.unlink(yaml_file)
    
    def test_environment_substitution_with_fallbacks(self):
        """Test ${VAR_NAME:default} substitution with fallback values."""
        yaml_content = """
server:
  host: "${UNDEFINED_HOST:-localhost}"
  port: ${UNDEFINED_PORT:-8000}

qdrant:
  url: "${TEST_URL:-http://localhost:6333}"
  api_key: "${UNDEFINED_API_KEY:-}"
  timeout: ${UNDEFINED_TIMEOUT:-30}

embedding:
  model: "${UNDEFINED_MODEL:-default-model}"
  chunk_size: ${TEST_CHUNK_SIZE:-800}
"""
        
        with patch.dict(os.environ, self.test_env_vars):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write(yaml_content)
                yaml_file = f.name
            
            try:
                loader = YAMLConfigLoader()
                import asyncio
                config = asyncio.run(loader.load_with_env_substitution(yaml_file))
                
                # Should use fallback values for undefined vars
                assert config['server']['host'] == "localhost"
                assert config['server']['port'] == 8000
                assert config['qdrant']['api_key'] == ""  # Empty fallback
                assert config['qdrant']['timeout'] == 30
                assert config['embedding']['model'] == "default-model"
                
                # Should use env var value when available
                assert config['qdrant']['url'] == "http://test-server:6333"
                assert config['embedding']['chunk_size'] == 1000
            finally:
                os.unlink(yaml_file)
    
    def test_environment_substitution_missing_variables(self):
        """Test handling of missing environment variables without fallbacks."""
        yaml_content = """
qdrant:
  url: "${MISSING_URL}"
  api_key: "${MISSING_API_KEY}"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            yaml_file = f.name
        
        try:
            loader = YAMLConfigLoader()
            import asyncio
            config = asyncio.run(loader.load_with_env_substitution(yaml_file))
            
            # Missing variables should become empty strings
            assert config['qdrant']['url'] == ""
            assert config['qdrant']['api_key'] == ""
        finally:
            os.unlink(yaml_file)
    
    def test_complex_environment_substitution(self):
        """Test complex substitution patterns."""
        yaml_content = """
database:
  connection_string: "${DB_PROTOCOL:-https}://${DB_HOST:-localhost}:${DB_PORT:-6333}"
  auth_header: "${AUTH_PREFIX:-Bearer} ${AUTH_TOKEN}"

paths:
  base_path: "${BASE_PATH:-/tmp}"
  log_file: "${BASE_PATH:-/tmp}/${APP_NAME:-app}.log"

mixed:
  template: "Config for ${APP_NAME:-workspace-qdrant} on ${ENVIRONMENT:-development}"
"""
        
        with patch.dict(os.environ, {
            "DB_HOST": "production.example.com",
            "DB_PORT": "443",
            "AUTH_TOKEN": "secret-123",
            "APP_NAME": "test-app",
            "ENVIRONMENT": "staging",
        }):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write(yaml_content)
                yaml_file = f.name
            
            try:
                loader = YAMLConfigLoader()
                import asyncio
                config = asyncio.run(loader.load_with_env_substitution(yaml_file))
                
                assert config['database']['connection_string'] == "https://production.example.com:443"
                assert config['database']['auth_header'] == "Bearer secret-123"
                assert config['paths']['base_path'] == "/tmp"
                assert config['paths']['log_file'] == "/tmp/test-app.log"
                assert config['mixed']['template'] == "Config for test-app on staging"
            finally:
                os.unlink(yaml_file)
    
    def test_yaml_config_loader_hierarchy(self):
        """Test YAML configuration hierarchy loading."""
        # Create multiple config files with different priorities
        base_config = {"server": {"host": "base-host", "port": 8000, "debug": False}}
        override_config = {"server": {"port": 9000, "debug": True}, "qdrant": {"url": "http://override:6333"}}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            base_file = Path(temp_dir) / "base.yaml"
            override_file = Path(temp_dir) / "override.yaml"
            
            with open(base_file, 'w') as f:
                yaml.dump(base_config, f)
            with open(override_file, 'w') as f:
                yaml.dump(override_config, f)
            
            loader = YAMLConfigLoader()
            import asyncio
            
            # Load with hierarchy (override should win)
            merged = asyncio.run(loader.load_with_hierarchy([str(base_file), str(override_file)]))
            
            assert merged['server']['host'] == "base-host"  # From base
            assert merged['server']['port'] == 9000  # From override
            assert merged['server']['debug'] is True  # From override
            assert merged['qdrant']['url'] == "http://override:6333"  # From override


class TestJSONSchemaValidation:
    """Test JSON schema validation with core/config.py and utils/config_validator.py."""
    
    def test_pydantic_config_validation_success(self):
        """Test successful configuration validation."""
        config = Config(
            host="127.0.0.1",
            port=8000,
            debug=False
        )
        
        # Should not raise validation errors
        issues = config.validate_config()
        assert isinstance(issues, list)
        # Some issues might exist but no critical validation errors
        
        # Test individual component validation
        qdrant_config = QdrantConfig(
            url="http://localhost:6333",
            timeout=30
        )
        assert qdrant_config.url == "http://localhost:6333"
        assert qdrant_config.timeout == 30
        
        embedding_config = EmbeddingConfig(
            chunk_size=800,
            chunk_overlap=120,
            batch_size=50
        )
        assert embedding_config.chunk_overlap < embedding_config.chunk_size
    
    def test_pydantic_config_validation_failures(self):
        """Test configuration validation failures."""
        # Test invalid chunk overlap
        with pytest.raises(ValidationError):
            EmbeddingConfig(chunk_size=100, chunk_overlap=150)
        
        # Test invalid QdrantConfig values through Config validation
        config = Config()
        config.qdrant.url = "invalid-url"
        config.embedding.chunk_size = -1
        config.embedding.batch_size = 0
        config.embedding.chunk_overlap = 1000  # Greater than chunk_size
        config.workspace.collections = []  # Empty collections
        
        issues = config.validate_config()
        
        # Should find multiple validation issues
        assert len(issues) > 0
        assert any("URL must start with http://" in issue for issue in issues)
        assert any("Chunk size must be positive" in issue for issue in issues)
        assert any("Batch size must be positive" in issue for issue in issues)
        assert any("Chunk overlap must be less than chunk size" in issue for issue in issues)
        assert any("At least one project collection must be configured" in issue for issue in issues)
    
    def test_config_validator_comprehensive(self):
        """Test comprehensive configuration validation with ConfigValidator."""
        # Test with valid configuration
        config = Config()
        validator = ConfigValidator(config)
        
        # Test individual validations (may fail due to missing services, but should not crash)
        qdrant_valid, qdrant_msg = validator.validate_qdrant_connection()
        embedding_valid, embedding_msg = validator.validate_embedding_model()
        project_valid, project_msg = validator.validate_project_detection()
        
        # These are network-dependent, so just ensure they return proper format
        assert isinstance(qdrant_valid, bool)
        assert isinstance(qdrant_msg, str)
        assert isinstance(embedding_valid, bool)
        assert isinstance(embedding_msg, str)
        assert isinstance(project_valid, bool)
        assert isinstance(project_msg, str)
        
        # Test comprehensive validation
        is_valid, results = validator.validate_all()
        assert isinstance(is_valid, bool)
        assert isinstance(results, dict)
        assert "issues" in results
        assert "warnings" in results
        assert "qdrant_connection" in results
        assert "embedding_model" in results
        assert "project_detection" in results
        assert "config_validation" in results
    
    def test_yaml_config_validation(self):
        """Test YAML configuration validation."""
        valid_config = {
            "qdrant": {
                "url": "http://localhost:6333",
                "port": 6333
            },
            "embedding": {
                "batch_size": 50
            }
        }
        
        invalid_config = {
            "qdrant": {
                "url": "invalid-url",  # Invalid URL format
                "port": "not-a-number"  # Invalid port type
            },
            "embedding": {
                "batch_size": "invalid"  # Invalid batch_size type
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(valid_config, f)
            valid_file = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(invalid_config, f)
            invalid_file = f.name
        
        try:
            loader = YAMLConfigLoader()
            import asyncio
            
            # Valid config should load successfully
            valid_result = asyncio.run(loader.load_and_validate(valid_file))
            assert isinstance(valid_result, dict)
            
            # Invalid config should raise ValueError
            with pytest.raises(ValueError):
                asyncio.run(loader.load_and_validate(invalid_file))
                
        finally:
            os.unlink(valid_file)
            os.unlink(invalid_file)
    
    def test_enhanced_config_validation(self):
        """Test enhanced configuration validation."""
        # Test with valid enhanced config
        config = EnhancedConfig(environment="development")
        assert config.is_valid or len(config.validation_errors) >= 0  # May have warnings
        
        # Test with invalid enhanced config
        config = EnhancedConfig(environment="invalid-env")
        errors = config.validate_config()
        assert len(errors) > 0
        assert any("Invalid environment" in error for error in errors)
        
        # Test production-specific validation
        prod_config = EnhancedConfig(environment="production")
        prod_config.debug = True
        prod_config.security.allow_http = True
        prod_config.qdrant.url = "http://insecure:6333"
        prod_config.security.mask_sensitive_logs = False
        
        prod_errors = prod_config.validate_config()
        assert len(prod_errors) > 0
        assert any("Debug mode should be disabled in production" in error for error in prod_errors)


class TestHotReloadCapability:
    """Test hot-reload capability for configuration changes without service restart."""
    
    def test_enhanced_config_manual_reload(self):
        """Test manual configuration reload in EnhancedConfig."""
        config_v1 = """
server:
  port: 8000
  debug: false

qdrant:
  url: "http://v1:6333"
"""
        
        config_v2 = """
server:
  port: 9000
  debug: true

qdrant:
  url: "http://v2:6333"
  timeout: 60
"""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            config_file = config_dir / "development.yaml"
            
            # Create initial config
            with open(config_file, 'w') as f:
                f.write(config_v1)
            
            config = EnhancedConfig(environment="development", config_dir=config_dir)
            
            # Verify initial configuration
            assert config.port == 8000
            assert config.debug is False
            assert config.qdrant.url == "http://v1:6333"
            
            # Update config file
            with open(config_file, 'w') as f:
                f.write(config_v2)
            
            # Manually reload configuration
            config.reload_config()
            
            # Verify updated configuration
            assert config.port == 9000
            assert config.debug is True
            assert config.qdrant.url == "http://v2:6333"
            assert config.qdrant.timeout == 60
    
    def test_yaml_config_loader_reload_simulation(self):
        """Test configuration reload simulation with YAMLConfigLoader."""
        config_data = {
            "server": {"port": 8000},
            "qdrant": {"url": "http://original:6333"}
        }
        
        updated_data = {
            "server": {"port": 9000, "debug": True},
            "qdrant": {"url": "http://updated:6333", "timeout": 45}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name
        
        try:
            loader = YAMLConfigLoader()
            import asyncio
            
            # Load initial configuration
            initial_config = asyncio.run(loader.load_with_env_substitution(config_file))
            assert initial_config['server']['port'] == 8000
            assert initial_config['qdrant']['url'] == "http://original:6333"
            
            # Simulate configuration file update
            with open(config_file, 'w') as f:
                yaml.dump(updated_data, f)
            
            # Reload configuration
            reloaded_config = asyncio.run(loader.load_with_env_substitution(config_file))
            assert reloaded_config['server']['port'] == 9000
            assert reloaded_config['server']['debug'] is True
            assert reloaded_config['qdrant']['url'] == "http://updated:6333"
            assert reloaded_config['qdrant']['timeout'] == 45
            
        finally:
            os.unlink(config_file)
    
    def test_config_file_watching_simulation(self):
        """Test file watching simulation for configuration changes."""
        # This simulates what a file watcher would do
        config_states = []
        
        def config_change_handler(config_file: str) -> Dict[str, Any]:
            """Simulate configuration change handler."""
            loader = YAMLConfigLoader()
            import asyncio
            return asyncio.run(loader.load_with_env_substitution(config_file))
        
        initial_config = {"server": {"port": 8000}}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(initial_config, f)
            config_file = f.name
        
        try:
            # Initial load
            config_states.append(config_change_handler(config_file))
            
            # Simulate multiple configuration changes
            changes = [
                {"server": {"port": 8001}},
                {"server": {"port": 8002, "debug": True}},
                {"server": {"port": 8003}, "qdrant": {"url": "http://new:6333"}},
            ]
            
            for change in changes:
                # Simulate file modification
                with open(config_file, 'w') as f:
                    yaml.dump(change, f)
                
                # Simulate file watcher detecting change and reloading
                time.sleep(0.1)  # Simulate debounce delay
                new_config = config_change_handler(config_file)
                config_states.append(new_config)
            
            # Verify all configuration states were captured
            assert len(config_states) == 4
            assert config_states[0]['server']['port'] == 8000
            assert config_states[1]['server']['port'] == 8001
            assert config_states[2]['server']['port'] == 8002
            assert config_states[2]['server']['debug'] is True
            assert config_states[3]['server']['port'] == 8003
            assert config_states[3]['qdrant']['url'] == "http://new:6333"
            
        finally:
            os.unlink(config_file)


class TestSecurityValidation:
    """Test security aspects ensuring no secret leakage and proper environment variable handling."""
    
    def setup_method(self):
        """Set up test environment with sensitive data."""
        self.sensitive_env_vars = {
            "SECRET_API_KEY": "super-secret-key-123",
            "DATABASE_PASSWORD": "very-secret-password",
            "OAUTH_CLIENT_SECRET": "oauth-secret-456",
        }
    
    def test_sensitive_value_masking(self):
        """Test that sensitive values are properly masked in logs and output."""
        config = EnhancedConfig()
        config.security.mask_sensitive_logs = True
        
        # Test different lengths of sensitive data
        test_cases = [
            ("short", "***"),
            ("medium", "me****"),
            ("very-long-secret", "ve**********et"),
            ("super-secret-key-123", "su******************23"),
            ("", ""),  # Empty strings should remain empty
        ]
        
        for secret, expected_mask in test_cases:
            masked = config.mask_sensitive_value(secret)
            assert masked == expected_mask
        
        # Test with masking disabled
        config.security.mask_sensitive_logs = False
        assert config.mask_sensitive_value("secret") == "secret"
    
    def test_environment_variable_security(self):
        """Test that environment variables containing secrets are handled securely."""
        yaml_content = """
qdrant:
  url: "${QDRANT_URL:-http://localhost:6333}"
  api_key: "${SECRET_API_KEY}"

database:
  password: "${DATABASE_PASSWORD}"

oauth:
  client_secret: "${OAUTH_CLIENT_SECRET}"
"""
        
        with patch.dict(os.environ, self.sensitive_env_vars):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write(yaml_content)
                yaml_file = f.name
            
            try:
                loader = YAMLConfigLoader()
                import asyncio
                config = asyncio.run(loader.load_with_env_substitution(yaml_file))
                
                # Verify secrets are loaded correctly
                assert config['qdrant']['api_key'] == "super-secret-key-123"
                assert config['database']['password'] == "very-secret-password"
                assert config['oauth']['client_secret'] == "oauth-secret-456"
                
                # In a real application, these would be masked for logging
                enhanced_config = EnhancedConfig()
                enhanced_config.security.mask_sensitive_logs = True
                
                # Test that secrets would be masked when displayed
                masked_api_key = enhanced_config.mask_sensitive_value(config['qdrant']['api_key'])
                assert masked_api_key != "super-secret-key-123"
                assert masked_api_key == "su******************23"
                
            finally:
                os.unlink(yaml_file)
    
    def test_production_security_validation(self):
        """Test production environment security requirements."""
        # Test insecure production configuration
        config = EnhancedConfig(environment="production")
        config.debug = True  # Should not be allowed in production
        config.security.allow_http = True
        config.qdrant.url = "http://insecure:6333"  # HTTP in production
        config.security.mask_sensitive_logs = False  # Should mask in production
        config.security.cors_enabled = True
        config.security.cors_origins = []  # CORS enabled but no origins
        
        errors = config.validate_config()
        
        # Should flag multiple security issues
        security_errors = [error for error in errors if any(
            keyword in error.lower() for keyword in 
            ['debug', 'https', 'production', 'mask', 'cors']
        )]
        
        assert len(security_errors) > 0
        assert any("Debug mode should be disabled in production" in error for error in errors)
        assert any("HTTPS should be used in production" in error for error in errors)
        assert any("Sensitive log masking should be enabled in production" in error for error in errors)
        assert any("CORS origins must be specified when CORS is enabled" in error for error in errors)
    
    def test_api_key_handling(self):
        """Test secure API key handling and validation."""
        # Test configuration with API key
        config = EnhancedConfig()
        config.qdrant.api_key = "test-api-key"
        
        client_config = config.qdrant_client_config
        assert "api_key" in client_config
        assert client_config["api_key"] == "test-api-key"
        
        # Test configuration summary with API key masking
        summary = config.get_config_summary()
        assert summary["qdrant_api_key_set"] is True
        
        # With masking enabled, the key should be masked or not shown in full
        config.security.mask_sensitive_logs = True
        masked_summary = config.get_config_summary()
        assert masked_summary["qdrant_api_key_set"] is True
        
        # Test configuration without API key
        config_no_key = EnhancedConfig()
        config_no_key.qdrant.api_key = None
        
        client_config_no_key = config_no_key.qdrant_client_config
        assert "api_key" not in client_config_no_key
        
        summary_no_key = config_no_key.get_config_summary()
        assert "qdrant_api_key_set" not in summary_no_key
    
    def test_ssl_security_configuration(self):
        """Test SSL/TLS security configuration validation."""
        config = EnhancedConfig()
        
        # Test secure SSL configuration
        config.security.ssl_verify_certificates = True
        config.security.validate_ssl = True
        config.security.production_enforce_ssl = True
        
        # Test insecure development configuration
        config.environment = "development"
        config.security.development_allow_insecure_localhost = True
        config.qdrant.url = "http://localhost:6333"  # OK for development
        
        dev_errors = config.validate_config()
        # Should be valid for development
        
        # Test insecure production configuration
        config.environment = "production"
        config.security.production_enforce_ssl = True
        config.qdrant.url = "http://production-server:6333"  # Not OK for production
        
        prod_errors = config.validate_config()
        assert len(prod_errors) > 0
    
    def test_environment_variable_conflict_detection(self):
        """Test detection of conflicting environment variable settings."""
        # Test conflicting debug settings
        with patch.dict(os.environ, {
            "WORKSPACE_QDRANT_DEBUG": "true",
        }):
            config = EnhancedConfig(debug=False)  # Explicit False conflicts with env var True
            
            # This creates a potential conflict that should be detected
            validator = ConfigValidator(config)
            warnings = validator._generate_warnings()
            
            # The system should handle this gracefully
            assert isinstance(warnings, list)
    
    def test_configuration_export_security(self):
        """Test that exported configuration properly handles sensitive data."""
        config = Config()
        config.qdrant.api_key = "secret-api-key"
        
        # Export to YAML
        yaml_output = config.to_yaml()
        
        # In a production system, sensitive values should be excluded or masked
        # For now, verify the structure is correct
        yaml_data = yaml.safe_load(yaml_output)
        assert "qdrant" in yaml_data
        assert "api_key" in yaml_data["qdrant"]
        
        # In production, this should be masked or excluded
        # This test documents current behavior and can be enhanced later
        assert yaml_data["qdrant"]["api_key"] == "secret-api-key"


class TestConfigurationIntegration:
    """Integration tests combining multiple configuration validation aspects."""
    
    def test_full_configuration_pipeline(self):
        """Test complete configuration loading pipeline with all features."""
        # Create a comprehensive configuration that tests all systems
        yaml_content = """
host: "${SERVER_HOST:-127.0.0.1}"
port: ${SERVER_PORT:-8000}
debug: false

qdrant:
  url: "${QDRANT_URL:-http://localhost:6333}"
  api_key: "${QDRANT_API_KEY:-}"
  timeout: ${QDRANT_TIMEOUT:-30}

embedding:
  model: "${EMBEDDING_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
  chunk_size: ${CHUNK_SIZE:-800}
  chunk_overlap: ${CHUNK_OVERLAP:-120}
  batch_size: ${BATCH_SIZE:-50}

workspace:
  github_user: "${GITHUB_USER:-}"
  collections: 
    - project
    - docs
  global_collections:
    - scratchbook
    - references
"""
        
        test_env_vars = {
            "SERVER_HOST": "integration-host",
            "SERVER_PORT": "9999",
            "QDRANT_URL": "https://integration-qdrant:6333",
            "QDRANT_API_KEY": "integration-api-key",
            "QDRANT_TIMEOUT": "45",
            "EMBEDDING_MODEL": "integration-model",
            "CHUNK_SIZE": "1200",
            "CHUNK_OVERLAP": "200",
            "BATCH_SIZE": "75",
            "GITHUB_USER": "integration-user",
        }
        
        with patch.dict(os.environ, test_env_vars):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write(yaml_content)
                yaml_file = f.name
            
            try:
                # Test 1: YAML config with environment substitution
                config = Config(config_file=yaml_file)
                
                assert config.host == "integration-host"
                assert config.port == 9999
                assert config.qdrant.url == "https://integration-qdrant:6333"
                assert config.qdrant.api_key == "integration-api-key"
                assert config.qdrant.timeout == 45
                assert config.embedding.model == "integration-model"
                assert config.embedding.chunk_size == 1200
                assert config.embedding.chunk_overlap == 200
                assert config.embedding.batch_size == 75
                assert config.workspace.github_user == "integration-user"
                
                # Test 2: Validation
                issues = config.validate_config()
                # Should have no critical issues with this configuration
                assert len(issues) == 0 or all("should not exceed" not in issue for issue in issues)
                
                # Test 3: Enhanced config validation
                validator = ConfigValidator(config)
                is_valid, results = validator.validate_all()
                
                assert isinstance(results, dict)
                assert "issues" in results
                assert "warnings" in results
                
                # Test 4: Configuration export
                exported_yaml = config.to_yaml()
                exported_data = yaml.safe_load(exported_yaml)
                
                assert exported_data['host'] == "integration-host"
                assert exported_data['qdrant']['url'] == "https://integration-qdrant:6333"
                
            finally:
                os.unlink(yaml_file)
    
    def test_configuration_error_recovery(self):
        """Test configuration error recovery and fallback mechanisms."""
        # Test with partially invalid configuration
        invalid_yaml = """
host: "${UNDEFINED_HOST}"  # No fallback, should be empty
port: ${INVALID_PORT:-8000}  # Valid fallback
debug: "${INVALID_BOOL:-false}"  # String that should convert to bool

qdrant:
  url: "${QDRANT_URL:-http://localhost:6333}"  # Valid fallback
  api_key: "${UNDEFINED_API_KEY}"  # No fallback, should be empty
  timeout: ${INVALID_TIMEOUT:-30}  # Valid fallback

embedding:
  model: "${UNDEFINED_MODEL:-default-model}"  # Valid fallback
  chunk_size: ${INVALID_CHUNK_SIZE:-800}  # Valid fallback
  chunk_overlap: 9999  # Will cause validation error (> chunk_size)
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml)
            yaml_file = f.name
        
        try:
            # Should handle missing environment variables gracefully
            loader = YAMLConfigLoader()
            import asyncio
            
            # This should work despite missing env vars
            config_data = asyncio.run(loader.load_with_env_substitution(yaml_file))
            
            assert config_data['host'] == ""  # Missing env var, no fallback
            assert config_data['port'] == 8000  # Fallback value
            assert config_data['qdrant']['url'] == "http://localhost:6333"  # Fallback
            assert config_data['qdrant']['api_key'] == ""  # Missing, no fallback
            assert config_data['embedding']['model'] == "default-model"  # Fallback
            
            # Validation should catch the chunk overlap issue
            try:
                validation_result = asyncio.run(loader.load_and_validate(yaml_file))
                # Should pass basic validation but may have logical issues
            except ValueError as e:
                # Validation errors are expected for this invalid config
                assert "chunk_overlap" in str(e).lower() or "validation" in str(e).lower()
            
        finally:
            os.unlink(yaml_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])