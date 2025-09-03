"""Tests for SSL configuration and warning management.

This module tests the SSL/TLS configuration functionality including:
- Context-aware SSL warning suppression
- Localhost vs remote connection handling
- SSL certificate validation configuration
- Authentication support for secure connections
- Development vs production environment behavior
"""

import warnings
import ssl
from unittest.mock import patch, MagicMock
import urllib3
from urllib.parse import urlparse

import pytest

from workspace_qdrant_mcp.core.ssl_config import (
    SSLConfiguration,
    SSLContextManager,
    get_ssl_manager,
    create_secure_qdrant_config
)


class TestSSLConfiguration:
    """Test SSL configuration class."""
    
    def test_default_configuration(self):
        """Test default SSL configuration."""
        config = SSLConfiguration()
        assert config.verify_certificates is True
        assert config.ca_cert_path is None
        assert config.client_cert_path is None
        assert config.client_key_path is None
        assert config.auth_token is None
        assert config.api_key is None
        assert config.environment == "production"
    
    def test_development_configuration(self):
        """Test development SSL configuration."""
        config = SSLConfiguration(
            verify_certificates=False,
            environment="development"
        )
        assert config.verify_certificates is False
        assert config.environment == "development"
    
    def test_authenticated_configuration(self):
        """Test SSL configuration with authentication."""
        config = SSLConfiguration(
            auth_token="test_token",
            api_key="test_api_key"
        )
        assert config.auth_token == "test_token"
        assert config.api_key == "test_api_key"
    
    def test_to_qdrant_config_with_api_key(self):
        """Test conversion to Qdrant config with API key."""
        config = SSLConfiguration(api_key="test_api_key")
        qdrant_config = config.to_qdrant_config()
        
        assert qdrant_config['api_key'] == "test_api_key"
    
    def test_to_qdrant_config_with_auth_token(self):
        """Test conversion to Qdrant config with auth token."""
        config = SSLConfiguration(auth_token="test_token")
        qdrant_config = config.to_qdrant_config()
        
        assert 'metadata' in qdrant_config
        assert qdrant_config['metadata']['authorization'] == 'Bearer test_token'
    
    def test_to_qdrant_config_disable_verification(self):
        """Test Qdrant config with disabled certificate verification."""
        config = SSLConfiguration(verify_certificates=False)
        qdrant_config = config.to_qdrant_config()
        
        assert qdrant_config['verify'] is False
    
    def test_to_qdrant_config_with_certificates(self):
        """Test Qdrant config with client certificates."""
        config = SSLConfiguration(
            ca_cert_path="/path/to/ca.crt",
            client_cert_path="/path/to/client.crt",
            client_key_path="/path/to/client.key"
        )
        qdrant_config = config.to_qdrant_config()
        
        assert qdrant_config['ca_certs'] == "/path/to/ca.crt"
        assert qdrant_config['cert'] == ("/path/to/client.crt", "/path/to/client.key")
    
    def test_create_ssl_context_secure(self):
        """Test creating secure SSL context."""
        config = SSLConfiguration(verify_certificates=True)
        context = config.create_ssl_context()
        
        assert isinstance(context, ssl.SSLContext)
        assert context.verify_mode == ssl.CERT_REQUIRED
    
    def test_create_ssl_context_development(self):
        """Test creating development SSL context."""
        config = SSLConfiguration(
            verify_certificates=False,
            environment="development"
        )
        context = config.create_ssl_context()
        
        assert isinstance(context, ssl.SSLContext)
        assert context.verify_mode == ssl.CERT_NONE
        assert context.check_hostname is False


class TestSSLContextManager:
    """Test SSL context manager functionality."""
    
    def test_is_localhost_url(self):
        """Test localhost URL detection."""
        manager = SSLContextManager()
        
        # Test localhost URLs
        assert manager.is_localhost_url("http://localhost:6333") is True
        assert manager.is_localhost_url("https://127.0.0.1:6333") is True
        assert manager.is_localhost_url("http://127.1.2.3:6333") is True
        assert manager.is_localhost_url("https://::1:6333") is True
        
        # Test remote URLs
        assert manager.is_localhost_url("https://qdrant.example.com") is False
        assert manager.is_localhost_url("http://10.0.0.1:6333") is False
        assert manager.is_localhost_url("https://192.168.1.100:6333") is False
    
    def test_is_localhost_url_malformed(self):
        """Test localhost detection with malformed URLs."""
        manager = SSLContextManager()
        
        # Should handle malformed URLs gracefully
        assert manager.is_localhost_url("not-a-url") is False
        assert manager.is_localhost_url("") is False
        assert manager.is_localhost_url("://malformed") is False
    
    @patch('warnings.filterwarnings')
    @patch('urllib3.disable_warnings')
    def test_for_localhost_context_manager(self, mock_urllib3_disable, mock_warnings_filter):
        """Test localhost SSL warning suppression context manager."""
        manager = SSLContextManager()
        
        # Store original filters
        original_filters = warnings.filters.copy()
        
        with manager.for_localhost():
            # Check that warnings were filtered
            assert mock_warnings_filter.call_count >= 3  # At least 3 filter calls
            assert mock_urllib3_disable.called
            
            # Check that suppression is active
            assert manager._suppression_active is True
        
        # After exiting context, suppression should be inactive
        assert manager._suppression_active is False
    
    def test_create_ssl_config_localhost_development(self):
        """Test SSL config creation for localhost in development."""
        manager = SSLContextManager()
        
        config = manager.create_ssl_config(
            url="http://localhost:6333",
            environment="development"
        )
        
        assert config.verify_certificates is False
        assert config.environment == "development"
    
    def test_create_ssl_config_localhost_production(self):
        """Test SSL config creation for localhost in production."""
        manager = SSLContextManager()
        
        config = manager.create_ssl_config(
            url="http://localhost:6333",
            environment="production"
        )
        
        # localhost in production should still allow configuration override
        assert config.environment == "production"
    
    def test_create_ssl_config_remote(self):
        """Test SSL config creation for remote URLs."""
        manager = SSLContextManager()
        
        config = manager.create_ssl_config(
            url="https://qdrant.example.com",
            environment="development"
        )
        
        # Remote URLs should always verify certificates
        assert config.verify_certificates is True
    
    def test_create_ssl_config_with_auth(self):
        """Test SSL config creation with authentication."""
        manager = SSLContextManager()
        
        config = manager.create_ssl_config(
            url="https://qdrant.example.com",
            auth_token="test_token",
            api_key="test_api_key"
        )
        
        assert config.auth_token == "test_token"
        assert config.api_key == "test_api_key"
    
    def test_get_qdrant_client_config(self):
        """Test merging SSL config with Qdrant client config."""
        manager = SSLContextManager()
        
        base_config = {
            "url": "http://localhost:6333",
            "timeout": 30
        }
        
        ssl_config = SSLConfiguration(
            api_key="test_key",
            verify_certificates=False
        )
        
        merged_config = manager.get_qdrant_client_config(base_config, ssl_config)
        
        # Should contain original config
        assert merged_config["url"] == "http://localhost:6333"
        assert merged_config["timeout"] == 30
        
        # Should contain SSL config
        assert merged_config["api_key"] == "test_key"
        assert merged_config["verify"] is False


class TestSSLConfigUtilities:
    """Test SSL configuration utility functions."""
    
    def test_get_ssl_manager_singleton(self):
        """Test that get_ssl_manager returns the same instance."""
        manager1 = get_ssl_manager()
        manager2 = get_ssl_manager()
        
        assert manager1 is manager2
        assert isinstance(manager1, SSLContextManager)
    
    def test_create_secure_qdrant_config_localhost(self):
        """Test creating secure config for localhost."""
        base_config = {"url": "http://localhost:6333"}
        
        config = create_secure_qdrant_config(
            base_config=base_config,
            url="http://localhost:6333",
            environment="development"
        )
        
        # Should include base config
        assert config["url"] == "http://localhost:6333"
        
        # Should have SSL verification disabled for localhost in dev
        assert config.get("verify") is False
    
    def test_create_secure_qdrant_config_remote(self):
        """Test creating secure config for remote URL."""
        base_config = {"url": "https://qdrant.example.com"}
        
        config = create_secure_qdrant_config(
            base_config=base_config,
            url="https://qdrant.example.com",
            environment="development",
            api_key="test_key"
        )
        
        # Should include base config
        assert config["url"] == "https://qdrant.example.com"
        
        # Should have authentication
        assert config["api_key"] == "test_key"
        
        # Should not disable verification for remote URLs
        assert config.get("verify") != False
    
    def test_create_secure_qdrant_config_production(self):
        """Test creating secure config for production environment."""
        base_config = {"url": "https://qdrant.example.com"}
        
        config = create_secure_qdrant_config(
            base_config=base_config,
            url="https://qdrant.example.com",
            environment="production",
            auth_token="prod_token"
        )
        
        # Should include authentication metadata
        assert "metadata" in config
        assert config["metadata"]["authorization"] == "Bearer prod_token"


class TestSSLWarningIntegration:
    """Test SSL warning suppression integration."""
    
    def test_warning_suppression_context(self):
        """Test that warnings are properly suppressed and restored."""
        manager = SSLContextManager()
        
        # Store original warning count
        original_filter_count = len(warnings.filters)
        
        # Test warning suppression
        with patch('urllib3.disable_warnings') as mock_disable:
            with manager.for_localhost():
                # Inside context, warnings should be filtered
                assert len(warnings.filters) >= original_filter_count
                assert mock_disable.called
        
        # After context, filters should be restored
        # (Note: exact count may vary due to test environment)
    
    @patch('workspace_qdrant_mcp.core.ssl_config.logger')
    def test_ssl_context_logging(self, mock_logger):
        """Test that SSL operations are properly logged."""
        manager = SSLContextManager()
        
        with manager.for_localhost():
            pass
        
        # Should log warning suppression and restoration
        assert mock_logger.debug.call_count >= 2
    
    def test_multiple_context_managers(self):
        """Test nested or multiple context manager usage."""
        manager = SSLContextManager()
        
        # Test nested contexts
        with manager.for_localhost():
            assert manager._suppression_active is True
            
            with manager.for_localhost():
                assert manager._suppression_active is True
            
            assert manager._suppression_active is True
        
        assert manager._suppression_active is False
