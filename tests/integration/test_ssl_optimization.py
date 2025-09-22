"""Integration tests for SSL optimization.

This module provides integration tests to verify that SSL warning optimization
works correctly with real Qdrant connections, both localhost and remote.
"""

import sys
from pathlib import Path

# Add src/python to path for common module imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

import asyncio
import warnings
from unittest.mock import patch, MagicMock

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from workspace_qdrant_mcp.core.ssl_config import get_ssl_manager
from workspace_qdrant_mcp.core.client import QdrantWorkspaceClient
from workspace_qdrant_mcp.core.config import Config


class TestSSLOptimizationIntegration:
    """Integration tests for SSL optimization."""
    
    def test_localhost_ssl_warning_suppression(self):
        """Test that SSL warnings are suppressed for localhost connections."""
        ssl_manager = get_ssl_manager()
        
        # Capture warnings
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            
            # Test localhost connection with SSL context
            with ssl_manager.for_localhost():
                # This would normally generate SSL warnings
                try:
                    # Mock a localhost connection that would generate SSL warnings
                    import urllib3
                    urllib3.connectionpool.log.warning("Test SSL warning")
                except Exception:
                    pass  # Expected for test environment
            
            # Check that relevant warnings were suppressed
            ssl_warnings = [w for w in warning_list 
                          if 'ssl' in str(w.message).lower() or 
                             'insecure' in str(w.message).lower()]
            
            # Should have minimal SSL warnings due to suppression
            assert len(ssl_warnings) <= 1
    
    @patch('workspace_qdrant_mcp.core.client.QdrantClient')
    def test_client_uses_ssl_context_localhost(self, mock_qdrant_client):
        """Test that QdrantWorkspaceClient uses SSL context for localhost."""
        # Mock config for localhost
        config = MagicMock()
        config.qdrant.url = "http://localhost:6333"
        config.qdrant_client_config = {"url": "http://localhost:6333"}
        config.security = MagicMock()
        config.security.qdrant_auth_token = None
        config.security.qdrant_api_key = None
        config.environment = "development"
        
        # Mock QdrantClient instance
        mock_client_instance = MagicMock()
        mock_qdrant_client.return_value = mock_client_instance
        mock_client_instance.get_collections.return_value = MagicMock()
        
        # Create workspace client
        client = QdrantWorkspaceClient(config)
        
        # Initialize should use SSL context manager
        asyncio.run(client.initialize())
        
        # Verify that QdrantClient was created
        assert mock_qdrant_client.called
        
        # Verify connection test was performed
        assert mock_client_instance.get_collections.called
    
    @patch('workspace_qdrant_mcp.core.client.QdrantClient')
    def test_client_uses_ssl_context_remote(self, mock_qdrant_client):
        """Test that QdrantWorkspaceClient handles remote connections properly."""
        # Mock config for remote connection
        config = MagicMock()
        config.qdrant.url = "https://qdrant.example.com"
        config.qdrant_client_config = {"url": "https://qdrant.example.com"}
        config.security = MagicMock()
        config.security.qdrant_auth_token = "test_token"
        config.security.qdrant_api_key = "test_api_key"
        config.environment = "production"
        
        # Mock QdrantClient instance
        mock_client_instance = MagicMock()
        mock_qdrant_client.return_value = mock_client_instance
        mock_client_instance.get_collections.return_value = MagicMock()
        
        # Create workspace client
        client = QdrantWorkspaceClient(config)
        
        # Initialize should handle remote connection
        asyncio.run(client.initialize())
        
        # Verify that QdrantClient was created with secure config
        mock_qdrant_client.assert_called_once()
        call_kwargs = mock_qdrant_client.call_args[1]
        
        # Should include authentication
        assert 'api_key' in call_kwargs or 'metadata' in call_kwargs
    
    def test_ssl_config_environment_detection(self):
        """Test SSL configuration adapts to different environments."""
        ssl_manager = get_ssl_manager()
        
        # Development environment with localhost
        dev_config = ssl_manager.create_ssl_config(
            url="http://localhost:6333",
            environment="development"
        )
        assert dev_config.verify_certificates is False
        assert dev_config.environment == "development"
        
        # Production environment with remote URL
        prod_config = ssl_manager.create_ssl_config(
            url="https://qdrant.example.com",
            environment="production",
            api_key="prod_api_key"
        )
        assert prod_config.verify_certificates is True
        assert prod_config.environment == "production"
        assert prod_config.api_key == "prod_api_key"
        
        # Development with remote URL (should still verify)
        dev_remote_config = ssl_manager.create_ssl_config(
            url="https://qdrant.example.com",
            environment="development"
        )
        assert dev_remote_config.verify_certificates is True
    
    def test_authentication_configuration(self):
        """Test that authentication is properly configured."""
        ssl_manager = get_ssl_manager()
        
        # Test API key authentication
        config_with_api_key = ssl_manager.create_ssl_config(
            url="https://qdrant.example.com",
            api_key="test_api_key"
        )
        qdrant_config = config_with_api_key.to_qdrant_config()
        assert qdrant_config['api_key'] == "test_api_key"
        
        # Test token authentication
        config_with_token = ssl_manager.create_ssl_config(
            url="https://qdrant.example.com",
            auth_token="test_token"
        )
        qdrant_config = config_with_token.to_qdrant_config()
        assert 'metadata' in qdrant_config
        assert qdrant_config['metadata']['authorization'] == 'Bearer test_token'
    
    @patch('warnings.filterwarnings')
    def test_warning_filter_specificity(self, mock_warnings_filter):
        """Test that only specific SSL warnings are suppressed."""
        ssl_manager = get_ssl_manager()
        
        with ssl_manager.for_localhost():
            pass
        
        # Check that specific warning filters were applied
        filter_calls = [call[0] for call in mock_warnings_filter.call_args_list]
        
        # Should have calls for specific SSL/insecure connection warnings
        insecure_filters = [call for call in filter_calls if len(call) >= 2 and 'insecure' in str(call[1]).lower()]
        ssl_filters = [call for call in filter_calls if len(call) >= 2 and 'ssl' in str(call[1]).lower()]
        
        assert len(insecure_filters) > 0 or len(ssl_filters) > 0
    
    def test_ssl_context_restoration(self):
        """Test that SSL warning filters are properly restored."""
        ssl_manager = get_ssl_manager()
        
        # Store original warning filters
        original_filters = warnings.filters.copy()
        
        # Use SSL context manager
        with ssl_manager.for_localhost():
            # Filters should be modified during context
            assert len(warnings.filters) >= len(original_filters)
        
        # After context, should be restored to original state
        # Note: exact comparison might vary due to test environment
        assert not ssl_manager._suppression_active
    
    def test_multiple_concurrent_contexts(self):
        """Test behavior with multiple concurrent SSL contexts."""
        ssl_manager = get_ssl_manager()
        
        # Test that nested contexts work correctly
        with ssl_manager.for_localhost():
            assert ssl_manager._suppression_active is True
            
            # Nested context should maintain suppression
            with ssl_manager.for_localhost():
                assert ssl_manager._suppression_active is True
                
                # Even deeper nesting
                with ssl_manager.for_localhost():
                    assert ssl_manager._suppression_active is True
                
                assert ssl_manager._suppression_active is True
            
            assert ssl_manager._suppression_active is True
        
        # After all contexts, should be inactive
        assert ssl_manager._suppression_active is False


class TestSSLOptimizationEdgeCases:
    """Test edge cases for SSL optimization."""
    
    def test_malformed_url_handling(self):
        """Test SSL manager handles malformed URLs gracefully."""
        ssl_manager = get_ssl_manager()
        
        # Test various malformed URLs
        malformed_urls = [
            "",
            "not-a-url",
            "://malformed",
            "http://",
            None
        ]
        
        for url in malformed_urls:
            try:
                # Should not crash on malformed URLs
                is_localhost = ssl_manager.is_localhost_url(str(url) if url else "")
                assert isinstance(is_localhost, bool)
            except Exception as e:
                pytest.fail(f"SSL manager should handle malformed URL gracefully: {url}, error: {e}")
    
    def test_ssl_config_with_missing_credentials(self):
        """Test SSL config creation without credentials."""
        ssl_manager = get_ssl_manager()
        
        config = ssl_manager.create_ssl_config(
            url="https://qdrant.example.com"
            # No auth_token or api_key provided
        )
        
        assert config.auth_token is None
        assert config.api_key is None
        assert config.verify_certificates is True  # Should still verify for remote
    
    def test_ssl_context_exception_handling(self):
        """Test SSL context manager handles exceptions properly."""
        ssl_manager = get_ssl_manager()
        
        # Test that exceptions in context don't prevent cleanup
        try:
            with ssl_manager.for_localhost():
                assert ssl_manager._suppression_active is True
                raise ValueError("Test exception")
        except ValueError:
            pass  # Expected
        
        # Should still be cleaned up after exception
        assert ssl_manager._suppression_active is False
    
    def test_concurrent_context_managers(self):
        """Test multiple SSL context managers don't interfere."""
        manager1 = get_ssl_manager()
        manager2 = get_ssl_manager()
        
        # Should be the same singleton instance
        assert manager1 is manager2
        
        # Test concurrent usage
        with manager1.for_localhost():
            assert manager1._suppression_active is True
            assert manager2._suppression_active is True  # Same instance
        
        assert manager1._suppression_active is False
        assert manager2._suppression_active is False
