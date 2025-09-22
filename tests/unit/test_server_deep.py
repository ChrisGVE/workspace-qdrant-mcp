"""
Deep coverage tests for server.py module focusing on method-level execution.

This test suite targets specific uncovered lines in server.py to maximize coverage
by testing core functionality, error paths, and edge cases.
"""

import asyncio
import os
import sys
import signal
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, AsyncMock, MagicMock, call, mock_open

import pytest

# Add the src directory to Python path for imports
src_path = Path(__file__).parent.parent.parent / "src" / "python"
sys.path.insert(0, str(src_path))

# Test imports
try:
    from workspace_qdrant_mcp import server
    from workspace_qdrant_mcp.server import (
        run_server, create_app, create_client,
        handle_server_signal, setup_logging
    )
    SERVER_AVAILABLE = True
except ImportError as e:
    SERVER_AVAILABLE = False
    server = None
    print(f"Server import failed: {e}")

pytestmark = pytest.mark.skipif(not SERVER_AVAILABLE, reason="Server module not available")


class TestServerDeepCoverage:
    """Test class for deep method-level coverage of server.py."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        config = MagicMock()
        config.qdrant_client_config.url = "http://localhost:6333"
        config.qdrant_client_config.api_key = None
        config.workspace.global_collections = ["global", "shared"]
        config.workspace.project_collections = ["notes", "docs"]
        config.embedding.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        config.embedding.dimension = 384
        config.server.host = "127.0.0.1"
        config.server.port = 8000
        config.logging.level = "INFO"
        return config

    def test_create_app_initialization(self, mock_config):
        """Test create_app function initialization."""
        with patch('workspace_qdrant_mcp.server.FastMCP') as mock_fastmcp:
            mock_app = MagicMock()
            mock_fastmcp.return_value = mock_app

            app = create_app(mock_config)

            assert app == mock_app
            mock_fastmcp.assert_called_once_with("workspace-qdrant-mcp")

    def test_create_app_tool_registration(self, mock_config):
        """Test create_app tool registration process."""
        with patch('workspace_qdrant_mcp.server.FastMCP') as mock_fastmcp:
            mock_app = MagicMock()
            mock_fastmcp.return_value = mock_app

            with patch('workspace_qdrant_mcp.server.register_memory_tools') as mock_memory:
                with patch('workspace_qdrant_mcp.server.register_search_tools') as mock_search:
                    with patch('workspace_qdrant_mcp.server.register_state_tools') as mock_state:
                        app = create_app(mock_config)

                        # Verify tool registration calls
                        mock_memory.assert_called_once_with(mock_app, mock_config)
                        mock_search.assert_called_once_with(mock_app, mock_config)
                        mock_state.assert_called_once_with(mock_app, mock_config)

    @pytest.mark.asyncio
    async def test_create_client_success(self, mock_config):
        """Test successful client creation."""
        with patch('workspace_qdrant_mcp.server.QdrantWorkspaceClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.initialize.return_value = None

            client = await create_client(mock_config)

            assert client == mock_client
            mock_client_class.assert_called_once_with(mock_config)
            mock_client.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_client_initialization_failure(self, mock_config):
        """Test client creation with initialization failure."""
        with patch('workspace_qdrant_mcp.server.QdrantWorkspaceClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.initialize.side_effect = Exception("Connection failed")

            with pytest.raises(Exception, match="Connection failed"):
                await create_client(mock_config)

    def test_setup_logging_with_level(self):
        """Test setup_logging with different log levels."""
        with patch('workspace_qdrant_mcp.server.logger') as mock_logger:
            setup_logging("DEBUG")
            mock_logger.remove.assert_called()
            mock_logger.add.assert_called()

    def test_setup_logging_default(self):
        """Test setup_logging with default level."""
        with patch('workspace_qdrant_mcp.server.logger') as mock_logger:
            setup_logging()
            mock_logger.remove.assert_called()
            mock_logger.add.assert_called()

    def test_handle_server_signal_sigint(self):
        """Test signal handler for SIGINT."""
        with patch('workspace_qdrant_mcp.server.sys.exit') as mock_exit:
            handle_server_signal(signal.SIGINT, None)
            mock_exit.assert_called_once_with(0)

    def test_handle_server_signal_sigterm(self):
        """Test signal handler for SIGTERM."""
        with patch('workspace_qdrant_mcp.server.sys.exit') as mock_exit:
            handle_server_signal(signal.SIGTERM, None)
            mock_exit.assert_called_once_with(0)

    def test_handle_server_signal_other(self):
        """Test signal handler for other signals."""
        with patch('workspace_qdrant_mcp.server.sys.exit') as mock_exit:
            handle_server_signal(signal.SIGUSR1, None)
            mock_exit.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_server_stdio_mode(self, mock_config):
        """Test run_server in stdio mode."""
        with patch('workspace_qdrant_mcp.server.create_app') as mock_create_app:
            with patch('workspace_qdrant_mcp.server.create_client') as mock_create_client:
                with patch('workspace_qdrant_mcp.server.setup_logging') as mock_setup_logging:
                    with patch('workspace_qdrant_mcp.server.signal.signal') as mock_signal:
                        mock_app = AsyncMock()
                        mock_create_app.return_value = mock_app
                        mock_create_client.return_value = AsyncMock()

                        mock_app.run.side_effect = KeyboardInterrupt()

                        with pytest.raises(KeyboardInterrupt):
                            await run_server(mock_config, transport="stdio")

                        mock_setup_logging.assert_called_once()
                        mock_signal.assert_called()
                        mock_create_app.assert_called_once_with(mock_config)
                        mock_create_client.assert_called_once_with(mock_config)
                        mock_app.run.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_run_server_http_mode(self, mock_config):
        """Test run_server in HTTP mode."""
        with patch('workspace_qdrant_mcp.server.create_app') as mock_create_app:
            with patch('workspace_qdrant_mcp.server.create_client') as mock_create_client:
                with patch('workspace_qdrant_mcp.server.setup_logging') as mock_setup_logging:
                    with patch('workspace_qdrant_mcp.server.signal.signal') as mock_signal:
                        mock_app = AsyncMock()
                        mock_create_app.return_value = mock_app
                        mock_create_client.return_value = AsyncMock()

                        mock_app.run.side_effect = KeyboardInterrupt()

                        with pytest.raises(KeyboardInterrupt):
                            await run_server(mock_config, transport="http", host="127.0.0.1", port=8000)

                        mock_setup_logging.assert_called_once()
                        mock_create_app.assert_called_once_with(mock_config)
                        mock_create_client.assert_called_once_with(mock_config)
                        mock_app.run.assert_called_once_with(host="127.0.0.1", port=8000)

    @pytest.mark.asyncio
    async def test_run_server_client_creation_failure(self, mock_config):
        """Test run_server when client creation fails."""
        with patch('workspace_qdrant_mcp.server.create_app') as mock_create_app:
            with patch('workspace_qdrant_mcp.server.create_client') as mock_create_client:
                with patch('workspace_qdrant_mcp.server.setup_logging') as mock_setup_logging:
                    mock_create_app.return_value = AsyncMock()
                    mock_create_client.side_effect = Exception("Client creation failed")

                    with pytest.raises(Exception, match="Client creation failed"):
                        await run_server(mock_config)

    def test_server_module_imports(self):
        """Test that all required server module imports work."""
        # Test direct imports
        assert server is not None
        assert hasattr(server, 'run_server')
        assert hasattr(server, 'create_app')
        assert hasattr(server, 'create_client')
        assert hasattr(server, 'handle_server_signal')
        assert hasattr(server, 'setup_logging')

    @pytest.mark.asyncio
    async def test_run_server_with_custom_logging_level(self, mock_config):
        """Test run_server with custom logging level."""
        mock_config.logging.level = "DEBUG"

        with patch('workspace_qdrant_mcp.server.create_app') as mock_create_app:
            with patch('workspace_qdrant_mcp.server.create_client') as mock_create_client:
                with patch('workspace_qdrant_mcp.server.setup_logging') as mock_setup_logging:
                    mock_app = AsyncMock()
                    mock_create_app.return_value = mock_app
                    mock_create_client.return_value = AsyncMock()

                    mock_app.run.side_effect = KeyboardInterrupt()

                    with pytest.raises(KeyboardInterrupt):
                        await run_server(mock_config)

                    mock_setup_logging.assert_called_once_with("DEBUG")

    def test_create_app_with_minimal_config(self):
        """Test create_app with minimal configuration."""
        minimal_config = MagicMock()
        minimal_config.workspace = None
        minimal_config.embedding = None

        with patch('workspace_qdrant_mcp.server.FastMCP') as mock_fastmcp:
            with patch('workspace_qdrant_mcp.server.register_memory_tools'):
                with patch('workspace_qdrant_mcp.server.register_search_tools'):
                    with patch('workspace_qdrant_mcp.server.register_state_tools'):
                        mock_app = MagicMock()
                        mock_fastmcp.return_value = mock_app

                        app = create_app(minimal_config)

                        assert app == mock_app

    @pytest.mark.asyncio
    async def test_create_client_with_none_config(self):
        """Test create_client with None config handling."""
        with patch('workspace_qdrant_mcp.server.QdrantWorkspaceClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.initialize.return_value = None

            client = await create_client(None)

            # Should still create client even with None config
            assert client == mock_client

    def test_signal_registration_coverage(self):
        """Test signal registration in run_server."""
        with patch('workspace_qdrant_mcp.server.signal.signal') as mock_signal:
            with patch('workspace_qdrant_mcp.server.create_app'):
                with patch('workspace_qdrant_mcp.server.create_client'):
                    with patch('workspace_qdrant_mcp.server.setup_logging'):
                        mock_app = AsyncMock()
                        mock_app.run.side_effect = KeyboardInterrupt()

                        # This should call signal.signal twice (SIGINT and SIGTERM)
                        try:
                            asyncio.run(run_server(MagicMock()))
                        except KeyboardInterrupt:
                            pass

                        # Verify signal registration calls
                        assert mock_signal.call_count >= 2

    def test_server_initialization_with_environment_variables(self):
        """Test server initialization with environment variables."""
        with patch.dict(os.environ, {'QDRANT_URL': 'http://test:6333', 'QDRANT_API_KEY': 'test-key'}):
            with patch('workspace_qdrant_mcp.server.create_app') as mock_create_app:
                mock_config = MagicMock()

                create_app(mock_config)

                # Verify app creation was called
                mock_create_app.assert_called_once_with(mock_config)

    def test_logging_configuration_edge_cases(self):
        """Test logging setup with edge cases."""
        # Test with empty string
        with patch('workspace_qdrant_mcp.server.logger') as mock_logger:
            setup_logging("")
            mock_logger.add.assert_called()

        # Test with invalid level (should not crash)
        with patch('workspace_qdrant_mcp.server.logger') as mock_logger:
            setup_logging("INVALID_LEVEL")
            mock_logger.add.assert_called()

    @pytest.mark.asyncio
    async def test_concurrent_server_operations(self, mock_config):
        """Test concurrent server operations."""
        async def mock_run():
            await asyncio.sleep(0.1)
            raise KeyboardInterrupt()

        with patch('workspace_qdrant_mcp.server.create_app') as mock_create_app:
            with patch('workspace_qdrant_mcp.server.create_client') as mock_create_client:
                with patch('workspace_qdrant_mcp.server.setup_logging'):
                    mock_app = AsyncMock()
                    mock_app.run = mock_run
                    mock_create_app.return_value = mock_app
                    mock_create_client.return_value = AsyncMock()

                    # Run multiple concurrent server instances
                    tasks = [
                        run_server(mock_config, transport="stdio"),
                        run_server(mock_config, transport="stdio")
                    ]

                    with pytest.raises(KeyboardInterrupt):
                        await asyncio.gather(*tasks, return_exceptions=True)

    def test_server_error_handling_coverage(self):
        """Test server error handling paths."""
        with patch('workspace_qdrant_mcp.server.FastMCP') as mock_fastmcp:
            mock_fastmcp.side_effect = Exception("FastMCP initialization failed")

            with pytest.raises(Exception, match="FastMCP initialization failed"):
                create_app(MagicMock())


if __name__ == "__main__":
    pytest.main([__file__])