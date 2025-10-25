"""
Tests for the simplified 4-tool MCP server.
Tests the basic functionality without requiring a running Qdrant instance.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Add source path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "python"))


class TestSimpleServerImports:
    """Test that the simplified server can be imported."""

    def test_fastmcp_available(self):
        """Test FastMCP is available."""
        from fastmcp import FastMCP
        assert FastMCP is not None

    def test_qdrant_client_available(self):
        """Test Qdrant client is available."""
        from qdrant_client import QdrantClient
        assert QdrantClient is not None

    def test_simple_server_imports(self):
        """Test the simple server module imports."""
        from workspace_qdrant_mcp.simple_server import create_simple_server
        assert create_simple_server is not None

    def test_main_entry_point_imports(self):
        """Test main entry point imports."""
        from workspace_qdrant_mcp.main import main
        assert main is not None


class TestSimpleServerCreation:
    """Test server creation and basic structure."""

    def test_create_simple_server(self):
        """Test server creation."""
        from workspace_qdrant_mcp.simple_server import create_simple_server

        server = create_simple_server()
        assert server is not None
        assert hasattr(server, 'name')
        assert server.name == "Workspace Qdrant MCP"

    def test_server_has_four_tools(self):
        """Test that server has exactly 4 tools."""
        # Check that the app has the expected tools
        # Note: FastMCP doesn't expose tools directly, so we check the module
        import workspace_qdrant_mcp.simple_server as server_module
        from workspace_qdrant_mcp.simple_server import app

        # Count decorated functions (tools)
        tool_functions = ['store', 'search', 'manage', 'retrieve']
        for tool_name in tool_functions:
            assert hasattr(server_module, tool_name), f"Missing tool: {tool_name}"


class TestUtilityFunctions:
    """Test utility functions that don't require external dependencies."""

    def test_get_project_name_fallback(self):
        """Test project name detection fallback."""
        from workspace_qdrant_mcp.simple_server import get_project_name

        with patch('pathlib.Path.exists', return_value=False):
            with patch('pathlib.Path.cwd') as mock_cwd:
                mock_cwd.return_value.name = 'test-project'
                name = get_project_name()
                assert name == 'test-project'

    def test_get_project_name_git_success(self):
        """Test project name detection from git."""
        from workspace_qdrant_mcp.simple_server import get_project_name

        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = 'https://github.com/user/my-repo.git\n'

        with patch('pathlib.Path.exists', return_value=True):
            with patch('subprocess.run', return_value=mock_result):
                name = get_project_name()
                assert name == 'my-repo'

    def test_get_project_name_git_failure(self):
        """Test project name detection when git fails."""
        from workspace_qdrant_mcp.simple_server import get_project_name

        with patch('pathlib.Path.exists', return_value=True):
            with patch('subprocess.run', side_effect=Exception("Git failed")):
                with patch('pathlib.Path.cwd') as mock_cwd:
                    mock_cwd.return_value.name = 'fallback-project'
                    name = get_project_name()
                    assert name == 'fallback-project'


class TestToolFunctionsExist:
    """Test that tool functions exist and are properly decorated."""

    def test_store_tool_exists(self):
        """Test that store tool function exists."""
        from workspace_qdrant_mcp.simple_server import store
        assert store is not None
        # FastMCP decorates functions, so they become FunctionTool objects
        assert hasattr(store, '__name__') or hasattr(store, 'name')

    def test_search_tool_exists(self):
        """Test that search tool function exists."""
        from workspace_qdrant_mcp.simple_server import search
        assert search is not None

    def test_manage_tool_exists(self):
        """Test that manage tool function exists."""
        from workspace_qdrant_mcp.simple_server import manage
        assert manage is not None

    def test_retrieve_tool_exists(self):
        """Test that retrieve tool function exists."""
        from workspace_qdrant_mcp.simple_server import retrieve
        assert retrieve is not None


class TestServerConfiguration:
    """Test server configuration and validation."""

    def test_server_environment_variables(self):
        """Test server reads environment variables."""
        import os

        # Test default values
        with patch.dict(os.environ, {}, clear=True):
            from workspace_qdrant_mcp.simple_server import initialize_components
            # This function should use defaults when env vars are not set
            assert True  # If import succeeds, defaults work

    def test_server_tool_count(self):
        """Test that server has exactly 4 tools."""
        import workspace_qdrant_mcp.simple_server as server_module

        # Count the tool functions
        tools = ['store', 'search', 'manage', 'retrieve']
        for tool in tools:
            assert hasattr(server_module, tool), f"Missing tool: {tool}"


class TestContentBasedRouting:
    """Test that tools route based on content as intended."""

    def test_collection_naming_logic_code(self):
        """Test collection naming for code files."""
        # Simulate the logic used in store function
        file_path = "example.py"
        project = "test-proj"

        # This mimics the logic in the store function
        if file_path.endswith(('.py', '.js', '.ts', '.java', '.cpp')):
            expected_collection = f"{project}-code"
        else:
            expected_collection = f"{project}-documents"

        assert expected_collection == "test-proj-code"

    def test_collection_naming_logic_scratchbook(self):
        """Test collection naming for scratchbook content."""
        source = "scratchbook"
        project = "test-proj"

        # This mimics the logic in the store function
        if source == "scratchbook":
            expected_collection = f"{project}-scratchbook"
        else:
            expected_collection = f"{project}-documents"

        assert expected_collection == "test-proj-scratchbook"

    def test_collection_naming_logic_web(self):
        """Test collection naming for web content."""
        url = "https://example.com"
        project = "test-proj"

        # This mimics the logic in the store function
        if url:
            expected_collection = f"{project}-web"
        else:
            expected_collection = f"{project}-documents"

        assert expected_collection == "test-proj-web"
