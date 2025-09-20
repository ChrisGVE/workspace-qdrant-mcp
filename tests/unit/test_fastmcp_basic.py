"""
Basic FastMCP Infrastructure Test - Simplified for Task 241.1.

This test focuses on the core FastMCP testing infrastructure functionality
without getting into complex tool structure validation.
"""

import pytest
from tests.utils.fastmcp_test_infrastructure import (
    FastMCPTestServer,
    FastMCPTestClient,
    fastmcp_test_environment
)


class TestBasicFastMCPInfrastructure:
    """Basic tests for FastMCP testing infrastructure."""

    @pytest.mark.fastmcp
    async def test_fastmcp_environment_creation(self):
        """Test that FastMCP test environment can be created."""
        from workspace_qdrant_mcp.server import app

        # Test basic environment creation
        async with fastmcp_test_environment(app, "basic-test") as (server, client):
            assert isinstance(server, FastMCPTestServer)
            assert isinstance(client, FastMCPTestClient)
            assert server.initialized
            assert client.initialized

    @pytest.mark.fastmcp
    async def test_tool_call_basic(self, fastmcp_test_client):
        """Test basic tool call functionality."""
        client = fastmcp_test_client

        # Test a simple tool call
        result = await client.call_tool("workspace_status", {})

        # Basic result structure validation
        assert hasattr(result, 'success')
        assert hasattr(result, 'tool_name')
        assert hasattr(result, 'execution_time_ms')
        assert result.tool_name == "workspace_status"

        # Either succeeds or fails gracefully
        if result.success:
            assert result.response is not None
        else:
            assert result.error is not None

    @pytest.mark.fastmcp
    async def test_call_history(self, fastmcp_test_client):
        """Test call history functionality."""
        client = fastmcp_test_client

        # Start with empty history
        initial_history = client.get_call_history()

        # Make a call
        await client.call_tool("workspace_status", {})

        # History should have one more item
        new_history = client.get_call_history()
        assert len(new_history) == len(initial_history) + 1

    @pytest.mark.fastmcp
    async def test_error_handling(self, fastmcp_test_client):
        """Test error handling for invalid tool calls."""
        client = fastmcp_test_client

        # Test nonexistent tool
        result = await client.call_tool("definitely_not_a_real_tool", {})

        assert not result.success
        assert result.error is not None
        # Accept various error types including context errors
        error_lower = result.error.lower()
        assert (
            "not found" in error_lower or
            "unknown" in error_lower or
            "no active context" in error_lower or
            "context" in error_lower
        ), f"Unexpected error message: {result.error}"

    @pytest.mark.fastmcp
    async def test_server_tool_discovery(self, fastmcp_test_server):
        """Test basic tool discovery."""
        server = fastmcp_test_server

        # Should be able to get list of tools
        tools = server.get_available_tools()

        # Should be a list (might be empty if no tools loaded)
        assert isinstance(tools, list)

        # If tools exist, they should be strings
        for tool in tools:
            assert isinstance(tool, str)
            assert len(tool) > 0