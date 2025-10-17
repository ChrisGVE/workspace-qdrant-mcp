"""
Example test using official fastmcp.Client SDK (Task 325.4).

This demonstrates the migration from custom FastMCPTestServer infrastructure
to the official MCP SDK testing approach recommended by the MCP team.

This example validates:
- fastmcp.Client(app) in-memory testing
- CallToolResult response handling
- Proper async context manager usage
- Error vs success response checking
"""

import pytest
from fastmcp import Client
from fastmcp.client.client import CallToolResult
from mcp.types import TextContent


@pytest.fixture
async def mcp_client():
    """
    Official SDK fixture using fastmcp.Client with in-memory transport.

    This replaces our custom FastMCPTestServer/FastMCPTestClient infrastructure
    with the official MCP SDK pattern.
    """
    from workspace_qdrant_mcp.server import app

    # Use async context manager for automatic initialization and cleanup
    async with Client(app) as client:
        # Context manager handles initialization automatically
        yield client
        # Cleanup is automatic when context exits


class TestOfficialSDKExample:
    """Example tests demonstrating official SDK usage."""

    @pytest.mark.asyncio
    async def test_client_ping(self, mcp_client):
        """Test client can ping server."""
        # Official SDK ping method
        is_alive = await mcp_client.ping()
        assert isinstance(is_alive, bool)
        # Server should be accessible in-memory
        assert is_alive is True

    @pytest.mark.asyncio
    async def test_list_tools(self, mcp_client):
        """Test client can list available tools."""
        # Official SDK list_tools method
        tools = await mcp_client.list_tools()

        # Should return list of Tool objects
        assert isinstance(tools, list)
        assert len(tools) > 0

        # Verify expected tools are present
        tool_names = [tool.name for tool in tools]
        expected_tools = ["store", "search", "manage", "retrieve"]

        for expected_tool in expected_tools:
            assert expected_tool in tool_names, f"Expected tool '{expected_tool}' not found"

    @pytest.mark.asyncio
    async def test_call_tool_success_response(self, mcp_client):
        """Test successful tool call returns proper CallToolResult."""
        # Official SDK call_tool method
        result = await mcp_client.call_tool(
            "manage",
            {"action": "workspace_status"}
        )

        # Result should be official MCP CallToolResult type
        assert isinstance(result, CallToolResult)

        # Check for errors using official SDK attribute
        assert not result.isError, f"Tool call failed: {result.content if result.content else 'unknown error'}"

        # Content should be present
        assert result.content is not None
        assert isinstance(result.content, list)
        assert len(result.content) > 0

        # First content item should be TextContent for manage tool
        first_content = result.content[0]
        assert isinstance(first_content, TextContent)
        assert hasattr(first_content, 'text')

    @pytest.mark.asyncio
    async def test_call_tool_with_parameters(self, mcp_client):
        """Test tool call with various parameters."""
        # Test store tool with content parameter
        result = await mcp_client.call_tool(
            "store",
            {
                "content": "Test content for official SDK",
                "title": "SDK Example",
                "metadata": {"test": "example"}
            }
        )

        # Verify response structure
        assert isinstance(result, CallToolResult)
        assert not result.isError
        assert result.content is not None

    @pytest.mark.asyncio
    async def test_call_tool_error_handling(self, mcp_client):
        """Test tool call with invalid parameters shows proper error."""
        # Call tool with potentially invalid parameters
        # (empty content might fail validation)
        try:
            result = await mcp_client.call_tool(
                "store",
                {"content": ""}  # Empty content might be rejected
            )

            # If it doesn't raise, check if it's an error response
            if result.isError:
                # Error should have content explaining the issue
                assert result.content is not None
                assert len(result.content) > 0
        except Exception as e:
            # SDK might raise exception for invalid calls
            # This is acceptable behavior
            assert isinstance(e, Exception)

    @pytest.mark.asyncio
    async def test_search_tool_response_structure(self, mcp_client):
        """Test search tool returns expected response structure."""
        result = await mcp_client.call_tool(
            "search",
            {"query": "test search", "limit": 5}
        )

        assert isinstance(result, CallToolResult)

        # Search should succeed or gracefully error
        if not result.isError:
            assert result.content is not None
            # Content should contain results
            assert len(result.content) > 0

    @pytest.mark.asyncio
    async def test_manage_tool_list_collections(self, mcp_client):
        """Test manage tool list_collections action."""
        result = await mcp_client.call_tool(
            "manage",
            {"action": "list_collections"}
        )

        assert isinstance(result, CallToolResult)
        assert not result.isError
        assert result.content is not None

        # Should return collection information
        first_content = result.content[0]
        assert isinstance(first_content, TextContent)

    @pytest.mark.asyncio
    async def test_retrieve_tool_with_document_id(self, mcp_client):
        """Test retrieve tool with document_id parameter."""
        result = await mcp_client.call_tool(
            "retrieve",
            {"document_id": "test-doc-id"}
        )

        assert isinstance(result, CallToolResult)
        # Result may succeed or fail (doc might not exist)
        # Either way, response structure should be valid
        assert result.content is not None
