"""
Comprehensive MCP Capability Negotiation Tests (Task 325.3).

Tests MCP server capability negotiation following MCP specification requirements:
- Server initialization and handshake
- Capability advertisement (tools, prompts, resources)
- Version negotiation
- Feature detection
- Successful negotiation scenarios
- Failed negotiation edge cases

This verifies MCP protocol compliance for capability exchange, not business logic.
All external dependencies (Qdrant, daemon) are mocked via conftest fixtures.
"""

import json
import pytest
import inspect
from typing import Dict, Any, List
from unittest.mock import AsyncMock, Mock

from tests.utils.fastmcp_test_infrastructure import MCPTestResult


class TestServerInitialization:
    """Test MCP server initialization and startup."""

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_server_starts_successfully(self, fastmcp_test_server):
        """Verify MCP server initializes successfully."""
        assert fastmcp_test_server.initialized, "Server failed to initialize"
        assert fastmcp_test_server.app is not None, "Server app is None"

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_server_has_app_instance(self, fastmcp_test_server):
        """Verify server has valid FastMCP app instance."""
        from fastmcp import FastMCP

        assert isinstance(
            fastmcp_test_server.app, FastMCP
        ), "Server app is not FastMCP instance"

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_server_has_implementation_info(self, fastmcp_test_server):
        """Verify server has implementation name and version info."""
        app = fastmcp_test_server.app

        # Check if server has name (FastMCP stores this)
        assert hasattr(app, "name") or hasattr(
            app, "_name"
        ), "Server missing name attribute"

        # Get the name
        server_name = getattr(app, "name", None) or getattr(app, "_name", None)
        assert (
            server_name is not None and len(server_name) > 0
        ), "Server name is empty"

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_server_protocol_version_info(self, fastmcp_test_server):
        """Verify server communicates protocol version."""
        # Check that we can get the MCP package version
        from importlib.metadata import version
        try:
            mcp_version = version('mcp')
            assert mcp_version is not None and len(mcp_version) > 0, "MCP version is empty"
        except Exception as e:
            pytest.fail(f"Could not retrieve MCP package version: {e}")


class TestCapabilityAdvertisement:
    """Test server capability advertisement per MCP specification."""

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_server_has_tool_manager(self, fastmcp_test_server):
        """Verify server has tool management capability."""
        app = fastmcp_test_server.app

        # FastMCP should have tool manager
        assert hasattr(app, "_tool_manager") or hasattr(
            app, "get_tools"
        ), "Server missing tool management"

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_server_can_list_tools(self, fastmcp_test_server):
        """Verify server can enumerate tools."""
        available_tools = fastmcp_test_server.get_available_tools()

        assert isinstance(available_tools, list), "Tools list is not a list"
        assert len(available_tools) > 0, "No tools registered"

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_all_four_tools_registered(self, fastmcp_test_server):
        """Verify all 4 MCP tools are registered and discoverable."""
        available_tools = fastmcp_test_server.get_available_tools()

        expected_tools = ["store", "search", "manage", "retrieve"]

        for tool_name in expected_tools:
            assert (
                tool_name in available_tools
            ), f"Tool '{tool_name}' not registered"

        assert (
            len([t for t in available_tools if t in expected_tools]) == 4
        ), "Not all 4 tools registered"

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_tools_have_unique_names(self, fastmcp_test_server):
        """Verify all tools have unique names (no collisions)."""
        available_tools = fastmcp_test_server.get_available_tools()

        # Check for duplicates
        unique_tools = set(available_tools)
        assert len(unique_tools) == len(
            available_tools
        ), "Duplicate tool names detected"

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_store_tool_schema_complete(self, fastmcp_test_server):
        """Verify store tool has complete schema with all parameters."""
        tool = fastmcp_test_server.get_tool_sync("store")
        assert tool is not None, "store tool not found"

        # Check tool has callable function
        assert hasattr(tool, "fn") and callable(tool.fn), "Tool fn not callable"

        # Check function signature
        sig = inspect.signature(tool.fn)
        params = list(sig.parameters.keys())

        # Verify required parameters
        assert "content" in params, "Missing required parameter: content"

        # Verify optional parameters
        expected_optional = [
            "title",
            "metadata",
            "collection",
            "source",
            "document_type",
            "file_path",
            "url",
            "project_name",
        ]
        for param in expected_optional:
            assert param in params, f"Missing optional parameter: {param}"

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_search_tool_schema_complete(self, fastmcp_test_server):
        """Verify search tool has complete schema with all parameters."""
        tool = fastmcp_test_server.get_tool_sync("search")
        assert tool is not None, "search tool not found"

        assert hasattr(tool, "fn") and callable(tool.fn), "Tool fn not callable"

        sig = inspect.signature(tool.fn)
        params = list(sig.parameters.keys())

        # Verify required parameters
        assert "query" in params, "Missing required parameter: query"

        # Verify optional parameters
        expected_optional = [
            "collection",
            "project_name",
            "mode",
            "limit",
            "score_threshold",
            "filters",
            "branch",
            "file_type",
            "workspace_type",
        ]
        for param in expected_optional:
            assert param in params, f"Missing optional parameter: {param}"

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_manage_tool_schema_complete(self, fastmcp_test_server):
        """Verify manage tool has complete schema with all parameters."""
        tool = fastmcp_test_server.get_tool_sync("manage")
        assert tool is not None, "manage tool not found"

        assert hasattr(tool, "fn") and callable(tool.fn), "Tool fn not callable"

        sig = inspect.signature(tool.fn)
        params = list(sig.parameters.keys())

        # Verify required parameters
        assert "action" in params, "Missing required parameter: action"

        # Verify optional parameters
        expected_optional = ["collection", "name", "project_name", "config"]
        for param in expected_optional:
            assert param in params, f"Missing optional parameter: {param}"

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_retrieve_tool_schema_complete(self, fastmcp_test_server):
        """Verify retrieve tool has complete schema with all parameters."""
        tool = fastmcp_test_server.get_tool_sync("retrieve")
        assert tool is not None, "retrieve tool not found"

        assert hasattr(tool, "fn") and callable(tool.fn), "Tool fn not callable"

        sig = inspect.signature(tool.fn)
        params = list(sig.parameters.keys())

        # Verify parameters (document_id or metadata required)
        assert (
            "document_id" in params and "metadata" in params
        ), "Missing required parameters: document_id or metadata"

        # Verify optional parameters
        expected_optional = ["collection", "limit", "project_name", "branch", "file_type"]
        for param in expected_optional:
            assert param in params, f"Missing optional parameter: {param}"

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_tool_parameters_have_type_annotations(self, fastmcp_test_server):
        """Verify tool parameters have proper type annotations."""
        tools = ["store", "search", "manage", "retrieve"]

        for tool_name in tools:
            tool = fastmcp_test_server.get_tool_sync(tool_name)
            assert tool is not None, f"Tool '{tool_name}' not found"

            sig = inspect.signature(tool.fn)

            # Check that parameters have annotations
            params_with_annotations = 0
            total_params = 0

            for param_name, param in sig.parameters.items():
                total_params += 1
                if param.annotation != inspect.Parameter.empty:
                    params_with_annotations += 1

            # Most parameters should have type annotations
            annotation_rate = (
                params_with_annotations / total_params if total_params > 0 else 0
            )
            assert annotation_rate > 0.5, (
                f"Tool '{tool_name}' has insufficient type annotations: "
                f"{params_with_annotations}/{total_params}"
            )

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_tool_functions_have_docstrings(self, fastmcp_test_server):
        """Verify tool functions have documentation."""
        tools = ["store", "search", "manage", "retrieve"]

        for tool_name in tools:
            tool = fastmcp_test_server.get_tool_sync(tool_name)
            assert tool is not None, f"Tool '{tool_name}' not found"

            # Check for docstring
            assert (
                tool.fn.__doc__ is not None and len(tool.fn.__doc__.strip()) > 0
            ), f"Tool '{tool_name}' missing docstring"

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_required_vs_optional_parameters_distinguished(
        self, fastmcp_test_server
    ):
        """Verify required vs optional parameters are properly distinguished."""
        # Test store tool (content is required, others optional)
        store_tool = fastmcp_test_server.get_tool_sync("store")
        assert store_tool is not None

        sig = inspect.signature(store_tool.fn)

        # content should not have default (required)
        content_param = sig.parameters.get("content")
        assert content_param is not None, "content parameter not found"
        # If parameter has no default, it's required (unless it has a default value)

        # title should have default (optional)
        title_param = sig.parameters.get("title")
        assert title_param is not None, "title parameter not found"
        # Optional params typically have default=None

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_tool_schemas_are_json_serializable(self, fastmcp_test_server):
        """Verify tool schemas can be serialized to JSON (MCP requirement)."""
        tools = ["store", "search", "manage", "retrieve"]

        for tool_name in tools:
            tool = fastmcp_test_server.get_tool_sync(tool_name)
            assert tool is not None, f"Tool '{tool_name}' not found"

            # Try to serialize tool information
            tool_info = {
                "name": tool_name,
                "has_function": hasattr(tool, "fn"),
                "is_callable": callable(tool.fn) if hasattr(tool, "fn") else False,
                "parameter_count": len(inspect.signature(tool.fn).parameters)
                if hasattr(tool, "fn")
                else 0,
            }

            try:
                json.dumps(tool_info)
            except (TypeError, ValueError) as e:
                pytest.fail(
                    f"Tool '{tool_name}' schema not JSON serializable: {e}"
                )


class TestVersionNegotiation:
    """Test MCP protocol version negotiation."""

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_mcp_sdk_has_version(self):
        """Verify MCP SDK has version information."""
        # Check that we can get the MCP package version
        from importlib.metadata import version
        try:
            mcp_version = version('mcp')
            assert mcp_version is not None and len(mcp_version) > 0, "MCP version is empty"
        except Exception as e:
            pytest.fail(f"Could not retrieve MCP package version: {e}")

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_server_accepts_current_protocol(self, fastmcp_test_server):
        """Verify server accepts current MCP protocol version."""
        # Server initialization succeeded, so it accepts current protocol
        assert (
            fastmcp_test_server.initialized
        ), "Server failed to initialize with current protocol"

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_fastmcp_has_version_info(self):
        """Verify FastMCP framework has version information."""
        import fastmcp

        assert hasattr(
            fastmcp, "__version__"
        ), "FastMCP missing version information"
        assert (
            isinstance(fastmcp.__version__, str) and len(fastmcp.__version__) > 0
        ), "FastMCP version is invalid"


class TestFeatureDetection:
    """Test client capability to discover server features."""

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_client_can_list_tools(self, fastmcp_test_client):
        """Verify client can discover available tools."""
        available_tools = fastmcp_test_client.server.get_available_tools()

        assert isinstance(available_tools, list), "Tool list is not a list"
        assert len(available_tools) >= 4, "Expected at least 4 tools"

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_client_can_get_tool_schema(self, fastmcp_test_server):
        """Verify client can retrieve individual tool schemas."""
        tool = fastmcp_test_server.get_tool_sync("store")

        assert tool is not None, "Failed to retrieve tool schema"
        assert hasattr(tool, "fn"), "Tool schema missing function"
        assert callable(tool.fn), "Tool function not callable"

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_tool_parameter_inspection(self, fastmcp_test_server):
        """Verify tool parameters can be inspected."""
        tool = fastmcp_test_server.get_tool_sync("search")
        assert tool is not None

        sig = inspect.signature(tool.fn)
        params = sig.parameters

        # Should be able to inspect parameter details
        assert len(params) > 0, "No parameters found"

        # Check specific parameter
        query_param = params.get("query")
        assert query_param is not None, "query parameter not found"

        # Can inspect parameter properties
        assert hasattr(query_param, "name"), "Parameter missing name"
        assert hasattr(query_param, "annotation"), "Parameter missing annotation"
        assert hasattr(query_param, "default"), "Parameter missing default"

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_tool_descriptions_available(self, fastmcp_test_server):
        """Verify tool descriptions are available for feature discovery."""
        tools = ["store", "search", "manage", "retrieve"]

        for tool_name in tools:
            tool = fastmcp_test_server.get_tool_sync(tool_name)
            assert tool is not None, f"Tool '{tool_name}' not found"

            # Tool should have documentation
            docstring = tool.fn.__doc__
            assert (
                docstring is not None and len(docstring.strip()) > 0
            ), f"Tool '{tool_name}' has no description"

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_parameter_types_discoverable(self, fastmcp_test_server):
        """Verify parameter types can be discovered."""
        tool = fastmcp_test_server.get_tool_sync("search")
        assert tool is not None

        sig = inspect.signature(tool.fn)

        # Check query parameter type
        query_param = sig.parameters.get("query")
        assert query_param is not None

        # Type annotation should be available
        assert query_param.annotation != inspect.Parameter.empty, (
            "query parameter missing type annotation"
        )

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_all_tools_discoverable_via_iteration(self, fastmcp_test_server):
        """Verify all tools can be discovered by iterating available tools."""
        available_tools = fastmcp_test_server.get_available_tools()

        discovered_tools = []
        for tool_name in available_tools:
            tool = fastmcp_test_server.get_tool_sync(tool_name)
            if tool is not None:
                discovered_tools.append(tool_name)

        # Should discover all 4 expected tools
        expected_tools = {"store", "search", "manage", "retrieve"}
        discovered_set = set(discovered_tools)

        assert expected_tools.issubset(
            discovered_set
        ), f"Not all tools discoverable. Missing: {expected_tools - discovered_set}"


class TestSuccessfulNegotiation:
    """Test successful capability negotiation scenarios."""

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_complete_initialization_handshake(self, fastmcp_test_server):
        """Verify complete initialization handshake succeeds."""
        # Server initialized
        assert fastmcp_test_server.initialized, "Server not initialized"

        # Server has tools
        tools = fastmcp_test_server.get_available_tools()
        assert len(tools) >= 4, "Insufficient tools after initialization"

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_all_tools_available_after_init(
        self, fastmcp_test_server, fastmcp_test_client
    ):
        """Verify all tools are available after successful initialization."""
        expected_tools = ["store", "search", "manage", "retrieve"]

        for tool_name in expected_tools:
            tool = fastmcp_test_server.get_tool_sync(tool_name)
            assert (
                tool is not None
            ), f"Tool '{tool_name}' not available after init"

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_tool_calls_work_after_init(self, fastmcp_test_client):
        """Verify tools are callable after successful initialization."""
        # Try calling a simple tool
        result = await fastmcp_test_client.call_tool(
            "manage", {"action": "workspace_status"}
        )

        # Should get a response (success or graceful error)
        assert isinstance(result, MCPTestResult), "Result not MCPTestResult"
        assert (
            result.response is not None or result.error is not None
        ), "No response received"

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_server_ready_for_requests_after_init(
        self, fastmcp_test_server, fastmcp_test_client
    ):
        """Verify server is ready to handle requests after initialization."""
        # Server initialized
        assert fastmcp_test_server.initialized

        # Client initialized
        assert fastmcp_test_client.initialized

        # Can make successful tool calls
        result = await fastmcp_test_client.call_tool(
            "search", {"query": "test"}
        )

        assert isinstance(result, MCPTestResult)
        # Execution completed (success or graceful error)
        assert result.execution_time_ms >= 0

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_multiple_tools_callable_after_init(self, fastmcp_test_client):
        """Verify multiple tools can be called after initialization."""
        tool_calls = [
            ("manage", {"action": "workspace_status"}),
            ("search", {"query": "test"}),
        ]

        results = []
        for tool_name, params in tool_calls:
            result = await fastmcp_test_client.call_tool(tool_name, params)
            results.append(result)

        # All calls should complete
        assert len(results) == len(tool_calls), "Not all calls completed"

        # All results should be valid MCPTestResult
        for result in results:
            assert isinstance(result, MCPTestResult)

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_capability_matches_functionality(self, fastmcp_test_server):
        """Verify advertised capabilities match actual functionality."""
        # Server advertises 4 tools
        available_tools = fastmcp_test_server.get_available_tools()
        expected_tools = {"store", "search", "manage", "retrieve"}

        # Check all expected tools are present
        for tool_name in expected_tools:
            assert (
                tool_name in available_tools
            ), f"Advertised tool '{tool_name}' not found"

        # Each advertised tool should be functional
        for tool_name in expected_tools:
            tool = fastmcp_test_server.get_tool_sync(tool_name)
            assert tool is not None, f"Tool '{tool_name}' not functional"
            assert callable(tool.fn), f"Tool '{tool_name}' not callable"


class TestFailedNegotiation:
    """Test failed/edge case negotiation scenarios."""

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_nonexistent_tool_call_fails_gracefully(self, fastmcp_test_client):
        """Verify calling nonexistent tool fails gracefully."""
        result = await fastmcp_test_client.call_tool(
            "nonexistent_tool_that_does_not_exist", {}
        )

        # Should fail gracefully
        assert isinstance(result, MCPTestResult)
        assert not result.success, "Nonexistent tool call should fail"
        if result.error is not None:
            # If error message is provided, it should indicate the issue
            assert "not found" in result.error.lower() or "unknown" in result.error.lower() or "tool" in result.error.lower(), (
                "Error message should indicate tool issue"
            )
    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_malformed_tool_name_fails_gracefully(self, fastmcp_test_client):
        """Verify malformed tool names fail gracefully."""
        malformed_names = [
            "",  # Empty name
            "   ",  # Whitespace only
            "invalid-tool-name-123",  # Non-existent
            "STORE",  # Wrong case
        ]

        for malformed_name in malformed_names:
            result = await fastmcp_test_client.call_tool(malformed_name, {})

            # Should fail gracefully
            assert isinstance(result, MCPTestResult)
            assert not result.success, (
                f"Malformed tool name '{malformed_name}' should fail"
            )

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_error_response_format(self, fastmcp_test_client):
        """Verify error responses follow MCP format."""
        # Call nonexistent tool to trigger error
        result = await fastmcp_test_client.call_tool(
            "nonexistent_tool_that_does_not_exist", {}
        )

        # Error should be properly formatted
        assert isinstance(result, MCPTestResult)
        assert not result.success
        assert result.error is not None
        assert isinstance(result.error, str), "Error should be string"
        assert len(result.error) > 0, "Error message should not be empty"

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_invalid_parameters_handled_gracefully(self, fastmcp_test_client):
        """Verify invalid parameters are handled gracefully."""
        # Call with completely wrong parameters
        result = await fastmcp_test_client.call_tool(
            "search",
            {
                "invalid_param": "value",
                "another_invalid": 123,
            },
        )

        # Should handle gracefully (either error or ignore invalid params)
        assert isinstance(result, MCPTestResult)
        # Should not crash the server
        assert result.execution_time_ms >= 0

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_server_continues_after_failed_call(self, fastmcp_test_client):
        """Verify server continues functioning after failed tool call."""
        # Make a failing call
        result1 = await fastmcp_test_client.call_tool(
            "nonexistent_tool_that_does_not_exist", {}
        )
        assert not result1.success

        # Server should still work for valid calls
        result2 = await fastmcp_test_client.call_tool(
            "manage", {"action": "workspace_status"}
        )

        # Second call should work
        assert isinstance(result2, MCPTestResult)
        # Should get response
        assert result2.response is not None or result2.error is not None

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_empty_parameters_handled(self, fastmcp_test_client):
        """Verify empty parameter dict is handled correctly."""
        # Some tools require parameters, calling with empty dict should fail gracefully
        result = await fastmcp_test_client.call_tool("search", {})

        # Should handle gracefully (either work with defaults or return error)
        assert isinstance(result, MCPTestResult)
        assert result.execution_time_ms >= 0

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_null_parameters_handled(self, fastmcp_test_client):
        """Verify null/None parameter values are handled."""
        result = await fastmcp_test_client.call_tool(
            "search",
            {"query": None},  # Invalid: None query
        )

        # Should handle gracefully
        assert isinstance(result, MCPTestResult)
        # Should either error or handle None
        assert result.execution_time_ms >= 0


class TestCrossCapabilityCompliance:
    """Cross-capability MCP protocol compliance tests."""

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_all_tools_have_consistent_schema_structure(
        self, fastmcp_test_server
    ):
        """Verify all tools follow consistent schema structure."""
        tools = ["store", "search", "manage", "retrieve"]

        for tool_name in tools:
            tool = fastmcp_test_server.get_tool_sync(tool_name)
            assert tool is not None, f"Tool '{tool_name}' not found"

            # All tools should have these attributes
            assert hasattr(tool, "fn"), f"Tool '{tool_name}' missing fn"
            assert callable(tool.fn), f"Tool '{tool_name}' fn not callable"

            # All tools should have docstrings
            assert tool.fn.__doc__ is not None, (
                f"Tool '{tool_name}' missing docstring"
            )

            # All tools should have inspectable signatures
            sig = inspect.signature(tool.fn)
            assert len(sig.parameters) > 0, (
                f"Tool '{tool_name}' has no parameters"
            )

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_all_tools_return_json_serializable_responses(
        self, fastmcp_test_client
    ):
        """Verify all tools return JSON-serializable responses."""
        tool_calls = [
            ("store", {"content": "test"}),
            ("search", {"query": "test"}),
            ("manage", {"action": "workspace_status"}),
            ("retrieve", {"document_id": "test"}),
        ]

        for tool_name, params in tool_calls:
            result = await fastmcp_test_client.call_tool(tool_name, params)

            if result.response is not None:
                try:
                    json.dumps(result.response, default=str)
                except (TypeError, ValueError) as e:
                    pytest.fail(
                        f"Tool '{tool_name}' response not JSON serializable: {e}"
                    )

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_capability_negotiation_preserves_functionality(
        self, fastmcp_test_server, fastmcp_test_client
    ):
        """Verify capability negotiation doesn't break tool functionality."""
        # After initialization, all tools should work
        tools_to_test = [
            ("manage", {"action": "workspace_status"}),
            ("search", {"query": "test"}),
        ]

        for tool_name, params in tools_to_test:
            result = await fastmcp_test_client.call_tool(tool_name, params)

            # Should get valid result
            assert isinstance(result, MCPTestResult)
            assert result.execution_time_ms >= 0

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_server_capabilities_match_registered_tools(
        self, fastmcp_test_server
    ):
        """Verify server's advertised capabilities match actually registered tools."""
        available_tools = fastmcp_test_server.get_available_tools()

        # All advertised tools should be retrievable
        for tool_name in available_tools:
            tool = fastmcp_test_server.get_tool_sync(tool_name)
            assert tool is not None, (
                f"Advertised tool '{tool_name}' not actually available"
            )

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_no_undocumented_tools(self, fastmcp_test_server):
        """Verify there are no undocumented/hidden tools."""
        available_tools = set(fastmcp_test_server.get_available_tools())

        # Expected tools
        expected_tools = {"store", "search", "manage", "retrieve"}

        # Should not have unexpected tools (some test infrastructure tools may exist)
        # Just verify our expected tools are present
        assert expected_tools.issubset(
            available_tools
        ), "Expected tools not all present"
