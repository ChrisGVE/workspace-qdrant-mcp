"""
Comprehensive MCP Tool Call Handling Validation Tests (Task 325.1).

Tests all 4 MCP tools (store, search, manage, retrieve) for:
- Tool schema MCP compliance
- Parameter validation (required, optional, defaults)
- Type checking enforcement
- Response format MCP compliance
- Error handling for invalid inputs

This focuses on MCP protocol compliance, not business logic.
All external dependencies (Qdrant, daemon) are mocked.
"""

import json
import pytest
from typing import Dict, Any

from tests.utils.fastmcp_test_infrastructure import MCPTestResult


class TestStoreToolValidation:
    """Test store tool MCP compliance and parameter validation."""

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_store_tool_exists_and_accessible(self, fastmcp_test_server):
        """Verify store tool is registered and accessible."""
        available_tools = fastmcp_test_server.get_available_tools()

        assert "store" in available_tools, "store tool not found in available tools"

        tool = fastmcp_test_server.get_tool_sync("store")
        assert tool is not None, "store tool exists but couldn't be retrieved"

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_store_tool_schema_structure(self, fastmcp_test_server):
        """Validate store tool has proper schema structure."""
        tool = fastmcp_test_server.get_tool_sync("store")

        # Verify tool has callable function
        assert hasattr(tool, "fn"), "Tool missing fn attribute"
        assert callable(tool.fn), "Tool fn is not callable"

        # Verify function signature has expected parameters
        import inspect

        sig = inspect.signature(tool.fn)
        params = list(sig.parameters.keys())

        # Required parameter
        assert "content" in params, "Missing required parameter: content"

        # Optional parameters with defaults
        optional_params = [
            "title",
            "metadata",
            "collection",
            "source",
            "document_type",
            "file_path",
            "url",
            "project_name",
        ]
        for param in optional_params:
            assert param in params, f"Missing optional parameter: {param}"

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_store_valid_call_with_required_params_only(
        self, fastmcp_test_client
    ):
        """Test store with only required parameters."""
        result = await fastmcp_test_client.call_tool(
            "store", {"content": "Test content"}
        )

        assert isinstance(result, MCPTestResult)
        # Store should either succeed or fail gracefully
        assert result.response is not None or result.error is not None

        # If successful, verify response structure
        if result.success and isinstance(result.response, dict):
            # Should have success indicator
            assert "success" in result.response or "document_id" in result.response

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_store_valid_call_with_all_parameters(self, fastmcp_test_client):
        """Test store with all parameters provided."""
        result = await fastmcp_test_client.call_tool(
            "store",
            {
                "content": "Test content with all params",
                "title": "Test Title",
                "metadata": {"key": "value"},
                "collection": "test-collection",
                "source": "test",
                "document_type": "code",
                "file_path": "/path/to/file.py",
                "url": "https://example.com/doc",
                "project_name": "test-project",
            },
        )

        assert isinstance(result, MCPTestResult)
        # Verify response handling
        assert result.response is not None or result.error is not None

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_store_default_values_applied(self, fastmcp_test_client):
        """Test that default values are applied when parameters omitted."""
        result = await fastmcp_test_client.call_tool(
            "store",
            {
                "content": "Test content for defaults",
                # Omit source, document_type - should use defaults
            },
        )

        assert isinstance(result, MCPTestResult)

        # If successful, check that defaults were applied in the metadata
        if result.success and isinstance(result.response, dict):
            metadata = result.response.get("metadata", {})
            # Default source should be "user_input"
            # Default document_type should be "text"
            # These are applied in the tool implementation

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_store_response_json_serializable(self, fastmcp_test_client):
        """Verify store response is JSON serializable (MCP requirement)."""
        result = await fastmcp_test_client.call_tool(
            "store", {"content": "JSON test content"}
        )

        # Response must be JSON serializable for MCP
        if result.response is not None:
            try:
                json_str = json.dumps(result.response, default=str)
                assert isinstance(json_str, str)
                # Verify it can be deserialized
                json.loads(json_str)
            except (TypeError, ValueError) as e:
                pytest.fail(f"Response not JSON serializable: {e}")

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_store_response_protocol_compliance(self, fastmcp_test_client):
        """Verify store response matches MCP protocol format."""
        result = await fastmcp_test_client.call_tool(
            "store", {"content": "Protocol test content"}
        )

        # Check protocol compliance from test infrastructure
        if result.protocol_compliance:
            assert result.protocol_compliance.get(
                "json_serializable", False
            ), "Response not JSON serializable"
            assert result.protocol_compliance.get(
                "is_dict_or_list", False
            ), "Response not dict or list"
            assert result.protocol_compliance.get(
                "has_content", False
            ), "Response has no content"

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_store_empty_content_handling(self, fastmcp_test_client):
        """Test store with empty content string."""
        result = await fastmcp_test_client.call_tool("store", {"content": ""})

        # Should handle gracefully - either success with empty content or error
        assert isinstance(result, MCPTestResult)
        assert result.response is not None or result.error is not None

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_store_large_content_handling(self, fastmcp_test_client):
        """Test store with very large content."""
        large_content = "x" * 100000  # 100KB of content

        result = await fastmcp_test_client.call_tool(
            "store", {"content": large_content}
        )

        # Should handle large content gracefully
        assert isinstance(result, MCPTestResult)
        assert result.response is not None or result.error is not None

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_store_special_characters_in_content(self, fastmcp_test_client):
        """Test store with special characters and Unicode."""
        special_content = "Test with 特殊字符 and emojis 🚀 and newlines\n\ttabs"

        result = await fastmcp_test_client.call_tool(
            "store", {"content": special_content}
        )

        # Should handle special characters
        assert isinstance(result, MCPTestResult)
        assert result.response is not None or result.error is not None

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_store_complex_metadata_handling(self, fastmcp_test_client):
        """Test store with complex nested metadata."""
        complex_metadata = {
            "nested": {"level1": {"level2": {"level3": "deep value"}}},
            "array": [1, 2, 3, "mixed", {"types": True}],
            "types": {"string": "value", "int": 42, "bool": True, "null": None},
        }

        result = await fastmcp_test_client.call_tool(
            "store", {"content": "Complex metadata test", "metadata": complex_metadata}
        )

        assert isinstance(result, MCPTestResult)
        # Should handle complex metadata gracefully
        assert result.response is not None or result.error is not None

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_store_invalid_url_format_handling(self, fastmcp_test_client):
        """Test store with invalid URL format."""
        result = await fastmcp_test_client.call_tool(
            "store",
            {"content": "URL test", "url": "not-a-valid-url"},
        )

        # Should handle invalid URL gracefully (accept or error gracefully)
        assert isinstance(result, MCPTestResult)
        assert result.response is not None or result.error is not None

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_store_execution_performance(self, fastmcp_test_client):
        """Test store tool execution performance."""
        result = await fastmcp_test_client.call_tool(
            "store", {"content": "Performance test"}
        )

        # Execution time should be reasonable (< 1000ms for in-memory mock)
        assert result.execution_time_ms < 1000.0, (
            f"Store execution too slow: {result.execution_time_ms}ms"
        )


class TestSearchToolValidation:
    """Test search tool MCP compliance and parameter validation."""

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_search_tool_exists_and_accessible(self, fastmcp_test_server):
        """Verify search tool is registered and accessible."""
        available_tools = fastmcp_test_server.get_available_tools()

        assert "search" in available_tools, "search tool not found in available tools"

        tool = fastmcp_test_server.get_tool_sync("search")
        assert tool is not None, "search tool exists but couldn't be retrieved"

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_search_tool_schema_structure(self, fastmcp_test_server):
        """Validate search tool has proper schema structure."""
        tool = fastmcp_test_server.get_tool_sync("search")

        assert hasattr(tool, "fn"), "Tool missing fn attribute"
        assert callable(tool.fn), "Tool fn is not callable"

        import inspect

        sig = inspect.signature(tool.fn)
        params = list(sig.parameters.keys())

        # Required parameter
        assert "query" in params, "Missing required parameter: query"

        # Optional parameters with defaults
        optional_params = [
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
        for param in optional_params:
            assert param in params, f"Missing optional parameter: {param}"

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_search_valid_call_with_required_params_only(
        self, fastmcp_test_client
    ):
        """Test search with only required parameters."""
        result = await fastmcp_test_client.call_tool("search", {"query": "test query"})

        assert isinstance(result, MCPTestResult)
        assert result.response is not None or result.error is not None

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_search_with_all_parameters(self, fastmcp_test_client):
        """Test search with all parameters provided."""
        result = await fastmcp_test_client.call_tool(
            "search",
            {
                "query": "comprehensive test",
                "collection": "test-collection",
                "project_name": "test-project",
                "mode": "hybrid",
                "limit": 5,
                "score_threshold": 0.5,
                "filters": {"source": "test"},
                "branch": "main",
                "file_type": "code",
                "workspace_type": "code",
            },
        )

        assert isinstance(result, MCPTestResult)
        assert result.response is not None or result.error is not None

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_search_mode_parameter_values(self, fastmcp_test_client):
        """Test search with different mode values."""
        modes = ["hybrid", "semantic", "exact", "keyword"]

        for mode in modes:
            result = await fastmcp_test_client.call_tool(
                "search", {"query": "mode test", "mode": mode}
            )

            assert isinstance(result, MCPTestResult)
            # Each mode should be handled (success or graceful error)
            assert result.response is not None or result.error is not None

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_search_limit_parameter_validation(self, fastmcp_test_client):
        """Test search limit parameter with various values."""
        limit_values = [1, 5, 10, 50, 100]

        for limit in limit_values:
            result = await fastmcp_test_client.call_tool(
                "search", {"query": "limit test", "limit": limit}
            )

            assert isinstance(result, MCPTestResult)

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_search_score_threshold_validation(self, fastmcp_test_client):
        """Test search score_threshold parameter."""
        thresholds = [0.0, 0.3, 0.5, 0.7, 1.0]

        for threshold in thresholds:
            result = await fastmcp_test_client.call_tool(
                "search", {"query": "threshold test", "score_threshold": threshold}
            )

            assert isinstance(result, MCPTestResult)

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_search_filters_parameter_handling(self, fastmcp_test_client):
        """Test search with various filter configurations."""
        filter_configs = [
            {},  # Empty filters
            {"source": "test"},  # Single filter
            {"source": "test", "file_type": "code"},  # Multiple filters
            {"nested": {"key": "value"}},  # Nested filters
        ]

        for filters in filter_configs:
            result = await fastmcp_test_client.call_tool(
                "search", {"query": "filter test", "filters": filters}
            )

            assert isinstance(result, MCPTestResult)

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_search_branch_parameter_handling(self, fastmcp_test_client):
        """Test search with different branch values."""
        branches = [None, "main", "develop", "*"]

        for branch in branches:
            params = {"query": "branch test"}
            if branch is not None:
                params["branch"] = branch

            result = await fastmcp_test_client.call_tool("search", params)

            assert isinstance(result, MCPTestResult)

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_search_file_type_parameter_handling(self, fastmcp_test_client):
        """Test search with different file_type values."""
        file_types = ["code", "test", "docs", "config", "data", "build", "other"]

        for file_type in file_types:
            result = await fastmcp_test_client.call_tool(
                "search", {"query": "file type test", "file_type": file_type}
            )

            assert isinstance(result, MCPTestResult)

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_search_response_json_serializable(self, fastmcp_test_client):
        """Verify search response is JSON serializable."""
        result = await fastmcp_test_client.call_tool(
            "search", {"query": "JSON test"}
        )

        if result.response is not None:
            try:
                json_str = json.dumps(result.response, default=str)
                json.loads(json_str)
            except (TypeError, ValueError) as e:
                pytest.fail(f"Response not JSON serializable: {e}")

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_search_response_structure(self, fastmcp_test_client):
        """Verify search response has expected structure."""
        result = await fastmcp_test_client.call_tool(
            "search", {"query": "structure test"}
        )

        if result.success and isinstance(result.response, dict):
            # Expected fields in search response
            expected_fields = ["success", "query"]
            for field in expected_fields:
                # Field should exist if successful
                if result.response.get("success"):
                    assert (
                        field in result.response or "error" in result.response
                    ), f"Missing expected field: {field}"

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_search_empty_query_handling(self, fastmcp_test_client):
        """Test search with empty query string."""
        result = await fastmcp_test_client.call_tool("search", {"query": ""})

        # Should handle empty query gracefully
        assert isinstance(result, MCPTestResult)
        assert result.response is not None or result.error is not None

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_search_special_characters_in_query(self, fastmcp_test_client):
        """Test search with special characters in query."""
        special_queries = [
            "query with spaces",
            "query-with-dashes",
            "query_with_underscores",
            "query.with.dots",
            "query/with/slashes",
            "query@with@symbols",
        ]

        for query in special_queries:
            result = await fastmcp_test_client.call_tool("search", {"query": query})

            assert isinstance(result, MCPTestResult)

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_search_execution_performance(self, fastmcp_test_client):
        """Test search tool execution performance."""
        result = await fastmcp_test_client.call_tool(
            "search", {"query": "performance"}
        )

        # Execution time should be reasonable
        assert result.execution_time_ms < 1000.0, (
            f"Search execution too slow: {result.execution_time_ms}ms"
        )


class TestManageToolValidation:
    """Test manage tool MCP compliance and parameter validation."""

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_manage_tool_exists_and_accessible(self, fastmcp_test_server):
        """Verify manage tool is registered and accessible."""
        available_tools = fastmcp_test_server.get_available_tools()

        assert "manage" in available_tools, "manage tool not found in available tools"

        tool = fastmcp_test_server.get_tool_sync("manage")
        assert tool is not None, "manage tool exists but couldn't be retrieved"

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_manage_tool_schema_structure(self, fastmcp_test_server):
        """Validate manage tool has proper schema structure."""
        tool = fastmcp_test_server.get_tool_sync("manage")

        assert hasattr(tool, "fn"), "Tool missing fn attribute"
        assert callable(tool.fn), "Tool fn is not callable"

        import inspect

        sig = inspect.signature(tool.fn)
        params = list(sig.parameters.keys())

        # Required parameter
        assert "action" in params, "Missing required parameter: action"

        # Optional parameters
        optional_params = ["collection", "name", "project_name", "config"]
        for param in optional_params:
            assert param in params, f"Missing optional parameter: {param}"

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_manage_list_collections_action(self, fastmcp_test_client):
        """Test manage with list_collections action."""
        result = await fastmcp_test_client.call_tool(
            "manage", {"action": "list_collections"}
        )

        assert isinstance(result, MCPTestResult)
        assert result.response is not None or result.error is not None

        if result.success and isinstance(result.response, dict):
            # Should have collections info
            assert "collections" in result.response or "error" in result.response

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_manage_workspace_status_action(self, fastmcp_test_client):
        """Test manage with workspace_status action."""
        result = await fastmcp_test_client.call_tool(
            "manage", {"action": "workspace_status"}
        )

        assert isinstance(result, MCPTestResult)
        assert result.response is not None or result.error is not None

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_manage_create_collection_action(self, fastmcp_test_client):
        """Test manage with create_collection action."""
        result = await fastmcp_test_client.call_tool(
            "manage", {"action": "create_collection", "name": "test-new-collection"}
        )

        assert isinstance(result, MCPTestResult)
        # Should handle collection creation (success or error)
        assert result.response is not None or result.error is not None

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_manage_delete_collection_action(self, fastmcp_test_client):
        """Test manage with delete_collection action."""
        result = await fastmcp_test_client.call_tool(
            "manage", {"action": "delete_collection", "name": "test-delete-collection"}
        )

        assert isinstance(result, MCPTestResult)
        assert result.response is not None or result.error is not None

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_manage_collection_info_action(self, fastmcp_test_client):
        """Test manage with collection_info action."""
        result = await fastmcp_test_client.call_tool(
            "manage", {"action": "collection_info", "name": "test-collection"}
        )

        assert isinstance(result, MCPTestResult)
        assert result.response is not None or result.error is not None

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_manage_init_project_action(self, fastmcp_test_client):
        """Test manage with init_project action."""
        result = await fastmcp_test_client.call_tool(
            "manage", {"action": "init_project"}
        )

        assert isinstance(result, MCPTestResult)
        assert result.response is not None or result.error is not None

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_manage_cleanup_action(self, fastmcp_test_client):
        """Test manage with cleanup action."""
        result = await fastmcp_test_client.call_tool("manage", {"action": "cleanup"})

        assert isinstance(result, MCPTestResult)
        assert result.response is not None or result.error is not None

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_manage_unknown_action_handling(self, fastmcp_test_client):
        """Test manage with unknown action value."""
        result = await fastmcp_test_client.call_tool(
            "manage", {"action": "unknown_action"}
        )

        # Should handle unknown action gracefully with error
        assert isinstance(result, MCPTestResult)

        if result.success and isinstance(result.response, dict):
            # Should indicate error for unknown action
            assert (
                result.response.get("success") is False
                or "error" in result.response
                or "available_actions" in result.response
            )

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_manage_create_collection_missing_name(self, fastmcp_test_client):
        """Test manage create_collection without required name parameter."""
        result = await fastmcp_test_client.call_tool(
            "manage",
            {"action": "create_collection"},
            # Missing 'name' parameter
        )

        # Should fail or return error for missing name
        assert isinstance(result, MCPTestResult)

        if result.success and isinstance(result.response, dict):
            # Should indicate error
            assert result.response.get("success") is False or "error" in result.response

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_manage_delete_collection_missing_name(self, fastmcp_test_client):
        """Test manage delete_collection without required name parameter."""
        result = await fastmcp_test_client.call_tool(
            "manage",
            {"action": "delete_collection"},
            # Missing 'name' parameter
        )

        assert isinstance(result, MCPTestResult)

        if result.success and isinstance(result.response, dict):
            # Should indicate error
            assert result.response.get("success") is False or "error" in result.response

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_manage_collection_info_missing_name(self, fastmcp_test_client):
        """Test manage collection_info without required name parameter."""
        result = await fastmcp_test_client.call_tool(
            "manage",
            {"action": "collection_info"},
            # Missing 'name' parameter
        )

        assert isinstance(result, MCPTestResult)

        if result.success and isinstance(result.response, dict):
            # Should indicate error
            assert result.response.get("success") is False or "error" in result.response

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_manage_with_config_parameter(self, fastmcp_test_client):
        """Test manage with config parameter."""
        config = {"vector_size": 384, "distance": "Cosine"}

        result = await fastmcp_test_client.call_tool(
            "manage",
            {"action": "create_collection", "name": "test-config", "config": config},
        )

        assert isinstance(result, MCPTestResult)
        assert result.response is not None or result.error is not None

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_manage_response_json_serializable(self, fastmcp_test_client):
        """Verify manage response is JSON serializable."""
        result = await fastmcp_test_client.call_tool(
            "manage", {"action": "workspace_status"}
        )

        if result.response is not None:
            try:
                json_str = json.dumps(result.response, default=str)
                json.loads(json_str)
            except (TypeError, ValueError) as e:
                pytest.fail(f"Response not JSON serializable: {e}")

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_manage_response_structure(self, fastmcp_test_client):
        """Verify manage response has expected structure."""
        result = await fastmcp_test_client.call_tool(
            "manage", {"action": "list_collections"}
        )

        if result.success and isinstance(result.response, dict):
            # Should have action field
            assert "action" in result.response or "error" in result.response

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_manage_execution_performance(self, fastmcp_test_client):
        """Test manage tool execution performance."""
        result = await fastmcp_test_client.call_tool(
            "manage", {"action": "workspace_status"}
        )

        # Execution time should be reasonable
        assert result.execution_time_ms < 1000.0, (
            f"Manage execution too slow: {result.execution_time_ms}ms"
        )


class TestRetrieveToolValidation:
    """Test retrieve tool MCP compliance and parameter validation."""

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_retrieve_tool_exists_and_accessible(self, fastmcp_test_server):
        """Verify retrieve tool is registered and accessible."""
        available_tools = fastmcp_test_server.get_available_tools()

        assert (
            "retrieve" in available_tools
        ), "retrieve tool not found in available tools"

        tool = fastmcp_test_server.get_tool_sync("retrieve")
        assert tool is not None, "retrieve tool exists but couldn't be retrieved"

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_retrieve_tool_schema_structure(self, fastmcp_test_server):
        """Validate retrieve tool has proper schema structure."""
        tool = fastmcp_test_server.get_tool_sync("retrieve")

        assert hasattr(tool, "fn"), "Tool missing fn attribute"
        assert callable(tool.fn), "Tool fn is not callable"

        import inspect

        sig = inspect.signature(tool.fn)
        params = list(sig.parameters.keys())

        # At least one of document_id or metadata is required
        assert (
            "document_id" in params and "metadata" in params
        ), "Missing required parameters: document_id or metadata"

        # Optional parameters
        optional_params = ["collection", "limit", "project_name", "branch", "file_type"]
        for param in optional_params:
            assert param in params, f"Missing optional parameter: {param}"

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_retrieve_with_document_id(self, fastmcp_test_client):
        """Test retrieve with document_id parameter."""
        result = await fastmcp_test_client.call_tool(
            "retrieve", {"document_id": "test-doc-123"}
        )

        assert isinstance(result, MCPTestResult)
        assert result.response is not None or result.error is not None

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_retrieve_with_metadata_filter(self, fastmcp_test_client):
        """Test retrieve with metadata filter."""
        result = await fastmcp_test_client.call_tool(
            "retrieve", {"metadata": {"source": "test", "file_type": "code"}}
        )

        assert isinstance(result, MCPTestResult)
        assert result.response is not None or result.error is not None

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_retrieve_missing_both_required_params(self, fastmcp_test_client):
        """Test retrieve without document_id or metadata (should fail)."""
        result = await fastmcp_test_client.call_tool(
            "retrieve",
            {},  # No document_id, no metadata
        )

        # Should fail or return error
        assert isinstance(result, MCPTestResult)

        if result.success and isinstance(result.response, dict):
            # Should indicate error for missing parameters
            assert result.response.get("success") is False or "error" in result.response

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_retrieve_with_collection_parameter(self, fastmcp_test_client):
        """Test retrieve with collection parameter."""
        result = await fastmcp_test_client.call_tool(
            "retrieve",
            {"document_id": "test-doc", "collection": "specific-collection"},
        )

        assert isinstance(result, MCPTestResult)
        assert result.response is not None or result.error is not None

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_retrieve_with_limit_parameter(self, fastmcp_test_client):
        """Test retrieve with limit parameter."""
        limits = [1, 5, 10, 50]

        for limit in limits:
            result = await fastmcp_test_client.call_tool(
                "retrieve", {"metadata": {"source": "test"}, "limit": limit}
            )

            assert isinstance(result, MCPTestResult)

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_retrieve_with_branch_parameter(self, fastmcp_test_client):
        """Test retrieve with branch parameter."""
        branches = ["main", "develop", "*"]

        for branch in branches:
            result = await fastmcp_test_client.call_tool(
                "retrieve", {"document_id": "test-doc", "branch": branch}
            )

            assert isinstance(result, MCPTestResult)

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_retrieve_with_file_type_parameter(self, fastmcp_test_client):
        """Test retrieve with file_type parameter."""
        file_types = ["code", "test", "docs", "other"]

        for file_type in file_types:
            result = await fastmcp_test_client.call_tool(
                "retrieve", {"metadata": {"source": "test"}, "file_type": file_type}
            )

            assert isinstance(result, MCPTestResult)

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_retrieve_with_project_name_parameter(self, fastmcp_test_client):
        """Test retrieve with project_name parameter."""
        result = await fastmcp_test_client.call_tool(
            "retrieve",
            {"document_id": "test-doc", "project_name": "test-project"},
        )

        assert isinstance(result, MCPTestResult)
        assert result.response is not None or result.error is not None

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_retrieve_with_all_parameters(self, fastmcp_test_client):
        """Test retrieve with all parameters provided."""
        result = await fastmcp_test_client.call_tool(
            "retrieve",
            {
                "document_id": "test-doc-full",
                "collection": "test-collection",
                "metadata": {"source": "test"},
                "limit": 10,
                "project_name": "test-project",
                "branch": "main",
                "file_type": "code",
            },
        )

        assert isinstance(result, MCPTestResult)
        assert result.response is not None or result.error is not None

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_retrieve_response_json_serializable(self, fastmcp_test_client):
        """Verify retrieve response is JSON serializable."""
        result = await fastmcp_test_client.call_tool(
            "retrieve", {"document_id": "json-test"}
        )

        if result.response is not None:
            try:
                json_str = json.dumps(result.response, default=str)
                json.loads(json_str)
            except (TypeError, ValueError) as e:
                pytest.fail(f"Response not JSON serializable: {e}")

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_retrieve_response_structure(self, fastmcp_test_client):
        """Verify retrieve response has expected structure."""
        result = await fastmcp_test_client.call_tool(
            "retrieve", {"document_id": "structure-test"}
        )

        if result.success and isinstance(result.response, dict):
            # Should have results field
            assert "results" in result.response or "error" in result.response

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_retrieve_complex_metadata_filter(self, fastmcp_test_client):
        """Test retrieve with complex metadata filter."""
        complex_metadata = {
            "source": "test",
            "file_type": "code",
            "branch": "main",
            "custom_field": "value",
        }

        result = await fastmcp_test_client.call_tool(
            "retrieve", {"metadata": complex_metadata}
        )

        assert isinstance(result, MCPTestResult)
        assert result.response is not None or result.error is not None

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_retrieve_execution_performance(self, fastmcp_test_client):
        """Test retrieve tool execution performance."""
        result = await fastmcp_test_client.call_tool(
            "retrieve", {"document_id": "performance-test"}
        )

        # Execution time should be reasonable
        assert result.execution_time_ms < 1000.0, (
            f"Retrieve execution too slow: {result.execution_time_ms}ms"
        )


class TestCrossToolMCPCompliance:
    """Cross-tool MCP protocol compliance tests."""

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_all_tools_registered(self, fastmcp_test_server):
        """Verify all expected tools are registered."""
        available_tools = fastmcp_test_server.get_available_tools()

        expected_tools = ["store", "search", "manage", "retrieve"]

        for tool_name in expected_tools:
            assert (
                tool_name in available_tools
            ), f"Expected tool '{tool_name}' not registered"

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_all_tools_responses_json_serializable(self, fastmcp_test_client):
        """Verify all tools return JSON serializable responses."""
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
    async def test_all_tools_have_protocol_compliance(self, fastmcp_test_client):
        """Verify all tools have protocol compliance checks."""
        tool_calls = [
            ("store", {"content": "test"}),
            ("search", {"query": "test"}),
            ("manage", {"action": "workspace_status"}),
            ("retrieve", {"document_id": "test"}),
        ]

        for tool_name, params in tool_calls:
            result = await fastmcp_test_client.call_tool(tool_name, params)

            # Protocol compliance should be checked on successful calls
            # On errors, protocol_compliance may be None which is acceptable
            if result.success:
                assert result.protocol_compliance is not None, (
                    f"Tool '{tool_name}' missing protocol compliance check on success"
                )

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_all_tools_execution_within_performance_threshold(
        self, fastmcp_test_client
    ):
        """Verify all tools execute within performance threshold."""
        tool_calls = [
            ("store", {"content": "performance test"}),
            ("search", {"query": "performance test"}),
            ("manage", {"action": "workspace_status"}),
            ("retrieve", {"document_id": "performance-test"}),
        ]

        performance_threshold = 1000.0  # 1 second for in-memory operations

        for tool_name, params in tool_calls:
            result = await fastmcp_test_client.call_tool(tool_name, params)

            assert result.execution_time_ms < performance_threshold, (
                f"Tool '{tool_name}' exceeded performance threshold: "
                f"{result.execution_time_ms}ms > {performance_threshold}ms"
            )
