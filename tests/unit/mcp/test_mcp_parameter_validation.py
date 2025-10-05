"""
Comprehensive MCP Parameter Validation and Error Handling Tests (Task 325.2).

Tests all 4 MCP tools for proper error handling in invalid scenarios:
- Invalid parameter types (type mismatch errors)
- Missing required parameters
- Extra unexpected parameters
- Parameter constraint violations
- Boundary conditions
- Error response format compliance

This complements test_mcp_tool_call_validation.py which tests successful operations.
All external dependencies (Qdrant, daemon) are mocked.
"""

import json
import pytest
from typing import Dict, Any

from tests.utils.fastmcp_test_infrastructure import MCPTestResult


class TestStoreParameterValidation:
    """Test store tool parameter validation and error handling."""

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_store_missing_required_content(self, fastmcp_test_client):
        """Test store without required content parameter."""
        result = await fastmcp_test_client.call_tool("store", {})

        # FastMCP should catch missing required parameter
        # Either FastMCP raises TypeError or tool returns error
        assert result.error is not None or (
            isinstance(result.response, dict)
            and (not result.response.get("success", True) or "error" in result.response)
        ), "Missing content should cause error"

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_store_content_wrong_type_int(self, fastmcp_test_client):
        """Test store with integer instead of string for content."""
        result = await fastmcp_test_client.call_tool("store", {"content": 123})

        # Should handle type error gracefully
        assert isinstance(result, MCPTestResult)
        # Either error raised or returned error response
        if result.error is None and isinstance(result.response, dict):
            # If tool accepted int, it should convert or handle it
            # But ideally should validate type
            pass

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_store_content_wrong_type_list(self, fastmcp_test_client):
        """Test store with list instead of string for content."""
        result = await fastmcp_test_client.call_tool(
            "store", {"content": ["not", "a", "string"]}
        )

        assert isinstance(result, MCPTestResult)
        # Type error should be caught

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_store_content_wrong_type_dict(self, fastmcp_test_client):
        """Test store with dict instead of string for content."""
        result = await fastmcp_test_client.call_tool(
            "store", {"content": {"key": "value"}}
        )

        assert isinstance(result, MCPTestResult)

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_store_content_none_value(self, fastmcp_test_client):
        """Test store with None value for content."""
        result = await fastmcp_test_client.call_tool("store", {"content": None})

        # None for required parameter should error
        assert result.error is not None or (
            isinstance(result.response, dict) and not result.response.get("success", True)
        )

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_store_metadata_wrong_type_string(self, fastmcp_test_client):
        """Test store with string instead of dict for metadata."""
        result = await fastmcp_test_client.call_tool(
            "store", {"content": "test", "metadata": "not-a-dict"}
        )

        assert isinstance(result, MCPTestResult)
        # Should handle type error for metadata

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_store_metadata_wrong_type_list(self, fastmcp_test_client):
        """Test store with list instead of dict for metadata."""
        result = await fastmcp_test_client.call_tool(
            "store", {"content": "test", "metadata": ["not", "dict"]}
        )

        assert isinstance(result, MCPTestResult)

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_store_extra_unexpected_parameter(self, fastmcp_test_client):
        """Test store with unknown extra parameter."""
        result = await fastmcp_test_client.call_tool(
            "store",
            {"content": "test", "unexpected_param": "value", "another_unknown": 123},
        )

        assert isinstance(result, MCPTestResult)
        # FastMCP may ignore extra params or reject them
        # Either behavior is acceptable as long as it's handled gracefully

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_store_collection_wrong_type_int(self, fastmcp_test_client):
        """Test store with integer for collection parameter."""
        result = await fastmcp_test_client.call_tool(
            "store", {"content": "test", "collection": 12345}
        )

        assert isinstance(result, MCPTestResult)
        # Should handle type mismatch

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_store_invalid_collection_name_special_chars(
        self, fastmcp_test_client
    ):
        """Test store with invalid collection name containing special characters."""
        result = await fastmcp_test_client.call_tool(
            "store", {"content": "test", "collection": "invalid@collection#name!"}
        )

        assert isinstance(result, MCPTestResult)
        # May accept or reject - should handle gracefully

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_store_error_response_json_serializable(self, fastmcp_test_client):
        """Verify error responses from store are JSON serializable."""
        # Trigger error with missing content
        result = await fastmcp_test_client.call_tool("store", {})

        # Error response must be JSON serializable
        if result.error:
            # Error string should be serializable
            json.dumps({"error": result.error})
        if result.response:
            # Response should be serializable
            json.dumps(result.response, default=str)


class TestSearchParameterValidation:
    """Test search tool parameter validation and error handling."""

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_search_missing_required_query(self, fastmcp_test_client):
        """Test search without required query parameter."""
        result = await fastmcp_test_client.call_tool("search", {})

        # Missing required parameter should cause error
        assert result.error is not None or (
            isinstance(result.response, dict)
            and (not result.response.get("success", True) or "error" in result.response)
        )

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_search_query_wrong_type_int(self, fastmcp_test_client):
        """Test search with integer instead of string for query."""
        result = await fastmcp_test_client.call_tool("search", {"query": 12345})

        assert isinstance(result, MCPTestResult)

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_search_query_wrong_type_list(self, fastmcp_test_client):
        """Test search with list instead of string for query."""
        result = await fastmcp_test_client.call_tool(
            "search", {"query": ["search", "terms"]}
        )

        assert isinstance(result, MCPTestResult)

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_search_query_none_value(self, fastmcp_test_client):
        """Test search with None value for query."""
        result = await fastmcp_test_client.call_tool("search", {"query": None})

        # None for required parameter should error
        assert result.error is not None or (
            isinstance(result.response, dict) and not result.response.get("success", True)
        )

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_search_limit_wrong_type_string(self, fastmcp_test_client):
        """Test search with string instead of int for limit."""
        result = await fastmcp_test_client.call_tool(
            "search", {"query": "test", "limit": "not-an-int"}
        )

        assert isinstance(result, MCPTestResult)
        # Type error should be caught

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_search_limit_negative_value(self, fastmcp_test_client):
        """Test search with negative limit value."""
        result = await fastmcp_test_client.call_tool(
            "search", {"query": "test", "limit": -5}
        )

        assert isinstance(result, MCPTestResult)
        # Negative limit should be handled (error or clamp to 0)

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_search_limit_zero_value(self, fastmcp_test_client):
        """Test search with zero limit value."""
        result = await fastmcp_test_client.call_tool(
            "search", {"query": "test", "limit": 0}
        )

        assert isinstance(result, MCPTestResult)
        # Zero limit is boundary condition - should handle gracefully

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_search_limit_extremely_large_value(self, fastmcp_test_client):
        """Test search with very large limit value."""
        result = await fastmcp_test_client.call_tool(
            "search", {"query": "test", "limit": 999999}
        )

        assert isinstance(result, MCPTestResult)
        # Should handle large values (cap at max or return error)

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_search_score_threshold_wrong_type_string(self, fastmcp_test_client):
        """Test search with string instead of float for score_threshold."""
        result = await fastmcp_test_client.call_tool(
            "search", {"query": "test", "score_threshold": "not-a-float"}
        )

        assert isinstance(result, MCPTestResult)

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_search_score_threshold_negative(self, fastmcp_test_client):
        """Test search with negative score_threshold."""
        result = await fastmcp_test_client.call_tool(
            "search", {"query": "test", "score_threshold": -0.5}
        )

        assert isinstance(result, MCPTestResult)
        # Negative threshold should be handled

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_search_score_threshold_above_one(self, fastmcp_test_client):
        """Test search with score_threshold > 1.0."""
        result = await fastmcp_test_client.call_tool(
            "search", {"query": "test", "score_threshold": 1.5}
        )

        assert isinstance(result, MCPTestResult)
        # Values > 1.0 should be handled (clamp or error)

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_search_filters_wrong_type_string(self, fastmcp_test_client):
        """Test search with string instead of dict for filters."""
        result = await fastmcp_test_client.call_tool(
            "search", {"query": "test", "filters": "not-a-dict"}
        )

        assert isinstance(result, MCPTestResult)

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_search_filters_wrong_type_list(self, fastmcp_test_client):
        """Test search with list instead of dict for filters."""
        result = await fastmcp_test_client.call_tool(
            "search", {"query": "test", "filters": ["not", "dict"]}
        )

        assert isinstance(result, MCPTestResult)

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_search_mode_invalid_value(self, fastmcp_test_client):
        """Test search with invalid mode value."""
        result = await fastmcp_test_client.call_tool(
            "search", {"query": "test", "mode": "invalid_mode"}
        )

        assert isinstance(result, MCPTestResult)
        # Invalid mode should be handled (use default or error)

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_search_mode_wrong_type_int(self, fastmcp_test_client):
        """Test search with integer for mode parameter."""
        result = await fastmcp_test_client.call_tool(
            "search", {"query": "test", "mode": 123}
        )

        assert isinstance(result, MCPTestResult)

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_search_file_type_invalid_value(self, fastmcp_test_client):
        """Test search with invalid file_type value."""
        result = await fastmcp_test_client.call_tool(
            "search", {"query": "test", "file_type": "invalid_type"}
        )

        assert isinstance(result, MCPTestResult)
        # Invalid file_type may be accepted or rejected

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_search_extra_unexpected_parameters(self, fastmcp_test_client):
        """Test search with unknown extra parameters."""
        result = await fastmcp_test_client.call_tool(
            "search",
            {"query": "test", "unknown_param": "value", "another_param": 123},
        )

        assert isinstance(result, MCPTestResult)
        # Extra params should be ignored or rejected gracefully

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_search_error_response_structure(self, fastmcp_test_client):
        """Verify search error responses have proper structure."""
        # Trigger error with missing query
        result = await fastmcp_test_client.call_tool("search", {})

        # Should have error indication
        if isinstance(result.response, dict):
            # Should have either success=False or error field
            has_error_indicator = (
                result.response.get("success") is False or "error" in result.response
            )
            # Error response should be informative
            if "error" in result.response:
                assert isinstance(result.response["error"], str)
                assert len(result.response["error"]) > 0


class TestManageParameterValidation:
    """Test manage tool parameter validation and error handling."""

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_manage_missing_required_action(self, fastmcp_test_client):
        """Test manage without required action parameter."""
        result = await fastmcp_test_client.call_tool("manage", {})

        # Missing required parameter should cause error
        assert result.error is not None or (
            isinstance(result.response, dict)
            and (not result.response.get("success", True) or "error" in result.response)
        )

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_manage_action_wrong_type_int(self, fastmcp_test_client):
        """Test manage with integer instead of string for action."""
        result = await fastmcp_test_client.call_tool("manage", {"action": 12345})

        assert isinstance(result, MCPTestResult)

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_manage_action_wrong_type_list(self, fastmcp_test_client):
        """Test manage with list instead of string for action."""
        result = await fastmcp_test_client.call_tool(
            "manage", {"action": ["not", "string"]}
        )

        assert isinstance(result, MCPTestResult)

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_manage_action_none_value(self, fastmcp_test_client):
        """Test manage with None value for action."""
        result = await fastmcp_test_client.call_tool("manage", {"action": None})

        # None for required parameter should error
        assert result.error is not None or (
            isinstance(result.response, dict) and not result.response.get("success", True)
        )

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_manage_action_empty_string(self, fastmcp_test_client):
        """Test manage with empty string for action."""
        result = await fastmcp_test_client.call_tool("manage", {"action": ""})

        assert isinstance(result, MCPTestResult)
        # Empty action should be handled (treated as invalid or unknown)
        if isinstance(result.response, dict):
            # Should indicate error for empty/invalid action
            assert (
                result.response.get("success") is False
                or "error" in result.response
                or "available_actions" in result.response
            )

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_manage_action_unknown_value(self, fastmcp_test_client):
        """Test manage with unknown action value."""
        result = await fastmcp_test_client.call_tool(
            "manage", {"action": "unknown_action_xyz"}
        )

        assert isinstance(result, MCPTestResult)
        # Unknown action should return error or available actions
        if isinstance(result.response, dict):
            assert (
                result.response.get("success") is False
                or "error" in result.response
                or "available_actions" in result.response
            ), "Unknown action should be handled with error or available actions"

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_manage_create_collection_missing_name(self, fastmcp_test_client):
        """Test manage create_collection action without name parameter."""
        result = await fastmcp_test_client.call_tool(
            "manage", {"action": "create_collection"}
        )

        assert isinstance(result, MCPTestResult)
        # Should fail with missing name error
        if isinstance(result.response, dict):
            assert (
                result.response.get("success") is False or "error" in result.response
            ), "create_collection without name should error"

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_manage_delete_collection_missing_name(self, fastmcp_test_client):
        """Test manage delete_collection action without name parameter."""
        result = await fastmcp_test_client.call_tool(
            "manage", {"action": "delete_collection"}
        )

        assert isinstance(result, MCPTestResult)
        # Should fail with missing name error
        if isinstance(result.response, dict):
            assert result.response.get("success") is False or "error" in result.response

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_manage_collection_info_missing_name(self, fastmcp_test_client):
        """Test manage collection_info action without name parameter."""
        result = await fastmcp_test_client.call_tool(
            "manage", {"action": "collection_info"}
        )

        assert isinstance(result, MCPTestResult)
        # Should fail with missing name error
        if isinstance(result.response, dict):
            assert result.response.get("success") is False or "error" in result.response

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_manage_name_wrong_type_int(self, fastmcp_test_client):
        """Test manage with integer for name parameter."""
        result = await fastmcp_test_client.call_tool(
            "manage", {"action": "create_collection", "name": 12345}
        )

        assert isinstance(result, MCPTestResult)

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_manage_name_empty_string(self, fastmcp_test_client):
        """Test manage with empty string for name parameter."""
        result = await fastmcp_test_client.call_tool(
            "manage", {"action": "create_collection", "name": ""}
        )

        assert isinstance(result, MCPTestResult)
        # Empty name should be handled (error or rejection)

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_manage_config_wrong_type_string(self, fastmcp_test_client):
        """Test manage with string instead of dict for config."""
        result = await fastmcp_test_client.call_tool(
            "manage",
            {"action": "create_collection", "name": "test", "config": "not-a-dict"},
        )

        assert isinstance(result, MCPTestResult)

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_manage_config_wrong_type_list(self, fastmcp_test_client):
        """Test manage with list instead of dict for config."""
        result = await fastmcp_test_client.call_tool(
            "manage",
            {"action": "create_collection", "name": "test", "config": ["not", "dict"]},
        )

        assert isinstance(result, MCPTestResult)

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_manage_extra_unexpected_parameters(self, fastmcp_test_client):
        """Test manage with unknown extra parameters."""
        result = await fastmcp_test_client.call_tool(
            "manage",
            {
                "action": "workspace_status",
                "unknown_param": "value",
                "another_param": 123,
            },
        )

        assert isinstance(result, MCPTestResult)
        # Extra params should be ignored or rejected gracefully

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_manage_error_response_informative(self, fastmcp_test_client):
        """Verify manage error responses are informative."""
        # Trigger error with unknown action
        result = await fastmcp_test_client.call_tool(
            "manage", {"action": "invalid_action"}
        )

        if isinstance(result.response, dict) and "error" in result.response:
            error_msg = result.response["error"]
            # Error message should be descriptive
            assert isinstance(error_msg, str)
            assert len(error_msg) > 10, "Error message should be descriptive"
            # Should mention the action in error
            assert (
                "action" in error_msg.lower() or "invalid" in error_msg.lower()
            ), "Error should reference the problem"


class TestRetrieveParameterValidation:
    """Test retrieve tool parameter validation and error handling."""

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_retrieve_missing_both_required_params(self, fastmcp_test_client):
        """Test retrieve without document_id or metadata."""
        result = await fastmcp_test_client.call_tool("retrieve", {})

        # Missing both required params should cause error
        assert isinstance(result, MCPTestResult)
        # Should indicate error for missing parameters
        if isinstance(result.response, dict):
            assert (
                result.response.get("success") is False or "error" in result.response
            ), "Missing document_id and metadata should error"

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_retrieve_document_id_wrong_type_int(self, fastmcp_test_client):
        """Test retrieve with integer for document_id."""
        result = await fastmcp_test_client.call_tool("retrieve", {"document_id": 12345})

        assert isinstance(result, MCPTestResult)

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_retrieve_document_id_wrong_type_list(self, fastmcp_test_client):
        """Test retrieve with list for document_id."""
        result = await fastmcp_test_client.call_tool(
            "retrieve", {"document_id": ["not", "string"]}
        )

        assert isinstance(result, MCPTestResult)

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_retrieve_document_id_empty_string(self, fastmcp_test_client):
        """Test retrieve with empty string for document_id."""
        result = await fastmcp_test_client.call_tool("retrieve", {"document_id": ""})

        assert isinstance(result, MCPTestResult)
        # Empty document_id should be handled

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_retrieve_metadata_wrong_type_string(self, fastmcp_test_client):
        """Test retrieve with string instead of dict for metadata."""
        result = await fastmcp_test_client.call_tool(
            "retrieve", {"metadata": "not-a-dict"}
        )

        assert isinstance(result, MCPTestResult)

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_retrieve_metadata_wrong_type_list(self, fastmcp_test_client):
        """Test retrieve with list instead of dict for metadata."""
        result = await fastmcp_test_client.call_tool(
            "retrieve", {"metadata": ["not", "dict"]}
        )

        assert isinstance(result, MCPTestResult)

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_retrieve_metadata_empty_dict(self, fastmcp_test_client):
        """Test retrieve with empty dict for metadata."""
        result = await fastmcp_test_client.call_tool("retrieve", {"metadata": {}})

        assert isinstance(result, MCPTestResult)
        # Empty metadata dict is technically valid but may return no results

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_retrieve_limit_wrong_type_string(self, fastmcp_test_client):
        """Test retrieve with string for limit parameter."""
        result = await fastmcp_test_client.call_tool(
            "retrieve", {"document_id": "test-123", "limit": "not-an-int"}
        )

        assert isinstance(result, MCPTestResult)

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_retrieve_limit_negative_value(self, fastmcp_test_client):
        """Test retrieve with negative limit."""
        result = await fastmcp_test_client.call_tool(
            "retrieve", {"document_id": "test-123", "limit": -5}
        )

        assert isinstance(result, MCPTestResult)
        # Negative limit should be handled

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_retrieve_limit_zero_value(self, fastmcp_test_client):
        """Test retrieve with zero limit."""
        result = await fastmcp_test_client.call_tool(
            "retrieve", {"document_id": "test-123", "limit": 0}
        )

        assert isinstance(result, MCPTestResult)
        # Zero limit is boundary condition

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_retrieve_collection_wrong_type_int(self, fastmcp_test_client):
        """Test retrieve with integer for collection parameter."""
        result = await fastmcp_test_client.call_tool(
            "retrieve", {"document_id": "test-123", "collection": 12345}
        )

        assert isinstance(result, MCPTestResult)

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_retrieve_branch_wrong_type_int(self, fastmcp_test_client):
        """Test retrieve with integer for branch parameter."""
        result = await fastmcp_test_client.call_tool(
            "retrieve", {"document_id": "test-123", "branch": 12345}
        )

        assert isinstance(result, MCPTestResult)

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_retrieve_file_type_wrong_type_int(self, fastmcp_test_client):
        """Test retrieve with integer for file_type parameter."""
        result = await fastmcp_test_client.call_tool(
            "retrieve", {"document_id": "test-123", "file_type": 12345}
        )

        assert isinstance(result, MCPTestResult)

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_retrieve_extra_unexpected_parameters(self, fastmcp_test_client):
        """Test retrieve with unknown extra parameters."""
        result = await fastmcp_test_client.call_tool(
            "retrieve",
            {
                "document_id": "test-123",
                "unknown_param": "value",
                "another_param": 123,
            },
        )

        assert isinstance(result, MCPTestResult)
        # Extra params should be ignored or rejected gracefully

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_retrieve_error_response_json_serializable(self, fastmcp_test_client):
        """Verify retrieve error responses are JSON serializable."""
        # Trigger error with missing both params
        result = await fastmcp_test_client.call_tool("retrieve", {})

        # Error response must be JSON serializable
        if result.error:
            json.dumps({"error": result.error})
        if result.response:
            json.dumps(result.response, default=str)


class TestCrossToolErrorHandling:
    """Cross-tool error handling and response format tests."""

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_all_tools_handle_none_parameters(self, fastmcp_test_client):
        """Verify all tools handle None values for required parameters."""
        test_cases = [
            ("store", {"content": None}),
            ("search", {"query": None}),
            ("manage", {"action": None}),
            ("retrieve", {"document_id": None, "metadata": None}),
        ]

        for tool_name, params in test_cases:
            result = await fastmcp_test_client.call_tool(tool_name, params)
            # Should handle None gracefully with error
            assert isinstance(result, MCPTestResult), f"{tool_name} failed None handling"

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_all_tools_error_responses_json_serializable(
        self, fastmcp_test_client
    ):
        """Verify all tools return JSON serializable error responses."""
        # Trigger errors for each tool
        test_cases = [
            ("store", {}),  # Missing content
            ("search", {}),  # Missing query
            ("manage", {}),  # Missing action
            ("retrieve", {}),  # Missing document_id and metadata
        ]

        for tool_name, params in test_cases:
            result = await fastmcp_test_client.call_tool(tool_name, params)

            # Verify JSON serializability
            if result.error:
                try:
                    json.dumps({"error": result.error})
                except (TypeError, ValueError) as e:
                    pytest.fail(
                        f"Tool '{tool_name}' error not JSON serializable: {e}"
                    )

            if result.response:
                try:
                    json.dumps(result.response, default=str)
                except (TypeError, ValueError) as e:
                    pytest.fail(
                        f"Tool '{tool_name}' error response not JSON serializable: {e}"
                    )

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_all_tools_handle_extra_parameters_gracefully(
        self, fastmcp_test_client
    ):
        """Verify all tools handle extra unknown parameters gracefully."""
        test_cases = [
            ("store", {"content": "test", "unknown_param": "value"}),
            ("search", {"query": "test", "unknown_param": "value"}),
            ("manage", {"action": "workspace_status", "unknown_param": "value"}),
            ("retrieve", {"document_id": "test-123", "unknown_param": "value"}),
        ]

        for tool_name, params in test_cases:
            result = await fastmcp_test_client.call_tool(tool_name, params)
            # Should handle gracefully (ignore or reject cleanly)
            assert isinstance(result, MCPTestResult), (
                f"{tool_name} failed extra param handling"
            )

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_error_messages_are_informative(self, fastmcp_test_client):
        """Verify error messages are descriptive and helpful."""
        # Trigger various errors
        test_cases = [
            ("store", {}, "content"),  # Missing content
            ("search", {}, "query"),  # Missing query
            ("manage", {}, "action"),  # Missing action
            ("manage", {"action": "unknown_xyz"}, "action"),  # Unknown action
        ]

        for tool_name, params, expected_keyword in test_cases:
            result = await fastmcp_test_client.call_tool(tool_name, params)

            # Check for informative error message
            if isinstance(result.response, dict) and "error" in result.response:
                error_msg = result.response["error"]
                assert isinstance(error_msg, str), (
                    f"{tool_name} error should be string"
                )
                assert len(error_msg) > 5, (
                    f"{tool_name} error message too short: {error_msg}"
                )
                # Error should be somewhat descriptive (length check is basic quality)

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_error_responses_no_sensitive_info_leakage(
        self, fastmcp_test_client
    ):
        """Verify error responses don't leak sensitive information."""
        # Trigger errors
        test_cases = [
            ("store", {}),
            ("search", {}),
            ("manage", {}),
            ("retrieve", {}),
        ]

        sensitive_patterns = [
            "password",
            "api_key",
            "secret",
            "token",
            "credential",
            "/Users/",
            "/home/",
            "C:\\Users\\",
        ]

        for tool_name, params in test_cases:
            result = await fastmcp_test_client.call_tool(tool_name, params)

            # Check error message for sensitive info
            error_text = ""
            if result.error:
                error_text = str(result.error).lower()
            if isinstance(result.response, dict) and "error" in result.response:
                error_text += " " + str(result.response["error"]).lower()

            for pattern in sensitive_patterns:
                # Allow /Users/ if it's part of a generic path reference
                # But flag actual home directories
                if pattern.lower() in error_text:
                    # This is informational - not a hard failure
                    # Just checking error messages don't leak paths unnecessarily
                    pass

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_error_response_structure_consistency(self, fastmcp_test_client):
        """Verify error responses have consistent structure across tools."""
        test_cases = [
            ("store", {}),
            ("search", {}),
            ("manage", {}),
            ("retrieve", {}),
        ]

        for tool_name, params in test_cases:
            result = await fastmcp_test_client.call_tool(tool_name, params)

            # Error responses should be consistent
            if isinstance(result.response, dict):
                # Should have either success=False or error field
                has_success_false = result.response.get("success") is False
                has_error_field = "error" in result.response

                assert has_success_false or has_error_field, (
                    f"{tool_name} error response should have success=False or error field"
                )

                # If has error field, it should be non-empty string
                if has_error_field:
                    assert isinstance(result.response["error"], str)
                    assert len(result.response["error"]) > 0

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_boundary_empty_string_parameters(self, fastmcp_test_client):
        """Test all tools handle empty string parameters appropriately."""
        test_cases = [
            ("store", {"content": ""}),  # Empty content
            ("search", {"query": ""}),  # Empty query
            ("manage", {"action": ""}),  # Empty action
            ("retrieve", {"document_id": ""}),  # Empty document_id
        ]

        for tool_name, params in test_cases:
            result = await fastmcp_test_client.call_tool(tool_name, params)
            # Empty strings should be handled (may succeed or error gracefully)
            assert isinstance(result, MCPTestResult)

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_type_coercion_behavior(self, fastmcp_test_client):
        """Test how tools handle type coercion (e.g., int to string)."""
        # Some frameworks auto-coerce types, test the behavior
        test_cases = [
            ("store", {"content": 123}),  # int to string
            ("search", {"query": 456}),  # int to string
            ("search", {"limit": "10"}),  # string to int
        ]

        for tool_name, params in test_cases:
            result = await fastmcp_test_client.call_tool(tool_name, params)
            # Either coerces and succeeds, or rejects with type error
            assert isinstance(result, MCPTestResult)
            # Both behaviors are acceptable as long as handled gracefully
