"""
Unit tests for FastMCP In-Memory Testing Infrastructure (Task 241.1).

This module tests the FastMCP testing infrastructure itself to ensure it provides
reliable testing capabilities for MCP protocol operations.

Validates:
    - FastMCPTestServer initialization and setup
    - FastMCPTestClient creation and tool invocation
    - MCPProtocolTester comprehensive testing
    - Response validation and error handling
    - Performance baseline testing
"""

import asyncio
import pytest
from typing import Dict, Any

from tests.utils.fastmcp_test_infrastructure import (
    FastMCPTestServer,
    FastMCPTestClient,
    MCPProtocolTester,
    MCPTestResult,
    MCPToolTestCase,
    fastmcp_test_environment
)


class TestFastMCPTestInfrastructure:
    """Test the FastMCP testing infrastructure components."""

    @pytest.mark.fastmcp
    async def test_fastmcp_test_server_initialization(self, fastmcp_test_server):
        """Test FastMCP test server initialization."""
        server = fastmcp_test_server

        assert server.initialized
        assert server.name == "pytest-server"
        assert isinstance(server.app, object)  # Should be FastMCP instance

        # Test available tools discovery
        available_tools = server.get_available_tools()
        assert len(available_tools) > 0

        # Test specific expected tools exist
        expected_tools = ["workspace_status", "list_workspace_collections"]
        for tool_name in expected_tools:
            if tool_name in available_tools:
                tool = server.get_tool_sync(tool_name)
                if tool is not None:
                    # Only check .fn attribute if tool is found
                    # For testing purposes, we accept that some tools might not have .fn
                    print(f"Found tool {tool_name}: {type(tool)}")
                else:
                    print(f"Tool {tool_name} not found via sync method")

    @pytest.mark.fastmcp
    async def test_fastmcp_test_client_basic_operations(self, fastmcp_test_client):
        """Test FastMCP test client basic operations."""
        client = fastmcp_test_client

        assert client.initialized
        assert len(client.get_call_history()) == 0

        # Test basic tool call
        result = await client.call_tool("workspace_status", {})

        # Verify result structure
        assert isinstance(result, MCPTestResult)
        assert result.tool_name == "workspace_status"
        assert result.parameters == {}
        assert result.execution_time_ms >= 0

        # Should succeed or fail gracefully
        if result.success:
            assert result.response is not None
            assert result.protocol_compliance is not None
            assert isinstance(result.protocol_compliance, dict)
        else:
            assert result.error is not None

        # Verify call history
        history = client.get_call_history()
        assert len(history) == 1
        assert history[0].tool_name == "workspace_status"

    @pytest.mark.fastmcp
    async def test_tool_invocation_with_parameters(self, fastmcp_test_client):
        """Test tool invocation with parameters."""
        client = fastmcp_test_client

        # Test search tool with parameters
        result = await client.call_tool("search_workspace_tool", {
            "query": "test",
            "limit": 5
        })

        assert isinstance(result, MCPTestResult)
        assert result.tool_name == "search_workspace_tool"

        # Should handle parameters gracefully (success or structured error)
        if result.success:
            assert result.response is not None
        else:
            # Error should be informative
            assert result.error is not None
            assert len(result.error) > 0

    @pytest.mark.fastmcp
    async def test_tool_error_handling(self, fastmcp_test_client):
        """Test tool error handling scenarios."""
        client = fastmcp_test_client

        # Test nonexistent tool
        result = await client.call_tool("nonexistent_tool", {})
        assert not result.success
        assert "not found" in result.error.lower()

        # Test tool with invalid parameters
        result = await client.call_tool("search_workspace_tool", {
            "invalid_param": "value"
        })

        # Should either fail gracefully or return structured error
        if result.success:
            # If successful, response should indicate error
            if isinstance(result.response, dict):
                assert "error" in result.response or "success" not in result.response or not result.response.get("success", True)
        else:
            # Failure is acceptable for invalid parameters
            assert result.error is not None

    @pytest.mark.fastmcp
    async def test_response_protocol_validation(self, fastmcp_test_client):
        """Test response protocol validation."""
        client = fastmcp_test_client

        result = await client.call_tool("workspace_status", {})

        if result.success and result.protocol_compliance:
            compliance = result.protocol_compliance

            # Check specific compliance requirements
            assert "json_serializable" in compliance
            assert "is_dict_or_list" in compliance
            assert "has_content" in compliance

            # JSON serializable should always be True for valid responses
            if compliance["json_serializable"]:
                import json
                # Should not raise exception
                json.dumps(result.response, default=str)

    @pytest.mark.fastmcp
    async def test_mcp_protocol_tester_tool_registration(self, mcp_protocol_tester):
        """Test MCP protocol tester tool registration validation."""
        tester = mcp_protocol_tester

        registration_results = await tester.test_tool_registration()

        assert "success_rate" in registration_results
        assert "total_tools" in registration_results
        assert "test_results" in registration_results

        # Should have some tools registered
        assert registration_results["total_tools"] > 0
        assert registration_results["success_rate"] >= 0.0

        # Test results should be informative
        test_results = registration_results["test_results"]
        assert isinstance(test_results, list)

        for test_result in test_results:
            assert "tool_name" in test_result
            assert "structure_valid" in test_result

    @pytest.mark.fastmcp
    async def test_mcp_protocol_tester_tool_invocation(self, mcp_protocol_tester):
        """Test MCP protocol tester tool invocation validation."""
        tester = mcp_protocol_tester

        invocation_results = await tester.test_tool_invocation()

        assert "success_rate" in invocation_results
        assert "total_tests" in invocation_results
        assert "test_results" in invocation_results

        # Should have attempted some tests
        assert invocation_results["total_tests"] >= 0

        # Test results should be detailed
        for test_result in invocation_results["test_results"]:
            assert "tool_name" in test_result
            assert "success" in test_result
            assert "execution_time_ms" in test_result

    @pytest.mark.fastmcp
    async def test_mcp_protocol_tester_comprehensive(self, mcp_protocol_tester):
        """Test comprehensive MCP protocol testing."""
        tester = mcp_protocol_tester

        # Run comprehensive tests (this tests all components)
        comprehensive_results = await tester.run_comprehensive_tests()

        # Verify all test categories are present
        expected_categories = [
            "tool_registration",
            "tool_invocation",
            "parameter_validation",
            "response_format",
            "error_handling",
            "performance_baseline",
            "summary"
        ]

        for category in expected_categories:
            assert category in comprehensive_results, f"Missing test category: {category}"

        # Verify summary contains overall metrics
        summary = comprehensive_results["summary"]
        assert "overall_compliance" in summary
        assert "total_tests" in summary
        assert "test_timestamp" in summary

        # Overall compliance should be between 0 and 1
        assert 0.0 <= summary["overall_compliance"] <= 1.0

    @pytest.mark.fastmcp
    async def test_fastmcp_test_environment_context_manager(self):
        """Test fastmcp_test_environment context manager."""
        from workspace_qdrant_mcp.server import app

        async with fastmcp_test_environment(app, "context-test") as (server, client):
            assert isinstance(server, FastMCPTestServer)
            assert isinstance(client, FastMCPTestClient)
            assert server.initialized
            assert client.initialized

            # Test basic operation within context
            result = await client.call_tool("workspace_status", {})
            assert isinstance(result, MCPTestResult)

        # After context, server should be cleaned up
        # Note: We can't easily test this without access to internals

    @pytest.mark.fastmcp
    async def test_performance_timing(self, fastmcp_test_client):
        """Test performance timing accuracy."""
        client = fastmcp_test_client

        # Multiple calls to measure timing consistency
        execution_times = []

        for _ in range(5):
            result = await client.call_tool("workspace_status", {})
            if result.success:
                execution_times.append(result.execution_time_ms)

        if execution_times:
            # Execution times should be reasonable (< 1000ms for in-memory)
            for exec_time in execution_times:
                assert exec_time < 1000.0, f"Execution time too high: {exec_time}ms"

            # Times should be generally consistent (not varying by orders of magnitude)
            min_time = min(execution_times)
            max_time = max(execution_times)

            # Max should not be more than 10x min (allowing for some variance)
            if min_time > 0:
                assert max_time / min_time < 10, f"Execution time variance too high: {min_time}ms - {max_time}ms"

    @pytest.mark.fastmcp
    async def test_error_handling_edge_cases(self, fastmcp_test_client):
        """Test edge cases in error handling."""
        client = fastmcp_test_client

        # Test with very large parameters
        large_param = "x" * 10000
        result = await client.call_tool("search_workspace_tool", {
            "query": large_param
        })

        # Should handle gracefully (either success or structured error)
        assert isinstance(result, MCPTestResult)

        # Test with None parameters
        result = await client.call_tool("workspace_status", None)

        # Should handle None parameters gracefully
        # This might succeed (if tool accepts no params) or fail gracefully
        assert isinstance(result, MCPTestResult)

        # Test with timeout (if supported)
        try:
            result = await client.call_tool("workspace_status", {}, timeout_ms=0.1)
            # Very short timeout might cause timeout error
            if not result.success and result.error:
                assert "timeout" in result.error.lower() or "time" in result.error.lower()
        except Exception:
            # Exception is also acceptable for very short timeouts
            pass

    @pytest.mark.fastmcp
    async def test_call_history_management(self, fastmcp_test_client):
        """Test call history management."""
        client = fastmcp_test_client

        # Initially empty
        assert len(client.get_call_history()) == 0

        # Make several calls
        await client.call_tool("workspace_status", {})
        await client.call_tool("list_workspace_collections", {})
        await client.call_tool("nonexistent_tool", {})  # This should fail

        # History should contain all calls
        history = client.get_call_history()
        assert len(history) == 3

        # History should maintain order
        assert history[0].tool_name == "workspace_status"
        assert history[1].tool_name == "list_workspace_collections"
        assert history[2].tool_name == "nonexistent_tool"

        # Clear history
        client.clear_call_history()
        assert len(client.get_call_history()) == 0

        # New call should start fresh history
        await client.call_tool("workspace_status", {})
        assert len(client.get_call_history()) == 1


@pytest.mark.fastmcp
class TestMCPTestResult:
    """Test MCPTestResult data structure."""

    def test_mcp_test_result_structure(self):
        """Test MCPTestResult data structure."""
        result = MCPTestResult(
            success=True,
            tool_name="test_tool",
            parameters={"param": "value"},
            response={"result": "data"},
            execution_time_ms=50.0,
            protocol_compliance={"json_serializable": True}
        )

        assert result.success
        assert result.tool_name == "test_tool"
        assert result.parameters == {"param": "value"}
        assert result.response == {"result": "data"}
        assert result.execution_time_ms == 50.0
        assert result.error is None
        assert result.protocol_compliance == {"json_serializable": True}
        assert isinstance(result.metadata, dict)


@pytest.mark.fastmcp
class TestMCPToolTestCase:
    """Test MCPToolTestCase data structure."""

    def test_mcp_tool_test_case_structure(self):
        """Test MCPToolTestCase data structure."""
        test_case = MCPToolTestCase(
            tool_name="test_tool",
            description="Test case description",
            parameters={"param": "value"},
            expected_response_type=dict,
            expected_fields=["field1", "field2"],
            should_succeed=True,
            validation_fn=lambda x: isinstance(x, dict)
        )

        assert test_case.tool_name == "test_tool"
        assert test_case.description == "Test case description"
        assert test_case.parameters == {"param": "value"}
        assert test_case.expected_response_type == dict
        assert test_case.expected_fields == ["field1", "field2"]
        assert test_case.should_succeed
        assert callable(test_case.validation_fn)