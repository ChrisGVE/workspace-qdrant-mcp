"""
FastMCP In-Memory Testing Infrastructure for Task 241.1.

This module provides comprehensive in-memory testing utilities for FastMCP server-client
communication without network latency. Designed to support testing of all 11 MCP tools
with direct client-server connections for protocol compliance validation.

Key Features:
    - In-memory FastMCP server instances for zero-latency testing
    - Direct client-server connections bypassing network stack
    - Comprehensive MCP protocol testing utilities
    - Tool invocation testing with parameter validation
    - Response format compliance verification
    - Error handling testing framework
    - Performance benchmarking for MCP operations

Architecture:
    - FastMCPTestServer: In-memory server instances
    - FastMCPTestClient: Direct client connections
    - MCPProtocolTester: Protocol compliance validation
    - ToolInvocationTester: Individual tool testing
    - ResponseValidator: MCP response format validation

Example:
    ```python
    async with FastMCPTestServer(app) as server:
        client = await server.create_test_client()
        result = await client.call_tool("workspace_status", {})
        assert result["success"] == True
    ```
"""

import asyncio
import json
import logging
import traceback
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Union
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastmcp import FastMCP
from fastmcp.tools import FunctionTool

# Suppress logging during tests for clean output
logging.getLogger("fastmcp").setLevel(logging.WARNING)
logging.getLogger("workspace_qdrant_mcp").setLevel(logging.WARNING)


@dataclass
class MCPTestResult:
    """Result of an MCP protocol test operation."""
    success: bool
    tool_name: str
    parameters: dict[str, Any]
    response: Any
    execution_time_ms: float
    error: str | None = None
    protocol_compliance: dict[str, bool] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPToolTestCase:
    """Test case definition for MCP tool testing."""
    tool_name: str
    description: str
    parameters: dict[str, Any]
    expected_response_type: type
    expected_fields: list[str] = field(default_factory=list)
    should_succeed: bool = True
    validation_fn: Callable[[Any], bool] | None = None


class FastMCPTestServer:
    """
    In-memory FastMCP server for testing without network overhead.

    Provides direct access to FastMCP app instances and enables zero-latency
    testing of MCP protocol operations.
    """

    def __init__(self, app: FastMCP, name: str = "test-server"):
        """
        Initialize FastMCP test server.

        Args:
            app: FastMCP application instance to test
            name: Identifier for this test server instance
        """
        self.app = app
        self.name = name
        self.initialized = False
        self._clients: list[FastMCPTestClient] = []
        self._setup_patches: list[Any] = []

    async def __aenter__(self) -> 'FastMCPTestServer':
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()

    async def initialize(self) -> None:
        """Initialize the test server environment."""
        if self.initialized:
            return

        # Set up test environment isolation
        self._setup_test_environment()

        # Verify FastMCP app structure
        await self._verify_app_structure()

        self.initialized = True

    async def cleanup(self) -> None:
        """Clean up test server resources."""
        # Close all test clients
        for client in self._clients[:]:
            await client.close()

        # Remove patches
        for patch_obj in self._setup_patches:
            if hasattr(patch_obj, 'stop'):
                patch_obj.stop()

        self._setup_patches.clear()
        self.initialized = False

    def _setup_test_environment(self) -> None:
        """Set up isolated test environment."""
        # Mock qdrant_client to prevent initialization issues
        mock_qdrant_client = Mock()
        mock_qdrant_client.get_collections.return_value.collections = [
            Mock(name="_test_project_id"),
            Mock(name="memory"),
        ]

        # Apply qdrant_client patch
        qdrant_patch = patch(
            "workspace_qdrant_mcp.server.qdrant_client",
            mock_qdrant_client
        )
        qdrant_patch.start()
        self._setup_patches.append(qdrant_patch)

        # Mock daemon_client to prevent initialization issues
        mock_daemon_client = AsyncMock()
        mock_daemon_client.ping.return_value = {"status": "ok"}

        # Apply daemon_client patch
        daemon_patch = patch(
            "workspace_qdrant_mcp.server.daemon_client",
            mock_daemon_client
        )
        daemon_patch.start()
        self._setup_patches.append(daemon_patch)

    async def _verify_app_structure(self) -> None:
        """Verify FastMCP app has expected structure."""
        if not isinstance(self.app, FastMCP):
            raise ValueError(f"Expected FastMCP instance, got {type(self.app)}")

        # Check for tools registry using FastMCP API
        if not hasattr(self.app, 'get_tools'):
            raise ValueError("FastMCP app missing get_tools method")

        # Count registered tools using FastMCP API
        try:
            tools = await self.app.get_tools()
            tool_count = len(tools)
        except Exception:
            # If get_tools fails, try alternate approach
            tool_count = 0
            if hasattr(self.app, '_tool_manager') and hasattr(self.app._tool_manager, '_tools'):
                tool_count = len(self.app._tool_manager._tools)

        if tool_count < 1:  # Expect at least 1 tool
            raise ValueError(f"FastMCP app has insufficient tools: {tool_count}")

    async def create_test_client(self) -> 'FastMCPTestClient':
        """Create a new test client connected to this server."""
        client = FastMCPTestClient(self)
        await client.initialize()
        self._clients.append(client)
        return client

    def get_available_tools(self) -> list[str]:
        """Get list of available tool names."""
        try:
            # Use async method in sync context - this is for testing only
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create a new task if loop is already running
                asyncio.create_task(self.app.get_tools())
                # Note: This is a simplified approach for testing
                # In real async context, this would be awaited properly
                if hasattr(self.app, '_tool_manager') and hasattr(self.app._tool_manager, '_tools'):
                    return list(self.app._tool_manager._tools.keys())
            else:
                tools = loop.run_until_complete(self.app.get_tools())
                return [tool.name for tool in tools]
        except Exception:
            # Fallback to internal structure
            if hasattr(self.app, '_tool_manager') and hasattr(self.app._tool_manager, '_tools'):
                return list(self.app._tool_manager._tools.keys())
        return []

    async def get_tool(self, tool_name: str) -> FunctionTool | None:
        """Get a specific tool by name."""
        try:
            # Try FastMCP API first (it's async)
            return await self.app.get_tool(tool_name)
        except Exception:
            # Fallback to attribute access
            return getattr(self.app, tool_name, None)

    def get_tool_sync(self, tool_name: str) -> FunctionTool | None:
        """Get a specific tool by name (synchronous version for testing)."""
        try:
            # Try to use internal tool manager directly
            if hasattr(self.app, '_tool_manager') and hasattr(self.app._tool_manager, '_tools'):
                tool_data = self.app._tool_manager._tools.get(tool_name)
                if tool_data:
                    return tool_data
            # Fallback to attribute access
            return getattr(self.app, tool_name, None)
        except Exception:
            return None


class FastMCPTestClient:
    """
    In-memory FastMCP client for direct server communication.

    Provides direct access to FastMCP tools without network serialization,
    enabling high-performance testing of MCP protocol operations.
    """

    def __init__(self, server: FastMCPTestServer):
        """
        Initialize FastMCP test client.

        Args:
            server: FastMCP test server to connect to
        """
        self.server = server
        self.initialized = False
        self._call_history: list[MCPTestResult] = []

    async def initialize(self) -> None:
        """Initialize the test client."""
        if not self.server.initialized:
            raise ValueError("Server must be initialized before client")
        self.initialized = True

    async def close(self) -> None:
        """Close the test client connection."""
        if self in self.server._clients:
            self.server._clients.remove(self)
        self.initialized = False

    async def call_tool(
        self,
        tool_name: str,
        parameters: dict[str, Any],
        timeout_ms: float = 5000.0
    ) -> MCPTestResult:
        """
        Call a tool on the connected server.

        Args:
            tool_name: Name of the tool to call
            parameters: Parameters to pass to the tool
            timeout_ms: Timeout in milliseconds for the call

        Returns:
            MCPTestResult with call results and metadata
        """
        if not self.initialized:
            raise ValueError("Client not initialized")

        start_time = asyncio.get_event_loop().time()

        try:
            # Use FastMCP _call_tool method for direct invocation
            try:
                response = await asyncio.wait_for(
                    self.server.app._call_tool(tool_name, parameters),
                    timeout=timeout_ms / 1000.0
                )
            except asyncio.TimeoutError:
                execution_time = (asyncio.get_event_loop().time() - start_time) * 1000
                return MCPTestResult(
                    success=False,
                    tool_name=tool_name,
                    parameters=parameters,
                    response=None,
                    execution_time_ms=execution_time,
                    error=f"Tool call timed out after {timeout_ms}ms"
                )
            except Exception as tool_error:
                # Check if it's a tool not found error
                if "not found" in str(tool_error).lower() or "unknown" in str(tool_error).lower():
                    return MCPTestResult(
                        success=False,
                        tool_name=tool_name,
                        parameters=parameters,
                        response=None,
                        execution_time_ms=0.0,
                        error=f"Tool '{tool_name}' not found"
                    )
                # Re-raise other errors to be handled by outer try-catch
                raise tool_error

            execution_time = (asyncio.get_event_loop().time() - start_time) * 1000

            # Validate response format
            protocol_compliance = self._validate_response_protocol(response)

            result = MCPTestResult(
                success=True,
                tool_name=tool_name,
                parameters=parameters,
                response=response,
                execution_time_ms=execution_time,
                protocol_compliance=protocol_compliance,
                metadata={
                    "response_type": type(response).__name__,
                    "response_size": len(str(response)) if response else 0,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )

        except Exception as e:
            execution_time = (asyncio.get_event_loop().time() - start_time) * 1000
            result = MCPTestResult(
                success=False,
                tool_name=tool_name,
                parameters=parameters,
                response=None,
                execution_time_ms=execution_time,
                error=f"Tool execution error: {str(e)}",
                metadata={
                    "exception_type": type(e).__name__,
                    "traceback": traceback.format_exc(),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )

        # Store call history
        self._call_history.append(result)
        return result

    def _validate_response_protocol(self, response: Any) -> dict[str, bool]:
        """
        Validate response against MCP protocol requirements.

        Args:
            response: Response to validate

        Returns:
            Dictionary of compliance checks
        """
        compliance = {
            "json_serializable": True,
            "is_dict_or_list": isinstance(response, (dict, list)),
            "has_content": response is not None,
            "error_format_valid": True
        }

        # Test JSON serialization
        try:
            json.dumps(response, default=str)
        except Exception:
            compliance["json_serializable"] = False

        # Check error response format
        if isinstance(response, dict) and "error" in response:
            compliance["error_format_valid"] = isinstance(response["error"], str)

        return compliance

    def get_call_history(self) -> list[MCPTestResult]:
        """Get history of all tool calls made by this client."""
        return self._call_history.copy()

    def clear_call_history(self) -> None:
        """Clear the call history."""
        self._call_history.clear()


class MCPProtocolTester:
    """
    Comprehensive MCP protocol compliance tester.

    Provides systematic testing of MCP protocol requirements including
    tool registration, parameter validation, response formats, and error handling.
    """

    def __init__(self, server: FastMCPTestServer):
        """
        Initialize MCP protocol tester.

        Args:
            server: FastMCP test server to test against
        """
        self.server = server
        self.results: list[MCPTestResult] = []

    async def run_comprehensive_tests(self) -> dict[str, Any]:
        """
        Run comprehensive MCP protocol compliance tests.

        Returns:
            Dictionary containing test results and compliance metrics
        """
        test_results = {
            "tool_registration": await self.test_tool_registration(),
            "tool_invocation": await self.test_tool_invocation(),
            "parameter_validation": await self.test_parameter_validation(),
            "response_format": await self.test_response_format(),
            "error_handling": await self.test_error_handling(),
            "performance_baseline": await self.test_performance_baseline()
        }

        # Calculate overall compliance score
        compliance_scores = []
        for _category, results in test_results.items():
            if isinstance(results, dict) and "success_rate" in results:
                compliance_scores.append(results["success_rate"])

        overall_compliance = sum(compliance_scores) / len(compliance_scores) if compliance_scores else 0

        test_results["summary"] = {
            "overall_compliance": overall_compliance,
            "total_tests": sum(len(r.get("test_results", [])) for r in test_results.values() if isinstance(r, dict)),
            "test_timestamp": datetime.now(timezone.utc).isoformat()
        }

        return test_results

    async def test_tool_registration(self) -> dict[str, Any]:
        """Test tool registration and discovery."""
        available_tools = self.server.get_available_tools()

        # Expected core tools (minimum set)
        expected_tools = [
            "workspace_status",
            "list_workspace_collections",
            "search_workspace_tool",
            "add_document_tool",
            "get_document_tool"
        ]

        registered_tools = [tool for tool in expected_tools if tool in available_tools]
        registration_rate = len(registered_tools) / len(expected_tools)

        # Test tool structure
        tool_structure_tests = []
        for tool_name in registered_tools:
            tool = self.server.get_tool_sync(tool_name)
            structure_valid = (
                tool is not None and
                hasattr(tool, 'fn') and
                callable(tool.fn)
            )
            tool_structure_tests.append({
                "tool_name": tool_name,
                "structure_valid": structure_valid
            })

        structure_success_rate = sum(1 for t in tool_structure_tests if t["structure_valid"]) / len(tool_structure_tests) if tool_structure_tests else 0

        return {
            "success_rate": (registration_rate + structure_success_rate) / 2,
            "total_tools": len(available_tools),
            "expected_tools": len(expected_tools),
            "registered_tools": len(registered_tools),
            "registration_rate": registration_rate,
            "structure_success_rate": structure_success_rate,
            "test_results": tool_structure_tests
        }

    async def test_tool_invocation(self) -> dict[str, Any]:
        """Test basic tool invocation without parameters."""
        client = await self.server.create_test_client()

        # Test tools that should work without parameters
        no_param_tools = ["workspace_status", "list_workspace_collections"]

        test_results = []
        for tool_name in no_param_tools:
            if tool_name in self.server.get_available_tools():
                result = await client.call_tool(tool_name, {})
                test_results.append(result)

        await client.close()

        success_rate = sum(1 for r in test_results if r.success) / len(test_results) if test_results else 0
        avg_execution_time = sum(r.execution_time_ms for r in test_results) / len(test_results) if test_results else 0

        return {
            "success_rate": success_rate,
            "total_tests": len(test_results),
            "successful_tests": sum(1 for r in test_results if r.success),
            "average_execution_time_ms": avg_execution_time,
            "test_results": [
                {
                    "tool_name": r.tool_name,
                    "success": r.success,
                    "execution_time_ms": r.execution_time_ms,
                    "error": r.error
                } for r in test_results
            ]
        }

    async def test_parameter_validation(self) -> dict[str, Any]:
        """Test parameter validation for tools."""
        client = await self.server.create_test_client()

        # Test cases with valid and invalid parameters
        test_cases = [
            MCPToolTestCase(
                tool_name="search_workspace_tool",
                description="Valid search parameters",
                parameters={"query": "test", "limit": 5},
                expected_response_type=dict,
                should_succeed=True
            ),
            MCPToolTestCase(
                tool_name="search_workspace_tool",
                description="Invalid search parameters",
                parameters={"query": "", "limit": -1},
                expected_response_type=dict,
                should_succeed=False
            ),
            MCPToolTestCase(
                tool_name="add_document_tool",
                description="Valid document addition",
                parameters={"content": "test content", "collection": "test"},
                expected_response_type=dict,
                should_succeed=True
            )
        ]

        test_results = []
        for test_case in test_cases:
            if test_case.tool_name in self.server.get_available_tools():
                result = await client.call_tool(test_case.tool_name, test_case.parameters)

                # Evaluate if result matches expectation
                validation_passed = (
                    (result.success and test_case.should_succeed) or
                    (not result.success and not test_case.should_succeed)
                )

                test_results.append({
                    "test_case": test_case.description,
                    "tool_name": test_case.tool_name,
                    "parameters": test_case.parameters,
                    "should_succeed": test_case.should_succeed,
                    "actual_success": result.success,
                    "validation_passed": validation_passed,
                    "execution_time_ms": result.execution_time_ms,
                    "error": result.error
                })

        await client.close()

        success_rate = sum(1 for r in test_results if r["validation_passed"]) / len(test_results) if test_results else 0

        return {
            "success_rate": success_rate,
            "total_tests": len(test_results),
            "validation_passed": sum(1 for r in test_results if r["validation_passed"]),
            "test_results": test_results
        }

    async def test_response_format(self) -> dict[str, Any]:
        """Test MCP response format compliance."""
        client = await self.server.create_test_client()

        # Test various tools for response format
        tools_to_test = [
            ("workspace_status", {}),
            ("list_workspace_collections", {}),
            ("search_workspace_tool", {"query": "test", "limit": 3})
        ]

        test_results = []
        for tool_name, params in tools_to_test:
            if tool_name in self.server.get_available_tools():
                result = await client.call_tool(tool_name, params)

                if result.success and result.protocol_compliance:
                    compliance_score = sum(result.protocol_compliance.values()) / len(result.protocol_compliance)
                else:
                    compliance_score = 0.0

                test_results.append({
                    "tool_name": tool_name,
                    "success": result.success,
                    "protocol_compliance": result.protocol_compliance,
                    "compliance_score": compliance_score,
                    "response_type": result.metadata.get("response_type"),
                    "response_size": result.metadata.get("response_size", 0)
                })

        await client.close()

        success_rate = sum(r["compliance_score"] for r in test_results) / len(test_results) if test_results else 0

        return {
            "success_rate": success_rate,
            "total_tests": len(test_results),
            "average_compliance_score": success_rate,
            "test_results": test_results
        }

    async def test_error_handling(self) -> dict[str, Any]:
        """Test error handling compliance."""
        client = await self.server.create_test_client()

        # Test error scenarios
        error_test_cases = [
            ("nonexistent_tool", {}, "Tool should not exist"),
            ("search_workspace_tool", {"invalid": "params"}, "Invalid parameters"),
            ("add_document_tool", {}, "Missing required parameters")
        ]

        test_results = []
        for tool_name, params, description in error_test_cases:
            result = await client.call_tool(tool_name, params)

            # Error handling is good if:
            # 1. Tool fails gracefully (not success) OR
            # 2. Tool returns structured error response
            error_handled_gracefully = (
                not result.success or
                (result.success and isinstance(result.response, dict) and "error" in result.response)
            )

            test_results.append({
                "description": description,
                "tool_name": tool_name,
                "parameters": params,
                "error_handled_gracefully": error_handled_gracefully,
                "execution_time_ms": result.execution_time_ms,
                "error_message": result.error or result.response.get("error", "") if isinstance(result.response, dict) else ""
            })

        await client.close()

        success_rate = sum(1 for r in test_results if r["error_handled_gracefully"]) / len(test_results) if test_results else 0

        return {
            "success_rate": success_rate,
            "total_tests": len(test_results),
            "graceful_errors": sum(1 for r in test_results if r["error_handled_gracefully"]),
            "test_results": test_results
        }

    async def test_performance_baseline(self) -> dict[str, Any]:
        """Test performance baseline for MCP operations."""
        client = await self.server.create_test_client()

        # Performance test: multiple calls to fast operations
        performance_tests = []
        test_iterations = 10

        for tool_name in ["workspace_status", "list_workspace_collections"]:
            if tool_name in self.server.get_available_tools():
                execution_times = []

                for _ in range(test_iterations):
                    result = await client.call_tool(tool_name, {})
                    if result.success:
                        execution_times.append(result.execution_time_ms)

                if execution_times:
                    avg_time = sum(execution_times) / len(execution_times)
                    min_time = min(execution_times)
                    max_time = max(execution_times)

                    performance_tests.append({
                        "tool_name": tool_name,
                        "iterations": len(execution_times),
                        "average_time_ms": avg_time,
                        "min_time_ms": min_time,
                        "max_time_ms": max_time,
                        "performance_rating": "excellent" if avg_time < 10 else "good" if avg_time < 50 else "acceptable"
                    })

        await client.close()

        overall_performance = "excellent"
        if performance_tests:
            avg_performance_time = sum(p["average_time_ms"] for p in performance_tests) / len(performance_tests)
            if avg_performance_time > 100:
                overall_performance = "poor"
            elif avg_performance_time > 50:
                overall_performance = "acceptable"
            elif avg_performance_time > 10:
                overall_performance = "good"

        return {
            "success_rate": 1.0 if performance_tests else 0.0,
            "total_tests": len(performance_tests),
            "overall_performance": overall_performance,
            "test_results": performance_tests
        }


@asynccontextmanager
async def fastmcp_test_environment(app: FastMCP, name: str = "test-env") -> AsyncGenerator[tuple[FastMCPTestServer, FastMCPTestClient], None]:
    """
    Async context manager for FastMCP testing environment.

    Provides a complete testing environment with server and client ready for use.

    Args:
        app: FastMCP application instance
        name: Environment name for identification

    Yields:
        Tuple of (server, client) ready for testing

    Example:
        ```python
        async with fastmcp_test_environment(app) as (server, client):
            result = await client.call_tool("workspace_status", {})
            assert result.success
        ```
    """
    async with FastMCPTestServer(app, name) as server:
        client = await server.create_test_client()
        try:
            yield server, client
        finally:
            await client.close()
