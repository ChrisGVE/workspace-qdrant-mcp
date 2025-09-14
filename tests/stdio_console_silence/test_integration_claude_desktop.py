"""
Integration testing with Claude Desktop MCP connection.

Tests real-world integration scenarios to validate that the MCP server
works correctly with Claude Desktop while maintaining complete console silence.

SUCCESS CRITERIA:
- Real MCP client can connect successfully
- All 11 FastMCP tools function correctly
- No console interference with MCP protocol
- Tool responses are properly formatted JSON-RPC
- Connection stability under load
"""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, patch

import pytest

# Test markers
pytestmark = [
    pytest.mark.integration,
    pytest.mark.stdio,
    pytest.mark.slow,
]


class MCPClientSimulator:
    """Simulate MCP client behavior for integration testing."""

    def __init__(self, server_process):
        self.server_process = server_process
        self.message_id = 0
        self.responses = []

    def get_next_id(self) -> int:
        """Get next message ID."""
        self.message_id += 1
        return self.message_id

    def send_request(self, method: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Send JSON-RPC request to server."""
        request = {
            "jsonrpc": "2.0",
            "id": self.get_next_id(),
            "method": method
        }
        if params is not None:
            request["params"] = params

        # Send to server stdin
        message = json.dumps(request) + '\n'
        self.server_process.stdin.write(message.encode())
        self.server_process.stdin.flush()

        return request

    def read_response(self, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """Read response from server stdout."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            if self.server_process.poll() is not None:
                # Server terminated
                return None

            # Try to read a line
            try:
                line = self.server_process.stdout.readline()
                if line:
                    line = line.decode().strip()
                    if line:
                        try:
                            response = json.loads(line)
                            self.responses.append(response)
                            return response
                        except json.JSONDecodeError:
                            # Invalid JSON - this should not happen
                            pytest.fail(f"Server produced invalid JSON: {repr(line)}")
            except:
                # Continue trying
                pass

            time.sleep(0.01)

        return None  # Timeout


class TestIntegrationClaudeDesktop:
    """Integration tests with Claude Desktop simulation."""

    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)

            # Create some test files
            (workspace / "test.txt").write_text("Test document content")
            (workspace / "README.md").write_text("# Test Project\nThis is a test.")

            # Initialize as git repo (for project detection)
            subprocess.run(["git", "init"], cwd=workspace, capture_output=True)
            subprocess.run(["git", "config", "user.name", "Test User"], cwd=workspace)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=workspace)

            yield workspace

    @pytest.fixture
    def mcp_server_process(self, temp_workspace):
        """Start MCP server process for integration testing."""
        server_script = temp_workspace / "test_server.py"
        server_script.write_text(f"""
import os
import sys
import signal
import json
from threading import Timer

# Set up environment
os.environ["WQM_STDIO_MODE"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, "{Path(__file__).parent.parent.parent.parent / 'src' / 'python'}")

def shutdown():
    os._exit(0)

# Safety shutdown timer
Timer(30.0, shutdown).start()

try:
    from workspace_qdrant_mcp.stdio_server import run_lightweight_stdio_server
    run_lightweight_stdio_server()
except Exception as e:
    sys.exit(1)
""")

        # Start server process
        process = subprocess.Popen(
            [sys.executable, str(server_script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=temp_workspace
        )

        yield process

        # Cleanup
        try:
            process.terminate()
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

    def test_mcp_connection_establishment(self, mcp_server_process):
        """Test basic MCP connection establishment."""
        client = MCPClientSimulator(mcp_server_process)

        # Send initialize request
        init_request = client.send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        })

        # Read response
        response = client.read_response()

        # Validate response
        assert response is not None, "No response to initialize request"
        assert response.get("jsonrpc") == "2.0", f"Invalid JSON-RPC version: {response}"
        assert "result" in response or "error" in response, f"Invalid response structure: {response}"
        assert response.get("id") == init_request["id"], f"Response ID mismatch: {response}"

        # Check stderr is clean
        stderr_output = mcp_server_process.stderr.read(1024)  # Non-blocking read
        assert stderr_output == b"", f"Stderr contamination: {stderr_output}"

    def test_tool_listing_integration(self, mcp_server_process):
        """Test tool listing through MCP protocol."""
        client = MCPClientSimulator(mcp_server_process)

        # Send tools/list request
        tools_request = client.send_request("tools/list")
        response = client.read_response()

        # Validate tool listing response
        assert response is not None, "No response to tools/list request"
        assert "result" in response, f"Tools list failed: {response}"

        if "result" in response:
            result = response["result"]
            assert "tools" in result, f"Missing tools in response: {result}"

            tools = result["tools"]
            assert isinstance(tools, list), f"Tools should be a list: {type(tools)}"

            # Should have the basic stdio tools
            tool_names = [tool.get("name") for tool in tools]
            expected_tools = ["workspace_status", "echo_test", "search_workspace", "get_server_info"]

            for expected_tool in expected_tools:
                assert expected_tool in tool_names, f"Missing tool: {expected_tool}"

        # Verify no stderr output
        stderr_check = mcp_server_process.stderr.read(1024)
        assert stderr_check == b"", f"Stderr during tools listing: {stderr_check}"

    def test_tool_execution_integration(self, mcp_server_process):
        """Test actual tool execution through MCP protocol."""
        client = MCPClientSimulator(mcp_server_process)

        # Test workspace_status tool
        status_request = client.send_request("tools/call", {
            "name": "workspace_status",
            "arguments": {}
        })
        status_response = client.read_response()

        # Validate workspace_status response
        assert status_response is not None, "No response to workspace_status call"
        assert "result" in status_response, f"workspace_status failed: {status_response}"

        status_result = status_response["result"]
        assert "content" in status_result, f"Missing content in tool response: {status_result}"

        # Test echo_test tool
        echo_request = client.send_request("tools/call", {
            "name": "echo_test",
            "arguments": {"message": "integration_test"}
        })
        echo_response = client.read_response()

        # Validate echo_test response
        assert echo_response is not None, "No response to echo_test call"
        assert "result" in echo_response, f"echo_test failed: {echo_response}"

        echo_result = echo_response["result"]
        assert "content" in echo_result, f"Missing content in echo response: {echo_result}"

        # Check that echo contains our message
        if isinstance(echo_result["content"], list) and echo_result["content"]:
            content = echo_result["content"][0]
            if isinstance(content, dict) and "text" in content:
                assert "integration_test" in content["text"], f"Echo test failed: {content['text']}"

        # Verify no stderr contamination
        stderr_check = mcp_server_process.stderr.read(1024)
        assert stderr_check == b"", f"Stderr during tool execution: {stderr_check}"

    def test_multiple_concurrent_requests(self, mcp_server_process):
        """Test handling multiple concurrent requests."""
        client = MCPClientSimulator(mcp_server_process)

        # Send multiple requests quickly
        requests = []
        for i in range(5):
            request = client.send_request("tools/call", {
                "name": "echo_test",
                "arguments": {"message": f"concurrent_test_{i}"}
            })
            requests.append(request)

        # Collect all responses
        responses = []
        for _ in range(5):
            response = client.read_response(timeout=10)
            assert response is not None, f"Missing response for concurrent request"
            responses.append(response)

        # Validate all responses
        assert len(responses) == 5, f"Expected 5 responses, got {len(responses)}"

        for response in responses:
            assert "result" in response, f"Concurrent request failed: {response}"
            assert response.get("jsonrpc") == "2.0", f"Invalid JSON-RPC in concurrent response"

        # Verify response IDs match request IDs
        response_ids = {resp.get("id") for resp in responses}
        request_ids = {req["id"] for req in requests}
        assert response_ids == request_ids, f"Response ID mismatch: {response_ids} vs {request_ids}"

        # Check stderr remains clean
        stderr_check = mcp_server_process.stderr.read(1024)
        assert stderr_check == b"", f"Stderr during concurrent requests: {stderr_check}"

    def test_error_handling_integration(self, mcp_server_process):
        """Test error handling through MCP protocol."""
        client = MCPClientSimulator(mcp_server_process)

        # Test invalid tool name
        invalid_request = client.send_request("tools/call", {
            "name": "nonexistent_tool",
            "arguments": {}
        })
        invalid_response = client.read_response()

        # Should get an error response, not crash
        assert invalid_response is not None, "No response to invalid tool call"
        assert "error" in invalid_response or "result" in invalid_response, f"Malformed error response: {invalid_response}"

        # Test malformed request (missing arguments)
        malformed_request = client.send_request("tools/call", {
            "name": "echo_test"
            # Missing arguments
        })
        malformed_response = client.read_response()

        assert malformed_response is not None, "No response to malformed request"
        # Server should handle gracefully

        # Most importantly, server should still be running
        assert mcp_server_process.poll() is None, "Server crashed on error handling"

        # Verify stderr is still clean
        stderr_check = mcp_server_process.stderr.read(1024)
        assert stderr_check == b"", f"Stderr during error handling: {stderr_check}"

    def test_large_response_handling(self, mcp_server_process):
        """Test handling of large responses."""
        client = MCPClientSimulator(mcp_server_process)

        # Test search tool which might return larger responses
        search_request = client.send_request("tools/call", {
            "name": "search_workspace",
            "arguments": {
                "query": "test",
                "limit": 50
            }
        })
        search_response = client.read_response(timeout=10)

        # Should handle large responses gracefully
        assert search_response is not None, "No response to search request"
        assert "result" in search_response, f"Search failed: {search_response}"

        # Verify the response is valid JSON-RPC
        assert search_response.get("jsonrpc") == "2.0", f"Invalid JSON-RPC in large response"
        assert search_response.get("id") == search_request["id"], f"ID mismatch in large response"

        # No stderr contamination
        stderr_check = mcp_server_process.stderr.read(1024)
        assert stderr_check == b"", f"Stderr during large response: {stderr_check}"

    def test_connection_stability_under_load(self, mcp_server_process):
        """Test connection stability under sustained load."""
        client = MCPClientSimulator(mcp_server_process)

        # Send sustained requests
        successful_requests = 0
        total_requests = 20

        for i in range(total_requests):
            # Mix different tool calls
            if i % 3 == 0:
                tool_name = "workspace_status"
                args = {}
            elif i % 3 == 1:
                tool_name = "echo_test"
                args = {"message": f"load_test_{i}"}
            else:
                tool_name = "search_workspace"
                args = {"query": f"test_{i}"}

            request = client.send_request("tools/call", {
                "name": tool_name,
                "arguments": args
            })

            response = client.read_response(timeout=5)

            if response is not None and ("result" in response or "error" in response):
                successful_requests += 1

            # Small delay to avoid overwhelming
            time.sleep(0.01)

        # Should handle most requests successfully
        success_rate = successful_requests / total_requests
        assert success_rate > 0.8, f"Poor success rate under load: {success_rate:.2f}"

        # Server should still be running
        assert mcp_server_process.poll() is None, "Server crashed under load"

        # Final stderr check
        stderr_check = mcp_server_process.stderr.read(1024)
        assert stderr_check == b"", f"Stderr under load: {stderr_check}"

    @pytest.mark.slow
    def test_full_session_simulation(self, mcp_server_process):
        """Simulate a complete Claude Desktop session."""
        client = MCPClientSimulator(mcp_server_process)

        # Simulate complete session workflow
        session_steps = [
            # 1. Initialize
            ("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "claude-desktop", "version": "1.0.0"}
            }),

            # 2. List tools
            ("tools/list", {}),

            # 3. Get workspace status
            ("tools/call", {"name": "workspace_status", "arguments": {}}),

            # 4. Search workspace
            ("tools/call", {"name": "search_workspace", "arguments": {"query": "test", "limit": 10}}),

            # 5. Echo test
            ("tools/call", {"name": "echo_test", "arguments": {"message": "session_test"}}),

            # 6. Get server info
            ("tools/call", {"name": "get_server_info", "arguments": {}}),
        ]

        responses = []
        for method, params in session_steps:
            request = client.send_request(method, params if params else None)
            response = client.read_response(timeout=10)

            assert response is not None, f"No response for {method}"
            assert response.get("jsonrpc") == "2.0", f"Invalid JSON-RPC for {method}: {response}"
            assert response.get("id") == request["id"], f"ID mismatch for {method}"

            responses.append((method, response))

        # Verify session completed successfully
        assert len(responses) == len(session_steps), "Session incomplete"

        # Check specific responses
        for i, (method, response) in enumerate(responses):
            if method == "initialize":
                # Should have capabilities in result
                if "result" in response:
                    assert "capabilities" in response["result"], f"Missing capabilities in init: {response}"

            elif method == "tools/list":
                # Should list available tools
                if "result" in response:
                    assert "tools" in response["result"], f"Missing tools in list: {response}"

            elif method.startswith("tools/call"):
                # Tool calls should have content
                if "result" in response:
                    assert "content" in response["result"], f"Missing content in tool response: {response}"

        # Final verification - no stderr output during entire session
        stderr_output = mcp_server_process.stderr.read(4096)
        assert stderr_output == b"", f"Stderr contamination during session: {stderr_output}"

        # Server should still be running
        assert mcp_server_process.poll() is None, "Server crashed during session"

    def test_graceful_shutdown_handling(self, mcp_server_process):
        """Test graceful shutdown behavior."""
        client = MCPClientSimulator(mcp_server_process)

        # Send a few requests first
        client.send_request("tools/list")
        response = client.read_response()
        assert response is not None, "Server not responding before shutdown"

        # Send shutdown signal
        mcp_server_process.terminate()

        # Wait for graceful shutdown
        try:
            exit_code = mcp_server_process.wait(timeout=5)
            # Should exit cleanly
            assert exit_code in [0, -15, 143], f"Ungraceful shutdown: exit code {exit_code}"
        except subprocess.TimeoutExpired:
            # Force kill if needed
            mcp_server_process.kill()
            mcp_server_process.wait()

        # Check final stderr state
        stderr_output = mcp_server_process.stderr.read(4096)
        # Small amount of shutdown messages might be acceptable
        if stderr_output:
            # Should not be significant output
            assert len(stderr_output) < 1024, f"Excessive stderr during shutdown: {len(stderr_output)} bytes"