"""Test stdio communication protocol for MCP server.

This module verifies that the workspace-qdrant-mcp server properly implements
stdio-based communication as required by the MCP protocol for Claude Desktop/Code.
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


class StdioMCPClient:
    """Simple MCP client that communicates over stdio for testing."""

    def __init__(self, server_command: list[str]):
        self.server_command = server_command
        self.process: subprocess.Popen | None = None
        self.request_id = 0

    async def start(self) -> None:
        """Start the MCP server process."""
        self.process = subprocess.Popen(
            self.server_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0,
        )

    async def stop(self) -> None:
        """Stop the MCP server process."""
        if self.process:
            self.process.terminate()
            try:
                await asyncio.wait_for(
                    asyncio.create_task(asyncio.to_thread(self.process.wait)),
                    timeout=5.0,
                )
            except asyncio.TimeoutError:
                self.process.kill()
            finally:
                self.process = None

    async def send_request(
        self, method: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Send a JSON-RPC request to the server and return the response."""
        if not self.process or not self.process.stdin or not self.process.stdout:
            raise RuntimeError("Server not started")

        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": params or {},
        }

        # Send request
        request_line = json.dumps(request) + "\n"
        self.process.stdin.write(request_line)
        self.process.stdin.flush()

        # Read response
        response_line = self.process.stdout.readline()
        if not response_line:
            raise RuntimeError("No response from server")

        try:
            response = json.loads(response_line.strip())
            return response
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON response: {response_line}") from e


@pytest.fixture
async def stdio_client():
    """Create a stdio MCP client for testing."""
    # Get the path to the server module
    (Path(__file__).parents[2] / "src" / "workspace_qdrant_mcp" / "server.py")

    client = StdioMCPClient(
        [sys.executable, "-m", "workspace_qdrant_mcp.server", "--transport", "stdio"]
    )

    await client.start()
    yield client
    await client.stop()


@pytest.mark.integration
@pytest.mark.requires_qdrant
async def test_stdio_initialization(stdio_client: StdioMCPClient):
    """Test that the server can initialize over stdio."""
    # Send initialize request
    response = await stdio_client.send_request(
        "initialize",
        {
            "protocolVersion": "0.1.0",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0.0"},
        },
    )

    # Verify response structure
    assert "result" in response
    result = response["result"]
    assert "protocolVersion" in result
    assert "capabilities" in result
    assert "serverInfo" in result

    server_info = result["serverInfo"]
    assert server_info["name"] == "workspace-qdrant-mcp"
    assert "version" in server_info


@pytest.mark.integration
@pytest.mark.requires_qdrant
async def test_stdio_list_tools(stdio_client: StdioMCPClient):
    """Test that tools can be listed over stdio."""
    # Initialize first
    await stdio_client.send_request(
        "initialize",
        {
            "protocolVersion": "0.1.0",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0.0"},
        },
    )

    # List tools
    response = await stdio_client.send_request("tools/list")

    # Verify response
    assert "result" in response
    result = response["result"]
    assert "tools" in result
    assert isinstance(result["tools"], list)

    # Check that we have the expected tools
    tool_names = {tool["name"] for tool in result["tools"]}
    expected_tools = {
        "workspace_status",
        "list_workspace_collections",
        "search_workspace_tool",
        "add_document_tool",
        "get_document_tool",
    }

    # At least some core tools should be present
    assert expected_tools.intersection(tool_names), (
        f"Expected tools not found in {tool_names}"
    )


@pytest.mark.integration
@pytest.mark.requires_qdrant
async def test_stdio_error_handling(stdio_client: StdioMCPClient):
    """Test that errors are properly handled over stdio."""
    # Send invalid request
    response = await stdio_client.send_request("nonexistent/method")

    # Should get an error response
    assert "error" in response
    error = response["error"]
    assert "code" in error
    assert "message" in error


if __name__ == "__main__":
    # Simple manual test
    async def manual_test():
        client = StdioMCPClient(
            [
                sys.executable,
                "-m",
                "workspace_qdrant_mcp.server",
                "--transport",
                "stdio",
            ]
        )

        try:
            await client.start()
            response = await client.send_request(
                "initialize",
                {
                    "protocolVersion": "0.1.0",
                    "capabilities": {},
                    "clientInfo": {"name": "manual-test", "version": "1.0.0"},
                },
            )
            print("Initialize response:", response)

            tools_response = await client.send_request("tools/list")
            print("Tools response:", tools_response)

        finally:
            await client.stop()

    asyncio.run(manual_test())
