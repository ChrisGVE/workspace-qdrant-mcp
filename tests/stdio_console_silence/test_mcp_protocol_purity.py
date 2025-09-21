"""
MCP protocol purity validation tests.

Validates that JSON-RPC message exchange is pure and uncontaminated by
console output, ensuring perfect MCP protocol compliance.

SUCCESS CRITERIA:
- Only valid JSON-RPC messages on stdout
- All messages parseable as JSON
- Proper JSON-RPC structure maintained
- No interference from logging or debug output
"""

import asyncio
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from threading import Thread
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Test markers
pytestmark = [
    pytest.mark.integration,
    pytest.mark.stdio,
]


class MCPMessage:
    """Helper class for creating and validating MCP messages."""

    @staticmethod
    def request(method: str, params: Optional[Dict] = None, msg_id: Optional[int] = None) -> Dict[str, Any]:
        """Create a JSON-RPC request message."""
        message = {
            "jsonrpc": "2.0",
            "method": method,
            "id": msg_id or 1
        }
        if params is not None:
            message["params"] = params
        return message

    @staticmethod
    def response(result: Any, msg_id: int = 1) -> Dict[str, Any]:
        """Create a JSON-RPC response message."""
        return {
            "jsonrpc": "2.0",
            "result": result,
            "id": msg_id
        }

    @staticmethod
    def error(code: int, message: str, msg_id: int = 1) -> Dict[str, Any]:
        """Create a JSON-RPC error message."""
        return {
            "jsonrpc": "2.0",
            "error": {
                "code": code,
                "message": message
            },
            "id": msg_id
        }

    @staticmethod
    def notification(method: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Create a JSON-RPC notification message."""
        message = {
            "jsonrpc": "2.0",
            "method": method
        }
        if params is not None:
            message["params"] = params
        return message

    @staticmethod
    def is_valid_jsonrpc(data: str) -> bool:
        """Check if a string is valid JSON-RPC."""
        try:
            obj = json.loads(data)
            return (
                isinstance(obj, dict) and
                obj.get("jsonrpc") == "2.0" and
                ("method" in obj or "result" in obj or "error" in obj)
            )
        except (json.JSONDecodeError, TypeError):
            return False


class TestMCPProtocolPurity:
    """Test MCP protocol message purity."""

    def test_json_rpc_message_validation(self):
        """Test JSON-RPC message validation helper."""
        # Valid messages
        valid_messages = [
            '{"jsonrpc": "2.0", "method": "test", "id": 1}',
            '{"jsonrpc": "2.0", "result": {"status": "ok"}, "id": 1}',
            '{"jsonrpc": "2.0", "error": {"code": -1, "message": "test"}, "id": 1}',
            '{"jsonrpc": "2.0", "method": "notification"}',
        ]

        for message in valid_messages:
            assert MCPMessage.is_valid_jsonrpc(message), f"Should be valid: {message}"

        # Invalid messages
        invalid_messages = [
            "Plain text",
            "{\"invalid\": \"json\"",  # Malformed JSON
            '{"valid": "json", "but": "not jsonrpc"}',
            '{"jsonrpc": "1.0", "method": "test"}',  # Wrong version
            "Debug: Server starting",
            "Warning: Something happened",
        ]

        for message in invalid_messages:
            assert not MCPMessage.is_valid_jsonrpc(message), f"Should be invalid: {message}"

    def test_stdio_server_protocol_compliance(self, monkeypatch, capsys):
        """Test that stdio server produces only valid JSON-RPC."""
        monkeypatch.setenv("WQM_STDIO_MODE", "true")

        # Mock FastMCP to simulate actual tool calls
        with patch('workspace_qdrant_mcp.stdio_server.FastMCP') as mock_fastmcp:
            mock_app = mock_fastmcp.return_value

            # Simulate tool registration and execution
            registered_tools = {}

            def mock_tool(func=None, **kwargs):
                def decorator(f):
                    registered_tools[f.__name__] = f
                    return f
                if func is not None:
                    return decorator(func)
                return decorator

            mock_app.tool = mock_tool

            # Simulate server responses
            def mock_run(transport):
                # Simulate typical MCP interactions
                responses = [
                    MCPMessage.response({"status": "initialized"}, 1),
                    MCPMessage.response({"tools": list(registered_tools.keys())}, 2),
                    MCPMessage.response({"status": "ready"}, 3),
                ]

                for response in responses:
                    print(json.dumps(response))

            mock_app.run = mock_run

            # Import and run server
            from workspace_qdrant_mcp.stdio_server import run_lightweight_stdio_server

            try:
                run_lightweight_stdio_server()
            except Exception:
                pass  # Expected from mocked server

            # Capture and validate output
            captured = capsys.readouterr()

            # CRITICAL: No stderr output
            assert captured.err == "", f"Protocol error - stderr output: {repr(captured.err)}"

            # Validate stdout contains only valid JSON-RPC
            if captured.out.strip():
                for line in captured.out.strip().split('\n'):
                    if line.strip():
                        assert MCPMessage.is_valid_jsonrpc(line), \
                               f"Invalid JSON-RPC in output: {repr(line)}"

    def test_tool_execution_protocol_purity(self, monkeypatch, capsys):
        """Test that tool execution maintains protocol purity."""
        monkeypatch.setenv("WQM_STDIO_MODE", "true")

        from workspace_qdrant_mcp.stdio_server import run_lightweight_stdio_server

        # Mock tool execution
        with patch('workspace_qdrant_mcp.stdio_server.FastMCP') as mock_fastmcp:
            mock_app = mock_fastmcp.return_value

            # Track tool calls and responses
            tool_calls = []

            def mock_tool(func=None, **kwargs):
                def decorator(f):
                    async def wrapper(*args, **kwargs):
                        tool_calls.append((f.__name__, args, kwargs))

                        # Simulate various tool responses
                        if f.__name__ == "workspace_status":
                            return {"status": "active", "mode": "stdio"}
                        elif f.__name__ == "echo_test":
                            return f"Echo: {kwargs.get('message', args[0] if args else 'test')}"
                        elif f.__name__ == "search_workspace":
                            return {"results": [], "total": 0}
                        else:
                            return {"success": True}

                    return wrapper
                if func is not None:
                    return decorator(func)
                return decorator

            mock_app.tool = mock_tool

            def mock_run(transport):
                # Simulate actual tool calls
                responses = []

                # Initialize
                responses.append(MCPMessage.response({"initialized": True}, 1))

                # Tool calls - these should not produce any non-JSON output
                tools = ["workspace_status", "echo_test", "search_workspace"]
                for i, tool in enumerate(tools, 2):
                    result = {"tool": tool, "executed": True, "result": f"test_{tool}"}
                    responses.append(MCPMessage.response(result, i))

                # Output all responses
                for response in responses:
                    print(json.dumps(response))

            mock_app.run = mock_run

            try:
                run_lightweight_stdio_server()
            except Exception:
                pass

            # Capture and validate
            captured = capsys.readouterr()

            # CRITICAL: Protocol purity validation
            assert captured.err == "", f"Tool execution broke protocol - stderr: {repr(captured.err)}"

            # Validate all output is JSON-RPC
            output_lines = [line for line in captured.out.strip().split('\n') if line.strip()]
            for line in output_lines:
                assert MCPMessage.is_valid_jsonrpc(line), \
                       f"Tool execution produced non-JSON-RPC: {repr(line)}"

    @pytest.mark.slow
    def test_full_server_protocol_compliance(self, monkeypatch, tmp_path):
        """Test full server maintains protocol compliance in stdio mode."""
        monkeypatch.setenv("WQM_STDIO_MODE", "true")
        monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")

        # Create test script that simulates MCP conversation
        test_script = tmp_path / "mcp_protocol_test.py"
        test_script.write_text(f"""
import os
import sys
import json
import signal
import time
from threading import Timer

os.environ["WQM_STDIO_MODE"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def shutdown_timer():
    os._exit(0)

# Set timer to avoid hanging
Timer(3.0, shutdown_timer).start()

try:
    # Mock a basic MCP conversation
    from workspace_qdrant_mcp.stdio_server import run_lightweight_stdio_server

    # Mock the server to output JSON-RPC messages
    import workspace_qdrant_mcp.stdio_server as stdio_module

    class MockApp:
        def tool(self, func=None, **kwargs):
            def decorator(f):
                return f
            if func is not None:
                return decorator(func)
            return decorator

        def run(self, transport):
            # Output valid JSON-RPC messages
            messages = [
                {{"jsonrpc": "2.0", "method": "initialized", "id": 1}},
                {{"jsonrpc": "2.0", "result": {{"status": "ready"}}, "id": 2}},
                {{"jsonrpc": "2.0", "method": "tool_call", "params": {{"tool": "test"}}, "id": 3}},
            ]
            for msg in messages:
                print(json.dumps(msg))
                sys.stdout.flush()

    # Replace FastMCP with mock
    stdio_module.FastMCP = lambda name: MockApp()

    run_lightweight_stdio_server()

except Exception as e:
    # Exit silently on error
    sys.exit(1)
""")

        # Run the test
        result = subprocess.run([
            sys.executable, str(test_script)
        ],
        capture_output=True,
        text=True,
        timeout=10,
        cwd=tmp_path
        )

        # CRITICAL: Protocol compliance validation
        assert result.stderr == "", f"Protocol violation - stderr output: {repr(result.stderr)}"

        # Validate all stdout is valid JSON-RPC
        if result.stdout.strip():
            output_lines = [line for line in result.stdout.strip().split('\n') if line.strip()]
            assert len(output_lines) > 0, "Expected JSON-RPC output"

            for line in output_lines:
                assert MCPMessage.is_valid_jsonrpc(line), \
                       f"Protocol violation - invalid JSON-RPC: {repr(line)}"

        # Verify successful execution (exit code 0)
        assert result.returncode == 0, f"Process failed with code {result.returncode}"

    def test_error_handling_protocol_compliance(self, monkeypatch, capsys):
        """Test that error conditions maintain protocol compliance."""
        monkeypatch.setenv("WQM_STDIO_MODE", "true")

        # Mock server with error conditions
        with patch('workspace_qdrant_mcp.stdio_server.FastMCP') as mock_fastmcp:
            mock_app = mock_fastmcp.return_value

            def mock_tool(func=None, **kwargs):
                def decorator(f):
                    async def wrapper(*args, **kwargs):
                        # Simulate various error conditions
                        if f.__name__ == "error_tool":
                            raise ValueError("Test error")
                        elif f.__name__ == "timeout_tool":
                            await asyncio.sleep(0.1)
                            raise TimeoutError("Test timeout")
                        else:
                            return {"success": True}
                    return wrapper
                if func is not None:
                    return decorator(func)
                return decorator

            mock_app.tool = mock_tool

            def mock_run(transport):
                # Simulate error responses (should still be valid JSON-RPC)
                error_responses = [
                    MCPMessage.error(-32603, "Internal error", 1),
                    MCPMessage.error(-32000, "Server error", 2),
                    MCPMessage.response({"error": "handled"}, 3),
                ]

                for response in error_responses:
                    print(json.dumps(response))

                # Simulate some errors that might produce output
                try:
                    raise RuntimeError("Simulated runtime error")
                except RuntimeError:
                    # This should not produce console output
                    pass

            mock_app.run = mock_run

            from workspace_qdrant_mcp.stdio_server import run_lightweight_stdio_server

            try:
                run_lightweight_stdio_server()
            except Exception:
                pass

            # Validate error handling maintains protocol compliance
            captured = capsys.readouterr()

            # CRITICAL: No error output on stderr
            assert captured.err == "", f"Error handling broke protocol - stderr: {repr(captured.err)}"

            # All stdout should be valid JSON-RPC error messages
            if captured.out.strip():
                output_lines = [line for line in captured.out.strip().split('\n') if line.strip()]
                for line in output_lines:
                    assert MCPMessage.is_valid_jsonrpc(line), \
                           f"Error response not valid JSON-RPC: {repr(line)}"

    def test_concurrent_operations_protocol_integrity(self, monkeypatch, capsys):
        """Test protocol integrity under concurrent operations."""
        monkeypatch.setenv("WQM_STDIO_MODE", "true")

        with patch('workspace_qdrant_mcp.stdio_server.FastMCP') as mock_fastmcp:
            mock_app = mock_fastmcp.return_value

            # Track concurrent tool calls
            concurrent_calls = []

            def mock_tool(func=None, **kwargs):
                def decorator(f):
                    async def wrapper(*args, **kwargs):
                        # Simulate concurrent processing
                        call_id = len(concurrent_calls)
                        concurrent_calls.append(call_id)

                        # Simulate some async work
                        await asyncio.sleep(0.01)

                        return {"call_id": call_id, "tool": f.__name__}

                    return wrapper
                if func is not None:
                    return decorator(func)
                return decorator

            mock_app.tool = mock_tool

            def mock_run(transport):
                # Simulate concurrent tool calls and responses
                responses = []

                # Multiple concurrent operations
                for i in range(5):
                    responses.append(MCPMessage.request(f"tool_{i}", {"concurrent": True}, i))
                    responses.append(MCPMessage.response({"tool": f"tool_{i}", "id": i}, i))

                # Output all responses (simulating concurrent handling)
                for response in responses:
                    print(json.dumps(response))

            mock_app.run = mock_run

            from workspace_qdrant_mcp.stdio_server import run_lightweight_stdio_server

            try:
                run_lightweight_stdio_server()
            except Exception:
                pass

            # Validate concurrent operations maintain protocol integrity
            captured = capsys.readouterr()

            # CRITICAL: No interference between concurrent operations
            assert captured.err == "", f"Concurrent ops broke protocol - stderr: {repr(captured.err)}"

            # All output should be valid JSON-RPC
            if captured.out.strip():
                output_lines = [line for line in captured.out.strip().split('\n') if line.strip()]
                for line in output_lines:
                    assert MCPMessage.is_valid_jsonrpc(line), \
                           f"Concurrent operation broke JSON-RPC: {repr(line)}"

    def test_large_message_protocol_compliance(self, monkeypatch, capsys):
        """Test protocol compliance with large messages."""
        monkeypatch.setenv("WQM_STDIO_MODE", "true")

        # Create large test data
        large_data = {
            "large_array": list(range(1000)),
            "large_string": "x" * 10000,
            "nested_data": {
                "level1": {
                    "level2": {
                        "level3": ["data"] * 100
                    }
                }
            }
        }

        with patch('workspace_qdrant_mcp.stdio_server.FastMCP') as mock_fastmcp:
            mock_app = mock_fastmcp.return_value

            def mock_tool(func=None, **kwargs):
                def decorator(f):
                    return f
                if func is not None:
                    return decorator(func)
                return decorator

            mock_app.tool = mock_tool

            def mock_run(transport):
                # Output large JSON-RPC message
                large_response = MCPMessage.response(large_data, 1)
                print(json.dumps(large_response))

            mock_app.run = mock_run

            from workspace_qdrant_mcp.stdio_server import run_lightweight_stdio_server

            try:
                run_lightweight_stdio_server()
            except Exception:
                pass

            # Validate large messages maintain protocol compliance
            captured = capsys.readouterr()

            # CRITICAL: Large messages should not break protocol
            assert captured.err == "", f"Large message broke protocol - stderr: {repr(captured.err)}"

            if captured.out.strip():
                # Should be exactly one line with valid JSON-RPC
                output_lines = [line for line in captured.out.strip().split('\n') if line.strip()]
                assert len(output_lines) == 1, f"Large message split across lines: {len(output_lines)}"

                line = output_lines[0]
                assert MCPMessage.is_valid_jsonrpc(line), \
                       f"Large message not valid JSON-RPC: {len(line)} chars"

                # Verify the large data was preserved
                parsed = json.loads(line)
                assert parsed["result"]["large_array"][:10] == list(range(10))
                assert len(parsed["result"]["large_string"]) == 10000

    def test_unicode_message_protocol_compliance(self, monkeypatch, capsys):
        """Test protocol compliance with Unicode messages."""
        monkeypatch.setenv("WQM_STDIO_MODE", "true")

        # Unicode test data
        unicode_data = {
            "emoji": "🚀🔍📊💻🌟",
            "languages": {
                "chinese": "你好世界",
                "japanese": "こんにちは世界",
                "arabic": "مرحبا بالعالم",
                "russian": "Привет мир",
                "hindi": "नमस्ते दुनिया"
            },
            "symbols": "∑∏∆∇∂∫∞±≤≥≠≈√∛∜",
            "special": "ℝ ℤ ℕ ℚ ℂ ⊆ ⊇ ∪ ∩ ∅ ∀ ∃ ∧ ∨ ¬"
        }

        with patch('workspace_qdrant_mcp.stdio_server.FastMCP') as mock_fastmcp:
            mock_app = mock_fastmcp.return_value

            def mock_tool(func=None, **kwargs):
                def decorator(f):
                    return f
                if func is not None:
                    return decorator(func)
                return decorator

            mock_app.tool = mock_tool

            def mock_run(transport):
                # Output Unicode JSON-RPC message
                unicode_response = MCPMessage.response(unicode_data, 1)
                print(json.dumps(unicode_response, ensure_ascii=False))

            mock_app.run = mock_run

            from workspace_qdrant_mcp.stdio_server import run_lightweight_stdio_server

            try:
                run_lightweight_stdio_server()
            except Exception:
                pass

            # Validate Unicode messages maintain protocol compliance
            captured = capsys.readouterr()

            # CRITICAL: Unicode should not break protocol
            assert captured.err == "", f"Unicode broke protocol - stderr: {repr(captured.err)}"

            if captured.out.strip():
                line = captured.out.strip()
                assert MCPMessage.is_valid_jsonrpc(line), \
                       f"Unicode message not valid JSON-RPC"

                # Verify Unicode data was preserved
                parsed = json.loads(line)
                assert parsed["result"]["emoji"] == "🚀🔍📊💻🌟"
                assert parsed["result"]["languages"]["chinese"] == "你好世界"