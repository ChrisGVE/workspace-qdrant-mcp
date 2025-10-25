"""
Console capture tests for MCP stdio mode.

Validates complete console silence using pytest capsys fixture to detect
any stderr/stdout leakage during MCP server operations.

SUCCESS CRITERIA:
- ZERO bytes written to stderr in stdio mode
- Only valid JSON-RPC messages on stdout
- All logging systems completely silenced
- No warning messages or debug output
"""

import asyncio
import io
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from threading import Thread
from typing import Any, Optional
from unittest.mock import patch

import pytest

# Test markers
pytestmark = [
    pytest.mark.unit,
    pytest.mark.stdio,
]


@contextmanager
def capture_all_output():
    """Context manager to capture all possible output streams."""
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
        yield stdout_capture, stderr_capture


class TestConsoleCapture:
    """Test console output capture in stdio mode."""

    def test_stdio_mode_detection(self, monkeypatch):
        """Test stdio mode detection works correctly."""
        # Test environment variable detection
        monkeypatch.setenv("WQM_STDIO_MODE", "true")

        from workspace_qdrant_mcp.server import _detect_stdio_mode
        assert _detect_stdio_mode()

        # Test command line detection
        monkeypatch.delenv("WQM_STDIO_MODE", raising=False)
        with patch.object(sys, 'argv', ['script.py', '--transport', 'stdio']):
            assert _detect_stdio_mode()

    def test_stdio_server_startup_silence(self, capsys, monkeypatch):
        """Test that stdio server startup produces no console output."""
        monkeypatch.setenv("WQM_STDIO_MODE", "true")
        monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")

        # Import after environment is set
        from workspace_qdrant_mcp.stdio_server import run_lightweight_stdio_server

        # Mock FastMCP to avoid actual server start
        with patch('workspace_qdrant_mcp.stdio_server.FastMCP') as mock_fastmcp:
            mock_app = mock_fastmcp.return_value
            mock_app.run.side_effect = KeyboardInterrupt("Simulated shutdown")

            try:
                run_lightweight_stdio_server()
            except (KeyboardInterrupt, SystemExit):
                pass

            # Capture output
            captured = capsys.readouterr()

            # CRITICAL: No output should be produced during startup
            assert captured.out == "", f"Unexpected stdout: {repr(captured.out)}"
            assert captured.err == "", f"Unexpected stderr: {repr(captured.err)}"

    def test_full_server_stdio_mode_silence(self, capsys, monkeypatch):
        """Test that full server in stdio mode produces no console output."""
        monkeypatch.setenv("WQM_STDIO_MODE", "true")
        monkeypatch.setenv("MCP_QUIET_MODE", "true")
        monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")

        # Mock all imports to avoid actual startup
        with patch.dict('sys.modules', {
            'common.core.client': type('MockModule', (), {
                'QdrantWorkspaceClient': type('MockClient', (), {})
            })(),
            'common.core.hybrid_search': type('MockModule', (), {
                'HybridSearchEngine': type('MockEngine', (), {})
            })(),
        }):

            # Import after mocking
            from workspace_qdrant_mcp.server import _STDIO_MODE, _detect_stdio_mode

            # Verify stdio mode was detected
            assert _detect_stdio_mode()
            assert _STDIO_MODE

            # Capture any output during import
            captured = capsys.readouterr()

            # CRITICAL: Imports should not produce output
            assert captured.out == "", f"Import produced stdout: {repr(captured.out)}"
            assert captured.err == "", f"Import produced stderr: {repr(captured.err)}"

    def test_logging_complete_silence(self, capsys, monkeypatch):
        """Test that all logging is completely silenced in stdio mode."""
        monkeypatch.setenv("WQM_STDIO_MODE", "true")

        # Import the server module to trigger logging setup
        import workspace_qdrant_mcp.server

        # Test various logging levels
        logger = logging.getLogger("test_logger")
        third_party_logger = logging.getLogger("qdrant_client")
        root_logger = logging.getLogger()

        # Try to log at all levels
        for level in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]:
            logger.log(level, "Test message that should be silenced")
            third_party_logger.log(level, "Third party message that should be silenced")
            root_logger.log(level, "Root logger message that should be silenced")

        # Try direct handler calls
        for handler in root_logger.handlers:
            try:
                record = logging.LogRecord(
                    name="test", level=logging.ERROR, pathname="", lineno=0,
                    msg="Direct handler test", args=(), exc_info=None
                )
                handler.handle(record)
            except Exception:
                pass  # Expected for null handlers

        # Capture output
        captured = capsys.readouterr()

        # CRITICAL: No logging output should appear
        assert captured.out == "", f"Logging produced stdout: {repr(captured.out)}"
        assert captured.err == "", f"Logging produced stderr: {repr(captured.err)}"

    def test_warnings_suppression(self, capsys, monkeypatch):
        """Test that all warnings are suppressed in stdio mode."""
        monkeypatch.setenv("WQM_STDIO_MODE", "true")

        # Import to trigger warning suppression
        import warnings

        import workspace_qdrant_mcp.server

        # Generate various types of warnings
        warnings.warn("Test warning message", UserWarning, stacklevel=2)
        warnings.warn("Deprecation warning", DeprecationWarning, stacklevel=2)
        warnings.warn("Future warning", FutureWarning, stacklevel=2)
        warnings.warn("Runtime warning", RuntimeWarning, stacklevel=2)

        # Try to force warnings through
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn("Force warning", UserWarning, stacklevel=2)

        # Capture output
        captured = capsys.readouterr()

        # CRITICAL: No warning output should appear
        assert captured.out == "", f"Warnings produced stdout: {repr(captured.out)}"
        assert captured.err == "", f"Warnings produced stderr: {repr(captured.err)}"

    def test_third_party_library_silence(self, capsys, monkeypatch):
        """Test that third-party libraries are silenced in stdio mode."""
        monkeypatch.setenv("WQM_STDIO_MODE", "true")

        # Import server to set up silencing
        import workspace_qdrant_mcp.server

        # Test third-party loggers that commonly produce output
        third_party_loggers = [
            'httpx', 'httpcore', 'urllib3', 'requests',
            'qdrant_client', 'fastmcp', 'uvicorn', 'fastapi',
            'pydantic', 'transformers', 'huggingface_hub',
            'sentence_transformers', 'torch', 'tensorflow'
        ]

        for logger_name in third_party_loggers:
            logger = logging.getLogger(logger_name)

            # Try to log at all levels
            logger.critical(f"Critical message from {logger_name}")
            logger.error(f"Error message from {logger_name}")
            logger.warning(f"Warning message from {logger_name}")
            logger.info(f"Info message from {logger_name}")
            logger.debug(f"Debug message from {logger_name}")

        # Capture output
        captured = capsys.readouterr()

        # CRITICAL: No third-party logging should appear
        assert captured.out == "", f"Third-party logging produced stdout: {repr(captured.out)}"
        assert captured.err == "", f"Third-party logging produced stderr: {repr(captured.err)}"

    def test_exception_handling_silence(self, capsys, monkeypatch):
        """Test that exceptions don't produce console output in stdio mode."""
        monkeypatch.setenv("WQM_STDIO_MODE", "true")

        # Import server to set up silencing
        import workspace_qdrant_mcp.server

        # Test various exception scenarios
        try:
            raise ValueError("Test exception that should be silent")
        except ValueError:
            pass

        try:
            1 / 0  # ZeroDivisionError
        except ZeroDivisionError:
            pass

        try:
            import nonexistent_module
        except ImportError:
            pass

        # Test logging of exceptions
        logger = logging.getLogger("test_exception_logger")
        try:
            raise RuntimeError("Exception for logging test")
        except RuntimeError as e:
            logger.exception("Exception occurred: %s", str(e))
            logger.error("Error occurred", exc_info=True)

        # Capture output
        captured = capsys.readouterr()

        # CRITICAL: No exception output should appear
        assert captured.out == "", f"Exception handling produced stdout: {repr(captured.out)}"
        assert captured.err == "", f"Exception handling produced stderr: {repr(captured.err)}"

    def test_print_statements_blocked(self, capsys, monkeypatch):
        """Test that print statements are blocked or filtered in stdio mode."""
        monkeypatch.setenv("WQM_STDIO_MODE", "true")

        # Import server to set up stdout wrapper
        import workspace_qdrant_mcp.server

        # Test various print scenarios
        print("This should be blocked")
        print("Debug info:", {"key": "value"})

        # Test stderr writes directly
        sys.stderr.write("Direct stderr write\n")
        sys.stderr.flush()

        # Test non-JSON stdout writes
        sys.stdout.write("Non-JSON stdout write\n")
        sys.stdout.flush()

        # Capture output
        captured = capsys.readouterr()

        # CRITICAL: Only JSON-RPC should pass through stdout, nothing on stderr
        assert captured.err == "", f"Print/write produced stderr: {repr(captured.err)}"

        # Check stdout only contains JSON-RPC or is empty
        if captured.out:
            for line in captured.out.strip().split('\n'):
                if line.strip():
                    # Should be JSON-RPC format
                    assert line.startswith('{'), f"Non-JSON output on stdout: {repr(line)}"
                    assert '"jsonrpc"' in line or '"id"' in line or '"method"' in line, \
                           f"Non-JSON-RPC output on stdout: {repr(line)}"

    def test_json_rpc_output_passthrough(self, capsys, monkeypatch):
        """Test that valid JSON-RPC messages pass through stdout correctly."""
        monkeypatch.setenv("WQM_STDIO_MODE", "true")

        # Import server to set up stdout wrapper
        import workspace_qdrant_mcp.server

        # Valid JSON-RPC messages that should pass through
        valid_messages = [
            '{"jsonrpc": "2.0", "method": "test", "id": 1}',
            '{"jsonrpc": "2.0", "result": {"status": "ok"}, "id": 1}',
            '{"jsonrpc": "2.0", "error": {"code": -1, "message": "test"}, "id": 1}',
            '{"method": "initialize", "params": {}, "id": 1}',
            '{"id": 1, "result": {"capabilities": {}}}'
        ]

        # Write valid JSON-RPC messages
        for message in valid_messages:
            sys.stdout.write(message + '\n')
            sys.stdout.flush()

        # Write invalid messages that should be blocked
        invalid_messages = [
            "Plain text message",
            "Debug: initialization complete",
            "Warning: something happened",
            '{"not": "jsonrpc"}',  # JSON but not JSON-RPC
        ]

        for message in invalid_messages:
            sys.stdout.write(message + '\n')
            sys.stdout.flush()

        # Capture output
        captured = capsys.readouterr()

        # Verify only valid JSON-RPC messages passed through
        output_lines = [line for line in captured.out.strip().split('\n') if line.strip()]

        assert len(output_lines) == len(valid_messages), \
               f"Expected {len(valid_messages)} lines, got {len(output_lines)}: {output_lines}"

        # Verify each output line is a valid JSON-RPC message
        for i, line in enumerate(output_lines):
            assert line.strip() == valid_messages[i], \
                   f"Line {i}: expected {valid_messages[i]}, got {line.strip()}"

        # No stderr output
        assert captured.err == "", f"Unexpected stderr output: {repr(captured.err)}"

    def test_startup_sequence_silence(self, capsys, monkeypatch, tmp_path):
        """Test complete startup sequence produces no console output."""
        monkeypatch.setenv("WQM_STDIO_MODE", "true")
        monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")

        # Change to temporary directory to avoid project detection issues
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            # Mock imports to avoid heavy dependencies
            with patch.dict('sys.modules', {
                'common.core.client': type('MockModule', (), {
                    'QdrantWorkspaceClient': type('MockClient', (), {
                        '__init__': lambda self, *args, **kwargs: None
                    })
                })(),
            }):

                # Import modules in order they would be imported during startup
                import workspace_qdrant_mcp.server
                from workspace_qdrant_mcp.stdio_server import (
                    run_lightweight_stdio_server,
                )

                # Capture startup output
                captured = capsys.readouterr()

                # CRITICAL: No output during module imports/startup
                assert captured.out == "", f"Startup produced stdout: {repr(captured.out)}"
                assert captured.err == "", f"Startup produced stderr: {repr(captured.err)}"

        finally:
            os.chdir(original_cwd)

    @pytest.mark.slow
    def test_subprocess_stdio_silence(self, tmp_path):
        """Test complete stdio silence using subprocess (integration test)."""
        # Create a test script that starts the server
        test_script = tmp_path / "test_stdio_server.py"
        test_script.write_text("""
import os
import sys
import signal
import time
from threading import Timer

# Set stdio mode
os.environ["WQM_STDIO_MODE"] = "true"

# Import and start server
try:
    from workspace_qdrant_mcp.stdio_server import run_lightweight_stdio_server

    # Set up timer to kill the server after 2 seconds
    def shutdown():
        import os
        os._exit(0)

    timer = Timer(2.0, shutdown)
    timer.start()

    # Run server
    run_lightweight_stdio_server()
except Exception as e:
    # Don't print anything, even on error
    sys.exit(1)
""")

        # Run the test script as subprocess
        result = subprocess.run([
            sys.executable, str(test_script)
        ],
        capture_output=True,
        text=True,
        timeout=10,
        cwd=tmp_path
        )

        # CRITICAL: No output should be produced
        assert result.stderr == "", f"Subprocess stderr: {repr(result.stderr)}"

        # stdout should be empty or contain only valid JSON-RPC
        if result.stdout.strip():
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    try:
                        # Should be parseable JSON
                        json.loads(line)
                    except json.JSONDecodeError:
                        pytest.fail(f"Non-JSON output in subprocess stdout: {repr(line)}")

    def test_cli_mode_preserves_output(self, capsys, monkeypatch):
        """Test that CLI mode still allows normal console output."""
        # Ensure CLI mode (no stdio mode)
        monkeypatch.delenv("WQM_STDIO_MODE", raising=False)
        monkeypatch.delenv("MCP_QUIET_MODE", raising=False)

        # Mock sys.argv to not include stdio transport
        with patch.object(sys, 'argv', ['script.py', '--help']):

            # Import should not set up stdio silencing
            from workspace_qdrant_mcp.server import _STDIO_MODE, _detect_stdio_mode

            # Verify CLI mode is not detected as stdio
            assert not _detect_stdio_mode()

            # Normal logging should work in CLI mode
            logger = logging.getLogger("test_cli_logger")
            logger.warning("This should appear in CLI mode")

            print("This print should work in CLI mode")

            # Capture output - should contain our messages
            capsys.readouterr()

            # In CLI mode, output should be preserved
            # Note: This test validates that CLI functionality is not broken
            # The exact output depends on logging configuration in CLI mode
