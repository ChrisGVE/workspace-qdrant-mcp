"""
Console silence testing configuration and fixtures.

Provides shared fixtures, configuration, and utilities for comprehensive
console silence validation testing.
"""

import os
import sys
import tempfile
import logging
from pathlib import Path
from typing import Any, Dict, Generator

import pytest


@pytest.fixture(scope="session", autouse=True)
def clean_environment():
    """Clean environment for consistent testing."""
    # Store original environment
    original_env = os.environ.copy()

    # Clean known environment variables
    env_vars_to_clean = [
        "WQM_STDIO_MODE",
        "MCP_QUIET_MODE",
        "TOKENIZERS_PARALLELISM",
        "MCP_TRANSPORT"
    ]

    for var in env_vars_to_clean:
        os.environ.pop(var, None)

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def isolated_modules():
    """Provide isolated module import context."""
    # Store original modules
    modules_to_isolate = [
        "workspace_qdrant_mcp.server",
        "workspace_qdrant_mcp.stdio_server",
        "workspace_qdrant_mcp.launcher"
    ]

    original_modules = {}
    for module in modules_to_isolate:
        if module in sys.modules:
            original_modules[module] = sys.modules[module]
            del sys.modules[module]

    yield

    # Restore modules
    for module in modules_to_isolate:
        if module in sys.modules:
            del sys.modules[module]
        if module in original_modules:
            sys.modules[module] = original_modules[module]


@pytest.fixture
def temp_workspace():
    """Create temporary workspace for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        # Create basic workspace structure
        (workspace / "src").mkdir()
        (workspace / "tests").mkdir()
        (workspace / "docs").mkdir()

        # Create sample files
        (workspace / "README.md").write_text("# Test Project")
        (workspace / "src" / "main.py").write_text("print('Hello World')")
        (workspace / "tests" / "test_main.py").write_text("def test_main(): pass")

        yield workspace


@pytest.fixture
def stdio_environment():
    """Set up stdio mode environment."""
    os.environ["WQM_STDIO_MODE"] = "true"
    os.environ["MCP_QUIET_MODE"] = "true"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    yield

    # Cleanup
    for var in ["WQM_STDIO_MODE", "MCP_QUIET_MODE", "TOKENIZERS_PARALLELISM"]:
        os.environ.pop(var, None)


@pytest.fixture
def cli_environment():
    """Set up CLI mode environment (opposite of stdio)."""
    # Ensure stdio mode is disabled
    for var in ["WQM_STDIO_MODE", "MCP_QUIET_MODE"]:
        os.environ.pop(var, None)

    yield


@pytest.fixture
def console_capture_validator():
    """Provide console capture validation utilities."""
    class ConsoleValidator:
        @staticmethod
        def validate_complete_silence(captured):
            """Validate complete console silence."""
            assert captured.out == "", f"Unexpected stdout: {repr(captured.out)}"
            assert captured.err == "", f"Unexpected stderr: {repr(captured.err)}"

        @staticmethod
        def validate_json_rpc_only(captured):
            """Validate only JSON-RPC messages on stdout."""
            if captured.err:
                pytest.fail(f"Stderr output detected: {repr(captured.err)}")

            if captured.out.strip():
                import json
                lines = [line for line in captured.out.strip().split('\n') if line.strip()]
                for line in lines:
                    try:
                        obj = json.loads(line)
                        # Check if it's JSON-RPC
                        if not (isinstance(obj, dict) and
                               obj.get("jsonrpc") == "2.0" and
                               ("method" in obj or "result" in obj or "error" in obj)):
                            pytest.fail(f"Non-JSON-RPC output: {repr(line)}")
                    except json.JSONDecodeError:
                        pytest.fail(f"Non-JSON output on stdout: {repr(line)}")

    return ConsoleValidator()


@pytest.fixture(scope="session")
def test_performance_tracker():
    """Track performance metrics across tests."""
    class PerformanceTracker:
        def __init__(self):
            self.metrics = {
                "startup_times": [],
                "memory_usage": [],
                "tool_latencies": [],
                "error_counts": []
            }

        def record_startup_time(self, time_ms):
            self.metrics["startup_times"].append(time_ms)

        def record_memory_usage(self, memory_mb):
            self.metrics["memory_usage"].append(memory_mb)

        def record_tool_latency(self, latency_ms):
            self.metrics["tool_latencies"].append(latency_ms)

        def record_error(self):
            self.metrics["error_counts"].append(1)

        def get_summary(self):
            summary = {}
            for metric, values in self.metrics.items():
                if values:
                    if metric == "error_counts":
                        summary[f"total_{metric}"] = sum(values)
                    else:
                        summary[f"avg_{metric}"] = sum(values) / len(values)
                        summary[f"max_{metric}"] = max(values)
                        summary[f"min_{metric}"] = min(values)
            return summary

    tracker = PerformanceTracker()
    yield tracker

    # Print summary at end
    summary = tracker.get_summary()
    if summary:
        print("\n=== PERFORMANCE SUMMARY ===")
        for metric, value in summary.items():
            print(f"{metric}: {value:.2f}")


# Test markers for categorization
pytest_markers = [
    "stdio: Tests related to stdio mode",
    "console_silence: Tests validating complete console silence",
    "protocol_purity: Tests validating MCP protocol purity",
    "performance: Performance benchmark tests",
    "integration: Integration tests with MCP clients",
    "regression: Regression tests for console silence",
]


def pytest_configure(config):
    """Configure pytest with custom markers."""
    for marker in pytest_markers:
        config.addinivalue_line("markers", marker)


def pytest_collection_modifyitems(config, items):
    """Modify test collection for console silence testing."""
    # Add stdio marker to all tests in this directory
    stdio_marker = pytest.mark.stdio

    for item in items:
        if "stdio_console_silence" in str(item.fspath):
            item.add_marker(stdio_marker)

            # Add specific markers based on test name
            if "performance" in item.name:
                item.add_marker(pytest.mark.performance)
            if "integration" in item.name:
                item.add_marker(pytest.mark.integration)
            if "protocol" in item.name:
                item.add_marker(pytest.mark.protocol_purity)
            if "capture" in item.name:
                item.add_marker(pytest.mark.console_silence)


@pytest.fixture(autouse=True)
def prevent_test_pollution():
    """Prevent tests from polluting each other's environments."""
    # Store original state
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    original_logging_level = logging.root.level
    original_logging_handlers = logging.root.handlers.copy()

    yield

    # Restore original state
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    logging.root.level = original_logging_level

    # Clear and restore handlers
    logging.root.handlers.clear()
    logging.root.handlers.extend(original_logging_handlers)


@pytest.fixture
def mock_qdrant_environment(monkeypatch):
    """Mock Qdrant-related imports to avoid dependencies."""
    # Mock the heavy imports that might cause issues
    mock_modules = {
        'qdrant_client': type('MockQdrantClient', (), {}),
        'fastembed': type('MockFastEmbed', (), {}),
        'common.core.client': type('MockClient', (), {
            'QdrantWorkspaceClient': type('MockWorkspaceClient', (), {})
        }),
    }

    for module_name, mock_module in mock_modules.items():
        monkeypatch.setitem(sys.modules, module_name, mock_module)

    yield


def validate_test_requirements():
    """Validate test environment requirements."""
    # Check Python version
    if sys.version_info < (3, 10):
        pytest.skip("Requires Python 3.10+")

    # Check required packages
    try:
        import pytest_benchmark
        import psutil
    except ImportError as e:
        pytest.skip(f"Missing required package for console silence tests: {e}")


# Run requirement validation at import
validate_test_requirements()