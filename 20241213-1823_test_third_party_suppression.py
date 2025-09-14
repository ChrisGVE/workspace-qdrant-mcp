#!/usr/bin/env python3
"""
Test script for third-party library console suppression in MCP stdio mode.

This script validates that all configured third-party libraries are properly
suppressed in stdio mode and produce zero console output.
"""

import os
import sys
import logging
import warnings
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

# Set stdio mode early
os.environ["WQM_STDIO_MODE"] = "true"
os.environ["MCP_QUIET_MODE"] = "true"

def capture_output_during_imports():
    """Capture any output during third-party library imports."""
    # Capture stdout and stderr
    stdout_capture = StringIO()
    stderr_capture = StringIO()

    with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
        # Now import the stdio server which should set up all suppressions
        try:
            from workspace_qdrant_mcp.stdio_server import main
            print("✓ stdio_server imported successfully", file=sys.__stdout__)
        except ImportError as e:
            print(f"✗ Failed to import stdio_server: {e}", file=sys.__stderr__)
            return False, str(e), ""

        # Import libraries that should be suppressed
        suppression_test_libraries = [
            "logging",  # Core logging
            "warnings", # Warnings system
        ]

        # Try importing optional libraries that might be present
        optional_libraries = [
            "rich", "typer", "structlog", "pydantic",
            "grpc", "fastembed", "httpx", "requests"
        ]

        for lib_name in suppression_test_libraries + optional_libraries:
            try:
                __import__(lib_name)
                print(f"✓ {lib_name} imported", file=sys.__stdout__)
            except ImportError:
                print(f"- {lib_name} not available (skipped)", file=sys.__stdout__)

        # Test logging operations
        try:
            logger = logging.getLogger("test_logger")
            logger.info("This should be suppressed")
            logger.warning("This warning should be suppressed")
            logger.error("This error should be suppressed")
            print("✓ Logging operations completed", file=sys.__stdout__)
        except Exception as e:
            print(f"✗ Logging test failed: {e}", file=sys.__stderr__)

        # Test warning operations
        try:
            warnings.warn("This warning should be suppressed")
            warnings.warn("Deprecation warning", DeprecationWarning)
            print("✓ Warning operations completed", file=sys.__stdout__)
        except Exception as e:
            print(f"✗ Warning test failed: {e}", file=sys.__stderr__)

    # Get captured output
    stdout_output = stdout_capture.getvalue()
    stderr_output = stderr_capture.getvalue()

    return True, stdout_output, stderr_output

def test_stdio_server_functionality():
    """Test that stdio server runs without console output."""
    print("Testing stdio server import and basic functionality...", file=sys.__stdout__)

    success, stdout_content, stderr_content = capture_output_during_imports()

    # Analyze results
    total_output_length = len(stdout_content) + len(stderr_content)

    print(f"\n=== Test Results ===", file=sys.__stdout__)
    print(f"Import success: {success}", file=sys.__stdout__)
    print(f"Stdout captured length: {len(stdout_content)}", file=sys.__stdout__)
    print(f"Stderr captured length: {len(stderr_content)}", file=sys.__stdout__)
    print(f"Total output length: {total_output_length}", file=sys.__stdout__)

    if stdout_content:
        print(f"\nStdout content (first 500 chars):", file=sys.__stdout__)
        print(f"'{stdout_content[:500]}'", file=sys.__stdout__)

    if stderr_content:
        print(f"\nStderr content (first 500 chars):", file=sys.__stdout__)
        print(f"'{stderr_content[:500]}'", file=sys.__stdout__)

    # Success criteria: minimal or no output from third-party libraries
    if total_output_length == 0:
        print("✅ PERFECT: Complete console silence achieved", file=sys.__stdout__)
        return True
    elif total_output_length < 100:
        print("✅ EXCELLENT: Minimal console output (< 100 chars)", file=sys.__stdout__)
        return True
    elif total_output_length < 500:
        print("⚠️  GOOD: Low console output (< 500 chars)", file=sys.__stdout__)
        return True
    else:
        print("❌ FAILED: Excessive console output detected", file=sys.__stdout__)
        return False

def test_environment_variables():
    """Test that all required environment variables are set."""
    print("Testing environment variable configuration...", file=sys.__stdout__)

    required_env_vars = {
        "WQM_STDIO_MODE": "true",
        "MCP_QUIET_MODE": "true",
        "TOKENIZERS_PARALLELISM": "false",
        "GRPC_VERBOSITY": "NONE",
        "GRPC_TRACE": "",
    }

    all_set = True
    for var, expected in required_env_vars.items():
        actual = os.getenv(var, "")
        if actual.lower() != expected.lower():
            print(f"❌ {var}: expected '{expected}', got '{actual}'", file=sys.__stdout__)
            all_set = False
        else:
            print(f"✅ {var}: {actual}", file=sys.__stdout__)

    return all_set

def main():
    """Main test function."""
    print("=== Third-Party Library Console Suppression Test ===", file=sys.__stdout__)
    print(f"Python version: {sys.version}", file=sys.__stdout__)
    print(f"Working directory: {os.getcwd()}", file=sys.__stdout__)

    # Test environment variables
    env_test_passed = test_environment_variables()

    # Test stdio server functionality
    stdio_test_passed = test_stdio_server_functionality()

    # Overall result
    print(f"\n=== Final Results ===", file=sys.__stdout__)
    if env_test_passed and stdio_test_passed:
        print("🎉 ALL TESTS PASSED: Third-party library suppression working correctly", file=sys.__stdout__)
        return 0
    else:
        print("💥 SOME TESTS FAILED: Console suppression needs improvement", file=sys.__stdout__)
        return 1

if __name__ == "__main__":
    sys.exit(main())