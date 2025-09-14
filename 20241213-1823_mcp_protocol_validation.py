#!/usr/bin/env python3
"""
MCP JSON-RPC protocol validation test.

This script validates that MCP stdio mode produces clean JSON-RPC communication
without any console interference from third-party libraries.
"""

import os
import sys
import json
import asyncio
import subprocess
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

def test_mcp_stdio_protocol():
    """Test MCP stdio protocol for clean JSON-RPC communication."""
    print("=== MCP JSON-RPC Protocol Validation ===", file=sys.__stdout__)

    # Set stdio mode environment
    env = os.environ.copy()
    env["WQM_STDIO_MODE"] = "true"
    env["MCP_QUIET_MODE"] = "true"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["GRPC_VERBOSITY"] = "NONE"
    env["GRPC_TRACE"] = ""

    # Test initialize request
    initialize_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0.0"}
        }
    }

    # Test list_tools request
    list_tools_request = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list"
    }

    # Test call tool request
    call_tool_request = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "workspace_status",
            "arguments": {}
        }
    }

    print("Testing MCP stdio server communication...", file=sys.__stdout__)

    try:
        # Launch stdio server
        process = subprocess.Popen(
            [sys.executable, "-m", "workspace_qdrant_mcp.stdio_server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True
        )

        # Send requests and collect responses
        requests = [initialize_request, list_tools_request, call_tool_request]
        stdin_data = ""
        for request in requests:
            stdin_data += json.dumps(request) + "\n"

        # Communicate with timeout
        try:
            stdout_data, stderr_data = process.communicate(input=stdin_data, timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout_data, stderr_data = process.communicate()
            print("❌ Server timeout - process killed", file=sys.__stdout__)
            return False

        # Analyze results
        print(f"Process exit code: {process.returncode}", file=sys.__stdout__)
        print(f"Stdout length: {len(stdout_data)} chars", file=sys.__stdout__)
        print(f"Stderr length: {len(stderr_data)} chars", file=sys.__stdout__)

        # Check for clean JSON-RPC output
        valid_json_responses = 0
        invalid_lines = []

        if stdout_data:
            for line_num, line in enumerate(stdout_data.strip().split('\n'), 1):
                if line.strip():
                    try:
                        response = json.loads(line.strip())
                        if "jsonrpc" in response:
                            valid_json_responses += 1
                            print(f"✅ Line {line_num}: Valid JSON-RPC response", file=sys.__stdout__)
                        else:
                            invalid_lines.append((line_num, line[:100]))
                    except json.JSONDecodeError:
                        invalid_lines.append((line_num, line[:100]))

        # Check stderr for unwanted output
        if stderr_data.strip():
            print(f"⚠️  Stderr contains data: {stderr_data[:200]}...", file=sys.__stdout__)

        # Report results
        print(f"\n=== Protocol Validation Results ===", file=sys.__stdout__)
        print(f"Valid JSON-RPC responses: {valid_json_responses}", file=sys.__stdout__)
        print(f"Invalid/non-JSON lines: {len(invalid_lines)}", file=sys.__stdout__)

        if invalid_lines:
            print("Invalid lines found:", file=sys.__stdout__)
            for line_num, content in invalid_lines[:5]:  # Show first 5
                print(f"  Line {line_num}: {content}", file=sys.__stdout__)

        # Success criteria
        success = (
            process.returncode == 0 and
            valid_json_responses >= 2 and  # At least initialize and list_tools responses
            len(invalid_lines) == 0 and
            len(stderr_data.strip()) == 0
        )

        if success:
            print("🎉 MCP JSON-RPC protocol validation PASSED", file=sys.__stdout__)
        else:
            print("💥 MCP JSON-RPC protocol validation FAILED", file=sys.__stdout__)

        return success

    except Exception as e:
        print(f"❌ Test failed with exception: {e}", file=sys.__stdout__)
        return False

def test_library_import_silence():
    """Test that importing all libraries produces no console output."""
    print("\n=== Library Import Silence Test ===", file=sys.__stdout__)

    # Set stdio mode
    os.environ["WQM_STDIO_MODE"] = "true"

    # Capture any output during imports
    stdout_capture = StringIO()
    stderr_capture = StringIO()

    with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
        try:
            # Import all the modules that should be silent
            import warnings
            import logging
            warnings.filterwarnings("ignore")

            # Import common logging
            from common.logging.core import THIRD_PARTY_LOGGERS, StructuredLogger

            # Test logger creation and usage
            logger = StructuredLogger("test")
            logger.info("This should be completely silent")
            logger.warning("Silent warning")
            logger.error("Silent error")

            # Test third-party logger suppression
            for logger_name in THIRD_PARTY_LOGGERS[:10]:  # Test first 10
                test_logger = logging.getLogger(logger_name)
                test_logger.info(f"Testing {logger_name} suppression")

        except Exception as e:
            print(f"❌ Import test failed: {e}", file=sys.__stdout__)
            return False

    stdout_content = stdout_capture.getvalue()
    stderr_content = stderr_capture.getvalue()
    total_output = len(stdout_content) + len(stderr_content)

    print(f"Import output length: {total_output} chars", file=sys.__stdout__)

    if total_output == 0:
        print("✅ Perfect silence achieved during imports", file=sys.__stdout__)
        return True
    elif total_output < 50:
        print("✅ Acceptable minimal output during imports", file=sys.__stdout__)
        return True
    else:
        print(f"❌ Excessive output during imports: {total_output} chars", file=sys.__stdout__)
        if stdout_content:
            print(f"Stdout: {stdout_content[:200]}", file=sys.__stdout__)
        if stderr_content:
            print(f"Stderr: {stderr_content[:200]}", file=sys.__stdout__)
        return False

def main():
    """Main validation function."""
    print("Comprehensive MCP Protocol and Third-Party Suppression Validation", file=sys.__stdout__)
    print(f"Python version: {sys.version}", file=sys.__stdout__)
    print(f"Working directory: {os.getcwd()}\n", file=sys.__stdout__)

    # Run all tests
    library_test_passed = test_library_import_silence()
    protocol_test_passed = test_mcp_stdio_protocol()

    # Final results
    print(f"\n=== Final Validation Results ===", file=sys.__stdout__)
    print(f"Library import silence: {'✅ PASS' if library_test_passed else '❌ FAIL'}", file=sys.__stdout__)
    print(f"MCP JSON-RPC protocol: {'✅ PASS' if protocol_test_passed else '❌ FAIL'}", file=sys.__stdout__)

    if library_test_passed and protocol_test_passed:
        print("\n🎉 ALL VALIDATION TESTS PASSED", file=sys.__stdout__)
        print("✅ Third-party library console suppression is working correctly", file=sys.__stdout__)
        print("✅ MCP stdio mode produces clean JSON-RPC communication", file=sys.__stdout__)
        return 0
    else:
        print("\n💥 SOME VALIDATION TESTS FAILED", file=sys.__stdout__)
        print("❌ Further improvements needed for complete stdio compatibility", file=sys.__stdout__)
        return 1

if __name__ == "__main__":
    sys.exit(main())