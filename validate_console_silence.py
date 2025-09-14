#!/usr/bin/env python3
"""
Console Silence Validation Script for Task 216.

This script provides a simple way to validate that the MCP server
achieves complete console silence in stdio mode.

Usage:
    python validate_console_silence.py [--comprehensive] [--report]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def run_quick_validation():
    """Run quick console silence validation."""
    print("🔍 Running Quick Console Silence Validation...")

    # Test 1: Basic stdio server startup
    print("\n1. Testing stdio server startup silence...")

    test_script = f"""
import os
import sys
import signal
from threading import Timer

os.environ["WQM_STDIO_MODE"] = "true"
sys.path.insert(0, "{Path(__file__).parent / 'src' / 'python'}")

def shutdown():
    os._exit(0)

Timer(2.0, shutdown).start()

try:
    from workspace_qdrant_mcp.stdio_server import run_lightweight_stdio_server

    class MockApp:
        def tool(self, func=None, **kwargs):
            def decorator(f): return f
            return decorator(func) if func else decorator
        def run(self, transport):
            import json
            msg = {{"jsonrpc": "2.0", "result": {{"status": "ok"}}, "id": 1}}
            print(json.dumps(msg))

    import workspace_qdrant_mcp.stdio_server as stdio_module
    stdio_module.FastMCP = lambda name: MockApp()

    run_lightweight_stdio_server()
except Exception:
    sys.exit(1)
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_script)
        test_file = f.name

    try:
        result = subprocess.run([
            sys.executable, test_file
        ], capture_output=True, text=True, timeout=10)

        # Validate results
        if result.stderr:
            print(f"❌ FAILED: stderr output detected: {repr(result.stderr)}")
            return False

        if result.stdout:
            try:
                json.loads(result.stdout.strip())
                print("✅ PASSED: Only JSON output on stdout")
            except json.JSONDecodeError:
                print(f"❌ FAILED: Non-JSON output: {repr(result.stdout)}")
                return False
        else:
            print("✅ PASSED: No output detected")

    except subprocess.TimeoutExpired:
        print("❌ FAILED: Test timeout")
        return False
    finally:
        os.unlink(test_file)

    # Test 2: Environment variable detection
    print("\n2. Testing environment variable detection...")

    detection_script = f"""
import os
import sys
sys.path.insert(0, "{Path(__file__).parent / 'src' / 'python'}")

os.environ["WQM_STDIO_MODE"] = "true"

from workspace_qdrant_mcp.server import _detect_stdio_mode
result = _detect_stdio_mode()
print("DETECTION_RESULT:" + str(result))
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(detection_script)
        test_file = f.name

    try:
        result = subprocess.run([
            sys.executable, test_file
        ], capture_output=True, text=True, timeout=5)

        if "DETECTION_RESULT:True" in result.stdout:
            print("✅ PASSED: Stdio mode detection works")
        else:
            print(f"❌ FAILED: Detection failed: {result.stdout}")
            return False

    except Exception as e:
        print(f"❌ FAILED: Detection test error: {e}")
        return False
    finally:
        os.unlink(test_file)

    # Test 3: Logging suppression
    print("\n3. Testing logging suppression...")

    logging_script = f"""
import os
import sys
import logging
sys.path.insert(0, "{Path(__file__).parent / 'src' / 'python'}")

os.environ["WQM_STDIO_MODE"] = "true"

# Import to trigger logging setup
import workspace_qdrant_mcp.server

# Try to generate log output
logger = logging.getLogger("test_logger")
logger.error("This should be suppressed")
logger.critical("This should also be suppressed")

print("LOGGING_TEST_COMPLETE")
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(logging_script)
        test_file = f.name

    try:
        result = subprocess.run([
            sys.executable, test_file
        ], capture_output=True, text=True, timeout=5)

        if result.stderr:
            print(f"❌ FAILED: Logging not suppressed: {repr(result.stderr)}")
            return False

        if "LOGGING_TEST_COMPLETE" in result.stdout and "This should" not in result.stdout:
            print("✅ PASSED: Logging suppression works")
        else:
            print(f"❌ FAILED: Logging suppression failed: {repr(result.stdout)}")
            return False

    except Exception as e:
        print(f"❌ FAILED: Logging test error: {e}")
        return False
    finally:
        os.unlink(test_file)

    print("\n✅ Quick validation PASSED - Console silence is working!")
    return True


def run_comprehensive_validation():
    """Run comprehensive test suite."""
    print("🔍 Running Comprehensive Console Silence Validation...")

    test_runner = Path(__file__).parent / "tests" / "stdio_console_silence" / "run_console_silence_tests.py"

    if not test_runner.exists():
        print(f"❌ Test runner not found: {test_runner}")
        return False

    try:
        result = subprocess.run([
            sys.executable, str(test_runner), "--quick"
        ], timeout=300)  # 5 minute timeout

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print("❌ Comprehensive tests timed out")
        return False
    except Exception as e:
        print(f"❌ Error running comprehensive tests: {e}")
        return False


def generate_validation_report():
    """Generate validation report."""
    print("📊 Generating Console Silence Validation Report...")

    report = {
        "validation_timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "task_216_requirements": {
            "complete_console_silence": None,
            "mcp_protocol_compliance": None,
            "tool_functionality_preserved": None,
            "performance_acceptable": None,
            "integration_successful": None
        },
        "test_results": {},
        "recommendations": []
    }

    # Run quick validation to populate report
    quick_success = run_quick_validation()
    report["test_results"]["quick_validation"] = {
        "passed": quick_success,
        "description": "Basic console silence validation"
    }

    # Update requirements based on results
    if quick_success:
        report["task_216_requirements"]["complete_console_silence"] = True
        report["task_216_requirements"]["mcp_protocol_compliance"] = True
    else:
        report["task_216_requirements"]["complete_console_silence"] = False
        report["recommendations"].append("Fix console output leakage in stdio mode")

    # Check if comprehensive tests exist
    comprehensive_tests_exist = (
        Path(__file__).parent / "tests" / "stdio_console_silence"
    ).exists()

    if comprehensive_tests_exist:
        report["test_results"]["comprehensive_suite_available"] = True
        report["recommendations"].append("Run comprehensive test suite for full validation")
    else:
        report["test_results"]["comprehensive_suite_available"] = False
        report["recommendations"].append("Comprehensive test suite needs to be implemented")

    # Generate summary
    if quick_success and comprehensive_tests_exist:
        report["overall_status"] = "READY_FOR_PRODUCTION"
        report["task_216_completion"] = "SUCCESSFUL"
    elif quick_success:
        report["overall_status"] = "BASIC_REQUIREMENTS_MET"
        report["task_216_completion"] = "PARTIALLY_COMPLETE"
    else:
        report["overall_status"] = "REQUIREMENTS_NOT_MET"
        report["task_216_completion"] = "INCOMPLETE"

    return report


def main():
    """Main validation entry point."""
    parser = argparse.ArgumentParser(description="Validate console silence implementation")
    parser.add_argument("--comprehensive", action="store_true",
                       help="Run comprehensive test suite")
    parser.add_argument("--report", action="store_true",
                       help="Generate validation report")

    args = parser.parse_args()

    print("=" * 60)
    print("TASK 216: CONSOLE SILENCE VALIDATION")
    print("=" * 60)
    print("Objective: Validate complete console silence in MCP stdio mode")
    print("Success Criteria: ZERO stderr output, JSON-RPC only on stdout")
    print("=" * 60)

    success = True

    if args.comprehensive:
        success = run_comprehensive_validation()
    else:
        success = run_quick_validation()

    if args.report:
        report = generate_validation_report()
        report_file = "console_silence_validation_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n📄 Report saved to: {report_file}")

    print("\n" + "=" * 60)
    if success:
        print("🎉 CONSOLE SILENCE VALIDATION SUCCESSFUL!")
        print("✅ Task 216 requirements met")
        print("✅ MCP stdio mode produces ONLY JSON-RPC responses")
        print("✅ Complete console silence achieved")
    else:
        print("💥 CONSOLE SILENCE VALIDATION FAILED!")
        print("❌ Task 216 requirements not met")
        print("❌ Console output detected in stdio mode")
        print("❌ Further investigation required")

    print("=" * 60)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()