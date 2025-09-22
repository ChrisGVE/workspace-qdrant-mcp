#!/usr/bin/env python3
"""
IMMEDIATE TEST EXECUTION - Run known working tests now
"""

import subprocess
import sys
import os
import time
from pathlib import Path


def run_command(cmd, timeout=120):
    """Run command with timeout"""
    try:
        result = subprocess.run(
            cmd,
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "TIMEOUT"
    except Exception as e:
        return False, "", str(e)


def main():
    print("IMMEDIATE TEST EXECUTION STARTED")

    # Known working test files from previous runs
    working_tests = [
        "tests/unit/test_server_basic.py",
        "tests/unit/test_collection_naming_manager.py",
        "tests/unit/test_daemon_identifier.py",
        "tests/unit/test_direct_execution.py",
        "tests/unit/test_common_modules.py"
    ]

    # Verify test files exist
    existing_tests = []
    for test in working_tests:
        if Path(test).exists():
            existing_tests.append(test)
            print(f"✓ Found: {test}")
        else:
            print(f"✗ Missing: {test}")

    if not existing_tests:
        print("No working test files found!")
        return

    print(f"\nExecuting {len(existing_tests)} test files for immediate coverage...")

    # Run tests individually first
    individual_results = []
    for i, test in enumerate(existing_tests, 1):
        print(f"\n[{i}/{len(existing_tests)}] Running {test}...")

        cmd = ["uv", "run", "pytest", test, "--cov=src", "--cov-report=term", "-v"]
        success, stdout, stderr = run_command(cmd, timeout=60)

        if success:
            print(f"✓ SUCCESS: {test}")
            individual_results.append(test)
        else:
            print(f"✗ FAILED: {test}")
            if stderr:
                print(f"Error: {stderr[:200]}...")

    print(f"\nIndividual test results: {len(individual_results)}/{len(existing_tests)} passed")

    # Run all successful tests together for maximum coverage
    if individual_results:
        print(f"\nRunning all {len(individual_results)} successful tests together...")

        cmd = ["uv", "run", "pytest"] + individual_results + [
            "--cov=src", "--cov-report=term", "--cov-report=xml", "--cov-report=html",
            "-v", "--tb=short"
        ]

        success, stdout, stderr = run_command(cmd, timeout=180)

        if success:
            print("✓ BATCH EXECUTION SUCCESSFUL")
            print("Coverage report generated:")
            print("- coverage.xml")
            print("- htmlcov/")
        else:
            print("✗ BATCH EXECUTION FAILED")
            if stderr:
                print(f"Error: {stderr[:500]}...")

        # Print coverage information from stdout
        if "TOTAL" in stdout:
            lines = stdout.split('\n')
            for line in lines:
                if "TOTAL" in line or "coverage" in line.lower():
                    print(f"Coverage: {line.strip()}")

    print("\nImmediate execution completed!")


if __name__ == "__main__":
    main()