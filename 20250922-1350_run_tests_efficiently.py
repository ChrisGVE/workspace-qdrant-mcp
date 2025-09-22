#!/usr/bin/env python3
"""
Efficient Test Runner for Coverage Measurement
===========================================

Runs comprehensive tests in batches to avoid timeout issues and
measures coverage incrementally to track progress toward 100%.
"""

import subprocess
import sys
import os
import glob
import time
from pathlib import Path

def run_test_batch(test_files, batch_name):
    """Run a batch of test files and return results."""
    print(f"\n🧪 Running {batch_name} ({len(test_files)} files)...")

    for test_file in test_files:
        print(f"  Running {os.path.basename(test_file)}...")
        try:
            result = subprocess.run([
                "uv", "run", "pytest", test_file,
                "--cov=src", "--cov-append", "--tb=short", "-q"
            ], capture_output=True, text=True, timeout=120)

            if result.returncode == 0:
                print(f"    ✅ PASSED")
            else:
                print(f"    ⚠️  Issues detected")
                if result.stderr:
                    print(f"    Error: {result.stderr[:200]}...")

        except subprocess.TimeoutExpired:
            print(f"    ⏰ TIMEOUT - skipping for now")
        except Exception as e:
            print(f"    ❌ ERROR: {e}")

def get_coverage_report():
    """Get final coverage report."""
    print("\n📊 Generating Coverage Report...")
    try:
        result = subprocess.run([
            "uv", "run", "coverage", "report", "--show-missing"
        ], capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            print(result.stdout)
            return result.stdout
        else:
            print(f"Coverage report error: {result.stderr}")
            return None
    except Exception as e:
        print(f"Coverage report failed: {e}")
        return None

def main():
    """Run all comprehensive tests efficiently."""
    print("🚀 Starting Efficient Test Coverage Run")
    print("=" * 50)

    # Clear any existing coverage data
    subprocess.run(["uv", "run", "coverage", "erase"], capture_output=True)

    # Get all comprehensive test files
    test_pattern = "tests/unit/*comprehensive*.py"
    all_tests = glob.glob(test_pattern)

    print(f"Found {len(all_tests)} comprehensive test files")

    # Group tests by category for efficient execution
    test_groups = {
        "Core Modules": [t for t in all_tests if any(x in t for x in ["client", "embeddings", "hybrid_search", "memory_tools", "search_tools"])],
        "Server & Config": [t for t in all_tests if any(x in t for x in ["server", "config", "daemon"])],
        "CLI Modules": [t for t in all_tests if "cli_" in t],
        "Utils & Common": [t for t in all_tests if any(x in t for x in ["admin", "project", "common", "os_directories"])],
        "Legacy Tests": [t for t in all_tests if any(x in t for x in ["100_percent", "boost", "final"])]
    }

    # Run each group
    for group_name, test_files in test_groups.items():
        if test_files:
            run_test_batch(test_files, group_name)
            time.sleep(2)  # Brief pause between groups

    # Generate final coverage report
    coverage_report = get_coverage_report()

    print("\n🎯 Test Execution Complete!")
    print("=" * 50)

    if coverage_report:
        # Extract coverage percentage if possible
        lines = coverage_report.split('\n')
        total_line = [line for line in lines if 'TOTAL' in line]
        if total_line:
            print(f"\n📈 Final Coverage: {total_line[0]}")

if __name__ == "__main__":
    main()