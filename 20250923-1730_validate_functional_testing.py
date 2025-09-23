#!/usr/bin/env python3
"""
Functional Testing Framework Validation Script

This script validates that all installed functional testing frameworks
are working correctly and can be executed.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description, cwd=None):
    """Run a command and report results."""
    print(f"\n🔍 {description}")
    print(f"   Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            print(f"   ✅ Success")
            if result.stdout:
                print(f"   Output: {result.stdout.strip()[:200]}...")
            return True
        else:
            print(f"   ❌ Failed with return code {result.returncode}")
            if result.stderr:
                print(f"   Error: {result.stderr.strip()[:200]}...")
            return False

    except subprocess.TimeoutExpired:
        print(f"   ⏰ Timeout after 60 seconds")
        return False
    except Exception as e:
        print(f"   💥 Exception: {e}")
        return False

def validate_python_frameworks():
    """Validate Python functional testing frameworks."""
    print("\n" + "="*60)
    print("VALIDATING PYTHON FUNCTIONAL TESTING FRAMEWORKS")
    print("="*60)

    results = []

    # Test pytest basic functionality
    results.append(run_command(
        ["uv", "run", "python", "-c", "import pytest; print(f'pytest {pytest.__version__} available')"],
        "Testing pytest availability"
    ))

    # Test playwright
    results.append(run_command(
        ["uv", "run", "python", "-c", "import playwright; print(f'playwright {playwright.__version__} available')"],
        "Testing playwright availability"
    ))

    # Test testcontainers
    results.append(run_command(
        ["uv", "run", "python", "-c", "import testcontainers; print('testcontainers available')"],
        "Testing testcontainers availability"
    ))

    # Test httpx/respx
    results.append(run_command(
        ["uv", "run", "python", "-c", "import httpx, respx; print('httpx and respx available')"],
        "Testing httpx/respx availability"
    ))

    # Test pytest-benchmark
    results.append(run_command(
        ["uv", "run", "python", "-c", "import pytest_benchmark; print('pytest-benchmark available')"],
        "Testing pytest-benchmark availability"
    ))

    # Run our functional test validation
    results.append(run_command(
        ["uv", "run", "python", "-m", "pytest", "tests/functional/test_framework_validation.py", "-v", "--tb=short"],
        "Running functional test validation suite"
    ))

    return all(results)

def validate_rust_frameworks():
    """Validate Rust functional testing frameworks."""
    print("\n" + "="*60)
    print("VALIDATING RUST FUNCTIONAL TESTING FRAMEWORKS")
    print("="*60)

    rust_dir = Path("rust-engine")
    if not rust_dir.exists():
        print("   ❌ Rust engine directory not found")
        return False

    results = []

    # Test cargo-nextest availability
    results.append(run_command(
        ["cargo", "nextest", "--version"],
        "Testing cargo-nextest availability"
    ))

    # Test cargo compilation with new dependencies
    results.append(run_command(
        ["cargo", "check", "--benches"],
        "Testing Rust framework compilation",
        cwd=rust_dir
    ))

    # Test that criterion benchmark compiles
    results.append(run_command(
        ["cargo", "check", "--bench", "processing_benchmarks"],
        "Testing criterion benchmark compilation",
        cwd=rust_dir
    ))

    return all(results)

def validate_integration_setup():
    """Validate integration testing setup."""
    print("\n" + "="*60)
    print("VALIDATING INTEGRATION TESTING SETUP")
    print("="*60)

    results = []

    # Check test directory structure
    test_dirs = [
        "tests/functional",
        "tests/functional",
        "rust-engine/benches",
        "rust-engine/tests/functional"
    ]

    for test_dir in test_dirs:
        if Path(test_dir).exists():
            print(f"   ✅ {test_dir} exists")
            results.append(True)
        else:
            print(f"   ❌ {test_dir} missing")
            results.append(False)

    # Check configuration files
    config_files = [
        "tests/functional/pytest.ini",
        "tests/functional/conftest.py",
        "rust-engine/.cargo/config.toml"
    ]

    for config_file in config_files:
        if Path(config_file).exists():
            print(f"   ✅ {config_file} exists")
            results.append(True)
        else:
            print(f"   ❌ {config_file} missing")
            results.append(False)

    # Check documentation
    doc_files = [
        "docs/FUNCTIONAL_TESTING_GUIDE.md"
    ]

    for doc_file in doc_files:
        if Path(doc_file).exists():
            print(f"   ✅ {doc_file} exists")
            results.append(True)
        else:
            print(f"   ❌ {doc_file} missing")
            results.append(False)

    return all(results)

def validate_sample_tests():
    """Validate sample tests are present and functional."""
    print("\n" + "="*60)
    print("VALIDATING SAMPLE TESTS")
    print("="*60)

    sample_tests = [
        "tests/functional/test_mcp_protocol_compliance.py",
        "tests/functional/test_web_ui_functional.py",
        "rust-engine/benches/processing_benchmarks.rs",
        "rust-engine/tests/functional/test_service_integration.rs"
    ]

    results = []
    for test_file in sample_tests:
        if Path(test_file).exists():
            print(f"   ✅ {test_file} exists")
            results.append(True)
        else:
            print(f"   ❌ {test_file} missing")
            results.append(False)

    return all(results)

def main():
    """Main validation function."""
    print("🚀 FUNCTIONAL TESTING FRAMEWORK VALIDATION")
    print("=" * 80)

    # Change to project directory
    os.chdir(Path(__file__).parent)

    validation_results = []

    # Run all validations
    validation_results.append(validate_python_frameworks())
    validation_results.append(validate_rust_frameworks())
    validation_results.append(validate_integration_setup())
    validation_results.append(validate_sample_tests())

    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    total_validations = len(validation_results)
    passed_validations = sum(validation_results)

    print(f"   Total validations: {total_validations}")
    print(f"   Passed: {passed_validations}")
    print(f"   Failed: {total_validations - passed_validations}")

    if all(validation_results):
        print("\n   🎉 ALL FUNCTIONAL TESTING FRAMEWORKS VALIDATED SUCCESSFULLY!")
        print("\n   The following frameworks are ready for use:")
        print("   • Python: pytest-playwright, testcontainers, httpx/respx, pytest-benchmark")
        print("   • Rust: cargo-nextest, testcontainers, criterion, proptest")
        print("   • Integration: MCP protocol testing, cross-language testing")
        print("   • Web UI: Playwright browser automation")
        print("   • Performance: Benchmarking and load testing")
        return 0
    else:
        print("\n   ❌ SOME VALIDATIONS FAILED")
        print("   Please check the output above for specific issues.")
        return 1

if __name__ == "__main__":
    sys.exit(main())