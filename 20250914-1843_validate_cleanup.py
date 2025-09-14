#!/usr/bin/env python3
"""
Script to validate the loguru migration cleanup was successful.
Tests import paths, server startup, and CLI functionality.
"""

import sys
import subprocess
import importlib
from pathlib import Path

def test_imports():
    """Test that all imports work correctly after cleanup."""
    print("Testing imports...")

    test_modules = [
        "common.logging.loguru_config",
        "workspace_qdrant_mcp.server",
        "workspace_qdrant_mcp.stdio_server",
        "workspace_qdrant_mcp.tools.memory",
        "workspace_qdrant_mcp.tools.search",
        "common.core.config",
        "common.core.client",
    ]

    failed_imports = []

    for module in test_modules:
        try:
            importlib.import_module(module)
            print(f"  ✓ {module}")
        except Exception as e:
            print(f"  ✗ {module}: {e}")
            failed_imports.append((module, str(e)))

    return failed_imports

def test_server_startup():
    """Test that the server can start without import errors."""
    print("\nTesting server startup...")

    # Test import of main server module
    try:
        # Set environment to prevent actual server startup
        env = {"WQM_STDIO_MODE": "true", "PYTHONPATH": "src/python"}

        # Test server import
        result = subprocess.run([
            sys.executable, "-c",
            "import sys; sys.path.insert(0, 'src/python'); "
            "from workspace_qdrant_mcp import server; print('Server import: OK')"
        ], capture_output=True, text=True, timeout=10, env=env)

        if result.returncode == 0:
            print("  ✓ Server module imports successfully")
            return True
        else:
            print(f"  ✗ Server import failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"  ✗ Server startup test failed: {e}")
        return False

def test_cli_startup():
    """Test that the CLI can start without import errors."""
    print("\nTesting CLI startup...")

    try:
        # Test CLI help command
        result = subprocess.run([
            sys.executable, "-c",
            "import sys; sys.path.insert(0, 'src/python'); "
            "from wqm_cli.cli_wrapper import main; print('CLI import: OK')"
        ], capture_output=True, text=True, timeout=10)

        if result.returncode == 0:
            print("  ✓ CLI module imports successfully")
            return True
        else:
            print(f"  ✗ CLI import failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"  ✗ CLI startup test failed: {e}")
        return False

def check_remaining_old_imports():
    """Check for any remaining imports of old logging systems."""
    print("\nChecking for remaining old logging imports...")

    old_patterns = [
        "from common.logging.core import",
        "from common.logging.config import",
        "from common.logging.handlers import",
        "from common.logging.formatters import",
        "from common.observability.logger import",
        "import structlog",
        "from structlog import"
    ]

    src_dir = Path("src/python")
    issues = []

    for py_file in src_dir.rglob("*.py"):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()

            for pattern in old_patterns:
                if pattern in content:
                    issues.append(f"{py_file}: {pattern}")
        except Exception as e:
            print(f"  Warning: Could not read {py_file}: {e}")

    if issues:
        print("  ✗ Found remaining old logging imports:")
        for issue in issues:
            print(f"    {issue}")
    else:
        print("  ✓ No old logging imports found")

    return issues

def verify_files_removed():
    """Verify that old logging files were properly removed."""
    print("\nVerifying file removal...")

    files_that_should_not_exist = [
        "src/python/common/logging/config.py",
        "src/python/common/logging/formatters.py",
        "src/python/common/logging/handlers.py",
        "src/python/common/logging/core.py",
        "src/python/common/logging/__init__.py",
        "src/python/common/logging/migration.py",
        "src/python/common/observability/logger.py",
    ]

    still_exist = []
    for file_path in files_that_should_not_exist:
        if Path(file_path).exists():
            still_exist.append(file_path)

    if still_exist:
        print("  ✗ Old files still exist:")
        for file_path in still_exist:
            print(f"    {file_path}")
    else:
        print("  ✓ All old logging files properly removed")

    # Verify loguru_config.py still exists
    if Path("src/python/common/logging/loguru_config.py").exists():
        print("  ✓ loguru_config.py preserved")
    else:
        print("  ✗ loguru_config.py missing!")
        still_exist.append("loguru_config.py missing")

    return still_exist

def main():
    """Run all validation tests."""
    print("=== Loguru Migration Cleanup Validation ===\n")

    failed_imports = test_imports()
    server_ok = test_server_startup()
    cli_ok = test_cli_startup()
    old_imports = check_remaining_old_imports()
    missing_removals = verify_files_removed()

    print("\n=== SUMMARY ===")

    success = True

    if failed_imports:
        print(f"✗ {len(failed_imports)} import failures")
        success = False
    else:
        print("✓ All imports successful")

    if not server_ok:
        print("✗ Server startup failed")
        success = False
    else:
        print("✓ Server startup successful")

    if not cli_ok:
        print("✗ CLI startup failed")
        success = False
    else:
        print("✓ CLI startup successful")

    if old_imports:
        print(f"✗ {len(old_imports)} old logging imports still present")
        success = False
    else:
        print("✓ No old logging imports found")

    if missing_removals:
        print(f"✗ {len(missing_removals)} files not properly removed")
        success = False
    else:
        print("✓ All old files properly removed")

    if success:
        print("\n🎉 CLEANUP VALIDATION SUCCESSFUL!")
        print("Loguru migration cleanup is complete and working correctly.")
        return 0
    else:
        print("\n❌ CLEANUP VALIDATION FAILED!")
        print("Issues found - see details above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())