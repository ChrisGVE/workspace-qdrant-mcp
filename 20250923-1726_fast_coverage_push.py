#!/usr/bin/env python3
"""
Fast coverage achievement script - targets specific uncovered modules.
Focuses on lightweight utility and validation modules that can reach 100% quickly.
"""

import subprocess
import sys
from pathlib import Path

def run_focused_tests():
    """Run focused tests on high-value, low-complexity modules."""

    # Target specific modules that should be easy to achieve 100% coverage
    target_modules = [
        "src/python/workspace_qdrant_mcp/utils/config_validator.py",
        "src/python/workspace_qdrant_mcp/utils/project_detection.py",
        "src/python/workspace_qdrant_mcp/utils/project_collection_validator.py",
        "src/python/workspace_qdrant_mcp/core/ssl_config.py",
        "src/python/workspace_qdrant_mcp/core/embeddings.py",
        "src/python/workspace_qdrant_mcp/core/config.py",
    ]

    # Simple test commands that should pass quickly
    test_commands = [
        # Run basic import tests for these modules
        ["python", "-c", "import workspace_qdrant_mcp.utils.config_validator; print('config_validator imported')"],
        ["python", "-c", "import workspace_qdrant_mcp.utils.project_detection; print('project_detection imported')"],
        ["python", "-c", "import workspace_qdrant_mcp.core.ssl_config; print('ssl_config imported')"],
    ]

    for cmd in test_commands:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print(f"✓ {' '.join(cmd)}")
            else:
                print(f"✗ {' '.join(cmd)}: {result.stderr}")
        except subprocess.TimeoutExpired:
            print(f"⏰ {' '.join(cmd)}: Timeout")
        except Exception as e:
            print(f"💥 {' '.join(cmd)}: {e}")

    # Now run pytest on working modules only
    working_tests = [
        "tests/unit/test_core_ssl_config.py",
        "tests/unit/test_config_validator.py",
        "tests/unit/test_project_detection.py",
    ]

    for test_file in working_tests:
        test_path = Path(test_file)
        if test_path.exists():
            print(f"\n🧪 Running {test_file}")
            try:
                cmd = ["uv", "run", "python", "-m", "pytest", str(test_path), "-v", "--tb=short", "--timeout=60"]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                if result.returncode == 0:
                    print(f"✅ {test_file} PASSED")
                else:
                    print(f"❌ {test_file} FAILED:")
                    print(result.stdout[-500:])  # Last 500 chars
            except subprocess.TimeoutExpired:
                print(f"⏰ {test_file}: Test timeout")
            except Exception as e:
                print(f"💥 {test_file}: {e}")
        else:
            print(f"📝 {test_file}: Test file missing - needs creation")

if __name__ == "__main__":
    # Set proper PYTHONPATH for imports
    sys.path.insert(0, str(Path(__file__).parent / "src" / "python"))
    run_focused_tests()