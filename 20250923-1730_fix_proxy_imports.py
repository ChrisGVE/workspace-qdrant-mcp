#!/usr/bin/env python3
"""
Fix proxy import statements in workspace_qdrant_mcp modules.
Many files have incorrect import paths that need to be corrected.
"""

import os
import re
from pathlib import Path

def fix_proxy_imports():
    """Fix all proxy import statements in workspace_qdrant_mcp modules."""

    workspace_dir = Path("src/python/workspace_qdrant_mcp")
    common_dir = Path("src/python/common")

    # Pattern to match the incorrect imports
    import_pattern = re.compile(r'^from python\.common\.(.+) import \*$', re.MULTILINE)

    fixed_files = []
    errors = []

    for py_file in workspace_dir.rglob("*.py"):
        try:
            content = py_file.read_text()

            # Skip files that don't have proxy imports
            if "from python.common." not in content:
                continue

            print(f"🔧 Fixing {py_file}")

            # Fix the import path
            new_content = import_pattern.sub(r'from ...common.\1 import *', content)

            if new_content != content:
                py_file.write_text(new_content)
                fixed_files.append(str(py_file))
                print(f"  ✅ Fixed import in {py_file}")
            else:
                print(f"  ⚠️  No changes needed for {py_file}")

        except Exception as e:
            error_msg = f"Error fixing {py_file}: {e}"
            print(f"  ❌ {error_msg}")
            errors.append(error_msg)

    print(f"\n📊 Summary:")
    print(f"  Fixed files: {len(fixed_files)}")
    print(f"  Errors: {len(errors)}")

    if fixed_files:
        print(f"\n✅ Fixed files:")
        for file_path in fixed_files:
            print(f"  - {file_path}")

    if errors:
        print(f"\n❌ Errors:")
        for error in errors:
            print(f"  - {error}")

    return len(fixed_files), len(errors)

if __name__ == "__main__":
    fixed_count, error_count = fix_proxy_imports()
    print(f"\n🎯 Completed: {fixed_count} files fixed, {error_count} errors")