#!/usr/bin/env python3
"""Script to fix import paths from 'python.common' to 'common'."""

import os
import re
from pathlib import Path

def fix_imports_in_file(file_path: Path):
    """Fix import paths in a single file."""
    try:
        content = file_path.read_text()
        original_content = content

        # Replace import patterns
        content = re.sub(r'from python\.common\.', 'from common.', content)
        content = re.sub(r'import python\.common\.', 'import common.', content)

        if content != original_content:
            file_path.write_text(content)
            print(f"Fixed imports in {file_path}")
            return True
        return False

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Main function to fix all imports."""
    project_root = Path(__file__).parent
    wqm_cli_dir = project_root / "src" / "python" / "wqm_cli"

    if not wqm_cli_dir.exists():
        print(f"Directory {wqm_cli_dir} does not exist")
        return

    fixed_count = 0

    # Find all Python files in wqm_cli directory
    for py_file in wqm_cli_dir.rglob("*.py"):
        if fix_imports_in_file(py_file):
            fixed_count += 1

    print(f"\nFixed imports in {fixed_count} files")

if __name__ == "__main__":
    main()