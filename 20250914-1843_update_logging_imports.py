#!/usr/bin/env python3
"""
Script to update all common.logging imports to use loguru_config directly.
Part of final cleanup phase to complete loguru migration.
"""

import os
import re
from pathlib import Path

# Base directory for the search
BASE_DIR = Path("src/python")

def update_imports_in_file(file_path: Path) -> bool:
    """Update imports in a single file. Returns True if changes were made."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Pattern to match: from common.logging import ...
        # Replace with: from common.logging.loguru_config import ...
        patterns_to_replace = [
            (r'from common\.logging import get_logger', 'from common.logging.loguru_config import get_logger'),
            (r'from common\.logging import (.+)', r'from common.logging.loguru_config import \1'),
        ]

        for pattern, replacement in patterns_to_replace:
            content = re.sub(pattern, replacement, content)

        # If content changed, write it back
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Updated: {file_path}")
            return True

        return False

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Main function to update all imports."""
    files_updated = []

    # Find all Python files
    for py_file in BASE_DIR.rglob("*.py"):
        if update_imports_in_file(py_file):
            files_updated.append(py_file)

    print(f"\nUpdated {len(files_updated)} files:")
    for file_path in files_updated:
        print(f"  - {file_path}")

if __name__ == "__main__":
    main()