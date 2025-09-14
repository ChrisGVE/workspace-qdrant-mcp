#!/usr/bin/env python3
"""
Script to replace over-engineered logging imports with direct loguru usage.
This embraces loguru's simplicity philosophy - no complex wrappers needed!
"""

import os
import re
from pathlib import Path

def replace_logging_imports(file_path: Path) -> bool:
    """Replace complex logging imports with direct loguru usage."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Replace imports from common.logging.loguru_config
        content = re.sub(
            r'from common\.logging\.loguru_config import get_logger.*?\n',
            'from loguru import logger\n',
            content
        )

        # Replace imports from relative paths
        content = re.sub(
            r'from \.\.logging\.loguru_config import get_logger.*?\n',
            'from loguru import logger\n',
            content
        )

        # Replace imports from observability that provide get_logger
        content = re.sub(
            r'from \.\.observability import get_logger.*?\n',
            'from loguru import logger\n',
            content
        )
        content = re.sub(
            r'from common\.observability import get_logger.*?\n',
            'from loguru import logger\n',
            content
        )

        # Replace logger = get_logger(__name__) patterns
        content = re.sub(
            r'logger = get_logger\(__name__\)',
            '# logger imported from loguru',
            content
        )

        # Replace other get_logger calls with direct logger usage
        content = re.sub(
            r'get_logger\(__name__\)',
            'logger',
            content
        )
        content = re.sub(
            r'get_logger\("[^"]*"\)',
            'logger',
            content
        )

        # Replace configure_logging calls - these need manual review
        if 'configure_logging' in content:
            content = content.replace(
                'configure_logging',
                '# TODO: Replace with setup_logging from common.logging.loguru_config'
            )

        # Write back if changed
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
    """Main function to process all Python files."""
    src_root = Path("/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp/src")

    # Find all Python files with logging imports
    python_files = []
    for root, dirs, files in os.walk(src_root):
        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if any(pattern in content for pattern in [
                            'from common.logging.loguru_config import',
                            'from ..logging.loguru_config import',
                            'get_logger',
                            'configure_logging'
                        ]):
                            python_files.append(file_path)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    print(f"Found {len(python_files)} files to update")

    updated_count = 0
    for file_path in python_files:
        if replace_logging_imports(file_path):
            updated_count += 1

    print(f"\nUpdated {updated_count} files")
    print("\nNOTE: Files with 'configure_logging' calls need manual review!")

if __name__ == "__main__":
    main()