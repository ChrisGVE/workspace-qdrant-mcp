#!/usr/bin/env python3
"""
Script to fix observability imports to use loguru instead of the deleted logger.py.
Part of final cleanup phase to complete loguru migration.
"""

import os
import re
from pathlib import Path

def update_file(file_path: Path) -> bool:
    """Update imports in a file. Returns True if changes were made."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Replace imports from .logger with imports from loguru system
        patterns_to_replace = [
            (r'from \.logger import ([^\\n]+)', r'from common.logging.loguru_config import \1'),
            (r'from \.logger import get_logger', r'from common.logging.loguru_config import get_logger'),
            (r'from \.logger import LogContext, get_logger', r'from common.logging.loguru_config import get_logger\nfrom common.logging import LogContext'),
            (r'from \.logger import LogContext', r'from common.logging import LogContext'),
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
    """Main function to update observability imports."""
    files_to_update = [
        "src/python/common/observability/monitoring.py",
        "src/python/common/observability/endpoints.py",
        "src/python/common/observability/health.py",
        "src/python/common/observability/metrics.py",
    ]

    files_updated = []

    for file_path_str in files_to_update:
        file_path = Path(file_path_str)
        if file_path.exists():
            if update_file(file_path):
                files_updated.append(file_path)

    print(f"\nUpdated {len(files_updated)} files:")
    for file_path in files_updated:
        print(f"  - {file_path}")

if __name__ == "__main__":
    main()