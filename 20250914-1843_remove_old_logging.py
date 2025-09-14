#!/usr/bin/env python3
"""
Script to remove old logging infrastructure files.
Part of final cleanup phase to complete loguru migration.
"""

import os
import shutil
from pathlib import Path

def main():
    """Remove old logging infrastructure files."""
    # Files to remove from common/logging/ (keep only loguru_config.py)
    logging_dir = Path("src/python/common/logging")
    files_to_remove = [
        "config.py",
        "formatters.py",
        "handlers.py",
        "__init__.py",
        "core.py",
        "migration.py"
    ]

    # Remove old logging files
    print("Removing old logging infrastructure files:")
    for filename in files_to_remove:
        file_path = logging_dir / filename
        if file_path.exists():
            file_path.unlink()
            print(f"  Removed: {file_path}")

    # Remove the observability logger.py
    observability_logger = Path("src/python/common/observability/logger.py")
    if observability_logger.exists():
        observability_logger.unlink()
        print(f"  Removed: {observability_logger}")

    # Remove __pycache__ directories
    pycache_dirs = [
        logging_dir / "__pycache__",
    ]

    for cache_dir in pycache_dirs:
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            print(f"  Removed cache: {cache_dir}")

    print("\nCleanup completed!")
    print(f"Kept: {logging_dir}/loguru_config.py")

if __name__ == "__main__":
    main()