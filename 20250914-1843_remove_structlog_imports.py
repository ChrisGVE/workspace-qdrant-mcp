#!/usr/bin/env python3
"""
Script to remove remaining structlog imports from the codebase.
Part of final cleanup phase to complete loguru migration.
"""

import os
import re
from pathlib import Path

def update_file(file_path: Path) -> bool:
    """Remove structlog imports from a file. Returns True if changes were made."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Remove import structlog lines
        content = re.sub(r'^import structlog.*\n', '', content, flags=re.MULTILINE)
        content = re.sub(r'^from structlog import.*\n', '', content, flags=re.MULTILINE)

        # Remove structlog usage patterns (replace with loguru equivalent)
        # This is a simplified replacement - manual review may be needed
        patterns_to_replace = [
            (r'structlog\.get_logger\(\)', 'get_logger(__name__)'),
            (r'structlog\.configure\([^)]+\)', '# structlog.configure removed - using loguru'),
            (r'\.bind\(', '.bind('),  # loguru supports bind too
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
    """Main function to remove structlog imports."""
    # Files that had structlog imports according to validation
    files_to_update = [
        "src/python/workspace_qdrant_mcp/stdio_server.py",
        "src/python/workspace_qdrant_mcp/tools/type_search.py",
        "src/python/workspace_qdrant_mcp/tools/dependency_analyzer.py",
        "src/python/workspace_qdrant_mcp/tools/symbol_resolver.py",
        "src/python/workspace_qdrant_mcp/tools/code_search.py",
        "src/python/common/core/logging_config.py",
        "src/python/common/core/lsp_metadata_extractor.py",
        "src/python/common/core/smart_ingestion_router.py",
        "src/python/common/core/lsp_health_monitor.py",
        "src/python/common/core/lsp_client.py",
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