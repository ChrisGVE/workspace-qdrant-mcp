#!/usr/bin/env python3
"""
Final script to clean up all remaining structlog references.
Part of final cleanup phase to complete loguru migration.
"""

import os
import re
from pathlib import Path

def update_file(file_path: Path) -> bool:
    """Fix structlog references in a file. Returns True if changes were made."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Replace all structlog.get_logger references
        content = re.sub(r'structlog\.get_logger\(__name__\)', 'get_logger(__name__)', content)
        content = re.sub(r'structlog\.get_logger\(([^)]+)\)', r'get_logger(\1)', content)

        # Remove or replace other structlog references
        content = re.sub(r'structlog\.contextvars\.bind_contextvars\([^)]+\)', '# structlog context binding removed', content)
        content = re.sub(r'structlog\.contextvars\.clear_contextvars\(\)', '# structlog context clearing removed', content)

        # Add get_logger import if structlog.get_logger was used and get_logger import is missing
        if 'get_logger(' in content and 'from' in content and 'logging' in content:
            # Check if get_logger import is already present
            if 'from common.logging.loguru_config import get_logger' not in content:
                if 'from ..logging.loguru_config import get_logger' not in content:
                    # Add import at the top after existing imports
                    lines = content.split('\\n')
                    import_inserted = False
                    for i, line in enumerate(lines):
                        if line.startswith('from common.') and 'import' in line and not import_inserted:
                            lines.insert(i, 'from common.logging.loguru_config import get_logger')
                            import_inserted = True
                            break
                        elif line.startswith('from ..') and 'import' in line and not import_inserted:
                            lines.insert(i, 'from ..logging.loguru_config import get_logger')
                            import_inserted = True
                            break

                    if not import_inserted:
                        # Find a good place to insert the import
                        for i, line in enumerate(lines):
                            if line.strip() == '' and i > 0 and not lines[i-1].startswith(('import ', 'from ')):
                                lines.insert(i, 'from common.logging.loguru_config import get_logger')
                                break

                    content = '\\n'.join(lines)

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
    """Main function to clean up all structlog references."""

    # First, find all files with structlog references
    result = os.popen('grep -r "structlog" src/python/ --include="*.py" -l').read()
    files_with_structlog = [line.strip() for line in result.split('\\n') if line.strip()]

    print(f"Found {len(files_with_structlog)} files with structlog references:")
    for f in files_with_structlog:
        print(f"  - {f}")

    files_updated = []

    for file_path_str in files_with_structlog:
        if file_path_str:  # Skip empty strings
            file_path = Path(file_path_str)
            if file_path.exists():
                if update_file(file_path):
                    files_updated.append(file_path)

    print(f"\\nUpdated {len(files_updated)} files:")
    for file_path in files_updated:
        print(f"  - {file_path}")

if __name__ == "__main__":
    main()