#!/usr/bin/env python3
"""
Task 215: Comprehensive Logging Migration Script

This script migrates all direct logging calls to the unified logging system
for MCP stdio compliance.

OBJECTIVE: Replace 113+ direct logging.getLogger() calls and sys.stderr writes
with unified logging system.

MIGRATION STEPS:
1. Find all files with direct logging.getLogger() usage
2. Replace with 'from common.observability.logger import get_logger; logger = get_logger(__name__)'
3. Remove direct sys.stderr.write() and sys.__stderr__.write() calls
4. Replace debug prints with logger.debug() calls
5. Replace traceback.print_exc() with logger.exception()
6. Add stdio mode detection to all debug outputs
"""

import os
import re
import glob
from pathlib import Path
from typing import List, Tuple, Dict

# Base directory for the project
BASE_DIR = Path("/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp/src/python")

def find_python_files() -> List[Path]:
    """Find all Python files in the project that may need migration."""
    patterns = [
        "**/*.py",
    ]

    files = []
    for pattern in patterns:
        files.extend(BASE_DIR.glob(pattern))

    # Filter out __pycache__ and test files for now
    filtered_files = []
    for file in files:
        if "__pycache__" in str(file):
            continue
        if "/.pytest_cache/" in str(file):
            continue
        filtered_files.append(file)

    return sorted(filtered_files)

def analyze_file_logging_usage(file_path: Path) -> Dict[str, List[int]]:
    """Analyze a file for various logging patterns that need migration."""

    patterns = {
        "logging.getLogger": r"logging\.getLogger\s*\(",
        "sys.stderr.write": r"sys\.stderr\.write\s*\(",
        "sys.__stderr__.write": r"sys\.__stderr__\.write\s*\(",
        "traceback.print_exc": r"traceback\.print_exc\s*\(",
        "print_debug": r"print\s*\([^)]*debug[^)]*\)",  # Debug print statements
        "import_logging": r"^import logging$|^from logging import",
        "import_sys": r"^import sys$|^from sys import",
    }

    results = {key: [] for key in patterns.keys()}

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line_num, line in enumerate(lines, 1):
            for pattern_name, pattern in patterns.items():
                if re.search(pattern, line, re.IGNORECASE):
                    results[pattern_name].append(line_num)

    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")

    return results

def generate_migration_report():
    """Generate a comprehensive report of all files needing migration."""

    print("Task 215: Logging Migration Analysis")
    print("=" * 50)

    python_files = find_python_files()
    print(f"Found {len(python_files)} Python files to analyze")
    print()

    files_needing_migration = []
    total_issues = 0

    for file_path in python_files:
        analysis = analyze_file_logging_usage(file_path)

        # Check if file has any logging issues
        has_issues = any(issues for issues in analysis.values())

        if has_issues:
            files_needing_migration.append((file_path, analysis))
            file_issues = sum(len(issues) for issues in analysis.values())
            total_issues += file_issues

            print(f"📁 {file_path.relative_to(BASE_DIR)}")
            for pattern_name, line_numbers in analysis.items():
                if line_numbers:
                    print(f"   - {pattern_name}: lines {line_numbers}")
            print()

    print("MIGRATION SUMMARY")
    print("=" * 50)
    print(f"Files needing migration: {len(files_needing_migration)}")
    print(f"Total logging issues found: {total_issues}")
    print()

    return files_needing_migration

def create_migration_template():
    """Create the standard migration template for logging."""

    template = '''
# MIGRATION TEMPLATE for Task 215

# OLD PATTERN:
# import logging
# logger = logging.getLogger(__name__)

# NEW PATTERN:
from common.observability.logger import get_logger
logger = get_logger(__name__)

# OLD PATTERNS TO REPLACE:
# sys.stderr.write("message") -> logger.error("message") (with stdio detection)
# sys.__stderr__.write("message") -> logger.error("message") (with stdio detection)
# traceback.print_exc() -> logger.exception("Error occurred")
# print("debug:", msg) -> logger.debug(msg)

# STDIO MODE DETECTION PATTERN:
import os
if not os.getenv("WQM_STDIO_MODE", "").lower() == "true":
    logger.debug("This will only log when not in stdio mode")
'''

    with open(BASE_DIR.parent.parent / "20250113-1641_migration_template.txt", "w") as f:
        f.write(template)

    print("Migration template created: 20250113-1641_migration_template.txt")

if __name__ == "__main__":
    print("Starting Task 215: Logging Migration Analysis")

    # Create backup info
    print("\n1. Create backup branch first:")
    print("   bash 20250113-1641_backup_before_logging_migration.sh")
    print()

    # Run analysis
    migration_files = generate_migration_report()

    # Create template
    create_migration_template()

    print("\nNext Steps:")
    print("1. Review the files listed above")
    print("2. Start migration with high-priority files (server.py, stdio_server.py)")
    print("3. Test each module after migration")
    print("4. Commit atomically by functional area")
    print("5. Validate complete system integration")