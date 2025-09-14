#!/usr/bin/env python3
"""
Task 215: Find all files with direct logging usage that need migration.
"""

import os
import re
from pathlib import Path

BASE_DIR = Path("/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp/src/python")

def find_files_with_logging():
    """Find Python files with direct logging usage."""

    patterns_to_find = [
        r'logging\.getLogger\s*\(',
        r'import logging$',
        r'from logging import',
        r'sys\.stderr\.write',
        r'sys\.__stderr__\.write',
        r'traceback\.print_exc',
    ]

    files_with_issues = {}

    # Find all Python files
    for py_file in BASE_DIR.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue

        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')

            file_issues = []

            for i, line in enumerate(lines):
                line_num = i + 1
                for pattern in patterns_to_find:
                    if re.search(pattern, line):
                        file_issues.append((line_num, pattern, line.strip()))

            if file_issues:
                rel_path = py_file.relative_to(BASE_DIR)
                files_with_issues[str(rel_path)] = file_issues

        except Exception as e:
            print(f"Error reading {py_file}: {e}")

    return files_with_issues

def main():
    print("Task 215: Finding files with direct logging usage")
    print("=" * 60)

    files_with_issues = find_files_with_logging()

    total_issues = 0
    for file_path, issues in files_with_issues.items():
        print(f"\n📁 {file_path}")
        for line_num, pattern, line_content in issues:
            print(f"   Line {line_num:3d}: {pattern}")
            print(f"             {line_content[:80]}")
            total_issues += 1

    print(f"\n" + "=" * 60)
    print(f"SUMMARY:")
    print(f"Files with logging issues: {len(files_with_issues)}")
    print(f"Total logging issues found: {total_issues}")

    # Priority files for migration
    priority_files = [
        "workspace_qdrant_mcp/server.py",
        "workspace_qdrant_mcp/stdio_server.py",
        "common/core/client.py",
        "common/core/hybrid_search.py",
        "wqm_cli/cli_wrapper.py"
    ]

    print(f"\nPRIORITY FILES FOR MIGRATION:")
    for priority_file in priority_files:
        if priority_file in files_with_issues:
            issue_count = len(files_with_issues[priority_file])
            print(f"✓ {priority_file} ({issue_count} issues)")
        else:
            print(f"✓ {priority_file} (clean)")

if __name__ == "__main__":
    main()