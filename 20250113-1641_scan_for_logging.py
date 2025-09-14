#!/usr/bin/env python3
"""
Task 215: Quick scan for remaining files with direct logging usage.
"""

import os
import re
from pathlib import Path

BASE_DIR = Path("/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp/src/python")

def scan_files():
    """Quick scan for files with logging issues."""

    patterns = {
        "direct_logging": r"logging\.getLogger\s*\(",
        "import_logging": r"^import logging$",
        "from_logging": r"^from logging import",
        "stderr_write": r"sys\.stderr\.write|sys\.__stderr__\.write",
        "traceback_print": r"traceback\.print_exc\s*\(",
    }

    print("Task 215: Quick scan for remaining logging migration targets")
    print("=" * 60)

    total_issues = 0

    for py_file in BASE_DIR.rglob("*.py"):
        if "__pycache__" in str(py_file) or ".pytest_cache" in str(py_file):
            continue

        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()

            file_issues = []
            for pattern_name, pattern in patterns.items():
                matches = re.findall(pattern, content, re.MULTILINE)
                if matches:
                    file_issues.append((pattern_name, len(matches)))

            if file_issues:
                rel_path = py_file.relative_to(BASE_DIR)
                issue_count = sum(count for _, count in file_issues)
                total_issues += issue_count

                print(f"📁 {rel_path} ({issue_count} issues)")
                for pattern_name, count in file_issues:
                    print(f"   - {pattern_name}: {count}")

        except Exception as e:
            print(f"Error reading {py_file}: {e}")

    print(f"\n" + "=" * 60)
    print(f"TOTAL REMAINING ISSUES: {total_issues}")

    # Prioritize common/core and workspace_qdrant_mcp modules
    priority_patterns = [
        "common/core/*.py",
        "workspace_qdrant_mcp/*.py",
        "wqm_cli/*.py"
    ]

    print(f"\nPRIORITY MIGRATION TARGETS:")
    for pattern in priority_patterns:
        print(f"- {pattern}")

if __name__ == "__main__":
    scan_files()