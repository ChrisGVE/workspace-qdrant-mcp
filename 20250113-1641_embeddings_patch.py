#!/usr/bin/env python3
"""
Task 215: Patch embeddings.py with exact line replacements
"""

embeddings_file = "/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp/src/python/common/core/embeddings.py"

# Read all lines
with open(embeddings_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Patch specific lines
for i, line in enumerate(lines):
    line_num = i + 1

    # Line 45: import logging
    if line_num == 45 and line.strip() == "import logging":
        lines[i] = "# Task 215: Direct logging import replaced with unified system\n"
        lines.insert(i + 1, "# import logging  # MIGRATED\n")
        print(f"✓ Fixed line 45: import logging")

    # Line 55: logger = logging.getLogger(__name__)
    elif line_num == 55 and "logger = logging.getLogger(__name__)" in line:
        lines[i] = "# Task 215: Use unified logging system\n"
        lines.insert(i + 1, "from ..observability.logger import get_logger\n")
        lines.insert(i + 2, "logger = get_logger(__name__)\n")
        print(f"✓ Fixed line 55: logger initialization")

# Write back the file
with open(embeddings_file, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Task 215: embeddings.py patched successfully")