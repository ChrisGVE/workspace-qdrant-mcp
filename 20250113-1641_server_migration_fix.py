#!/usr/bin/env python3
"""
Task 215: Direct server.py migration fix

Replace the direct logging calls in server.py with unified logging system calls.
"""

# Read the current server.py
server_path = "/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp/src/python/workspace_qdrant_mcp/server.py"

with open(server_path, 'r') as f:
    lines = f.readlines()

# Process lines to fix logging issues
fixed_lines = []
in_logging_setup = False

for i, line in enumerate(lines):
    line_num = i + 1

    # Direct logging.getLogger() replacement - lines 139 and 162
    if 'root_logger = logging.getLogger()' in line:
        fixed_lines.append('    # Task 215: Direct logging replaced with unified system\n')
        fixed_lines.append('    from common.logging import get_logger\n')
        fixed_lines.append('    root_logger = get_logger("root").logger  # Access underlying logger for compatibility\n')
        continue

    if 'third_logger = logging.getLogger(logger_name)' in line:
        fixed_lines.append('        # Task 215: Direct logging replaced with unified system\n')
        fixed_lines.append('        third_logger = get_logger(logger_name).logger  # Access underlying logger\n')
        continue

    # Comment out the problematic import at line 40
    if line_num == 40 and 'import logging' in line:
        fixed_lines.append('# Task 215: Direct logging import commented out, using unified system instead\n')
        fixed_lines.append('# import logging  # MIGRATED to unified system\n')
        continue

    # Add unified logging import after imports section (around line 204-210)
    if 'from common.logging import (' in line:
        # This import already exists but needs to be moved earlier
        # Add the get_logger import here
        fixed_lines.append('# Task 215: Import unified logging system early\n')
        fixed_lines.append('from common.logging import get_logger\n')
        fixed_lines.append(line)  # Keep the existing import
        continue

    # Keep all other lines as-is
    fixed_lines.append(line)

# Write the fixed version
with open(server_path, 'w') as f:
    f.writelines(fixed_lines)

print("Task 215: server.py logging migration completed")
print("- Replaced direct logging.getLogger() calls with unified system")
print("- Fixed import order for unified logging system")
print("- Maintained compatibility with existing logging setup")