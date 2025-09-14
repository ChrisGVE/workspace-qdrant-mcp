#!/usr/bin/env python3
"""
Task 215: Clean up direct logging import in client.py
"""

from pathlib import Path

client_file = Path("/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp/src/python/common/core/client.py")

# Read the file
with open(client_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Check if direct logging import is actually used
if 'logging.' in content:
    print("Direct logging usage found in client.py")
    # Count occurrences
    import re
    direct_logging_calls = re.findall(r'logging\.\w+', content)
    print(f"Direct logging calls: {direct_logging_calls}")
else:
    print("No direct logging usage found")

# Check if we can remove the direct import
if 'import logging' in content and 'logging.' not in content:
    print("Direct logging import can be safely removed")

    # Remove the direct logging import
    lines = content.split('\n')
    cleaned_lines = []

    for line in lines:
        if line.strip() == 'import logging':
            cleaned_lines.append('# Task 215: Direct logging import removed - using unified logging system')
            cleaned_lines.append('# import logging  # MIGRATED')
        else:
            cleaned_lines.append(line)

    # Write back the cleaned version
    with open(client_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(cleaned_lines))

    print("✓ client.py logging import cleaned up")
else:
    print("Direct logging import still needed or already cleaned")