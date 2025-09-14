#!/usr/bin/env python3
"""
Task 215: Fix embeddings.py logging

Replace lines 45 and 55 in embeddings.py
"""

import re

embeddings_file = "/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp/src/python/common/core/embeddings.py"

# Read the file
with open(embeddings_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace direct logging import and logger initialization
replacements = [
    # Replace import logging
    (r'^import logging$', '# Task 215: Direct logging import replaced with unified system\n# import logging  # MIGRATED'),

    # Replace logger = logging.getLogger(__name__)
    (r'^logger = logging\.getLogger\(__name__\)$', '# Task 215: Use unified logging system\nfrom ..observability.logger import get_logger\nlogger = get_logger(__name__)'),
]

# Apply replacements
modified_content = content
for pattern, replacement in replacements:
    modified_content = re.sub(pattern, replacement, modified_content, flags=re.MULTILINE)

# Write back the file
with open(embeddings_file, 'w', encoding='utf-8') as f:
    f.write(modified_content)

print("Task 215: embeddings.py logging migration completed")
print("- Replaced direct logging import with unified system")
print("- Replaced logging.getLogger(__name__) with get_logger(__name__)")