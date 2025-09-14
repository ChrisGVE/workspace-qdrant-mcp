#!/usr/bin/env python3
"""
Task 215: Apply embeddings.py logging fix directly
"""

embeddings_file = "/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp/src/python/common/core/embeddings.py"

print("Task 215: Applying embeddings.py logging migration...")

# Read the entire file
with open(embeddings_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Apply exact replacements
replacements = [
    (
        'import logging',
        '# Task 215: Direct logging import replaced with unified system\n# import logging  # MIGRATED'
    ),
    (
        'logger = logging.getLogger(__name__)',
        '# Task 215: Use unified logging system\nfrom ..observability.logger import get_logger\nlogger = get_logger(__name__)'
    )
]

# Apply each replacement
migrated_content = content
for old, new in replacements:
    if old in migrated_content:
        migrated_content = migrated_content.replace(old, new)
        print(f"✓ Replaced: {old}")
    else:
        print(f"⚠ Not found: {old}")

# Write the migrated content
with open(embeddings_file, 'w', encoding='utf-8') as f:
    f.write(migrated_content)

print("✓ embeddings.py logging migration completed")

# Verify changes
with open(embeddings_file, 'r', encoding='utf-8') as f:
    verification_content = f.read()

if 'get_logger(__name__)' in verification_content and '# import logging  # MIGRATED' in verification_content:
    print("✓ Migration verification successful")
else:
    print("✗ Migration verification failed")