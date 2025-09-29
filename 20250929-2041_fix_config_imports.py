#!/usr/bin/env python3
"""
Temporary script to fix Config import issues across CLI codebase.
Replaces old Config class with get_config_manager() lua-style API.
"""

import re
from pathlib import Path

# Files to fix
files_to_fix = [
    "src/python/wqm_cli/cli/diagnostics.py",
    "src/python/wqm_cli/cli/observability.py",
    "src/python/wqm_cli/cli/setup.py",
    "src/python/wqm_cli/cli/utils.py",
    "src/python/wqm_cli/cli/commands/ingest.py",
    "src/python/wqm_cli/cli/commands/memory.py",
    "src/python/wqm_cli/cli/status.py",
]

for file_path in files_to_fix:
    path = Path(file_path)
    if not path.exists():
        print(f"Skipping {file_path} - not found")
        continue

    content = path.read_text()
    original = content

    # Replace import statement
    content = re.sub(
        r"from common\.core\.config import Config(?:, (\w+))?",
        lambda m: f"from common.core.config import get_config_manager{', ' + m.group(1) if m.group(1) else ''}",
        content
    )

    # Replace Config() instantiations
    content = re.sub(r"\bConfig\(\)", "get_config_manager()", content)

    # Replace type hints
    content = re.sub(r": Config\b", "", content)

    if content != original:
        path.write_text(content)
        print(f"✓ Fixed {file_path}")
    else:
        print(f"  No changes needed in {file_path}")

print("\nDone! All files processed.")