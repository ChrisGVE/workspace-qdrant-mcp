"""Unit test configuration - isolated from shared fixtures."""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src" / "python"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# NOTE: pytest_plugins cannot be defined in non-root conftest files
# Unit tests should avoid relying on shared fixtures where possible
