"""Unit test configuration - isolated from shared fixtures."""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src" / "python"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Override pytest_plugins to prevent loading shared fixtures
pytest_plugins = []
