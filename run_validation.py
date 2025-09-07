#!/usr/bin/env python3
"""
Simplified validation runner for bug fixes.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Run the validation
if __name__ == "__main__":
    from validate_bug_fixes import main
    main()