#!/usr/bin/env python3
"""Execute the quick service test and capture output."""
import subprocess
import sys

# Execute the test
result = subprocess.run([sys.executable, "20250111-1647_quick_service_test.py"])
print(f"Test completed with exit code: {result.returncode}")