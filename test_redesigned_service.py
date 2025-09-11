#!/usr/bin/env python3
import subprocess
import sys

# Test the redesigned service manager
result = subprocess.run([sys.executable, "20250111-1655_service_redesign.py"])
print(f"\nRedesigned service test completed with exit code: {result.returncode}")