#!/usr/bin/env python3
import subprocess
import sys

# Run the daemon check
result = subprocess.run([sys.executable, "20250111-1651_check_daemon_binary.py"])
print(f"Check completed with exit code: {result.returncode}")