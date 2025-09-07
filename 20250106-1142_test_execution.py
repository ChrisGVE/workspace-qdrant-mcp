#!/usr/bin/env python3
"""Execute comprehensive bug fix validation."""

import subprocess
import sys
from pathlib import Path

def run_validation():
    """Run the validation script."""
    script_path = Path(__file__).parent / "20250106-1142_quick_validation.py"
    
    try:
        # Execute the validation script
        result = subprocess.run([
            sys.executable, str(script_path)
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
        return result.returncode
        
    except Exception as e:
        print(f"Error running validation: {e}")
        return 1

if __name__ == "__main__":
    exit_code = run_validation()
    print(f"\nValidation completed with exit code: {exit_code}")
    sys.exit(exit_code)