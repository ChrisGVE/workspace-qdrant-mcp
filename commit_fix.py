#!/usr/bin/env python3
"""Commit the service fix."""
import subprocess

def main():
    # Add and commit the fix
    subprocess.run(["git", "add", "src/python/wqm_cli/cli/commands/service.py"])
    subprocess.run([
        "git", "commit", "-m", 
        "fix(service): complete redesign of service management\n\n" +
        "- Replace missing Rust binary with working Python daemon\n" +
        "- Robust error handling for all OS operations\n" +
        "- Simplified, testable architecture\n" +
        "- Proper state management for macOS and Linux\n" +
        "- Fixed all launchctl I/O errors and status detection"
    ])
    
    print("✅ Service fix committed")

if __name__ == "__main__":
    main()