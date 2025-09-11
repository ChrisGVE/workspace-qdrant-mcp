#!/usr/bin/env python3
"""
Manual WQM Service Testing - Execute commands one by one with detailed logging
"""

import subprocess
import time
import os
from datetime import datetime

def run_cmd(cmd, description=""):
    """Execute a single command and log results"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"\n[{timestamp}] {description}")
    print(f"COMMAND: {cmd}")
    print("-" * 50)
    
    try:
        # Change to project directory
        os.chdir("/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp")
        
        # Execute command
        result = subprocess.run(
            cmd.split(), 
            capture_output=True, 
            text=True, 
            timeout=30
        )
        
        print(f"EXIT CODE: {result.returncode}")
        
        if result.stdout.strip():
            print(f"STDOUT:\n{result.stdout}")
            
        if result.stderr.strip():
            print(f"STDERR:\n{result.stderr}")
            
        # Check process status after each command
        ps_result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
        memexd_lines = [line for line in ps_result.stdout.split('\n') if 'memexd' in line and 'grep' not in line]
        
        if memexd_lines:
            print(f"PROCESSES FOUND ({len(memexd_lines)}):")
            for line in memexd_lines:
                print(f"  {line}")
        else:
            print("NO MEMEXD PROCESSES FOUND")
            
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("TIMEOUT: Command exceeded 30 seconds")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def main():
    print("=== WQM SERVICE MANUAL TESTING ===")
    print(f"Start time: {datetime.now()}")
    print(f"Working directory: {os.getcwd()}")
    
    # Test sequence
    test_commands = [
        # Initial state
        ("ps aux | grep memexd", "Check initial processes"),
        ("uv run wqm service status", "Initial status check"),
        
        # Clean slate
        ("uv run wqm service stop", "Stop any existing service"),
        ("uv run wqm service uninstall", "Uninstall any existing service"),
        ("uv run wqm service status", "Status after cleanup"),
        
        # Install sequence
        ("uv run wqm service install", "Install service"),
        ("uv run wqm service status", "Status after install"),
        ("uv run wqm service start", "Start service"),
        ("uv run wqm service status", "Status after start"),
        
        # Control operations
        ("uv run wqm service stop", "Stop service"),
        ("uv run wqm service status", "Status after stop"),
        ("uv run wqm service restart", "Restart service"),
        ("uv run wqm service status", "Status after restart"),
        
        # Error conditions
        ("uv run wqm service start", "Start already running service"),
        ("uv run wqm service install", "Install already installed service"),
        
        # Cleanup
        ("uv run wqm service stop", "Final stop"),
        ("uv run wqm service uninstall", "Final uninstall"),
        ("uv run wqm service status", "Final status check")
    ]
    
    results = []
    
    for cmd, desc in test_commands:
        if cmd.startswith("ps aux"):
            # Handle ps command differently
            try:
                result = subprocess.run(["sh", "-c", cmd], capture_output=True, text=True)
                print(f"\n{desc}")
                print(f"COMMAND: {cmd}")
                print("-" * 50)
                memexd_lines = [line for line in result.stdout.split('\n') if 'memexd' in line and 'grep' not in line]
                if memexd_lines:
                    print(f"MEMEXD PROCESSES ({len(memexd_lines)}):")
                    for line in memexd_lines:
                        print(f"  {line}")
                else:
                    print("NO MEMEXD PROCESSES")
                results.append(True)
            except Exception as e:
                print(f"Process check failed: {e}")
                results.append(False)
        else:
            success = run_cmd(cmd, desc)
            results.append(success)
            
        time.sleep(1)  # Brief pause between commands
    
    # Summary
    total = len(results)
    successes = sum(results)
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total commands: {total}")
    print(f"Successful: {successes}")
    print(f"Failed: {total - successes}")
    print(f"Success rate: {(successes/total*100):.1f}%")
    print(f"End time: {datetime.now()}")

if __name__ == "__main__":
    main()