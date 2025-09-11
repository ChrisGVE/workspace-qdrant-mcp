#!/usr/bin/env python3
"""
Individual WQM Service Command Tests - Run one command at a time
"""
import subprocess
import os
import sys

def test_single_command(cmd, desc):
    """Test a single command"""
    print(f"\n{'='*50}")
    print(f"TEST: {desc}")
    print(f"CMD:  {cmd}")
    print('='*50)
    
    # Change to correct directory
    os.chdir("/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp")
    
    try:
        # Split command and run
        cmd_parts = cmd.split()
        result = subprocess.run(cmd_parts, capture_output=True, text=True, timeout=30)
        
        print(f"EXIT CODE: {result.returncode}")
        
        if result.stdout.strip():
            print(f"STDOUT:")
            print(result.stdout)
            
        if result.stderr.strip():
            print(f"STDERR:")  
            print(result.stderr)
            
        return result.returncode, result.stdout, result.stderr
        
    except subprocess.TimeoutExpired:
        print("TIMEOUT: Command took too long")
        return -1, "", "Timeout"
    except Exception as e:
        print(f"ERROR: {e}")
        return -2, "", str(e)

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py 'command to test' 'description'")
        print("\nAvailable test commands:")
        print("  'uv run wqm service status' 'Check status'")
        print("  'uv run wqm service install' 'Install service'")
        print("  'uv run wqm service start' 'Start service'")
        print("  'uv run wqm service stop' 'Stop service'")
        sys.exit(1)
    
    command = sys.argv[1]
    description = sys.argv[2] if len(sys.argv) > 2 else "Manual test"
    
    exit_code, stdout, stderr = test_single_command(command, description)
    
    print(f"\nRESULT: {'SUCCESS' if exit_code == 0 else 'FAILED'}")
    print(f"Exit code: {exit_code}")

if __name__ == "__main__":
    main()