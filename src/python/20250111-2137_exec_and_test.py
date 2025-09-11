#!/usr/bin/env python3
"""
Execute structure exploration and then run basic WQM tests
"""
import subprocess
import os

def main():
    project_root = "/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp"
    os.chdir(project_root)
    
    print("WQM SERVICE TESTING - Step by Step")
    print("="*50)
    
    # Step 1: List project structure
    print("\n1. PROJECT STRUCTURE:")
    try:
        result = subprocess.run(["python3", "src/python/20250111-2137_ls_project.py"], 
                              capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Structure listing errors:", result.stderr)
    except Exception as e:
        print(f"Structure listing failed: {e}")
    
    # Step 2: Find CLI files
    print("\n2. FINDING CLI/SERVICE FILES:")
    try:
        result = subprocess.run(["python3", "src/python/20250111-2137_find_cli.py"], 
                              capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("File search errors:", result.stderr)
    except Exception as e:
        print(f"File search failed: {e}")
    
    # Step 3: Test basic wqm commands
    print("\n3. BASIC WQM COMMAND TESTS:")
    
    commands_to_test = [
        (["uv", "run", "wqm", "--help"], "WQM help"),
        (["uv", "run", "wqm", "service", "--help"], "Service help"),
        (["uv", "run", "wqm", "service", "status"], "Service status")
    ]
    
    for cmd, desc in commands_to_test:
        print(f"\nTesting: {desc}")
        print(f"Command: {' '.join(cmd)}")
        print("-" * 30)
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            print(f"Exit code: {result.returncode}")
            
            if result.stdout.strip():
                print("STDOUT:")
                print(result.stdout[:500])
                if len(result.stdout) > 500:
                    print("... (truncated)")
                    
            if result.stderr.strip():
                print("STDERR:")
                print(result.stderr[:300])
                if len(result.stderr) > 300:
                    print("... (truncated)")
                    
        except subprocess.TimeoutExpired:
            print("TIMEOUT: Command took too long")
        except Exception as e:
            print(f"ERROR: {e}")
    
    print(f"\n{'='*50}")
    print("Basic exploration and testing complete.")

if __name__ == "__main__":
    main()