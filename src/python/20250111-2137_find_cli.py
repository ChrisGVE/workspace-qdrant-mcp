#!/usr/bin/env python3
import os
import subprocess

def find_files_with_service():
    """Find files that might contain service command implementation"""
    project_root = "/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp"
    
    print("Looking for CLI and service-related files...")
    
    # Use find command to locate Python files
    try:
        result = subprocess.run([
            "find", project_root, "-name", "*.py", "-type", "f"
        ], capture_output=True, text=True)
        
        python_files = result.stdout.strip().split('\n')
        
        service_files = []
        cli_files = []
        
        for file in python_files:
            if 'cli' in file.lower():
                cli_files.append(file)
            
            # Check if file contains 'service' in content
            try:
                with open(file, 'r') as f:
                    content = f.read()
                    if 'service' in content.lower() and ('install' in content or 'start' in content):
                        service_files.append(file)
            except:
                pass
                
        print("\nCLI-related files:")
        for f in cli_files:
            print(f"  {f}")
            
        print("\nService-related files:")
        for f in service_files:
            print(f"  {f}")
            
        return cli_files, service_files
        
    except Exception as e:
        print(f"Error: {e}")
        return [], []

if __name__ == "__main__":
    find_files_with_service()