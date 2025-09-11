#!/usr/bin/env python3
"""Debug script to understand exactly what happens during the stop command."""

import asyncio
import subprocess
import os
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

def check_memexd_processes():
    """Check for running memexd processes."""
    try:
        result = subprocess.run(
            ["ps", "aux"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        memexd_lines = [
            line for line in result.stdout.split('\n') 
            if 'memexd' in line and 'grep' not in line
        ]
        return memexd_lines
    except Exception as e:
        print(f"Error checking processes: {e}")
        return []

async def debug_stop_sequence():
    """Debug the stop sequence step by step."""
    print("=== DEBUGGING STOP SEQUENCE ===")
    
    # Check initial state
    print("\n1. Initial state:")
    initial_procs = check_memexd_processes()
    print(f"Found {len(initial_procs)} memexd processes:")
    for proc in initial_procs:
        print(f"  {proc}")
    
    # Test the unload command directly
    print("\n2. Testing manual unload command:")
    plist_path = "/Users/chris/Library/LaunchAgents/com.workspace-qdrant-mcp.memexd.plist"
    
    if os.path.exists(plist_path):
        print(f"plist exists at: {plist_path}")
        
        # Run unload command
        print("Running: launchctl unload " + plist_path)
        result = await asyncio.create_subprocess_exec(
            "launchctl", "unload", plist_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = await result.communicate()
        print(f"Return code: {result.returncode}")
        print(f"Stdout: {stdout.decode()}")  
        print(f"Stderr: {stderr.decode()}")
        
        # Wait a moment
        print("Waiting 3 seconds...")
        await asyncio.sleep(3)
        
        # Check processes after unload
        print("\n3. State after unload:")
        after_unload = check_memexd_processes()
        print(f"Found {len(after_unload)} memexd processes:")
        for proc in after_unload:
            print(f"  {proc}")
        
        # Test if unloaded service shows in launchctl list
        print("\n4. Check launchctl list:")
        list_result = await asyncio.create_subprocess_exec(
            "launchctl", "list", "com.workspace-qdrant-mcp.memexd",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        list_stdout, list_stderr = await list_result.communicate()
        print(f"launchctl list return code: {list_result.returncode}")
        print(f"launchctl list stdout: {list_stdout.decode()}")
        print(f"launchctl list stderr: {list_stderr.decode()}")
        
        print("\n5. Testing wqm service stop command:")
        stop_result = await asyncio.create_subprocess_exec(
            "wqm", "service", "stop",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stop_stdout, stop_stderr = await stop_result.communicate()
        print(f"wqm stop return code: {stop_result.returncode}")
        print(f"wqm stop stdout: {stop_stdout.decode()}")
        print(f"wqm stop stderr: {stop_stderr.decode()}")
        
        print("Waiting 3 seconds...")
        await asyncio.sleep(3)
        
        print("\n6. Final state after wqm stop:")
        final_procs = check_memexd_processes()
        print(f"Found {len(final_procs)} memexd processes:")
        for proc in final_procs:
            print(f"  {proc}")
            
        if len(final_procs) == 0:
            print("🎉 SUCCESS: All processes stopped!")
        else:
            print("❌ FAILURE: Processes still running")
    else:
        print(f"plist not found at: {plist_path}")

if __name__ == "__main__":
    asyncio.run(debug_stop_sequence())