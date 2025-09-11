#!/usr/bin/env python3
"""Debug script to understand why unload isn't working in Python code."""

import asyncio
import subprocess
import os
from pathlib import Path

async def debug_unload_issue():
    """Debug why the unload command isn't working in our Python code."""
    print("🔍 DEBUGGING UNLOAD ISSUE")
    print("=" * 50)
    
    service_id = "com.workspace-qdrant-mcp.memexd"
    plist_dir = Path.home() / "Library" / "LaunchAgents"
    plist_path = plist_dir / f"{service_id}.plist"
    
    print(f"Service ID: {service_id}")
    print(f"Plist path: {plist_path}")
    print(f"Plist exists: {plist_path.exists()}")
    
    # Check current launchctl status
    print(f"\n📋 Current launchctl status:")
    try:
        result = subprocess.run(
            ["launchctl", "list", service_id], 
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"✅ Service is loaded: {result.stdout.strip()}")
        else:
            print(f"❌ Service not found: {result.stderr.strip()}")
    except Exception as e:
        print(f"❌ Error checking status: {e}")
    
    # Test manual unload
    if plist_path.exists():
        print(f"\n🔧 Testing manual unload with Python subprocess...")
        
        cmd = ["launchctl", "unload", str(plist_path)]
        print(f"Command: {' '.join(cmd)}")
        
        try:
            result = await asyncio.create_subprocess_exec(
                *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            print(f"Return code: {result.returncode}")
            print(f"Stdout: '{stdout.decode().strip()}'")
            print(f"Stderr: '{stderr.decode().strip()}'")
            
            if result.returncode == 0:
                print("✅ Unload command succeeded")
                
                # Wait a moment
                await asyncio.sleep(2)
                
                # Check status after unload
                print(f"\n📋 Status after unload:")
                try:
                    check_result = subprocess.run(
                        ["launchctl", "list", service_id], 
                        capture_output=True, text=True
                    )
                    if check_result.returncode == 0:
                        print(f"❌ Service still loaded: {check_result.stdout.strip()}")
                    else:
                        print(f"✅ Service no longer found: {check_result.stderr.strip()}")
                except Exception as e:
                    print(f"❌ Error checking status: {e}")
                
                # Check processes
                print(f"\n🔍 Process check after unload:")
                try:
                    ps_result = subprocess.run(
                        ["ps", "aux"], 
                        capture_output=True, text=True
                    )
                    memexd_procs = [
                        line for line in ps_result.stdout.split('\n')
                        if 'memexd' in line and 'grep' not in line
                    ]
                    
                    if memexd_procs:
                        print(f"❌ Found {len(memexd_procs)} memexd processes:")
                        for proc in memexd_procs:
                            print(f"  {proc}")
                    else:
                        print("✅ No memexd processes found")
                        
                except Exception as e:
                    print(f"❌ Error checking processes: {e}")
            else:
                print(f"❌ Unload command failed: {stderr.decode().strip()}")
                
        except Exception as e:
            print(f"❌ Exception during unload: {e}")
    
    else:
        print(f"\n❌ Plist file not found at {plist_path}")

if __name__ == "__main__":
    asyncio.run(debug_unload_issue())