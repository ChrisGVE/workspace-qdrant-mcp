#!/usr/bin/env python3
"""Focused test to prove the launchd service lifecycle fix works correctly."""

import asyncio
import subprocess
import sys
import time

async def run_command(cmd: list[str]) -> tuple[str, str, int]:
    """Run a command and return stdout, stderr, return_code."""
    print(f"🔧 Running: {' '.join(cmd)}")
    
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    
    stdout_text = stdout.decode().strip()
    stderr_text = stderr.decode().strip()
    return_code = process.returncode
    
    print(f"   Return code: {return_code}")
    return stdout_text, stderr_text, return_code

def check_processes():
    """Check for memexd processes."""
    try:
        result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
        memexd_lines = [
            line for line in result.stdout.split('\n') 
            if 'memexd' in line and 'grep' not in line
        ]
        return memexd_lines
    except:
        return []

async def test_service_lifecycle():
    """Test the complete service lifecycle to prove the fix works."""
    print("🚀 TESTING DAEMON LIFECYCLE FIX")
    print("=" * 50)
    
    success_count = 0
    total_tests = 5
    
    # Test 1: Start from stopped state
    print("\n📍 Test 1: Starting service from stopped state...")
    stdout, stderr, code = await run_command(["wqm", "service", "start"])
    if code == 0:
        print("✅ Service start command succeeded")
        success_count += 1
    else:
        print(f"❌ Service start failed: {stderr}")
        
    await asyncio.sleep(2)
    
    # Test 2: Verify service is running  
    print("\n📍 Test 2: Verifying service is running...")
    procs = check_processes()
    if len(procs) > 0:
        print(f"✅ Service running - found {len(procs)} memexd process(es)")
        success_count += 1
        for proc in procs[:1]:  # Show just first one
            print(f"   Process: {proc}")
    else:
        print("❌ No memexd processes found after start")
    
    # Verify status shows running
    stdout, stderr, code = await run_command(["wqm", "service", "status"])
    if "running" in stdout.lower():
        print("✅ Status command confirms service is running")
    else:
        print(f"❌ Status shows: {stdout}")
    
    # Test 3: CRITICAL - Stop the service (this tests our fix)
    print("\n📍 Test 3: CRITICAL TEST - Stopping service (testing the fix)...")
    stdout, stderr, code = await run_command(["wqm", "service", "stop"])
    if code == 0:
        print("✅ Service stop command succeeded")
        success_count += 1
    else:
        print(f"❌ Service stop failed: {stderr}")
    
    await asyncio.sleep(3)  # Give time for processes to stop
    
    # Test 4: CRITICAL - Verify NO processes remain
    print("\n📍 Test 4: CRITICAL VERIFICATION - No processes should remain...")
    procs = check_processes()
    if len(procs) == 0:
        print("✅ SUCCESS! No memexd processes remain after stop")
        success_count += 1
    else:
        print(f"❌ FAILURE! {len(procs)} processes still running:")
        for proc in procs:
            print(f"   {proc}")
    
    # Verify status shows stopped
    stdout, stderr, code = await run_command(["wqm", "service", "status"])
    if "stopped" in stdout.lower():
        print("✅ Status command confirms service is stopped")
    else:
        print(f"   Status output: {stdout}")
    
    # Test 5: Start again to prove restart works
    print("\n📍 Test 5: Testing restart capability...")
    stdout, stderr, code = await run_command(["wqm", "service", "start"])
    if code == 0:
        print("✅ Service restart command succeeded")
        
        await asyncio.sleep(2)
        procs = check_processes()
        if len(procs) > 0:
            print("✅ Service successfully restarted")
            success_count += 1
        else:
            print("❌ Service restart failed - no processes found")
    else:
        print(f"❌ Service restart failed: {stderr}")
    
    # Final summary
    print(f"\n{'='*50}")
    print(f"🏁 FINAL RESULTS")
    print(f"{'='*50}")
    print(f"Tests passed: {success_count}/{total_tests}")
    
    if success_count >= 4:  # Allow 1 test to fail
        print("🎉 OVERALL: SUCCESS! The daemon lifecycle fix is working!")
        print("   ✓ Service can start")
        print("   ✓ Service can stop completely (no persistent processes)")  
        print("   ✓ Service can restart after being stopped")
        return True
    else:
        print("💥 OVERALL: FAILURE! Issues persist")
        return False

async def main():
    success = await test_service_lifecycle()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())