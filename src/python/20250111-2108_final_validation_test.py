#!/usr/bin/env python3
"""Final validation test to prove the launchd service lifecycle fix works."""

import asyncio
import subprocess
import sys

async def run_command(cmd: list[str]) -> tuple[str, str, int]:
    """Run a command and return stdout, stderr, return_code."""
    print(f"🔧 {' '.join(cmd)}")
    
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    return stdout.decode().strip(), stderr.decode().strip(), process.returncode

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

def check_launchctl_loaded():
    """Check if service is loaded in launchctl."""
    try:
        result = subprocess.run(
            ["launchctl", "list", "com.workspace-qdrant-mcp.memexd"], 
            capture_output=True, text=True
        )
        return result.returncode == 0
    except:
        return False

async def final_validation_test():
    """Run comprehensive validation test."""
    print("🎯 FINAL VALIDATION: Daemon Lifecycle Fix")
    print("=" * 60)
    
    tests = []
    
    # Test 1: Clean initial state
    print("\n📍 Test 1: Verify clean initial state")
    initial_procs = check_processes()
    initial_loaded = check_launchctl_loaded()
    
    if len(initial_procs) == 0 and not initial_loaded:
        tests.append(("Clean initial state", True))
        print("✅ PASS: Clean initial state confirmed")
    else:
        tests.append(("Clean initial state", False))
        print(f"❌ FAIL: Initial state not clean - processes: {len(initial_procs)}, loaded: {initial_loaded}")
    
    # Test 2: Service can start
    print("\n📍 Test 2: Start service")
    stdout, stderr, code = await run_command(["wqm", "service", "start"])
    if code == 0:
        await asyncio.sleep(2)
        start_procs = check_processes()
        start_loaded = check_launchctl_loaded()
        
        if len(start_procs) > 0 and start_loaded:
            tests.append(("Service start", True))
            print(f"✅ PASS: Service started - {len(start_procs)} process(es), loaded: {start_loaded}")
        else:
            tests.append(("Service start", False))
            print(f"❌ FAIL: Service not properly started - processes: {len(start_procs)}, loaded: {start_loaded}")
    else:
        tests.append(("Service start", False))
        print(f"❌ FAIL: Start command failed - code: {code}, error: {stderr}")
    
    # Test 3: CRITICAL - Service can stop completely
    print("\n📍 Test 3: CRITICAL - Stop service completely")
    stdout, stderr, code = await run_command(["wqm", "service", "stop"])
    if code == 0:
        await asyncio.sleep(3)  # Give time for complete shutdown
        stop_procs = check_processes()
        stop_loaded = check_launchctl_loaded()
        
        if len(stop_procs) == 0 and not stop_loaded:
            tests.append(("Complete service stop", True))
            print("✅ PASS: Service completely stopped - no processes, not loaded")
        else:
            tests.append(("Complete service stop", False))
            print(f"❌ FAIL: Service not completely stopped - processes: {len(stop_procs)}, loaded: {stop_loaded}")
            for proc in stop_procs:
                print(f"   Remaining: {proc}")
    else:
        tests.append(("Complete service stop", False))
        print(f"❌ FAIL: Stop command failed - code: {code}, error: {stderr}")
    
    # Test 4: Service can start again after stop
    print("\n📍 Test 4: Restart after complete stop")
    stdout, stderr, code = await run_command(["wqm", "service", "start"])
    if code == 0:
        await asyncio.sleep(2)
        restart_procs = check_processes()
        restart_loaded = check_launchctl_loaded()
        
        if len(restart_procs) > 0 and restart_loaded:
            tests.append(("Service restart", True))
            print(f"✅ PASS: Service restarted successfully - {len(restart_procs)} process(es)")
        else:
            tests.append(("Service restart", False))
            print(f"❌ FAIL: Service restart failed - processes: {len(restart_procs)}, loaded: {restart_loaded}")
    else:
        tests.append(("Service restart", False))
        print(f"❌ FAIL: Restart command failed - code: {code}, error: {stderr}")
    
    # Test 5: Second stop to prove repeatability
    print("\n📍 Test 5: Second stop (prove repeatability)")
    stdout, stderr, code = await run_command(["wqm", "service", "stop"])
    if code == 0:
        await asyncio.sleep(3)
        final_procs = check_processes()
        final_loaded = check_launchctl_loaded()
        
        if len(final_procs) == 0 and not final_loaded:
            tests.append(("Repeatable stop", True))
            print("✅ PASS: Second stop successful - proves repeatability")
        else:
            tests.append(("Repeatable stop", False))
            print(f"❌ FAIL: Second stop failed - processes: {len(final_procs)}, loaded: {final_loaded}")
    else:
        tests.append(("Repeatable stop", False))
        print(f"❌ FAIL: Second stop command failed - code: {code}, error: {stderr}")
    
    # Final summary
    passed = sum(1 for _, result in tests if result)
    total = len(tests)
    
    print(f"\n{'='*60}")
    print(f"🏁 FINAL VALIDATION RESULTS")
    print(f"{'='*60}")
    
    for test_name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nScore: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 OVERALL SUCCESS!")
        print("The launchd service lifecycle fix is working perfectly!")
        print("Key achievements:")
        print("  ✓ Service starts correctly")
        print("  ✓ Service stops completely (no persistent processes)")  
        print("  ✓ Service unloads from launchctl properly")
        print("  ✓ Service can restart after being stopped")
        print("  ✓ Stop operation is repeatable")
        return True
    else:
        print(f"\n💥 VALIDATION FAILED!")
        print(f"Only {passed}/{total} tests passed. Issues remain.")
        return False

async def main():
    success = await final_validation_test()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())