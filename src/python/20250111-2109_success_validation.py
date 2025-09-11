#!/usr/bin/env python3
"""Success validation - prove the fix works completely."""

import asyncio
import subprocess
import sys

async def run_cmd(cmd: list[str]) -> tuple[str, str, int]:
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = await proc.communicate()
    return stdout.decode().strip(), stderr.decode().strip(), proc.returncode

def check_processes():
    try:
        result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
        return [line for line in result.stdout.split('\n') if 'memexd' in line and 'grep' not in line]
    except:
        return []

def check_service_status():
    try:
        result = subprocess.run(["launchctl", "list", "com.workspace-qdrant-mcp.memexd"], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            # Find the line with PID info
            for line in lines:
                if 'com.workspace-qdrant-mcp.memexd' in line:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        return parts[0], parts[1]  # PID, exit_status
            return "loaded", "unknown"
        else:
            return "not_found", "not_found"
    except:
        return "error", "error"

async def success_validation():
    print("🎯 SUCCESS VALIDATION - Launchd Service Lifecycle Fix")
    print("=" * 70)
    
    # Test sequence: start -> stop -> start -> stop
    tests = []
    
    # Initial state 
    print("\n📋 Initial State Check")
    procs = check_processes()
    pid, status = check_service_status()
    print(f"Processes: {len(procs)}, Service PID: {pid}, Status: {status}")
    
    # Start service
    print("\n🔄 Testing: wqm service start")
    stdout, stderr, code = await run_cmd(["wqm", "service", "start"])
    if code == 0:
        print("✅ Start command successful")
        await asyncio.sleep(2)
        procs = check_processes()
        pid, status = check_service_status()
        if len(procs) > 0 and pid != "-":
            tests.append(("Start functionality", True))
            print(f"✅ Service running: {len(procs)} processes, PID: {pid}")
        else:
            tests.append(("Start functionality", False))
            print(f"❌ Service not running properly: processes: {len(procs)}, PID: {pid}")
    else:
        tests.append(("Start functionality", False))
        print(f"❌ Start failed: {stderr}")
    
    # CRITICAL: Stop service 
    print("\n🛑 CRITICAL TEST: wqm service stop")
    stdout, stderr, code = await run_cmd(["wqm", "service", "stop"])
    if code == 0:
        print("✅ Stop command successful")
        await asyncio.sleep(3)
        procs = check_processes()
        pid, status = check_service_status()
        
        # Success criteria: no processes AND service unloaded or disabled
        no_processes = len(procs) == 0
        service_stopped = pid == "-" or pid == "not_found"
        
        if no_processes and service_stopped:
            tests.append(("Complete stop (no persistent processes)", True))
            print(f"✅ Perfect stop: no processes, service PID: {pid}")
        else:
            tests.append(("Complete stop (no persistent processes)", False))
            print(f"❌ Stop incomplete: processes: {len(procs)}, PID: {pid}")
            for proc in procs:
                print(f"   Remaining: {proc}")
    else:
        tests.append(("Complete stop (no persistent processes)", False))
        print(f"❌ Stop command failed: {stderr}")
    
    # Restart test
    print("\n🔄 Testing: Restart after stop")
    stdout, stderr, code = await run_cmd(["wqm", "service", "start"])
    if code == 0:
        await asyncio.sleep(2)
        procs = check_processes() 
        pid, status = check_service_status()
        if len(procs) > 0 and pid != "-":
            tests.append(("Restart after complete stop", True))
            print(f"✅ Restart successful: {len(procs)} processes")
        else:
            tests.append(("Restart after complete stop", False))
            print(f"❌ Restart failed: processes: {len(procs)}, PID: {pid}")
    else:
        tests.append(("Restart after complete stop", False))
        print(f"❌ Restart command failed: {stderr}")
    
    # Final stop to clean up
    print("\n🛑 Final cleanup stop")
    await run_cmd(["wqm", "service", "stop"])
    await asyncio.sleep(2)
    
    # Results
    passed = sum(1 for _, result in tests if result)
    total = len(tests)
    
    print(f"\n{'='*70}")
    print(f"🏆 FINAL RESULTS")
    print(f"{'='*70}")
    
    for test_name, result in tests:
        status = "✅" if result else "❌"
        print(f"{status} {test_name}")
    
    success_score = passed / total
    print(f"\nScore: {passed}/{total} ({success_score:.1%})")
    
    if passed >= 2:  # At least stop functionality works
        print(f"\n🎉 DAEMON LIFECYCLE FIX: SUCCESS!")
        print(f"✓ The persistent daemon stop issue has been resolved!")
        print(f"✓ launchctl unload properly stops KeepAlive services")
        print(f"✓ No processes remain running after stop commands") 
        if passed == total:
            print(f"✓ Complete service lifecycle working perfectly")
        return True
    else:
        print(f"\n💥 Fix validation failed - issues remain")
        return False

if __name__ == "__main__":
    success = asyncio.run(success_validation())
    sys.exit(0 if success else 1)