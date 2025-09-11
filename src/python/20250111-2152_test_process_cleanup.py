#!/usr/bin/env python3
"""
Test process cleanup functionality
"""

import asyncio
import subprocess
import sys
from pathlib import Path

async def main():
    print("🧪 Testing process cleanup functionality...")
    
    # Create some dummy memexd processes
    print("📝 Creating dummy memexd processes...")
    processes = []
    
    for i in range(3):
        # Create a process that looks like memexd
        proc = await asyncio.create_subprocess_exec(
            "sh", "-c", f"exec -a 'memexd-test-{i}' sleep 300"
        )
        processes.append(proc)
        print(f"   ✅ Created process {i+1} with PID {proc.pid}")
    
    # Wait a moment for them to start
    await asyncio.sleep(2)
    
    # Check how many memexd processes exist
    result = await asyncio.create_subprocess_exec(
        "pgrep", "-f", "memexd",
        stdout=subprocess.PIPE
    )
    stdout, _ = await result.communicate()
    
    if result.returncode == 0:
        pids = [int(line) for line in stdout.decode().strip().split('\n') if line.strip().isdigit()]
        print(f"🔍 Found {len(pids)} memexd processes: {pids}")
    else:
        print("🔍 No memexd processes found")
        pids = []
    
    # Now test the stop command
    print("⏹️ Running wqm service stop...")
    stop_result = await asyncio.create_subprocess_exec(
        "uv", "run", "wqm", "service", "stop",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd="/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp/src/python"
    )
    
    stdout, stderr = await stop_result.communicate()
    print(f"📤 Stop command return code: {stop_result.returncode}")
    if stdout.decode().strip():
        print(f"📄 Output: {stdout.decode()[:200]}")
    if stderr.decode().strip():
        print(f"⚠️ Error: {stderr.decode()[:200]}")
    
    # Wait for cleanup to complete
    await asyncio.sleep(3)
    
    # Check remaining processes
    result2 = await asyncio.create_subprocess_exec(
        "pgrep", "-f", "memexd",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout2, _ = await result2.communicate()
    
    if result2.returncode == 0:
        remaining_pids = [int(line) for line in stdout2.decode().strip().split('\n') if line.strip().isdigit()]
        print(f"🔍 After cleanup: {len(remaining_pids)} memexd processes remain: {remaining_pids}")
    else:
        print("✅ After cleanup: No memexd processes found")
        remaining_pids = []
    
    # Summary
    if len(remaining_pids) < len(pids):
        print(f"✅ SUCCESS: Process cleanup working! Reduced from {len(pids)} to {len(remaining_pids)} processes")
    elif len(remaining_pids) == 0:
        print("✅ EXCELLENT: All processes cleaned up successfully!")
    else:
        print(f"⚠️ WARNING: Process cleanup may not be fully effective. Still have {len(remaining_pids)} processes")
    
    # Force cleanup any remaining test processes
    print("🧹 Cleaning up test processes...")
    for proc in processes:
        try:
            if proc.returncode is None:  # Still running
                proc.terminate()
                await proc.wait()
                print(f"   🗑️ Terminated process {proc.pid}")
        except:
            pass
    
    print("✅ Test complete!")

if __name__ == "__main__":
    asyncio.run(main())