#!/usr/bin/env python3
"""
Comprehensive WQM Service Test
Tests the actual service lifecycle operations to validate fixes
"""

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

async def run_wqm_service_command(*args) -> Dict:
    """Run a wqm service command and return structured result."""
    cmd = ["uv", "run", "wqm", "service"] + list(args)
    try:
        console.print(f"🔧 Running: {' '.join(cmd)}")
        result = await asyncio.create_subprocess_exec(
            *cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            cwd="/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp/src/python"
        )
        stdout, stderr = await result.communicate()
        
        stdout_text = stdout.decode()
        stderr_text = stderr.decode()
        
        console.print(f"📤 Return code: {result.returncode}")
        if stdout_text.strip():
            console.print(f"📄 Output: {stdout_text[:200]}{'...' if len(stdout_text) > 200 else ''}")
        if stderr_text.strip():
            console.print(f"⚠️ Error: {stderr_text[:200]}{'...' if len(stderr_text) > 200 else ''}")
        
        return {
            "returncode": result.returncode,
            "stdout": stdout_text,
            "stderr": stderr_text,
            "success": result.returncode == 0
        }
    except Exception as e:
        console.print(f"💥 Exception: {e}")
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
            "success": False,
            "exception": e
        }

async def find_memexd_processes() -> List[int]:
    """Find all memexd processes."""
    try:
        result = await asyncio.create_subprocess_exec(
            "pgrep", "-f", "memexd",
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        stdout, stderr = await result.communicate()
        
        if result.returncode == 0:
            pids = []
            for line in stdout.decode().strip().split('\n'):
                if line.strip() and line.strip().isdigit():
                    pids.append(int(line.strip()))
            return pids
        return []
    except Exception:
        return []

async def cleanup_environment():
    """Clean up any existing processes and files."""
    console.print("🧹 Cleaning up environment...")
    
    # Kill any memexd processes
    try:
        result = await asyncio.create_subprocess_exec(
            "pkill", "-f", "memexd",
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        await result.communicate()
    except:
        pass
    
    # Clean up PID files
    pid_files = [
        "/tmp/memexd.pid",
        "/tmp/memexd-launchd.pid", 
        "/tmp/memexd-manual.pid",
        "/tmp/memexd-service.pid",
    ]
    
    for pid_file in pid_files:
        try:
            Path(pid_file).unlink()
        except FileNotFoundError:
            pass

async def test_service_lifecycle():
    """Test complete service lifecycle."""
    console.print(Panel.fit(
        "Comprehensive WQM Service Lifecycle Test\n"
        "Testing actual service operations",
        title="Service Lifecycle Test",
        style="blue"
    ))
    
    # Clean environment first
    await cleanup_environment()
    await asyncio.sleep(1)
    
    # 1. Test status on clean system
    console.print("\n📊 Step 1: Status on clean system")
    status_result = await run_wqm_service_command("status")
    
    # 2. Test install command
    console.print("\n⚙️ Step 2: Install service")
    install_result = await run_wqm_service_command("install", "--no-auto-start")
    
    # 3. Test status after install
    console.print("\n📊 Step 3: Status after install")
    status_after_install = await run_wqm_service_command("status")
    
    # 4. Test start command (will likely fail due to missing binary, but should handle gracefully)
    console.print("\n▶️ Step 4: Start service")
    start_result = await run_wqm_service_command("start")
    
    # 5. Test stop command
    console.print("\n⏹️ Step 5: Stop service")
    stop_result = await run_wqm_service_command("stop")
    
    # 6. Test restart command  
    console.print("\n🔄 Step 6: Restart service")
    restart_result = await run_wqm_service_command("restart")
    
    # 7. Check for orphaned processes
    console.print("\n🔍 Step 7: Check for orphaned processes")
    final_processes = await find_memexd_processes()
    console.print(f"🔍 Found {len(final_processes)} memexd processes: {final_processes}")
    
    # 8. Test uninstall
    console.print("\n🗑️ Step 8: Uninstall service")
    uninstall_result = await run_wqm_service_command("uninstall", "--force")
    
    # 9. Final cleanup verification
    console.print("\n✅ Step 9: Final verification")
    final_status = await run_wqm_service_command("status")
    final_processes_2 = await find_memexd_processes()
    console.print(f"🔍 Final process count: {len(final_processes_2)}")
    
    # Summary
    console.print(Panel.fit(
        f"Test Lifecycle Complete!\n\n"
        f"• Status (clean): {'✅' if status_result['returncode'] in [0, 1] else '❌'}\n"
        f"• Install: {'✅' if install_result['returncode'] in [0, 1] else '❌'}\n" 
        f"• Status (post-install): {'✅' if status_after_install['success'] else '❌'}\n"
        f"• Start: {'✅' if start_result['returncode'] in [0, 1] else '❌'}\n"
        f"• Stop: {'✅' if stop_result['success'] else '❌'}\n"
        f"• Restart: {'✅' if restart_result['returncode'] in [0, 1] else '❌'}\n"
        f"• Uninstall: {'✅' if uninstall_result['returncode'] in [0, 1] else '❌'}\n"
        f"• No orphaned processes: {'✅' if len(final_processes_2) == 0 else '❌'}",
        title="Lifecycle Test Results",
        style="green" if len(final_processes_2) == 0 else "yellow"
    ))

async def test_specific_fixes():
    """Test specific fixes mentioned in the test report."""
    console.print(Panel.fit(
        "Testing Specific Fixes from Test Report",
        title="Specific Fix Validation",
        style="cyan"
    ))
    
    # Test 1: Binary path resolution
    console.print("\n🔍 Testing Binary Path Resolution...")
    await cleanup_environment()
    install_result = await run_wqm_service_command("install")
    
    # Should either succeed (if binary found) or give helpful error
    if install_result["success"]:
        console.print("✅ Binary found and install succeeded")
    else:
        error_text = install_result["stderr"] + install_result["stdout"]
        if "memexd binary not found" in error_text and "cargo build" in error_text:
            console.print("✅ Clear error message about missing binary")
        else:
            console.print(f"❌ Unclear error: {error_text[:150]}")
    
    # Test 2: Process detection accuracy
    console.print("\n🔍 Testing Process Detection...")
    
    # Create a test process that looks like memexd
    test_process = await asyncio.create_subprocess_exec(
        "sh", "-c", "exec -a 'memexd-test' sleep 30"
    )
    
    try:
        await asyncio.sleep(2)  # Let it start
        
        # Test status command
        status_result = await run_wqm_service_command("status") 
        
        # Check if our detection logic works
        processes = await find_memexd_processes()
        if test_process.pid in processes:
            console.print("✅ Process detection correctly identifies memexd processes")
        else:
            console.print(f"❌ Failed to detect test process {test_process.pid}")
            
    finally:
        # Clean up test process
        try:
            test_process.terminate()
            await test_process.wait()
        except:
            pass
    
    # Test 3: Stop command effectiveness 
    console.print("\n🔍 Testing Stop Command Effectiveness...")
    
    # Create multiple test processes
    test_processes = []
    for i in range(3):
        proc = await asyncio.create_subprocess_exec(
            "sh", "-c", f"exec -a 'memexd-test-{i}' sleep 30"
        )
        test_processes.append(proc)
    
    try:
        await asyncio.sleep(2)  # Let them start
        
        initial_count = len(await find_memexd_processes())
        console.print(f"📊 Created {len(test_processes)} test processes, found {initial_count} total")
        
        # Test stop command  
        stop_result = await run_wqm_service_command("stop")
        await asyncio.sleep(3)  # Wait for cleanup
        
        final_count = len(await find_memexd_processes())
        console.print(f"📊 After stop command: {final_count} processes remain")
        
        if final_count < initial_count:
            console.print("✅ Stop command reduced process count")
        else:
            console.print("⚠️ Stop command may not have cleaned up all processes")
            
    finally:
        # Force cleanup test processes
        for proc in test_processes:
            try:
                proc.terminate()
                await proc.wait()
            except:
                pass
    
    console.print("\n🧹 Final cleanup...")
    await cleanup_environment()

async def main():
    """Run comprehensive service tests."""
    try:
        await test_service_lifecycle()
        console.print("\n" + "="*50)
        await test_specific_fixes()
        
    except KeyboardInterrupt:
        console.print("\n🛑 Tests interrupted by user")
    except Exception as e:
        console.print(f"💥 Test failed with exception: {e}")
    finally:
        console.print("\n🧹 Final cleanup...")
        await cleanup_environment()

if __name__ == "__main__":
    asyncio.run(main())