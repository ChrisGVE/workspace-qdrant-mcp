#!/usr/bin/env python3
"""Test script to validate the launchd service lifecycle fix.

This script tests the complete daemon lifecycle to ensure the persistent
daemon stop issues have been resolved.

Test sequence:
1. Verify initial state
2. Install service
3. Start service
4. Verify running
5. Stop service
6. Verify stopped
7. Start again
8. Verify running again

Expected behavior:
- wqm service stop should properly terminate all memexd processes
- No processes should remain after stop
- Service should start successfully after being stopped
"""

import asyncio
import subprocess
import sys
import time
from pathlib import Path


class DaemonLifecycleTest:
    """Test the complete daemon lifecycle."""
    
    def __init__(self):
        self.failed_tests = []
        self.passed_tests = []
    
    async def run_command(self, cmd: list[str], check_return_code: bool = True) -> tuple[str, str, int]:
        """Run a command and return stdout, stderr, return_code."""
        print(f"Running: {' '.join(cmd)}")
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        stdout_text = stdout.decode().strip()
        stderr_text = stderr.decode().strip()
        return_code = process.returncode
        
        print(f"Return code: {return_code}")
        if stdout_text:
            print(f"Stdout: {stdout_text}")
        if stderr_text:
            print(f"Stderr: {stderr_text}")
        
        if check_return_code and return_code != 0:
            self.failed_tests.append(f"Command failed: {' '.join(cmd)}")
        
        return stdout_text, stderr_text, return_code
    
    def check_memexd_processes(self) -> list[str]:
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
    
    def log_test_result(self, test_name: str, success: bool, details: str = ""):
        """Log test result."""
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"\n{status}: {test_name}")
        if details:
            print(f"Details: {details}")
        
        if success:
            self.passed_tests.append(test_name)
        else:
            self.failed_tests.append(test_name)
    
    async def test_lifecycle(self):
        """Test the complete service lifecycle."""
        print("=" * 60)
        print("DAEMON LIFECYCLE TEST - Testing launchd service fix")
        print("=" * 60)
        
        # Test 1: Verify wqm command exists
        try:
            stdout, stderr, code = await self.run_command(["wqm", "--version"], check_return_code=False)
            if code == 0:
                self.log_test_result("wqm command available", True, stdout)
            else:
                self.log_test_result("wqm command available", False, f"wqm not found: {stderr}")
                return
        except Exception as e:
            self.log_test_result("wqm command available", False, str(e))
            return
        
        # Test 2: Check initial state
        print(f"\n{'-'*40}")
        print("Step 1: Checking initial state...")
        initial_processes = self.check_memexd_processes()
        print(f"Initial memexd processes: {len(initial_processes)}")
        for proc in initial_processes:
            print(f"  {proc}")
        
        # Test 3: Install service
        print(f"\n{'-'*40}")
        print("Step 2: Installing service...")
        stdout, stderr, code = await self.run_command(["wqm", "service", "install"], check_return_code=False)
        install_success = code == 0 or "already installed" in stderr.lower() or "already installed" in stdout.lower()
        self.log_test_result("Service install", install_success, f"stdout: {stdout}, stderr: {stderr}")
        
        # Test 4: Start service
        print(f"\n{'-'*40}")
        print("Step 3: Starting service...")
        stdout, stderr, code = await self.run_command(["wqm", "service", "start"], check_return_code=False)
        start_success = code == 0 or "already running" in stderr.lower() or "already running" in stdout.lower()
        self.log_test_result("Service start", start_success, f"stdout: {stdout}, stderr: {stderr}")
        
        # Wait for service to start
        print("Waiting 3 seconds for service to start...")
        await asyncio.sleep(3)
        
        # Test 5: Verify service is running
        print(f"\n{'-'*40}")
        print("Step 4: Verifying service is running...")
        stdout, stderr, code = await self.run_command(["wqm", "service", "status"], check_return_code=False)
        running_processes = self.check_memexd_processes()
        
        status_success = "running" in stdout.lower() or "running" in stderr.lower()
        processes_success = len(running_processes) > 0
        overall_running = status_success and processes_success
        
        self.log_test_result(
            "Service running verification", 
            overall_running,
            f"Status check: {status_success}, Processes found: {len(running_processes)}"
        )
        
        for proc in running_processes:
            print(f"  Running: {proc}")
        
        # Test 6: Critical Test - Stop service (this is the fix we're testing)
        print(f"\n{'-'*40}")
        print("Step 5: CRITICAL TEST - Stopping service (testing the fix)...")
        stdout, stderr, code = await self.run_command(["wqm", "service", "stop"], check_return_code=False)
        stop_success = code == 0
        self.log_test_result("Service stop command", stop_success, f"stdout: {stdout}, stderr: {stderr}")
        
        # Wait for processes to stop
        print("Waiting 5 seconds for processes to stop...")
        await asyncio.sleep(5)
        
        # Test 7: Verify NO processes remain (this is the key test)
        print(f"\n{'-'*40}")
        print("Step 6: CRITICAL VERIFICATION - Checking for remaining processes...")
        remaining_processes = self.check_memexd_processes()
        no_processes_remain = len(remaining_processes) == 0
        
        self.log_test_result(
            "No processes remain after stop", 
            no_processes_remain,
            f"Remaining processes: {len(remaining_processes)}"
        )
        
        if remaining_processes:
            print("❌ CRITICAL FAILURE: Processes still running after stop:")
            for proc in remaining_processes:
                print(f"  {proc}")
        else:
            print("✅ SUCCESS: No memexd processes remain after stop")
        
        # Test 8: Verify status shows stopped
        print(f"\n{'-'*40}")
        print("Step 7: Verifying status shows stopped...")
        stdout, stderr, code = await self.run_command(["wqm", "service", "status"], check_return_code=False)
        status_stopped = "stopped" in stdout.lower() or "not running" in stdout.lower() or code != 0
        self.log_test_result("Service status shows stopped", status_stopped, f"Status: {stdout}")
        
        # Test 9: Start service again
        print(f"\n{'-'*40}")
        print("Step 8: Starting service again...")
        stdout, stderr, code = await self.run_command(["wqm", "service", "start"], check_return_code=False)
        restart_success = code == 0
        self.log_test_result("Service restart", restart_success, f"stdout: {stdout}, stderr: {stderr}")
        
        # Wait for service to start
        print("Waiting 3 seconds for service to start...")
        await asyncio.sleep(3)
        
        # Test 10: Verify service is running again
        print(f"\n{'-'*40}")
        print("Step 9: Final verification - Service running after restart...")
        final_processes = self.check_memexd_processes()
        final_success = len(final_processes) > 0
        
        self.log_test_result(
            "Service running after restart",
            final_success,
            f"Processes found: {len(final_processes)}"
        )
        
        # Print final summary
        print(f"\n{'='*60}")
        print("FINAL TEST SUMMARY")
        print(f"{'='*60}")
        print(f"✅ Tests passed: {len(self.passed_tests)}")
        print(f"❌ Tests failed: {len(self.failed_tests)}")
        
        if self.failed_tests:
            print("\nFailed tests:")
            for test in self.failed_tests:
                print(f"  - {test}")
        
        # Critical success criteria
        critical_success = (
            no_processes_remain and  # No processes after stop
            stop_success and         # Stop command succeeded
            restart_success and      # Restart succeeded
            final_success           # Service running after restart
        )
        
        if critical_success:
            print(f"\n🎉 OVERALL RESULT: SUCCESS - Daemon lifecycle fix is working!")
            print("The launchd service lifecycle has been properly fixed.")
            return True
        else:
            print(f"\n💥 OVERALL RESULT: FAILURE - Daemon lifecycle issues persist")
            print("The fix may need additional work.")
            return False


async def main():
    """Run the daemon lifecycle test."""
    tester = DaemonLifecycleTest()
    success = await tester.test_lifecycle()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())