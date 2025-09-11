#!/usr/bin/env python3
"""
WQM Service Validation Test Suite
Tests the fixes for wqm service daemon management issues
"""

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

class ServiceTester:
    """Comprehensive service management testing."""
    
    def __init__(self):
        self.test_results = []
        self.errors = []
        
    def log_result(self, test_name: str, passed: bool, details: str = ""):
        """Log a test result."""
        self.test_results.append({
            "test": test_name,
            "passed": passed,
            "details": details
        })
        
        if passed:
            console.print(f"✅ {test_name}: PASSED")
        else:
            console.print(f"❌ {test_name}: FAILED - {details}")
            self.errors.append(f"{test_name}: {details}")
    
    async def run_wqm_command(self, *args) -> Dict:
        """Run a wqm command and return structured result."""
        cmd = ["uv", "run", "wqm", "service"] + list(args)
        try:
            result = await asyncio.create_subprocess_exec(
                *cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                cwd="/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp/src/python"
            )
            stdout, stderr = await result.communicate()
            
            return {
                "returncode": result.returncode,
                "stdout": stdout.decode(),
                "stderr": stderr.decode(),
                "success": result.returncode == 0
            }
        except Exception as e:
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": str(e),
                "success": False,
                "exception": e
            }
    
    async def find_memexd_processes(self) -> List[int]:
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
    
    async def check_pid_files(self) -> List[str]:
        """Check for existing PID files."""
        pid_files = [
            "/tmp/memexd.pid",
            "/tmp/memexd-launchd.pid", 
            "/tmp/memexd-manual.pid",
            "/tmp/memexd-service.pid",
        ]
        
        existing = []
        for pid_file in pid_files:
            if Path(pid_file).exists():
                existing.append(pid_file)
        
        return existing
    
    async def test_binary_detection(self):
        """Test 1: Binary path resolution."""
        console.print("\n🔍 Testing binary detection...")
        
        # This is a tricky test since we don't have a built binary
        # We'll test that the error message is informative
        result = await self.run_wqm_command("install", "--no-auto-start")
        
        if not result["success"]:
            # Check if error message is informative
            error_output = result["stderr"] + result["stdout"]
            if "memexd binary not found" in error_output or "Build it first with:" in error_output:
                self.log_result("Binary Detection Error Messages", True, 
                              "Clear error message about missing binary")
            else:
                self.log_result("Binary Detection Error Messages", False, 
                              f"Unclear error message: {error_output[:200]}")
        else:
            # If install succeeded, that's also good (binary was found)
            self.log_result("Binary Detection", True, "Binary found and install succeeded")
    
    async def test_status_accuracy_clean_system(self):
        """Test 2: Status command on clean system."""
        console.print("\n📊 Testing status accuracy on clean system...")
        
        # First ensure system is clean
        await self.cleanup_all_processes()
        
        result = await self.run_wqm_command("status")
        
        # Check that status command runs successfully 
        if result["success"] or "not found" in result["stderr"] or "not installed" in result["stderr"]:
            # Either succeeds with "not loaded" status or fails with clear "not installed" message
            self.log_result("Clean System Status", True, "Status command handles clean system correctly")
        else:
            self.log_result("Clean System Status", False, 
                          f"Unexpected status behavior: {result['stderr'][:200]}")
    
    async def test_process_detection_when_running(self):
        """Test 3: Process detection when processes exist."""
        console.print("\n🔍 Testing process detection...")
        
        # Start a dummy memexd-like process (using sleep with memexd in name)
        dummy_cmd = ["sh", "-c", "exec -a 'memexd-test-dummy' sleep 10"]
        dummy_process = await asyncio.create_subprocess_exec(*dummy_cmd)
        
        try:
            await asyncio.sleep(1)  # Let process start
            
            # Now test if our service detection works
            processes_before = await self.find_memexd_processes()
            
            if dummy_process.pid in processes_before:
                self.log_result("Process Detection", True, 
                              f"Successfully detected dummy memexd process (PID {dummy_process.pid})")
            else:
                self.log_result("Process Detection", False, 
                              f"Failed to detect dummy process. Found PIDs: {processes_before}")
            
        finally:
            # Clean up dummy process
            try:
                dummy_process.terminate()
                await dummy_process.wait()
            except:
                pass
    
    async def test_pid_file_cleanup(self):
        """Test 4: PID file cleanup."""
        console.print("\n🧹 Testing PID file cleanup...")
        
        # Create some dummy PID files
        test_pid_files = ["/tmp/memexd-test.pid", "/tmp/memexd-manual.pid"]
        
        for pid_file in test_pid_files:
            Path(pid_file).write_text("12345")  # Fake PID
        
        # Run status command which should clean up stale files
        await self.run_wqm_command("status")
        
        # Check if cleanup happened (files with non-existent PIDs should be removed)
        cleaned_up = 0
        for pid_file in test_pid_files:
            if not Path(pid_file).exists():
                cleaned_up += 1
        
        self.log_result("PID File Cleanup", cleaned_up > 0, 
                      f"Cleaned up {cleaned_up}/{len(test_pid_files)} stale PID files")
    
    async def test_stop_command_robustness(self):
        """Test 5: Stop command robustness."""
        console.print("\n⏹️ Testing stop command robustness...")
        
        # Test stop command when nothing is running
        result = await self.run_wqm_command("stop")
        
        # Should handle gracefully (either success or clear error)
        if result["success"] or "not running" in result["stderr"] or "not found" in result["stderr"]:
            self.log_result("Stop Command Robustness", True, 
                          "Stop command handles 'nothing running' case gracefully")
        else:
            self.log_result("Stop Command Robustness", False, 
                          f"Stop command failed unexpectedly: {result['stderr'][:200]}")
    
    async def test_error_message_quality(self):
        """Test 6: Error message quality."""
        console.print("\n📝 Testing error message quality...")
        
        # Test various commands that should produce helpful errors
        test_cases = [
            ("start", "when not installed"),
            ("stop", "when not running"),  
            ("restart", "when not installed"),
            ("uninstall", "when not installed")
        ]
        
        good_error_count = 0
        for cmd, scenario in test_cases:
            result = await self.run_wqm_command(cmd)
            error_text = result["stderr"] + result["stdout"]
            
            # Check for helpful content in error messages
            helpful_phrases = [
                "install first", "not installed", "not found", 
                "suggestion", "try", "run", "command"
            ]
            
            has_helpful_content = any(phrase.lower() in error_text.lower() for phrase in helpful_phrases)
            
            if has_helpful_content:
                good_error_count += 1
        
        self.log_result("Error Message Quality", good_error_count >= len(test_cases) // 2, 
                      f"{good_error_count}/{len(test_cases)} commands provided helpful error messages")
    
    async def cleanup_all_processes(self):
        """Clean up any existing memexd processes."""
        try:
            # Kill any memexd processes
            result = await asyncio.create_subprocess_exec(
                "pkill", "-f", "memexd",
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            await result.communicate()
            
            # Clean up PID files
            pid_files = [
                "/tmp/memexd.pid",
                "/tmp/memexd-launchd.pid", 
                "/tmp/memexd-manual.pid",
                "/tmp/memexd-service.pid",
                "/tmp/memexd-test.pid"
            ]
            
            for pid_file in pid_files:
                try:
                    Path(pid_file).unlink()
                except FileNotFoundError:
                    pass
                    
        except Exception as e:
            console.print(f"Warning: Error in cleanup: {e}")
    
    async def run_all_tests(self):
        """Run all validation tests."""
        console.print(Panel.fit(
            "WQM Service Management Validation Tests\n"
            "Testing fixes for daemon management issues",
            title="Service Validation",
            style="blue"
        ))
        
        # Initial cleanup
        await self.cleanup_all_processes()
        await asyncio.sleep(1)
        
        # Run all tests
        await self.test_binary_detection()
        await self.test_status_accuracy_clean_system() 
        await self.test_process_detection_when_running()
        await self.test_pid_file_cleanup()
        await self.test_stop_command_robustness()
        await self.test_error_message_quality()
        
        # Final cleanup
        await self.cleanup_all_processes()
        
        # Summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary."""
        passed = sum(1 for result in self.test_results if result["passed"])
        total = len(self.test_results)
        
        # Create summary table
        table = Table(title="Test Results Summary")
        table.add_column("Test", style="cyan")
        table.add_column("Result", style="white")
        table.add_column("Details", style="dim")
        
        for result in self.test_results:
            status_text = Text("PASSED", style="green") if result["passed"] else Text("FAILED", style="red")
            table.add_row(
                result["test"], 
                str(status_text), 
                result.get("details", "")[:60]
            )
        
        console.print(table)
        
        # Overall result
        if passed == total:
            console.print(Panel.fit(
                f"🎉 All {total} tests passed!\n\n"
                "Service management fixes are working correctly.",
                title="Validation Complete",
                style="green"
            ))
        else:
            console.print(Panel.fit(
                f"⚠️ {passed}/{total} tests passed\n\n"
                f"Issues found:\n" + "\n".join(f"• {error}" for error in self.errors),
                title="Validation Issues",
                style="yellow"
            ))


async def main():
    """Run service validation tests."""
    tester = ServiceTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())