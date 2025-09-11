#!/usr/bin/env python3
"""
Comprehensive service testing plan to systematically test and fix all wqm service commands.

This script will execute every test combination and fix failures as they occur.
"""
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

class ServiceTestPlan:
    def __init__(self):
        self.wqm_path = "/Users/chris/.local/bin/wqm"
        self.test_results = []
        self.failure_count = 0
        self.success_count = 0
        
    def run_command(self, cmd: List[str]) -> Dict[str, Any]:
        """Run a command and return the result."""
        print(f"📋 Running: {' '.join(cmd)}")
        
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=30
            )
            duration = time.time() - start_time
            
            success = result.returncode == 0
            
            result_data = {
                "command": ' '.join(cmd),
                "success": success,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration": duration,
                "timestamp": time.time()
            }
            
            if success:
                print(f"✅ SUCCESS ({duration:.2f}s)")
                self.success_count += 1
            else:
                print(f"❌ FAILED ({duration:.2f}s) - Return code: {result.returncode}")
                print(f"   STDOUT: {result.stdout}")
                print(f"   STDERR: {result.stderr}")
                self.failure_count += 1
            
            self.test_results.append(result_data)
            return result_data
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            print(f"⏰ TIMEOUT after {duration:.2f}s")
            result_data = {
                "command": ' '.join(cmd),
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": "Command timed out",
                "duration": duration,
                "timestamp": time.time()
            }
            self.test_results.append(result_data)
            self.failure_count += 1
            return result_data
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"💥 EXCEPTION: {e}")
            result_data = {
                "command": ' '.join(cmd),
                "success": False,
                "returncode": -2,
                "stdout": "",
                "stderr": str(e),
                "duration": duration,
                "timestamp": time.time()
            }
            self.test_results.append(result_data)
            self.failure_count += 1
            return result_data

    def test_sequence_1_clean_install_full_lifecycle(self):
        """Test Sequence 1: Fresh system → install → start → status → stop → status → restart → status → uninstall"""
        print("\n" + "="*80)
        print("TEST SEQUENCE 1: Complete service lifecycle from clean state")
        print("="*80)
        
        # Ensure clean state
        print("\n🧹 Cleaning up any existing service...")
        self.run_command([self.wqm_path, "service", "uninstall"])
        time.sleep(2)
        
        # Test sequence
        commands = [
            [self.wqm_path, "service", "install"],
            [self.wqm_path, "service", "start"], 
            [self.wqm_path, "service", "status"],
            [self.wqm_path, "service", "stop"],
            [self.wqm_path, "service", "status"],
            [self.wqm_path, "service", "restart"],
            [self.wqm_path, "service", "status"],
            [self.wqm_path, "service", "uninstall"]
        ]
        
        for cmd in commands:
            result = self.run_command(cmd)
            if not result["success"]:
                print(f"🚨 CRITICAL FAILURE in sequence 1 at command: {result['command']}")
                print("   Stopping sequence due to failure")
                break
            time.sleep(2)  # Brief pause between commands

    def test_sequence_2_install_status_uninstall(self):
        """Test Sequence 2: Fresh system → install → status → uninstall"""
        print("\n" + "="*80)
        print("TEST SEQUENCE 2: Minimal install/check/uninstall")
        print("="*80)
        
        # Ensure clean state
        print("\n🧹 Cleaning up any existing service...")
        self.run_command([self.wqm_path, "service", "uninstall"])
        time.sleep(2)
        
        commands = [
            [self.wqm_path, "service", "install"],
            [self.wqm_path, "service", "status"],
            [self.wqm_path, "service", "uninstall"]
        ]
        
        for cmd in commands:
            result = self.run_command(cmd)
            if not result["success"]:
                print(f"🚨 CRITICAL FAILURE in sequence 2 at command: {result['command']}")
                print("   Stopping sequence due to failure")
                break
            time.sleep(2)

    def test_sequence_3_start_stop_cycling(self):
        """Test Sequence 3: Fresh system → install → start → stop → start → stop → uninstall"""
        print("\n" + "="*80)
        print("TEST SEQUENCE 3: Start/stop cycling")
        print("="*80)
        
        # Ensure clean state
        print("\n🧹 Cleaning up any existing service...")
        self.run_command([self.wqm_path, "service", "uninstall"])
        time.sleep(2)
        
        commands = [
            [self.wqm_path, "service", "install"],
            [self.wqm_path, "service", "start"],
            [self.wqm_path, "service", "stop"],
            [self.wqm_path, "service", "start"],
            [self.wqm_path, "service", "stop"],
            [self.wqm_path, "service", "uninstall"]
        ]
        
        for cmd in commands:
            result = self.run_command(cmd)
            if not result["success"]:
                print(f"🚨 CRITICAL FAILURE in sequence 3 at command: {result['command']}")
                print("   Stopping sequence due to failure")
                break
            time.sleep(2)

    def test_sequence_4_restart_testing(self):
        """Test Sequence 4: Fresh system → install → start → restart → status → stop → uninstall"""
        print("\n" + "="*80)
        print("TEST SEQUENCE 4: Restart functionality")
        print("="*80)
        
        # Ensure clean state
        print("\n🧹 Cleaning up any existing service...")
        self.run_command([self.wqm_path, "service", "uninstall"])
        time.sleep(2)
        
        commands = [
            [self.wqm_path, "service", "install"],
            [self.wqm_path, "service", "start"],
            [self.wqm_path, "service", "restart"],
            [self.wqm_path, "service", "status"],
            [self.wqm_path, "service", "stop"],
            [self.wqm_path, "service", "uninstall"]
        ]
        
        for cmd in commands:
            result = self.run_command(cmd)
            if not result["success"]:
                print(f"🚨 CRITICAL FAILURE in sequence 4 at command: {result['command']}")
                print("   Stopping sequence due to failure")
                break
            time.sleep(2)

    def test_sequence_5_error_recovery(self):
        """Test Sequence 5: Error recovery scenarios"""
        print("\n" + "="*80)
        print("TEST SEQUENCE 5: Error recovery testing")
        print("="*80)
        
        # Test double install
        print("\n🔄 Testing double install...")
        self.run_command([self.wqm_path, "service", "uninstall"])
        time.sleep(2)
        self.run_command([self.wqm_path, "service", "install"])
        time.sleep(2)
        result = self.run_command([self.wqm_path, "service", "install"])  # Should handle gracefully
        
        # Test stop when not running
        print("\n🔄 Testing stop when not running...")
        self.run_command([self.wqm_path, "service", "stop"])  # Ensure stopped
        time.sleep(2)
        self.run_command([self.wqm_path, "service", "stop"])  # Should handle gracefully
        
        # Test start when already running
        print("\n🔄 Testing start when already running...")
        self.run_command([self.wqm_path, "service", "start"])
        time.sleep(2)
        self.run_command([self.wqm_path, "service", "start"])  # Should handle gracefully
        
        # Cleanup
        print("\n🧹 Final cleanup...")
        self.run_command([self.wqm_path, "service", "uninstall"])

    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        print("\n" + "="*80)
        print("TEST SEQUENCE 6: Edge cases and boundary conditions")
        print("="*80)
        
        # Test status when service doesn't exist
        print("\n🔄 Testing status when service doesn't exist...")
        self.run_command([self.wqm_path, "service", "uninstall"])
        time.sleep(2)
        self.run_command([self.wqm_path, "service", "status"])
        
        # Test start when service doesn't exist
        print("\n🔄 Testing start when service doesn't exist...")
        self.run_command([self.wqm_path, "service", "start"])
        
        # Test stop when service doesn't exist
        print("\n🔄 Testing stop when service doesn't exist...")
        self.run_command([self.wqm_path, "service", "stop"])
        
        # Test restart when service doesn't exist
        print("\n🔄 Testing restart when service doesn't exist...")
        self.run_command([self.wqm_path, "service", "restart"])

    def run_all_tests(self):
        """Execute all test sequences"""
        print("🚀 Starting comprehensive service testing...")
        print(f"📍 Using wqm binary: {self.wqm_path}")
        
        # Verify wqm exists
        if not Path(self.wqm_path).exists():
            print(f"❌ wqm binary not found at {self.wqm_path}")
            return False
            
        start_time = time.time()
        
        try:
            self.test_sequence_1_clean_install_full_lifecycle()
            self.test_sequence_2_install_status_uninstall()
            self.test_sequence_3_start_stop_cycling()
            self.test_sequence_4_restart_testing()
            self.test_sequence_5_error_recovery()
            self.test_edge_cases()
            
        except KeyboardInterrupt:
            print("\n🛑 Testing interrupted by user")
            
        end_time = time.time()
        total_duration = end_time - start_time
        
        self.print_summary(total_duration)
        self.save_results()
        
        return self.failure_count == 0

    def print_summary(self, duration: float):
        """Print test summary"""
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        print(f"Total tests run: {len(self.test_results)}")
        print(f"Successes: {self.success_count}")
        print(f"Failures: {self.failure_count}")
        print(f"Success rate: {(self.success_count / len(self.test_results) * 100):.1f}%")
        print(f"Total duration: {duration:.2f} seconds")
        
        if self.failure_count > 0:
            print(f"\n🚨 FAILED COMMANDS:")
            for result in self.test_results:
                if not result["success"]:
                    print(f"   ❌ {result['command']}")
                    print(f"      Return code: {result['returncode']}")
                    if result["stderr"]:
                        print(f"      Error: {result['stderr'][:200]}")
        
        if self.failure_count == 0:
            print("\n🎉 ALL TESTS PASSED! Service implementation is working correctly.")
        else:
            print(f"\n🔥 {self.failure_count} TESTS FAILED! Service implementation needs fixes.")

    def save_results(self):
        """Save test results to JSON file"""
        results_file = Path("20250111-1643_service_test_results.json")
        
        summary = {
            "test_run": {
                "timestamp": time.time(),
                "total_tests": len(self.test_results),
                "successes": self.success_count,
                "failures": self.failure_count,
                "success_rate": self.success_count / len(self.test_results) * 100 if self.test_results else 0
            },
            "test_results": self.test_results
        }
        
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"\n💾 Test results saved to: {results_file}")

if __name__ == "__main__":
    tester = ServiceTestPlan()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)