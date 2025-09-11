#!/usr/bin/env python3
"""
Comprehensive test matrix for ALL wqm service command combinations.

This implements the EXACT test requirements:
1. Test EVERY combination
2. Actually execute each test  
3. Fix the root cause
4. Make it work on macOS properly
5. Don't stop until ALL work
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

class ComprehensiveServiceTester:
    def __init__(self):
        self.wqm_path = "/Users/chris/.local/bin/wqm"
        self.test_results = []
        self.failure_count = 0
        self.success_count = 0
        
    def execute_command(self, cmd: List[str], expected_success: bool = True) -> Dict:
        """Execute a command and return detailed results."""
        print(f"🔧 Executing: {' '.join(cmd)}")
        
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=45
            )
            duration = time.time() - start_time
            
            # Determine if this was successful based on expectations
            actual_success = result.returncode == 0
            test_passed = actual_success == expected_success
            
            result_data = {
                "command": ' '.join(cmd),
                "expected_success": expected_success,
                "actual_success": actual_success,
                "test_passed": test_passed,
                "returncode": result.returncode,
                "stdout": result.stdout.strip() if result.stdout else "",
                "stderr": result.stderr.strip() if result.stderr else "",
                "duration": duration,
                "timestamp": time.time()
            }
            
            if test_passed:
                print(f"   ✅ PASS ({duration:.2f}s) - Exit code: {result.returncode}")
                self.success_count += 1
            else:
                print(f"   ❌ FAIL ({duration:.2f}s) - Expected success: {expected_success}, Got success: {actual_success}")
                print(f"   Exit code: {result.returncode}")
                if result.stdout:
                    print(f"   STDOUT: {result.stdout}")
                if result.stderr:
                    print(f"   STDERR: {result.stderr}")
                self.failure_count += 1
            
            self.test_results.append(result_data)
            return result_data
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            print(f"   ⏰ TIMEOUT after {duration:.2f}s")
            result_data = {
                "command": ' '.join(cmd),
                "expected_success": expected_success,
                "actual_success": False,
                "test_passed": False,
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
            print(f"   💥 EXCEPTION: {e}")
            result_data = {
                "command": ' '.join(cmd),
                "expected_success": expected_success,
                "actual_success": False,
                "test_passed": False,
                "returncode": -2,
                "stdout": "",
                "stderr": str(e),
                "duration": duration,
                "timestamp": time.time()
            }
            self.test_results.append(result_data)
            self.failure_count += 1
            return result_data

    def clean_state(self):
        """Ensure clean state before testing."""
        print("\n🧹 Cleaning state...")
        # Try to uninstall any existing service (ignore failures)
        self.execute_command([self.wqm_path, "service", "uninstall"], expected_success=False)
        time.sleep(2)

    def test_sequence_1_complete_lifecycle(self):
        """Test Sequence 1: Fresh system → install → start → status → stop → status → restart → status → uninstall"""
        print("\n" + "="*80)
        print("TEST SEQUENCE 1: Complete service lifecycle from clean state")
        print("="*80)
        
        self.clean_state()
        
        test_cases = [
            ([self.wqm_path, "service", "install"], True, "Install service"),
            ([self.wqm_path, "service", "start"], True, "Start service"),
            ([self.wqm_path, "service", "status"], True, "Check status - should be running"),
            ([self.wqm_path, "service", "stop"], True, "Stop service"),
            ([self.wqm_path, "service", "status"], True, "Check status - should be stopped"),
            ([self.wqm_path, "service", "restart"], True, "Restart service"),
            ([self.wqm_path, "service", "status"], True, "Check status - should be running again"),
            ([self.wqm_path, "service", "uninstall"], True, "Uninstall service")
        ]
        
        for cmd, expected, description in test_cases:
            print(f"\n📋 {description}")
            result = self.execute_command(cmd, expected)
            if not result["test_passed"]:
                print(f"🚨 CRITICAL FAILURE in sequence 1: {description}")
                break
            time.sleep(3)  # Proper pause between commands

    def test_sequence_2_minimal_install(self):
        """Test Sequence 2: Fresh system → install → status → uninstall"""
        print("\n" + "="*80)
        print("TEST SEQUENCE 2: Minimal install/check/uninstall")
        print("="*80)
        
        self.clean_state()
        
        test_cases = [
            ([self.wqm_path, "service", "install"], True, "Install service"),
            ([self.wqm_path, "service", "status"], True, "Check status after install"),
            ([self.wqm_path, "service", "uninstall"], True, "Uninstall service")
        ]
        
        for cmd, expected, description in test_cases:
            print(f"\n📋 {description}")
            result = self.execute_command(cmd, expected)
            if not result["test_passed"]:
                print(f"🚨 CRITICAL FAILURE in sequence 2: {description}")
                break
            time.sleep(3)

    def test_sequence_3_start_stop_cycling(self):
        """Test Sequence 3: Fresh system → install → start → stop → start → stop → uninstall"""
        print("\n" + "="*80)
        print("TEST SEQUENCE 3: Start/stop cycling")
        print("="*80)
        
        self.clean_state()
        
        test_cases = [
            ([self.wqm_path, "service", "install"], True, "Install service"),
            ([self.wqm_path, "service", "start"], True, "First start"),
            ([self.wqm_path, "service", "stop"], True, "First stop"),
            ([self.wqm_path, "service", "start"], True, "Second start"),
            ([self.wqm_path, "service", "stop"], True, "Second stop"),
            ([self.wqm_path, "service", "uninstall"], True, "Uninstall service")
        ]
        
        for cmd, expected, description in test_cases:
            print(f"\n📋 {description}")
            result = self.execute_command(cmd, expected)
            if not result["test_passed"]:
                print(f"🚨 CRITICAL FAILURE in sequence 3: {description}")
                break
            time.sleep(3)

    def test_sequence_4_restart_functionality(self):
        """Test Sequence 4: Fresh system → install → start → restart → status → stop → uninstall"""
        print("\n" + "="*80)
        print("TEST SEQUENCE 4: Restart functionality")
        print("="*80)
        
        self.clean_state()
        
        test_cases = [
            ([self.wqm_path, "service", "install"], True, "Install service"),
            ([self.wqm_path, "service", "start"], True, "Start service"),
            ([self.wqm_path, "service", "restart"], True, "Restart service"),
            ([self.wqm_path, "service", "status"], True, "Check status after restart"),
            ([self.wqm_path, "service", "stop"], True, "Stop service"),
            ([self.wqm_path, "service", "uninstall"], True, "Uninstall service")
        ]
        
        for cmd, expected, description in test_cases:
            print(f"\n📋 {description}")
            result = self.execute_command(cmd, expected)
            if not result["test_passed"]:
                print(f"🚨 CRITICAL FAILURE in sequence 4: {description}")
                break
            time.sleep(3)

    def test_sequence_5_error_recovery(self):
        """Test Sequence 5: Error recovery scenarios"""
        print("\n" + "="*80)
        print("TEST SEQUENCE 5: Error recovery testing")
        print("="*80)
        
        # Test double install
        print("\n🔄 Testing double install...")
        self.clean_state()
        self.execute_command([self.wqm_path, "service", "install"], True)
        time.sleep(2)
        # Double install should be handled gracefully
        self.execute_command([self.wqm_path, "service", "install"], True)
        
        # Test stop when not running
        print("\n🔄 Testing stop when not running...")
        self.execute_command([self.wqm_path, "service", "stop"], True)  # Ensure stopped
        time.sleep(2)
        self.execute_command([self.wqm_path, "service", "stop"], True)  # Should handle gracefully
        
        # Test start when already running
        print("\n🔄 Testing start when already running...")
        self.execute_command([self.wqm_path, "service", "start"], True)
        time.sleep(3)
        self.execute_command([self.wqm_path, "service", "start"], True)  # Should handle gracefully
        
        # Cleanup
        print("\n🧹 Final cleanup...")
        self.execute_command([self.wqm_path, "service", "uninstall"], True)

    def test_sequence_6_edge_cases(self):
        """Test Sequence 6: Edge cases and boundary conditions"""
        print("\n" + "="*80)
        print("TEST SEQUENCE 6: Edge cases and boundary conditions")
        print("="*80)
        
        self.clean_state()
        
        # Test status when service doesn't exist
        print("\n🔄 Testing status when service doesn't exist...")
        # Status should succeed but show "not installed"
        self.execute_command([self.wqm_path, "service", "status"], True)
        
        # Test start when service doesn't exist
        print("\n🔄 Testing start when service doesn't exist...")
        # Start should fail gracefully when service not installed
        self.execute_command([self.wqm_path, "service", "start"], False)
        
        # Test stop when service doesn't exist
        print("\n🔄 Testing stop when service doesn't exist...")
        # Stop should fail gracefully when service not installed
        self.execute_command([self.wqm_path, "service", "stop"], False)
        
        # Test restart when service doesn't exist
        print("\n🔄 Testing restart when service doesn't exist...")
        # Restart should fail gracefully when service not installed
        self.execute_command([self.wqm_path, "service", "restart"], False)

    def test_sequence_7_logs_functionality(self):
        """Test Sequence 7: Logs functionality"""
        print("\n" + "="*80)
        print("TEST SEQUENCE 7: Logs functionality")
        print("="*80)
        
        self.clean_state()
        
        test_cases = [
            ([self.wqm_path, "service", "install"], True, "Install service"),
            ([self.wqm_path, "service", "start"], True, "Start service"),
            ([self.wqm_path, "service", "logs"], True, "Check logs while running"),
            ([self.wqm_path, "service", "logs", "--lines", "10"], True, "Check logs with line limit"),
            ([self.wqm_path, "service", "stop"], True, "Stop service"),
            ([self.wqm_path, "service", "logs"], True, "Check logs after stopping"),
            ([self.wqm_path, "service", "uninstall"], True, "Uninstall service")
        ]
        
        for cmd, expected, description in test_cases:
            print(f"\n📋 {description}")
            result = self.execute_command(cmd, expected)
            if not result["test_passed"]:
                print(f"🚨 CRITICAL FAILURE in sequence 7: {description}")
                break
            time.sleep(3)

    def run_all_tests(self):
        """Execute ALL test sequences comprehensively."""
        print("🚀 Starting COMPREHENSIVE service testing...")
        print(f"📍 Using wqm binary: {self.wqm_path}")
        
        # Verify wqm exists
        if not Path(self.wqm_path).exists():
            print(f"❌ wqm binary not found at {self.wqm_path}")
            return False
            
        start_time = time.time()
        
        try:
            # Execute ALL test sequences
            self.test_sequence_1_complete_lifecycle()
            self.test_sequence_2_minimal_install()
            self.test_sequence_3_start_stop_cycling()
            self.test_sequence_4_restart_functionality()
            self.test_sequence_5_error_recovery()
            self.test_sequence_6_edge_cases()
            self.test_sequence_7_logs_functionality()
            
        except KeyboardInterrupt:
            print("\n🛑 Testing interrupted by user")
            
        end_time = time.time()
        total_duration = end_time - start_time
        
        self.print_final_results(total_duration)
        self.save_results()
        
        return self.failure_count == 0

    def print_final_results(self, duration: float):
        """Print comprehensive test results."""
        print("\n" + "="*80)
        print("COMPREHENSIVE TEST RESULTS")
        print("="*80)
        print(f"Total tests executed: {len(self.test_results)}")
        print(f"Tests passed: {self.success_count}")
        print(f"Tests failed: {self.failure_count}")
        print(f"Success rate: {(self.success_count / len(self.test_results) * 100):.1f}%")
        print(f"Total duration: {duration:.2f} seconds")
        
        if self.failure_count > 0:
            print(f"\n🚨 FAILED TESTS ({self.failure_count}):")
            for result in self.test_results:
                if not result["test_passed"]:
                    print(f"   ❌ {result['command']}")
                    print(f"      Expected success: {result['expected_success']}, Got: {result['actual_success']}")
                    print(f"      Exit code: {result['returncode']}")
                    if result["stderr"]:
                        print(f"      Error: {result['stderr'][:200]}")
        
        if self.failure_count == 0:
            print("\n🎉 ALL TESTS PASSED! Service implementation is fully working!")
            print("✅ Every single service command combination works correctly.")
        else:
            print(f"\n🔥 {self.failure_count} TESTS FAILED! Service implementation still has issues.")
            print("❌ Critical failures that must be fixed before service is production-ready.")

    def save_results(self):
        """Save comprehensive test results."""
        results_file = Path("20250111-1710_comprehensive_service_test_results.json")
        
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
            
        print(f"\n💾 Comprehensive test results saved to: {results_file}")

def main():
    tester = ComprehensiveServiceTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n🏆 MISSION ACCOMPLISHED: ALL service commands work perfectly!")
    else:
        print("\n💣 MISSION FAILED: Service implementation still broken!")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())