#!/usr/bin/env python3
"""
WQM Service Testing Script
Tests all service command combinations and logs results.
"""

import subprocess
import time
import sys
from datetime import datetime
import os

class WQMServiceTester:
    def __init__(self):
        self.test_results = []
        self.report_lines = []
        
    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line)
        self.report_lines.append(line)
        
    def run_command(self, cmd, description=""):
        """Run command and capture result"""
        self.log(f"Testing: {cmd}")
        if description:
            self.log(f"Description: {description}")
            
        try:
            result = subprocess.run(
                cmd.split(), 
                capture_output=True, 
                text=True, 
                timeout=30,
                cwd="/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp"
            )
            
            self.log(f"Exit code: {result.returncode}")
            if result.stdout.strip():
                self.log(f"STDOUT: {result.stdout.strip()}")
            if result.stderr.strip():
                self.log(f"STDERR: {result.stderr.strip()}")
                
            return {
                'command': cmd,
                'description': description,
                'exit_code': result.returncode,
                'stdout': result.stdout.strip(),
                'stderr': result.stderr.strip(),
                'success': result.returncode == 0
            }
            
        except subprocess.TimeoutExpired:
            self.log(f"TIMEOUT: Command exceeded 30 seconds")
            return {
                'command': cmd,
                'description': description,
                'exit_code': -1,
                'stdout': '',
                'stderr': 'Command timeout',
                'success': False
            }
        except Exception as e:
            self.log(f"ERROR: {str(e)}")
            return {
                'command': cmd,
                'description': description,
                'exit_code': -2,
                'stdout': '',
                'stderr': str(e),
                'success': False
            }
    
    def check_process_status(self):
        """Check if memexd process is running"""
        try:
            result = subprocess.run(
                ["ps", "aux"], 
                capture_output=True, 
                text=True
            )
            
            memexd_processes = [line for line in result.stdout.split('\n') if 'memexd' in line]
            
            self.log(f"Process check - Found {len(memexd_processes)} memexd processes:")
            for proc in memexd_processes:
                self.log(f"  {proc}")
                
            return len(memexd_processes) > 0, memexd_processes
            
        except Exception as e:
            self.log(f"Process check failed: {e}")
            return False, []
    
    def test_initial_state(self):
        self.log("=== INITIAL STATE ASSESSMENT ===")
        
        # Check if wqm command is available
        wqm_check = self.run_command("which wqm", "Check if wqm command exists")
        self.test_results.append(wqm_check)
        
        if not wqm_check['success']:
            self.log("Trying uv run wqm instead...")
            wqm_check = self.run_command("uv run wqm --help", "Check if uv run wqm works")
            self.test_results.append(wqm_check)
        
        # Check initial process state
        running, processes = self.check_process_status()
        
        # Check initial service status
        status_result = self.run_command("uv run wqm service status", "Initial service status check")
        self.test_results.append(status_result)
        
        self.log("=== INITIAL STATE COMPLETE ===\n")
        
    def test_service_commands(self):
        """Test all service commands systematically"""
        
        # Test sequence 1: Fresh install scenario
        self.log("=== TEST SEQUENCE 1: FRESH INSTALL ===")
        
        # Stop any existing service first
        self.run_command("uv run wqm service stop", "Ensure clean state")
        self.run_command("uv run wqm service uninstall", "Ensure clean state") 
        
        commands = [
            ("uv run wqm service status", "Status check before install"),
            ("uv run wqm service install", "Fresh install"),
            ("uv run wqm service status", "Status after install"),
            ("uv run wqm service start", "Start after install"), 
            ("uv run wqm service status", "Status after start"),
            ("uv run wqm service stop", "Stop service"),
            ("uv run wqm service status", "Status after stop"),
            ("uv run wqm service restart", "Restart when stopped"),
            ("uv run wqm service status", "Status after restart"),
        ]
        
        for cmd, desc in commands:
            result = self.run_command(cmd, desc)
            self.test_results.append(result)
            self.check_process_status()
            time.sleep(2)  # Brief pause between commands
            
        self.log("=== SEQUENCE 1 COMPLETE ===\n")
        
        # Test sequence 2: Error conditions
        self.log("=== TEST SEQUENCE 2: ERROR CONDITIONS ===")
        
        error_commands = [
            ("uv run wqm service start", "Start when already running"),
            ("uv run wqm service install", "Double install"),
            ("uv run wqm service stop", "Stop running service"),
            ("uv run wqm service stop", "Double stop"),
            ("uv run wqm service restart", "Restart when stopped"),
            ("uv run wqm service uninstall", "Uninstall while stopped"),
            ("uv run wqm service uninstall", "Double uninstall"),
            ("uv run wqm service start", "Start when not installed"),
        ]
        
        for cmd, desc in error_commands:
            result = self.run_command(cmd, desc)
            self.test_results.append(result)
            self.check_process_status()
            time.sleep(2)
            
        self.log("=== SEQUENCE 2 COMPLETE ===\n")
        
    def generate_report(self):
        """Generate final test report"""
        self.log("=== GENERATING FINAL REPORT ===")
        
        # Count successes and failures
        successes = sum(1 for r in self.test_results if r['success'])
        failures = len(self.test_results) - successes
        
        self.log(f"Total commands tested: {len(self.test_results)}")
        self.log(f"Successful: {successes}")
        self.log(f"Failed: {failures}")
        
        # Identify problem areas
        failed_commands = [r for r in self.test_results if not r['success']]
        
        if failed_commands:
            self.log("\n=== FAILED COMMANDS ===")
            for fail in failed_commands:
                self.log(f"FAIL: {fail['command']}")
                self.log(f"  Exit code: {fail['exit_code']}")
                self.log(f"  Error: {fail['stderr']}")
                self.log("")
        
        return {
            'total': len(self.test_results),
            'successes': successes, 
            'failures': failures,
            'failed_commands': failed_commands,
            'all_results': self.test_results,
            'log_lines': self.report_lines
        }

if __name__ == "__main__":
    tester = WQMServiceTester()
    
    try:
        tester.test_initial_state()
        tester.test_service_commands()
        report = tester.generate_report()
        
        # Write detailed report
        with open("20250111-2137_wqm_service_detailed_results.txt", "w") as f:
            f.write("WQM Service Testing - Detailed Results\n")
            f.write("=" * 50 + "\n\n")
            
            for line in report['log_lines']:
                f.write(line + "\n")
                
            f.write("\n\n" + "=" * 50 + "\n")
            f.write("SUMMARY ANALYSIS\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total Commands: {report['total']}\n")
            f.write(f"Successful: {report['successes']}\n") 
            f.write(f"Failed: {report['failures']}\n")
            f.write(f"Success Rate: {(report['successes']/report['total']*100):.1f}%\n")
        
        print(f"\nDetailed results written to: 20250111-2137_wqm_service_detailed_results.txt")
        print(f"Summary: {report['successes']}/{report['total']} commands successful")
        
    except KeyboardInterrupt:
        print("\nTesting interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Testing failed with error: {e}")
        sys.exit(1)