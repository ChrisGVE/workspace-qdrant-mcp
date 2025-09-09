#!/usr/bin/env python3
"""
Emergency Stop - Immediate kill switch for all testing processes
Can be run manually or triggered by system guardian
"""

import psutil
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

class EmergencyStop:
    def __init__(self):
        self.setup_logging()
        self.test_process_names = [
            "python3",  # General Python processes
            "stress_test",
            "load_test", 
            "sync_test",
            "baseline_test",
            "autonomous_test_runner"
        ]
        
    def setup_logging(self):
        """Setup emergency logging"""
        log_dir = Path(__file__).parent.parent / "monitoring_logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"emergency_stop_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def find_test_processes(self) -> List[Dict[str, Any]]:
        """Find all running test processes"""
        test_processes = []
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
                try:
                    # Check if it's a test process
                    if self.is_test_process(proc):
                        test_processes.append({
                            'pid': proc.info['pid'],
                            'name': proc.info['name'],
                            'cmdline': ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else '',
                            'create_time': proc.info['create_time']
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error finding test processes: {e}")
        
        return test_processes
    
    def is_test_process(self, proc: psutil.Process) -> bool:
        """Check if a process is related to testing"""
        try:
            cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
            
            # Check for testing sandbox path
            if 'LT_20250906-1822_testing_sandbox' in cmdline:
                return True
            
            # Check for test script names
            test_keywords = [
                'stress_test', 'load_test', 'sync_test', 'baseline_test',
                'autonomous_test_runner', 'system_guardian',
                'memory_monitor', 'cpu_monitor', 'disk_monitor'
            ]
            
            for keyword in test_keywords:
                if keyword in cmdline.lower():
                    return True
            
            # Check if Python process running testing scripts
            if proc.info['name'] == 'python3' or proc.info['name'] == 'python':
                if any(keyword in cmdline.lower() for keyword in ['test', 'monitor', 'stress', 'load']):
                    return True
                    
            return False
            
        except Exception:
            return False
    
    def terminate_process(self, pid: int, name: str, force: bool = False) -> bool:
        """Terminate a specific process"""
        try:
            if not psutil.pid_exists(pid):
                self.logger.info(f"Process {name} (PID: {pid}) already terminated")
                return True
            
            process = psutil.Process(pid)
            
            if force:
                self.logger.warning(f"Force killing process {name} (PID: {pid})")
                process.kill()
            else:
                self.logger.info(f"Terminating process {name} (PID: {pid})")
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=10)
                except psutil.TimeoutExpired:
                    self.logger.warning(f"Process {name} didn't terminate gracefully, force killing")
                    process.kill()
                    process.wait(timeout=5)
            
            return True
            
        except psutil.NoSuchProcess:
            self.logger.info(f"Process {name} (PID: {pid}) already terminated")
            return True
        except Exception as e:
            self.logger.error(f"Error terminating process {name} (PID: {pid}): {e}")
            return False
    
    def stop_all_tests(self, force: bool = False) -> Dict[str, Any]:
        """Stop all test processes"""
        self.logger.critical("EMERGENCY STOP: Terminating all test processes")
        
        # Find all test processes
        test_processes = self.find_test_processes()
        
        if not test_processes:
            self.logger.info("No test processes found running")
            return {
                "success": True,
                "terminated_processes": [],
                "failed_processes": [],
                "message": "No test processes were running"
            }
        
        terminated = []
        failed = []
        
        self.logger.info(f"Found {len(test_processes)} test processes to terminate")
        
        # Sort by creation time (newest first) to avoid dependency issues
        test_processes.sort(key=lambda x: x['create_time'], reverse=True)
        
        for proc_info in test_processes:
            pid = proc_info['pid']
            name = proc_info['name']
            
            if self.terminate_process(pid, name, force):
                terminated.append(proc_info)
                self.logger.info(f"Successfully terminated {name} (PID: {pid})")
            else:
                failed.append(proc_info)
                self.logger.error(f"Failed to terminate {name} (PID: {pid})")
        
        # Log summary
        self.logger.critical(f"Emergency stop complete: {len(terminated)} terminated, {len(failed)} failed")
        
        # Save emergency stop report
        self.save_emergency_report(terminated, failed)
        
        return {
            "success": len(failed) == 0,
            "terminated_processes": terminated,
            "failed_processes": failed,
            "message": f"Terminated {len(terminated)} processes, {len(failed)} failures"
        }
    
    def save_emergency_report(self, terminated: List[Dict], failed: List[Dict]):
        """Save emergency stop report"""
        report_dir = Path(__file__).parent.parent / "results_summary"
        report_dir.mkdir(exist_ok=True)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "emergency_stop",
            "summary": {
                "terminated_count": len(terminated),
                "failed_count": len(failed),
                "total_processes": len(terminated) + len(failed)
            },
            "terminated_processes": terminated,
            "failed_processes": failed
        }
        
        report_file = report_dir / f"emergency_stop_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            self.logger.info(f"Emergency stop report saved to {report_file}")
        except Exception as e:
            self.logger.error(f"Failed to save emergency stop report: {e}")
    
    def check_system_health(self) -> Dict[str, Any]:
        """Check system health after emergency stop"""
        try:
            # Get current system metrics
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            disk = psutil.disk_usage('/')
            
            return {
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "cpu_percent": cpu_percent,
                "disk_percent": (disk.used / disk.total) * 100,
                "disk_free_gb": disk.free / (1024**3),
                "active_processes": len(psutil.pids())
            }
        except Exception as e:
            self.logger.error(f"Error checking system health: {e}")
            return {}


def main():
    """Main entry point"""
    emergency = EmergencyStop()
    
    print("EMERGENCY STOP ACTIVATED")
    print("This will terminate all testing processes immediately.")
    
    # Check for force flag
    force = '--force' in sys.argv
    
    if not force:
        response = input("Are you sure you want to proceed? (yes/no): ")
        if response.lower() != 'yes':
            print("Emergency stop cancelled")
            return
    
    # Execute emergency stop
    result = emergency.stop_all_tests(force=force)
    
    print(f"\nEmergency stop complete: {result['message']}")
    
    # Check system health
    health = emergency.check_system_health()
    if health:
        print(f"System health after emergency stop:")
        print(f"  Memory: {health['memory_percent']:.1f}% used")
        print(f"  CPU: {health['cpu_percent']:.1f}% used")
        print(f"  Disk: {health['disk_percent']:.1f}% used")


if __name__ == "__main__":
    main()